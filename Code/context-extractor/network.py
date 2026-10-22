from typing import TypedDict, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import interrupt, Command

MAX_CLARIFICATIONS = 5

STOP_MESSAGE = "I do not need any more information"

CLARIFICATION_SYSTEM_PROMPT = f"""
You are an intermediate tool between a user and a larger conversation model.
Your purpose is to analyze a user's prompt and try to derive additional context to help towards a more complete answer.
Do not attempt to answer the original query itself.
Do not make any assumptions about information the user has not provided.
Strictly ask exactly one question at a time.
Do not ask for information the user has already provided.
A maximum of {MAX_CLARIFICATIONS} clarification questions may eventually be asked,
but you must only ask one question in the current response.

When no further clarification is required, respond exactly with:

{STOP_MESSAGE}

When asking a clarification question, output only the question.
Do not provide reasoning, explanation, headings, numbering, or an
answer to the original request.
"""

SYNTHESIS_SYSTEM_PROMPT = """
You are a query synthesis model.
You sit between two LLMs one responsible to ask clarifications from the user based on the original query and one
responsible to take your synthesized query and answer it.

You will receive an original user query followed by a conversation
containing clarification questions and the user's answers.

Your task is to rewrite the original query into a concise,
self-contained query that incorporates all relevant information
provided by the user during clarification.

Rules:
- Preserve the user's original intent.
- Include ALL relevant information from the clarification answers in a concise manner.
- Try to use natural flowing language. No bullets, lists or anything similar.
- Do not add information that the user did not provide.
- Do not answer the query.
- Do not explain your reasoning.
- Do not mention the clarification process.
- Output only the synthesized query.
"""

def empty_token_usage() -> TokenUsage:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }

class LocalModel:
    model: BaseChatModel

    def __init__(self, model: BaseChatModel):
        self.model = model

    def invoke(self, messages: Any):
        return self.model.invoke(input=messages)

class Clarification(TypedDict):
    question: str
    answer: str

class TokenUsage(TypedDict):
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int

class ClarificationState(MessagesState):
    original_query: str
    clarifications: list[Clarification]
    current_question: str | None
    extraction_complete: bool
    expanded_query: str | None

    extractor_usage: TokenUsage
    synthesis_usage: TokenUsage

class ContextNetwork:
    extractor_model: LocalModel
    synthesis_model: LocalModel
    builder: StateGraph

    def __init__(self, extractor_model_name: str, extractor_temperature: float=0, synthesis_model_name: str=None, synthesis_temperature: float=0):

        extractor_base_model = ChatOllama(model=extractor_model_name, temperature=extractor_temperature)
        self.extractor_model = LocalModel(extractor_base_model)
        synthesis_base_model = ChatOllama(model=synthesis_model_name, temperature=synthesis_temperature)
        self.synthesis_model = LocalModel(synthesis_base_model)
        self.builder = StateGraph(ClarificationState)

    def generate_question(self, state: ClarificationState):
        """
        Generate exactly one clarification question.

        If the model returns STOP_MESSAGE, mark the
        extraction process as complete.
        """
        response = self.extractor_model.invoke(state["messages"])

        response = self.extractor_model.invoke(
            state["messages"]
        )

        call_usage = get_token_usage(response)
        previous_usage = state["extractor_usage"]

        extractor_usage = {
            "input_tokens":
                previous_usage["input_tokens"]
                + call_usage["input_tokens"],

            "output_tokens":
                previous_usage["output_tokens"]
                + call_usage["output_tokens"],

            "reasoning_tokens":
                previous_usage["reasoning_tokens"]
                + call_usage["reasoning_tokens"],

            "total_tokens":
                previous_usage["total_tokens"]
                + call_usage["total_tokens"],
        }

        content = response.content.strip()

        extraction_complete = (
                content.strip().rstrip(".").lower()
                ==
                STOP_MESSAGE.strip().rstrip(".").lower()
        )

        return {
            "current_question": (
                None
                if extraction_complete
                else content
            ),
            "extraction_complete":
                extraction_complete,
            "messages": [
                AIMessage(content=content)
            ],
            "extractor_usage": extractor_usage,
        }

    def route_after_question(self, state: ClarificationState) -> Literal["ask_user", "end"]:
        """
        Decide whether to present the generated question
        to the user or stop because the model needs no
        further information.
        """

        if state["extraction_complete"]:
            return "end"

        if state["current_question"] is None:
            return "end"

        return "ask_user"

    def ask_user(self, state: ClarificationState):
        """
        Pause the graph and wait for the user's answer.

        Command(resume=...) supplies the value returned by interrupt().
        """
        question = state["current_question"]

        if question is None:
            raise ValueError(
                "No clarification question is available."
            )

        answer = interrupt({
            "type": "clarification",
            "question": question,
            "question_number": (
                    len(state["clarifications"]) + 1
            ),
            "maximum_questions":
                MAX_CLARIFICATIONS,
        })

        answer = str(answer).strip()

        clarification: Clarification = {
            "question":question,
            "answer":answer,
        }

        return {
            "clarifications": (
                    state["clarifications"] + [clarification]
            ),
            "messages": [
                HumanMessage(content=answer)
            ],
            "current_question": None,
        }

    def route_after_answer(self, state: ClarificationState) -> Literal["generate_question", "end"]:
        """
        Stop after five completed Q/A pairs.

        Otherwise, call the model again so it can derive
        the next question from the updated message history.
        """

        if (len(state["clarifications"])>= MAX_CLARIFICATIONS):
            return "end"

        return "generate_question"

    def synthesize_query(
            self,
            state: ClarificationState,
    ):
        conversation = []

        for message in state["messages"]:

            # Do not forward the extractor's system prompt.
            if isinstance(message, SystemMessage):
                continue

            # Do not forward the extractor's STOP marker.
            if (
                    isinstance(message, AIMessage)
                    and
                    message.content.strip().rstrip(".").lower()
                    ==
                    STOP_MESSAGE.strip().rstrip(".").lower()
            ):
                continue

            conversation.append(message)

        synthesis_messages = [
            SystemMessage(
                content=SYNTHESIS_SYSTEM_PROMPT
            ),
            *conversation,
        ]

        response = self.synthesis_model.invoke(
            synthesis_messages
        )

        synthesis_usage = get_token_usage(response)

        return {
            "expanded_query": response.content.strip(),
            "synthesis_usage": synthesis_usage,
        }

    def build(self):
        self.builder.add_node("generate_question", self.generate_question)
        self.builder.add_node("ask_user", self.ask_user)
        self.builder.add_node("synthesize_query", self.synthesize_query)

        self.builder.add_edge(START, "generate_question")
        self.builder.add_conditional_edges(
            "generate_question",
            self.route_after_question,
            {
                "ask_user": "ask_user",
                "end": "synthesize_query",
            },
        )

        self.builder.add_conditional_edges(
            "ask_user",
            self.route_after_answer,
            {
                "generate_question":
                    "generate_question",
                "end": "synthesize_query",
            },
        )

        self.builder.add_edge(
            "synthesize_query",
            END,
        )

        memory = InMemorySaver()

        return self.builder.compile(checkpointer=memory)

def get_interrupt_payload(result: dict) -> dict | None:
    """
    Return the current interrupt payload, if the graph
    is waiting for external input.
    """

    interrupts = result.get(
        "__interrupt__",
        []
    )

    if not interrupts:
        return None

    return interrupts[0].value

def get_token_usage(response) -> TokenUsage:
    usage = response.usage_metadata or {}

    output_details = (
        usage.get("output_token_details") or {}
    )

    return {
        "input_tokens":
            usage.get("input_tokens", 0),

        "output_tokens":
            usage.get("output_tokens", 0),

        "reasoning_tokens":
            output_details.get("reasoning", 0),

        "total_tokens":
            usage.get("total_tokens", 0),
    }

if __name__ == "__main__":
    network = ContextNetwork(extractor_model_name="gemma4:e4b", extractor_temperature=0, synthesis_model_name="gemma4:e4b", synthesis_temperature=0)
    graph = network.build()

    query = (
        "I have a headache. What do you recommend I do?"
        #"What can I bake for dinner with courgettes?"
    )

    config = {
        "configurable": {
            "thread_id": "context-extraction-001",
        }
    }

    initial_state: ClarificationState = {
        "messages": [
            (
                "system",
                CLARIFICATION_SYSTEM_PROMPT,
            ),
            (
                "human",
                query,
            ),
        ],
        "original_query": query,
        "clarifications": [],
        "current_question": None,
        "extraction_complete": False,
        "expanded_query": None,

        "extractor_usage": empty_token_usage(),
        "synthesis_usage": empty_token_usage(),
    }

    # Start the graph.
    result = graph.invoke(
        initial_state,
        config=config,
    )

    # Each interrupt returns control to this console loop.
    while True:
        payload = get_interrupt_payload(result)

        # No interrupt means the graph reached END.
        if payload is None:
            break

        question = payload["question"]
        question_number = payload[
            "question_number"
        ]
        maximum = payload[
            "maximum_questions"
        ]

        print(
            f"\nClarification "
            f"{question_number}/{maximum}:"
        )

        print(question)

        user_answer = input(
            "\nYour answer: "
        ).strip()

        # Optional safeguard against an empty answer.
        while not user_answer:
            print(
                "Please provide a short answer."
            )

            user_answer = input(
                "Your answer: "
            ).strip()

        # Resume the same paused graph thread.
        result = graph.invoke(
            Command(
                resume=user_answer
            ),
            config=config,
        )

    final_state = result

    if final_state["extraction_complete"]:
        print(
            "Termination: sufficient context"
        )
    else:
        print(
            "Termination: maximum questions reached"
        )

    print("\nContext extraction complete.")

    print(
        "\nClarifications recorded:",
        len(final_state["clarifications"]),
    )

    for index, clarification in enumerate(
            final_state["clarifications"],
            start=1,
    ):
        print(
            f"\nQ{index}: "
            f"{clarification['question']}"
        )

        print(
            f"A{index}: "
            f"{clarification['answer']}"
        )

    expanded_query = final_state.get(
        "expanded_query"
    )

    if not expanded_query:
        raise RuntimeError(
            "The synthesis node did not produce an expanded query."
        )

    print("\nSynthesized query:")
    print(expanded_query)

    extractor_usage = final_state["extractor_usage"]
    synthesis_usage = final_state["synthesis_usage"]

    print("\nToken usage")

    print("\nExtractor:")
    print(
        f"  Input:     "
        f"{extractor_usage['input_tokens']}"
    )
    print(
        f"  Output:    "
        f"{extractor_usage['output_tokens']}"
    )
    print(
        f"  Reasoning: "
        f"{extractor_usage['reasoning_tokens']}"
    )
    print(
        f"  Total:     "
        f"{extractor_usage['total_tokens']}"
    )

    print("\nSynthesis:")
    print(
        f"  Input:     "
        f"{synthesis_usage['input_tokens']}"
    )
    print(
        f"  Output:    "
        f"{synthesis_usage['output_tokens']}"
    )
    print(
        f"  Reasoning: "
        f"{synthesis_usage['reasoning_tokens']}"
    )
    print(
        f"  Total:     "
        f"{synthesis_usage['total_tokens']}"
    )

    print("\nContextExtractor total:")

    print(
        "  Input:",
        extractor_usage["input_tokens"]
        + synthesis_usage["input_tokens"]
    )

    print(
        "  Output:",
        extractor_usage["output_tokens"]
        + synthesis_usage["output_tokens"]
    )

    print(
        "  Total:",
        extractor_usage["total_tokens"]
        + synthesis_usage["total_tokens"]
    )