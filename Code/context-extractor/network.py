from typing import TypedDict, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import interrupt, Command

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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

MAIN_MODEL_SYSTEM_PROMPT = """
Answer the user's request accurately and directly.

Use only the information given in the request.
Do not assume facts the user has not provided.
"""

def empty_token_usage() -> TokenUsage:
    return {
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": None,
        "total_tokens": 0,
        "cached_input_tokens": None,
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
    reasoning_tokens: int | None
    total_tokens: int
    cached_input_tokens: int | None

class ClarificationState(MessagesState):
    original_query: str
    clarifications: list[Clarification]
    current_question: str | None
    extraction_complete: bool
    expanded_query: str | None

    original_answer: str | None
    expanded_answer: str | None

    extractor_usage: TokenUsage
    synthesis_usage: TokenUsage
    original_answer_usage: TokenUsage
    expanded_answer_usage: TokenUsage

class ContextNetwork:
    extractor_model: LocalModel
    synthesis_model: LocalModel
    builder: StateGraph

    def __init__(self, extractor_model_name: str, extractor_temperature: float=0, synthesis_model_name: str=None, synthesis_temperature: float=0,
                 main_model_provider: str="openai", main_model_name: str = "gpt-6.1-sol", main_model_temperature: float=0):

        extractor_base_model = ChatOllama(model=extractor_model_name, temperature=extractor_temperature)
        self.extractor_model = LocalModel(extractor_base_model)
        synthesis_base_model = ChatOllama(model=synthesis_model_name, temperature=synthesis_temperature)
        self.synthesis_model = LocalModel(synthesis_base_model)
        self.builder = StateGraph(ClarificationState)

        self.main_model = self.create_main_model(
            provider=main_model_provider,
            model_name=main_model_name,
            temperature=main_model_temperature,
        )

    def create_main_model(self,
            provider: Literal["openai", "gemini"],
            model_name: str,
            temperature: float = 0,
    ) -> BaseChatModel:

        if provider == "openai":
            return ChatOpenAI(
                model=model_name,
            )

        if provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=model_name,
            )

        raise ValueError(
            f"Unsupported provider: {provider}"
        )

    def generate_question(self, state: ClarificationState):
        """
        Generate exactly one clarification question.

        If the model returns STOP_MESSAGE, mark the
        extraction process as complete.
        """
        response = self.extractor_model.invoke(state["messages"])

        call_usage = get_token_usage(response)
        previous_usage = state["extractor_usage"]

        extractor_usage = {
            "input_tokens":
                previous_usage["input_tokens"]
                + call_usage["input_tokens"],

            "output_tokens":
                previous_usage["output_tokens"]
                + call_usage["output_tokens"],

            "reasoning_tokens": sum_optional([
                previous_usage["reasoning_tokens"],
                call_usage["reasoning_tokens"],
            ]),

            "total_tokens":
                previous_usage["total_tokens"]
                + call_usage["total_tokens"],

            "cached_input_tokens": sum_optional([
                previous_usage["cached_input_tokens"],
                call_usage["cached_input_tokens"],
            ]),
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

    def answer_original_query(
            self,
            state: ClarificationState,
    ):
        messages = [
            SystemMessage(
                content=MAIN_MODEL_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=state["original_query"]
            ),
        ]

        response = self.main_model.invoke(
            messages
        )

        return {
            "original_answer":
                response.content.strip(),

            "original_answer_usage":
                get_token_usage(response),
        }

    def answer_expanded_query(
            self,
            state: ClarificationState,
    ):
        expanded_query = state["expanded_query"]

        if not expanded_query:
            raise ValueError(
                "No expanded query is available."
            )

        messages = [
            SystemMessage(
                content=MAIN_MODEL_SYSTEM_PROMPT
            ),
            HumanMessage(
                content=expanded_query
            ),
        ]

        response = self.main_model.invoke(
            messages
        )

        return {
            "expanded_answer":
                response.content.strip(),

            "expanded_answer_usage":
                get_token_usage(response),
        }

    def build(self):
        self.builder.add_node("generate_question", self.generate_question)
        self.builder.add_node("ask_user", self.ask_user)
        self.builder.add_node("synthesize_query", self.synthesize_query)
        self.builder.add_node("answer_original_query", self.answer_original_query)
        self.builder.add_node("answer_expanded_query",self.answer_expanded_query)

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
            "answer_original_query",
        )

        self.builder.add_edge(
            "answer_original_query",
            "answer_expanded_query",
        )

        self.builder.add_edge(
            "answer_expanded_query",
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

    input_details = (
        usage.get("input_token_details") or {}
    )

    output_details = (
        usage.get("output_token_details") or {}
    )

    return {
        "input_tokens":
            usage.get("input_tokens", 0),

        "output_tokens":
            usage.get("output_tokens", 0),

        "reasoning_tokens":
            output_details.get("reasoning"),

        "total_tokens":
            usage.get("total_tokens", 0),

        "cached_input_tokens":
            input_details.get("cache_read"),
    }

def sum_optional(
    values: list[int | None]
) -> int | None:

    reported = [
        value
        for value in values
        if value is not None
    ]

    if not reported:
        return None

    return sum(reported)

def add_usage(
    *items: TokenUsage,
) -> TokenUsage:
    return {
        "input_tokens": sum(
            item["input_tokens"]
            for item in items
        ),
        "output_tokens": sum(
            item["output_tokens"]
            for item in items
        ),
        "total_tokens": sum(
            item["total_tokens"]
            for item in items
        ),
        "cached_input_tokens":
            sum_optional([
                item["cached_input_tokens"]
                for item in items
            ]),

        "reasoning_tokens":
            sum_optional([
                item["reasoning_tokens"]
                for item in items
            ]),
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
        "original_answer_usage": empty_token_usage(),
        "expanded_answer_usage": empty_token_usage(),

        "original_answer": None,
        "expanded_answer": None,
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

    baseline_usage = (
        final_state["original_answer_usage"]
    )

    enhanced_remote_usage = (
        final_state["expanded_answer_usage"]
    )

    context_extractor_usage = add_usage(
        final_state["extractor_usage"],
        final_state["synthesis_usage"],
    )

    enhanced_total_usage = add_usage(
        final_state["extractor_usage"],
        final_state["synthesis_usage"],
        final_state["expanded_answer_usage"],
    )

    print("\nOriginal query:")
    print(final_state["original_query"])

    print("\nOriginal answer:")
    print(final_state["original_answer"])

    print("\nSynthesized query:")
    print(final_state["expanded_query"])

    print("\nExpanded-query answer:")
    print(final_state["expanded_answer"])

    print("\n--- Token comparison ---")

    print(
        "\nBaseline main-model call:",
        baseline_usage,
    )

    print(
        "\nContextExtractor only:",
        context_extractor_usage,
    )

    print(
        "\nExpanded-query main-model call:",
        enhanced_remote_usage,
    )

    print(
        "\nComplete enhanced pipeline:",
        enhanced_total_usage,
    )

    remote_token_delta = (
            enhanced_remote_usage["total_tokens"]
            - baseline_usage["total_tokens"]
    )

    pipeline_token_delta = (
            enhanced_total_usage["total_tokens"]
            - baseline_usage["total_tokens"]
    )

    print(
        "\nRemote token delta:",
        remote_token_delta,
    )

    print(
        "Full pipeline token delta:",
        pipeline_token_delta,
    )