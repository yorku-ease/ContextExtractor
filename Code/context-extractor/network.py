from typing import TypedDict, Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import interrupt, Command

MAX_CLARIFICATIONS = 5

STOP_MESSAGE = "I do not need any more information"

SYSTEM_PROMPT = f"""
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


class ContextExtractorModel:
    model: BaseChatModel

    def __init__(self, model: BaseChatModel):
        self.model = model

    def invoke(self, messages: Any):
        return self.model.invoke(input=messages)

class Clarification(TypedDict):
    question: str
    answer: str

class ClarificationState(MessagesState):
    original_query: str
    clarifications: list[Clarification]
    current_question: str | None
    extraction_complete: bool
    expanded_query: str | None

class ContextNetwork:
    model: ContextExtractorModel
    builder: StateGraph

    def __init__(self, model_name: str, temperature: float=0):

        base_model = ChatOllama(model=model_name, temperature=temperature)
        self.model = ContextExtractorModel(base_model)
        self.builder = StateGraph(ClarificationState)

    def generate_question(self, state: ClarificationState):
        """
        Generate exactly one clarification question.

        If the model returns STOP_MESSAGE, mark the
        extraction process as complete.
        """
        response = self.model.invoke(state["messages"])
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

    def build(self):
        self.builder.add_node("generate_question", self.generate_question)
        self.builder.add_node("ask_user", self.ask_user)
        self.builder.add_edge(START, "generate_question")
        self.builder.add_conditional_edges(
            "generate_question",
            self.route_after_question,
            {
                "ask_user": "ask_user",
                "end": END,
            },
        )

        self.builder.add_conditional_edges(
            "ask_user",
            self.route_after_answer,
            {
                "generate_question":
                    "generate_question",
                "end": END,
            },
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



if __name__ == "__main__":
    network = ContextNetwork(model_name="gemma4:e4b", temperature=0)
    graph = network.build()

    query = (
        "I have a headache. What do you recommend I do?"
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
                SYSTEM_PROMPT,
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