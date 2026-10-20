from typing import TypedDict, Any

from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from langgraph.types import interrupt, Command
from pydantic import BaseModel

SYSTEM_PROMPT = """
You are an intermediate tool between a user and a larger conversation model.
Your purpose is to analyze a user's prompt and try to derive additional context to help towards a more complete answer.
Do not attempt to answer the original query itself.
Do not make any assumptions about information the user has not provided.
Strictly ask exactly one question at a time.
Do not ask for information the user has already provided.
A maximum of five clarification questions may eventually be asked,
but you must only ask one question in the current response.
If you think you cannot extract more context from the user, you can stop before asking five questions. In this case, simply say "I do not need any more information".
Pose the question directly to the user without any reasoning or explanation.
Expect a short answer from the user and keep the additional information in your memory.
When done, simply say "I do not need any more information".
"""


class ContextExtractorModel:
    model: BaseChatModel

    def __init__(self, model: BaseChatModel):
        self.model = model

    def invoke(self, messages: Any):
        return self.model.invoke(input=messages)

class Clarification(BaseModel):
    question: str
    answer: str

class ClarificationState(MessagesState):
    original_query: str
    clarifications: list[Clarification]
    current_question: str | None
    expanded_query: str | None

class ContextNetwork:

    def __init__(self, model_name: str, temperature: float=0):

        base_model = ChatOllama(model=model_name, temperature=temperature)
        self.model = ContextExtractorModel(base_model)
        self.builder = StateGraph(ClarificationState)

    def generate_question(self, state: ClarificationState):
        """
        Generate exactly one clarification question.

        This node finishes before the interrupt node begins, so
        resuming the interrupt does not call the model again.
        """
        response = self.model.invoke(state["messages"])
        question = response.content

        return {
            "current_question": question,
            "messages": [
                AIMessage(content=question)
            ],
        }

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
        })

        clarification = Clarification(
            question=question,
            answer=str(answer),
        )

        return {
            "clarifications": (
                    state["clarifications"] + [clarification]
            ),
            "messages": [
                HumanMessage(content=str(answer))
            ],
            "current_question": None,
        }

    def build(self):
        self.builder.add_node("generate_question", self.generate_question)
        self.builder.add_node("ask_user", self.ask_user)
        self.builder.add_edge(START, "generate_question")
        self.builder.add_edge("generate_question", "ask_user")
        self.builder.add_edge("ask_user", END)

        memory = InMemorySaver()

        return self.builder.compile(checkpointer=memory)

def get_interrupt_payload(result: dict) -> dict:
    """
    Extract the first interrupt payload from graph_invoke() output.
    """

    interrupts = result.get("__interrupt__", [])

    if not interrupts:
        raise RuntimeError(
            "The graph did not return an interrupt."
        )

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
        "expanded_query": None,
    }

    paused_result  = graph.invoke(
        initial_state,
        config=config,
    )

    # ---------------------------------------------------------
    # Invocation 1:
    # generate_question runs, then ask_user reaches interrupt().
    # ---------------------------------------------------------

    interrupt_payload = get_interrupt_payload(paused_result )

    question = interrupt_payload["question"]

    print(f"\nClarification question: \n{question}")

    user_answer = input("\nYour answer: ").strip()

    # ---------------------------------------------------------
    # Invocation 2:
    # Resume the same thread and provide the user's answer.
    # ---------------------------------------------------------

    final_state = graph.invoke(
        Command(resume=user_answer),
        config=config,
    )

    print("\nClarifications recorded:")

    for clarification in final_state["clarifications"]:
        print(f"Q: {clarification.question}")
        print(f"A: {clarification.answer}")

    print("\nCurrent message history:")

    for message in final_state["messages"]:
        if message.__class__ != SystemMessage:
            print(
                f"{message.__class__.__name__}: "
                f"{message.content}"
            )