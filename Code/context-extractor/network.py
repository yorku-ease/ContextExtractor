from typing import TypedDict

from langchain_core.language_models import BaseChatModel
from langchain.chat_models import init_chat_model
from langchain_ollama import ChatOllama
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState
from pydantic import BaseModel

SYSTEM_PROMPT = """
You are an intermediate tool between a user and a larger conversation model.
Your purpose is to analyze a user's prompt and try to derive additional context to help towards a more complete answer.
Do not attempt to answer the query itself.
Do not make any assumptions concerning the answer to the query.
Ask up to five clarification questions. 
If you think you cannot extract more context from the user, you can stop before asking five questions. In this case, simply say "I do not need any more information".
Pose the question directly to the user without any reasoning or explanation.
Expect a short answer from the user and keep the additional information in your memory.
When done, simply say "I do not need any more information".
"""


class ContextExtractorModel:
    model: BaseChatModel

    def __init__(self, model: BaseChatModel):
        self.model = model

    def invoke(self, text: str):
        return self.model.invoke(input=text)

class Clarification(BaseModel):
    question: str
    answer: str

class ClarificationState(MessagesState):
    original_query: str
    clarifications: list[Clarification]
    expanded_query: str | None

class ContextNetwork():
    model: ContextExtractorModel
    graph: StateGraph

    def __init__(self, model_name: str, temperature: int):
        self.graph = StateGraph(ClarificationState)
        base_model = ChatOllama(model=model_name, temperature=temperature)
        self.model = ContextExtractorModel(base_model)

    def call_model(self, state: ClarificationState):
        response = self.model.invoke(state["messages"])
        return {"messages": [response]}

    def build(self):
        self.graph.add_node("model", self.call_model)
        self.graph.add_edge(START, "model")
        self.graph.add_edge("model", END)

        return self.graph.compile()



if __name__ == "__main__":
    network = ContextNetwork(model_name="gemma4:e4b", temperature=0)
    graph = network.build()
    response = graph.invoke({
        "messages": [
            (
                "system",
                SYSTEM_PROMPT
            ),
            (
                "human",
                "I have a headache. What do you recommend I do?"
            )
        ],
        "original_query":
            "I have a headache. What do you recommend I do?",
        "clarifications": [],
        "expanded_query": None,
    })

    print(response)