"""
LangGraph ReAct agent — reference implementation.

Copy the contents into src/agent.py to use this as your starting point.
Demonstrates both `run` (non-streaming) and `stream` (token-level streaming
via astream_events).
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in a string."""
    return len(text.split())


_llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

_graph = create_react_agent(
    _llm,
    tools=[add, word_count],
    prompt="You are a helpful assistant. Use the tools when they help answer the question.",
    checkpointer=MemorySaver(),
)


def _cfg(session_id):
    return {"configurable": {"thread_id": session_id}} if session_id else {}


def run(prompt: str, session_id: str | None = None) -> str:
    result = _graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=_cfg(session_id),
    )
    return result["messages"][-1].content


async def stream(prompt: str, session_id: str | None = None):
    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=_cfg(session_id),
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk
