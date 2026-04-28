"""
LangChain AgentExecutor with OpenAI tool calling — reference implementation.

Copy into src/agent.py to use as your starting point. Demonstrates both `run`
and `stream` using `astream_events` on the executor.
"""

import os

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


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

_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. Use tools when they help."),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

_tools = [add, word_count]
_agent = create_openai_tools_agent(_llm, _tools, _prompt)
_executor = AgentExecutor(agent=_agent, tools=_tools, verbose=False)


def run(prompt: str, session_id: str | None = None) -> str:
    return _executor.invoke({"input": prompt})["output"]


async def stream(prompt: str, session_id: str | None = None):
    async for event in _executor.astream_events({"input": prompt}, version="v2"):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk
