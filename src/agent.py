"""
Your agent lives here.

Implement ONE or BOTH of these functions — the runtime picks the right one:

    def run(prompt: str, session_id: str | None = None) -> str:
        '''Non-streaming. Return the full reply as a string.'''

    async def stream(prompt: str, session_id: str | None = None):
        '''Streaming. Async generator yielding chunks (str or dict).'''
        async for chunk in ...:
            yield chunk

If you only implement `stream`, `run` is auto-derived (chunks are joined).
If you only implement `run`, streaming callers get the full result as one chunk.

Put whatever framework logic you want inside — LangGraph, CrewAI, LangChain,
LlamaIndex, raw API calls. The example below is a LangGraph ReAct agent.
"""

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent


# ── Observability (OPTIONAL — uncomment to enable) ──────────────────────────
# Arize Phoenix tracing for LangChain/LangGraph. Captures every LLM call,
# tool call, and graph step as OpenTelemetry spans. Two backends:
#
#   1) Arize Phoenix Cloud — free, hosted. Sign up at https://app.phoenix.arize.com
#      and grab PHOENIX_API_KEY + PHOENIX_COLLECTOR_ENDPOINT.
#   2) Self-hosted Phoenix — `docker run -p 6006:6006 arizephoenix/phoenix`.
#
# Add to root requirements.txt:
#     arize-phoenix-otel
#     openinference-instrumentation-langchain
#
# Then uncomment the block below. Env vars to set at deploy time:
#     PHOENIX_API_KEY=...
#     PHOENIX_COLLECTOR_ENDPOINT=https://app.phoenix.arize.com   # or your host
#     PHOENIX_PROJECT_NAME=my-agent                              # optional
#
# from phoenix.otel import register
# from openinference.instrumentation.langchain import LangChainInstrumentor
#
# _tracer_provider = register(
#     project_name=os.environ.get("PHOENIX_PROJECT_NAME", "agent"),
#     endpoint=os.environ.get("PHOENIX_COLLECTOR_ENDPOINT"),
#     auto_instrument=False,
# )
# LangChainInstrumentor().instrument(tracer_provider=_tracer_provider)


# ── Tools ───────────────────────────────────────────────────────────────────
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in a string."""
    return len(text.split())


# ── Model + graph (built once at import time) ───────────────────────────────
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


def _config(session_id: str | None):
    return {"configurable": {"thread_id": session_id}} if session_id else {}


# ── Non-streaming ───────────────────────────────────────────────────────────
def run(prompt: str, session_id: str | None = None) -> str:
    """Handle one invocation. Return the full reply as a string."""
    result = _graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=_config(session_id),
    )
    return result["messages"][-1].content


# ── Streaming ───────────────────────────────────────────────────────────────
async def stream(prompt: str, session_id: str | None = None):
    """Yield chunks as LangGraph produces them."""
    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=_config(session_id),
        version="v2",
    ):
        # Emit only token deltas from the final LLM — tweak to taste.
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk
