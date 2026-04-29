"""
Your invocation logic lives here.

This file is the ONE thing you edit to ship a deploy. The runtime wrappers
(aws/entrypoint.py, gcp/agent_engine_wrapper.py) import these functions and
hand off requests — you never touch the cloud-specific glue.

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
# Each `@tool` decorator turns a Python function into a LangChain Tool the
# agent can call. The docstring is shown to the LLM as the tool's description,
# and the type hints become its JSON-schema parameter spec — keep both clean.
@tool
def add(a: float, b: float) -> float:
    """Add two numbers and return the sum."""
    return a + b


@tool
def word_count(text: str) -> int:
    """Count the number of whitespace-separated words in a string."""
    return len(text.split())


# ── Model + graph (built once at import time) ───────────────────────────────
# Heavy-lifting that doesn't change per request goes at module scope so it
# runs once on container start, not on every invocation. Both AgentCore and
# Agent Engine reuse the same module across many requests in a hot worker.

# OPENAI_API_KEY (and any other secrets) are provided at deploy time via the
# UI. If you want a different model, set OPENAI_MODEL — falls back to
# gpt-4o-mini for cost-friendly local testing.
_llm = ChatOpenAI(
    model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    temperature=0,
)

# `create_react_agent` is LangGraph's prebuilt ReAct loop: call LLM, optionally
# call a tool, feed the tool's result back, repeat until the LLM stops asking
# for tools. Returns a compiled graph exposing .invoke / .astream / .astream_events.
#
# `checkpointer=MemorySaver()` enables per-thread conversation memory. The
# wrappers feed `session_id` in as `thread_id` (see _config below), so multiple
# turns in the same session share state automatically. MemorySaver is in-process
# only — swap for PostgresSaver/RedisSaver if you need persistence.
_graph = create_react_agent(
    _llm,
    tools=[add, word_count],
    prompt="You are a helpful assistant. Use the tools when they help answer the question.",
    checkpointer=MemorySaver(),
)


def _config(session_id: str | None):
    """
    Build the LangGraph config for a request.

    The runtime gives us a `session_id` per conversation (AWS:
    `runtimeSessionId`; GCP: caller-provided). We forward it as `thread_id`
    so MemorySaver can scope memory to the right conversation.
    """
    return {"configurable": {"thread_id": session_id}} if session_id else {}


# ── Non-streaming ───────────────────────────────────────────────────────────
def run(prompt: str, session_id: str | None = None) -> str:
    """
    Handle one invocation and return the full reply as a string.

    Called by:
      - aws/entrypoint.py for non-streaming /invocations requests
      - gcp/agent_engine_wrapper.py for `class_method=query`
    """
    # `messages` is the standard LangGraph state shape. `HumanMessage` is the
    # canonical user-turn marker; downstream nodes inspect message types.
    result = _graph.invoke(
        {"messages": [HumanMessage(content=prompt)]},
        config=_config(session_id),
    )
    # Final assistant reply is always the last message in the state.
    return result["messages"][-1].content


# ── Streaming ───────────────────────────────────────────────────────────────
async def stream(prompt: str, session_id: str | None = None):
    """
    Yield chunks as LangGraph produces them.

    Called by:
      - aws/entrypoint.py when registered as an async-generator @app.entrypoint
        (AgentCore turns it into Server-Sent Events automatically)
      - gcp/agent_engine_wrapper.py for `class_method=stream_query`
    """
    # `astream_events` emits typed events for every node, every LLM call,
    # every tool call. We filter to `on_chat_model_stream` to surface only
    # token deltas from the LLM. Tweak the filter if you also want tool
    # calls / intermediate state visible to the caller.
    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=_config(session_id),
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            # Skip empty deltas — common at the start/end of a stream.
            if chunk:
                yield chunk
