# Template Guide

## The contract

`src/agent.py` must expose **at least one** of these two functions:

```python
def run(prompt: str, session_id: str | None = None) -> str:
    """Non-streaming. Return the full reply as a string."""

async def stream(prompt: str, session_id: str | None = None):
    """Streaming. Async generator yielding chunks (str or dict)."""
    async for chunk in ...:
        yield chunk
```

- Implement **only `run`** → the agent works everywhere. Streaming callers
  receive the full result as a single chunk.
- Implement **only `stream`** → streaming works; non-streaming callers get
  the joined chunks back as one string.
- Implement **both** → each caller uses its native path (best performance
  for both sync and stream users).

Framework choice is yours — LangGraph, LangChain, CrewAI, LlamaIndex, raw
OpenAI/Anthropic SDK, custom code. The wrappers only care about these two
signatures.

## Layout

```
src/                     your code (agent.py plus any helpers)
requirements.txt         your Python deps
aws/                     AgentCore wrapper + Dockerfile + AgentCore SDK deps
gcp/                     Agent Engine wrapper + Agent Engine SDK deps
```

## Requirements

Edit the **root** `requirements.txt` only. The build pipeline concatenates it
with the cloud-specific runtime deps (`aws/requirements.txt` or
`gcp/requirements.txt`) and installs the combined list. You never add
`bedrock-agentcore` or `google-cloud-aiplatform` yourself.

## Examples

### LangGraph (streaming + non-streaming)

```python
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent

_graph = create_react_agent(ChatOpenAI(model="gpt-4o-mini"), tools=[...])

def _cfg(session_id):
    return {"configurable": {"thread_id": session_id}} if session_id else {}

def run(prompt, session_id=None):
    r = _graph.invoke({"messages": [HumanMessage(content=prompt)]}, config=_cfg(session_id))
    return r["messages"][-1].content

async def stream(prompt, session_id=None):
    async for event in _graph.astream_events(
        {"messages": [HumanMessage(content=prompt)]},
        config=_cfg(session_id),
        version="v2",
    ):
        if event["event"] == "on_chat_model_stream":
            chunk = event["data"]["chunk"].content
            if chunk:
                yield chunk
```

### LangChain `AgentExecutor`

```python
from langchain.agents import AgentExecutor, create_openai_tools_agent
# ... build agent_executor ...

def run(prompt, session_id=None):
    return agent_executor.invoke({"input": prompt})["output"]
```

### CrewAI

```python
from crewai import Crew
# ... build crew ...

def run(prompt, session_id=None):
    return str(crew.kickoff(inputs={"prompt": prompt}))
```

### Raw OpenAI with streaming

```python
from openai import AsyncOpenAI
_client = AsyncOpenAI()

async def stream(prompt, session_id=None):
    resp = await _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
```

## Environment variables

Anything your code reads from `os.environ` must be provided at deploy time
via the UI. Common examples:

- `OPENAI_API_KEY`, `OPENAI_MODEL`
- `ANTHROPIC_API_KEY`
- `TAVILY_API_KEY`

Document them in `.env.example`.

## Session memory

`session_id` is the runtime's per-conversation identifier. Use it however
your framework wants — most commonly as a LangGraph `thread_id` with a
checkpointer:

```python
from langgraph.checkpoint.memory import MemorySaver

_graph = create_react_agent(llm, tools=[...], checkpointer=MemorySaver())
```

- **AWS**: `session_id` comes from `runtimeSessionId` in the invoke request.
- **GCP**: pass `session_id` in the request body's `input` object.

`MemorySaver` is in-process (resets on container recycle). For persistence
use `PostgresSaver` / `RedisSaver` with the DB URL as an env var.

## Observability (optional)

`src/agent.py` ships with a commented-out **Arize Phoenix** instrumentation
block. Uncomment it to capture every LLM call, tool call, and graph step as
OpenTelemetry traces — works with both Phoenix Cloud (free) and self-hosted
Phoenix.

Steps:
1. Add to root `requirements.txt`:
   ```
   arize-phoenix-otel
   openinference-instrumentation-langchain
   ```
2. Uncomment the `phoenix.otel.register(...)` block in `src/agent.py`.
3. Provide env vars at deploy time:
   - `PHOENIX_API_KEY`
   - `PHOENIX_COLLECTOR_ENDPOINT` (e.g. `https://app.phoenix.arize.com`)
   - `PHOENIX_PROJECT_NAME` (optional)

Same recipe works for CrewAI / LangChain / raw OpenAI — swap
`openinference-instrumentation-langchain` for the matching instrumentor
package (`-crewai`, `-openai`, etc.).

## Do NOT

- Edit `aws/entrypoint.py`, `aws/Dockerfile`, or `gcp/agent_engine_wrapper.py`.
- Bind to a specific port — the runtime handles that.
- Add a root-level `Dockerfile`. The one under `aws/` is ARM64 (required —
  AgentCore is Graviton-only).

## Invocation shapes

### AWS (Bedrock AgentCore)

Non-streaming:
```
POST https://bedrock-agentcore.<region>.amazonaws.com/runtimes/<id>/invocations
X-Amzn-Bedrock-AgentCore-Runtime-Session-Id: <uuid>

{"prompt": "what is 2+2?"}
```
Response: `{"result": "4", "session_id": "<uuid>"}`

Streaming: same URL, same body — if the agent implements `stream()`, the
response is Server-Sent Events (`data: <chunk>\n\n`).

### GCP (Vertex Agent Engine)

Non-streaming:
```
POST .../reasoningEngines/<id>:query
{"class_method": "query", "input": {"input": "what is 2+2?"}}
```

Streaming:
```
POST .../reasoningEngines/<id>:streamQuery
{"class_method": "stream_query", "input": {"input": "..."}}
```
Response: newline-delimited JSON.

## Local test

```bash
pip install -r requirements.txt -r aws/requirements.txt       # or gcp/

# Non-streaming
PYTHONPATH=src python -c "from agent import run; print(run('what is 2+2?'))"

# Streaming
PYTHONPATH=src python -c "
import asyncio
from agent import stream
async def main():
    async for c in stream('count to 3'):
        print(c, end='', flush=True)
asyncio.run(main())
"
```

Implement `run` and/or `stream`, push to GitHub, deploy.
