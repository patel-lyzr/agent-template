# Example agents

Each file in this directory is a drop-in replacement for `src/agent.py`.
Copy the one that matches your framework, rename it (or paste its contents)
into `src/agent.py`, then customize.

| File | Framework | Notes |
|---|---|---|
| [`langgraph_agent.py`](./langgraph_agent.py) | LangGraph (ReAct) | Token streaming via `astream_events`. **Default example.** |
| [`langchain_agent.py`](./langchain_agent.py) | LangChain `AgentExecutor` | OpenAI tools agent with streaming. |
| [`crewai_crew.py`](./crewai_crew.py) | CrewAI `Crew` | Sequential two-agent workflow. `run` only (CrewAI doesn't stream tokens). |
| [`crewai_flow.py`](./crewai_flow.py) | CrewAI `Flow` | `run` + `stream` — `stream` yields one dict per flow step. |
| [`openai_raw.py`](./openai_raw.py) | Raw OpenAI SDK (no framework) | Native `stream=True` for token streaming. |

## Dependencies

Add what the example needs to your **root** `requirements.txt`:

- LangGraph → `langchain-core`, `langchain-openai`, `langgraph`
- LangChain → `langchain`, `langchain-openai`
- CrewAI (both) → `crewai`
- OpenAI raw → `openai`

The runtime SDK (`bedrock-agentcore` / `google-cloud-aiplatform`) is already
handled by the cloud-specific `requirements.txt` under `aws/` and `gcp/`.

## Swapping in an example

```bash
# From the repo root:
cp src/examples/crewai_crew.py src/agent.py
```

Commit, push, redeploy.
