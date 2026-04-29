# Universal Agent Template

> One template. Any framework. Any cloud. Powered by **Lyzr Control Plane**.

A starter for deploying any LangChain / LangGraph / CrewAI / custom Python
agent to **AWS Bedrock AgentCore** or **GCP Vertex Agent Engine** — same
`src/` code, both clouds.

## Layout

```
.
├── src/
│   └── invocation.py            ← YOU EDIT — implements run() / stream()
├── requirements.txt             ← YOU EDIT — your deps
├── aws/
│   ├── entrypoint.py            ← managed (AgentCore wrapper)
│   ├── Dockerfile               ← managed (ARM64, port 8080)
│   └── requirements.txt         ← managed (AgentCore runtime deps)
├── gcp/
│   ├── agent_engine_wrapper.py  ← managed (Agent Engine wrapper)
│   └── requirements.txt         ← managed (Agent Engine runtime deps)
├── .env.example
├── README.md
└── TEMPLATE_GUIDE.md
```

## Quickstart

1. Click **Use this template** (or fork/clone this repo).
2. Edit [`src/invocation.py`](./src/invocation.py) — implement `run` and/or `stream`.
   Reference framework examples live in [`src/examples/`](./src/examples/).
3. Add any extra deps to the root [`requirements.txt`](./requirements.txt).
4. Copy `.env.example` → `.env` for local testing.
5. Push to your own GitHub repo.
6. In the **Lyzr Control Plane** UI: paste the repo URL, pick AWS or GCP,
   provide env vars, hit **Deploy**.

## The only rule

`src/invocation.py` must expose at least one of:

```python
def run(prompt: str, session_id: str | None = None) -> str: ...
async def stream(prompt: str, session_id: str | None = None): ...
```

Whatever framework you use inside is up to you. See
[TEMPLATE_GUIDE.md](./TEMPLATE_GUIDE.md) for the details and examples.

## Requirements

Put your dependencies in the **root** `requirements.txt`. At build time, the
build pipeline concatenates it with the cloud-specific runtime deps under
`aws/requirements.txt` or `gcp/requirements.txt` and installs the combined
list. You never need to add `bedrock-agentcore` or `google-cloud-aiplatform`
yourself.

## Local test

```bash
pip install -r requirements.txt -r aws/requirements.txt     # or gcp/
PYTHONPATH=src python -c "from invocation import run; print(run('what is 2+2?'))"
```

## Files you should NOT edit

| File | Why |
|---|---|
| `aws/entrypoint.py` | AgentCore wrapper that calls your `run`/`stream` |
| `aws/Dockerfile` | ARM64 container that AgentCore requires |
| `aws/requirements.txt` | AgentCore SDK deps |
| `gcp/agent_engine_wrapper.py` | Agent Engine wrapper that calls your `run`/`stream` |
| `gcp/requirements.txt` | Agent Engine SDK deps |

Everything else — `src/`, root `requirements.txt`, `.env.example` — is yours.
