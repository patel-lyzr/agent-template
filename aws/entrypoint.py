"""
AWS Bedrock AgentCore runtime wrapper — DO NOT EDIT.

Imports user-defined `run` and/or `stream` from src/agent.py and registers the
appropriate entrypoint. AgentCore streams when the entrypoint is an async
generator (yields), and returns one-shot JSON otherwise.
"""

import os
import sys

# Make `src/` importable.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import agent as user_agent  # src/agent.py

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext


_run = getattr(user_agent, "run", None)
_stream = getattr(user_agent, "stream", None)

if _run is None and _stream is None:
    raise RuntimeError(
        "src/agent.py must define `run(prompt, session_id=None) -> str` "
        "and/or `async def stream(prompt, session_id=None)`."
    )


app = BedrockAgentCoreApp()


if _stream is not None:
    # Streaming path — AgentCore auto-SSE when entrypoint is an async generator.
    @app.entrypoint
    async def invoke(payload, context: RequestContext):
        prompt = payload.get("prompt", "")
        async for chunk in _stream(prompt, session_id=context.session_id):
            yield chunk
else:
    # Non-streaming fallback.
    @app.entrypoint
    def invoke(payload, context: RequestContext):
        prompt = payload.get("prompt", "")
        return {
            "result": _run(prompt, session_id=context.session_id),
            "session_id": context.session_id,
        }


if __name__ == "__main__":
    app.run()
