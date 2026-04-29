"""
AWS Bedrock AgentCore runtime wrapper — DO NOT EDIT.

This file boots a Starlette HTTP server (provided by the bedrock-agentcore
SDK) on port 8080 and registers ONE entrypoint that AgentCore Runtime calls
at /invocations. We import the user's code from src/invocation.py and adapt
its run/stream functions to the AgentCore contract:

  * Sync function returning a dict  → one-shot JSON response
  * Async generator (yields)        → AgentCore auto-converts to SSE

We pick streaming if the user implemented `stream`, otherwise non-streaming.
"""

import os
import sys

# Make `src/` importable. We add two paths so this works whether the file is
# imported with cwd at repo root (local `python aws/entrypoint.py`) or from
# inside the container (cwd = /app, where src/ is at /app/src/).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import invocation as user_invocation  # src/invocation.py — defined by the user

from bedrock_agentcore import BedrockAgentCoreApp
from bedrock_agentcore.runtime.context import RequestContext


# Pull the user's two optional functions. Either may be None — we validate
# below that at least one is defined.
_run = getattr(user_invocation, "run", None)
_stream = getattr(user_invocation, "stream", None)

if _run is None and _stream is None:
    raise RuntimeError(
        "src/invocation.py must define `run(prompt, session_id=None) -> str` "
        "and/or `async def stream(prompt, session_id=None)`."
    )


# `BedrockAgentCoreApp` wires up /invocations (POST), /ping (GET), and
# auto-SSE conversion for async-generator entrypoints.
app = BedrockAgentCoreApp()


# Pick the entrypoint shape based on what the user implemented. We register
# only ONE @app.entrypoint — the SDK can't have both sync + streaming at once.
if _stream is not None:
    # Streaming path. Each `yield` becomes one SSE `data:` frame downstream.
    @app.entrypoint
    async def invoke(payload, context: RequestContext):
        prompt = payload.get("prompt", "")
        async for chunk in _stream(prompt, session_id=context.session_id):
            yield chunk
else:
    # Non-streaming fallback. The dict is JSON-serialized as the response body.
    @app.entrypoint
    def invoke(payload, context: RequestContext):
        prompt = payload.get("prompt", "")
        return {
            "result": _run(prompt, session_id=context.session_id),
            "session_id": context.session_id,
        }


if __name__ == "__main__":
    # `app.run()` starts the server on PORT (default 8080). The Dockerfile's
    # CMD runs this file directly, which is how AgentCore boots the agent.
    app.run()
