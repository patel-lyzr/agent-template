"""
GCP Vertex Agent Engine runtime wrapper — DO NOT EDIT.

Agent Engine deploys this file as a serverless reasoning engine. It expects
a top-level instance (here: `wrapper`) with:

    set_up()                    — called once per cold-start container
    query(*, input, ...)        — sync method for non-streaming requests
    stream_query(*, input, ...) — sync generator for streaming requests
    register_operations()       — declares which methods to expose

The instance must be PICKLE-able at deploy time (Agent Engine ships it via
cloudpickle), which is why heavy imports live inside set_up(), not __init__.

User code lives in src/invocation.py and must define at least one of:
    def run(prompt: str, session_id: str | None = None) -> str
    async def stream(prompt: str, session_id: str | None = None)
"""


class AgentEngineWrapper:
    def __init__(self):
        # Keep this trivial — must stay pickle-able.
        # No DB clients, no LLM clients, no `import` of heavy libs.
        pass

    def set_up(self):
        """
        Runs once per container after unpickling. This is where we resolve
        sys.path, import the user's invocation module, and capture references
        to their run/stream functions.
        """
        import os
        import sys

        # Make `src/` importable. Two paths handle both layouts:
        # cwd = repo root (local) vs cwd = /app (Agent Engine worker).
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))

        import invocation as user_invocation

        self._run = getattr(user_invocation, "run", None)
        self._stream = getattr(user_invocation, "stream", None)

        if self._run is None and self._stream is None:
            raise RuntimeError(
                "src/invocation.py must define `run(prompt, session_id=None) -> str` "
                "and/or `async def stream(prompt, session_id=None)`."
            )

    def query(self, *, input: str = None, prompt: str = None, session_id: str = None, **kwargs) -> str:
        """
        Non-streaming endpoint — Agent Engine calls this for `class_method=query`.

        We accept both `input` (Agent Engine convention) and `prompt`
        (LangchainAgent convention) so callers from either ecosystem work.
        """
        text = input or prompt or ""
        if self._run is not None:
            return self._run(text, session_id=session_id)
        # User implemented only `stream` — drain it and join the chunks.
        return "".join(str(c) for c in self._drain_stream(text, session_id))

    def stream_query(self, *, input: str = None, prompt: str = None, session_id: str = None, **kwargs):
        """
        Streaming endpoint — Agent Engine calls this for
        `class_method=stream_query`. Must be a SYNC generator (not async).
        """
        text = input or prompt or ""
        if self._stream is not None:
            yield from self._drain_stream(text, session_id)
        else:
            # User implemented only `run` — emit the full result as one chunk.
            yield self._run(text, session_id=session_id)

    def _drain_stream(self, text, session_id):
        """
        Run the user's async `stream()` to completion from a sync context.

        Agent Engine workers are sync; LangChain/LangGraph streaming APIs are
        async. Bridge the gap by spinning up an event loop and collecting
        chunks. The cost is real-time-streaming → batched-streaming, but
        Agent Engine's stream_query response is line-delimited JSON anyway.
        """
        import asyncio

        async def _collect():
            chunks = []
            async for c in self._stream(text, session_id=session_id):
                chunks.append(c)
            return chunks

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Defensive: shouldn't happen in Agent Engine's worker,
                # but if it does, nest_asyncio lets us reuse the running loop.
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            # No loop in this thread — make a fresh one.
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_collect())

    def register_operations(self):
        """
        Tell Agent Engine which methods it can call.
        Empty string "" is the default class_method namespace; "stream" is the
        streaming namespace. This dict is what makes `:query` and `:streamQuery`
        REST endpoints route to the right method.
        """
        return {"": ["query"], "stream": ["stream_query"]}


# This top-level name is what Agent Engine pickles + ships at deploy time.
# `build_service/deploy_to_gcp.js` references `agent_engine_wrapper:wrapper`
# as the engine's entrypoint object.
wrapper = AgentEngineWrapper()
