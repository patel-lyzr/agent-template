"""
GCP Vertex Agent Engine runtime wrapper — DO NOT EDIT.

Agent Engine loads the `wrapper` instance at the bottom of this file, calls
`set_up()` once per cold start, then `query(...)` or `stream_query(...)` per
request.

User code in src/agent.py must define at least one of:
    def run(prompt: str, session_id: str | None = None) -> str
    async def stream(prompt: str, session_id: str | None = None)
"""


class AgentEngineWrapper:
    def __init__(self):
        # Must stay pickle-able. No heavy imports, no network clients here.
        pass

    def set_up(self):
        import os
        import sys

        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
        sys.path.insert(0, os.path.join(os.getcwd(), "src"))

        import agent as user_agent

        self._run = getattr(user_agent, "run", None)
        self._stream = getattr(user_agent, "stream", None)

        if self._run is None and self._stream is None:
            raise RuntimeError(
                "src/agent.py must define `run(prompt, session_id=None) -> str` "
                "and/or `async def stream(prompt, session_id=None)`."
            )

    def query(self, *, input: str = None, prompt: str = None, session_id: str = None, **kwargs) -> str:
        text = input or prompt or ""
        if self._run is not None:
            return self._run(text, session_id=session_id)
        # Only `stream` defined — drain it and join.
        return "".join(str(c) for c in self._drain_stream(text, session_id))

    def stream_query(self, *, input: str = None, prompt: str = None, session_id: str = None, **kwargs):
        """Sync generator (Agent Engine requirement) yielding chunks."""
        text = input or prompt or ""
        if self._stream is not None:
            yield from self._drain_stream(text, session_id)
        else:
            # Only `run` defined — emit the full result as a single chunk.
            yield self._run(text, session_id=session_id)

    def _drain_stream(self, text, session_id):
        """Run the user's async `stream()` to completion from a sync context."""
        import asyncio

        async def _collect():
            chunks = []
            async for c in self._stream(text, session_id=session_id):
                chunks.append(c)
            return chunks

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # Defensive: shouldn't happen in Agent Engine's worker, but handle it.
                import nest_asyncio
                nest_asyncio.apply()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(_collect())

    def register_operations(self):
        return {"": ["query"], "stream": ["stream_query"]}


wrapper = AgentEngineWrapper()
