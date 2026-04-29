"""
Raw OpenAI SDK — reference implementation (no framework).

Copy into src/invocation.py. Demonstrates native streaming via the OpenAI
`stream=True` API.

Requires `openai` in your root requirements.txt.
"""

import os

from openai import AsyncOpenAI, OpenAI


_sync_client = OpenAI()
_async_client = AsyncOpenAI()
_model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")


def run(prompt: str, session_id: str | None = None) -> str:
    resp = _sync_client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content or ""


async def stream(prompt: str, session_id: str | None = None):
    resp = await _async_client.chat.completions.create(
        model=_model,
        messages=[{"role": "user", "content": prompt}],
        stream=True,
    )
    async for chunk in resp:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta
