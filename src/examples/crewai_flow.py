"""
CrewAI Flow — reference implementation.

Copy into src/invocation.py. Flows support async kickoff, so we can wire both
`run` (sync wrapper) and `stream` (yields per-step progress as each listener
fires).

Requires `crewai` in your root requirements.txt.
"""

import asyncio
import os
from typing import AsyncIterator

from crewai import Agent, Task
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel


_agent = Agent(
    role="Assistant",
    goal="Answer the user's question clearly.",
    backstory="Helpful generalist.",
    llm=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    verbose=False,
    allow_delegation=False,
)


class _State(BaseModel):
    prompt: str = ""
    draft: str = ""
    final: str = ""


class QAFlow(Flow[_State]):
    @start()
    def take_prompt(self):
        return self.state.prompt

    @listen(take_prompt)
    def draft(self, prompt: str):
        task = Task(
            description=f"Write a short draft answer to: {prompt}",
            expected_output="A 2-3 sentence draft.",
            agent=_agent,
        )
        self.state.draft = str(task.execute_sync())
        return self.state.draft

    @listen(draft)
    def polish(self, draft: str):
        task = Task(
            description=f"Polish this draft into a clear final answer:\n\n{draft}",
            expected_output="The polished answer.",
            agent=_agent,
        )
        self.state.final = str(task.execute_sync())
        return self.state.final


def run(prompt: str, session_id: str | None = None) -> str:
    flow = QAFlow()
    flow.state.prompt = prompt
    return str(flow.kickoff())


async def stream(prompt: str, session_id: str | None = None) -> AsyncIterator[dict]:
    """Yields a dict per flow step so callers see progress."""
    flow = QAFlow()
    flow.state.prompt = prompt

    # Run kickoff in a thread so we can observe state between steps.
    loop = asyncio.get_event_loop()
    task = loop.run_in_executor(None, flow.kickoff)

    emitted = {"draft": False, "final": False}
    while not task.done():
        await asyncio.sleep(0.25)
        if flow.state.draft and not emitted["draft"]:
            emitted["draft"] = True
            yield {"step": "draft", "content": flow.state.draft}
        if flow.state.final and not emitted["final"]:
            emitted["final"] = True
            yield {"step": "final", "content": flow.state.final}

    # Ensure we emit the final even if the loop missed it.
    if not emitted["final"] and flow.state.final:
        yield {"step": "final", "content": flow.state.final}
