"""
CrewAI Crew — reference implementation.

Copy into src/agent.py. CrewAI's `kickoff()` is synchronous and doesn't expose
token-level streaming, so this example implements only `run`. The wrapper
automatically makes streaming callers receive the full result as one chunk.

Requires `crewai` in your root requirements.txt.
"""

import os

from crewai import Agent, Crew, Process, Task


_researcher = Agent(
    role="Researcher",
    goal="Find concise, accurate information relevant to the user's question.",
    backstory="Seasoned analyst who returns terse, factual answers.",
    llm=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    verbose=False,
    allow_delegation=False,
)

_writer = Agent(
    role="Writer",
    goal="Turn the researcher's notes into a short, clear reply for the user.",
    backstory="Professional copywriter who prizes clarity over length.",
    llm=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    verbose=False,
    allow_delegation=False,
)


def _build_crew(prompt: str) -> Crew:
    research_task = Task(
        description=f"Research and summarize: {prompt}",
        expected_output="A short factual summary (2-4 sentences).",
        agent=_researcher,
    )
    write_task = Task(
        description="Rewrite the research summary as a direct answer to the user.",
        expected_output="A polished reply addressed to the user.",
        agent=_writer,
        context=[research_task],
    )
    return Crew(
        agents=[_researcher, _writer],
        tasks=[research_task, write_task],
        process=Process.sequential,
        verbose=False,
    )


def run(prompt: str, session_id: str | None = None) -> str:
    result = _build_crew(prompt).kickoff(inputs={"prompt": prompt})
    return str(result)
