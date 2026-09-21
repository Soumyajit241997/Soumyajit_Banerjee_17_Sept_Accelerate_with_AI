"""Shared ReAct agent construction (LangGraph's create_react_agent).

Every specialist agent in RADAR is built the same way: an LLM, a system
prompt describing its role, and a set of closures-bound tools. Centralising
that here keeps each agent module focused on its tools and transformation
logic rather than boilerplate agent wiring.
"""
from __future__ import annotations

from langgraph.prebuilt import create_react_agent

from core.config import get_llm


def build_agent(system_prompt: str, tools: list):
    llm = get_llm()
    return create_react_agent(llm, tools, prompt=system_prompt)


def run_agent(agent, goal: str):
    """Invoke a ReAct agent with a single human goal message and return the
    full message history (used for both the final answer and observability
    tracing)."""
    result = agent.invoke({"messages": [("human", goal)]})
    return result["messages"]
