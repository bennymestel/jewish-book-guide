"""
Multi-agent supervisor graph for the Jewish Book Guide.

Architecture:
  SUPERVISOR (existing ReAct StateGraph, tools = [consult_books, consult_sefaria, consult_youtube])
      │                   │                   │
  Books agent         Sefaria agent       YouTube agent
  (create_react_agent, 4 books tools)
                      (create_react_agent, all Sefaria tools)
                                          (create_react_agent, searchVideos)

Each specialist is a full ReAct agent exposed to the supervisor as a single async tool.
The supervisor never sees raw MCP tools — only the three consult_* wrappers.
"""
from __future__ import annotations

import os

from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

import config
from agent.graph import LLM_TIMEOUT_SECONDS, build_graph
from agent.prompts_multi import (
    BOOKS_AGENT_PROMPT,
    SEFARIA_AGENT_PROMPT,
    SUPERVISOR_PROMPT,
    YOUTUBE_AGENT_PROMPT,
)


async def build_multi_graph(
    books_tools: list,
    sefaria_tools: list,
    youtube_tools: list,
):
    """Build and return the supervisor multi-agent graph.

    Each specialist is created with create_react_agent (prebuilt) and a focused
    prompt, then wrapped as a single async tool the supervisor can call.
    The supervisor itself is the existing raw ReAct StateGraph (build_graph),
    re-used with a different system prompt and the three consult_* tools.

    Returns a compiled graph with the same AgentState shape as the simple graph,
    so session handling in server.py is identical across modes.
    """
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
        timeout=LLM_TIMEOUT_SECONDS,
    )

    # ── Specialist agents ──────────────────────────────────────────────────────
    books_agent = create_react_agent(llm, books_tools, prompt=BOOKS_AGENT_PROMPT)
    sefaria_agent = create_react_agent(llm, sefaria_tools, prompt=SEFARIA_AGENT_PROMPT)
    youtube_agent = create_react_agent(llm, youtube_tools, prompt=YOUTUBE_AGENT_PROMPT)

    # ── Consult tools (one per specialist) ────────────────────────────────────
    # Async so that when the supervisor emits multiple independent tool calls in
    # one turn, LangGraph's ToolNode can run them concurrently.

    @tool
    async def consult_books(request: str) -> str:
        """Recommend, look up, browse, or theme-search books in the curated Jewish collection."""
        result = await books_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return result["messages"][-1].content

    @tool
    async def consult_sefaria(request: str) -> str:
        """Fetch or search actual Jewish text passages from the Sefaria library. Include the title or reference in the request."""
        result = await sefaria_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return result["messages"][-1].content

    @tool
    async def consult_youtube(request: str) -> str:
        """Find YouTube shiurim and lectures. Include the book title or topic in the request."""
        result = await youtube_agent.ainvoke({"messages": [HumanMessage(content=request)]})
        return result["messages"][-1].content

    # ── Assemble supervisor ────────────────────────────────────────────────────
    consult_tools = [consult_books, consult_sefaria]
    if youtube_tools:
        # Only include the YouTube specialist when the API key is present;
        # otherwise the supervisor won't try to call a dead specialist.
        consult_tools.append(consult_youtube)

    # The supervisor is the existing ReAct StateGraph with the consult_* tools
    # bound in place of the raw MCP tools.
    return await build_graph(tools=consult_tools, system_prompt=SUPERVISOR_PROMPT)
