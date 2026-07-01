"""
Builds the LangGraph ReAct agent graph.
"""
from __future__ import annotations

import logging
import os

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

import config
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def build_graph(tools: list = [], system_prompt: str = SYSTEM_PROMPT):
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
    )
    llm_with_tools = llm.bind_tools(tools)
    tool_node = ToolNode(tools)

    async def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=system_prompt)] + state["messages"]
        response = await llm_with_tools.ainvoke(messages)
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                logger.info("[AGENT] calling tool: %s  args=%r", tc["name"], tc["args"])
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tool_node)
    graph.set_entry_point("agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")

    return graph.compile()


async def _load_mcp_tools(
    server_name: str,
    server_config: dict,
    tool_filter: set[str] | None = None,
) -> tuple[list, object | None]:
    try:
        client = MultiServerMCPClient({server_name: server_config})
        all_tools = await client.get_tools()
        tools = [t for t in all_tools if t.name in tool_filter] if tool_filter else all_tools
        logger.info("Loaded %d %s MCP tools: %s", len(tools), server_name, [t.name for t in tools])
        return tools, client
    except Exception as e:
        logger.warning("Failed to load %s MCP tools: %s", server_name, e)
        return [], None


async def load_youtube_tools() -> tuple[list, object | None]:
    youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not youtube_api_key:
        logger.warning("YOUTUBE_API_KEY not set — YouTube search will be unavailable")
        return [], None
    return await _load_mcp_tools(
        "youtube",
        {
            "command": "npx",
            "args": ["-y", "@kirbah/mcp-youtube"],
            "transport": "stdio",
            "env": {**os.environ, "YOUTUBE_API_KEY": youtube_api_key},
        },
        tool_filter={"searchVideos"},
    )


async def load_books_tools() -> tuple[list, object | None]:
    url = os.getenv("BOOKS_MCP_URL", "http://localhost:8001/mcp")
    return await _load_mcp_tools(
        "books",
        {"transport": "streamable_http", "url": url},
    )


async def load_sefaria_tools() -> tuple[list, object | None]:
    return await _load_mcp_tools(
        "sefaria-texts",
        {"transport": "sse", "url": "https://mcp.sefaria.org/sse"},
    )
