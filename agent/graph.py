"""
Builds the LangGraph ReAct agent graph.
"""
from __future__ import annotations

import logging
import os

from langchain_core.messages import SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition

import config
from agent.prompts import SYSTEM_PROMPT
from agent.state import AgentState

logger = logging.getLogger(__name__)


async def build_graph(extra_tools: list = []):
    llm = ChatGoogleGenerativeAI(
        model=config.GEMINI_MODEL,
        google_api_key=os.environ["GOOGLE_API_KEY"],
        temperature=0.3,
    )
    all_tools = extra_tools
    llm_with_tools = llm.bind_tools(all_tools)
    tool_node = ToolNode(all_tools)

    async def agent_node(state: AgentState) -> dict:
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + state["messages"]
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


async def load_local_tools() -> tuple[list, object | None]:
    """Load the Jewish book guide tools from the local MCP server via stdio."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "jewish-books": {
                "command": "python",
                "args": ["-m", "mcp_server.server"],
                "transport": "stdio",
                "env": {**os.environ},
            }
        })
        tools = await client.get_tools()
        logger.info("Loaded %d local MCP tools: %s", len(tools), [t.name for t in tools])
        return tools, client
    except Exception as e:
        logger.warning("Failed to load local MCP tools: %s", e)
        return [], None


async def load_youtube_tools() -> tuple[list, object | None]:
    """Load YouTube MCP tools. Returns (tools, mcp_client) — caller must close client."""
    youtube_api_key = os.environ.get("YOUTUBE_API_KEY", "")
    if not youtube_api_key:
        logger.warning("YOUTUBE_API_KEY not set — YouTube search will be unavailable")
        return [], None

    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "youtube": {
                "command": "npx",
                "args": ["-y", "@kirbah/mcp-youtube"],
                "transport": "stdio",
                "env": {**os.environ, "YOUTUBE_API_KEY": youtube_api_key},
            }
        })
        all_tools = await client.get_tools()
        tools = [t for t in all_tools if t.name in {"searchVideos"}]
        logger.info("Loaded %d YouTube MCP tools: %s", len(tools), [t.name for t in tools])
        return tools, client
    except Exception as e:
        logger.warning("Failed to load YouTube MCP tools: %s", e)
        return [], None


async def load_sefaria_tools() -> tuple[list, object | None]:
    """Load Sefaria MCP tools. Returns (tools, mcp_client) — caller must close client."""
    try:
        from langchain_mcp_adapters.client import MultiServerMCPClient

        client = MultiServerMCPClient({
            "sefaria-texts": {
                "transport": "sse",
                "url": "https://mcp.sefaria.org/sse",
            }
        })
        tools = await client.get_tools()
        logger.info("Loaded %d Sefaria MCP tools: %s", len(tools), [t.name for t in tools])
        return tools, client
    except Exception as e:
        logger.warning("Failed to load Sefaria MCP tools: %s", e)
        return [], None
