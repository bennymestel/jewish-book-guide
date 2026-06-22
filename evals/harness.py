"""
Eval harness: builds the agent graph once and runs a single user message through it.

Returns both the reply text and the full message history so that checks.py can inspect
tool calls and grounding without needing to re-run the graph.
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage

from agent.graph import build_graph, load_books_tools, load_youtube_tools, load_sefaria_tools

logger = logging.getLogger(__name__)


def _extract_reply(messages: list) -> str:
    """Extract plain text from the final message in a message list.
    Mirrors the extraction logic in agent/server.py — keep in sync."""
    last = messages[-1]
    content = last.content if hasattr(last, "content") else str(last)
    if isinstance(content, list):
        return "".join(
            block["text"] for block in content if isinstance(block, dict) and "text" in block
        )
    return content or ""


async def build_eval_graph():
    """Build the full agent graph the same way the server does at startup."""
    books_tools, _ = await load_books_tools()
    youtube_tools, _ = await load_youtube_tools()
    sefaria_tools, _ = await load_sefaria_tools()
    all_tools = books_tools + youtube_tools + sefaria_tools
    logger.info("[harness] loaded %d tools: %s", len(all_tools), [t.name for t in all_tools])
    return await build_graph(tools=all_tools)


async def run_message(graph, message: str) -> tuple[str, list]:
    """
    Send a single user message to the graph.
    Returns (reply_text, full_message_history).
    The message history includes all AIMessage/ToolMessage turns so checks.py
    can inspect tool_calls and tool outputs.
    """
    state = {"messages": [HumanMessage(content=message)]}
    result = await graph.ainvoke(state)
    messages = result["messages"]
    reply = _extract_reply(messages)
    return reply, messages
