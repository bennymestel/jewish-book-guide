"""
Eval harness: builds the agent graph once and runs a single user message through it.

Returns both the reply text and the full message history so that checks.py can inspect
tool calls and grounding without needing to re-run the graph.

Two graphs can be evaluated:
  - "simple": the flat ReAct graph (agent/graph.py) — the supervisor sees every raw tool.
  - "multi":  the supervisor graph (agent/multi_graph.py) — the supervisor only sees
              three consult_* wrapper tools; each wrapper runs a nested specialist
              ReAct agent whose own tool calls are NOT included in the top-level
              graph.ainvoke() message history.

For "multi", pass flatten_subagents=True to run_message/run_conversation so that the
nested specialists' real tool calls (get_recommendations, get_text, searchVideos, ...)
are reconstructed into the message history checks.py inspects. This is done by driving
the graph with astream_events instead of ainvoke, in a single pass (no extra LLM calls).
"""
from __future__ import annotations

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.graph import build_graph, load_books_tools, load_youtube_tools, load_sefaria_tools
from agent.multi_graph import build_multi_graph

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


async def build_eval_graph(mode: str = "simple"):
    """Build the agent graph the same way the server does at startup.

    mode="simple" (default): the flat ReAct graph, unchanged from before.
    mode="multi": the supervisor/agents-as-tools graph.
    """
    books_tools, _ = await load_books_tools()
    youtube_tools, _ = await load_youtube_tools()
    sefaria_tools, _ = await load_sefaria_tools()

    if mode == "multi":
        logger.info(
            "[harness] building multi-agent graph: books=%d sefaria=%d youtube=%d",
            len(books_tools), len(sefaria_tools), len(youtube_tools),
        )
        return await build_multi_graph(books_tools, sefaria_tools, youtube_tools)

    all_tools = books_tools + youtube_tools + sefaria_tools
    logger.info("[harness] loaded %d tools: %s", len(all_tools), [t.name for t in all_tools])
    return await build_graph(tools=all_tools)


async def _run_streamed(graph, state: dict) -> tuple[str, list, dict]:
    """
    Drive the graph via astream_events instead of ainvoke, in one pass, so nested
    specialist tool calls (hidden behind consult_* at the top level) are surfaced.
    Returns (reply, synthetic_tool_trace, real_final_state) — the real state is what
    callers should thread forward as conversation history; the trace is only for
    checks.py to inspect this turn's tool usage.
    """
    messages: list = []
    final_state = None

    async for event in graph.astream_events(state, version="v2"):
        kind = event["event"]
        if kind == "on_tool_start":
            name = event.get("name", "")
            run_id = str(event.get("run_id", ""))
            args = event["data"].get("input")
            if not isinstance(args, dict):
                args = {"input": args}
            messages.append(
                AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": run_id}])
            )
        elif kind == "on_tool_end":
            run_id = str(event.get("run_id", ""))
            output = event["data"].get("output")
            content = output.content if hasattr(output, "content") else str(output)
            messages.append(ToolMessage(content=content, tool_call_id=run_id))
        elif kind == "on_chain_end":
            output = event["data"].get("output")
            if isinstance(output, dict) and "messages" in output:
                final_state = output

    if final_state is None:
        final_state = state
    reply = _extract_reply(final_state["messages"])
    return reply, messages, final_state


async def run_message(graph, message: str, flatten_subagents: bool = False) -> tuple[str, list]:
    """
    Send a single user message to the graph.
    Returns (reply_text, full_message_history).
    The message history includes all AIMessage/ToolMessage turns so checks.py
    can inspect tool_calls and tool outputs.

    Set flatten_subagents=True (use for mode="multi") to reconstruct nested
    specialist tool calls into the returned history; see _run_streamed.
    """
    state = {"messages": [HumanMessage(content=message)]}
    if flatten_subagents:
        reply, messages, _ = await _run_streamed(graph, state)
        return reply, messages
    result = await graph.ainvoke(state)
    messages = result["messages"]
    reply = _extract_reply(messages)
    return reply, messages


async def run_conversation(
    graph, turns: list[str], flatten_subagents: bool = False
) -> tuple[str, list]:
    """
    Thread multiple user turns through the graph, carrying message history forward.
    Returns (final_reply_text, full_message_history_across_all_turns).
    Use for multi-turn eval cases (cases with "inputs" instead of "input").

    Set flatten_subagents=True (use for mode="multi") to reconstruct nested
    specialist tool calls for checks.py; only the last turn's trace is returned,
    consistent with the non-flattened path returning only the final message list.
    """
    all_messages: list = []
    reply = ""
    turn_messages: list = []
    for turn in turns:
        state = {"messages": all_messages + [HumanMessage(content=turn)]}
        if flatten_subagents:
            reply, turn_messages, final_state = await _run_streamed(graph, state)
            all_messages = final_state["messages"]
        else:
            result = await graph.ainvoke(state)
            all_messages = result["messages"]
            reply = _extract_reply(all_messages)
    if flatten_subagents:
        return reply, turn_messages
    return reply, all_messages