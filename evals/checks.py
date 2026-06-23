"""
Deterministic correctness checks for agent eval cases.

All functions are pure/sync and require no API key.
"""
from __future__ import annotations

import re
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage


def tools_called(messages: list) -> list[dict]:
    """Return [{name, args}, ...] for every tool call across the message history."""
    calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                calls.append({"name": tc["name"], "args": tc["args"]})
    return calls


def assert_tool_used(messages: list, allowed: set[str]) -> bool:
    """True if at least one tool from `allowed` was called (necessary condition, loose on path)."""
    called_names = {tc["name"] for tc in tools_called(messages)}
    return bool(called_names & allowed)


def extract_titles(reply: str) -> set[str]:
    """
    Extract candidate book titles from a reply string.

    Recommendations are formatted "Title - Author - Difficulty: N - desc" per the
    system prompt. They may appear as plain lines, or prefixed by a list marker.
    We capture the segment before the first dash, then strip markdown bold markers.
    """
    titles: set[str] = set()

    # Match optional list prefix (e.g. "1. " or "* " or "- "), then capture up to the first dash.
    # Also matches bare "Title - ..." lines with no list prefix.
    for match in re.finditer(r"^(?:[\d\-\*•]+\.?\s+)?(\*{0,2}[^:\n\-]{4,79}\*{0,2})\s+-", reply, re.MULTILINE):
        candidate = re.sub(r"\*+", "", match.group(1)).strip()
        if 3 < len(candidate) < 80:
            titles.add(candidate)

    return titles


def assert_grounded(titles: set[str]) -> tuple[bool, list[str]]:
    """
    True if every extracted title resolves to a real book in the local collection.
    Returns (passed, list_of_unresolved_titles).
    """
    from recommender.query import find_book

    unresolved = []
    for title in titles:
        if find_book(title) is None:
            unresolved.append(title)
    return (len(unresolved) == 0, unresolved)


def assert_tool_args(
    messages: list,
    predicate,
) -> tuple[bool, str]:
    """
    Run a case-supplied predicate over the list of tool calls.
    The predicate receives [{name, args}, ...] and returns (bool, reason_str).
    Use this to assert on specific argument values, e.g. that both seeds were passed.
    """
    calls = tools_called(messages)
    return predicate(calls)


def assert_difficulty_max(titles: set[str], max_diff: int) -> tuple[bool, list[str]]:
    """
    True if no resolved title has a difficulty above max_diff.
    Returns (passed, list_of_violating_titles).
    """
    from recommender.query import find_book

    violations = []
    for title in titles:
        book = find_book(title)
        if book and book.get("difficulty") and book["difficulty"] > max_diff:
            violations.append(f"{title} (difficulty={book['difficulty']})")
    return (len(violations) == 0, violations)
