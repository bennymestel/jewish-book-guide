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

    Recommendations are formatted "Title - Author - Difficulty: N - desc", usually
    rendered as a numbered or bulleted list. On each such line we capture the segment
    before the first dash/colon, yielding candidate strings to check for grounding.
    """
    titles: set[str] = set()

    # Numbered or bulleted list items: capture segment before the first dash/colon/newline.
    for match in re.finditer(r"^[\d\-\*•]+\.?\s+([^:\n\-]+)", reply, re.MULTILINE):
        candidate = match.group(1).strip()
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
