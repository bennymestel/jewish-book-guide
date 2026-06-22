"""
Agent eval cases.

Each case declares:
  id              — unique slug
  input           — user message sent to the agent
  required_tools  — set of tool names; at least one must have been called (None = no check)
  max_difficulty  — if set, no recommended title may exceed this difficulty level
  expect_grounded — if True, every extracted title in the reply must exist in the local DB
"""
from __future__ import annotations

CASES: list[dict] = [
    {
        "id": "seed_recommendation",
        "input": "I loved Mesillat Yesharim. What should I read next?",
        "required_tools": {"get_recommendations"},
        "max_difficulty": None,
        "expect_grounded": True,
    },
    {
        "id": "multi_seed_recommendation",
        # Tests that get_recommendations handles a list of seeds, not just one —
        # a common failure point where agents only use the first seed.
        "input": "I loved both Tanya and Mesillat Yesharim. What should I read next?",
        "required_tools": {"get_recommendations"},
        "max_difficulty": None,
        "expect_grounded": True,
    },
    {
        "id": "theme_search_teshuvah",
        "input": "I want to learn about teshuvah.",
        "required_tools": {"search_by_theme"},
        "max_difficulty": None,
        "expect_grounded": True,
    },
    {
        "id": "category_and_difficulty_filter",
        "input": "Recommend an intermediate-level Musar book for me.",
        "required_tools": {"get_recommendations", "browse_collection"},
        "max_difficulty": 3,
        "expect_grounded": True,
    },
    {
        "id": "beginner_filtering",
        "input": "I am completely new to Jewish texts. Where do I start?",
        "required_tools": {"browse_collection"},
        "max_difficulty": 2,
        "expect_grounded": True,
    },
    {
        "id": "specific_lookup_local",
        "input": "Tell me about the book Tanya — who wrote it and what is it about?",
        "required_tools": {"lookup_book"},
        "max_difficulty": None,
        "expect_grounded": True,
    },
    {
        "id": "sefaria_fallback",
        # Vayikra (Leviticus) is a core Sefaria text but will never be in the curated
        # Chasidut/Musar/Jewish Thought local collection. The agent should fall back to
        # the Sefaria MCP tools (get_text or get_text_catalogue_info) to answer.
        "input": "Can you tell me about the book of Vayikra and what it covers?",
        "required_tools": {"get_text_catalogue_info", "get_text"},
        "max_difficulty": None,
        "expect_grounded": False,
    },
    {
        "id": "unknown_book",
        # Deliberately nonexistent title — tests graceful not-found handling without hallucination.
        # Grounding check catches invention: if the agent makes up metadata, extract_titles will
        # surface names that don't resolve in the DB.
        "input": "Can you look up a book called 'Sefer HaDavar HaLo Kayam'?",
        "required_tools": {"lookup_book", "get_text_catalogue_info"},
        "max_difficulty": None,
        "expect_grounded": True,
    },
    {
        "id": "reading_plan_beginner",
        "input": "Can you create a reading plan on the topic of prayer for a complete beginner?",
        "required_tools": {"search_by_theme", "browse_collection"},
        "max_difficulty": 2,
        "expect_grounded": True,
    },
]
