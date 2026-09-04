"""
Unit tests for pure functions that require no DB or network access.

Covers:
- ingestion.embed.build_profile  — composite text profile for embeddings
- recommender.query._score       — re-ranking score calculation
"""
import sys
import os

# Allow importing project modules without installing the package
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from ingestion.embed import build_profile
from recommender.query import _score
from mcp_server.server import _escape_like


# ── build_profile ─────────────────────────────────────────────────────────────

def _book(**kwargs) -> dict:
    """Minimal book row with sensible defaults."""
    defaults = {
        "title_en": "Tanya",
        "author_en": "Rabbi Schneur Zalman of Liadi",
        "difficulty": 3,
        "themes": ["soul", "divine service", "repentance"],
        "desc_en": "The foundational text of Chabad Chasidut.",
        "desc_en_short": "Foundational Chabad text.",
    }
    return {**defaults, **kwargs}


def test_build_profile_includes_title():
    profile = build_profile(_book())
    assert "Tanya" in profile


def test_build_profile_includes_author():
    profile = build_profile(_book())
    assert "Rabbi Schneur Zalman of Liadi" in profile


def test_build_profile_includes_themes():
    profile = build_profile(_book())
    assert "soul" in profile


def test_build_profile_handles_missing_author():
    profile = build_profile(_book(author_en=None))
    assert "Tanya" in profile
    assert "None" not in profile


def test_build_profile_handles_missing_themes():
    profile = build_profile(_book(themes=None))
    # Should not raise; title still present
    assert "Tanya" in profile


# ── _score ────────────────────────────────────────────────────────────────────

def _candidate(**kwargs) -> dict:
    defaults = {
        "cosine_sim": 0.80,
        "category": "Chasidut",
        "subcategory": "Chabad",
        "difficulty": 3,
        "themes": ["soul", "divine service"],
    }
    return {**defaults, **kwargs}


def _seed(**kwargs) -> dict:
    defaults = {
        "category": "Chasidut",
        "subcategory": "Chabad",
        "themes": ["soul"],
    }
    return {**defaults, **kwargs}


def test_score_same_category_adds_bonus():
    import config
    # Use no subcategory so that's not a confounding factor
    base = _score(_candidate(subcategory=None), _seed(subcategory=None), difficulty_pref=None)
    other = _score(_candidate(category="Musar", subcategory=None), _seed(subcategory=None), difficulty_pref=None)
    assert abs((base - other) - config.WEIGHT_SAME_CATEGORY) < 1e-9


def test_score_difficulty_gap_applies_penalty():
    import config
    no_gap = _score(_candidate(difficulty=3), _seed(), difficulty_pref=3)
    two_gap = _score(_candidate(difficulty=5), _seed(), difficulty_pref=3)
    assert round(no_gap - two_gap, 6) == round(2 * config.WEIGHT_PER_DIFFICULTY, 6)


# ── _escape_like ──────────────────────────────────────────────────────────────

def test_escape_like_percent():
    assert _escape_like("100%") == "100\\%"


def test_escape_like_underscore():
    assert _escape_like("soul_search") == "soul\\_search"


def test_escape_like_backslash():
    assert _escape_like("a\\b") == "a\\\\b"


def test_escape_like_no_special_chars():
    assert _escape_like("prayer") == "prayer"


def test_escape_like_combined():
    assert _escape_like("100%_test\\end") == "100\\%\\_test\\\\end"


# ── _score ────────────────────────────────────────────────────────────────────

def test_score_theme_overlap_adds_bonus():
    import config
    two_overlap = _score(_candidate(themes=["soul", "prayer"]), _seed(themes=["soul", "prayer"]), difficulty_pref=None)
    no_overlap = _score(_candidate(themes=["soul", "prayer"]), _seed(themes=["humility"]), difficulty_pref=None)
    assert round(two_overlap - no_overlap, 6) == round(2 * config.WEIGHT_PER_THEME, 6)


# ── _score: cross-encoder term ─────────────────────────────────────────────────

def test_score_cross_score_none_matches_no_cross_encoder():
    """Omitting cross_score (the default) must score identically to passing None
    explicitly — this is what makes an absent user_query a no-op."""
    with_default = _score(_candidate(), _seed(), difficulty_pref=None)
    with_none = _score(_candidate(), _seed(), difficulty_pref=None, cross_score=None)
    assert with_default == with_none


def test_score_higher_cross_score_wins_at_equal_cosine():
    low = _score(_candidate(), _seed(), difficulty_pref=None, cross_score=-5.0)
    high = _score(_candidate(), _seed(), difficulty_pref=None, cross_score=5.0)
    assert high > low


def test_score_cross_score_adds_sigmoid_weighted_term():
    import config
    from recommender.query import _sigmoid
    base = _score(_candidate(), _seed(), difficulty_pref=None)
    with_cross = _score(_candidate(), _seed(), difficulty_pref=None, cross_score=2.0)
    expected = config.WEIGHT_CROSS_ENCODER * _sigmoid(2.0)
    assert round(with_cross - base, 6) == round(expected, 6)


def test_score_equal_cross_scores_preserve_cosine_order():
    """A flat cross_score across candidates (e.g. all near sigmoid's saturated
    tail, as with a vague query) must not flip the order cosine alone gives."""
    weaker = _score(_candidate(cosine_sim=0.70), _seed(), difficulty_pref=None, cross_score=-9.0)
    stronger = _score(_candidate(cosine_sim=0.80), _seed(), difficulty_pref=None, cross_score=-9.0)
    assert stronger > weaker
