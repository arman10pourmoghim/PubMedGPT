# tests/test_freshness.py
from app.ranking import freshness_score
def test_freshness_monotonic():
    assert freshness_score(2025, 2025, 5.0) > freshness_score(2015, 2025, 5.0)
