from src.scoring import compute_numeric_score


def test_numeric_score_bounds():
    answers = {"intensite": 5, "duree": "chronique", "red_flags": ["douleur thoracique"]}
    score = compute_numeric_score(answers)
    assert 0 <= score <= 1


def test_numeric_score_increases_with_intensity():
    low = compute_numeric_score({"intensite": 1, "duree": "moins de 24h", "red_flags": []})
    high = compute_numeric_score({"intensite": 5, "duree": "chronique", "red_flags": []})
    assert high > low
