from src.pipeline import run_pipeline
from src.pipeline import run_pipeline_with_generation


def test_pipeline_runs_with_minimal_input():
    answers = {
        "description": "douleur thoracique et essoufflement",
        "intensite": 4,
        "duree": "1-3 jours",
        "localisation": "poitrine",
        "red_flags": ["douleur intense"],
    }
    out = run_pipeline(answers, ref_path="data/medical_referential.csv")
    assert "top3" in out
    assert len(out["top3"]) == 3
    assert out["user_text"]
    assert "retrieved" in out
    assert len(out["retrieved"]) > 0


def test_pipeline_with_generation_uses_cache_and_mock(monkeypatch, tmp_path):
    answers = {
        "description": "douleur thoracique",
        "intensite": 3,
        "duree": "1-3 jours",
        "localisation": "poitrine",
        "red_flags": [],
    }

    # Mock provider to avoid network/API calls
    import src.genai as genai

    calls = {"count": 0}

    def fake_provider(context):
        calls["count"] += 1
        return "texte généré de test"

    monkeypatch.setattr(genai, "generate_with_provider", fake_provider)

    cache_path = tmp_path / "cache.json"
    out1 = run_pipeline_with_generation(answers, ref_path="data/medical_referential.csv", cache_path=cache_path)
    out2 = run_pipeline_with_generation(answers, ref_path="data/medical_referential.csv", cache_path=cache_path)

    assert out1["genai_text"] == "texte généré de test"
    assert out2["genai_text"] == "texte généré de test"
    # provider should be called only once thanks to cache
    assert calls["count"] == 1
