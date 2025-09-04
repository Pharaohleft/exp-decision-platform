from src.stats.srm import srm_from_counts


def test_srm_balanced():
    out = srm_from_counts(500, 500, ratioA=0.5)
    assert out["flag"] is False
    assert 0 <= out["p"] <= 1
