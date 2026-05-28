from citations import build_numbered_sources, enforce_citations, validate_citations


def test_build_numbered_sources():
    passages = [{"passage_id": "p0", "source_type": "chunk", "source_id": "c1", "text": "Hello world " * 5, "page": 1}]
    sources, block = build_numbered_sources(passages)
    assert len(sources) == 1
    assert sources[0]["citation_id"] == 1
    assert "[1]" in block


def test_validate_citations_valid():
    result = validate_citations("Alice founded Acme [1].", 3)
    assert result["valid"] is True
    assert result["cited_ids"] == [1]


def test_validate_citations_invalid_id():
    result = validate_citations("Claim [9] here.", 3)
    assert result["valid"] is False
    assert 9 in result["invalid_ids"]


def test_enforce_missing_citations():
    answer, check = enforce_citations("Alice founded Acme without cite.", 2)
    assert check["missing_citations"] is True
    assert "citation" in answer.lower()
