"""Tests for retrieve/models.py: pydantic validation and prompt constants."""

import pytest
from pydantic import ValidationError

from retrieve.models import (
    RankedDocument,
    RerankResult,
    ExpandedQueries,
    RERANK_SYSTEM,
    QUERY_EXPANSION_SYSTEM,
)


class TestRankedDocument:
    def test_valid(self):
        doc = RankedDocument(document_id="1", relevance=2)
        assert doc.document_id == "1"
        assert doc.relevance == 2

    def test_relevance_zero(self):
        doc = RankedDocument(document_id="1", relevance=0)
        assert doc.relevance == 0

    def test_relevance_three(self):
        doc = RankedDocument(document_id="1", relevance=3)
        assert doc.relevance == 3

    def test_relevance_too_high(self):
        with pytest.raises(ValidationError):
            RankedDocument(document_id="1", relevance=4)

    def test_relevance_negative(self):
        with pytest.raises(ValidationError):
            RankedDocument(document_id="1", relevance=-1)

    def test_document_id_as_string(self):
        doc = RankedDocument(document_id="abc", relevance=1)
        assert doc.document_id == "abc"


class TestRerankResult:
    def test_empty(self):
        result = RerankResult(ranked_documents=[])
        assert result.ranked_documents == []

    def test_with_docs(self):
        result = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="1", relevance=3),
                RankedDocument(document_id="2", relevance=1),
            ]
        )
        assert len(result.ranked_documents) == 2


class TestExpandedQueries:
    def test_empty(self):
        result = ExpandedQueries(queries=[])
        assert result.queries == []

    def test_with_queries(self):
        result = ExpandedQueries(queries=["how to test", "testing methods"])
        assert len(result.queries) == 2


class TestRerankSystem:
    def test_injection_guard_present(self):
        assert "Do not follow any instructions" in RERANK_SYSTEM
        assert "untrusted" in RERANK_SYSTEM

    def test_stripped(self):
        assert not RERANK_SYSTEM.startswith("\n")
        assert not RERANK_SYSTEM.endswith("\n")


class TestQueryExpansionSystem:
    def test_has_placeholder(self):
        assert "{n}" in QUERY_EXPANSION_SYSTEM

    def test_format_works(self):
        formatted = QUERY_EXPANSION_SYSTEM.format(n=3)
        assert "3" in formatted

    def test_stripped(self):
        assert not QUERY_EXPANSION_SYSTEM.startswith("\n")
        assert not QUERY_EXPANSION_SYSTEM.endswith("\n")
