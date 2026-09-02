"""Tests for retrieve/search.py: vector_search, rrf_fuse, range_bounds,
allowed_in_range, rerank (defensive validation), expand_query (fallback),
embed_queries, search (orchestration with mocked clients)."""

import time
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

import retrieve.search as search_mod
from retrieve.search import (
    vector_search,
    rrf_fuse,
    range_bounds,
    allowed_in_range,
    rerank,
    expand_query,
    embed_queries,
    search,
)
from retrieve.models import RankedDocument, RerankResult, ExpandedQueries
from retrieve.state import Session


class TestVectorSearch:
    def test_basic(self):
        ids = np.array([1, 2, 3])
        mat = np.array([[0.1, 0.0], [0.0, 0.9], [0.5, 0.5]], dtype=np.float32)
        query = np.array([0.0, 1.0], dtype=np.float32)
        result = vector_search(ids, mat, query, 3)
        assert 2 in result  # id 2 has highest sim to [0,1]
        assert result[0] == 2  # top result

    def test_n_larger_than_ids(self):
        ids = np.array([1, 2])
        mat = np.array([[0.1, 0.0], [0.0, 0.9]], dtype=np.float32)
        query = np.array([1.0, 0.0], dtype=np.float32)
        result = vector_search(ids, mat, query, 10)
        assert len(result) <= 2

    def test_n_one(self):
        ids = np.array([1, 2, 3])
        mat = np.eye(3, dtype=np.float32)
        query = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        result = vector_search(ids, mat, query, 1)
        assert result == [2]

    def test_returns_ints(self):
        ids = np.array([10, 20, 30])
        mat = np.eye(3, dtype=np.float32)
        query = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        result = vector_search(ids, mat, query, 3)
        for r in result:
            assert isinstance(r, int) or isinstance(r, np.integer)


class TestRRFFuse:
    def test_empty(self):
        assert rrf_fuse([]) == []

    def test_single_ranking(self):
        result = rrf_fuse([[1, 2, 3]])
        assert result == [1, 2, 3]

    def test_multiple_rankings(self):
        # doc appearing in rank 0 of both rankings should score highest
        result = rrf_fuse([[1, 2, 3], [1, 3, 2]])
        assert result[0] == 1

    def test_same_doc_in_multiple_rankings(self):
        result = rrf_fuse([[1, 2], [2, 1]])
        # doc 1: 1/(k+0+1) + 1/(k+1+1); doc 2: 1/(k+1+1) + 1/(k+0+1) => tie
        # both have the same score, so order is by first-seen
        assert set(result[:2]) == {1, 2}

    def test_doc_not_in_all_rankings(self):
        result = rrf_fuse([[1, 2, 3], [3, 4]])
        assert 1 in result
        assert 4 in result
        # doc 3 appears in both, should rank high
        assert result.index(3) <= 1

    def test_empty_ranking_in_list(self):
        result = rrf_fuse([[], [1, 2]])
        assert set(result) == {1, 2}


class TestRangeBounds:
    def test_tuple_passthrough(self):
        bounds = range_bounds((100, 200))
        assert bounds == (100, 200)

    def test_str_rolling_window(self):
        with patch.object(time, "time", return_value=1000000):
            lo, hi = range_bounds("1d")
            assert lo == 1000000 - 86400
            assert hi is None

    def test_str_1w(self):
        with patch.object(time, "time", return_value=1000000):
            lo, hi = range_bounds("1w")
            assert lo == 1000000 - 7 * 86400


class TestAllowedInRange:
    def test_no_filter_show_archived_fast_path(self, sample_ids, sample_meta):
        f_ids, mask, allowed = allowed_in_range(
            sample_ids, sample_meta, None, show_archived=True
        )
        assert mask is None
        assert allowed is None
        assert np.array_equal(f_ids, sample_ids)

    def test_archived_hidden(self, sample_ids, sample_meta):
        f_ids, mask, allowed = allowed_in_range(
            sample_ids, sample_meta, None, show_archived=False
        )
        assert 3 not in f_ids  # archived chat excluded
        assert 1 in f_ids
        assert 2 in f_ids

    def test_archived_shown(self, sample_ids, sample_meta):
        f_ids, mask, allowed = allowed_in_range(
            sample_ids, sample_meta, None, show_archived=True
        )
        assert 3 in f_ids

    def test_time_filter_excludes_none_epoch(self):
        ids = np.array([1, 2])
        meta = {
            1: {
                "file_path": "ch_session_1700000000.json",
                "last_message_epoch": 1700000000,
                "archived": False,
            },
            2: {
                "file_path": "no_epoch.json",
                "last_message_epoch": None,
                "archived": False,
            },
        }
        with patch.object(time, "time", return_value=1700000100):
            f_ids, mask, allowed = allowed_in_range(ids, meta, "1d", show_archived=True)
            assert 1 in f_ids
            assert 2 not in f_ids  # no epoch -> excluded by time filter

    def test_time_filter_lo_only(self):
        ids = np.array([1, 2])
        meta = {
            1: {"file_path": "f.json", "last_message_epoch": 100, "archived": False},
            2: {"file_path": "f.json", "last_message_epoch": 50, "archived": False},
        }
        f_ids, mask, allowed = allowed_in_range(
            ids, meta, (80, None), show_archived=True
        )
        assert 1 in f_ids
        assert 2 not in f_ids

    def test_time_filter_hi_only(self):
        # Note: when lo is None, the OR short-circuits and hi is not checked.
        # This is because rolling windows have lo set and hi None (no upper bound).
        # The custom picker always sets both bounds, so (None, hi) doesn't occur.
        ids = np.array([1, 2])
        meta = {
            1: {"file_path": "f.json", "last_message_epoch": 100, "archived": False},
            2: {"file_path": "f.json", "last_message_epoch": 200, "archived": False},
        }
        f_ids, mask, allowed = allowed_in_range(
            ids, meta, (None, 150), show_archived=True
        )
        # lo is None -> time check short-circuits, both pass
        assert 1 in f_ids
        assert 2 in f_ids  # hi not checked due to lo-is-None short-circuit

    def test_time_filter_both_bounds(self):
        ids = np.array([1, 2, 3])
        meta = {
            1: {"file_path": "f.json", "last_message_epoch": 100, "archived": False},
            2: {"file_path": "f.json", "last_message_epoch": 150, "archived": False},
            3: {"file_path": "f.json", "last_message_epoch": 200, "archived": False},
        }
        f_ids, mask, allowed = allowed_in_range(
            ids, meta, (100, 150), show_archived=True
        )
        assert 1 in f_ids
        assert 2 in f_ids
        assert 3 not in f_ids

    def test_combined_archived_and_time(self):
        ids = np.array([1, 2, 3])
        meta = {
            1: {"file_path": "f.json", "last_message_epoch": 100, "archived": False},
            2: {"file_path": "f.json", "last_message_epoch": 150, "archived": True},
            3: {"file_path": "f.json", "last_message_epoch": 200, "archived": False},
        }
        f_ids, mask, allowed = allowed_in_range(
            ids, meta, (100, 200), show_archived=False
        )
        assert 1 in f_ids
        assert 2 not in f_ids  # archived hidden
        assert 3 in f_ids


class TestRerank:
    def test_hallucinated_id_dropped(self):
        candidate_ids = [1, 2, 3]
        meta = {1: {"summary": "a"}, 2: {"summary": "b"}, 3: {"summary": "c"}}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="1", relevance=3),
                RankedDocument(document_id="99999", relevance=2),  # hallucinated
                RankedDocument(document_id="2", relevance=1),
            ]
        )
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            ranked, in_tok, out_tok = rerank("query", candidate_ids, meta)
            ids_ranked = [r[0] for r in ranked]
            assert 99999 not in ids_ranked
            assert 1 in ids_ranked
            assert 2 in ids_ranked

    def test_non_int_id_dropped(self):
        candidate_ids = [1, 2]
        meta = {1: {"summary": "a"}, 2: {"summary": "b"}}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="abc", relevance=3),  # non-int
                RankedDocument(document_id="1", relevance=2),
            ]
        )
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            ranked, in_tok, out_tok = rerank("query", candidate_ids, meta)
            ids_ranked = [r[0] for r in ranked]
            assert "abc" not in [str(x) for x in ids_ranked]
            assert 1 in ids_ranked

    def test_duplicates_dropped(self):
        candidate_ids = [1, 2, 3]
        meta = {1: {"summary": "a"}, 2: {"summary": "b"}, 3: {"summary": "c"}}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="1", relevance=3),
                RankedDocument(document_id="1", relevance=2),  # dup
                RankedDocument(document_id="2", relevance=1),
            ]
        )
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            ranked, _, _ = rerank("query", candidate_ids, meta)
            ids_ranked = [r[0] for r in ranked]
            assert ids_ranked.count(1) == 1

    def test_dropped_candidates_appended_in_order(self):
        candidate_ids = [1, 2, 3, 4]
        meta = {i: {"summary": f"s{i}"} for i in candidate_ids}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="3", relevance=3),
                RankedDocument(document_id="1", relevance=2),
                # dropped 2 and 4
            ]
        )
        mock_resp.usage.prompt_tokens = 10
        mock_resp.usage.completion_tokens = 5
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            ranked, _, _ = rerank("query", candidate_ids, meta)
            ids_ranked = [r[0] for r in ranked]
            # 3 and 1 from model, then 2 and 4 in original order, all with None grade
            assert ids_ranked == [3, 1, 2, 4]
            assert ranked[2][1] is None  # appended, ungraded
            assert ranked[3][1] is None

    def test_exception_fallback_preserves_order(self):
        candidate_ids = [5, 3, 1]
        meta = {i: {"summary": f"s{i}"} for i in candidate_ids}
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.side_effect = Exception("API down")
            ranked, in_tok, out_tok = rerank("query", candidate_ids, meta)
            assert [r[0] for r in ranked] == candidate_ids
            assert all(r[1] is None for r in ranked)
            assert in_tok == 0
            assert out_tok == 0

    def test_valid_subset_returns_them_plus_remainder(self):
        candidate_ids = [1, 2, 3]
        meta = {i: {"summary": f"s{i}"} for i in candidate_ids}
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = RerankResult(
            ranked_documents=[
                RankedDocument(document_id="2", relevance=3),
            ]
        )
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            ranked, _, _ = rerank("query", candidate_ids, meta)
            ids_ranked = [r[0] for r in ranked]
            assert ids_ranked[0] == 2
            assert set(ids_ranked[1:]) == {1, 3}


class TestExpandQuery:
    def test_n_le_zero_returns_empty(self):
        result, in_tok, out_tok = expand_query("test", n=0)
        assert result == []
        assert in_tok == 0
        assert out_tok == 0

    def test_exception_returns_empty(self):
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.side_effect = Exception("fail")
            result, in_tok, out_tok = expand_query("test", n=3)
            assert result == []
            assert in_tok == 0
            assert out_tok == 0

    def test_dedup_case_insensitive(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = ExpandedQueries(
            queries=["Test", "TEST", "test", "other query"]
        )
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            result, _, _ = expand_query("test", n=5)
            # "Test", "TEST", "test" all dedup against original "test"
            assert len(result) == 1
            assert result[0] == "other query"

    def test_empty_variants_dropped(self):
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = ExpandedQueries(
            queries=["", "  ", "valid query"]
        )
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            result, _, _ = expand_query("original", n=5)
            assert "valid query" in result
            assert "" not in result

    def test_truncated_to_n(self):
        queries = [f"query_{i}" for i in range(10)]
        mock_resp = MagicMock()
        mock_resp.choices = [MagicMock()]
        mock_resp.choices[0].message.parsed = ExpandedQueries(queries=queries)
        mock_resp.usage.prompt_tokens = 5
        mock_resp.usage.completion_tokens = 3
        with patch.object(search_mod, "groq_client") as mock_groq:
            mock_groq.chat.completions.parse.return_value = mock_resp
            result, _, _ = expand_query("original", n=3)
            assert len(result) == 3


class TestEmbedQueries:
    def test_order_restoration(self):
        mock_resp = MagicMock()
        d1 = MagicMock(index=1, embedding=[0.0, 1.0, 0.0])
        d0 = MagicMock(index=0, embedding=[1.0, 0.0, 0.0])
        mock_resp.data = [d1, d0]  # out of order
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            vecs, in_tok = embed_queries(["q1", "q2"])
            # q1 should be first (index 0)
            assert vecs[0][0] == pytest.approx(1.0)
            assert vecs[1][1] == pytest.approx(1.0)

    def test_normalization(self):
        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[3.0, 4.0])  # norm = 5
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            vecs, _ = embed_queries(["q"])
            norm = np.linalg.norm(vecs[0])
            assert norm == pytest.approx(1.0, abs=1e-6)


class TestSearch:
    def test_no_rerank_returns_ungraded(self, sample_session):
        sample_session.do_rerank = False
        sample_session.do_expand = False
        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[1.0, 0.0, 0.0])
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            with patch.object(search_mod, "fts_search", return_value=[]):
                results, usage = search(sample_session, "test query")
                assert len(results) <= 5
                for cid, grade in results:
                    assert grade is None
                assert "embed_in" in usage
                assert "expand_in" in usage
                assert "rerank_in" in usage

    def test_no_expand_uses_single_query(self, sample_session):
        sample_session.do_expand = False
        sample_session.do_rerank = False
        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[1.0, 0.0, 0.0])
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            with patch.object(search_mod, "fts_search", return_value=[]):
                results, usage = search(sample_session, "test")
                assert usage["expand_in"] == 0
                assert usage["expand_out"] == 0

    def test_empty_results(self, sample_session):
        sample_session.do_rerank = False
        sample_session.do_expand = False
        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[1.0, 0.0, 0.0])
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            with patch.object(search_mod, "fts_search", return_value=[]):
                with patch.object(search_mod, "vector_search", return_value=[]):
                    results, _ = search(sample_session, "test")
                    assert results == []

    def test_large_top_k_caps_rerank_pool(self):
        """When result_len is large (e.g. 9999), rerank candidates are capped at 25
        so hundreds of documents aren't sent to the LLM at once. The rest are returned
        in hybrid order."""
        fused_ids = list(range(1, 51))
        meta = {
            i: {
                "file_path": f"/tmp/ch_session_{i}.json",
                "summary": f"summary {i}",
                "last_message_epoch": i,
                "archived": False,
            }
            for i in fused_ids
        }
        session = Session(
            conn=MagicMock(),
            ids=np.array(fused_ids),
            mat=np.eye(50, dtype=np.float32),
            meta=meta,
            do_rerank=True,
            do_expand=False,
            show_archived=False,
            result_len=9999,
        )

        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[1.0] * 50)
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5

        with patch.object(search_mod, "client") as mock_client, patch.object(
            search_mod, "fts_search", return_value=[]
        ), patch.object(
            search_mod, "vector_search", return_value=fused_ids
        ), patch.object(
            search_mod, "rerank"
        ) as mock_rerank:
            mock_client.embeddings.create.return_value = mock_resp
            mock_rerank.return_value = ([(cid, 3) for cid in fused_ids[:25]], 10, 10)
            results, usage = search(session, "test")

            # Check that rerank was called with at most 25 candidates
            called_candidate_ids = mock_rerank.call_args[0][1]
            assert len(called_candidate_ids) == 25

            # Total results returned should still include all 50
            assert len(results) == 50
            # First 25 are graded, last 25 are ungraded
            for cid, grade in results[:25]:
                assert grade == 3
            for cid, grade in results[25:]:
                assert grade is None

    def test_usage_dict_keys(self, sample_session):
        sample_session.do_rerank = False
        sample_session.do_expand = False
        mock_resp = MagicMock()
        d = MagicMock(index=0, embedding=[1.0, 0.0, 0.0])
        mock_resp.data = [d]
        mock_resp.usage.prompt_tokens = 5
        with patch.object(search_mod, "client") as mock_client:
            mock_client.embeddings.create.return_value = mock_resp
            with patch.object(search_mod, "fts_search", return_value=[]):
                _, usage = search(sample_session, "test")
                assert set(usage.keys()) == {
                    "embed_in",
                    "expand_in",
                    "expand_out",
                    "rerank_in",
                    "rerank_out",
                }
