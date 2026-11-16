from __future__ import annotations

import json
import math

import httpx
import numpy as np
import pytest

from vidlens.cloud_embedding import CloudEmbeddingBackend
from vidlens.bm25_retrieval import BM25Retriever, tokenize
from vidlens.embedding_retrieval import HybridRrfRetriever
from vidlens import runtime_retrieval
from scripts.compare_retrieval_methods import method_profiles, FrozenVectors


def segment(cid, content, source="ASR"):
    return {"segmentId": cid, "content": content, "source": source, "startMs": 0, "endMs": 1000}


def test_bm25_formula_frequency_and_length():
    retriever = BM25Retriever([segment("a", "alpha alpha beta"), segment("b", "alpha beta gamma delta")])
    result = retriever.search("alpha")
    score = math.log(1 + .5 / 2.5) * 2 * 2.5 / (2 + 1.5 * (.25 + .75 * 3 / 3.5))
    assert result["matches"][0]["segmentId"] == "a"
    assert result["matches"][0]["score"] == pytest.approx(score)
    assert retriever.search("absent")["matches"] == []
    assert tokenize("订单缓存 ASR 12") == ["asr", "12", "订单", "单缓", "缓存"]


def test_bm25_source_filter_and_hybrid_uses_requested_component():
    segs = [segment("a", "alpha"), segment("b", "beta", "OCR")]
    backend = FrozenVectors(["alpha", "beta"], np.eye(2, dtype=np.float32))
    sparse = BM25Retriever(segs)
    assert sparse.search("beta", sources=["ASR"])["matches"] == []
    hybrid = HybridRrfRetriever(segs, backend, lexical_retriever=sparse)
    assert hybrid.lexical is sparse
    assert hybrid.search("alpha")["matches"][0]["segmentId"] == "a"
    profiles = method_profiles(backend, {"embeddingModel": "test"})
    assert len(profiles) == 5
    assert profiles["bm25_dense_rrf"].settings["fusion"] == profiles["idf_dense_rrf"].settings["fusion"]


def test_cloud_batch_order_normalization_cache_and_usage():
    calls = []
    def handle(request):
        body = json.loads(request.content)
        calls.append(body)
        assert request.headers["authorization"] == "Bearer test-key"
        rows = [{"index": n, "embedding": [3., 4.]} for n, _ in enumerate(body["input"])]
        return httpx.Response(200, json={"data": rows[::-1], "usage": {"total_tokens": 7}})
    backend = CloudEmbeddingBackend("https://example.invalid/v1", "test-key", dimensions=2,
                                    batch_size=2, transport=httpx.MockTransport(handle))
    result = backend.encode(["a", "b", "c", "a"])
    assert len(calls) == 2
    assert result.shape == (4, 2)
    assert np.allclose(result, [.6, .8])
    backend.encode(["b"], query=True)
    assert len(calls) == 2
    assert backend.usage()["totalTokens"] == 14


@pytest.mark.parametrize("rows", [
    [{"index": 0, "embedding": [0, 0]}],
    [{"index": 0, "embedding": [1, 2, 3]}],
    [{"index": 1, "embedding": [1, 2]}],
    [{"index": 0, "embedding": [float('nan'), 2]}],
])
def test_cloud_rejects_invalid_vectors(rows):
    # JSON encoding of NaN is not accepted by httpx; supply raw response body.
    transport = httpx.MockTransport(lambda r: httpx.Response(200, content=json.dumps({"data": rows})))
    backend = CloudEmbeddingBackend("https://example.invalid/v1", "key", dimensions=2, transport=transport)
    with pytest.raises(RuntimeError, match="invalid vectors"):
        backend.encode(["a"])


def test_cloud_error_sanitization():
    backend = CloudEmbeddingBackend("https://example.invalid/v1", "secret", transport=httpx.MockTransport(
        lambda r: httpx.Response(401, text="secret private account")))
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        backend.encode(["test"])
    assert "secret" not in str(error.value)


def test_runtime_cloud_selection_reuses_dashscope_key(monkeypatch):
    monkeypatch.setenv("EVIDENCE_EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "test-key")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_API_KEY", "")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setattr(runtime_retrieval, "_embedding_backend", None)
    monkeypatch.setattr(runtime_retrieval, "_embedding_key", None)
    backend = runtime_retrieval._embedding()
    assert isinstance(backend, CloudEmbeddingBackend)
    assert backend.api_key == "test-key"
    assert runtime_retrieval._embedding() is backend
    monkeypatch.setenv("EVIDENCE_EMBEDDING_MODEL", "another-model")
    assert runtime_retrieval._embedding().cache_identity != backend.cache_identity


def test_cloud_collection_isolates_model_and_dimensions(monkeypatch):
    import qdrant_client
    from vidlens import qdrant_retrieval
    monkeypatch.setenv("EVIDENCE_EMBEDDING_PROVIDER", "cloud")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_MODEL", "text-embedding-v4")
    monkeypatch.setenv("EVIDENCE_EMBEDDING_DIMENSIONS", "1024")
    monkeypatch.setattr(runtime_retrieval, "_qdrant_store", None)
    monkeypatch.setattr(runtime_retrieval, "_qdrant_key", None)
    monkeypatch.setattr(qdrant_client, "QdrantClient", lambda **kw: object())
    monkeypatch.setattr(qdrant_retrieval, "QdrantEvidenceStore", lambda client, collection_name: collection_name)
    first = runtime_retrieval._store()
    monkeypatch.setenv("EVIDENCE_EMBEDDING_DIMENSIONS", "512")
    second = runtime_retrieval._store()
    assert first != second
    assert "_cloud_" in first
