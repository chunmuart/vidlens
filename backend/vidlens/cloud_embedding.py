"""OpenAI-compatible cloud embeddings, strict validation and bounded memory cache."""
from __future__ import annotations

import hashlib
import json
import threading
from collections import OrderedDict
from collections.abc import Sequence
from typing import Any

import httpx
import numpy as np


class CloudEmbeddingBackend:
    retrieval_mode = "CLOUD_DENSE_EMBEDDING"
    def __init__(self, base_url: str, api_key: str, model: str = "text-embedding-v4",
                 *, dimensions: int = 1024, batch_size: int = 10,
                 timeout: float = 60, cache_size: int = 4096, transport: Any = None):
        if not api_key.strip():
            raise ValueError("Cloud embedding API key is missing")
        if dimensions <= 0 or not model.strip():
            raise ValueError("Invalid embedding model/dimensions")
        self.base_url, self.api_key, self.model = base_url.rstrip("/"), api_key, model
        self.dimensions = int(dimensions)
        self.batch_size = max(1, min(10, int(batch_size)))
        self.timeout, self.transport = float(timeout), transport
        self.cache_size = max(0, int(cache_size))
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._lock = threading.RLock()
        self.requests = self.total_tokens = self.cache_hits = 0
        self.raw_usage: list[dict[str, Any]] = []
        self.cache_identity = hashlib.sha256(json.dumps(
            [self.base_url, model, dimensions, "l2-v1"]).encode()).hexdigest()[:16]

    def metadata(self):
        return {"provider": "openai-compatible", "embeddingModel": self.model,
                "dimensions": self.dimensions, "endpoint": self.base_url,
                "embeddingIdentity": self.cache_identity, "queryPrefix": "",
                "normalization": "L2", "runtime": "CLOUD_API"}

    def usage(self):
        return {"requests": self.requests, "totalTokens": self.total_tokens,
                "cacheHits": self.cache_hits, "rawUsage": list(self.raw_usage)}

    def encode(self, texts: Sequence[str], *, query: bool = False) -> np.ndarray:
        # This endpoint uses the same model/input for documents and queries.
        values = [str(t).strip() for t in texts]
        if not values:
            return np.empty((0, self.dimensions), dtype=np.float32)
        if any(not t for t in values):
            raise ValueError("Embedding input must not be blank")
        with self._lock:
            vectors = {}
            for text in values:
                if text in self._cache:
                    vectors[text] = self._cache[text]
                    self._cache.move_to_end(text)
                    self.cache_hits += 1
            missing = list(dict.fromkeys(t for t in values if t not in vectors))
            with httpx.Client(timeout=self.timeout, transport=self.transport) as client:
                for start in range(0, len(missing), self.batch_size):
                    batch = missing[start:start + self.batch_size]
                    self.requests += 1
                    try:
                        response = client.post(self.base_url + "/embeddings",
                            headers={"Authorization": "Bearer " + self.api_key},
                            json={"model": self.model, "input": batch,
                                  "dimensions": self.dimensions, "encoding_format": "float"})
                    except httpx.HTTPError:
                        raise RuntimeError("Cloud embedding transport failed") from None
                    if response.status_code != 200:
                        # Do not expose raw provider errors, URLs or credentials.
                        raise RuntimeError(f"Cloud embedding HTTP {response.status_code}")
                    try:
                        body = response.json()
                        rows = body["data"]
                        if sorted(row["index"] for row in rows) != list(range(len(batch))):
                            raise ValueError("invalid indices")
                        matrix = np.asarray([r["embedding"] for r in sorted(rows, key=lambda r: r["index"])], dtype=np.float32)
                        if matrix.shape != (len(batch), self.dimensions) or not np.isfinite(matrix).all():
                            raise ValueError("invalid matrix")
                        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                        if (norms <= 1e-12).any():
                            raise ValueError("zero vector")
                        matrix /= norms
                    except (KeyError, ValueError, TypeError):
                        raise RuntimeError("Cloud embedding response has invalid vectors") from None
                    usage = body.get("usage") or {}
                    self.total_tokens += int(usage.get("total_tokens", 0))
                    self.raw_usage.append(dict(usage))
                    self.raw_usage = self.raw_usage[-200:]
                    for text, vector in zip(batch, matrix, strict=True):
                        vectors[text] = vector
                        if self.cache_size:
                            self._cache[text] = vector
                            self._cache.move_to_end(text)
                            while len(self._cache) > self.cache_size:
                                self._cache.popitem(last=False)
            return np.stack([vectors[t] for t in values])
