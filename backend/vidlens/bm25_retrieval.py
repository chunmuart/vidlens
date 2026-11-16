"""BM25 with documented deterministic Chinese bigram/ASCII tokenization."""
from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text):
    normalized = str(text).lower()
    result = re.findall(r"[a-z0-9_]+", normalized)
    for run in re.findall(r"[\u4e00-\u9fff]+", normalized):
        result.extend([run] if len(run) == 1 else [run[i:i+2] for i in range(len(run)-1)])
    return result


class BM25Retriever:
    def __init__(self, segments, *, k1=1.5, b=.75):
        if k1 <= 0 or not 0 <= b <= 1:
            raise ValueError("Invalid BM25 parameters")
        self.segments = [dict(s) for s in segments]
        self.k1, self.b = k1, b
        self.tf = [Counter(tokenize(s.get("content", ""))) for s in segments]
        self.lengths = [sum(t.values()) for t in self.tf]
        self.avgdl = sum(self.lengths) / max(1, len(segments))
        self.df = Counter(t for document in self.tf for t in document)

    def search(self, query, *, top_k=8, sources=None):
        if not str(query).strip():
            raise ValueError("Empty query")
        terms = set(tokenize(query))
        allowed = {s.upper() for s in sources or []}
        matches = []
        for segment, tf, length in zip(self.segments, self.tf, self.lengths, strict=True):
            if allowed and segment.get("source", "").upper() not in allowed:
                continue
            score = 0.0
            for term in terms:
                frequency = tf.get(term, 0)
                if not frequency:
                    continue
                idf = math.log(1 + (len(self.segments) - self.df[term] + .5) / (self.df[term] + .5))
                score += idf * frequency * (self.k1+1) / (frequency + self.k1 * (1-self.b + self.b*length/max(self.avgdl, 1e-12)))
            if score > 0:
                matches.append({**segment, "score": score, "scoreDetails": {"bm25": score}})
        matches.sort(key=lambda s: (-s["score"], s.get("startMs", 0), str(s.get("segmentId", ""))))
        return {"ok": True, "query": query, "retrievalMode": "BM25_CHINESE_BIGRAM",
                "matches": matches[:max(1, min(int(top_k), 100))], "matchedCount": len(matches),
                "fallbackToTimelineStart": False}
