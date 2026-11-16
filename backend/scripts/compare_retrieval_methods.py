"""Fair BM25/IDF/dense/RRF comparison on one frozen evidence snapshot."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.evaluate_suite import check_dataset, digest
from scripts.prepare_eval_suite import save
from scripts.evaluate_real_evidence import evaluate_profile, load_snapshots
from scripts.evaluate_retrieval_ab import comparison
from vidlens.bm25_retrieval import BM25Retriever
from vidlens.cloud_embedding import CloudEmbeddingBackend
from vidlens.embedding_retrieval import DenseEmbeddingRetriever, HybridRrfRetriever
from vidlens.retrieval import LEXICAL_PROFILE, RetrieverProfile
from vidlens.evidence_pipeline import atomic_json


def resumable_embeddings(texts, backend, directory, workers=4):
    """One saved paid request per ten texts; independent clients allow bounded concurrency."""
    directory.mkdir(parents=True,exist_ok=True)
    meta=backend.metadata()
    def batch_one(start):
        values=texts[start:start+10]
        identity=digest({'metadata':meta,'texts':values})
        path=directory/(identity+'.npz');record=directory/(identity+'.json')
        if path.exists() and record.exists():
            with np.load(path,allow_pickle=False) as saved:matrix=saved['vectors']
            return start,matrix,{'requests':0,'totalTokens':0,'cacheHits':len(values),'rawUsage':[]}
        if os.getenv('VIDLENS_EVAL_OFFLINE')=='1':raise RuntimeError('Cached-only mode: missing embedding batch')
        pending=directory/(identity+'.pending.json')
        if pending.exists():raise RuntimeError('Unresolved embedding request; inspect before retry')
        atomic_json(pending,{'inputFingerprint':identity,'status':'STARTED_USAGE_UNKNOWN'})
        client=CloudEmbeddingBackend(backend.base_url,backend.api_key,backend.model,dimensions=backend.dimensions)
        matrix=client.encode(values)
        temporary=directory/(identity+'.tmp.npz')
        np.savez_compressed(temporary,vectors=matrix);temporary.replace(path)
        atomic_json(record,{'inputFingerprint':identity,'usage':client.usage()});pending.unlink()
        return start,matrix,client.usage()
    output={};usage={'requests':0,'totalTokens':0,'cacheHits':0,'rawUsage':[]}
    with ThreadPoolExecutor(max_workers=max(1,min(6,workers))) as pool:
        futures=[pool.submit(batch_one,i) for i in range(0,len(texts),10)]
        for f in as_completed(futures):
            start,matrix,event=f.result();output[start]=matrix
            for key in ('requests','totalTokens','cacheHits'):usage[key]+=event[key]
            usage['rawUsage'].extend(event['rawUsage'])
            if len(output)%100==0:print(json.dumps({'embeddingBatchesCompleted':len(output),'totalBatches':len(futures)}),flush=True)
    return np.concatenate([output[i] for i in sorted(output)]),usage


class FrozenVectors:
    retrieval_mode = "CLOUD_DENSE_EMBEDDING"
    def __init__(self, texts, matrix):
        self.vectors = dict(zip(texts, matrix, strict=True))

    def encode(self, texts, *, query=False):
        return np.stack([self.vectors[str(t).strip()] for t in texts])


def method_profiles(backend, settings):
    fusion = {"rrfK": 60, "candidateDepth": 20, "weights": [.5, .5]}
    return {
        "lexical": LEXICAL_PROFILE,
        "bm25": RetrieverProfile("bm25-bigram-v1", "BM25_CHINESE_BIGRAM", BM25Retriever,
                                 "BM25，中文二元组/ASCII分词", {"k1": 1.5, "b": .75, "tokenizer": "chinese-bigram-ascii-v1"}),
        "dense": RetrieverProfile("cloud-dense-v1", "CLOUD_DENSE_COSINE",
                                  lambda s: DenseEmbeddingRetriever(s, backend), "纯向量余弦检索", settings),
        "bm25_dense_rrf": RetrieverProfile("bm25-cloud-rrf-v1", "BM25_DENSE_RRF",
            lambda s: HybridRrfRetriever(s, backend, lexical_retriever=BM25Retriever(s)),
            "BM25+云端向量，等权RRF", {**settings, "fusion": fusion, "bm25": {"k1": 1.5, "b": .75}}),
        "idf_dense_rrf": RetrieverProfile("idf-cloud-rrf-v1", "IDF_DENSE_RRF",
            lambda s: HybridRrfRetriever(s, backend), "现有IDF词法+云端向量，等权RRF",
            {**settings, "fusion": fusion, "lexical": "existing-phrase-bigram-idf"}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT.parent / ".runtime/evals/embedding_cache")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env")
    parser.add_argument("--allow-candidates", action="store_true")
    parser.add_argument("--execute", action="store_true", help="生成云端向量或复用同版本向量缓存后运行")
    parser.add_argument("--embedding-workers",type=int,default=4)
    args = parser.parse_args()
    load_dotenv(args.env_file, override=False)
    data = json.loads(args.dataset.read_text(encoding="utf-8-sig"))
    check_dataset(data, args.allow_candidates)
    snapshots = load_snapshots(args.snapshot)
    ids = {c["videoId"] for c in data["cases"]}
    if ids - snapshots.keys():
        parser.error("Missing video snapshots")
    snapshots = {k: snapshots[k] for k in sorted(ids)}
    exact = data.get("matchingPolicy") == "segment-id-v1"
    if exact:
        for case in data["cases"]:
            known = {str(s["segmentId"]) for s in snapshots[case["videoId"]]}
            if any(not e.get("segmentIds") or set(e["segmentIds"]) - known for e in case["goldEvidence"]):
                raise ValueError("Exact-ID gold refers to missing evidence")
    texts = sorted({str(s.get("content", "")).strip() for segs in snapshots.values() for s in segs}
                   | {c["question"].strip() for c in data["cases"]})
    backend = CloudEmbeddingBackend(
        os.getenv("EVIDENCE_EMBEDDING_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        os.getenv("EVIDENCE_EMBEDDING_API_KEY") or os.getenv("DASHSCOPE_API_KEY", ""),
        os.getenv("EVIDENCE_EMBEDDING_MODEL") or "text-embedding-v4",
        dimensions=int(os.getenv("EVIDENCE_EMBEDDING_DIMENSIONS") or "1024"))
    meta = backend.metadata()
    output = {"datasetId": data["datasetId"], "datasetFingerprint": digest(data),
              "snapshotFingerprint": digest(snapshots), "kind": data.get("kind"),
              "caseCount": len(data["cases"]), "sourceCount": len(ids), "uniqueTexts": len(texts),
              "candidateDebug": args.allow_candidates, "embedding": meta,
              "scorerVersion": "retrieval-method-matrix-exact-id-v2" if exact else "retrieval-method-matrix-v1",
              "matchingPolicy": data.get("matchingPolicy", "source-time-overlap-v1"), "executed": args.execute,
              "note": "五组固定同题/证据/云端向量；不加入邻接扩展/覆盖门禁。追问只输入本轮问题，不输入历史。受控样例不能代表真实视频准确率。耗时为预生成向量后的检索耗时。"}
    if not args.execute:
        save(args.output, output)
        print(json.dumps(output, ensure_ascii=False, indent=2))
        return
    cache_id = digest({"embedding": meta, "texts": texts})
    cache_path = args.cache_dir / (cache_id + ".npz")
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as cache:
            matrix = cache["vectors"]
        output["embeddingCacheReused"] = True
        usage=backend.usage()
    else:
        matrix,usage = resumable_embeddings(texts,backend,args.cache_dir/'batches',args.embedding_workers)
        args.cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, vectors=matrix)
        output["embeddingCacheReused"] = False
    if (matrix.shape != (len(texts), backend.dimensions) or not np.isfinite(matrix).all()
            or not np.allclose(np.linalg.norm(matrix, axis=1), 1.0, atol=1e-4)):
        raise ValueError("Invalid cached vectors")
    frozen = FrozenVectors(texts, matrix)
    results = {name: evaluate_profile(data, snapshots, profile)
               for name, profile in method_profiles(frozen, meta).items()}
    output.update(profiles=results, comparison=comparison(results), providerUsage=usage,
                  embeddingCacheId=cache_id)
    generation_usage=usage
    if args.output.exists():
        previous=json.loads(args.output.read_text(encoding='utf-8-sig'))
        if previous.get('embeddingCacheId')==cache_id:
            generation_usage=previous.get('embeddingGenerationUsage',previous.get('providerUsage',usage))
    output['embeddingGenerationUsage']=generation_usage
    dense = results["dense"]["aggregate"]
    output["deltaVsDense"] = {name: {k: round(result["aggregate"][k]-dense[k], 4)
        for k in ("recallAt1", "recallAt3", "recallAt8", "mrr", "completeHitRateAt8")}
        for name, result in results.items()}
    save(args.output, output)
    print(json.dumps({"datasetId": output["datasetId"], "providerUsage": {k:v for k,v in output["providerUsage"].items() if k!='rawUsage'},
                      "metrics": {name: {k: r["aggregate"][k] for k in (
                          "recallAt1", "recallAt3", "recallAt8", "mrr", "completeHitRateAt8")}
                          for name, r in results.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
