"""Build reproducible controlled fixtures and deduplicate existing real candidates."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ["real_video_eval.json", "heldout_retrieval_eval.json",
           "final_heldout_retrieval_eval.json", "coverage_aware_v2_final_heldout_retrieval_eval.json"]


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def controlled(seeds):
    videos, cases, snapshots = [], [], {}
    for index, (topic, purpose, parameter, value, limit, contrast) in enumerate(seeds["rows"], 1):
        vid = f"fixture-{index:03}"
        split = "development" if index <= 33 else "test"
        videos.append({"videoId": vid, "sourceGroupId": vid, "title": f"虚构演示：{topic}", "split": split})
        texts = [f"{topic}的目的是{purpose}。", f"{topic}配置面板：{parameter}为{value}。",
                 f"{topic}的限制：{limit}。", f"对照模块的配置：{contrast}。",
                 "课程安排：下周讨论性能调优，本段没有介绍负责人或上线日期。"]
        snapshots[vid] = [
            {"segmentId": f"{vid}-s{n}", "source": "OCR" if n in (2, 4) else "ASR",
             "startMs": n * 40000, "endMs": n * 40000 + 4000, "content": text}
            for n, text in enumerate(texts, 1)
        ]
        # Add related distractors; top-8 must not trivially include the whole corpus.
        pool = seeds["rows"][:33] if split == "development" else seeds["rows"][33:]
        local_index = index - 1 if split == "development" else index - 34
        for offset in range(1, 5):
            other, purpose2, param2, value2, limit2, _ = pool[(local_index + offset) % len(pool)]
            distractors = [f"{other}的目的是{purpose2}。", f"{other}配置面板：{param2}为{value2}。",
                           f"{other}的限制：{limit2}。", f"{other}的负责人是演示员{offset}，不负责{topic}。"]
            for n, content in enumerate(distractors):
                start = 300000 + offset * 50000 + n * 10000
                snapshots[vid].append({"segmentId": f"{vid}-noise-{offset}-{n}",
                                       "source": "OCR" if n == 1 else "ASR", "startMs": start,
                                       "endMs": start + 4000, "content": content})
        # Deterministically mix timeline positions so gold is not always first.
        snapshots[vid].sort(key=lambda item: hashlib.sha256(item["segmentId"].encode()).hexdigest())
        for position, segment in enumerate(snapshots[vid], 1):
            segment.update(startMs=position * 40000, endMs=position * 40000 + 4000)
        target = {n: next(s for s in snapshots[vid] if s["segmentId"] == f"{vid}-s{n}") for n in (1, 2, 3)}
        templates = [
            ("asr", f"{topic}的目的是什么？", purpose, [[purpose]], [1]),
            ("ocr", f"{topic}配置面板上的{parameter}是多少？", value, [[value]], [2]),
            ("multi_evidence", f"{topic}的目的和{parameter}分别是什么？", f"{purpose}；{value}", [[purpose], [value]], [1, 2]),
            ("constraint", f"{topic}有什么限制？", limit, [[limit]], [3]),
            ("followup", "它有什么限制？", limit, [[limit]], [3]),
            ("unanswerable", f"{topic}的负责人是谁？", "", [], []),
        ]
        for n, (kind, question, answer, groups, evidence) in enumerate(templates, 1):
            case = {"caseId": f"{vid}-q{n}", "videoId": vid, "sourceGroupId": vid,
                    "split": split, "kind": "SYNTHETIC_CONTROLLED", "reviewStatus": "FIXTURE_DEFINED",
                    "type": kind, "question": question, "answerable": bool(evidence),
                    "goldAnswer": answer, "requiredAnswerGroups": groups, "forbiddenAnswerPatterns": [],
                    "evidenceRequirement": "all", "goldEvidence": [
                        {"evidenceId": f"{vid}-e{e}", "startMs": target[e]["startMs"],
                         "endMs": target[e]["endMs"], "sources": [target[e]["source"]]}
                        for e in evidence],
                    "goldPlan": ([f"{topic}的目的是什么？", f"{topic}的{parameter}是多少？"]
                                 if kind == "multi_evidence" else [question])}
            if kind == "followup":
                case["history"] = [{"role": "user", "content": f"请介绍{topic}。"},
                                   {"role": "assistant", "content": f"{topic}的目的是{purpose}。"}]
                case["standaloneQuestion"] = f"{topic}有什么限制？"
                case["goldPlan"] = [case["standaloneQuestion"]]
            if not evidence:
                case["goldAnswer"] = "材料未提供负责人，无法确定。"
                case["requiredAnswerGroups"] = [["未提供", "没有提供", "无法确定", "证据不足"]]
            cases.append(case)
    return {"schemaVersion": "2.0", "datasetId": "vidlens-controlled-300-v1",
            "kind": "SYNTHETIC_CONTROLLED", "reviewStatus": "FIXTURE_DEFINED", "topK": [1, 3, 8],
            "description": "50个虚构证据场景×6种问题；模板相关，不能解释为300个独立真实视频样本。",
            "videos": videos, "cases": cases}, snapshots


def real_candidates(root):
    videos, seen, cases, fingerprints = {}, {}, [], {}
    for filename in SOURCES:
        path = root / filename
        fingerprints[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
        data = json.loads(path.read_text(encoding="utf-8"))
        ids = {}
        for video in data["videos"]:
            canonical = video.get("bvid") or video["videoId"]
            ids[video["videoId"]] = canonical
            videos[canonical] = {**video, "videoId": canonical, "sourceGroupId": canonical}
        for original in data["cases"]:
            vid = ids[original["videoId"]]
            key = (vid, "".join(original["question"].split()))
            if key in seen:
                seen[key]["sourceAnnotations"].append({"file": filename, "caseId": original["caseId"]})
                continue
            case = {**original, "caseId": f"real-candidate-{len(cases)+1:03}", "videoId": vid,
                    "sourceGroupId": vid, "kind": "REAL_VIDEO_CANDIDATE", "split": "unassigned",
                    "reviewStatus": "PENDING_VIDEO_REVIEW",
                    "sourceAnnotations": [{"file": filename, "caseId": original["caseId"]}]}
            seen[key] = case
            cases.append(case)
    return {"schemaVersion": "2.0", "datasetId": "vidlens-real-candidates-v2",
            "kind": "REAL_VIDEO_CANDIDATE", "reviewStatus": "PENDING_VIDEO_REVIEW", "topK": [1, 3, 8],
            "description": "按BV号+去空白问题去重；不是语义去重，重叠题仍需复核；重新核对原视频后才能发布正式集。",
            "sourceFingerprints": fingerprints, "videos": list(videos.values()), "cases": cases}


def validate(dataset):
    cases = dataset["cases"]
    ids = [c["caseId"] for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("duplicate caseId")
    groups = {}
    video_ids = {v["videoId"] for v in dataset["videos"]}
    for case in cases:
        if case["videoId"] not in video_ids or not isinstance(case["answerable"], bool):
            raise ValueError("unknown video or non-boolean answerable")
        group = case["sourceGroupId"]
        if group in groups and groups[group] != case["split"]:
            raise ValueError("source group leaks across splits")
        groups[group] = case["split"]
        if case["answerable"] and not case["goldEvidence"]:
            raise ValueError("answerable case lacks evidence")
        if not case["answerable"] and case["goldEvidence"]:
            raise ValueError("unanswerable case must not carry supporting gold evidence")
    return {"cases": len(cases), "sourceGroups": len(groups),
            "bySplit": dict(Counter(c["split"] for c in cases)),
            "byType": dict(Counter(c["type"] for c in cases)),
            "reviewedRealCases": sum(c["reviewStatus"] == "VIDEO_VERIFIED" and c["kind"] != "SYNTHETIC_CONTROLLED" for c in cases)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evals" / "suite_v2")
    args = parser.parse_args()
    seeds = json.loads((ROOT / "evals/scenario_seeds_v1.json").read_text(encoding="utf-8"))
    data, snapshots = controlled(seeds)
    out = args.output_dir
    save(out / "controlled_300.json", data)
    save(out / "controlled_snapshots.json", snapshots)
    for split in ("development", "test"):
        save(out / f"controlled_{split}.json", {**data, "datasetId": data["datasetId"] + "-" + split,
             "videos": [v for v in data["videos"] if v["split"] == split],
             "cases": [c for c in data["cases"] if c["split"] == split]})
    real = real_candidates(ROOT / "evals")
    save(out / "real_candidates.json", real)
    with (out / "real_review.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["caseId", "videoId", "question", "goldAnswer", "goldEvidence", "reviewStatus", "reviewer", "notes"])
        writer.writeheader()
        for case in real["cases"]:
            writer.writerow({**{k: case.get(k, "") for k in writer.fieldnames},
                             "goldEvidence": json.dumps(case["goldEvidence"], ensure_ascii=False)})
    manifest = {"controlled": validate(data), "realCandidates": validate(real),
                "note": "test是受控回归保留场景，不是真实视频未见集；CSV仅供审核，修改CSV不会自动改变JSON金标。"}
    save(out / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
