"""Publish provenance, stratified results, video-cluster intervals and a manual review CSV."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_real_video_benchmark import read
from scripts.evaluate_suite import digest
from scripts.evaluate_real_evidence import summarize_metrics
from vidlens.evidence_pipeline import atomic_json

LABELS={'lexical':'现有IDF','bm25':'BM25','dense':'纯向量','bm25_dense_rrf':'BM25＋向量','idf_dense_rrf':'IDF＋向量'}


def cluster_delta(rows,baseline,*,metric='recallAt3',seed=20261116,repeats=3000):
    groups=sorted({r['videoId'] for r in rows})
    left={r['caseId']:r for r in rows};right={r['caseId']:r for r in baseline}
    if left.keys()!=right.keys():raise ValueError('Unpaired case predictions')
    values=[np.array([left[cid][metric]-right[cid][metric] for cid in left if left[cid]['videoId']==v]) for v in groups]
    rng=np.random.default_rng(seed)
    draws=[]
    for sample in rng.integers(0,len(groups),size=(repeats,len(groups))):
        draws.append(float(np.concatenate([values[i] for i in sample]).mean()))
    return {'difference':float(np.concatenate(values).mean()),'lower':float(np.quantile(draws,.025)),
            'upper':float(np.quantile(draws,.975)),'resamplingUnit':'video','repeats':repeats,'seed':seed}


def table(stats):
    lines=['| 方案 | Recall@1 | Recall@3 | Recall@8 | MRR |','| --- | ---: | ---: | ---: | ---: |']
    for name,label in LABELS.items():
        a=stats[name]
        lines.append(f'| {label} | {a["recallAt1"]:.2%} | {a["recallAt3"]:.2%} | {a["recallAt8"]:.2%} | {a["mrr"]:.4f} |')
    return '\n'.join(lines)


def write_review_if_new(path, fields, rows):
    if path.exists():return
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT.parent/'.runtime/evals/real_bilibili_v1')
    p.add_argument('--output-dir',type=Path,default=ROOT/'evals/real_bilibili_v1')
    a=p.parse_args();d=read(a.directory/'audited_dataset.json');r=read(a.directory/'method_matrix.json')
    if r['datasetFingerprint']!=digest(d):raise ValueError('Report/dataset fingerprints differ')
    snapshots=read(a.directory/'snapshots.json')
    if r['snapshotFingerprint']!=digest({v:snapshots[v] for v in sorted({c['videoId'] for c in d['cases']})}):
        raise ValueError('Snapshot changed')
    stats={k:v['aggregate'] for k,v in r['profiles'].items()}
    mapping={c['caseId']:c for c in d['cases']}
    strata={}
    for split in ['development','test']:
        strata[split]={k:summarize_metrics([c for c in v['cases'] if mapping[c['caseId']]['split']==split],[1,3,8]) for k,v in r['profiles'].items()}
    per_video={v['videoId']:{k:summarize_metrics([c for c in result['cases'] if c['videoId']==v['videoId']],[1,3,8])
                           for k,result in r['profiles'].items()} for v in d['videos']}
    intervals={k:cluster_delta(v['cases'],r['profiles']['bm25']['cases']) for k,v in r['profiles'].items() if k!='bm25'}
    video_macro={k:round(float(np.mean([values[k]['recallAt3'] for values in per_video.values()])),4) for k in LABELS}
    summary={'date':'2026-11-16','status':'EXPLORATORY_MACHINE_SILVER','datasetFingerprint':r['datasetFingerprint'],
             'snapshotFingerprint':r['snapshotFingerprint'],'embedding':r['embedding'],'scorerVersion':r['scorerVersion'],
             'caseCount':len(d['cases']),'videoCount':len(d['videos']),'minutes':round(sum(v['durationMs'] for v in d['videos'])/60000,2),
             'evidenceSegments':sum(len(s) for s in snapshots.values()),'byType':dict(Counter(c['type'] for c in d['cases'])),
             'bySplit':dict(Counter(c['split'] for c in d['cases'])),'labelAudit':d['labelAudit'],
             'aggregate':stats,'bySplitMetrics':strata,'byVideo':per_video,'videoMacroRecallAt3':video_macro,
             'pairedClusterDeltaRecallAt3VsBm25':intervals,'embeddingUsage':r['providerUsage'],
             'matchingPolicy':r['matchingPolicy'],'candidateDebug':r['candidateDebug']}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    atomic_json(a.output_dir/'summary.json',summary)
    atomic_json(a.output_dir/'questions_silver.json',d)
    fields=['caseId','videoId','videoUrl','split','type','question','goldAnswer','goldEvidence','reviewStatus','humanReviewer','humanVerdict','corrections']
    review_rows=[]
    for c in d['cases']:
        start=min(g['startMs'] for g in c['goldEvidence'])//1000
        review_rows.append({**{k:c[k] for k in ['caseId','videoId','split','type','question','goldAnswer','reviewStatus']},
                            'videoUrl':f'https://www.bilibili.com/video/{c["videoId"]}/?t={start}',
                            'goldEvidence':json.dumps(c['goldEvidence'],ensure_ascii=False)})
    write_review_if_new(a.output_dir/'manual_review.csv',fields,review_rows)
    paired={name:{c['caseId']:c for c in result['cases']} for name,result in r['profiles'].items()}
    fields=['caseId','videoId','question','goldAnswer','bm25RecallAt3','idfDenseRecallAt3','delta',
            'bm25Top3','idfDenseTop3','humanErrorType','reviewer','notes']
    error_rows=[]
    for c in d['cases']:
        b=paired['bm25'][c['caseId']];m=paired['idf_dense_rrf'][c['caseId']]
        error_rows.append({**{k:c[k] for k in ['caseId','videoId','question','goldAnswer']},
                           'bm25RecallAt3':b['recallAt3'],'idfDenseRecallAt3':m['recallAt3'],
                           'delta':m['recallAt3']-b['recallAt3'],
                           'bm25Top3':json.dumps(b['retrieved'][:3],ensure_ascii=False),
                           'idfDenseTop3':json.dumps(m['retrieved'][:3],ensure_ascii=False)})
    write_review_if_new(a.output_dir/'paired_errors.csv',fields,error_rows)
    lines=['# 真实B站视频五组检索对照结果（2026-11-16）','',
           f'成功采集 **{len(d["videos"])}条真实视频、{summary["minutes"]}分钟、{summary["evidenceSegments"]}条证据**。',
           f'初审候选{d["labelAudit"]["inputCases"]}题，经不同模型Qwen-plus再审后保留 **{len(d["cases"])}题**。',
           '全部是机器审查的银标，人工回看视频核验为0题；以下是真实材料上的探索结果，不是正式人工测试集准确率。','',
           '## 全部银标题','',table(stats),'',
           '## 预先按视频划分的开发/测试子集','']
    for split in ['development','test']:
        lines.extend([f'### {split}（{summary["bySplit"].get(split,0)}题）','',table(strata[split]),''])
    lines.extend(['## 主指标Recall@3的成对差异','',
                  '以下相对纯BM25，单位为百分点；95%区间按视频成组重采样3000次。区间跨0时，不能仅凭均值宣称稳定提升。银标误差和便利抽样偏差不包含在这个区间里。','',
                  '| 方案 | 差异 | 95%区间 | 按视频平均Recall@3 |','| --- | ---: | --- | ---: |'])
    for k,x in intervals.items():
        lines.append(f'| {LABELS[k]} | {100*x["difference"]:+.2f} | [{100*x["lower"]:+.2f}, {100*x["upper"]:+.2f}] | {video_macro[k]:.2%} |')
    lines.extend(['','## 来源与每视频表现','','| BV号 | 标题 | 分组 | 题数 | IDF R@3 | BM25 R@3 | 向量 R@3 | BM25混合 R@3 | IDF混合 R@3 |',
                  '| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |'])
    for v in d['videos']:
        z=per_video[v['videoId']]
        lines.append(f'| {v["bvid"]} | {v["title"].replace("|","/")} | {v["split"]} | {z["lexical"]["caseCount"]} | '+' | '.join(f'{z[k]["recallAt3"]:.2%}' for k in LABELS)+' |')
    lines.extend(['','## 实验边界','',
                  '- 选片16条，1条因ASR任务失败排除；成功15条。不是从检索错题中挑视频，也没有按成绩改题。',
                  '- 同问题、同证据、同百炼text-embedding-v4/1024维向量；RRF统一深度20、k=60、等权。BM25使用中文二元组/ASCII分词，k1=1.5、b=0.75；现有IDF含短语匹配，分词不完全相同，不能把全部差异归因于BM25公式。',
                  '- 对每个必需事实，命中所列替代片段之一即算找到；多个必需事实分别计分。具体片段ID计分，非相邻时间重叠。等价支持可能漏标，银标有噪声。',
                  '- 每题只搜自己的视频，不测跨视频搜索；仅测有答案题，不测拒答、答案生成、规划或视觉/语音提取准确率。',
                  '- OCR全程抽最多12帧，实际公开视频画质见dataset；不是生产默认全量提取设置。已固定后供五组复用。',
                  '- 未进行检索参数调优，未根据这批test结果选择参数。数据按视频隔离，但未按作者隔离。',
                  '- 脚本里的检索耗时不含预生成向量或网络调用；不能用来宣称线上低延迟。',
                  '- 原始视频/转写/逐题检索输出在.runtime/evals/real_bilibili_v1/；可复核题集questions_silver.json和manual_review.csv位于本目录。',
                  '- 机器出题初期有思考输出截断和中断请求，用量可能不完整；嵌入实际usage保存在summary.json，供应商账单为费用依据。',
                  '', '下一步应回看原视频，补全等价支持和纠正错标，再冻结一份人工测试集。当前可说“做了真实视频上的对照探索”，不能说“人工测试集证明答案正确率提升”。',''])
    (a.output_dir/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'cases':len(d['cases']),'videos':len(d['videos']),'metrics':{k:{m:v[m] for m in ['recallAt1','recallAt3','recallAt8','mrr']} for k,v in stats.items()},'intervals':intervals},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
