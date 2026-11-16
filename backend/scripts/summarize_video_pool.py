"""Verify local video-pool artifacts and export inventory/review workbooks without model calls."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.build_real_video_benchmark import read
from vidlens.evidence_pipeline import atomic_json


def sha256_file(path):
    value=hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def verify_video(row):
    path=Path(row['evidencePath']);media=Path(row['mediaPath'])
    value=read(path)
    if not media.is_file() or sha256_file(media)!=value['mediaSha256']:
        raise ValueError('Missing or changed raw media: '+row['bvid'])
    segments=value['segments']
    ids=[s.get('segmentId') for s in segments]
    if not all(ids) or len(set(ids))!=len(ids):raise ValueError('Invalid segment IDs: '+row['bvid'])
    if not any(s['source']=='ASR' and str(s.get('content','')).strip() for s in segments):
        raise ValueError('No speech evidence: '+row['bvid'])
    for s in segments:
        if s['source'] not in ('ASR','OCR') or not 0<=s['startMs']<=s['endMs']:
            raise ValueError('Invalid source/timing: '+row['bvid'])
    branches=value['extraction']['branches']
    events=branches.get('OCR',{}).get('requestEvents',[])
    return {'bvid':row['bvid'],'mediaVerified':True,'evidenceSha256':sha256_file(path),
            'asrStatus':branches.get('ASR',{}).get('status'),
            'ocrStatus':branches.get('OCR',{}).get('status'),
            'ocrRequests':len(events),'ocrSucceeded':sum(e.get('status')=='SUCCESS' for e in events),
            'ocrReportedTokens':sum((e.get('usage') or {}).get('total_tokens',0) for e in events),
            'asrReportedDuration':(branches.get('ASR',{}).get('usage') or {}).get('duration')}


def write_new_csv(path,fields,rows):
    # Filled review work must never be reset by rerunning the inventory summary.
    if path.exists():return
    with path.open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def append_new_tasks(path,fields,rows,key):
    """Extend a growing inventory while preserving every existing review cell."""
    if not path.exists():
        write_new_csv(path,fields,rows)
        return
    with path.open(encoding='utf-8-sig',newline='') as handle:
        reader=csv.DictReader(handle)
        if reader.fieldnames!=fields:raise ValueError('Review worksheet schema changed; use a new output')
        existing=list(reader)
    ids=[row[key] for row in existing]
    if not all(ids) or len(set(ids))!=len(ids):raise ValueError('Duplicate/empty review identity')
    known=set(ids)
    additional=[row for row in rows if row[key] not in known]
    if additional:
        with path.open('a',encoding='utf-8',newline='') as handle:
            csv.DictWriter(handle,fieldnames=fields).writerows(additional)


def split_proposal(rows):
    groups={}
    for row in rows:groups.setdefault(str(row['uploaderId']),[]).append(row)
    targets={'development':round(len(rows)*.6),'validation':round(len(rows)*.2)}
    targets['test']=len(rows)-sum(targets.values())
    counts=Counter();assignments={}
    for author,videos in groups.items():
        if any(v['origin']=='pilot' for v in videos):
            assignments[author]='development';counts['development']+=len(videos)
    unseen=[author for author in groups if author not in assignments]
    unseen.sort(key=lambda author:hashlib.sha256(('20261116:'+author).encode()).hexdigest())
    for author in unseen:
        split=max(targets,key=lambda key:targets[key]-counts[key])
        assignments[author]=split;counts[split]+=len(groups[author])
    return {'status':'DRAFT_AUTHOR_GROUPED_REQUIRES_NEAR_DUPLICATE_REVIEW','targets':targets,
            'actualCounts':dict(counts),'authorCount':len(groups),'seed':'20261116',
            'note':'Pilot authors forced into development because pilot scores were inspected. No semantic near-duplicate review yet; not a frozen test set.',
            'videos':[{'bvid':r['bvid'],'uploaderId':r['uploaderId'],'proposedSplit':assignments[str(r['uploaderId'])]} for r in rows]}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,default=ROOT.parent/'.runtime/evals/video_pool_100_v1')
    parser.add_argument('--output',type=Path,default=ROOT/'evals/video_pool_100_v1')
    args=parser.parse_args()
    inventory=read(args.directory/'inventory.json')
    rows=inventory['videos']
    if len(rows)!=inventory['completed'] or len({r['bvid'] for r in rows})!=len(rows):
        raise ValueError('Inventory count/identity mismatch')
    if len({r['mediaSha256'] for r in rows})!=len(rows):raise ValueError('Duplicate media bytes')
    verified=[verify_video(row) for row in rows]
    args.output.mkdir(parents=True,exist_ok=True)
    proposal=split_proposal(rows)
    atomic_json(args.output/'split_proposal.json',proposal)
    atomic_json(args.output/'integrity.json',{'completed':len(rows),'videos':verified,
        'mediaIdentityPolicy':'unique BV and exact file SHA256; semantic near duplicates not yet human checked',
        'usageScope':'Successful video artifacts only. Failed/incomplete extraction usage may be unknown; provider billing is authoritative.'})
    failed_extractions=[]
    for failed in inventory['failures']:
        media=(args.directory/failed['bvid']/(failed['bvid']+'.mp4')).resolve()
        cache=args.directory/'extraction-cache/evidence-cache'/hashlib.sha256(str(media).encode()).hexdigest()
        branches={}
        for branch in ['ASR','OCR']:
            artifact=cache/(branch.lower()+'.json')
            if artifact.exists():
                raw=read(artifact)
                allowed=['status','provider','model','errorType','errorCode','retryable','taskId','usage','requestEvents']
                branches[branch]={k:raw[k] for k in allowed if k in raw}
        failed_extractions.append({**failed,'branches':branches,'countedAsCompleted':False})
    atomic_json(args.output/'failed_extractions.json',{'videos':failed_extractions,
        'note':'Known branch usage retained; absent usage is unknown, not zero. No failed paid job automatically resubmitted.'})
    fields=['bvid','title','domain','uploaderId','durationSeconds','segmentCount','asrSegments','ocrSegments',
            'labelStatus','mediaPath','evidencePath','mediaSha256']
    with (args.output/'videos.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader()
        writer.writerows({k:row[k] for k in fields} for row in rows)
    append_new_tasks(args.output/'video_review_tasks.csv',
        ['bvid','title','url','uploaderId','origin','suggestedUse','finalSplit','nearDuplicateGroup','speechChecked',
         'ocrChecked','questionCount','reviewer','notes'],
        [{'bvid':r['bvid'],'title':r['title'],'url':'https://www.bilibili.com/video/'+r['bvid'],
          'uploaderId':r['uploaderId'],'origin':r['origin'],
          'suggestedUse':'development_already_scored' if r['origin']=='pilot' else 'unassigned_unscored'} for r in rows], 'bvid')
    pilot=read(ROOT/'evals/real_bilibili_v1/questions_silver.json')
    cases=pilot['cases']
    write_new_csv(args.output/'planning_annotation_candidates.csv',
        ['caseId','videoId','question','candidateGoldAnswer','candidateGoldEvidence','requiredGoalsJSON',
         'acceptableMergedGoalsJSON','unrequestedGoalsJSON','reviewStatus','reviewer','notes'],
        [{'caseId':c['caseId'],'videoId':c['videoId'],'question':c['question'],
          'candidateGoldAnswer':c['goldAnswer'],'candidateGoldEvidence':json.dumps(c['goldEvidence'],ensure_ascii=False),
          'reviewStatus':'DRAFT_NEEDS_INTENT_ANNOTATION'} for c in cases])
    atomic_json(args.output/'stage_annotation_template.json',{
        'status':'EMPTY_TEMPLATE_NOT_A_SCORED_DATASET','cases':[],
        'fields':{
            'planning':['caseId','videoId','question','requiredAtomicGoals','acceptableMerges','reviewer','reviewStatus'],
            'verification':['caseId','sourceCaseId','videoId','question','plan','ledger','provenanceKind',
                            'goldSufficient','goldMissingGoalIds','supportingSegmentIds','reviewer','reviewStatus'],
            'allocation':['caseId','frozenPlan','directCandidates','adjacentCandidates','goldGoalToSegments',
                          'maxEvidence','maxChars','reviewer'],
            'repair':['caseId','initialLedger','initialVerification','retrievalQueries','finalLedger',
                      'goldBeforeSufficient','goldAfterSufficient','usageBefore','usageAfter','reviewer'],
            'answer':['caseId','requiredAnswerClaims','forbiddenClaims','claimToEvidence','answerable',
                      'reviewer','reviewStatus']},
        'separation':'Only question/plan/ledger/history enter the tested Agent; gold/reviewer fields are scorer-only.'})
    domains=Counter(r['domain'] for r in rows)
    tokens=sum(r['ocrReportedTokens'] for r in verified)
    ocr_requests=sum(r['ocrRequests'] for r in verified)
    statuses=Counter((v['asrStatus'],v['ocrStatus']) for v in verified)
    report=f'''# 后续评测视频素材池（2026-11-16）

实际完成 **{len(rows)}/{inventory['target']}条视频**，总时长 **{inventory['durationMinutes']}分钟**，共 **{inventory['segments']}条ASR/OCR证据**。
其中首批15条复用已完成的原始文件和证据，新增{sum(r['origin']!='pilot' for r in rows)}条；按BV和媒体SHA256去重，全部原始媒体校验通过。
不是只存100个链接。每条均有本地MP4、ASR文字、抽帧OCR结果、时间戳、稳定片段ID与供应商提取记录。

## 目前能用于什么

- 15条有已跑过检索的274道机器银标题；这些来源已是探索集，不能再充当未见终测。
- 新增视频是证据素材，尚未出题或人工标注；当前人工视频金标为0。
- inventory中的split保留首批实验的历史分组；未来数据划分以人工工作表的finalSplit为准，目前未冻结。
- ASR/OCR是机器输出，文件校验通过不表示转写正确。未来可从同一视频派生规划、检索、验证、修复和最终回答任务。
- 发现来源分层（不是人工主题金标）：{dict(domains)}。便利抽样来自公开视频相关列表，不代表B站总体；作者和近重复需在冻结划分前再审。
- 成功文件中的提取状态：{dict(statuses)}。零OCR文字不一定是失败，可能抽取画面没有文字，逐条见videos.csv。

## 在哪里找

- `videos.csv`：{len(rows)}条素材的标题、来源、时长、证据条数和本地文件位置。
- `inventory.json`：机器可读完整清单，含失败替补记录；失败不算完成。
- `integrity.json`：逐视频媒体/证据指纹、提取状态和已知usage。
- `failed_extractions.json`：{len(failed_extractions)}条失败来源的分路状态、错误码与已有用量；原始素材保留，可另作困难案例复核。
- `video_review_tasks.csv`：视频复核、作者/近重复分组、最终划分工作表；标签空白是待做。
- `split_proposal.json`：按作者整体分组的草案，{proposal['actualCounts']}；首批作者固定开发，{proposal['authorCount']}位作者无跨组，但尚未审核跨作者近重复，未冻结终测。
- `planning_annotation_candidates.csv`：复用274题的需求标注候选表；尚无原子目标金标，不能直接报规划分。
- `stage_annotation_template.json`：后续各阶段所需字段，空模板，不计入题量。
- `EXPERIMENT_PLAN.md`：完整对照设计、指标、标注和实施顺序。
- 原视频和完整证据：`.runtime/evals/video_pool_100_v1/<BV号>/`，不进入Git；首批媒体采用本地硬链接复用，不重复计费。

## 复现与续跑

在项目根目录运行：

```powershell
backend/.venv/Scripts/python.exe backend/scripts/prepare_video_pool.py
backend/.venv/Scripts/python.exe backend/scripts/prepare_video_pool.py --execute
backend/.venv/Scripts/python.exe backend/scripts/summarize_video_pool.py
```

第一条只发现/检查来源，不调用提取模型。第二条才实际下载和调用已有百炼ASR/OCR配置，最多两条并行，成功产物断点复用；已记录失败不自动重提付费任务，使用预先发现的候补补齐。
公开视频最高720p，实际画质逐条记录；最多抽12帧，OCR宽度上限1280，生产默认抽帧配置不同。固定快照是为了比较下游策略，不能据此声称测了生产全量OCR质量。
汇总脚本保留已存在的人工工作表，只为新增视频追加待办行；修改素材后应发布新版本，避免把旧审查表用于新产物。

成功视频文件记录OCR **{ocr_requests}次请求、{tokens}个供应商报告Token**（包括复用首批），ASR原始usage逐条保存。
这不是本轮新增账单，也不含可能失败/中断请求的未知费用；失败用量不能当0，费用以平台账单为准。
'''
    (args.output/'README.md').write_text(report,encoding='utf-8')
    print(json.dumps({'videos':len(rows),'minutes':inventory['durationMinutes'],'segments':inventory['segments'],
                      'pilotSilverCases':len(cases),'humanVideoVerified':0,'ocrRequestsRecorded':ocr_requests},ensure_ascii=False))


if __name__=='__main__':main()
