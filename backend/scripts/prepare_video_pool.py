"""Build a resumable 100-video evidence pool, independently of the scored 15-video pilot."""
from __future__ import annotations

import argparse
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import logging
import os
from pathlib import Path
import shutil
import sys
import time

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_real_video_benchmark import collect_one, read
from vidlens.bilibili import BILIBILI_HEADERS
from vidlens.evidence_pipeline import atomic_json

POOL = ROOT.parent / '.runtime/evals/video_pool_100_v1'
PUBLIC = ROOT / 'evals/video_pool_100_v1'
PILOT = ROOT.parent / '.runtime/evals/real_bilibili_v1'


def discover(directory: Path):
    target = directory / 'candidates.json'
    if target.exists():
        return read(target)['videos']
    seeds = read(ROOT / 'evals/real_bilibili_v1/videos.json')['videos']
    rows, seen = [], set()
    queue = deque()
    author_counts = Counter()
    excluded = []
    for seed in seeds:
        path = PILOT / seed['bvid'] / 'evidence.json'
        if path.exists():
            video = read(path)
            row = {**seed, 'title':video['title'], 'durationSeconds':video['durationMs']//1000,
                   'uploaderId':video['uploaderId'], 'origin':'pilot', 'selectionOrder':len(rows)}
            rows.append(row)
            author_counts[row['uploaderId']] += 1
            seen.add(seed['bvid'])
            queue.append((seed['bvid'], seed['domain'], 0))
        else:
            seen.add(seed['bvid'])
            excluded.append({'bvid':seed['bvid'], 'reason':'prior_extraction_failure_not_retried'})
    queried = set()
    with httpx.Client(headers=BILIBILI_HEADERS, timeout=25, follow_redirects=True) as client:
        while queue and len(rows) < 220 and len(queried) < 48:
            bvid, domain, depth = queue.popleft()
            if bvid in queried:
                continue
            queried.add(bvid)
            time.sleep(0.5)
            try:
                response = client.get('https://api.bilibili.com/x/web-interface/archive/related', params={'bvid':bvid})
                response.raise_for_status()
                payload = response.json()
                if payload.get('code') != 0 or not isinstance(payload.get('data'),list):
                    raise ValueError('Unavailable related list')
            except Exception as exc:
                excluded.append({'bvid':bvid, 'reason':'discovery_'+type(exc).__name__})
                continue
            added = 0
            for item in payload['data']:
                bv = item.get('bvid')
                if not bv or bv in seen:
                    continue
                seen.add(bv)
                seconds = int(item.get('duration') or 0)
                title = str(item.get('title') or '')
                owner = (item.get('owner') or {}).get('mid')
                # Bounded short explanatory material. These labels are discovery strata, not semantic gold.
                if not 90 <= seconds <= 420 or not owner or author_counts[owner] >= 4:
                    continue
                if any(word in title.lower() for word in ['纯音乐','完整版歌曲','翻唱','舞蹈','游戏实况','纯享','抽奖','福利视频']):
                    continue
                row = {'bvid':bv,'title':title,'durationSeconds':seconds,'uploaderId':owner,
                       'domain':domain,'origin':'related','discoveredFrom':bvid,'discoveryDepth':depth+1,
                       'selectionOrder':len(rows), 'split':'future_unassigned'}
                rows.append(row)
                author_counts[owner] += 1
                if depth < 2:
                    queue.append((bv,domain,depth+1))
                added += 1
                if added >= 12 or len(rows)>=220:
                    break
            print(json.dumps({'stage':'discover','candidates':len(rows),'pages':len(queried)}),flush=True)
    # Interleave discovery strata so a partial run is not dominated by the first seed.
    first = [r for r in rows if r['origin']=='pilot']
    groups = {domain:deque(r for r in rows if r['origin']!='pilot' and r['domain']==domain)
              for domain in ['AI','science','consumer','education']}
    ordered = first[:]
    while any(groups.values()):
        for group in groups.values():
            if group: ordered.append(group.popleft())
    for index,row in enumerate(ordered):row['collectionOrder']=index
    value = {'protocolVersion':'video-pool-100-v1','videos':ordered,'excluded':excluded,
             'selection':'Public related lists; 90–420s for new videos; max 4 per uploader; interleaved discovery strata; no retrieval scores used.',
             'note':'Pilot retains its original split. New videos are unassigned, not a frozen test set.'}
    atomic_json(target,value)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic_json(PUBLIC/'candidates.json',value)
    return ordered


def publish(directory, rows, failures, target):
    inventory, hashes = [], set()
    duplicate = []
    for row in rows:
        path = directory/row['bvid']/'evidence.json'
        if not path.exists():continue
        video = read(path)
        if video['mediaSha256'] in hashes:
            duplicate.append(row['bvid'])
            continue
        hashes.add(video['mediaSha256'])
        sources = Counter(s['source'] for s in video['segments'])
        inventory.append({**row,'durationSeconds':video['durationMs']/1000,'title':video['title'],
                          'uploaderId':video['uploaderId'],'mediaSha256':video['mediaSha256'],
                          'segmentCount':len(video['segments']),'asrSegments':sources['ASR'],'ocrSegments':sources['OCR'],
                          'evidencePath':str(path),'mediaPath':str(directory/row['bvid']/(row['bvid']+'.mp4')),
                          'collectionProtocol':video['collectionProtocol'],
                          'labelStatus':'PILOT_SILVER_QUESTIONS_AVAILABLE' if row['origin']=='pilot' else 'EVIDENCE_ONLY_UNLABELED'})
    value = {'protocolVersion':'video-pool-100-v1','target':target,'completed':len(inventory),
             'durationMinutes':round(sum(v['durationSeconds'] for v in inventory)/60,2),
             'segments':sum(v['segmentCount'] for v in inventory), 'videos':inventory,'failures':failures,
             'duplicateMediaExcluded':duplicate,'humanVideoVerified':0}
    atomic_json(directory/'inventory.json',value)
    PUBLIC.mkdir(parents=True,exist_ok=True)
    atomic_json(PUBLIC/'inventory.json',value)
    print(json.dumps({k:value[k] for k in ['completed','target','durationMinutes','segments']},ensure_ascii=False),flush=True)
    return value


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--target',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.target<=100:raise ValueError('Target must be 1–100')
    POOL.mkdir(parents=True,exist_ok=True)
    load_dotenv(ROOT/'.env')
    rows=discover(POOL)
    if not args.execute:
        print(json.dumps({'candidates':len(rows),'target':args.target,'execute':False}));return
    os.environ['UPLOAD_ROOT']=str(POOL/'extraction-cache')
    os.environ['DATABASE_URL']='sqlite:///'+str(POOL/'unused_eval.db').replace('\\','/')
    os.environ['PADDLEOCR_MAX_FRAMES']='12'
    os.environ['PADDLEOCR_FRAME_MAX_WIDTH']='1280'
    os.environ['ASR_LOCAL_FALLBACK_ENABLED']='false'
    os.environ['ASR_POLL_TIMEOUT_SECONDS']='300'
    logging.getLogger('httpx').setLevel(logging.WARNING)
    for row in rows:
        if row['origin']!='pilot':continue
        src=PILOT/row['bvid'];dst=POOL/row['bvid']
        if not dst.exists():
            # Hardlinks keep the independent pool paths without duplicating large pilot files.
            shutil.copytree(src,dst,copy_function=os.link)
    failure_path=POOL/'failures.json'
    failures=read(failure_path) if failure_path.exists() else []
    failed={f['bvid'] for f in failures}
    current=publish(POOL,rows,failures,args.target)
    pending=[r for r in rows if not (POOL/r['bvid']/'evidence.json').exists() and r['bvid'] not in failed]
    # Only submit the remaining required slots; substitute failures from the prespecified reserve.
    with ThreadPoolExecutor(max_workers=2) as pool:
        while current['completed']<args.target and pending:
            batch=pending[:min(2,args.target-current['completed'])]
            pending=pending[len(batch):]
            futures={pool.submit(collect_one,row,POOL):row for row in batch}
            for future in as_completed(futures):
                row=futures[future]
                try:future.result()
                except Exception as exc:
                    failures.append({'bvid':row['bvid'],'errorType':type(exc).__name__,
                                     'retryPolicy':'No automatic resubmission of failed paid extraction'})
                    atomic_json(failure_path,failures)
                    print(json.dumps({'stage':'failed','video':row['bvid'],'errorType':type(exc).__name__}),flush=True)
                current=publish(POOL,rows,failures,args.target)
    if current['completed']<args.target:
        raise RuntimeError('Candidate reserve exhausted; see inventory, do not count failures as completed')


if __name__=='__main__':main()
