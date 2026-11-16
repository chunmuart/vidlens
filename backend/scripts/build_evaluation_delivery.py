"""Publish the actual video100 experiment, scorer inputs, error cases and interview guide.

Read-only with respect to providers: no model/network calls. Does not invent human labels.
"""
from __future__ import annotations
import argparse
from collections import Counter,defaultdict
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.build_real_video_benchmark import read
from scripts.evaluate_suite import digest
from scripts.evaluate_real_evidence import summarize_metrics
from scripts.evaluate_agent_stages import aggregate
from scripts.evaluate_agent_answers import summarize as summarize_answers
from vidlens.evidence_pipeline import atomic_json

METHODS={'lexical':'现有 IDF 词法','bm25':'BM25','dense':'云端向量','bm25_dense_rrf':'BM25＋向量 RRF','idf_dense_rrf':'IDF＋向量 RRF'}
def pct(v):return '无可用分母' if v is None else f'{v:.2%}'
def pp(v):return f'{v*100:+.2f} 个百分点'
def write(path,text):path.write_text(text.strip()+'\n',encoding='utf-8')


def paired_interval(left,right,authors,metric='recallAt3',repeats=2000):
    r={v['caseId']:v for v in right};groups=defaultdict(list)
    for v in left:
        assert v['caseId'] in r
        groups[str(authors[v['videoId']])].append(float(v[metric])-float(r[v['caseId']][metric]))
    values=[np.asarray(v) for _,v in sorted(groups.items())]
    if not values:return None
    rng=np.random.default_rng(20261116)
    draws=[float(np.concatenate([values[i] for i in sample]).mean()) for sample in rng.integers(0,len(values),size=(repeats,len(values)))]
    return {'delta':float(np.concatenate(values).mean()),'lower':float(np.quantile(draws,.025)),
        'upper':float(np.quantile(draws,.975)),'unit':'uploader','clusters':len(values),'repeats':repeats,'seed':20261116}


def annotation_qa(data,snapshots):
    ids=[];authors={};errors=[]
    for v in data['videos']:
        author=str(v['uploaderId'])
        if author in authors and authors[author]!=v['split']:errors.append('AUTHOR_SPLIT_LEAK')
        authors[author]=v['split']
    for c in data['cases']:
        ids.append(c['caseId']);segs={s['segmentId']:s for s in snapshots[c['videoId']]}
        gids={g['goalId'] for g in c['goals']}
        if gids!={f['goalId'] for f in c['facts']}:errors.append('GOAL_FACT_MISMATCH')
        for f in c['facts']:
            if not set(f['segmentIds'])<=segs.keys():errors.append('UNKNOWN_EVIDENCE')
            elif f['quote'] not in segs[f['segmentIds'][0]]['content']:errors.append('NONVERBATIM_QUOTE')
        if c['humanVideoVerified'] is not False:errors.append('UNEXPECTED_HUMAN_LABEL')
    if len(ids)!=len(set(ids)):errors.append('DUPLICATE_CASE_ID')
    if errors:raise ValueError('Dataset integrity failed: '+','.join(sorted(set(errors))))
    return {'caseCount':len(ids),'videoCount':len(data['videos']),'labeledVideos':len({c['videoId'] for c in data['cases']}),
        'humanVideoVerified':0,'uniqueCaseIds':True,'evidenceIdsExist':True,'quotesVerbatim':True,
        'authorsDoNotCrossSplits':True,'semanticEquivalenceHumanChecked':False,
        'byType':dict(Counter(c['type'] for c in data['cases'])),'bySplit':dict(Counter(c['split'] for c in data['cases']))}


def publish_stage_datasets(data,predictions,output):
    atomic_json(output/'planning_cases.json',{'kind':'REAL_VIDEO_SILVER','humanVerified':False,
        'note':'测试输入只取question，goals仅评分器可见；合理合并允许。',
        'cases':[{k:c[k] for k in ('caseId','videoId','split','question','goals')} for c in data['cases']]})
    mapping={c['caseId']:c for c in data['cases']};verifier=[];repair=[];allocation=[]
    for r in predictions['results']:
        c=mapping[r['caseId']]
        for variant,v in r['verification'].items():
            verifier.append({'caseId':r['caseId']+'::'+variant,'videoId':r['videoId'],'split':r['split'],
                'question':c['question'],'kind':'REAL_DERIVED_CONTROLLED_LEDGER','plan':{'answerMode':v['ledger']['answerMode'],'requirements':v['ledger']['requirements']},
                'ledger':v['ledger'],'goldSufficient':v['gold']['sufficient'],'goldMissingGoalIds':v['gold']['missingGoalIds'],
                'machineReason':v['gold']['reason'],'humanVerified':False})
        for key,kind in (('repair','NATURAL_RETRIEVAL'),('controlledRepair','REAL_DERIVED_CONTROLLED_LEDGER')):
            v=r[key]
            repair.append({'caseId':r['caseId']+'::'+key,'videoId':r['videoId'],'split':r['split'],'kind':kind,
                'question':c['question'],'initialState':v['stateBefore'],'goldFacts':c['facts'],
                'semanticVerdicts':v['independentSemanticVerdicts'],'triggered':v['triggered'],
                'methodOutputs':v['variants'],'humanVerified':False})
        allocation.append({'caseId':r['caseId'],'videoId':r['videoId'],'question':c['question'],'goldFacts':c['facts'],
            'inputAndOutputs':r['boundaries'],'humanVerified':False})
    atomic_json(output/'verifier_cases.json',{'cases':verifier,'note':'180受控证据清单，不等于180独立视频；缺失标签由独立模型逐清单核对。'})
    atomic_json(output/'repair_cases.json',{'cases':repair,'note':'自然与受控分开，参考facts不传给待测节点；共享相同初始状态、一次修复额度和18条证据上限。'})
    atomic_json(output/'allocation_cases.json',{'cases':allocation,'note':'相同候选，分别用标注目标/实际规划，比较顺序填充和轮流分配；18条×每条最多360字。'})


def usage_inventory(directory):
    totals=defaultdict(lambda:{'requests':0,'promptTokens':0,'completionTokens':0,'totalTokens':0})
    unknown=[]
    for p in directory.rglob('*.json'):
        if p.name.endswith('.pending.json'):unknown.append(str(p.relative_to(directory)));continue
        if '.attempt-' in p.name:
            stem=p.name.split('.attempt-')[0]
            if (p.parent/(stem+'.json')).exists():continue
            d=read(p)
            if d.get('status')!='RESPONSE_RECEIVED':unknown.append(str(p.relative_to(directory)));continue
            usage=d.get('usage') or {};model=d.get('model','unknown')
        else:
            d=read(p)
            if 'output' in d and 'usage' in d:
                usage=d.get('usage') or {};model=d.get('model','unknown')
            elif 'request' in d and 'message' in d:
                event=d['request']['usage'];usage=event.get('rawUsage') or {};model=event.get('model','unknown')
            else:continue
        t=totals[model];t['requests']+=1
        for a,b in [('promptTokens','prompt_tokens'),('completionTokens','completion_tokens'),('totalTokens','total_tokens')]:t[a]+=int(usage.get(b,0))
    return {'scope':'Annotation plus final stage/answer request caches; excludes extraction and separate smoke sandbox',
        'byModel':dict(totals),'unresolvedUsageArtifacts':unknown,'moneyCost':None,
        'note':'保留供应商真实usage；未配置并核对价格不臆算金额。缓存回放不代表又付费一次。'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT.parent/'.runtime/evals/agent_video100_v2')
    p.add_argument('--output',type=Path,default=ROOT/'evals/agent_video100_v2');a=p.parse_args()
    d=read(a.directory/'dataset.json');snapshots=read(a.directory/'snapshots.json')
    matrix=read(a.directory/'method_matrix.json');pred=read(a.output/'stage_predictions.json');answers=read(a.output/'answer_summary.json')
    answer_predictions=read(a.output/'answer_predictions.json')
    assert answer_predictions['datasetFingerprint']==digest(d)
    answers['metrics']=summarize_answers(answer_predictions['results'],answers['selected'])
    atomic_json(a.output/'answer_summary.json',answers)
    assert matrix['datasetFingerprint']==digest(d)==pred['selection']['datasetFingerprint']
    assert matrix['snapshotFingerprint']==digest({k:snapshots[k] for k in sorted({c['videoId'] for c in d['cases']})})
    assert not pred['failures'] and len(pred['results'])==len(pred['selection']['selectedCaseIds']), 'Incomplete stage run'
    qa=annotation_qa(d,snapshots);a.output.mkdir(parents=True,exist_ok=True)
    atomic_json(ROOT/'evals/video_pool_100_v1/annotation_status_v2.json',{
        'dataset':'../agent_video100_v2/questions_silver.json','questions':len(d['cases']),
        'kind':'MACHINE_SILVER','humanVideoVerified':0,
        'videos':[{'videoId':v['videoId'],'acceptedQuestions':sum(c['videoId']==v['videoId'] for c in d['cases']),
            'status':'MACHINE_REVIEWED_HAS_ACCEPTED_CASES' if any(c['videoId']==v['videoId'] for c in d['cases']) else 'ALL_CANDIDATES_REJECTED'} for v in d['videos']],
        'note':'原inventory.labelStatus保留采集时快照状态，现行标注看本文件。'})
    atomic_json(a.output/'quality_checks.json',qa);atomic_json(a.output/'evidence_snapshots.json',snapshots)
    atomic_json(a.output/'retrieval_predictions.json',matrix)
    mapping={c['caseId']:c for c in d['cases']};authors={v['videoId']:v['uploaderId'] for v in d['videos']}
    bysplit={s:{name:summarize_metrics([r for r in v['cases'] if mapping[r['caseId']]['split']==s],[1,3,8])
        for name,v in matrix['profiles'].items()} for s in ('development','validation','test')}
    intervals={name:paired_interval(v['cases'],matrix['profiles']['bm25']['cases'],authors) for name,v in matrix['profiles'].items() if name!='bm25'}
    retrieval={'aggregate':{k:v['aggregate'] for k,v in matrix['profiles'].items()},'bySplit':bysplit,
        'deltaRecallAt3VsBm25AuthorBootstrap95':intervals,'providerUsageThisRun':matrix['providerUsage'],
        'embeddingGenerationUsage':matrix.get('embeddingGenerationUsage',matrix['providerUsage']),
        'embedding':matrix['embedding'],'matchingPolicy':matrix['matchingPolicy']}
    atomic_json(a.output/'retrieval_summary.json',retrieval)
    selected=[mapping[cid] for cid in pred['selection']['selectedCaseIds']]
    stages={'aggregate':aggregate(pred['results'],selected),'bySplit':{s:aggregate([r for r in pred['results'] if r['split']==s],selected)
        for s in ('development','validation','test') if any(r['split']==s for r in pred['results'])},
        'selectedCount':len(selected),'completedCount':len(pred['results']),'selection':pred['selection'],'failures':[]}
    atomic_json(a.output/'stage_summary.json',stages);publish_stage_datasets(d,pred,a.output)
    context=[]
    for r in pred['results']:
        for origin,b in r['boundaries'].items():
            for method,ledger in b['ledgers'].items():
                context.append({'caseId':r['caseId'],'planOrigin':origin,'method':method,
                    'evidenceCount':len(ledger['evidence']),
                    'evidenceContentChars':sum(len(s['content']) for s in ledger['evidence']),
                    'serializedCharsRecomputed':len(json.dumps(ledger,ensure_ascii=False,separators=(',',':')))})
    atomic_json(a.output/'context_budget_summary.json',{'limits':{'evidenceItems':18,'charsPerItem':360},'cases':context,
        'note':'从实际保存清单重算；仅条数和单条截断相同，不保证每种方案总字符数完全一致。'})
    costs={}
    for name in ('direct','agent'):
        calls=[];tokens=[];latencies=[]
        for row in answer_predictions['results']:
            method=row['predictions'].get(name)
            if not method:continue
            events=method['requests'];calls.append(len(events))
            tokens.append(sum(e['usage'].get('totalTokens',0) for e in events))
            latencies.append(sum(e['elapsedMs'] for e in events))
        costs[name]={'completed':len(calls),'meanModelCalls':float(np.mean(calls)),
            'meanProviderTokens':float(np.mean(tokens)),'summedModelLatencyP50Ms':float(np.quantile(latencies,.5)),
            'summedModelLatencyP95Ms':float(np.quantile(latencies,.95)),
            'note':'串行模型请求耗时之和；不含提取、排队、工具计算，非完整用户端响应时间；独立评分调用不计入待测方法成本。'}
    atomic_json(a.output/'answer_cost_summary.json',costs)
    usage=usage_inventory(a.directory);atomic_json(a.output/'provider_usage.json',usage)
    inventory=read(ROOT/'evals/video_pool_100_v1/inventory.json')
    minutes=sum(v['durationSeconds'] for v in inventory['videos'])/60
    evidence_count=sum(len(v) for v in snapshots.values())
    stats=stages['aggregate'];lines=[
        '# VidLens：100 视频评测实际成果',
        '', '状态：真实素材＋机器银标＋独立模型复核的探索性实验。日期：2026-11-16。不是人工视频金标，不能当作线上总体准确率。',
        '',f"## 数据规模\n\n采集100条视频，{minutes:.2f}分钟，{evidence_count:,}条ASR/OCR证据，{len(set(authors.values()))}位作者。100条均完成标注尝试；{qa['labeledVideos']}条有通过审核的问题。",
        f"保留{len(d['cases'])}道题，{sum(len(c['goals'])>1 for c in d['cases'])}道多目标题，{sum(len(c['facts']) for c in d['cases'])}个原子事实。按作者分组的临时划分：{json.dumps(qa['bySplit'],ensure_ascii=False)}。",
        '一条视频候选全部未通过，保留拒绝记录；不补容易题凑数。旧15条视频的作者全部进入开发组。跨作者语义近重复尚未人工排查，test是临时分组名称，不能宣称严格独立终测。',
        '题集以能在单片段直接支持的原子事实为主；多目标题需要多个事实，但不代表已经覆盖深层跨片段推理。',
        '', '## 检索：837题上的五种方案',
        '', '| 方案 | Recall@1 | Recall@3 | Recall@8 | MRR |','| --- | ---: | ---: | ---: | ---: |']
    for name,label in METHODS.items():
        v=retrieval['aggregate'][name];lines.append(f"| {label} | {pct(v['recallAt1'])} | {pct(v['recallAt3'])} | {pct(v['recallAt8'])} | {v['mrr']:.4f} |")
    lines.extend(['','主指标预先用Recall@3；Recall@8观察给Agent更多材料时的覆盖。相同题、相同视频证据、相同云端向量；按单视频检索，未测全库跨视频搜索。',
        'Recall按必要事实计分：同事实的多个等价片段是“或”，不同事实是“且”；不会把多个替代片段当多个必需事实。MRR是首个支持事实的倒数排名，不等于所有必要事实都已找齐。',
        '云端text-embedding-v4，1024维，L2归一化；BM25 k1=1.5、b=0.75；两种融合均RRF k=60、候选深度20、等权。当前IDF还含短语匹配与不同分词，差异不能全部归因于IDF公式。',
        '检索耗时是预生成向量后的计算耗时，不能当在线端到端响应速度。',
        '', '| 对比 BM25 的 Recall@3 差值 | 差值 | 按作者重采样95%区间 |','| --- | ---: | --- |'])
    for name,v in intervals.items():lines.append(f"| {METHODS[name]} | {pp(v['delta'])} | [{pp(v['lower'])}, {pp(v['upper'])}] |")
    lines.extend(['','区间跨0时，不能说存在稳定提升。没有提升、甚至下降的方案也保留。各临时分组完整结果见retrieval_summary.json。',
        '',f"## 规划：{len(selected)}条视频各一道题",'',
        '| 方案 | 目标覆盖率 | 全目标覆盖率 | 原子槽位比例 | 多余槽位比例 |','| --- | ---: | ---: | ---: | ---: |'])
    for name,label in [('original_query','原问题一条'),('rule','已有规则规划'),('structured','生产结构化规划')]:
        v=stats['planning'][name];lines.append(f"| {label} | {pct(v['coverage'])} | {pct(v['complete'])} | {pct(v['atomicPrecision'])} | {pct(v['extraRate'])} |")
    lines.extend(['','规划器只看问题，不看答案与标注目标。独立评分模型看参考目标和输出槽位，按含义匹配。原问题合在一条通常仍覆盖全部需求，所以“拆得更细”不自动代表“覆盖更高”；真正作用还要看后续证据。',
        '', '## 证据分配：固定同一批候选与18条上限', '', '| 规划来源 | 分配方式 | 事实召回 | 全事实覆盖率 | 最后目标覆盖率 |','| --- | --- | ---: | ---: | ---: |'])
    for origin,label in [('reference_goals','标注目标'),('actual_planner','真实规划')]:
        for method,text in [('sequential','逐槽填充含相邻证据'),('round_robin','先直接证据轮流分配'),('sequential_reversed','逐槽填充·逆序'),('round_robin_reversed','轮流分配·逆序')]:
            v=stats['allocation'][origin][method];lines.append(f"| {label} | {text} | {pct(v['factRecall'])} | {pct(v['allFacts'])} | {pct(v['lastGoalCovered'])} |")
    lines.extend(['','每条证据最多360字，均使用同一生产格式化函数；总条数上限相同，实际总字符数可能不同并已记录。这里隔离使用现有IDF检索器，避免同时换检索与Agent流程后无法归因；云端混合另表比较。覆盖按标注ID计，不等于最终回答语义正确。',
        '', '## 验证：识别当前证据是否不足', '', '| 方案 | 不足检出率 | 不足错误放行率 | 充分错误拦截率 | Macro-F1 |','| --- | ---: | ---: | ---: | ---: |'])
    for name,label in [('nonempty','有候选就放行'),('production','生产语义验证器')]:
        v=stats['verifier'][name];lines.append(f"| {label} | {pct(v['insufficientRecall'])} | {pct(v['falsePassRate'])} | {pct(v['falseRejectRate'])} | {v['macroF1']:.4f} |")
    lines.extend(['','正类是“证据不足”。每题构造完整、缺末项目标、相似干扰三份清单，独立模型重新核对实际充分性；不能按构造名称直接决定正确标签。180份清单来自60条视频，不是180个独立视频。它衡量受控场景识别能力，不能解释为自然发生率。',
        '', '## 一次补检索：自然缺失与受控缺失分开', '', '| 场景 | 方式 | 初始不足数 | 补救成功数 | 不足样本补救率 | 最终充分率 |','| --- | --- | ---: | ---: | ---: | ---: |'])
    for key,label in [('repair','实际规划与检索'),('controlledRepair','真实证据构造缺失')]:
        for method,text in [('none','不补'),('repeat_original','重复原问题'),('targeted','针对验证器缺项补一次')]:
            v=stats[key][method];lines.append(f"| {label} | {text} | {v['initialInsufficientCount']} | {v['rescued']} | {pct(v['rescueRate'])} | {pct(v['finalSemanticSufficiency'])} |")
    lines.extend(['','只把生产验证器识别出的缺项交给修复，不把标准答案或金标缺项交给Agent。同一初始清单、一次共享修复额度、18条证据上限；多缺项时每个缺项可调用检索，邻接窗口调用数可能不同，toolTrace逐项记录，所以不是严格同等调用成本。若验证器误放行或审计已耗尽修复额度，两种方案都不能继续修复。',
        '补救成功率分母是独立评分确认初始不足的全部样本，包含验证器漏检而没有触发修复的样本。另有finalSemanticSufficiency、harmCount及精确ID覆盖可核对。',
        '', '## 最终答案与引用：12题小样本对照', '', '| 方式 | 完成题数 | 主张支持精确率 | 答案事实召回 | 完整正确率（含失败） | 引用支持精确率 |','| --- | ---: | ---: | ---: | ---: | ---: |'])
    for name,label in [('direct','直接检索＋一次回答'),('agent','完整生产Agent')]:
        v=answers['metrics'][name];lines.append(f"| {label} | {v['completed']}/{answers['selected']} | {pct(v['claimPrecision'])} | {pct(v['referenceFactRecall'])} | {pct(v['completeAnswerRateAllSelected'])} | {pct(v['citationSupportPrecision'])} |")
    lines.extend(['','| 方式 | 平均模型调用 | 平均供应商Token | 模型耗时和P50 | 模型耗时和P95 |','| --- | ---: | ---: | ---: | ---: |'])
    for name,label in [('direct','直接检索＋一次回答'),('agent','完整生产Agent')]:
        v=costs[name];lines.append(f"| {label} | {v['meanModelCalls']:.2f} | {v['meanProviderTokens']:.0f} | {v['summedModelLatencyP50Ms']/1000:.2f}秒 | {v['summedModelLatencyP95Ms']/1000:.2f}秒 |")
    lines.extend(['','Token取实际供应商usage，包含该方法所有阶段；耗时是串行模型请求之和，不含提取/排队/工具计算，也不把离线评分模型耗时算入方法成本。更多阶段不免费，必须连同效果与成本一起判断。'])
    lines.extend(['','两种方案都使用固定真实证据和相同IDF检索后端；这是整条链路对照，不是单独证明Critic有效。独立模型核查原子主张和实际引用语义；ID存在不等于支持。12题只用于流程与错例验证，不能用于宣称总体准确率。当前题目全部可回答，尚不能报告真正视频无答案的拒答性能。',
        '', '## 复现与边界', '',
        '从项目根目录运行：', '', '```powershell',
        r'backend\.venv\Scripts\python.exe backend\scripts\run_video100_benchmark.py --report-only', '```','',
        '上面只重建报告。完整流程需要显式执行，使用已有云端配置并优先复用同输入缓存：', '', '```powershell',
        r'backend\.venv\Scripts\python.exe backend\scripts\run_video100_benchmark.py --execute', '```','',
        '如需完整回放已有缓存且禁止新模型请求，把 --execute 换成 --cached-only；缺缓存会停止。',
        '每次模型请求有输入指纹、原始用量和响应缓存；未确认完成的付费请求不会盲目重试。数据/提示/模型改变时使用新版本目录，不能覆盖同一缓存后比较。',
        '本轮未修改生产Agent策略，未切换默认检索器。新功能是离线数据与评测工具；Windows原子文件保存补了短暂占用重试。',
        '尚未覆盖：人工回看视频金标、跨作者语义重复清理、全库跨视频检索、追问专用对照、真正无答案题、ASR/OCR识别精度及并发压测。不要把这些写成已完成。',
        '文件入口见README.md；面试叙述见INTERVIEW.md；具体错例见ERROR_CASES.md。'])
    write(a.output/'RESULTS.md','\n'.join(lines))
    errors=['# 实际错例与对照样本','','这些是本次运行的真实输出，不是为了展示提升而编写的例子。']
    errors_csv=[]
    for r in pred['results']:
        c=mapping[r['caseId']]
        seq=r['allocation']['actual_planner']['sequential']['factRecall'];rr=r['allocation']['actual_planner']['round_robin']['factRecall']
        natural={v['name']:v['sufficient'] for v in r['repair']['independentSemanticVerdicts']}
        controlled={v['name']:v['sufficient'] for v in r['controlledRepair']['independentSemanticVerdicts']}
        wrong=[k for k,v in r['verification'].items() if v['prediction']['verification']['overallSufficient']!=v['gold']['sufficient']]
        if seq!=rr or wrong or controlled.get('targeted')!=controlled.get('repeat_original'):
            errors_csv.append({'caseId':r['caseId'],'question':c['question'],'sequentialRecall':seq,'roundRobinRecall':rr,
                'verifierWrongVariants':json.dumps(wrong,ensure_ascii=False),'naturalRepair':json.dumps(natural),'controlledRepair':json.dumps(controlled),
                'humanDiagnosis':'','humanCorrection':''})
    for row in errors_csv[:15]:
        errors.extend(['',f"## {row['caseId']}",f"问题：{row['question']}",
            f"顺序/轮流分配事实召回：{pct(row['sequentialRecall'])} / {pct(row['roundRobinRecall'])}。验证器错判变体：{row['verifierWrongVariants']}。",
            f"受控补检索判断：`{row['controlledRepair']}`。",'原始证据、缺项解释和检索工具输出可按caseId在stage_predictions.json中核查；以下不把机器分歧自动认定为生产错误。'])
    write(a.output/'ERROR_CASES.md','\n'.join(errors))
    review=a.output/'stage_error_review.csv'
    if not review.exists():
        with review.open('w',encoding='utf-8-sig',newline='') as f:
            fields=list(errors_csv[0]) if errors_csv else ['caseId','question','humanDiagnosis','humanCorrection']
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(errors_csv)
    write(a.output/'README.md',f'''# VidLens 真实视频 Agent 评测交付

先读 RESULTS.md（本次真实数字），再读 INTERVIEW.md（如何解释），遇到问题查 ERROR_CASES.md。

本次素材100视频／{minutes:.2f}分钟；{len(d['cases'])}题，机器银标，人工视频核验0题。

| 文件 | 内容 |
| --- | --- |
| questions_silver.json | 全量问句、答案、需求目标、原子事实、替代证据ID、时间与审核理由 |
| evidence_snapshots.json | 100视频固定ASR/OCR文本证据；不需要再下载视频即可评测 |
| planning_cases.json | 全量规划输入与目标标注 |
| verifier_cases.json | 60题×3种真实证据清单与独立充分性标签 |
| repair_cases.json | 自然/受控初始状态、修复结果、评分标签 |
| allocation_cases.json | 同批候选、两种分配与逆序实验 |
| *_predictions.json | 各方案逐题实际输出、生产节点返回和工具轨迹 |
| *_summary.json | 总体及分组指标 |
| manual_review.csv | 全量问题的人工复核表；人工栏为空 |
| stage_error_review.csv | 阶段差异与人工诊断表 |
| annotation_rejections.json | 结构或语义审核拒绝记录 |
| quality_checks.json | ID、精确引文、作者分组等结构验证 |
| provider_usage.json | 模型调用真实usage；不臆算金额 |
| manifest.json | 交付文件SHA256与版本指纹 |

克隆仓库后可直接查看本目录的标注、快照、逐题预测和汇总。
仅在保留原运行目录 .runtime/evals/agent_video100_v2 时，从项目根运行 backend/.venv/Scripts/python.exe backend/scripts/run_video100_benchmark.py --report-only 重建报告。
--cached-only 完整离线回放同样需要原运行缓存，缺缓存即停止；公开交付目录不包含这些缓存。
完整流程加 --execute；需要 backend/.env 中现有云端配置。原媒体与模型缓存仍在 .runtime/evals/ 下，交付包不包含密钥或原始视频。

目录可独立查看；运行源码在同一项目backend/scripts中，压缩包附带本轮相关脚本。运行时需项目依赖与生产vidlens包。
''')
    best=max(METHODS,key=lambda k:retrieval['aggregate'][k]['recallAt3']);idf=retrieval['aggregate']['idf_dense_rrf'];bm=retrieval['aggregate']['bm25']
    guide=f'''# 面试讲解：从数据到真实结果

## 先说人话

这套评测相当于你做CV时保存验证图片、标注框与预测结果。区别是这里分阶段：问题拆解的标注是需求目标，检索的标注是证据片段，验证的标注是“这批证据是否足够”，补检索看前后有没有补全，回答再看事实与引用。

评测集是检验工具，不会让Agent自动训练或自进化。必须据错例提出修改，再固定条件做对照。

## 可以直接讲的一分钟版本

“我围绕视频问答Agent建立了分阶段离线评测流程，采集100条视频，保存ASR/OCR证据快照，生成并经独立模型审核后保留{len(d['cases'])}道机器银标题，并标注需求目标、原子答案事实、时间位置与等价证据。
我先在相同题集上比较IDF、BM25、云端向量及两种RRF混合，再从60条视频各抽一道题，分离规划、18条证据预算下的分配、证据充分性验证和一次补检索。最终用12道题检查完整Agent的答案及引用。
结果不是混合一定更好：本轮Recall@3最高的是{METHODS[best]}。我保留分组、置信区间和负向错例，能定位是检索找不到、分配挤掉、验证误放行还是回答越过证据。当前是自动标注的探索实验，下一步需要人工核验后再形成正式指标。”

## 简历建议写法

搭建视频问答Agent分阶段离线评测流程，基于100条真实视频构建{len(d['cases'])}道经独立模型审核的机器银标题，覆盖检索、规划、证据分配、充分性验证及一次补检索；保存证据快照、逐题轨迹与模型usage，完成五种检索策略和Agent阶段消融对照。

如果写数字，明确标注“离线机器银标探索集”：BM25 Recall@3={pct(bm['recallAt3'])}，IDF＋向量 Recall@3={pct(idf['recallAt3'])}。不要只挑对自己有利的K，不写尚未证明的总体提升。

## 面试追问与回答

**1. 题集怎么来，都是人工标的吗？**

不是。题目来自真实视频ASR/OCR证据，出题模型给需求、答案事实和证据，独立模型核对；程序再校验ID和精确引文。现在人工回看核验是0，所以明确叫机器银标，不能包装成人工金标。

**2. 为何不直接按视频标题编题？**

标题只帮助确定语境，答案必须由实际证据支持。只按标题很容易问到视频没讲的内容。

**3. Recall@3到底算什么？和CV的IoU一样吗？**

不是IoU。比如一道题需要3个独立事实，前三条检索结果找到2个，事实Recall@3就是2/3。替代证据任意命中一个就算该事实命中；全部事实命中率另算。IoU比较区域重叠，这里比较必要事实的覆盖。

**4. MRR说明回答完整吗？**

不能。MRR看第一个有用事实出现多靠前，多事实问题只找一个也可能得高分，所以同时看事实Recall和完整覆盖率。

**5. 为什么原问题不拆解也有很高目标覆盖率？**

用户原句本来就包含这些需求。拆解价值在于形成可独立检索和验证的槽位，不能把“没拆”直接当漏需求，要继续看检索与证据覆盖。

**6. 怎么证明证据分配有用？**

固定同一规划和同一批直接/邻接候选，同样最多18条，再比较逐槽填满和先直接证据轮流分配，另把槽位顺序反转。用标注目标和实际规划各跑一次，区分规划错误与分配错误。

**7. 验证器会不会自己给自己打分？**

生产验证器只看问题、规划和当前证据；独立评分模型看参考事实与证据来判充分性。仍有模型偏差，最终人工核验才能进一步确认。

**8. 怎么构造不足证据题？**

用真实证据构造缺少某目标或加入相似干扰的清单，再让独立模型检查实际缺失。删一个标注ID不代表答案一定没了，因为其他ASR/OCR可能提供等价信息。

**9. 验证器全判不足不就Recall很高吗？**

所以同时报告充分样本误拦率、混淆矩阵和Macro-F1，避免只追一个指标。

**10. 补检索是不是把标准答案拿去搜了？**

没有。待测修复只拿生产验证器指出的缺项和原规划。标准答案与标注事实只给评分器，模型请求轨迹可核查。

**11. 补救率分母是什么？**

独立评分确认初始不足的全部样本，包括验证器没发现而没触发修复的样本。不能只统计成功触发修复、容易补回的题。

**12. 成本真的完全公平吗？**

两种修复共享相同初始证据、一次额度和18条上限，缺项数相同；邻接窗口调用数可能不同，因此报告实际toolTrace与usage，不能宣称严格同成本。

**13. 引用ID对了就是有依据吗？**

不是。ID存在只是格式检查，还要看该片段是否支持答案中的对应主张。最终回答实验另外做语义核对。

**14. 100视频够不够？能否把一题的3个变体当3条视频？**

100视频可做起步实验，但规模不能弥补标签噪声。三种变体相关性很高，仍来自同一题同一视频；统计时按视频或作者分组，不能当完全独立样本。

**15. 为什么不能现在就写“准确率提升X%”？**

要说明哪个阶段、哪个指标、从哪个基线到哪个方案，以及是百分点还是相对提升。还要考虑机器标签、分组和置信区间。当前可以讲完成了真实对照和发现了哪些边界，不保证每个复杂模块都有提升。

## 建议学习顺序与检查

先看RESULTS.md五个表，再选ERROR_CASES.md一道真实问题，把“用户目标→候选→最终18条→验证→补检索→答案”顺着走一遍。

练习1：某题有3个事实，Top3命中2个，Recall@3是多少？答案：2/3；三个等价片段只支持同一个事实时不能算3次。

练习2：验证器把所有样本判不足，是否优秀？答案：不足检出率可能高，但充分样本误拦率很高，必须一起看。

练习3：针对缺项补检索后只新增5条证据，是否证明有效？答案：不能，要看必要事实是否增加、最终是否足够以及新增成本。

复习安排：当天讲清一条真实错例；次日不看文档复述各阶段输入与标签；第3天解释Recall/MRR/误放行；第7天自己复跑报告并核查一个数字；第14天练习一分钟项目介绍。
'''
    write(a.output/'INTERVIEW.md',guide)
    files={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file() and p.name!='manifest.json'}
    manifest={'datasetFingerprint':digest(d),'snapshotFingerprint':digest(snapshots),'files':files,
        'status':'EXPLORATORY_MACHINE_SILVER','rawMediaIncluded':False,'credentialsIncluded':False,
        'questions':len(d['cases']),'stageCases':len(selected),'answerCases':answers['selected']}
    atomic_json(a.output/'manifest.json',manifest)
    archive=a.directory.parent/'vidlens-video100-evaluation-delivery.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in a.output.rglob('*'):
            if p.is_file():z.write(p,'evaluation/'+str(p.relative_to(a.output)).replace('\\','/'))
        for name in ('build_agent_video_dataset.py','evaluate_agent_stages.py','evaluate_agent_answers.py','compare_retrieval_methods.py','build_evaluation_delivery.py','run_video100_benchmark.py',
            'build_real_video_benchmark.py','evaluate_suite.py','prepare_eval_suite.py','evaluate_real_video.py','evaluate_real_evidence.py','evaluate_retrieval_ab.py'):
            p=ROOT/'scripts'/name
            if p.exists():z.write(p,'scripts/'+name)
    print(json.dumps({'questions':len(d['cases']),'stageCases':len(selected),'archive':str(archive),'bytes':archive.stat().st_size},ensure_ascii=False))


if __name__=='__main__':main()
