from copy import deepcopy
import pytest
from scripts.build_agent_video_dataset import structural_cases, guarded_call
from scripts.evaluate_agent_stages import fact_coverage, confusion, select_cases, sequential_ledger, reference_plan
from vidlens.agent_structured import _assemble_ledger


def case():
    return {'caseId':'v-a','videoId':'v','split':'development','question':'比较两款产品的续航与售价？',
        'goals':[{'goalId':'G1','objective':'产品续航'},{'goalId':'G2','objective':'产品售价'}],
        'facts':[{'factId':'F1','goalId':'G1','segmentIds':['s1','s1-alt']},
                 {'factId':'F2','goalId':'G2','segmentIds':['s2']}]}


def test_alternative_ids_are_or_but_distinct_facts_are_and():
    c=case()
    assert fact_coverage(c,{'evidence':[{'segmentId':'s1-alt'}]})['factRecall']==.5
    assert not fact_coverage(c,{'evidence':[{'segmentId':'s1'},{'segmentId':'s1-alt'}]})['allFacts']
    assert fact_coverage(c,{'evidence':[{'segmentId':'s1-alt'},{'segmentId':'s2'}]})['allFacts']


def test_empty_ledger_is_not_full_coverage():
    assert fact_coverage(case(),{'evidence':[]})['goalCoverage']==0


def test_insufficiency_is_positive_class_not_answerability():
    rows=[{'gold':{'sufficient':False},'pred':True}, {'gold':{'sufficient':False},'pred':False},
          {'gold':{'sufficient':True},'pred':False},{'gold':{'sufficient':True},'pred':True}]
    r=confusion(rows,'pred')
    assert (r['tp'],r['fn'],r['fp'],r['tn'])==(1,1,1,1)
    assert r['insufficientRecall']==r['falsePassRate']==r['falseRejectRate']==.5


def test_absent_class_is_undefined_not_perfect():
    r=confusion([{'gold':{'sufficient':True},'p':True}],'p')
    assert r['insufficientRecall'] is None


def test_selection_one_question_per_video_and_no_predicted_scores():
    rows=[]
    for i in range(20):
        for n in range(2):
            c=deepcopy(case());c.update(caseId=f'v{i}-{n}',videoId=f'v{i}')
            if n==0:c['goals']=c['goals'][:1]
            c['split']=['development','validation','test'][i%3]
            rows.append(c)
    a=select_cases(rows,12);b=select_cases(list(reversed(rows)),12)
    assert [c['caseId'] for c in a]==[c['caseId'] for c in b]
    assert len({c['videoId'] for c in a})==12
    assert all(len(c['goals'])==2 for c in a)


def test_both_allocation_policies_have_same_18_item_cap():
    c=case();p=reference_plan(c)
    def seg(i):return {'segmentId':f's{i}','source':'ASR','startMs':i,'endMs':i+1,'content':'证据正文'}
    direct={'G1':[seg(i) for i in range(1,9)],'G2':[seg(i) for i in range(30,38)]}
    adjacent={'G1':[seg(i) for i in range(9,22)],'G2':[seg(i) for i in range(38,50)]}
    seq=sequential_ledger(p,c['question'],direct,adjacent,{})
    rr=_assemble_ledger(p,c['question'],direct,adjacent,{})
    assert len(seq['evidence'])==len(rr['evidence'])==18
    assert not any('G2' in s['requirementIds'] for s in seq['evidence'])
    assert any('G2' in s['requirementIds'] for s in rr['evidence'])


def test_structural_labels_reject_nonexistent_or_nonverbatim_support():
    seg=[{'segmentId':'s1','source':'ASR','content':'续航可以达到十小时'}]
    raw={'cases':[{'question':'视频里产品能续航多久？','answer':'十小时','type':'fact',
        'goals':[{'goalId':'G1','objective':'续航时长'}],
        'facts':[{'factId':'F1','goalId':'G1','claim':'十小时','segmentIds':['s1'],'quote':'可以达到十小时'}]}]}
    assert len(structural_cases(raw,seg)[0])==1
    raw['cases'][0]['facts'][0]['quote']='可以达到二十小时'
    assert not structural_cases(raw,seg)[0]


def test_unresolved_paid_attempt_is_not_automatically_retried(tmp_path):
    p=tmp_path/'draft.json'
    (tmp_path/'draft.attempt-old.json').write_text('{}')
    with pytest.raises(RuntimeError,match='previous model attempt'):
        guarded_call([],p)


def test_atomic_json_retries_transient_windows_sharing_error(tmp_path,monkeypatch):
    import vidlens.evidence_pipeline as m
    if m.os.name!='nt':pytest.skip('Windows sharing behavior')
    original=m.os.replace;attempts=[]
    def transient(a,b):
        attempts.append(1)
        if len(attempts)<3:
            err=PermissionError('sharing violation');err.winerror=32;raise err
        return original(a,b)
    monkeypatch.setattr(m.os,'replace',transient)
    monkeypatch.setattr(m.time,'sleep',lambda _:None)
    p=tmp_path/'record.json';m.atomic_json(p,{'complete':True})
    assert len(attempts)==3 and m.read_json(p)=={'complete':True}


def test_answer_complete_flag_cannot_override_missing_facts_or_refusal():
    from scripts.evaluate_agent_answers import summarize
    r={'referenceFactCount':2,'predictions':{'direct':{'report':{'answerable':True}}},
       'reviews':[{'name':'direct','claimCount':1,'supportedClaimCount':1,'citationCount':1,
           'supportedCitationCount':1,'coveredReferenceFactIds':['F1'],
           'allRequiredFactsCovered':True,'answerableDecisionCorrect':True}]}
    assert summarize([r])['direct']['completeAnswerRateAllSelected']==0
    r['reviews'][0]['coveredReferenceFactIds']=['F1','F2']
    r['predictions']['direct']['report']['answerable']=False
    assert summarize([r])['direct']['completeAnswerRateAllSelected']==0


def test_failed_answer_cases_remain_in_all_selected_denominator():
    from scripts.evaluate_agent_answers import summarize
    r={'referenceFactCount':1,'predictions':{'direct':{'report':{'answerable':True}}},
       'reviews':[{'name':'direct','claimCount':1,'supportedClaimCount':1,'citationCount':1,
           'supportedCitationCount':1,'coveredReferenceFactIds':['F1'],
           'allRequiredFactsCovered':True,'answerableDecisionCorrect':True}]}
    v=summarize([r],2)['direct']
    assert v['completeAnswerRateAllSelected']==.5 and v['failed']==1
