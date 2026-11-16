from scripts.evaluate_real_evidence import requirements, matched_requirements, retrieval_metrics
from scripts.summarize_real_retrieval_comparison import cluster_delta
import pytest
import json
from scripts import prepare_video_pool
from scripts.summarize_video_pool import verify_video, write_new_csv, sha256_file, append_new_tasks, split_proposal


def test_exact_gold_alternatives_are_one_fact_and_not_adjacent_time():
    gold = {"evidenceId":"fact", "sources":["ASR","OCR"], "segmentIds":["a","b"], "startMs":0,"endMs":30000}
    req = requirements({"goldEvidence":[gold]})
    assert len(req) == 1
    assert matched_requirements([{"segmentId":"adjacent","source":"ASR","startMs":0,"endMs":1000}],req) == set()
    segments = [{"segmentId":"b","source":"OCR","startMs":20000,"endMs":30000}]
    assert matched_requirements(segments,req) == {("fact","ANY")}
    metrics = retrieval_metrics(segments,segments,req,[1,3,8])
    assert metrics["recallAt1"] == 1


def test_legacy_time_gold_keeps_source_requirement():
    gold = {"evidenceId":"fact", "sources":["ASR","OCR"], "startMs":0,"endMs":1000}
    req = requirements({"goldEvidence":[gold]})
    assert len(req) == 2
    assert matched_requirements([{"source":"ASR","startMs":0,"endMs":1000}],req) == {("fact","ASR")}


def test_paired_video_interval_and_pairing_guard():
    rows=[{"caseId":"a","videoId":"v1","recallAt3":1.0},{"caseId":"b","videoId":"v2","recallAt3":.5}]
    base=[{"caseId":"a","videoId":"v1","recallAt3":.5},{"caseId":"b","videoId":"v2","recallAt3":0.0}]
    result=cluster_delta(rows,base,repeats=100)
    assert result["difference"] == result["lower"] == result["upper"] == .5
    with pytest.raises(ValueError,match="Unpaired"):
        cluster_delta(rows,base[:1],repeats=100)


def test_pool_does_not_count_failed_sources_or_duplicate_media(tmp_path, monkeypatch):
    monkeypatch.setattr(prepare_video_pool, 'PUBLIC', tmp_path/'public')
    rows=[]
    for bvid in ['first','duplicate','failed']:
        rows.append({'bvid':bvid,'origin':'related'})
        if bvid=='failed':continue
        folder=tmp_path/bvid;folder.mkdir()
        (folder/'evidence.json').write_text(json.dumps({
            'mediaSha256':'same-file','durationMs':60000,'title':bvid,'uploaderId':1,
            'segments':[{'source':'ASR'}],'collectionProtocol':'test'}),encoding='utf-8')
    result=prepare_video_pool.publish(tmp_path,rows,[{'bvid':'failed'}],100)
    assert result['completed']==1
    assert result['duplicateMediaExcluded']==['duplicate']
    assert result['durationMinutes']==1


def test_inventory_integrity_rejects_changed_media(tmp_path):
    media=tmp_path/'media.mp4';media.write_bytes(b'original-video-fixture')
    evidence=tmp_path/'evidence.json'
    evidence.write_text(json.dumps({'mediaSha256':sha256_file(media),
        'segments':[{'segmentId':'s1','source':'ASR','content':'speech','startMs':0,'endMs':1000}],
        'extraction':{'branches':{'ASR':{'status':'SUCCESS'},'OCR':{'status':'SUCCESS'}}}}),encoding='utf-8')
    row={'bvid':'fixture','mediaPath':str(media),'evidencePath':str(evidence)}
    assert verify_video(row)['mediaVerified'] is True
    media.write_bytes(b'changed-video')
    with pytest.raises(ValueError,match='changed raw media'):
        verify_video(row)


def test_review_export_never_resets_filled_human_work(tmp_path):
    path=tmp_path/'review.csv'
    write_new_csv(path,['reviewer'],[{'reviewer':''}])
    path.write_text('reviewer\nAlice\n',encoding='utf-8')
    write_new_csv(path,['reviewer'],[{'reviewer':''}])
    assert path.read_text(encoding='utf-8')=='reviewer\nAlice\n'


def test_growing_video_pool_appends_only_new_review_tasks(tmp_path):
    import csv
    path=tmp_path/'tasks.csv';fields=['bvid','reviewer']
    write_new_csv(path,fields,[{'bvid':'a','reviewer':'Alice'}])
    additions=[{'bvid':'a','reviewer':''},{'bvid':'b','reviewer':''}]
    append_new_tasks(path,fields,additions,'bvid')
    append_new_tasks(path,fields,additions,'bvid')
    with path.open(encoding='utf-8-sig',newline='') as handle:rows=list(csv.DictReader(handle))
    assert rows==[{'bvid':'a','reviewer':'Alice'},{'bvid':'b','reviewer':''}]


def test_future_split_keeps_authors_together_and_seen_pilot_in_development():
    rows=[{'bvid':str(i),'uploaderId':i//2,'origin':'pilot' if i==0 else 'related'} for i in range(20)]
    proposal=split_proposal(rows)
    by_author={}
    for row in proposal['videos']:
        by_author.setdefault(row['uploaderId'],set()).add(row['proposedSplit'])
    assert all(len(splits)==1 for splits in by_author.values())
    assert by_author[0]=={'development'}
    assert sum(proposal['actualCounts'].values())==len(rows)
    assert split_proposal(rows)==proposal
