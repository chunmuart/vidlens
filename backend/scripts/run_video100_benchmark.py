"""One entry point for the video100 silver evaluation. Default is a non-billable plan."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]


def commands(directory,output):
    scripts=ROOT/'scripts'
    return [
        [str(scripts/'build_agent_video_dataset.py'),'--directory',str(directory),'--output',str(output),'--execute'],
        [str(scripts/'compare_retrieval_methods.py'),'--dataset',str(directory/'dataset.json'),
            '--snapshot',str(directory/'snapshots.json'),'--output',str(directory/'method_matrix.json'),'--allow-candidates','--execute'],
        [str(scripts/'evaluate_agent_stages.py'),'--directory',str(directory),'--output',str(output),'--max-cases','60','--execute'],
        [str(scripts/'evaluate_agent_answers.py'),'--directory',str(directory),'--output',str(output),'--max-cases','12','--execute'],
        [str(scripts/'build_evaluation_delivery.py'),'--directory',str(directory),'--output',str(output)]]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,default=ROOT.parent/'.runtime/evals/agent_video100_v2')
    p.add_argument('--output',type=Path,default=ROOT/'evals/agent_video100_v2')
    modes=p.add_mutually_exclusive_group();modes.add_argument('--execute',action='store_true');modes.add_argument('--report-only',action='store_true')
    modes.add_argument('--cached-only',action='store_true',help='完整回放已有数据/响应；缺缓存即停止，禁止新模型请求')
    a=p.parse_args();steps=commands(a.directory.resolve(),a.output.resolve())
    if a.report_only:steps=steps[-1:]
    elif not (a.execute or a.cached_only):
        print(json.dumps({'executed':False,'note':'--report-only只读取已有结果；--execute允许调用已配置云端模型，优先复用缓存。',
            'commands':[[sys.executable,*step] for step in steps]},ensure_ascii=False,indent=2));return
    if a.cached_only:os.environ['VIDLENS_EVAL_OFFLINE']='1'
    for i,step in enumerate(steps):
        print(json.dumps({'step':i+1,'script':Path(step[0]).name}),flush=True)
        subprocess.run([sys.executable,*step],cwd=ROOT.parent,check=True)
        if Path(step[0]).name=='build_agent_video_dataset.py':
            summary=json.loads((a.output/'annotation_summary.json').read_text(encoding='utf-8'))
            if summary['failures']:raise RuntimeError('Annotation incomplete; successful paid calls retained, inspect failure records')
    print(json.dumps({'completed':True,'report':str(a.output/'RESULTS.md')},ensure_ascii=False))


if __name__=='__main__':main()
