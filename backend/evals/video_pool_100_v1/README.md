> 当前标注已完成：见 [837题与阶段实测成果](../agent_video100_v2/RESULTS.md) 和 annotation_status_v2.json。下文“尚未标注”等是采集完成时的历史记录；inventory.labelStatus保留当时快照，人工视频核验仍为0。

# 后续评测视频素材池（2026-11-16）

实际完成 **100/100条视频**，总时长 **437.58分钟**，共 **5982条ASR/OCR证据**。
其中首批15条复用已完成的原始文件和证据，新增85条；按BV和媒体SHA256去重，全部原始媒体校验通过。
不是只存100个链接。每条均有本地MP4、ASR文字、抽帧OCR结果、时间戳、稳定片段ID与供应商提取记录。

## 目前能用于什么

- 15条有已跑过检索的274道机器银标题；这些来源已是探索集，不能再充当未见终测。
- 新增视频是证据素材，尚未出题或人工标注；当前人工视频金标为0。
- inventory中的split保留首批实验的历史分组；未来数据划分以人工工作表的finalSplit为准，目前未冻结。
- ASR/OCR是机器输出，文件校验通过不表示转写正确。未来可从同一视频派生规划、检索、验证、修复和最终回答任务。
- 发现来源分层（不是人工主题金标）：{'AI': 31, 'science': 26, 'consumer': 26, 'education': 17}。便利抽样来自公开视频相关列表，不代表B站总体；作者和近重复需在冻结划分前再审。
- 成功文件中的提取状态：{('SUCCESS', 'SUCCESS'): 100}。零OCR文字不一定是失败，可能抽取画面没有文字，逐条见videos.csv。

## 在哪里找

- `videos.csv`：100条素材的标题、来源、时长、证据条数和本地文件位置。
- `inventory.json`：机器可读完整清单，含失败替补记录；失败不算完成。
- `integrity.json`：逐视频媒体/证据指纹、提取状态和已知usage。
- `failed_extractions.json`：8条失败来源的分路状态、错误码与已有用量；原始素材保留，可另作困难案例复核。
- `video_review_tasks.csv`：视频复核、作者/近重复分组、最终划分工作表；标签空白是待做。
- `split_proposal.json`：按作者整体分组的草案，{'development': 60, 'validation': 20, 'test': 20}；首批作者固定开发，74位作者无跨组，但尚未审核跨作者近重复，未冻结终测。
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

成功视频文件记录OCR **1146次请求、560747个供应商报告Token**（包括复用首批），ASR原始usage逐条保存。
这不是本轮新增账单，也不含可能失败/中断请求的未知费用；失败用量不能当0，费用以平台账单为准。
