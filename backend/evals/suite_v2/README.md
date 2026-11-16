# VidLens 分阶段评测 V2

本目录把数据、证据快照和测评程序分开，类似 CV 项目中的标注集、输入图像和 evaluation.py。模型运行时只看问题、历史和证据，标准答案只交给评分器。

## 已提供的数据

| 文件 | 数量 | 用途与边界 |
| --- | ---: | --- |
| controlled_300.json | 300题，50场景 | 虚构材料的受控行为回归；同模板题相关，不是300个独立真实样本 |
| controlled_development.json | 198题，33场景 | 开发、排查失败、对照实验 |
| controlled_test.json | 102题，17场景 | 保留场景；不用于反复调参，也不代表真实视频泛化 |
| controlled_snapshots.json | 每场景21片段 | ASR/OCR混合，包含其他主题的相似参数、限制和负责人作为干扰项 |
| real_candidates.json | 58题，8视频 | 汇总旧题集，按BV号+去空白问题去重；全部待回查原视频 |
| real_review.csv | 58行 | 回看视频时修正答案、来源、时间窗、记录审核人和备注 |
| manifest.json | — | 实际数量、类别、划分和审核状态 |

六类受控题各50道：ASR事实、OCR参数、多证据、限制、历史追问、无答案。按场景整体划分，不能随机按问题拆分。同一材料的多道题并不是多个独立来源；真实候选题也还需要语义去重。real_review.csv只是审核工作表，修改它不会自动改变JSON金标；审核完成后需同步修正JSON并发布新版本，保留原始候选集。

真实题的数量需要靠增加视频、抽取真实证据、逐条核对来增长。不能把这300道受控题记成“300道真实视频人工金标”，也不能把两种数据混在一起报一个准确率。未经审核的真实候选题默认被新入口拒绝，调试时显式加 `--allow-candidates`，报告会记录调试标志。

## 单独测试哪些部分

| 阶段 | 固定输入 | 输出与指标 | 能说明什么 |
| --- | --- | --- | --- |
| 检索 | 问题+已固定的证据快照 | Recall@1/3/8、MRR、完整命中率、检索耗时 | 是否找回标注证据；按来源+时间重叠计分，不是语义相似度或文字识别准确率 |
| 规划 | 实际Agent子问题+参考规划 | 人工判断目标覆盖、是否扩展未问目标 | 不能靠拆得多或模型自评分证明规划好 |
| 回答 | 实际回答+标准要点 | 要点规则通过率、拒答精确率/召回率、人工正确率 | 关键词只做筛查，出现关键词也可能语义错误 |
| 引用 | 实际引用+原始材料 | 人工支持率 | 没有引用的拒答题填NA，不能自动算引用满分 |
| Agent重放 | 固定证据+问题/历史 | 规划、回答、工具轨迹、供应商usage、耗时、失败类型 | 排除ASR/OCR提取和队列影响，测Agent本身 |
| 完整流程 | 原视频+问题 | 原有API评测脚本的结果 | 覆盖提取、后台任务和Agent；COMPLETED只表示执行完成 |

真实视频ASR/OCR提取质量需另有转写/文字金标；本套快照检索不能测WER/CER。没有标准拆分的真实候选题，规划表的goldPlan会空，审核人需先补参考需求。不同分阶段指标不应混成单一“精度”。

## 从项目根目录运行

所有命令均在 `C:\code\vidlens` 运行。重建数据会重建候选审核CSV，填过的表请先另存到 `.runtime/evals/`。

```powershell
backend/.venv/Scripts/python.exe backend/scripts/prepare_eval_suite.py

backend/.venv/Scripts/python.exe backend/scripts/evaluate_suite.py retrieval --dataset backend/evals/suite_v2/controlled_development.json --snapshot backend/evals/suite_v2/controlled_snapshots.json --output .runtime/evals/dev_lexical.json

backend/.venv/Scripts/python.exe backend/scripts/evaluate_suite.py retrieval --dataset backend/evals/suite_v2/controlled_development.json --snapshot backend/evals/suite_v2/controlled_snapshots.json --profile contextual-lexical --output .runtime/evals/dev_contextual.json
```

前两种检索配置均是本地词法配置；需要比较BGE/混合检索时复用 `backend/scripts/evaluate_retrieval_ab.py`，传相同数据与快照。更换问题、金标或模型后，不能把结果差异全部归因于检索策略。

先检查Agent输入，不调用模型：

```powershell
backend/.venv/Scripts/python.exe backend/scripts/run_agent_replay.py --dataset backend/evals/suite_v2/controlled_development.json --snapshot backend/evals/suite_v2/controlled_snapshots.json --output .runtime/evals/replay_check.json
```

实际跑默认3题，增加 `--execute`；读取 `backend/.env` 中的AI_BASE_URL、AI_API_KEY、AI_MODEL，可用 `--env-file` 指向已有配置文件。默认整批上限36调用、50000Token、600秒，越界停止后续调用；单次响应可能越过Token上限，因此不是预先锁定费用。不会回退Mock凑结果。首次先跑3题检查输出，再逐批扩展，调整 `--max-cases`、`--max-calls` 等参数。

```powershell
backend/.venv/Scripts/python.exe backend/scripts/run_agent_replay.py --dataset backend/evals/suite_v2/controlled_development.json --snapshot backend/evals/suite_v2/controlled_snapshots.json --output .runtime/evals/agent_dev.json --execute

backend/.venv/Scripts/python.exe backend/scripts/evaluate_suite.py answer --dataset backend/evals/suite_v2/controlled_development.json --predictions .runtime/evals/agent_dev.json --output .runtime/evals/answer_dev.json --export-review .runtime/evals/agent_review.csv

backend/.venv/Scripts/python.exe backend/scripts/evaluate_suite.py planning --dataset backend/evals/suite_v2/controlled_development.json --predictions .runtime/evals/agent_dev.json --review .runtime/evals/agent_review.csv --output .runtime/evals/planning_dev.json
```

只跑3题后若用198题文件评分，未运行的195题会作为缺失预测，不能拿这个分数当模型效果。冒烟检查时先把这3题导出成独立的数据文件；正式报告必须使用完整计划题集。审核表字段0/1，未知留空，不适用NA；必须填审核人。没有人工评分时语义正确率为null，不是0，也不是100%。每条评分绑定对应预测指纹，换输出后要重新审核。

Agent重放复用生产结构化Agent、工具箱、归一化、门禁和问题改写，但历史追问在改写后接结构化Agent，与线上简短追问的回答链路不同，报告已经注明。检索入口仅输入本轮问题，不偷用gold的standaloneQuestion。因此检索中的追问题分数不能替代完整历史理解测试。

已有完整流程入口为 `backend/scripts/evaluate_real_video.py`，需要运行中的API、真实媒体ID映射和登录Token；使用 `--max-cases` 等上限。它仍兼容旧数据，审核状态保护在本次新增的入口中实现。不要用受控虚构videoId直接请求真实视频API。

## 预测文件与审核规则

评分器兼容原API评测输出的results数组，也支持以下最小结构。内容必须来自实际运行，不能把goldAnswer复制进去当模型结果。

```json
{"results":[{"caseId":"fixture-001-q1","state":"COMPLETED","answerableResult":true,"finalAnswer":"这里填写实际输出","plan":["实际子问题"],"trace":{}}]}
```

未知/重复caseId会报错；缺预测、空答案、错误answerable类型和预算中止都不能算正确拒答。规则缺失标记NOT_SCORED。人工引用支持率与生产门禁的字符串/时间匹配分开记录。

## 下一步怎样得到能写简历的结果

1. 扩充真实视频来源，按视频先分开发与最终测试，候选题要回看时间点，校正文字/证据/无答案判断。建议先把现有58题核对完整，再加新视频；不靠同义改写堆数量。
2. 在开发集固定快照、模型、Prompt、预算、评分版本。保存改动前结果，仅更换一个要验证的策略，再跑同题。
3. 统计检索完整命中、多子问题覆盖、答案正确、引用支持、拒答与用量；看错题而非只看平均值。
4. 最终未见视频只在方案冻结后跑。报告实际题数、视频数和人工审核数，再考虑写“某策略使某指标从A提升到B”。

本次已运行范围与结果见项目内 `.runtime/evals/suite_v2_execution.json`；没有执行批量付费Agent调用，没有生成真实视频准确率或提升数字。


2026-11-16更新：已完成百炼云端嵌入接入及五组检索受控开发集对照，包含实际付费嵌入调用。详细条件、结果和限制见 [RETRIEVAL_COMPARISON.md](RETRIEVAL_COMPARISON.md)。前文“未调用”仅对应更早的套件初次落地。
