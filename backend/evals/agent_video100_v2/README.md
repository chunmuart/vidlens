# VidLens 真实视频 Agent 评测交付

先读 RESULTS.md（本次真实数字），再读 INTERVIEW.md（如何解释），遇到问题查 ERROR_CASES.md。

本次素材100视频／437.58分钟；837题，机器银标，人工视频核验0题。

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
