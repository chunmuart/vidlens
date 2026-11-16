# VidLens评测资料入口

先看数据来源和审核状态，再看分数。视频数量、题目数量和人工金标数量分别统计。

| 目录/文件 | 内容 | 当前用途 |
| --- | --- | --- |
| [agent_video100_v2](agent_video100_v2/RESULTS.md) | 100视频素材、837机器银标题、60题阶段对照、12题答案对照 | 当前主要成果；包含目标/事实/证据标注、真实结果、错例、复跑入口与面试讲解；人工视频核验仍为0 |
| [agent_video100_v2/INTERVIEW.md](agent_video100_v2/INTERVIEW.md) | 项目介绍、简历建议、15组面试追问 | 仅使用本次实测数字，明确机器银标与小样本边界 |
| [video_pool_100_v1](video_pool_100_v1/README.md) | 已完成100条真实B站视频，原媒体和云端ASR/OCR逐条留存 | 原始素材池；现行标注与评分见agent_video100_v2，旧labelStatus为采集时状态 |
| [video_pool_100_v1/EXPERIMENT_PLAN.md](video_pool_100_v1/EXPERIMENT_PLAN.md) | 最初的阶段实验设计及尚未覆盖事项 | 历史方案；已执行范围以agent_video100_v2/RESULTS.md为准 |
| [real_bilibili_v1](real_bilibili_v1/RESULTS.md) | 首批15条真实视频、274道机器审查银标题，五组检索实际对照 | 探索结果；人工视频核验0题，已看过成绩的来源不再算未见终测 |
| [real_bilibili_v1/ERROR_REVIEW.md](real_bilibili_v1/ERROR_REVIEW.md) | 正反错例与274题配对输出 | 开发错例复核，不能只挑提升的题 |
| [suite_v2](suite_v2/README.md) | 300道虚构材料受控题，另有58道旧真实候选题 | 回归、理解评分器、排查失败；非真实人工效果数据 |
| 其他已有JSON | 更早的基线或留出题资料 | 保留兼容，不能仅凭文件名认定已人工审核或仍未被看过 |

原视频与付费模型缓存保存在项目`.runtime/evals/`。当前交付目录也包含固定文本快照和逐题输出，便于无需重新提取即可核查；没有密钥或原始视频。完整压缩包在`.runtime/evals/vidlens-video100-evaluation-delivery.zip`。
各目录README给出运行命令。不要用标准答案替代实际预测，也不要把空白人工审核表算成已标注数据。
