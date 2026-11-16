# 真实B站视频检索对照（2026-11-16）

本批选择16条公开讲解视频，选片及按视频划分development/test均先于检索结果。采用便利抽样，覆盖AI、科普、数码、英语备考；不代表B站总体分布。同作者视频可能相关，未作作者隔离。

## 固定协议

- 复用项目生产ASR/OCR提取函数：云端Fun-ASR与已有OCR模型，真实时间戳。公开视频接口能提供的最高不超过720p画质，实际画质逐视频记录；每视频全程均匀抽最多12帧。为了控制成本，抽帧上限与生产默认40帧不同，实验只比较固定证据上的检索。
- 单独的`.runtime/evals/real_bilibili_v1/`存原视频、提取缓存、来源哈希、完整证据和生成审查记录；不修改用户现有媒体和数据库。
- 已有DeepSeek模型基于真实证据出题，先校验片段ID与原文引用，再用单独提示进行初审；之后百炼Qwen-plus另作证据支持审查，排除其判定不可靠的题。两轮审核都看不到检索结果，但仍可能有模型共同偏差。明确标记`MACHINE_EVIDENCE_REVIEWED`，不是人工回看原视频的金标。
- 不凑固定数量，不加入未验证的无答案题。只测有答案题的证据检索，不能证明拒答、规划、最终答案正确率或ASR/OCR准确率。
- 五组固定：现有IDF、BM25、纯云端向量、BM25＋向量RRF、IDF＋向量RRF。云端模型统一百炼text-embedding-v4/1024维；两组RRF深度20、k=60、等权0.5。未使用本批检索结果调整参数或选题。
- 检索范围是每题所属视频内的全部证据，符合项目“给定视频问答”场景；不测全库跨视频检索。
- 主看Recall@3，辅看Recall@1/8、MRR与完整命中率。对每个标注事实按精确片段ID命中；多个片段可作为同一事实的替代支持。不会因邻接时间重叠算命中。标注可能遗漏等价支持，分数存在偏低风险。
- 逐题宏平均；另列按视频平均及按视频重采样的成对差异区间，避免把同一视频内的题当完全独立样本。检索计时不包含预生成云端向量的耗时。
- 本轮不据受控题的结果选择策略；先固定上述参数，再一次性观察真实题的development/test。数据标签仍需人工复核，因此即使文件叫test，也不是正式人工测试集结论。

## 运行

在项目根目录：

```powershell
backend/.venv/Scripts/python.exe backend/scripts/build_real_video_benchmark.py collect --max-videos 16 --execute
backend/.venv/Scripts/python.exe backend/scripts/build_real_video_benchmark.py questions --max-videos 16 --execute
backend/.venv/Scripts/python.exe backend/scripts/audit_real_video_candidates.py --execute
backend/.venv/Scripts/python.exe backend/scripts/compare_retrieval_methods.py --dataset .runtime/evals/real_bilibili_v1/audited_dataset.json --snapshot .runtime/evals/real_bilibili_v1/snapshots.json --output .runtime/evals/real_bilibili_v1/method_matrix.json --allow-candidates --execute
backend/.venv/Scripts/python.exe backend/scripts/summarize_real_retrieval_comparison.py
```

前两步去掉`--execute`只展示任务。完成的提取和模型结果按视频保存，可复用。不能将未经核对的数据改为`VIDEO_VERIFIED`来绕过审核。首次出题时供应商默认思考模式导致部分输出被截断，已保存收到的响应、改为关闭思考后重新出题；被中断请求的用量未知，不能把账单当成只包含最终成功调用。

标注复核应对每题回看证据时间点，校正转写错误、必要事实和所有等价支持；不能只复核某方案的错题后立即宣称最终测试提升。
