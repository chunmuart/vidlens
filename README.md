# VidLens

长视频内容理解平台。输入一段视频和一个问题，输出一份**每句话都能追溯到具体时间点**的分析报告。

课程回放、会议录像、技术分享这类长视频有两个现实问题：一小时的内容，人只能记住前十分钟；而直接丢给大模型做总结，模型会在没听清的地方编造，且你无从验证。

VidLens 的做法是把视频拆成「证据」再作答：ASR 转写音频、OCR 提取画面文字，按时间轴融合成带来源标记的证据流；回答问题时先检索证据，再要求模型只能基于证据作答，无证据即拒答。

```
视频 ──┬─ FFmpeg 抽音频 → ASR 转写 ─┐
       └─ 抽关键帧 → OCR 文字识别 ─┴─→ 时间轴融合 → 证据库 → 检索 → Agent → 带引用的报告
```

---

## 一、三个核心问题与我的解法

### 1. 长耗时任务不能塞进 HTTP 请求

一段 40 分钟的视频，ASR 要跑几分钟，OCR 再几分钟，加上 LLM 分析，整个链路是分钟级的。同步接口必然超时，用户刷新就前功尽弃。

**做法**：`POST /analysis/ai` 只做一件事——落库建任务、返回 task_id，然后把任务投进 RocketMQ。独立的 Worker 消费消息、跑重活、逐步更新进度，前端轮询 `GET /analysis/analysis-status` 恢复状态。

**取舍**：引入 MQ 的代价是部署多一个组件、多一条排查链路（消息是否投递、Worker 是否存活、消息是否重复）。换来的是接口毫秒级返回、Worker 可以单独限流限内存、失败可重试。对分钟级任务，这个交换是值得的；如果是秒级任务，我会直接同步做。

### 2. 大文件在弱网下传不完

GB 级视频一次性上传，网络一抖就全部重来。

**做法**：`init-upload` 创建上传会话 → `upload-chunk` 分片上传（服务端记录每片状态）→ `complete-upload` 合并校验。断点重连时先查 `upload-status`，只补传缺失的分片。同时用 MD5 作为内容指纹做用户级去重，同一用户重复上传直接命中已有记录。

**取舍**：分片带来合并时的一致性问题（如何保证分片原子写入、合并幂等），所以合并阶段加了 MD5 全校验——宁可慢一步，也不让半截文件进入分析队列。

### 3. 检索不准，模型就会开始编

这是这类产品真正的难点。朴素 RAG 做「问题 → topK → 塞进 prompt」，在视频场景下会系统性失效：

- 专有名词、缩写、精确数字用向量检索经常召回不到——实际标注数据里就有一道题专门考「某技术名词在模型里是几个 token」，靠语义相似度是捞不到这类精确信息的
- 一道题往往需要多个证据点（「分别讲了哪三点」「A 和 B 有什么区别」），单次 topK 天然覆盖不全

**做法**（`backend/vidlens/retrieval.py`）：

1. **双路召回**：lexical（关键词）+ 本地 BGE 稠密向量，用 **RRF** 融合。选 RRF 而不是加权求和，是因为两路分数量纲不同，按排名融合不需要调权重。
2. **需求拆解**：先用 `plan_evidence_requirements` 判断问题类型（SINGLE / MULTI_PART / EXHAUSTIVE_LIST / COMPARISON / SEQUENCE / TEMPORAL / SYNTHESIS），拆成若干证据需求，每个需求单独检索。
3. **邻接扩展 + 充分性门禁**：检索到证据后向时间轴前后扩展找上下文；再判断是否真的覆盖了所有需求，不满足就继续检索或明确拒答。

**取舍**：这套「Coverage-Aware」策略比朴素 RAG 慢（开发集上查询均值约 8ms → 15ms），且在留出集上并没有泛化（见下文）。但它把「答错了却看不出来」变成了「答不出来就说答不出来」，对可信度优先的场景，我认为后者更重要。

---

## 二、Agent 与幻觉控制

`backend/vidlens/agent_structured.py` 里的分析 Agent 走五段式流程，而不是自由 ReAct：

```
Planner → Retriever → Verifier → Writer → Critic
```

- **Planner**：把问题拆成结构化槽位（要回答什么、每条需求依赖什么证据）
- **Retriever**：按槽位检索，构建 Evidence Ledger（证据账本）
- **Verifier**：独立判断每条结论是否被证据支持，分四级
- **Writer**：只能使用 Ledger 中的证据 ID 成文
- **Critic**：检查完整性、矛盾、外部推断、是否该拒答

其中最关键的是 **supportLevel 四级分档**：

| 级别 | 含义 | 处理方式 |
|---|---|---|
| `DIRECT` | 视频里逐字说了 | 正常回答 + 引用 |
| `SYNTHESIS` | 由多条证据概括得出 | 正常回答，标注为概括 |
| `GROUNDED_INFERENCE` | 视频没明说，但可由证据合理推出 | **程序强制**追加「视频没有明确说明」 |
| `NONE` | 证据不支持 | 拒答 |

**为什么不用 prompt 让模型自己标注**：模型对「自己是否在推理」的判断并不稳定。所以这四级判定由 Verifier 给出，但「追加声明」和「拒答」这两个动作由程序执行——不依赖模型自觉。

另有一条 `agent_graph.py`（LangGraph 状态机：`plan → prefetch → model_step ⇄ tool_execution → finalize`），用于需要多轮动态工具调用的场景。两条链路并存是为了能对比和回滚，通过 `AGENT_PIPELINE_VERSION` 切换。

---

## 三、评测：我如何知道它真的有效

这一块我认为比功能本身更能说明问题。

评测指标：Recall@1/3/8、MRR、多证据完整命中率、无答案误召回率、分 ASR/OCR 来源的命中、延迟。脚本在 `backend/scripts/`，数据集在 `backend/evals/`。

**开发集上的结果**：加入 Contextual 策略后，Recall@8 从 0.867 提升到 1.000，多证据完整命中率从 0.800 提升到 1.000；后续 Coverage-Aware v2 把 MRR 从 0.850 提到 0.875、Recall@8 从 0.775 提到 1.000。

**留出集上的结果——这才是重点**：在一条**从未用于调参**的视频集上，Coverage-Aware v2 的 MRR 从 lexical 的 0.568 掉到 0.465，Recall@8 从 0.682 掉到 0.591，且 5 道不可回答题全部未在检索层拒答。

也就是说：**开发集上的提升没有泛化**。我做了一个容易被忽略但很重要的决定——承认这个结果，冻结该留出集不再重跑，而不是换一个指标或继续在这个集合上调参。同时定位到根因不是证据抽取（282 段证据的金标覆盖率是 1.000，说明抽取层没丢东西），而是检索覆盖与拒答泛化。

这个结论被我写进了 README 而不是删掉，因为它才是我对这个系统真实水平的判断。

---

## 四、工程细节

**用户隔离**：JWT 登录，所有媒体、证据、报告按 `user_id` 强制过滤。向量检索同样带 `video_id` 与 `source` 的 payload filter——这是安全底线，不是性能优化：不加过滤，一次检索就可能把别的视频的证据串进来。

**证据融合**：ASR 与 OCR 各带 `(start_ms, end_ms, text, source)`，按时间排序合并。两条流任意一条为空系统仍可用；时间戳漂移靠邻接扩展兜底。

**OCR 隔离子进程**：Paddle 的 oneDNN 线程模型会和主进程冲突并推高内存峰值，所以 OCR 放在独立子进程里跑、跑完回收；ASR 完成后先释放 Whisper 模型再启动 OCR，峰值内存因此能压到 4C8G 单机可承受的范围。

**MCP Server**（`backend/vidlens/mcp_server.py`）：14 个工具 + 4 个资源模板，透传用户 JWT，所以 MCP 客户端只能访问该用户自己的视频。提供 Streamable HTTP 与 stdio 两种传输。

**可替换的 AI Provider**：未配置真实模型密钥时自动退回确定性 Mock Provider，保证整条业务链路（上传 → 队列 → Worker → 报告）在无网络时也能跑通和测试。

---

## 五、技术栈

| 层 | 选型 | 理由 |
|---|---|---|
| API | FastAPI | 原生 async，适合大量 IO 等待（ASR 调用、向量检索） |
| 存储 | MySQL + Alembic | 结构化数据需事务；迁移需版本化 |
| 队列 | RocketMQ | 长耗时任务异步化，消费失败可重试 |
| 锁 | Redis | 防重复提交，瞬时互斥 |
| 向量 | Qdrant | 跨进程共享、payload 过滤、快照复用 |
| ASR | faster-whisper (int8) | 本地推理，CPU 可跑，成本可控 |
| OCR | PaddleOCR | 中文场景识别效果好，可 CPU 推理 |
| Agent | LangGraph + 自研五段式 | 前者管动态路由，后者管可控可审计 |
| 前端 | Vue 3 | — |

---

## 六、本地运行

需要 Docker Desktop 与 Node.js 22+。

```bash
cp backend/.env.example backend/.env      # 填入 AI_API_KEY；留空则走 Mock
docker compose up --build                 # API :9090，Worker、MySQL、Redis、RocketMQ、Qdrant

cd client && npm install && npm run dev   # 前端 :5173
```

首次启动会下载 ASR / OCR / BGE 模型（约 185MB）到独立 volume，后续复用。接口文档见 `http://localhost:9090/docs`。

后端测试：

```bash
cd backend && pytest -q
```

---

## 七、已知不足

- Coverage-Aware 检索在留出集上未泛化（见第三节），当前默认策略仍需在更多真实视频上验证
- 拒答泛化是主要短板：不可回答题在检索层几乎不拒答，拒答实际只发生在 Verifier 层
- 长视频（>1h）的处理耗时尚不理想，Worker 为串行处理
- 评测集规模偏小（留出集 2 条视频、16 题），统计意义有限，结论需谨慎外推
