# VidLens

视频证据分析项目：上传视频、提出问题，检索带时间戳的语音和画面文字，生成可回看的报告，并支持继续追问。

## Windows 启动

安装 Docker Desktop。首次使用时将 `backend/.env.example` 复制为 `backend/.env`，填入模型配置；已有 `.env` 时直接保留。云端识别参数见 [Agent 与 API 配置](docs/Agent优化与API配置.md)。

在项目根目录双击 `start-vidlens-docker.cmd`，或执行：

```powershell
.\start-vidlens-docker.cmd
```

脚本读取 `backend/.env`，构建并启动网页、API、Worker、MySQL、Redis、RocketMQ、Qdrant 和 MCP 服务。

- 网页：`http://localhost:5174`
- API 文档：`http://localhost:9090/docs`
- 停止：双击 `stop-vidlens-docker.cmd`，保留数据库、上传文件和模型缓存。

单条视频默认最多 15 分钟。第一次使用先选 1～3 分钟视频验证语音、画面文字和报告引用。

## 分析流程

```text
视频 → FFmpeg ┬→ ASR 语音转写 ────┐
              └→ 抽帧与 OCR 识别 ─┴→ 时间轴证据
                                        ↓
需求拆解 → 混合检索 → 证据核验 → 报告生成 → Critic 审查
```

云端 ASR 与 OCR 分路并行，各自成功后保存；OCR 分路内按帧顺序调用。纯本地识别保持串行以控制内存。

- **上传与任务**：分片上传、断点补传、内容去重；任务入库后交给 RocketMQ Worker，前端轮询进度。
- **证据检索**：IDF 关键词与云端 `text-embedding-v4`（1024维）向量检索，通过 RRF 融合，按问题需求分配证据，再扩展时间邻接片段；可选本地 BGE 模型。
- **Agent**：Planner → Retriever → Verifier → Writer → Critic；补检索与报告修订共享一次修复额度。
- **继续追问**：必要时结合历史补全问题，在当前视频内检索，校验引用与语义支持；指代不清时要求澄清。
- **恢复与预算**：复用成功识别分路、OCR 帧和 Planner／初始 Retriever；同一任务重试累计调用预算。
- **权限与 MCP**：JWT 用户隔离；MCP 通过用户令牌调用 API，提供视频检索、报告和引用校验工具。

## 项目结构

```text
backend/
  vidlens/       API、Worker、Agent、检索与识别实现
  alembic/       数据库迁移
  tests/         自动化测试
  scripts/       评测、候选集生成与结果比较
  evals/         评测标注与候选题集
  vendor/        RocketMQ 构建依赖说明
  .env.example   完整开发配置模板
client/          Vue 3 网页与 Nginx 配置
scripts/         Windows Docker 启停脚本
skills/          基于 MCP 的课程笔记、会议复盘等使用流程
deploy/         生产部署、备份与 API 配置片段
docs/           当前 Agent、识别 API 与评测操作说明
.github/         自动化检查
```

根目录的两个 `.cmd` 是双击入口，调用 `scripts/` 中对应脚本。开发配置统一使用 `backend/.env`；服务器使用 `deploy/.env.production`。本机环境、测试输出、模型缓存和上传数据不提交 Git。

## 开发与验证

后端独立开发和 MCP 运行见 [后端说明](backend/README.md)，网页独立开发见 [前端说明](client/README.md)。服务器部署见 [部署说明](deploy/README.md)。

```powershell
# 在项目根目录，用独立测试目录避免污染本地数据
$env:VIDLENS_TEST_ROOT = Join-Path $PWD '.test-runtime/local'
& backend/.venv/Scripts/python.exe -m pytest -q backend/tests

# 前端构建
npm --prefix client run build

# 检查本地配置，不打印配置内容
docker compose --env-file backend/.env -f docker-compose.yml -f docker-compose.local.yml config --quiet
```

2026-11-16 后端回归记录为 **378 passed、2 skipped**；跳过项需要真实 Qdrant 服务。核验不再根据解释关键词放行；主流程与追问审查均检查本轮未引用的候选反证。测试包含模拟 API 响应，不能据此声称真实识别准确率或速度提高。

## 评测与边界

当前成果见 [100视频评测结果](backend/evals/agent_video100_v2/RESULTS.md)、[面试讲解](backend/evals/agent_video100_v2/INTERVIEW.md) 和 [评测资料入口](backend/evals/README.md)。保存100条视频的文本证据快照、837道机器银标题、60题阶段对照、180组验证输入和12题答案对照；人工视频核验仍为0，作者分组划分为暂定版本，尚未人工排查语义重复。

同一批证据上对比 IDF、BM25、纯向量和两种 RRF 混合检索。BM25＋向量在837题上的 Recall@3 为67.46%，BM25为64.01%；这是同视频检索的离线实验，生产默认仍使用 IDF＋向量。自然缺证据的9题中，定向补检索补全6题。12题答案实验样本很小，不能据此声称总体准确率。

克隆仓库后可直接查看标注、快照和逐题预测；重新调用模型需要自己的配置并产生费用。`--report-only` 和完整缓存回放需要原运行目录 `.runtime/evals/`，仓库不含原视频、密钥或付费响应缓存。旧20题基线和历史检索结果保留用于回归，不代表本轮成绩。

checkpoint 保存 Planner 和初始 Retriever，后续阶段重新执行；识别分路缓存不保证跨进程恰好一次付费调用。OCR 只识别抽到画面的文字，可能漏掉短暂显示的内容。当前没有生产吞吐量或完整的新旧版本真实 API 对照结论。
