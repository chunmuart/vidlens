# VidLens 后端

FastAPI 提供用户认证、视频分片上传、任务状态、报告和追问接口。RocketMQ Worker 执行识别、证据检索和 Agent 分析；数据库由 SQLAlchemy 与 Alembic 管理。

完整项目建议从根目录的 `start-vidlens-docker.cmd` 启动。Docker 使用 MySQL，以下单独开发方式默认使用 SQLite，两者的数据互不自动迁移。

## 单独开发

推荐 Python 3.11。以下命令在 `backend` 目录执行：

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
# 保留已有配置
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
.venv/Scripts/python.exe -m alembic upgrade head
.venv/Scripts/python.exe -m uvicorn vidlens.main:app --reload --port 9090
```

单独运行不启动 MySQL、Redis、RocketMQ 或 Qdrant。RocketMQ 客户端需要 Linux 动态库，完整队列链路使用 Docker；镜像通过 `requirements-rocketmq.txt` 安装相应依赖。前端独立运行时，后端的 `CORS_ALLOWED_ORIGINS` 需允许 `http://localhost:5174`。

模型密钥和运行配置放在本目录 `.env`，从本目录启动以便加载。云端 ASR／OCR、Agent 预算、checkpoint 和评测操作见 [配置指南](../docs/Agent优化与API配置.md)。未配置 Agent 密钥时使用 Mock Provider。

## 代码入口

| 文件 | 职责 |
|---|---|
| `vidlens/main.py` | API、上传、任务编排、识别接入和追问 |
| `vidlens/worker.py` | RocketMQ 消费和任务执行 |
| `vidlens/agent_structured.py` | 五阶段 Agent、证据账本、一次有限修复 |
| `vidlens/retrieval.py` | 关键词、向量与混合检索 |
| `vidlens/followup.py` | 追问改写、证据回答和语义复核 |
| `vidlens/evidence_api.py` | 云端 ASR／OCR 协议与鉴权 |
| `vidlens/evidence_pipeline.py` | 独立分路缓存、恢复和并行 |
| `vidlens/checkpoint.py` | Planner 与初始 Retriever 快照 |
| `vidlens/budget.py` | 累计调用次数、Token、费用与时间预算 |
| `vidlens/mcp_server.py` | MCP 工具及资源入口 |

`agent.py`、`agent_graph.py` 保留旧版动态工具调用链路，用于显式回滚与对比，不能仅因不是默认链路就删除。

## 测试与评测

```powershell
$env:VIDLENS_TEST_ROOT = Join-Path (Resolve-Path ..) '.test-runtime/backend'
.venv/Scripts/python.exe -m pytest -q
```

`tests/` 检查程序行为；`evals/` 保存题集；`scripts/` 保存可重复执行的评测工具。真实 API 评测需要登录令牌、视频 ID 映射和人工核对后的题集，会产生模型调用。不要用候选标注直接宣称准确率。

## MCP

先启动 API 并登录获得用户令牌。在本目录执行：

```powershell
$env:VIDLENS_API_URL = 'http://127.0.0.1:9090'
$env:VIDLENS_MCP_TOKEN = '填写本机登录令牌'
.venv/Scripts/python.exe -m vidlens.mcp_server --transport stdio
```

HTTP 服务使用 `--transport streamable-http --host 0.0.0.0 --port 8001`。客户端通过 `Authorization: Bearer <token>` 传入网站登录令牌。示例流程见根目录 `skills/`；它们是产品使用流程，不是项目学习讲义。

服务器部署、数据库备份和生产参数见 [部署说明](../deploy/README.md)。
