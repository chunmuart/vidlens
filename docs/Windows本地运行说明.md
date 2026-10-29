# VidLens Windows 本地运行说明

## 当前运行方式

本机没有安装 Docker Desktop，因此当前开发环境没有启动 MySQL、RocketMQ 和 Qdrant，而是采用项目已支持的本地运行组合：

| 能力 | 当前实现 |
| --- | --- |
| 数据库 | SQLite，文件位于 `backend/data/vidlens.db` |
| 异步队列 | FastAPI 进程内线程池 |
| Redis | 复用本机 `127.0.0.1:6379` |
| 检索 | `coverage-aware-lexical-v3`，不依赖 Qdrant |
| AI Provider | 按 `VIDLENS_ENV_FILE` → `C:\code\shoppilot\.env` → `backend\.env` 的顺序选择完整的 OpenAI 兼容配置 |
| ASR | 本地 `faster-whisper base/int8` |
| OCR | 本地 PaddleOCR `PP-OCRv5_mobile` |
| 视频处理 | 项目运行时目录中的 FFmpeg 7.1 和 FFprobe |

API 密钥只从本机 env 文件读入启动进程，没有复制到 VidLens 仓库。脚本会优先使用 `VIDLENS_ENV_FILE`，否则依次尝试 `C:\code\shoppilot\.env` 和 `backend\.env`，选择第一个同时包含 `LLM_BASE_URL`、`LLM_API_KEY`、`LLM_MODEL`（或 `AI_*` 同名变量）的文件。

## 启动

在项目根目录双击 `start-vidlens.cmd`，脚本会启动前后端并自动打开 `http://127.0.0.1:5174`。

也可以在终端执行：

```powershell
.\start-vidlens.cmd
```

重启 VidLens 前后端：

```powershell
.\start-vidlens.cmd -Restart
```

只启动、不自动打开浏览器：

```powershell
.\start-vidlens.cmd -NoBrowser
```

访问地址：

- 前端：`http://127.0.0.1:5174`
- API：`http://127.0.0.1:9090`
- 接口文档：`http://127.0.0.1:9090/docs`
- 健康检查：`http://127.0.0.1:9090/health`

运行日志和进程编号保存在 `.runtime/`：

```text
.runtime/logs/backend.err.log
.runtime/logs/frontend.out.log
.runtime/pids.json
```

## 模型缓存

首次处理真实视频时会下载并缓存模型，后续复用：

```text
backend/data/models/faster-whisper-base/
backend/data/models/paddlex/
```

Whisper 模型从 Hugging Face 镜像下载；PaddleOCR 模型使用项目默认的 BOS 模型源。

## 重建依赖

后端：

```powershell
cd C:\code\vidlens\backend
uv venv --python 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -r requirements-windows-local.txt
```

前端：

```powershell
cd C:\code\vidlens\client
npm.cmd ci
```

FFprobe 运行文件：

```powershell
cd C:\code\vidlens
npm.cmd install --prefix .runtime\ffmpeg @ffprobe-installer\win32-x64
```

FFmpeg 由 `imageio-ffmpeg` 提供，启动脚本会把它复制到 `.runtime/tools/ffmpeg.exe` 并加入子进程的 `PATH`。

## 验证结果

2026-10-27 已完成一次不使用旁车证据的真实视频验收：

```text
本地视频上传 -> Whisper ASR -> PaddleOCR -> 时间轴证据
-> Coverage-Aware 检索 -> DeepSeek 报告 -> 引用支持性校验
```

结果为 `COMPLETED`，生成 2 段证据、2 条引用，引用支持数为 2。
