# Agent优化与API配置

更新：2026-11-15。本机已保留原有DeepSeek密钥，接入百炼密钥，并通过两家接口的鉴权与模型列表检查；尚未执行真实视频的收费识别。

## 1. 已接入的功能

| 项目 | 当前实现 | 范围与验证重点 |
|---|---|---|
| 可信评测 | 无评分规则标记`NOT_SCORED`，空答案不通过；可回答／应拒答分别统计 | 20题候选基线仍待人工回查，不能直接当作已验证成绩 |
| 证据分配 | 先收集每个需求的候选，轮流选直接命中，去重后再补邻接片段 | 账本仍最多18条；分到一条不等于证据充分 |
| 有限修复 | 缺证据定向补检索；与Critic修订共享一次修复额度 | 每次图执行最多一次；任务重试还受累计预算限制 |
| 可信追问 | 必要时结合历史补全检索问句；当前媒体内检索，答案引用ID并复核支持度 | 指代不清则澄清，复核未通过则拒答；模型复核仍有误判可能 |
| 两阶段恢复 | checkpoint v2，重做Planner清旧Retriever，版本与证据变化使相应缓存失效 | 保存Planner与初始Retriever；Verifier、Writer、Critic重新执行 |
| 调用预算 | 模型调用轮次、已返回Token、已知费用与累计执行时间；同任务重试不重置 | Token／费用通常在响应后得知；远端未知收费不当作零 |
| ASR／OCR API | Fun-ASR异步录音识别；兼容接口的OCR图片识别 | 需要真实时间戳；不把纯文本伪装成带时间证据 |
| 分路缓存与并行 | ASR／OCR各自完成后立即保存；失败后复用成功分路；OCR已成功帧也保存 | 云端两路并行，OCR路内图片顺序调用；纯本地两路串行以控制内存 |

代码入口：`backend/vidlens/agent_structured.py`、`followup.py`、`checkpoint.py`、`budget.py`、`evidence_api.py`、`evidence_pipeline.py`、`main.py`。

## 2. 你需要申请什么API

| 用途 | 推荐平台 | 配置模型名 | 用哪个Key |
|---|---|---|---|
| Agent规划、写报告、复核、追问 | DeepSeek官方开放平台 | `deepseek-flash` | `AI_API_KEY`；已有可用Key可以继续用 |
| 中文录音转写ASR | 阿里云百炼，优先北京地域 | `fun-asr` | `DASHSCOPE_API_KEY` |
| 视频抽帧OCR | 同一百炼业务空间 | `qwen3.5-ocr` | 同一个`DASHSCOPE_API_KEY` |

先准备这两家即可。百炼同一业务空间的Key可用于上述ASR与OCR；还需确保账户已开通对应模型调用。这里是适配性推荐，尚未对你的视频实测准确率或速度。

截至2026-11-15，DeepSeek官方推荐使用`deepseek-flash`，当前对应`DeepSeek-V4.1-Flash-1017`；旧的`deepseek-v4-flash`名称仍有兼容映射，原模型已退役。新模板使用当前名称。

申请入口：

- [DeepSeek开放平台](https://platform.deepseek.com/)：创建API Key，填入`AI_API_KEY`。
- [阿里云百炼控制台](https://bailian.console.aliyun.com/)：选择北京地域和业务空间，创建API Key，填入`DASHSCOPE_API_KEY`。

Key、地域、业务空间与API地址需要匹配。老业务空间常用地址如下；新业务空间如果给出带WorkspaceId的专属域名，以控制台该空间的API调用示例为准，替换这两个可配置地址。

```dotenv
ASR_BASE_URL=https://dashscope.aliyuncs.com
OCR_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
```

ASR使用百炼原生异步接口，不给它加`/compatible-mode/v1`。OCR使用兼容接口，需要该后缀。这里的`OCR_PROVIDER=dashscope`与`openai`都选择兼容格式适配器，前者便于辨认当前平台。

Fun-ASR按输入音频时长计费；该OCR按输入／输出Token计费。LLM费用另算，不把它们合成一个Token数。具体金额查平台当前价格页；本项目暂不自动汇总跨币种账单。

## 3. 配置到哪里

打开项目里的`deploy/.env.api.example`，把其中同名配置更新到**现有**`backend/.env`。保留当前数据库、Redis、RocketMQ、Qdrant及登录配置。该片段本身不会自动生效，也不应覆盖整份`.env`。

最主要的两项填Key：

```dotenv
AI_API_KEY=填写DeepSeek的Key
DASHSCOPE_API_KEY=填写百炼的Key
```

配套片段已设置`ASR_PROVIDER=dashscope`、`ASR_MODEL=fun-asr`、`OCR_PROVIDER=dashscope`、`OCR_MODEL=qwen3.5-ocr`、并行与checkpoint开关。单独的`ASR_API_KEY`／`OCR_API_KEY`为空时会使用`DASHSCOPE_API_KEY`。本机`backend/.env`现已合并上述识别参数，保留原有DeepSeek密钥和数据库等配置；仓库中的模板不含真实密钥。

首次试跑将画面上限设为12帧，采样间隔至少15秒，并随时长放宽以覆盖全片；这用于控制试跑调用量，快速闪过的文字可能漏掉。确认链路后可将`PADDLEOCR_MAX_FRAMES`改为30或60再比较召回与费用。

默认不在远端失败后自动启动本地ASR；需要时可显式开启本地降级。选择API后关闭本地模型预加载，可避免启动时仍占用这部分内存。

## 4. 如何让代码和配置生效

你现在的Docker启动脚本会读取`backend/.env`。修改后从项目根目录执行：

```powershell
Set-Location C:\code\vidlens
.\start-vidlens-docker.cmd
```

脚本重新构建并启动服务，网页入口为`http://localhost:5174`。API与Worker现在共用识别配置，避免后台分析走API而追问服务仍走另一套本地配置。配置文件写入后，须等待容器构建和健康检查通过，才算运行生效。

如果直接在Windows启动Python，需要从`backend`目录启动，让现有的`.env`加载逻辑找到配置；修改后重启API进程，另有Worker时也需重启。不要把Docker中的服务名地址直接用于Windows本地进程。

## 5. 第一次用什么视频检查

先用1～3分钟、语音清楚且有几页文字画面的视频。检查以下现象：

1. ASR和OCR都完成，证据带真实时间和来源；无声、无文字与接口失败应区别记录。
2. 问一个语音里有答案的问题，再问一个必须看画面的问题，最后问视频未讲的问题。
3. 先问两个方案的差异，再问“第二个的限制呢”，核对检索问句和引用是否仍对应当前视频。
4. 用测试故障验证分路复用；不要用反复提交真实收费任务代替自动化故障测试。

查看任务状态接口返回的`trace.evidenceExtraction`、`providerRequests`、`graph`、`checkpoint`、`budget`。追问的最近一次记录可在已有`/analysis/agent-memory?id=媒体ID`返回的会话中查看`latestFollowUpTrace`，需要该视频所属用户的登录状态。

分路产物在上传目录的`evidence-cache`，追问记录在`followup-traces`，Docker中都在现有上传数据卷里。模型／采样等输出配置变化会使对应分路缓存失效；新产物先落盘、数据库未提交时中断，后续也会比对内容再决定复用。

原生ASR提交成功后立即保存远端任务ID，暂时超时可继续轮询。任务有平台有效期；提交请求已被远端接受但本地没收到ID的窗口，仍可能重复提交。当前采用平台临时上传机制便于本地试跑，生产长期运行可换成自有OSS；这次不引入额外存储服务。

## 6. 小基线怎样评测

`backend/evals/agent_baseline_v1.json`有20题候选：15可答、5应拒答，全部标记`PENDING_VIDEO_REVIEW`。先回查视频，确认答案规则、必要来源、时间范围，并把人工核对结果另存为`agent_baseline_reviewed.json`。其中包含以前调试过的视频，属于开发／回归集；不能再当作全新留出集。

从项目`backend`目录生成一份候选文件（会覆盖指定输出，别指向已人工标注的文件）：

```powershell
& '.venv\Scripts\python.exe' scripts/prepare_agent_baseline.py --output '.test-runtime/candidate-baseline.json'
```

运行真实评测需要服务已启动、对应视频已导入，以及本机的登录Token。`media-map.json`的形式是`{"video-001":11,"video-002":12}`，键必须与标注集的videoId对应；先只运行1题，确认后再扩大：

```powershell
$env:VIDLENS_API_URL = 'http://localhost:5174/api'
# VIDLENS_EVAL_TOKEN填本机登录Token，再执行：
& '.venv\Scripts\python.exe' scripts/evaluate_real_video.py --dataset evals/agent_baseline_reviewed.json --media-map .test-runtime/media-map.json --output .test-runtime/after.json --max-cases 1
```

已有同题、同证据、同评分版本的优化前后报告时，离线比较不再请求模型：

```powershell
& '.venv\Scripts\python.exe' scripts/compare_agent_runs.py --before .test-runtime/before.json --after .test-runtime/after.json --output .test-runtime/comparison.json
```

对照脚本会检查标注、评分版本与证据身份，条件不一致时不会宣称Agent提升。要隔离Agent效果，前后固定同一份证据；要测ASR／OCR质量，则单独用人工转写／画面文字比较。没有规则的题从答案得分分母排除；`NOT_SCORED`不是零分，也不是满分。

预算片段的12次请求、60000 Token、180秒只是初次试跑上限，后续按真实样本调整。时间预算针对Agent调用，不含前面的媒体识别；单次模型请求的超时会受剩余时间约束。价格未填时保持费用上限为0，不能据此声称已有精确金额控制。价格配置按每百万Token的美元值填写，平台若报人民币需先统一币种。

## 7. 验证与面试口径

自动化测试覆盖了证据分配、一次修复、评分、checkpoint一致性、追问门禁、预算，以及API请求格式、鉴权隔离、真实并行、独立缓存和远端任务恢复。测试使用模拟响应，不产生远端收费。

本次最终验证：后端完整测试 **326 passed、2 skipped**；跳过的是未配置`QDRANT_TEST_URL`的真实服务集成测试。开发与生产Compose配置解析通过。额外用实际ffmpeg生成1.2秒视频，确认抽帧起点为0和1000ms、末段结束1200ms。依赖有4条弃用／配置提示，未导致测试失败。前端业务逻辑未改动；目录整理仅更新说明和开发端口注释。

可以说：“我给Agent补了证据分配、一次有限修复和带证据复核的追问，修复已有两阶段checkpoint，并接入云端ASR／OCR和独立缓存；用自动化回归检查边界。”

暂时不能说准确率、耗时或费用提升了多少；20题尚待人工核对，也尚未跑真实API对照。全流程断点恢复、跨进程恰好一次付费调用、完整分布式任务治理不在这次实现范围。

官方参考（2026-11-15核对）：

- [DeepSeek模型与命名](https://api-docs.deepseek.com/)
- [DeepSeek更新日志](https://api-docs.deepseek.com/updates)
- [Fun-ASR录音识别HTTP接口](https://help.aliyun.com/zh/model-studio/fun-asr-recorded-speech-recognition-http-api)
- [Qwen OCR接口](https://help.aliyun.com/zh/model-studio/qwen-vl-ocr-api-reference)
- [百炼模型价格](https://help.aliyun.com/zh/model-studio/model-pricing)


## 8. 本机接入与启动验收（2026-11-15）

本机保留原有DeepSeek密钥和API地址，根据实际模型列表将`AI_MODEL`设为`deepseek-flash`；百炼密钥已写入私有`backend/.env`。两家模型列表接口均返回HTTP 200，OCR模型在可用列表中。以上不代表真实识别调用或账户额度已验收。

Docker Desktop缺少当前用户安装路径记录，已恢复该记录并启动引擎。首次构建已完成，网页、API、Worker、MySQL、Redis、RocketMQ nameserver／broker、Qdrant、MCP共9个服务运行。网页首页与`/api/health`均返回HTTP 200；API和Worker的模型、密钥及识别开关与私有配置一致，检查过程未输出密钥。

已将现有BGE模型缓存复用到Docker模型卷，实际生成一条512维有限数值向量。容器内Qdrant、Redis、MySQL连通检查通过；Worker日志包含`worker_started`且无启动异常栈。这些是启动与连通验收，尚未证明完整视频任务成功。

Windows已有Memurai占用6379，因此`docker-compose.local.yml`将本项目Redis映射到`127.0.0.1:6380`。容器间仍用`redis:6379`；原有Memurai继续运行。本机覆盖文件使用Compose的`!override`合并端口，已在当前Compose 5.7.0上校验。

下一步从`http://localhost:5174`上传1～3分钟视频，验收语音题、画面文字题、无答案题和两轮追问。当前未调用收费ASR／OCR或Agent生成接口，没有新旧效果提升数据。

## 9. 本轮最终收口：核验与反证审查（2026-11-15）

Agent提示版本更新为`video-evidence-agent-v5.4-full-ledger-audit`。本轮只收紧已有核验与审查，不增加节点、请求次数或修复额度：

1. 删除根据解释中“证据直接支持”等关键词自动放行的逻辑。即使解释中包含合法证据ID，也不能把`supported=false`改成通过；只有实际布尔值`true`能通过对应结构检查，字符串和数字不作肯定判定。结果不通过时继续走已有的一次补检索／拒答流程。
2. Critic保留实际引用子集，同时接收本轮完整Evidence Ledger，检查未引用候选中的反证和适用条件。关键结论仍必须有实际引用支持，不能靠未引用材料替答案补齐依据。
3. 追问复核同样接收实际引用与本轮完整候选。发现遗漏必要限制或与候选矛盾时拒绝当前答案；引用存在不等于语义支持。

这里的“完整”仅指当前已入选的账本／追问候选，不代表整段视频的所有证据。18条账本上限、单条360字符截断和检索来源筛选仍保留；是否调整应由真实失败案例和基线决定。未增加部分回答状态、更多修复循环或全阶段checkpoint。

回归覆盖否定句误触发、解释与结构判定冲突、错误布尔类型，以及未引用的限制条件／反证进入审查并阻止报告通过。模型响应由本地测试模拟；这些测试验证数据传递和门禁，不能证明真实模型一定识别所有反证。提示版本变化会使旧版本Agent checkpoint失效。

本轮最终验证为后端 **339 passed、2 skipped、4 warnings**；跳过项仍是需要`QDRANT_TEST_URL`的服务集成测试。前端生产构建、本地与生产Compose配置校验通过。第7节的326项通过为上一轮历史记录；以本节为本轮最终结果。20题人工复核和真实视频/API效果对照仍未完成，不将当前版本称为经过效果验收的生产版本。

本机容器已重新构建并启动，API与Worker均确认加载上述v5.4提示版本，网页代理的`/api/health`返回HTTP 200。本轮没有调用收费模型进行效果评测。
