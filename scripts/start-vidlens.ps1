# VidLens Windows 本地一键启动脚本
param(
    [switch]$Restart,
    [switch]$NoBrowser,
    [switch]$WithVector
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$BackendRoot = Join-Path $RepoRoot "backend"
$ClientRoot = Join-Path $RepoRoot "client"
$BackendPython = Join-Path $BackendRoot ".venv\Scripts\python.exe"
$RuntimeRoot = Join-Path $RepoRoot ".runtime"
$LogRoot = Join-Path $RuntimeRoot "logs"
$PidPath = Join-Path $RuntimeRoot "pids.json"
$ToolDir = Join-Path $RuntimeRoot "tools"
$FfmpegExe = Join-Path $ToolDir "ffmpeg.exe"
$FfprobeDir = Join-Path $RuntimeRoot "ffmpeg\node_modules\@ffprobe-installer\win32-x64"
$BackendPort = 9090
$FrontendPort = 5174

function Read-DotEnv {
    param([Parameter(Mandatory = $true)][string]$Path)

    $values = @{}
    foreach ($line in Get-Content -LiteralPath $Path -Encoding UTF8) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $separator = $trimmed.IndexOf("=")
        if ($separator -lt 1) {
            continue
        }
        $name = $trimmed.Substring(0, $separator).Trim()
        $value = $trimmed.Substring($separator + 1).Trim()
        $values[$name] = $value
    }
    return $values
}

function Resolve-AiConfig {
    param([string[]]$Candidates)

    foreach ($path in $Candidates) {
        if (-not $path -or -not (Test-Path -LiteralPath $path)) {
            continue
        }
        $values = Read-DotEnv -Path $path
        $baseUrl = $values["LLM_BASE_URL"]
        if (-not $baseUrl) { $baseUrl = $values["AI_BASE_URL"] }
        $apiKey = $values["LLM_API_KEY"]
        if (-not $apiKey) { $apiKey = $values["AI_API_KEY"] }
        $model = $values["LLM_MODEL"]
        if (-not $model) { $model = $values["AI_MODEL"] }
        if ($baseUrl -and $apiKey -and $model) {
            return [pscustomobject]@{
                Path = $path
                BaseUrl = $baseUrl
                ApiKey = $apiKey
                Model = $model
            }
        }
    }
    return $null
}

function Get-ListeningProcessId {
    param([Parameter(Mandatory = $true)][int]$Port)

    $connection = Get-NetTCPConnection -State Listen -LocalPort $Port -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($connection) {
        return [int]$connection.OwningProcess
    }
    return $null
}

function Stop-TrackedProcess {
    param(
        [Parameter(Mandatory = $true)][string]$Name,
        [Parameter(Mandatory = $true)][int]$Port
    )

    $processId = Get-ListeningProcessId -Port $Port
    if (-not $processId) {
        return
    }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) {
        return
    }
    Stop-Process -Id $processId
    for ($attempt = 0; $attempt -lt 20; $attempt++) {
        if (-not (Get-Process -Id $processId -ErrorAction SilentlyContinue)) {
            return
        }
        Start-Sleep -Milliseconds 250
    }
    throw "Could not stop old $Name process (PID $processId)"
}

if (-not (Test-Path -LiteralPath $BackendPython)) {
    throw "Backend virtual environment not found. Run: uv venv --python 3.11 backend\.venv"
}
if (-not (Test-Path -LiteralPath (Join-Path $ClientRoot "node_modules"))) {
    throw "Frontend dependencies are missing. Run: npm.cmd ci"
}
if (-not (Test-Path -LiteralPath (Join-Path $FfprobeDir "ffprobe.exe"))) {
    throw "Bundled FFprobe was not found in $FfprobeDir"
}

$FfmpegSource = & $BackendPython -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"
if (-not $FfmpegSource -or -not (Test-Path -LiteralPath $FfmpegSource)) {
    throw "Modern FFmpeg was not found. Run: backend\.venv\Scripts\python.exe -m pip install imageio-ffmpeg"
}

$envCandidates = @()
if ($env:VIDLENS_ENV_FILE) { $envCandidates += $env:VIDLENS_ENV_FILE }
$envCandidates += "C:\code\shoppilot\.env"
$envCandidates += (Join-Path $BackendRoot ".env")
$aiConfig = Resolve-AiConfig -Candidates $envCandidates
if (-not $aiConfig) {
    throw "未找到完整的 AI 配置。请设置 VIDLENS_ENV_FILE，或在 backend\.env / C:\code\shoppilot\.env 中提供 LLM_BASE_URL、LLM_API_KEY、LLM_MODEL（或 AI_* 同名变量）。"
}

if ($Restart) {
    Stop-TrackedProcess -Name "VidLens backend" -Port $BackendPort
    Stop-TrackedProcess -Name "VidLens frontend" -Port $FrontendPort
}

New-Item -ItemType Directory -Force -Path $RuntimeRoot, $LogRoot, $ToolDir | Out-Null
if (-not (Test-Path -LiteralPath $FfmpegExe) -or
    (Get-Item -LiteralPath $FfmpegExe).Length -ne (Get-Item -LiteralPath $FfmpegSource).Length) {
    Copy-Item -LiteralPath $FfmpegSource -Destination $FfmpegExe -Force
}
$env:PATH = "$ToolDir;$FfprobeDir;$env:PATH"

$env:DATABASE_URL = "sqlite:///./data/vidlens.db"
$env:APP_ENV = "development"
$env:AUTO_CREATE_SCHEMA = "true"
$env:REDIS_URL = "redis://127.0.0.1:6379/0"
$env:ROCKETMQ_NAMESERVER = ""
$env:CORS_ALLOWED_ORIGINS = "http://localhost:$FrontendPort,http://127.0.0.1:$FrontendPort"
$qdrantPid = $null
if ($WithVector) {
    $QdrantDir = Join-Path $RuntimeRoot "qdrant"
    $QdrantExe = Join-Path $QdrantDir "qdrant.exe"
    $EmbeddingDir = Join-Path $RuntimeRoot "models\bge-small-zh"
    if (-not (Test-Path -LiteralPath $QdrantExe)) {
        throw "未找到 Qdrant：$QdrantExe。请把 qdrant-x86_64-pc-windows-msvc.zip 解压到该目录，或去掉 -WithVector。"
    }
    if (-not (Test-Path -LiteralPath (Join-Path $EmbeddingDir "onnx\model_quantized.onnx"))) {
        throw "未找到 BGE 模型：$EmbeddingDir\onnx\model_quantized.onnx"
    }
    $qdrantPid = Get-ListeningProcessId -Port 6333
    if (-not $qdrantPid) {
        $qdrantProcess = Start-Process `
            -FilePath $QdrantExe `
            -WorkingDirectory $QdrantDir `
            -WindowStyle Hidden `
            -RedirectStandardOutput (Join-Path $LogRoot "qdrant.out.log") `
            -RedirectStandardError (Join-Path $LogRoot "qdrant.err.log") `
            -PassThru
        $qdrantPid = $qdrantProcess.Id
    }
    $qdrantReady = $false
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        try {
            $qdrantHealth = Invoke-RestMethod -Uri "http://127.0.0.1:6333/collections" -TimeoutSec 2
            if ($qdrantHealth.status -eq "ok") {
                $qdrantReady = $true
                break
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    }
    if (-not $qdrantReady) {
        throw "Qdrant 未在 30 秒内就绪。查看 $LogRoot\qdrant.err.log"
    }
    $env:QDRANT_URL = "http://127.0.0.1:6333"
    $env:EVIDENCE_RETRIEVER_PROFILE = "coverage-aware-qdrant-hybrid-v2"
    $env:EVIDENCE_EMBEDDING_MODEL_DIR = $EmbeddingDir
    $env:EVIDENCE_EMBEDDING_ALLOW_DOWNLOAD = "false"
} else {
    $env:EVIDENCE_RETRIEVER_PROFILE = "coverage-aware-lexical-v3"
    $env:EVIDENCE_EMBEDDING_ALLOW_DOWNLOAD = "false"
}
$env:LOCAL_ASR_ENABLED = "true"
$env:OCR_ENABLED = "true"
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HUB_DISABLE_TELEMETRY = "1"
$env:HF_HUB_DISABLE_XET = "1"
$env:LOCAL_ASR_MODEL = "./data/models/faster-whisper-base"
$env:AI_BASE_URL = $aiConfig.BaseUrl
$env:AI_API_KEY = $aiConfig.ApiKey
$env:AI_MODEL = $aiConfig.Model
$env:AI_THINKING_MODE = "disabled"
$env:VITE_API_BASE_URL = "http://127.0.0.1:$BackendPort"

$backendPid = Get-ListeningProcessId -Port $BackendPort
if (-not $backendPid) {
    $backendProcess = Start-Process `
        -FilePath $BackendPython `
        -ArgumentList @("-m", "uvicorn", "vidlens.main:app", "--host", "127.0.0.1", "--port", "$BackendPort") `
        -WorkingDirectory $BackendRoot `
        -RedirectStandardOutput (Join-Path $LogRoot "backend.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "backend.err.log") `
        -WindowStyle Hidden `
        -PassThru
    $backendPid = $backendProcess.Id
}

$healthy = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $health = Invoke-RestMethod -Uri "http://127.0.0.1:$BackendPort/health" -TimeoutSec 2
        if ($health.status -eq "ok") {
            $healthy = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $healthy) {
    throw "VidLens backend was not ready within 60 seconds. See $LogRoot\backend.err.log"
}
$backendPid = Get-ListeningProcessId -Port $BackendPort

$frontendPid = Get-ListeningProcessId -Port $FrontendPort
if (-not $frontendPid) {
    $frontendProcess = Start-Process `
        -FilePath "npm.cmd" `
        -ArgumentList @("run", "dev", "--", "--host", "127.0.0.1", "--port", "$FrontendPort") `
        -WorkingDirectory $ClientRoot `
        -RedirectStandardOutput (Join-Path $LogRoot "frontend.out.log") `
        -RedirectStandardError (Join-Path $LogRoot "frontend.err.log") `
        -WindowStyle Hidden `
        -PassThru
    $frontendPid = $frontendProcess.Id
}

$frontendReady = $false
for ($attempt = 0; $attempt -lt 30; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$FrontendPort" -TimeoutSec 2 -UseBasicParsing
        if ($response.StatusCode -eq 200) {
            $frontendReady = $true
            break
        }
    } catch {
        Start-Sleep -Seconds 1
    }
}
if (-not $frontendReady) {
    throw "VidLens frontend was not ready within 30 seconds. See $LogRoot\frontend.err.log"
}
$frontendPid = Get-ListeningProcessId -Port $FrontendPort

[pscustomobject]@{
    backendPid = $backendPid
    frontendPid = $frontendPid
    qdrantPid = $qdrantPid
    backendUrl = "http://127.0.0.1:$BackendPort"
    frontendUrl = "http://127.0.0.1:$FrontendPort"
    apiProvider = "Configured / $($aiConfig.Model)"
} | ConvertTo-Json | Set-Content -LiteralPath $PidPath -Encoding UTF8

Write-Output "VidLens started."
Write-Output "Frontend: http://127.0.0.1:$FrontendPort"
Write-Output "API docs: http://127.0.0.1:$BackendPort/docs"
Write-Output "AI config: $($aiConfig.Path) / $($aiConfig.Model)"
if ($WithVector) {
    Write-Output "Retrieval: coverage-aware-qdrant-hybrid-v2 (Qdrant 127.0.0.1:6333)"
} else {
    Write-Output "Retrieval: coverage-aware-lexical-v3 (lexical only)"
}
if (-not $NoBrowser) {
    try {
        Start-Process "http://127.0.0.1:$FrontendPort"
    } catch {
        Write-Warning "无法自动打开浏览器，请手动访问 http://127.0.0.1:$FrontendPort"
    }
}
