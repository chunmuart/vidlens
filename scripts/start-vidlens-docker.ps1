param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
$dockerPaths = @(
    (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'),
    'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
)
$dockerExe = if ($dockerCommand) { $dockerCommand.Source } else {
    $dockerPaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $dockerExe) { throw 'Docker Desktop is not installed.' }
$env:PATH = (Split-Path -Parent $dockerExe) + ';' + $env:PATH

function Test-DockerEngine {
    $ErrorActionPreference = 'Continue'
    & $dockerExe info --format '{{.ServerVersion}}' >$null 2>$null
    return ($LASTEXITCODE -eq 0)
}

if (-not (Test-DockerEngine)) {
    $desktopPaths = @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\Docker Desktop.exe'),
        'C:\Program Files\Docker\Docker\Docker Desktop.exe'
    )
    $desktopExe = $desktopPaths | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
    if (-not $desktopExe) { throw 'Open Docker Desktop and wait until the engine is running.' }
    Start-Process -FilePath $desktopExe -WindowStyle Hidden
    $ready = $false
    for ($attempt = 0; $attempt -lt 60; $attempt++) {
        Start-Sleep -Seconds 2
        if (Test-DockerEngine) { $ready = $true; break }
    }
    if (-not $ready) { throw 'Docker engine did not become ready. Check Docker Desktop.' }
}

$configPath = Join-Path $RepoRoot 'backend\.env'
if (-not (Test-Path -LiteralPath $configPath)) { throw 'Missing backend/.env configuration.' }
$composeArgs = @(
    'compose', '--project-directory', $RepoRoot,
    '--env-file', $configPath,
    '-f', (Join-Path $RepoRoot 'docker-compose.yml'),
    '-f', (Join-Path $RepoRoot 'docker-compose.local.yml')
)
& $dockerExe @composeArgs up -d --build
if ($LASTEXITCODE -ne 0) { throw 'Docker Compose could not start the project.' }

$webUrl = 'http://localhost:5174'
$webReady = $false
for ($attempt = 0; $attempt -lt 60; $attempt++) {
    try {
        $response = Invoke-WebRequest -Uri "$webUrl/api/health" -UseBasicParsing -TimeoutSec 3
        if ($response.StatusCode -eq 200) { $webReady = $true; break }
    } catch { }
    Start-Sleep -Seconds 2
}
if (-not $webReady) { throw 'Containers started, but the web health check timed out. Check Docker Compose logs.' }
Write-Host "VidLens is ready: $webUrl"
Write-Host 'Database: MySQL in Docker. Previous SQLite data remains in its original file.'
if (-not $NoBrowser) { Start-Process $webUrl }
