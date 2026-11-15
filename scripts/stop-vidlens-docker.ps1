$ErrorActionPreference = 'Stop'
$RepoRoot = Split-Path -Parent $PSScriptRoot
$dockerCommand = Get-Command docker.exe -ErrorAction SilentlyContinue
$dockerExe = if ($dockerCommand) { $dockerCommand.Source } else {
    @(
        (Join-Path $env:LOCALAPPDATA 'Programs\DockerDesktop\resources\bin\docker.exe'),
        'C:\Program Files\Docker\Docker\resources\bin\docker.exe'
    ) | Where-Object { Test-Path -LiteralPath $_ } | Select-Object -First 1
}
if (-not $dockerExe) { throw 'Docker Desktop is not installed.' }
$env:PATH = (Split-Path -Parent $dockerExe) + ';' + $env:PATH
& $dockerExe compose --project-directory $RepoRoot --env-file (Join-Path $RepoRoot 'backend\.env') -f (Join-Path $RepoRoot 'docker-compose.yml') -f (Join-Path $RepoRoot 'docker-compose.local.yml') stop
if ($LASTEXITCODE -ne 0) { throw 'Could not stop VidLens containers.' }
Write-Host 'VidLens stopped. Database, uploads and models are preserved.'
