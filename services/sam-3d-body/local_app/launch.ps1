param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv/Scripts/python.exe'
$app = Join-Path $PSScriptRoot 'app.py'
$logs = Join-Path $root 'local_output'
New-Item -ItemType Directory -Path $logs -Force | Out-Null
if (-not (Test-Path -LiteralPath $python)) { throw 'Missing SAM 3D Body virtual environment.' }
$port = $null
foreach ($candidate in 7872..7892) {
    try {
        $config = Invoke-RestMethod -Uri "http://127.0.0.1:$candidate/sam3d-health" -TimeoutSec 1
        if ($config.app -eq 'sam-3d-body-local' -and $config.sequential_video_decode -eq 1) {
            if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$candidate" }
            Write-Output "http://127.0.0.1:$candidate"
            exit 0
        }
    } catch {}
    $socket = [System.Net.Sockets.TcpListener]::new([System.Net.IPAddress]::Loopback, $candidate)
    try { $socket.Start(); $port = $candidate } catch {} finally { $socket.Stop() }
    if ($null -ne $port) { break }
}
if ($null -eq $port) { throw 'No available port between 7872 and 7892.' }
$process = Start-Process -FilePath $python -ArgumentList ('"{0}" --port {1}' -f $app, $port) -WorkingDirectory $root -WindowStyle Hidden -RedirectStandardOutput (Join-Path $logs 'server.log') -RedirectStandardError (Join-Path $logs 'server-error.log') -PassThru
for ($attempt = 0; $attempt -lt 90; $attempt++) {
    Start-Sleep -Seconds 1
    $process.Refresh()
    if ($process.HasExited) { throw "SAM 3D Body exited. See $logs/server-error.log" }
    try {
        $config = Invoke-RestMethod -Uri "http://127.0.0.1:$port/sam3d-health" -TimeoutSec 2
        if ($config.app -eq 'sam-3d-body-local' -and $config.sequential_video_decode -eq 1) {
            if (-not $NoBrowser) { Start-Process "http://127.0.0.1:$port" }
            Write-Output "http://127.0.0.1:$port"
            exit 0
        }
    } catch {}
}
throw "Startup timed out. See $logs/server-error.log"
