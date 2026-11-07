param([string]$PythonPath, [int]$Port = 7872)
$ErrorActionPreference = 'Stop'
$captureRoot = Split-Path -Parent $PSScriptRoot
$captureService = Join-Path $captureRoot 'services/sam-3d-body'
if (-not $PythonPath) {
    $PythonPath = Join-Path $captureService '.venv/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
    throw 'Missing service Python. See docs/INSTALL.md, or pass -PythonPath.'
}
$env:MOCAP_ROOT = $captureRoot
$captureScript = Join-Path $captureService 'local_app/app.py'
& $PythonPath $captureScript --port $Port
exit $LASTEXITCODE
