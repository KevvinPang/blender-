$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$desktop = [Environment]::GetFolderPath('Desktop')
$shell = New-Object -ComObject WScript.Shell
$launcher = Join-Path $PSScriptRoot 'launch.ps1'
$shortcut = $shell.CreateShortcut((Join-Path $desktop 'SAM 3D Body 本地测试.lnk'))
$shortcut.TargetPath = Join-Path $env:SystemRoot 'System32/WindowsPowerShell/v1.0/powershell.exe'
$shortcut.Arguments = '-NoProfile -WindowStyle Hidden -ExecutionPolicy Bypass -File "{0}"' -f $launcher
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'SAM 3D Body - local image and video-frame evaluation'
$shortcut.IconLocation = "$env:SystemRoot\System32\shell32.dll,13"
$shortcut.Save()
$shortcut = $shell.CreateShortcut((Join-Path $desktop 'SAM 3D Body 模型登录下载.lnk'))
$shortcut.TargetPath = Join-Path $root '.venv/Scripts/python.exe'
$shortcut.Arguments = '"{0}"' -f (Join-Path $PSScriptRoot 'model_login.py')
$shortcut.WorkingDirectory = $root
$shortcut.Description = 'Hugging Face local login and authorized model download'
$shortcut.Save()
Write-Output 'Desktop shortcuts created.'
