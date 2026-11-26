param([string]$Python = 'C:\Users\Darshan\anaconda3\python.exe', [switch]$Cuda)
$ErrorActionPreference = 'Stop'
$Repo = Split-Path $PSScriptRoot -Parent
$EnvPath = Join-Path $Repo '.venv'
if (!(Test-Path -LiteralPath $EnvPath)) { & $Python -m venv $EnvPath; if ($LASTEXITCODE) { exit $LASTEXITCODE } }
$Interpreter = Join-Path $EnvPath 'Scripts\python.exe'
$Index = if ($Cuda) { 'https://download.pytorch.org/whl/cu128' } else { 'https://download.pytorch.org/whl/cpu' }
& $Interpreter -m pip install --index-url $Index torch==2.9.1
if ($LASTEXITCODE) { exit $LASTEXITCODE }
& $Interpreter -m pip install -r (Join-Path $Repo 'requirements-training.txt') -e (Join-Path $Repo 'env')
exit $LASTEXITCODE
