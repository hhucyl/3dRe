$ErrorActionPreference = 'Stop'

$projectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$condaPython = 'D:\software\anaconda\install\python.exe'

if (Test-Path -LiteralPath $condaPython) {
    $python = $condaPython
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $python = (Get-Command python).Source
} else {
    throw '未找到 Python。请安装 Python 3.9+ 与 PyQt5。'
}

& $python (Join-Path $projectDir 'main.py') @args

