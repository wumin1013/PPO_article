# 设置本仓库会话优先使用 Anaconda Python，避免 WindowsApps 的占位符。
$anacondaRoot = "D:\Anaconda"
if (-not (Test-Path "$anacondaRoot\python.exe")) {
    Write-Error "未找到 $anacondaRoot\python.exe，请根据实际安装路径调整脚本。"
    exit 1
}

$env:PATH = "$anacondaRoot;$anacondaRoot\Scripts;$anacondaRoot\Library\bin;" + $env:PATH
python --version
