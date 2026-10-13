param(
    [Parameter(Mandatory = $true)]
    [string]$NewRun,
    [string]$OldRun = "saved_models/phase31_mainline/20260228_164732",
    [string]$PythonExe = "D:\\Enviroment\\Anaconda3\\envs\\PPO\\python.exe"
)

$ErrorActionPreference = "Stop"

$newRunPath = [System.IO.Path]::GetFullPath($NewRun)
$oldRunPath = [System.IO.Path]::GetFullPath($OldRun)
New-Item -ItemType Directory -Force -Path $newRunPath | Out-Null

Write-Output "[PIPELINE] new_run=$newRunPath"
Write-Output "[PIPELINE] old_run=$oldRunPath"
Write-Output "[PIPELINE] python=$PythonExe"

if (-not (Test-Path $PythonExe)) {
    throw "Python executable not found: $PythonExe"
}

$env:DISABLE_FINAL_PLOT = "1"

& $PythonExe main.py --mode train --config configs/default.yaml --experiment_dir $newRunPath

$bestModel = Join-Path $newRunPath "checkpoints/best_model.pth"
if (-not (Test-Path $bestModel)) {
    $bestModel = Join-Path $newRunPath "checkpoints/tracking_model_final.pth"
}
if (-not (Test-Path $bestModel)) {
    throw "No model checkpoint found under $newRunPath/checkpoints"
}

$newConfigPath = Join-Path $newRunPath "config.yaml"
$testOut = Join-Path $newRunPath "best_eval"
& $PythonExe main.py --mode test --config $newConfigPath --model $bestModel --experiment_dir $testOut

$reportPath = Join-Path $newRunPath "comparison_vs_20260228_164732.md"
& $PythonExe tools/compare_run_metrics.py --old $oldRunPath --new $newRunPath --out $reportPath

Write-Output "[PIPELINE] report=$reportPath"
