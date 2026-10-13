param([string]$CondaRoot = 'D:/Anaconda')

# Dot-source this file in each fresh PowerShell session before running Python.
# Older Conda wrappers pass empty _CE_M/_CE_CONDA arguments incorrectly under
# PowerShell 7's Standard native-argument mode. Limit Legacy mode to activation.
$ppoHookPath = Join-Path $CondaRoot 'shell/condabin/conda-hook.ps1'
if (-not (Test-Path -LiteralPath $ppoHookPath -PathType Leaf)) {
    throw "Conda initialization script not found: $ppoHookPath"
}

$ppoHasNativeArgumentMode = Test-Path Variable:PSNativeCommandArgumentPassing
$ppoSavedNativeArgumentMode = $PSNativeCommandArgumentPassing
try {
    if ($ppoHasNativeArgumentMode) { $PSNativeCommandArgumentPassing = 'Legacy' }
    . $ppoHookPath
    conda activate PPO
} finally {
    if ($ppoHasNativeArgumentMode) { $PSNativeCommandArgumentPassing = $ppoSavedNativeArgumentMode }
}

if ($env:CONDA_DEFAULT_ENV -ne 'PPO' -or -not $env:CONDA_PREFIX) {
    throw 'Conda did not activate PPO; do not run training in another environment.'
}
$ppoExpectedPython = Join-Path $env:CONDA_PREFIX 'python.exe'
$ppoActualPython = (Get-Command python -ErrorAction Stop).Source
if (-not [string]::Equals($ppoActualPython, $ppoExpectedPython, [StringComparison]::OrdinalIgnoreCase)) {
    throw "Python does not resolve to the active PPO environment: $ppoActualPython"
}
$env:PYTHONIOENCODING = 'utf-8'
Write-Output "PPO Python: $ppoActualPython"
