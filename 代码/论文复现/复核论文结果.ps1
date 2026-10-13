$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$syncRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$archivePath = Join-Path $syncRoot '新EWSA发布包\08_supplementary_reproduction\supplementary_reproduction.zip'
$outputRoot = Join-Path 'D:\PPO_article_joint_runs\paper_reproduction' (Get-Date -Format 'yyyyMMdd_HHmmss_fff')
if (Test-Path -LiteralPath $outputRoot) { throw 'Output folder already exists.' }
New-Item -ItemType Directory -Path $outputRoot | Out-Null
Expand-Archive -LiteralPath $archivePath -DestinationPath $outputRoot
$savedArgumentMode = $global:PSNativeCommandArgumentPassing
try {
    $global:PSNativeCommandArgumentPassing = 'Legacy'
    . 'D:\PPO_article_work\PPO_project\tools\activate_ppo.ps1'
} finally {
    $global:PSNativeCommandArgumentPassing = $savedArgumentMode
}
python -B (Join-Path $outputRoot 'supplementary_reproduction\verify_records.py') --check-actions --output (Join-Path $outputRoot 'result_verification.json')
if ($LASTEXITCODE -ne 0) { throw 'Record verification failed.' }
Write-Output ('Result: ' + (Join-Path $outputRoot 'result_verification.json'))
