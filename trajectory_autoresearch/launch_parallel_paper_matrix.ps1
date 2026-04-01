param(
    [int]$Hours = 24,
    [string]$DeadlineTime = "",
    [string]$CondaExe = "D:\Anaconda\Scripts\conda.exe",
    [string]$CondaEnv = "PPO",
    [string]$Paths = "square,circle,butterfly",
    [int]$TotalEpisodes = 420,
    [double]$TimeBudgetSeconds = 3600.0,
    [double]$ProcessTimeoutSeconds = 86400.0,
    [int]$EvalEpisodes = 5,
    [string]$Variants = "baseline_policy,abl_fixed_lookahead,abl_no_kcm,abl_no_lookahead_obs,abl_no_dual_reward",
    [string]$SuitePrefix = "paper_matrix_24h"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$deadline = if ([string]::IsNullOrWhiteSpace($DeadlineTime)) {
    (Get-Date).AddHours($Hours)
} else {
    [datetime]::Parse($DeadlineTime)
}
$deadlineIso = $deadline.ToString("s")
$activeLogsDir = Join-Path $scriptRoot "paper_runs\active_logs"
$null = New-Item -ItemType Directory -Force -Path $activeLogsDir
$statusPath = Join-Path $activeLogsDir ("{0}_{1}.status.json" -f $timestamp, $SuitePrefix)

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $Value | ConvertTo-Json -Depth 10 | Out-File -FilePath $Path -Encoding utf8
}

$variantList = @()
foreach ($item in $Variants.Split(",")) {
    $name = $item.Trim()
    if (-not [string]::IsNullOrWhiteSpace($name)) {
        $variantList += $name
    }
}

$payload = [ordered]@{
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    deadline_time = $deadline.ToString("yyyy-MM-dd HH:mm:ss")
    suite_prefix = $SuitePrefix
    paths = $Paths
    processes = @()
    status = "running"
}

foreach ($variant in $variantList) {
    $suiteName = "{0}_{1}_{2}" -f $timestamp, $SuitePrefix, $variant
    $stdoutPath = Join-Path $activeLogsDir ("{0}.stdout.log" -f $suiteName)
    $stderrPath = Join-Path $activeLogsDir ("{0}.stderr.log" -f $suiteName)
    $proc = Start-Process -FilePath "C:\Program Files\PowerShell\7\pwsh.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-Command",
            "$CondaExe run -n $CondaEnv python trajectory_autoresearch\paper_suite.py --suite-name $suiteName --paths $Paths --variants $variant --total-episodes $TotalEpisodes --time-budget-seconds $TimeBudgetSeconds --process-timeout-seconds $ProcessTimeoutSeconds --eval-episodes $EvalEpisodes --deadline-time $deadlineIso --sync-after-each"
        ) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
    $payload.processes += [ordered]@{
        variant = $variant
        suite_name = $suiteName
        pid = $proc.Id
        stdout_log = $stdoutPath
        stderr_log = $stderrPath
    }
}

Write-JsonFile -Path $statusPath -Value $payload

while ((Get-Date) -lt $deadline) {
    $alive = @()
    foreach ($entry in $payload.processes) {
        if (Get-Process -Id $entry.pid -ErrorAction SilentlyContinue) {
            $alive += $entry
        }
    }
    $payload.updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    $payload.alive_count = $alive.Count
    if ($alive.Count -eq 0) {
        break
    }
    Write-JsonFile -Path $statusPath -Value $payload
    Start-Sleep -Seconds 30
}

$payload.updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$payload.status = "completed"
Write-JsonFile -Path $statusPath -Value $payload

& $CondaExe run -n $CondaEnv python trajectory_autoresearch\paper_sync.py --once | Out-Null
