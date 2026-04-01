param(
    [int]$TotalHours = 24,
    [int]$StageHours = 12,
    [string]$CondaExe = "D:\Anaconda\Scripts\conda.exe",
    [string]$CondaEnv = "PPO",
    [string]$Paths = "square,circle,butterfly",
    [string]$RunPrefix = "dual_stage_24h",
    [int]$Stage1CandidateBatchSize = 5,
    [double]$Stage1TimeBudgetSeconds = 240.0,
    [double]$Stage1ProcessTimeoutSeconds = 1500.0,
    [int]$Stage1ExtraEpisodes = 18,
    [int]$Stage2CandidateBatchSize = 4,
    [double]$Stage2TimeBudgetSeconds = 360.0,
    [double]$Stage2ProcessTimeoutSeconds = 2400.0,
    [int]$Stage2ExtraEpisodes = 30,
    [int]$EvalEpisodes = 3,
    [int]$AmpLookback = 6
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage1Deadline = (Get-Date).AddHours($StageHours)
$stage2Deadline = (Get-Date).AddHours($TotalHours)
$activeLogsDir = Join-Path $scriptRoot "paper_runs\active_logs"
$null = New-Item -ItemType Directory -Force -Path $activeLogsDir
$statusPath = Join-Path $activeLogsDir ("{0}_{1}.status.json" -f $timestamp, $RunPrefix)

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $Value | ConvertTo-Json -Depth 10 | Out-File -FilePath $Path -Encoding utf8
}

function Start-RefineWorker {
    param(
        [Parameter(Mandatory = $true)][datetime]$Deadline,
        [Parameter(Mandatory = $true)][string]$Label
    )
    $stdoutPath = Join-Path $activeLogsDir ("{0}_{1}_refine.stdout.log" -f $timestamp, $Label)
    $stderrPath = Join-Path $activeLogsDir ("{0}_{1}_refine.stderr.log" -f $timestamp, $Label)
    return Start-Process -FilePath "C:\Program Files\PowerShell\7\pwsh.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-Command",
            "$CondaExe run -n $CondaEnv python trajectory_autoresearch\refine_worker.py --deadline-time $($Deadline.ToString('s')) --paths $Paths --poll-seconds 45 --eval-episodes $EvalEpisodes --upgrade-top-k 3 --upgrade-min-pass-count 2 --upgrade-progress-threshold 0.985 --upgrade-extra-episodes 140 --upgrade-time-budget-seconds 2400 --upgrade-process-timeout-seconds 14400"
        ) `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath `
        -PassThru
}

function Stop-ProcessTree {
    param([int]$ProcId)
    if (Get-Process -Id $ProcId -ErrorAction SilentlyContinue) {
        taskkill /PID $ProcId /T /F | Out-Null
    }
}

$payload = [ordered]@{
    started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    stage1_deadline = $stage1Deadline.ToString("yyyy-MM-dd HH:mm:ss")
    stage2_deadline = $stage2Deadline.ToString("yyyy-MM-dd HH:mm:ss")
    status = "running"
    stages = @()
}
Write-JsonFile -Path $statusPath -Value $payload

$stagePlans = @(
    [ordered]@{
        label = "stage1"
        deadline = $stage1Deadline
        candidate_batch_size = $Stage1CandidateBatchSize
        time_budget_seconds = $Stage1TimeBudgetSeconds
        process_timeout_seconds = $Stage1ProcessTimeoutSeconds
        extra_episodes = $Stage1ExtraEpisodes
    },
    [ordered]@{
        label = "stage2"
        deadline = $stage2Deadline
        candidate_batch_size = $Stage2CandidateBatchSize
        time_budget_seconds = $Stage2TimeBudgetSeconds
        process_timeout_seconds = $Stage2ProcessTimeoutSeconds
        extra_episodes = $Stage2ExtraEpisodes
    }
)

foreach ($plan in $stagePlans) {
    $label = [string]$plan.label
    $deadline = [datetime]$plan.deadline
    $longRun = Start-Process -FilePath "C:\Program Files\PowerShell\7\pwsh.exe" `
        -ArgumentList @(
            "-NoProfile",
            "-ExecutionPolicy", "Bypass",
            "-File", (Join-Path $scriptRoot "launch_long_run.ps1"),
            "-DeadlineTime", $deadline.ToString("s"),
            "-RunName", ("{0}_{1}" -f $RunPrefix, $label),
            "-CondaExe", $CondaExe,
            "-CondaEnv", $CondaEnv,
            "-CandidateBatchSize", [string]$plan.candidate_batch_size,
            "-ScreenTopK", "0",
            "-ScreenEvalEpisodes", "1",
            "-EvalEpisodes", [string]$EvalEpisodes,
            "-ExtraEpisodes", [string]$plan.extra_episodes,
            "-TimeBudgetSeconds", [string]$plan.time_budget_seconds,
            "-ProcessTimeoutSeconds", [string]$plan.process_timeout_seconds,
            "-AmpLookback", [string]$AmpLookback,
            "-Paths", $Paths,
            "-PaperIntervalSeconds", "300",
            "-PaperTotalEpisodes", "320",
            "-PaperTimeBudgetSeconds", "1800",
            "-PaperProcessTimeoutSeconds", "21600",
            "-PaperEvalEpisodes", [string]$EvalEpisodes,
            "-PaperVariants", "full_method_snapshot"
        ) `
        -WorkingDirectory $repoRoot `
        -PassThru
    $refine = Start-RefineWorker -Deadline $deadline -Label $label
    $payload.stages += [ordered]@{
        label = $label
        started_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
        deadline = $deadline.ToString("yyyy-MM-dd HH:mm:ss")
        long_run_wrapper_pid = $longRun.Id
        refine_wrapper_pid = $refine.Id
    }
    Write-JsonFile -Path $statusPath -Value $payload

    while ((Get-Date) -lt $deadline) {
        if (-not (Get-Process -Id $longRun.Id -ErrorAction SilentlyContinue)) {
            break
        }
        Start-Sleep -Seconds 30
    }

    Stop-ProcessTree -Pid $refine.Id
    $payload.updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
    Write-JsonFile -Path $statusPath -Value $payload

    & $CondaExe run -n $CondaEnv python trajectory_autoresearch\paper_sync.py --once | Out-Null
}

$payload.updated_at = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$payload.status = "completed"
Write-JsonFile -Path $statusPath -Value $payload
