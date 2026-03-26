param(
    [int]$Hours = 16,
    [string]$RunName = "",
    [string]$CondaExe = "D:\Anaconda\Scripts\conda.exe",
    [string]$CondaEnv = "PPO",
    [int]$CandidateBatchSize = 3,
    [int]$ScreenTopK = 2,
    [int]$ScreenEvalEpisodes = 1,
    [int]$EvalEpisodes = 3,
    [int]$ExtraEpisodes = 40,
    [double]$TimeBudgetSeconds = 600.0,
    [double]$ProcessTimeoutSeconds = 3600.0,
    [int]$AmpLookback = 4,
    [string]$Paths = "square,circle,butterfly",
    [string]$ScreenPaths = "",
    [switch]$DeterministicEval
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$scriptRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent $scriptRoot
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$label = if ([string]::IsNullOrWhiteSpace($RunName)) { "{0}h" -f $Hours } else { $RunName }
$runId = "{0}_{1}" -f $timestamp, $label
$runDir = Join-Path $scriptRoot "long_runs\$runId"
$null = New-Item -ItemType Directory -Force -Path $runDir

function Write-JsonFile {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][object]$Value
    )
    $Value | ConvertTo-Json -Depth 10 | Out-File -FilePath $Path -Encoding utf8
}

function Copy-IfExists {
    param(
        [Parameter(Mandatory = $true)][string]$SourcePath,
        [Parameter(Mandatory = $true)][string]$TargetPath
    )
    if (Test-Path -LiteralPath $SourcePath) {
        Copy-Item -LiteralPath $SourcePath -Destination $TargetPath -Force
    }
}

function Read-JsonObject {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return Get-Content -LiteralPath $Path -Raw -Encoding utf8 | ConvertFrom-Json
}

function Get-ResultsLineCount {
    param([string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) {
        return 0
    }
    return (Get-Content -LiteralPath $Path).Count
}

function Write-AmpSnapshot {
    param(
        [Parameter(Mandatory = $true)][string]$Path,
        [Parameter(Mandatory = $true)][int]$Lookback,
        [Parameter(Mandatory = $true)][string]$LocalCondaExe,
        [Parameter(Mandatory = $true)][string]$LocalCondaEnv,
        [Parameter(Mandatory = $true)][string]$WorkingDirectory
    )
    $snapshotScriptPath = "$Path.snapshot.py"
    $stdoutPath = "$Path.stdout.tmp"
    $stderrPath = "$Path.stderr.tmp"
    $process = $null
    $workingDirectoryLiteral = $WorkingDirectory.Replace("\", "\\")
    try {
        @"
import sys
sys.path.insert(0, r"$workingDirectoryLiteral")

from train import candidate_specs, compute_candidate_amplitude
from prepare import read_results_history

history = read_results_history()
for spec in candidate_specs():
    if spec.name == "baseline":
        continue
    amp = compute_candidate_amplitude(spec, history, $Lookback)
    print(f"{spec.name}\t{amp:.3f}")
"@ | Out-File -FilePath $snapshotScriptPath -Encoding utf8

        $process = Start-Process -FilePath $LocalCondaExe `
            -ArgumentList @("run", "-n", $LocalCondaEnv, "python", $snapshotScriptPath) `
            -WorkingDirectory $WorkingDirectory `
            -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath `
            -PassThru
        if (-not $process.WaitForExit(60000)) {
            & taskkill /PID $process.Id /T /F | Out-Null
            "amp snapshot timed out after 60 seconds" | Out-File -FilePath $Path -Encoding utf8
            return
        }

        $lines = @()
        if (Test-Path -LiteralPath $stdoutPath) {
            $lines += Get-Content -LiteralPath $stdoutPath
        }
        if (Test-Path -LiteralPath $stderrPath) {
            $stderrLines = Get-Content -LiteralPath $stderrPath
            if ($stderrLines) {
                $lines += $stderrLines
            }
        }
        if (-not $lines) {
            $lines = @("amp snapshot completed with no output")
        }
        $lines | Out-File -FilePath $Path -Encoding utf8
    }
    catch {
        $_ | Out-File -FilePath $Path -Encoding utf8
    }
    finally {
        Remove-Item -LiteralPath $snapshotScriptPath, $stdoutPath, $stderrPath -ErrorAction SilentlyContinue
    }
}

function To-InvariantText {
    param([double]$Value)
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

$currentBestPath = Join-Path $scriptRoot "workspace\current_best.json"
$leaderboardMarkdownPath = Join-Path $scriptRoot "workspace\leaderboard.md"
$leaderboardJsonPath = Join-Path $scriptRoot "workspace\leaderboard.json"
$resultsPath = Join-Path $scriptRoot "results.tsv"

$resultsLineCountBefore = Get-ResultsLineCount -Path $resultsPath
$startTime = Get-Date
$deadline = $startTime.AddHours($Hours)
$branch = (& git -C $repoRoot branch --show-current).Trim()
$gitHead = (& git -C $repoRoot rev-parse HEAD).Trim()
$stdoutLogPath = Join-Path $runDir "train.stdout.log"
$stderrLogPath = Join-Path $runDir "train.stderr.log"
$statusPath = Join-Path $runDir "status.json"

$trainArgs = @(
    "run",
    "-n", $CondaEnv,
    "python",
    "trajectory_autoresearch\train.py",
    "--max-experiments", "0",
    "--candidate-batch-size", $CandidateBatchSize.ToString(),
    "--screen-top-k", $ScreenTopK.ToString(),
    "--screen-eval-episodes", $ScreenEvalEpisodes.ToString(),
    "--eval-episodes", $EvalEpisodes.ToString(),
    "--extra-episodes", $ExtraEpisodes.ToString(),
    "--time-budget-seconds", (To-InvariantText -Value $TimeBudgetSeconds),
    "--process-timeout-seconds", (To-InvariantText -Value $ProcessTimeoutSeconds),
    "--amp-lookback", $AmpLookback.ToString(),
    "--paths", $Paths
)

if (-not [string]::IsNullOrWhiteSpace($ScreenPaths)) {
    $trainArgs += @("--screen-paths", $ScreenPaths)
}
if ($DeterministicEval.IsPresent) {
    $trainArgs += "--deterministic-eval"
}

$status = [ordered]@{
    run_id = $runId
    status = "initializing"
    branch = $branch
    git_head = $gitHead
    start_time = $startTime.ToString("yyyy-MM-dd HH:mm:ss")
    deadline_time = $deadline.ToString("yyyy-MM-dd HH:mm:ss")
    wrapper_pid = $PID
    command = @($CondaExe) + $trainArgs
    stdout_log = $stdoutLogPath
    stderr_log = $stderrLogPath
    long_run_dir = $runDir
    results_tsv = $resultsPath
    results_line_count_before = $resultsLineCountBefore
    current_best_before = Read-JsonObject -Path $currentBestPath
}
Write-JsonFile -Path $statusPath -Value $status

Copy-IfExists -SourcePath $currentBestPath -TargetPath (Join-Path $runDir "current_best_before.json")
Copy-IfExists -SourcePath $leaderboardMarkdownPath -TargetPath (Join-Path $runDir "leaderboard_before.md")
Copy-IfExists -SourcePath $leaderboardJsonPath -TargetPath (Join-Path $runDir "leaderboard_before.json")
Write-AmpSnapshot -Path (Join-Path $runDir "amp_before.tsv") -Lookback $AmpLookback -LocalCondaExe $CondaExe -LocalCondaEnv $CondaEnv -WorkingDirectory $scriptRoot

$process = Start-Process -FilePath $CondaExe `
    -ArgumentList $trainArgs `
    -WorkingDirectory $repoRoot `
    -RedirectStandardOutput $stdoutLogPath `
    -RedirectStandardError $stderrLogPath `
    -PassThru

$status.status = "running"
$status.child_pid = $process.Id
Write-JsonFile -Path $statusPath -Value $status

while (-not $process.HasExited -and (Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 30
    $process.Refresh()
}

$exitReason = "process_exited"
if (-not $process.HasExited) {
    & taskkill /PID $process.Id /T /F | Out-Null
    $exitReason = "deadline_reached"
    Start-Sleep -Seconds 5
}

$exitCode = $null
try {
    $process.Refresh()
    if ($process.HasExited) {
        $exitCode = $process.ExitCode
    }
}
catch {
}

$resultsLineCountAfter = Get-ResultsLineCount -Path $resultsPath
Copy-IfExists -SourcePath $currentBestPath -TargetPath (Join-Path $runDir "current_best_after.json")
Copy-IfExists -SourcePath $leaderboardMarkdownPath -TargetPath (Join-Path $runDir "leaderboard_after.md")
Copy-IfExists -SourcePath $leaderboardJsonPath -TargetPath (Join-Path $runDir "leaderboard_after.json")
if (Test-Path -LiteralPath $resultsPath) {
    Get-Content -LiteralPath $resultsPath -Tail 20 | Out-File -FilePath (Join-Path $runDir "results_tail_after.tsv") -Encoding utf8
}
Write-AmpSnapshot -Path (Join-Path $runDir "amp_after.tsv") -Lookback $AmpLookback -LocalCondaExe $CondaExe -LocalCondaEnv $CondaEnv -WorkingDirectory $scriptRoot

$status.status = "completed"
$status.end_time = (Get-Date).ToString("yyyy-MM-dd HH:mm:ss")
$status.exit_reason = $exitReason
$status.exit_code = $exitCode
$status.results_line_count_after = $resultsLineCountAfter
$status.results_rows_added = [Math]::Max(0, $resultsLineCountAfter - $resultsLineCountBefore)
$status.current_best_after = Read-JsonObject -Path $currentBestPath
Write-JsonFile -Path $statusPath -Value $status
