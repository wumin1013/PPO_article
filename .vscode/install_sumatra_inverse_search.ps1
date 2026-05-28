[CmdletBinding()]
param()

function Get-ExistingSumatraSettingsPaths {
    $paths = New-Object System.Collections.Generic.List[string]
    $localSettingsPath = Join-Path $env:LOCALAPPDATA "SumatraPDF\SumatraPDF-settings.txt"
    [void]$paths.Add($localSettingsPath)

    $sumatraCandidates = @(
        "D:\SumatraPDF\SumatraPDF.exe",
        "$env:LOCALAPPDATA\SumatraPDF\SumatraPDF.exe",
        "$env:ProgramFiles\SumatraPDF\SumatraPDF.exe",
        "${env:ProgramFiles(x86)}\SumatraPDF\SumatraPDF.exe"
    )

    foreach ($candidate in $sumatraCandidates) {
        if (-not $candidate -or -not (Test-Path -LiteralPath $candidate)) {
            continue
        }

        $portableSettingsPath = Join-Path (Split-Path $candidate -Parent) "SumatraPDF-settings.txt"
        if (Test-Path -LiteralPath $portableSettingsPath) {
            [void]$paths.Add($portableSettingsPath)
        }
    }

    return $paths | Select-Object -Unique
}

function Resolve-CodeCliPath {
    $command = Get-Command "code" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command -and $command.Source -and (Test-Path -LiteralPath $command.Source)) {
        return $command.Source
    }

    $fallbackPaths = @(
        "D:\Microsoft VS Code\bin\code.cmd",
        "C:\Program Files\Microsoft VS Code\bin\code.cmd",
        "$env:LOCALAPPDATA\Programs\Microsoft VS Code\bin\code.cmd"
    )

    foreach ($candidate in $fallbackPaths) {
        if ($candidate -and (Test-Path -LiteralPath $candidate)) {
            return $candidate
        }
    }

    throw "Unable to locate the VS Code command line launcher code.cmd."
}

function Resolve-WScriptPath {
    $candidate = Join-Path $env:WINDIR "System32\wscript.exe"
    if (Test-Path -LiteralPath $candidate) {
        return $candidate
    }

    $command = Get-Command "wscript.exe" -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($command -and $command.Source -and (Test-Path -LiteralPath $command.Source)) {
        return $command.Source
    }

    throw "Unable to locate wscript.exe on this machine."
}

function Install-InverseSearchLauncher {
    $sourcePath = Join-Path $PSScriptRoot "vscode_inverse_search.vbs"
    if (-not (Test-Path -LiteralPath $sourcePath)) {
        throw "Unable to locate vscode_inverse_search.vbs beside this script."
    }

    $targetDir = Join-Path $env:LOCALAPPDATA "VSCodeLatexSync"
    $targetPath = Join-Path $targetDir "vscode_inverse_search.vbs"
    if (-not (Test-Path -LiteralPath $targetDir)) {
        New-Item -ItemType Directory -Path $targetDir | Out-Null
    }

    Copy-Item -LiteralPath $sourcePath -Destination $targetPath -Force
    return $targetPath
}

function Set-ConfigLine {
    param(
        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string[]]$Content,

        [Parameter(Mandatory = $true)]
        [string]$Name,

        [Parameter(Mandatory = $true)]
        [string]$Value
    )

    $line = "$Name = $Value"
    $pattern = "^\s*$([regex]::Escape($Name))\s*="
    $replaced = $false
    $result = foreach ($item in $Content) {
        if ($item -match $pattern) {
            if (-not $replaced) {
                $line
                $replaced = $true
            }

            continue
        }

        $item
    }

    if (-not $replaced) {
        return @($result) + $line
    }

    return @($result)
}

function Write-Utf8NoBomLines {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,

        [Parameter(Mandatory = $true)]
        [AllowEmptyString()]
        [string[]]$Content
    )

    $encoding = New-Object System.Text.UTF8Encoding $false
    [System.IO.File]::WriteAllLines($Path, $Content, $encoding)
}

$codePath = Resolve-CodeCliPath
$launcherPath = Install-InverseSearchLauncher
$wscriptPath = Resolve-WScriptPath
$inverseSearchCommand = "`"$wscriptPath`" `"$launcherPath`" `"$codePath`" `"%f`" `"%l`""
$settingsPaths = Get-ExistingSumatraSettingsPaths

foreach ($settingsPath in $settingsPaths) {
    $settingsDir = Split-Path $settingsPath -Parent
    if (-not (Test-Path -LiteralPath $settingsDir)) {
        New-Item -ItemType Directory -Path $settingsDir | Out-Null
    }

    if (Test-Path -LiteralPath $settingsPath) {
        $content = Get-Content -LiteralPath $settingsPath
    } else {
        $content = @()
    }

    $content = Set-ConfigLine -Content $content -Name "InverseSearchCmdLine" -Value $inverseSearchCommand
    $content = Set-ConfigLine -Content $content -Name "EnableTeXEnhancements" -Value "true"
    $content = Set-ConfigLine -Content $content -Name "ReuseInstance" -Value "true"

    Write-Utf8NoBomLines -Path $settingsPath -Content $content
    Write-Output "Updated $settingsPath"
}
