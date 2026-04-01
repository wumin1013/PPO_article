[CmdletBinding()]
param()

$settingsPath = Join-Path $env:LOCALAPPDATA "SumatraPDF\SumatraPDF-settings.txt"
$inverseSearchCommand = 'powershell.exe -NoProfile -ExecutionPolicy Bypass -Command "$dir = Split-Path ''%f'' -Parent; while ($dir -and -not (Test-Path (Join-Path $dir ''.vscode\latex_sync_bridge.ps1''))) { $parent = Split-Path $dir -Parent; if ($parent -eq $dir) { break }; $dir = $parent }; $script = if ($dir -and (Test-Path (Join-Path $dir ''.vscode\latex_sync_bridge.ps1''))) { Join-Path $dir ''.vscode\latex_sync_bridge.ps1'' } else { Join-Path $env:APPDATA ''Code\User\scripts\latex_sync_bridge.ps1'' }; & $script code-goto ''%f'' ''%l''"'

if (-not (Test-Path $settingsPath)) {
    $settingsDir = Split-Path $settingsPath -Parent
    if (-not (Test-Path $settingsDir)) {
        New-Item -ItemType Directory -Path $settingsDir | Out-Null
    }

    Set-Content -Path $settingsPath -Value @(
        "InverseSearchCmdLine = $inverseSearchCommand",
        "EnableTeXEnhancements = true",
        "ReuseInstance = true"
    ) -Encoding UTF8
    Write-Output "Created $settingsPath"
    exit 0
}

$content = Get-Content -Path $settingsPath

if ($content -match '^InverseSearchCmdLine = ') {
    $content = $content -replace '^InverseSearchCmdLine = .*$', "InverseSearchCmdLine = $inverseSearchCommand"
} else {
    $content += "InverseSearchCmdLine = $inverseSearchCommand"
}

if ($content -match '^EnableTeXEnhancements = ') {
    $content = $content -replace '^EnableTeXEnhancements = .*$', "EnableTeXEnhancements = true"
} else {
    $content += "EnableTeXEnhancements = true"
}

if ($content -match '^ReuseInstance = ') {
    $content = $content -replace '^ReuseInstance = .*$', "ReuseInstance = true"
} else {
    $content += "ReuseInstance = true"
}

Set-Content -Path $settingsPath -Value $content -Encoding UTF8
Write-Output "Updated $settingsPath"
