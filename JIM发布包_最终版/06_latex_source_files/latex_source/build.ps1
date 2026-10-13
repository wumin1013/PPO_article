param(
    [string]$TeXBin = '',
    [string]$OutputDirectory = '_build',
    [string]$Document = 'manuscript',
    [ValidateSet('pdflatex','xelatex')][string]$Engine = 'pdflatex'
)
$ErrorActionPreference = 'Stop'
if (-not $TeXBin) {
    $found = Get-Command ($Engine + '.exe') -ErrorAction SilentlyContinue
    if ($found) { $TeXBin = Split-Path $found.Source }
    elseif (Test-Path -LiteralPath ('D:\texlive\2025\bin\windows\' + $Engine + '.exe')) { $TeXBin = 'D:\texlive\2025\bin\windows' }
    else { throw 'TeX compiler not found; pass -TeXBin.' }
}
$sourceDirectory = $PSScriptRoot
$buildDirectory = if ([IO.Path]::IsPathRooted($OutputDirectory)) { $OutputDirectory } else { Join-Path $sourceDirectory $OutputDirectory }
$null = New-Item -ItemType Directory -Path $buildDirectory -Force
$buildDirectory = (Resolve-Path -LiteralPath $buildDirectory).Path
if ($buildDirectory -eq (Resolve-Path -LiteralPath $sourceDirectory).Path) { throw 'Use a separate output directory.' }
Get-ChildItem -LiteralPath $sourceDirectory -File | Where-Object { $_.Extension -in '.tex','.bib','.cls','.sty','.bst','.pdf','.eps','.png','.jpg' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $buildDirectory $_.Name)
}
$env:PATH = $TeXBin + [IO.Path]::PathSeparator + $env:PATH
$utf8 = New-Object Text.UTF8Encoding($false)
$receipts = New-Object 'System.Collections.Generic.List[object]'
function Invoke-Step([string]$Program, [string[]]$Arguments, [string]$LogName) {
    $exe = Join-Path $TeXBin ($Program + '.exe')
    $started = [DateTimeOffset]::Now.ToString('o')
    $output = & $exe @Arguments 2>&1
    $code = $LASTEXITCODE
    [IO.File]::WriteAllLines((Join-Path $buildDirectory $LogName), [string[]]$output, $utf8)
    $receipts.Add([pscustomobject]@{program=$exe;arguments=$Arguments;cwd=$buildDirectory;started=$started;exit_code=$code})
    [IO.File]::WriteAllText((Join-Path $buildDirectory 'build_receipt.json'), ($receipts | ConvertTo-Json -Depth 5), $utf8)
    if ($code -ne 0) { throw "Compilation failed: $Program, exit $code. See $buildDirectory\$LogName" }
}
Push-Location -LiteralPath $buildDirectory
try {
    $argsTeX = @('-interaction=nonstopmode','-halt-on-error','-file-line-error','-recorder',($Document + '.tex'))
    Invoke-Step $Engine $argsTeX '01_tex.txt'
    $aux = Get-Content -LiteralPath ($Document + '.aux') -Raw
    if ($aux -match '\\bibdata\{') { Invoke-Step 'bibtex' @($Document) '02_bibtex.txt' }
    Invoke-Step $Engine $argsTeX '03_tex.txt'
    Invoke-Step $Engine $argsTeX '04_tex.txt'
    $log = Get-Content -LiteralPath ($Document + '.log') -Raw
    if ($log -match 'Label\(s\) may have changed|Rerun to get cross-references') {
        Invoke-Step $Engine $argsTeX '05_tex.txt'
        $log = Get-Content -LiteralPath ($Document + '.log') -Raw
    }
    if ($log -match 'undefined references|undefined citations|Missing character|multiply defined') { throw 'Unresolved references, glyphs or labels; inspect TeX log.' }
    $digest = [Security.Cryptography.SHA256]::Create()
    try {
        $hash = ([BitConverter]::ToString($digest.ComputeHash([IO.File]::ReadAllBytes((Join-Path $buildDirectory ($Document + '.pdf')))))).Replace('-', '').ToLowerInvariant()
    } finally { $digest.Dispose() }
    [IO.File]::WriteAllText((Join-Path $buildDirectory ($Document + '.sha256')), ($hash + '  ' + $Document + '.pdf'), $utf8)
    Write-Output "Build succeeded: $buildDirectory\$Document.pdf"
} finally { Pop-Location }
