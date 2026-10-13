param([string]$TeXBin = '', [string]$OutputDirectory = '_build')
$ErrorActionPreference = 'Stop'
if (-not $TeXBin) {
    $found = Get-Command xelatex.exe -ErrorAction SilentlyContinue
    if ($found) { $TeXBin = Split-Path $found.Source }
    elseif (Test-Path -LiteralPath 'D:\texlive\2025\bin\windows\xelatex.exe') { $TeXBin = 'D:\texlive\2025\bin\windows' }
    else { throw 'XeLaTeX was not found. Pass -TeXBin with the existing TeX Live bin directory.' }
}
$sourceDirectory = $PSScriptRoot
if ([IO.Path]::IsPathRooted($OutputDirectory)) { $buildDirectory = $OutputDirectory }
else { $buildDirectory = Join-Path $sourceDirectory $OutputDirectory }
$null = New-Item -ItemType Directory -Path $buildDirectory -Force
$buildDirectory = (Resolve-Path -LiteralPath $buildDirectory).Path
if ($buildDirectory -eq (Resolve-Path -LiteralPath $sourceDirectory).Path) { throw 'Use a separate output directory.' }
Get-ChildItem -LiteralPath $sourceDirectory -File | Where-Object { $_.Extension -in '.tex','.bib','.bbl','.cls','.sty','.bst','.apc','.pdf' } | ForEach-Object {
    Copy-Item -LiteralPath $_.FullName -Destination (Join-Path $buildDirectory $_.Name)
}
$env:PATH = $TeXBin + [IO.Path]::PathSeparator + $env:PATH
$utf8 = New-Object Text.UTF8Encoding($false)
$receipts = New-Object 'System.Collections.Generic.List[object]'
function Invoke-TeXStep([string]$Program, [string[]]$Arguments, [string]$LogName) {
    $exe = Join-Path $TeXBin ($Program + '.exe')
    if (-not (Test-Path -LiteralPath $exe)) { throw "Missing compiler: $exe" }
    $started = (Get-Date).ToString('o')
    $output = & $exe @Arguments 2>&1
    $code = $LASTEXITCODE
    [IO.File]::WriteAllLines((Join-Path $buildDirectory $LogName), [string[]]$output, $utf8)
    $receipts.Add([pscustomobject]@{program=$exe;arguments=$Arguments;cwd=$buildDirectory;started=$started;exit_code=$code})
    [IO.File]::WriteAllText((Join-Path $buildDirectory 'build_receipt.json'), ($receipts | ConvertTo-Json -Depth 5), $utf8)
    if ($code -ne 0) { throw "Compilation failed ($Program, exit $code). See $buildDirectory\$LogName" }
}
Push-Location -LiteralPath $buildDirectory
try {
    $versions = & (Join-Path $TeXBin 'xelatex.exe') --version
    $versions += & (Join-Path $TeXBin 'bibtex.exe') --version
    [IO.File]::WriteAllLines((Join-Path $buildDirectory 'tool_versions.txt'), [string[]]$versions, $utf8)
    $texArguments = @('-interaction=nonstopmode','-halt-on-error','-file-line-error','-recorder','manuscript.tex')
    Invoke-TeXStep 'xelatex' $texArguments '01_xelatex.txt'
    Invoke-TeXStep 'bibtex' @('manuscript') '02_bibtex.txt'
    Invoke-TeXStep 'xelatex' $texArguments '03_xelatex.txt'
    Invoke-TeXStep 'xelatex' $texArguments '04_xelatex.txt'
    $log = Get-Content -LiteralPath (Join-Path $buildDirectory 'manuscript.log') -Raw
    if ($log -match 'undefined references|undefined citations|Missing character|multiply defined') { throw 'Unresolved references, glyphs or labels; inspect manuscript.log.' }
    $hash = Get-FileHash -LiteralPath (Join-Path $buildDirectory 'manuscript.pdf') -Algorithm SHA256
    [IO.File]::WriteAllText((Join-Path $buildDirectory 'manuscript.sha256'), ($hash.Hash.ToLower() + '  manuscript.pdf'), $utf8)
    Write-Output "Build succeeded: $buildDirectory\manuscript.pdf"
} finally { Pop-Location }
