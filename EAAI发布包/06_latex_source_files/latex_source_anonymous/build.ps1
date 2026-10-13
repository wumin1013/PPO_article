param()
$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath $PSScriptRoot
$utf8 = [Text.UTF8Encoding]::new($false)
$steps = [Collections.Generic.List[object]]::new()
try {
    [void][IO.Directory]::CreateDirectory((Join-Path $PSScriptRoot '_build'))
    Copy-Item -LiteralPath 'references.bib' -Destination '_build/references.bib'
    foreach ($program in @('xelatex','bibtex')) {
        if (-not (Get-Command $program -ErrorAction SilentlyContinue)) { throw "Missing compiler: $program" }
    }
    $versions = & xelatex --version
    $versions += & bibtex --version
    [IO.File]::WriteAllLines((Join-Path $PSScriptRoot '_build/tool_versions.txt'), [string[]]$versions, $utf8)
    function Invoke-Checked([string]$program, [string[]]$arguments, [string]$logName) {
        $output = & $program @arguments 2>&1
        $code = $LASTEXITCODE
        [IO.File]::WriteAllLines((Join-Path $PSScriptRoot ('_build/' + $logName)), [string[]]$output, $utf8)
        $steps.Add(@{program=$program; arguments=$arguments; exit_code=$code; cwd=$PSScriptRoot})
        [IO.File]::WriteAllText((Join-Path $PSScriptRoot '_build/build_receipt.json'), (ConvertTo-Json -InputObject @($steps.ToArray()) -Depth 5), $utf8)
        if ($code -ne 0) { throw "$program failed with exit code $code" }
    }
    $arguments = @('-interaction=nonstopmode','-halt-on-error','-file-line-error','-recorder','-output-directory=_build','manuscript.tex')
    Invoke-Checked 'xelatex' $arguments '01_xelatex.txt'
    Invoke-Checked 'bibtex' @('_build/manuscript') '02_bibtex.txt'
    Invoke-Checked 'xelatex' $arguments '03_xelatex.txt'
    Invoke-Checked 'xelatex' $arguments '04_xelatex.txt'
    Write-Output 'SUCCESS: _build/manuscript.pdf'
} finally { Pop-Location }
