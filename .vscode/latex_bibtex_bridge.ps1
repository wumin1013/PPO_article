param(
    [Parameter(Mandatory = $true)]
    [string]$DocFile,

    [Parameter(Mandatory = $true)]
    [string]$OutDir,

    [Parameter(Mandatory = $true)]
    [string]$SourceDir
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$docBase = [System.IO.Path]::GetFileNameWithoutExtension($DocFile)
if (-not $docBase) {
    throw "DocFile is empty or invalid: $DocFile"
}

$previousBibInputs = $env:BIBINPUTS
$previousBstInputs = $env:BSTINPUTS

try {
    # BibTeX should run inside the aux/output directory, but still search the source
    # directory for references.bib and local bst files.
    $env:BIBINPUTS = "$SourceDir;$previousBibInputs"
    $env:BSTINPUTS = "$SourceDir;$previousBstInputs"

    Push-Location -LiteralPath $OutDir
    & bibtex $docBase
    exit $LASTEXITCODE
}
finally {
    Pop-Location
    $env:BIBINPUTS = $previousBibInputs
    $env:BSTINPUTS = $previousBstInputs
}
