$ErrorActionPreference = 'Stop'
$stage = Join-Path $PSScriptRoot 'staged'
$targets = @(
    (Join-Path $stage '新EWSA发布包\03_cover_letter\cover_letter_ESWA.docx'),
    (Join-Path $stage '新EWSA发布包\07_highlights\Highlights.docx'),
    (Join-Path $stage '论文\给辅导老师\新旧稿改进清单.docx')
)
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    foreach ($file in $targets) {
        $document = $word.Documents.Open($file,$false,$true)
        try {
            $pdf = [IO.Path]::ChangeExtension($file,'.pdf')
            $document.ExportAsFixedFormat($pdf,17)
            Write-Output "Exported $pdf"
        } finally { $document.Close(0) }
    }
} finally {
    $word.Quit()
    [void][Runtime.InteropServices.Marshal]::ReleaseComObject($word)
}
