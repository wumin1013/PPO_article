$ErrorActionPreference = 'Stop'
$sync = [IO.Path]::GetFullPath('D:\Nutstore\1\基于强化学习的运动规划方法').TrimEnd('\')
$stage = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot 'staged')).TrimEnd('\')
$archive = [IO.Path]::GetFullPath('D:\PPO_article_cleanup_archive\20260924_before_five_folders').TrimEnd('\')
$keep = @('.git','.gitignore','AGENTS.md','PPO_存储与清理规则.md')
$categories = @('代码','论文用到的数据和模型','论文','EAAI发布包','新EWSA发布包')
if (-not (Test-Path -LiteralPath (Join-Path $PSScriptRoot 'sync_full_inventory_hashed.json'))) {throw 'Missing inventory'}
if (-not (Test-Path -LiteralPath (Join-Path $stage '新EWSA发布包\release_manifest.json'))) {throw 'Missing validated release'}
if (Test-Path -LiteralPath $archive) {throw 'Archive exists; inspect the receipt before any retry.'}
foreach ($name in $categories) {
    if (-not (Test-Path -LiteralPath (Join-Path $stage $name))) {throw "Missing category: $name"}
    if (Test-Path -LiteralPath (Join-Path $sync $name)) {throw "Destination already exists: $name"}
}
# An early failed copy created only empty directories beneath this exact staging path.
$empty = Join-Path $stage '论文用到的数据和模型\project_work'
if (Test-Path -LiteralPath $empty) {
    $resolvedEmpty = (Resolve-Path -LiteralPath $empty).Path
    if (-not $resolvedEmpty.StartsWith($stage+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Empty-tree scope violation'}
    if (@(Get-ChildItem -LiteralPath $empty -File -Recurse -Force).Count -ne 0) {throw 'Unexpected file in failed-copy directory'}
    Remove-Item -LiteralPath $resolvedEmpty -Recurse -Force
}
New-Item -ItemType Directory -Path $archive | Out-Null
$records = [Collections.Generic.List[object]]::new()
$items = @(Get-ChildItem -LiteralPath $sync -Force | Where-Object {$_.Name -notin $keep})
foreach ($item in $items) {
    $source = (Resolve-Path -LiteralPath $item.FullName).Path
    $dest = [IO.Path]::GetFullPath((Join-Path $archive $item.Name))
    if ([IO.Path]::GetDirectoryName($source) -ne $sync) {throw 'Source is not a direct child of the authorized root'}
    if (-not $dest.StartsWith($archive+'\',[StringComparison]::OrdinalIgnoreCase)) {throw 'Archive scope violation'}
    Move-Item -LiteralPath $source -Destination $dest
    $records.Add([PSCustomObject]@{operation='archive';source=$source;destination=$dest})
    $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'layout_move_receipt.json') -Encoding utf8NoBOM
}
foreach ($name in $categories) {
    $source = (Resolve-Path -LiteralPath (Join-Path $stage $name)).Path
    $dest = [IO.Path]::GetFullPath((Join-Path $sync $name))
    if ([IO.Path]::GetDirectoryName($source) -ne $stage) {throw 'Stage scope violation'}
    if ([IO.Path]::GetDirectoryName($dest) -ne $sync) {throw 'Final scope violation'}
    Move-Item -LiteralPath $source -Destination $dest
    $records.Add([PSCustomObject]@{operation='install';source=$source;destination=$dest})
    $records | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $PSScriptRoot 'layout_move_receipt.json') -Encoding utf8NoBOM
}
Write-Output "Installed five categories. Historical material is at $archive"
