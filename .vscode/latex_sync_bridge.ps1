[CmdletBinding()]
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [ValidateSet("sumatra-open", "sumatra-forward", "code-goto")]
    [string]$Mode,

    [Parameter(Position = 1)]
    [string]$Arg1,

    [Parameter(Position = 2)]
    [string]$Arg2,

    [Parameter(Position = 3)]
    [string]$Arg3
)

function Get-RegistryAppPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$ExecutableName
    )

    $registryKeys = @(
        "HKCU:\Software\Microsoft\Windows\CurrentVersion\App Paths\$ExecutableName",
        "HKLM:\Software\Microsoft\Windows\CurrentVersion\App Paths\$ExecutableName"
    )

    foreach ($key in $registryKeys) {
        if (-not (Test-Path $key)) {
            continue
        }

        try {
            $item = Get-Item $key -ErrorAction Stop
            $defaultValue = $item.GetValue("")
            if ($defaultValue -and (Test-Path $defaultValue)) {
                return $defaultValue
            }

            $pathValue = (Get-ItemProperty $key -Name Path -ErrorAction SilentlyContinue).Path
            if ($pathValue) {
                $candidate = Join-Path $pathValue $ExecutableName
                if (Test-Path $candidate) {
                    return $candidate
                }
            }
        } catch {
            continue
        }
    }

    return $null
}

function Resolve-AppPath {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$CommandNames,

        [Parameter(Mandatory = $true)]
        [string]$ExecutableName,

        [Parameter(Mandatory = $true)]
        [string[]]$FallbackPaths
    )

    foreach ($name in $CommandNames) {
        $command = Get-Command $name -ErrorAction SilentlyContinue | Select-Object -First 1
        if (-not $command) {
            continue
        }

        if ($command.Source -and (Test-Path $command.Source)) {
            return $command.Source
        }

        if ($command.Path -and (Test-Path $command.Path)) {
            return $command.Path
        }
    }

    $registryPath = Get-RegistryAppPath -ExecutableName $ExecutableName
    if ($registryPath) {
        return $registryPath
    }

    foreach ($candidate in $FallbackPaths) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    throw "Unable to locate $ExecutableName on this machine."
}

function Start-ResolvedProcess {
    param(
        [Parameter(Mandatory = $true)]
        [string]$FilePath,

        [Parameter(Mandatory = $true)]
        [string[]]$Arguments
    )

    Start-Process -FilePath $FilePath -ArgumentList $Arguments | Out-Null
}

try {
    switch ($Mode) {
        "sumatra-open" {
            if (-not $Arg1) {
                throw "Missing PDF path."
            }

            $sumatraPath = Resolve-AppPath `
                -CommandNames @("SumatraPDF.exe") `
                -ExecutableName "SumatraPDF.exe" `
                -FallbackPaths @(
                    "D:\SumatraPDF\SumatraPDF.exe",
                    "$env:LOCALAPPDATA\SumatraPDF\SumatraPDF.exe",
                    "C:\Program Files\SumatraPDF\SumatraPDF.exe",
                    "C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe"
                )

            Start-ResolvedProcess -FilePath $sumatraPath -Arguments @(
                "-reuse-instance",
                $Arg1
            )
        }

        "sumatra-forward" {
            if (-not $Arg1 -or -not $Arg2 -or -not $Arg3) {
                throw "Missing SyncTeX arguments."
            }

            $sumatraPath = Resolve-AppPath `
                -CommandNames @("SumatraPDF.exe") `
                -ExecutableName "SumatraPDF.exe" `
                -FallbackPaths @(
                    "D:\SumatraPDF\SumatraPDF.exe",
                    "$env:LOCALAPPDATA\SumatraPDF\SumatraPDF.exe",
                    "C:\Program Files\SumatraPDF\SumatraPDF.exe",
                    "C:\Program Files (x86)\SumatraPDF\SumatraPDF.exe"
                )

            Start-ResolvedProcess -FilePath $sumatraPath -Arguments @(
                "-reuse-instance",
                "-forward-search",
                $Arg1,
                $Arg2,
                $Arg3
            )
        }

        "code-goto" {
            if (-not $Arg1 -or -not $Arg2) {
                throw "Missing file or line argument."
            }

            $codePath = Resolve-AppPath `
                -CommandNames @("code", "Code.exe") `
                -ExecutableName "Code.exe" `
                -FallbackPaths @(
                    "D:\Microsoft VS Code\Code.exe",
                    "C:\Program Files\Microsoft VS Code\Code.exe",
                    "$env:LOCALAPPDATA\Programs\Microsoft VS Code\Code.exe"
                )

            Start-ResolvedProcess -FilePath $codePath -Arguments @(
                "-r",
                "-g",
                "${Arg1}:$Arg2"
            )
        }
    }
} catch {
    [Console]::Error.WriteLine($_.Exception.Message)
    exit 1
}
