param(
    [string]$Repository = "https://github.com/sibirica/PySPIDER.git",
    [string]$Branch = "main"
)

$ErrorActionPreference = "Stop"

$repositoryRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$target = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot "SPIDER"))
$expectedTarget = [System.IO.Path]::GetFullPath((Join-Path $repositoryRoot "SPIDER"))

if ($target -ne $expectedTarget -or [System.IO.Path]::GetDirectoryName($target) -ne $repositoryRoot) {
    throw "Refusing to replace unexpected target: $target"
}

$temporaryRoot = Join-Path ([System.IO.Path]::GetTempPath()) ("pyspider-sync-" + [guid]::NewGuid())
$clone = Join-Path $temporaryRoot "PySPIDER"
$staging = Join-Path $repositoryRoot (".pyspider-staging-" + [guid]::NewGuid())

try {
    New-Item -ItemType Directory -Path $temporaryRoot | Out-Null
    git clone --quiet --depth 1 --branch $Branch $Repository $clone
    if ($LASTEXITCODE -ne 0) {
        throw "Could not clone $Repository branch $Branch"
    }

    $source = Join-Path $clone "src\PySPIDER"
    if (-not (Test-Path -LiteralPath $source -PathType Container)) {
        throw "Upstream directory src/PySPIDER was not found"
    }

    $revision = git -C $clone rev-parse HEAD
    if ($LASTEXITCODE -ne 0) {
        throw "Could not determine the upstream revision"
    }

    New-Item -ItemType Directory -Path $staging | Out-Null
    Get-ChildItem -LiteralPath $source -Force | Copy-Item -Destination $staging -Recurse -Force

    if (Test-Path -LiteralPath $target) {
        Remove-Item -LiteralPath $target -Recurse -Force
    }
    Move-Item -LiteralPath $staging -Destination $target

    Set-Content -LiteralPath (Join-Path $repositoryRoot ".pyspider-revision") -Value $revision -Encoding ascii

    Write-Output "Synced PySPIDER src/PySPIDER at revision $revision"
}
finally {
    if (Test-Path -LiteralPath $staging) {
        Remove-Item -LiteralPath $staging -Recurse -Force
    }
    if (Test-Path -LiteralPath $temporaryRoot) {
        Remove-Item -LiteralPath $temporaryRoot -Recurse -Force
    }
}
