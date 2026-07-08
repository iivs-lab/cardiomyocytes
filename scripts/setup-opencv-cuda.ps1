#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Make the OpenCV CUDA wheel importable from any Python environment.

.DESCRIPTION
    The prebuilt OpenCV CUDA wheel loads the CUDA runtime from the toolkit
    (CUDA_PATH\bin\x64, which its bundled config.py already registers) but
    cannot find cuDNN: the cuDNN v9 installer keeps it in its own versioned
    folder and does not wire it into CUDA or PATH.

    This creates symbolic links to cuDNN's DLLs inside the CUDA toolkit's
    bin\x64 -- a directory the wheel already searches -- so a bare `import cv2`
    loads the CUDA build in vanilla Python, venv, uv, or conda alike, with no
    per-project code and without duplicating the DLLs. The links just point at
    the installed cuDNN, leaving its installer layout untouched.

    If cuDNN was instead installed from the zip/tarball straight into the CUDA
    bin, its DLLs are already where the wheel looks and the script reports that
    nothing needs doing. If the zip was unpacked elsewhere, point -CUDNN_PATH
    at that folder.

    Run once, elevated (symlinks require Administrator or Developer Mode).
    Re-run after upgrading cuDNN or CUDA to repoint the links.

.PARAMETER CUDNN_PATH
    Folder to link cuDNN from -- the installer's root (default) or the folder
    an extracted zip was unpacked to. The newest cuDNN found underneath is
    used. Ignored when cuDNN is already present in the CUDA bin.

.PARAMETER CUDA_PATH
    CUDA toolkit root to link into. Defaults to the CUDA_PATH environment
    variable set by the CUDA installer.
#>
[CmdletBinding()]
param(
    [string]$CUDNN_PATH = (Join-Path $env:ProgramFiles 'NVIDIA\CUDNN'),
    [string]$CUDA_PATH = $env:CUDA_PATH
)

$ErrorActionPreference = 'Stop'

if (-not $CUDA_PATH) {
    throw 'CUDA_PATH is not set. Install the CUDA Toolkit, or pass -CUDA_PATH.'
}

$cudaBin = Join-Path $CUDA_PATH 'bin\x64'
if (-not (Test-Path $cudaBin)) {
    throw "CUDA bin directory not found: $cudaBin"
}

# Zip/tarball install: if real cuDNN DLLs (not this script's own symlinks)
# already sit in the CUDA bin, cuDNN was extracted straight into the toolkit
# and there is nothing to do.
$present = Get-ChildItem -Path $cudaBin -Filter 'cudnn64_*.dll' -ErrorAction SilentlyContinue |
    Where-Object { -not $_.LinkType }
if ($present) {
    Write-Host "cuDNN already present in $cudaBin ($($present.Count) file(s)); nothing to do."
    return
}

# Otherwise locate cuDNN to link from: the installer layout by default, or the
# folder an extracted zip was unpacked to, passed via -CUDNN_PATH.
if (-not (Test-Path $CUDNN_PATH)) {
    throw "cuDNN not found at '$CUDNN_PATH'. Pass -CUDNN_PATH pointing at your cuDNN folder (installer dir or extracted zip), or copy its DLLs into $cudaBin."
}

$loader = Get-ChildItem -Path $CUDNN_PATH -Recurse -Filter 'cudnn64_*.dll' -ErrorAction SilentlyContinue |
    Sort-Object FullName | Select-Object -Last 1
if (-not $loader) {
    throw "No cuDNN DLLs (cudnn64_*.dll) found under '$CUDNN_PATH'. Install cuDNN, or pass -CUDNN_PATH pointing at the extracted zip."
}

$srcDir = $loader.Directory.FullName
if ($srcDir -eq $cudaBin) {
    Write-Host "cuDNN source is the CUDA bin itself; nothing to do."
    return
}

Write-Host "cuDNN source: $srcDir"
Write-Host "CUDA target:  $cudaBin"

$linked = 0
foreach ($dll in Get-ChildItem -Path $srcDir -Filter 'cudnn*.dll') {
    $link = Join-Path $cudaBin $dll.Name
    if (Test-Path $link) {
        Remove-Item -Path $link -Force
    }
    New-Item -ItemType SymbolicLink -Path $link -Target $dll.FullName | Out-Null
    Write-Host "  linked $($dll.Name)"
    $linked++
}

Write-Host "Done: $linked cuDNN symlinks created. A bare 'import cv2' now loads the CUDA build."
