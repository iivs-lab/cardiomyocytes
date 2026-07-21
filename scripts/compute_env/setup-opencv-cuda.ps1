<#
.SYNOPSIS
    Make the OpenCV CUDA wheel importable, without breaking PyTorch's cuDNN.

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

    IMPORTANT -- only the *core* cuDNN libraries are linked. That directory is
    on PATH, so every cuDNN consumer in the process searches it, PyTorch
    included -- and PyTorch bundles its OWN cuDNN in torch\lib. Linking a
    sub-library that torch does not bundle (newer cuDNN adds e.g.
    cudnn_engines_tensor_ir, cudnn_ext) makes torch's core load that foreign
    build, and every GPU convolution then fails with
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH. Non-core files are therefore
    skipped, and a stale link an earlier run left for them is removed -- so
    re-running this script repairs a machine already broken by it. Pass
    -LinkAll for the old link-everything behaviour (needed only if you use
    cv2.dnn's CUDA backend with those newer engines).

    If cuDNN was instead installed from the zip/tarball straight into the CUDA
    bin, its DLLs are already where the wheel looks and the script reports that
    nothing needs doing. If the zip was unpacked elsewhere, point -CUDNN_PATH
    at that folder.

    Linking requires elevation (Administrator or Developer Mode); -DryRun
    previews the changes and needs none. Re-run after upgrading cuDNN or CUDA
    to repoint the links.

.PARAMETER CUDNN_PATH
    Folder to link cuDNN from -- the installer's root (default) or the folder
    an extracted zip was unpacked to. The newest cuDNN found underneath is
    used. Ignored when cuDNN is already present in the CUDA bin.

.PARAMETER CUDA_PATH
    CUDA toolkit root to link into. Defaults to the CUDA_PATH environment
    variable set by the CUDA installer.

.PARAMETER LinkAll
    Link every cudnn*.dll, including the non-core sub-libraries that can break
    PyTorch's bundled cuDNN. Use only if cv2.dnn's CUDA backend needs them.

.PARAMETER DryRun
    Report the links and removals that would happen, changing nothing. Needs no
    elevation, so it doubles as a diagnostic.
#>
[CmdletBinding()]
param(
    [string]$CUDNN_PATH = (Join-Path $env:ProgramFiles 'NVIDIA\CUDNN'),
    [string]$CUDA_PATH = $env:CUDA_PATH,
    [switch]$LinkAll,
    [switch]$DryRun
)

$ErrorActionPreference = 'Stop'

# The standard cuDNN 9 set -- exactly what the PyTorch wheel bundles. Anything
# beyond it is an optional engine/extension a newer cuDNN added; linking those
# into the shared CUDA bin is what breaks torch (see .DESCRIPTION).
$CorePatterns = @(
    'cudnn64_*.dll'
    'cudnn_graph64_*.dll'
    'cudnn_ops64_*.dll'
    'cudnn_cnn64_*.dll'
    'cudnn_adv64_*.dll'
    'cudnn_engines_precompiled64_*.dll'
    'cudnn_engines_runtime_compiled64_*.dll'
    'cudnn_heuristic64_*.dll'
)

function Test-CoreCudnn {
    param([Parameter(Mandatory)][string]$Name)
    foreach ($pattern in $CorePatterns) {
        if ($Name -like $pattern) { return $true }
    }
    return $false
}

if (-not $DryRun) {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = [Security.Principal.WindowsPrincipal]::new($identity)
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw 'Creating symlinks requires elevation. Run as Administrator, or pass -DryRun to preview.'
    }
}

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
if ($DryRun) { Write-Host 'Mode:         DryRun (nothing will change)' }
if ($LinkAll) { Write-Host 'Mode:         LinkAll (non-core files linked too -- may break torch)' }
Write-Host ''

$linked = 0
$removed = 0
$skipped = 0

foreach ($dll in Get-ChildItem -Path $srcDir -Filter 'cudnn*.dll') {
    $link = Join-Path $cudaBin $dll.Name

    if ($LinkAll -or (Test-CoreCudnn -Name $dll.Name)) {
        if ($DryRun) {
            Write-Host "  would link   $($dll.Name)"
        }
        else {
            if (Test-Path $link) { Remove-Item -Path $link -Force }
            New-Item -ItemType SymbolicLink -Path $link -Target $dll.FullName | Out-Null
            Write-Host "  linked       $($dll.Name)"
        }
        $linked++
        continue
    }

    # Non-core: never link. Remove a symlink an earlier run left behind (only a
    # link -- a real DLL someone put there deliberately is left alone), so
    # re-running repairs a machine this script already broke.
    $skipped++
    $existing = Get-Item -Path $link -ErrorAction SilentlyContinue
    if ($existing -and $existing.LinkType) {
        if ($DryRun) {
            Write-Host "  would REMOVE $($dll.Name)  (stale link; breaks torch's cuDNN)"
        }
        else {
            Remove-Item -Path $link -Force
            Write-Host "  removed      $($dll.Name)  (stale link; breaks torch's cuDNN)"
        }
        $removed++
    }
    else {
        Write-Host "  skipped      $($dll.Name)  (non-core; would break torch's cuDNN)"
    }
}

$tense = if ($DryRun) { 'would be' } else { 'were' }
Write-Host ''
Write-Host "Done: $linked core link(s) $tense created, $removed stale non-core link(s) $tense removed."
if ($skipped -gt 0 -and -not $LinkAll) {
    Write-Host "Skipped $skipped non-core file(s) to keep PyTorch's bundled cuDNN intact (-LinkAll overrides)."
}
Write-Host "A bare 'import cv2' loads the CUDA build."
