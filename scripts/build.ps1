param(
    [string]$BuildDir = 'build',
    [string]$Architecture = '89'
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    # Import the MSVC environment into this process only; no system settings change.
    if (-not $env:INCLUDE -or -not $env:LIB) {
        $vswhere = Join-Path ${env:ProgramFiles(x86)} 'Microsoft Visual Studio\Installer\vswhere.exe'
        if (-not (Test-Path -LiteralPath $vswhere)) { throw 'Open an x64 Native Tools prompt, then run this script.' }
        # VsDevCmd also invokes vswhere by name in a plain PowerShell session.
        $env:PATH = (Split-Path -Parent $vswhere) + ';' + $env:PATH
        $vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if (-not $vs) { throw 'MSVC x64 tools were not found.' }
        $devcmd = Join-Path $vs 'Common7\Tools\VsDevCmd.bat'
        $lines = & $env:ComSpec /d /c "call `"$devcmd`" -no_logo -arch=x64 -host_arch=x64 >nul && set"
        if ($LASTEXITCODE -ne 0) { throw 'VsDevCmd failed.' }
        foreach ($line in $lines) {
            if ($line -match '^([^=]+)=(.*)$') {
                [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
            }
        }
    }
    & cmake -S . -B $BuildDir -G 'NMake Makefiles' '-DCMAKE_BUILD_TYPE=Release' "-DCMAKE_CUDA_ARCHITECTURES=$Architecture"
    if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }
    & cmake --build $BuildDir --config Release
    if ($LASTEXITCODE -ne 0) { throw 'Build failed.' }
} finally { Pop-Location }
