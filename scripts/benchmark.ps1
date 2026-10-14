param(
    [string]$Exe = 'build\gemm.exe',
    [ValidateSet('cublas', 'naive')][string]$Kernel = 'cublas',
    [string]$OutputDir = '',
    [int]$Iters = 0
)
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Push-Location $root
try {
    $Exe = (Resolve-Path -LiteralPath $Exe).Path
    if ($Iters -lt 0) { throw 'Iters must be zero (matrix default) or positive.' }
    if (-not $OutputDir) { $OutputDir = 'results\run-' + (Get-Date -Format 'yyyyMMdd-HHmmss') }
    if (Test-Path -LiteralPath $OutputDir) { throw "Output directory already exists: $OutputDir" }
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
    $OutputDir = (Resolve-Path -LiteralPath $OutputDir).Path
    Copy-Item -LiteralPath benchmarks\matrix.csv -Destination (Join-Path $OutputDir 'matrix.csv')
    & $Exe --mode info | Set-Content -Encoding UTF8 (Join-Path $OutputDir 'info.txt')
    if ($LASTEXITCODE -ne 0) { throw 'Device info failed.' }
    # Evidence includes the executable and source files, even without a git repository.
    $sources = @('CMakeLists.txt', 'src\gemm.cu', 'src\naive.cu', 'src\naive.cuh',
        'src\build_info.h.in', 'benchmarks\matrix.csv', 'scripts\benchmark.ps1')
    $sources += $Exe
    $sources | ForEach-Object { Get-FileHash -LiteralPath $_ -Algorithm SHA256 } |
        Select-Object Path,Hash | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputDir 'sha256.csv')
    "started_local=$([DateTimeOffset]::Now.ToString('o'))" | Set-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    "timezone=$([TimeZoneInfo]::Local.Id)" | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    & nvcc --version | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    & cmake --version | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    & powercfg /getactivescheme | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    & nvidia-smi | Set-Content -Encoding UTF8 (Join-Path $OutputDir 'nvidia-smi-before.txt')
    $telemetryArgs = @('--query-gpu=name,driver_version,pstate,temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit', '--format=csv')
    $combined = @()
    foreach ($row in (Import-Csv -LiteralPath benchmarks\matrix.csv)) {
        $count = if ($Iters -gt 0) { $Iters } else { [int]$row.iters }
        $caseCsv = Join-Path $OutputDir ($row.id + '.csv')
        $arguments = @('--mode','bench','--kernel',$Kernel,'--dtype',$row.dtype,
            '--m',$row.m,'--n',$row.n,'--k',$row.k,'--seed',$row.seed,
            '--warmup',$row.warmup,'--iters',"$count",'--rounds',$row.rounds,'--csv',$caseCsv)
        $command = '& "' + $Exe + '" ' + (($arguments | ForEach-Object { '"' + $_ + '"' }) -join ' ')
        $command | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'commands.txt')
        & nvidia-smi @telemetryArgs | Set-Content -Encoding UTF8 (Join-Path $OutputDir ($row.id + '-before.csv'))
        & $Exe @arguments 2>&1 | Tee-Object -FilePath (Join-Path $OutputDir ($row.id + '.log'))
        $caseExit = $LASTEXITCODE
        & nvidia-smi @telemetryArgs | Set-Content -Encoding UTF8 (Join-Path $OutputDir ($row.id + '-after.csv'))
        if ($caseExit -ne 0) { throw "Benchmark $($row.id) failed (exit $caseExit); no valid ranking." }
        $combined += Import-Csv -LiteralPath $caseCsv | Select-Object @{Name='id';Expression={$row.id}},*
    }
    $combined | Export-Csv -NoTypeInformation -Encoding UTF8 (Join-Path $OutputDir 'summary.csv')
    "finished_local=$([DateTimeOffset]::Now.ToString('o'))" | Add-Content -Encoding UTF8 (Join-Path $OutputDir 'environment.txt')
    Write-Host "Saved benchmark evidence: $OutputDir"
} finally { Pop-Location }
