param(
    [switch]$Foreground,
    [double]$MinFreeGiB = 8,
    [double]$MaxGpuUtilization = 35
)

$ErrorActionPreference = 'Stop'
$taskRoot = Split-Path -Parent $PSScriptRoot
$taskPython = Join-Path $taskRoot '.venv/Scripts/python.exe'
$taskRunner = Join-Path $taskRoot 'analysis/run_spatial_temporal_transformer.py'
$taskReports = Join-Path $taskRoot 'reports/spatial_temporal_transformer'
New-Item -ItemType Directory -Path $taskReports -Force | Out-Null

if ($Foreground) {
    Set-Location -LiteralPath $taskRoot
    $env:OMP_NUM_THREADS = '4'
    $env:MKL_NUM_THREADS = '4'
    & $taskPython -u $taskRunner --stage smoke --device cpu
    if ($LASTEXITCODE -ne 0) { throw 'CPU smoke failed; GPU work was not started.' }
    & $taskPython -u $taskRunner --stage smoke --device cuda --min-free-gib $MinFreeGiB --max-gpu-utilization $MaxGpuUtilization
    if ($LASTEXITCODE -ne 0) { throw 'CUDA smoke failed; experiment matrix was not started.' }
    & $taskPython -u $taskRunner --stage all --min-free-gib $MinFreeGiB --max-gpu-utilization $MaxGpuUtilization
    if ($LASTEXITCODE -ne 0) { throw 'Experiment worker failed; inspect worker.stderr.log before resuming.' }
    exit 0
}

if (Test-Path -LiteralPath (Join-Path $taskReports 'worker.lock')) {
    throw 'A worker currently owns this experiment. Inspect worker.lock and worker_status.json first.'
}
$taskStdout = Join-Path $taskReports 'worker.stdout.log'
$taskStderr = Join-Path $taskReports 'worker.stderr.log'
$taskArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PSCommandPath + '"'),
                   '-Foreground', '-MinFreeGiB', $MinFreeGiB, '-MaxGpuUtilization', $MaxGpuUtilization)
$taskProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList $taskArguments -WindowStyle Hidden `
    -WorkingDirectory $taskRoot -RedirectStandardOutput $taskStdout -RedirectStandardError $taskStderr -PassThru
@{ pid = $taskProcess.Id; stdout = $taskStdout; stderr = $taskStderr; min_free_gib = $MinFreeGiB;
   max_gpu_utilization = $MaxGpuUtilization; started = (Get-Date).ToString('o') } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $taskReports 'launcher.json') -Encoding UTF8
Write-Output "Started hidden experiment queue, PID $($taskProcess.Id). Log: $taskStdout"
