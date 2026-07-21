$ErrorActionPreference = "Stop"

foreach ($runName in @("iam_only_30", "identity_finetune")) {
    $runPath = Join-Path "registration_runs" $runName
    Write-Output "=== $runName ==="
    if (Test-Path -LiteralPath (Join-Path $runPath "training_complete.txt")) {
        Write-Output "Status: complete"
    } elseif (Test-Path -LiteralPath (Join-Path $runPath "training_failed.txt")) {
        Write-Output "Status: failed"
        Get-Content -LiteralPath (Join-Path $runPath "training_failed.txt")
    } elseif (Test-Path -LiteralPath $runPath) {
        Write-Output "Status: running or waiting"
    } else {
        Write-Output "Status: not started"
    }

    $historyPath = Join-Path $runPath "history.csv"
    if (Test-Path -LiteralPath $historyPath) {
        Import-Csv -LiteralPath $historyPath |
            Select-Object -Last 3 epoch,seconds,val_epe,val_ssim,learning_rate |
            Format-Table -AutoSize
    }
    $stderrPath = Join-Path $runPath "train_stderr.log"
    if ((Test-Path -LiteralPath $stderrPath) -and (Get-Item -LiteralPath $stderrPath).Length -gt 0) {
        Write-Output "Recent stderr:"
        Get-Content -LiteralPath $stderrPath -Tail 20
    }
}
