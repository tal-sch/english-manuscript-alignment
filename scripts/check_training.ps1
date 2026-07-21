param(
    [string]$RunDirectory = "registration_runs/final_combined"
)

$ErrorActionPreference = "Stop"
$runPath = Resolve-Path -LiteralPath $RunDirectory
$completePath = Join-Path $runPath "training_complete.txt"
$failedPath = Join-Path $runPath "training_failed.txt"
$historyPath = Join-Path $runPath "history.csv"
$stderrPath = Join-Path $runPath "train_stderr.log"

if (Test-Path -LiteralPath $completePath) {
    Write-Output "Status: complete"
} elseif (Test-Path -LiteralPath $failedPath) {
    Write-Output "Status: failed"
    Get-Content -LiteralPath $failedPath
} else {
    Write-Output "Status: running or waiting to start"
}

if (Test-Path -LiteralPath $historyPath) {
    Import-Csv -LiteralPath $historyPath |
        Select-Object -Last 5 epoch,seconds,train_total,val_epe,val_identity_epe,val_ssim,val_ink_dice,learning_rate |
        Format-Table -AutoSize
}

if ((Test-Path -LiteralPath $stderrPath) -and (Get-Item -LiteralPath $stderrPath).Length -gt 0) {
    Write-Output "Recent stderr:"
    Get-Content -LiteralPath $stderrPath -Tail 20
}
