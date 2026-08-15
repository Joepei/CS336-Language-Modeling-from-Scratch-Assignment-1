param(
    [long]$TargetTokens = 16777216,
    [int]$ContextLength = 256,
    [switch]$DryRun
)

$batchSizes = @(1, 2, 4, 8, 16, 32, 64, 128, 256)
$lrMultipliers = @(0.5, 1.0, 2.0)
$baseBatchSize = 32
$baseLearningRate = 1e-3
$logDir = "logs/batch_sweep_controlled"
$checkpointRoot = "checkpoints/batch_sweep_controlled"
$invariantCulture = [System.Globalization.CultureInfo]::InvariantCulture

if (-not $DryRun) {
    New-Item -ItemType Directory -Force -Path $logDir | Out-Null
    New-Item -ItemType Directory -Force -Path $checkpointRoot | Out-Null
}

$stopSweep = $false

foreach ($batchSize in $batchSizes) {
    # Every batch size processes the same number of tokens. Because the batch
    # sizes are powers of two, the default target is exactly divisible.
    $totalSteps = [int][math]::Ceiling($TargetTokens / ($batchSize * $ContextLength))
    $actualTokens = $totalSteps * $batchSize * $ContextLength
    $warmupSteps = [math]::Max(1, [int][math]::Round($totalSteps * 0.1))
    $logInterval = [math]::Max(1, [int][math]::Floor($totalSteps / 50))
    $valInterval = [math]::Max(1, [int][math]::Floor($totalSteps / 8))

    # Square-root scaling is a starting point, not an assumption that one LR
    # fits every batch. Test a small local grid and select by validation loss.
    $scaledLearningRate = $baseLearningRate * [math]::Sqrt($batchSize / $baseBatchSize)

    foreach ($multiplier in $lrMultipliers) {
        $learningRate = $scaledLearningRate * $multiplier
        $minimumLearningRate = $learningRate / 10
        $lrArgument = $learningRate.ToString("G17", $invariantCulture)
        $lrMinArgument = $minimumLearningRate.ToString("G17", $invariantCulture)
        $lrLabel = $learningRate.ToString("0.0E+0", $invariantCulture).ToLowerInvariant()
        $runName = "bs_${batchSize}_lr_${lrLabel}"

        Write-Host (
            "=== {0}: steps={1}, tokens={2}, lr={3} ===" -f `
                $runName, $totalSteps, $actualTokens, $lrArgument
        ) -ForegroundColor Cyan

        if ($DryRun) {
            continue
        }

        uv run python -m cs336_basics.train `
            --train_path data/ts_train_tokens.bin `
            --val_path data/ts_valid_tokens.bin `
            --vocab_size 10000 `
            --d_model 512 `
            --num_heads 16 `
            --num_layers 4 `
            --batch_size $batchSize `
            --context_length $ContextLength `
            --total_steps $totalSteps `
            --warmup_steps $warmupSteps `
            --lr_max $lrArgument `
            --lr_min $lrMinArgument `
            --log_interval $logInterval `
            --val_interval $valInterval `
            --val_batch_size 32 `
            --val_batches 20 `
            --save_interval 999999999 `
            --checkpoint_dir "$checkpointRoot/$runName" `
            --log_file "$logDir/$runName.csv"

        if ($LASTEXITCODE -ne 0) {
            Write-Host "Error or memory limit reached at batch_size=$batchSize; stopping sweep." -ForegroundColor Red
            $stopSweep = $true
            break
        }
    }

    if ($stopSweep) {
        break
    }
}

if ($DryRun) {
    Write-Host "Dry run complete; no training was started."
} else {
    Write-Host "Controlled sweep complete. Logs are in $logDir/."
}
