$lrs = @("1e-4", "3e-4", "1e-3", "3e-3")
$lr_min = "1e-5"
$log_dir = "logs/lr_sweep"
New-Item -ItemType Directory -Force -Path $log_dir | Out-Null

foreach ($lr in $lrs) {
    Write-Host "=== lr_max=$lr ===" -ForegroundColor Cyan
    uv run python -m cs336_basics.train `
        --train_path data/ts_train_tokens.bin `
        --val_path   data/ts_valid_tokens.bin `
        --vocab_size 10000 `
        --d_model 512 `
        --num_heads 16 `
        --num_layers 4 `
        --batch_size 32 `
        --context_length 256 `
        --total_steps 2000 `
        --warmup_steps 200 `
        --lr_max $lr `
        --lr_min $lr_min `
        --log_interval 50 `
        --val_interval 500 `
        --save_interval 99999 `
        --log_file "$log_dir/lr_$lr.csv"
}

Write-Host "Sweep done. Logs in $log_dir/"
