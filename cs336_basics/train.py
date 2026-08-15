import argparse
import csv
import json
import numpy as np
import os
import sys
import time
from datetime import datetime, timezone
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import (
    AdamW,
    capture_rng_state,
    cross_entropy,
    data_loading,
    gradient_clipping,
    learning_rate_schedule,
    load_checkpoint,
    restore_rng_state,
    save_checkpoint,
)
import torch


RESUME_CONFIG_KEYS = (
    "vocab_size",
    "context_length",
    "d_model",
    "num_heads",
    "num_layers",
    "d_ff",
    "theta",
    "lr_max",
    "lr_min",
    "warmup_steps",
    "total_steps",
    "beta1",
    "beta2",
    "eps",
    "weight_decay",
    "grad_clip",
    "train_path",
    "val_path",
    "batch_size",
    "seed",
    "val_seed",
    "val_batch_size",
    "val_batches",
)

LOG_COLUMNS = (
    "step",
    "wall_time",
    "tokens_processed",
    "lr",
    "train_loss",
    "val_loss",
    "grad_norm",
    "tokens_per_second",
    "peak_memory_mib",
)


def validate_resume_config(saved_config, args):
    mismatches = []
    current_config = vars(args)
    for key in RESUME_CONFIG_KEYS:
        if key in saved_config and saved_config[key] != current_config[key]:
            mismatches.append(f"{key}: checkpoint={saved_config[key]!r}, current={current_config[key]!r}")
    if mismatches:
        details = "\n  ".join(mismatches)
        raise ValueError(f"Resume configuration does not match the checkpoint:\n  {details}")

def parse_args():
    p = argparse.ArgumentParser()
    # Model
    p.add_argument("--vocab_size",     type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model",        type=int, default=512)
    p.add_argument("--num_heads",      type=int, default=8)
    p.add_argument("--num_layers",     type=int, default=4)
    p.add_argument("--d_ff",           type=int, default=None)  # None → auto 8/3*d_model
    p.add_argument("--theta",          type=float, default=10000.0)
    # Optimizer
    p.add_argument("--lr_max",    type=float, default=1e-3)
    p.add_argument("--lr_min",    type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--total_steps",  type=int, default=5000)
    p.add_argument("--beta1",     type=float, default=0.9)
    p.add_argument("--beta2",     type=float, default=0.999)
    p.add_argument("--eps",       type=float, default=1e-8)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # Data / training
    p.add_argument("--train_path", type=str, required=True)
    p.add_argument("--val_path",   type=str, required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device",     type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",           type=int, default=42)
    p.add_argument("--val_seed",       type=int, default=1337)
    p.add_argument("--val_batch_size", type=int, default=32)
    p.add_argument("--val_batches",    type=int, default=20)
    # Logging / checkpointing
    p.add_argument("--log_interval",  type=int, default=100)
    p.add_argument("--val_interval",  type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume",         type=str,  default=None)  # path to checkpoint
    p.add_argument("--early_stop_val_loss", type=float, default=None)
    p.add_argument("--overfit_batch",  action="store_true")
    p.add_argument("--log_file",       type=str, default=None)  # path to CSV log
    return p.parse_args()


def train():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    train_rng = np.random.default_rng(args.seed)
    train_data = np.memmap(args.train_path, dtype=np.uint16, mode='r')
    val_data = np.memmap(args.val_path, dtype=np.uint16, mode='r')

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        args.lr_max,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    wall_time_offset = 0.0
    tokens_processed_offset = 0
    if args.resume:
        saved_iteration, rng_state, training_state = load_checkpoint(
            args.resume,
            model,
            optimizer,
            return_extra_state=True,
        )
        start_step = saved_iteration + 1
        if training_state:
            validate_resume_config(training_state.get("config", {}), args)
            wall_time_offset = float(training_state.get("wall_time", 0.0))
            tokens_processed_offset = int(training_state.get("tokens_processed", 0))
        else:
            # Legacy checkpoints did not store RNG or cumulative counters.
            # Reconstruct the training-sampler position from the configured seed.
            tokens_processed_offset = start_step * args.batch_size * args.context_length
            if not args.overfit_batch and start_step > 0:
                train_rng.integers(
                    0,
                    len(train_data) - args.context_length,
                    size=start_step * args.batch_size,
                )
            print("warning: legacy checkpoint has no saved timing/RNG metadata")
        restore_rng_state(rng_state, train_rng)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    is_cuda = torch.device(args.device).type == "cuda"
    if is_cuda:
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))

    if args.overfit_batch:
        fixed_inputs, fixed_targets = data_loading(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
            rng=train_rng,
        )
        fixed_inputs, fixed_targets = fixed_inputs.long(), fixed_targets.long()

    if args.log_file:
        log_dir = os.path.dirname(os.path.abspath(args.log_file))
        os.makedirs(log_dir, exist_ok=True)

        config_path = f"{args.log_file}.config.json"
        now_utc = datetime.now(timezone.utc).isoformat()
        resume_existing_log = bool(
            args.resume and os.path.exists(args.log_file) and os.path.getsize(args.log_file) > 0
        )

        if resume_existing_log:
            with open(args.log_file, newline="") as existing_log_f:
                existing_reader = csv.DictReader(existing_log_f)
                if tuple(existing_reader.fieldnames or ()) != LOG_COLUMNS:
                    raise ValueError(
                        "Cannot append to a log with an older schema; choose a new --log_file."
                    )
                logged_steps = [int(row["step"]) for row in existing_reader]
            if logged_steps and max(logged_steps) > saved_iteration:
                raise ValueError(
                    "The log contains steps newer than the resume checkpoint; "
                    "choose a new --log_file to preserve both trajectories."
                )

            if os.path.exists(config_path):
                with open(config_path, encoding="utf-8") as config_f:
                    config = json.load(config_f)
            else:
                config = {}
            config.setdefault("resume_events", []).append(
                {
                    "resumed_at_utc": now_utc,
                    "checkpoint": args.resume,
                    "command": [sys.executable, *sys.argv],
                    "arguments": vars(args).copy(),
                }
            )
            log_mode = "a"
        else:
            config = vars(args).copy()
            config.update(
                {
                    "command": [sys.executable, *sys.argv],
                    "started_at_utc": now_utc,
                    "torch_version": str(torch.__version__),
                    "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                    "train_dataset_tokens": len(train_data),
                    "val_dataset_tokens": len(val_data),
                    "gpu_name": torch.cuda.get_device_name(torch.device(args.device)) if is_cuda else None,
                    "resume_events": [],
                }
            )
            log_mode = "w"

        with open(config_path, "w", encoding="utf-8") as config_f:
            json.dump(config, config_f, indent=2)

        log_f = open(args.log_file, log_mode, newline="")
        log_writer = csv.writer(log_f)
        if log_mode == "w":
            log_writer.writerow(LOG_COLUMNS)

    t0 = time.time()

    def save_training_checkpoint(step, path, elapsed, tokens_processed):
        save_checkpoint(
            model,
            optimizer,
            step,
            path,
            rng_state=capture_rng_state(train_rng),
            training_state={
                "wall_time": elapsed,
                "tokens_processed": tokens_processed,
                "config": vars(args).copy(),
            },
        )

    for step in range(start_step, args.total_steps):

        # 1. LR schedule
        lr = learning_rate_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        # 2. Batch creation
        if args.overfit_batch:
            inputs, targets = fixed_inputs, fixed_targets
        else:
            inputs, targets = data_loading(
                train_data,
                args.batch_size,
                args.context_length,
                args.device,
                rng=train_rng,
            )
            inputs, targets = inputs.long(), targets.long()

        # 3, 4. Forward pass & Loss
        logits = model(inputs)
        loss = cross_entropy(logits, targets) #cross_entropy takes raw logits and compute softmax + log + negative mean internally

        # 5-8. Backward & Update
        optimizer.zero_grad()
        loss.backward()
        grad_norm = gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        session_elapsed = time.time() - t0
        elapsed = wall_time_offset + session_elapsed
        run_steps = step - start_step + 1
        run_tokens_processed = run_steps * args.batch_size * args.context_length
        tokens_processed = tokens_processed_offset + run_tokens_processed
        tokens_per_second = (
            tokens_processed / elapsed
            if wall_time_offset > 0
            else run_tokens_processed / session_elapsed
        )
        peak_memory_mib = (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024**2)
            if is_cuda
            else ""
        )

        # Logging, validation, checkpointing
        if step % args.log_interval == 0:
            print(f"step {step:6d} | {elapsed:8.1f}s | lr {lr:.2e} | train_loss {loss.item():.4f}")
            if args.log_file:
                log_writer.writerow(
                    [
                        step,
                        f"{elapsed:.2f}",
                        tokens_processed,
                        f"{lr:.8e}",
                        f"{loss.item():.6f}",
                        "",
                        f"{grad_norm:.6f}",
                        f"{tokens_per_second:.2f}",
                        f"{peak_memory_mib:.2f}" if is_cuda else "",
                    ]
                )
                log_f.flush()

        if step % args.val_interval == 0 or step == args.total_steps - 1:
            model.eval()
            with torch.no_grad():
                val_losses = []
                # Reset the validation RNG so every evaluation uses the same
                # token windows, independent of the training RNG and batch size.
                val_rng = np.random.default_rng(args.val_seed)
                for _ in range(args.val_batches):
                    x, y = data_loading(
                        val_data,
                        args.val_batch_size,
                        args.context_length,
                        args.device,
                        rng=val_rng,
                    )
                    val_losses.append(cross_entropy(model(x.long()), y.long()).item())

            val_loss = sum(val_losses) / len(val_losses)
            session_elapsed = time.time() - t0
            elapsed = wall_time_offset + session_elapsed
            tokens_per_second = (
                tokens_processed / elapsed
                if wall_time_offset > 0
                else run_tokens_processed / session_elapsed
            )
            peak_memory_mib = (
                torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024**2)
                if is_cuda
                else ""
            )
            print(f"step {step:6d} | {elapsed:8.1f}s | val_loss {val_loss:.4f}")
            if args.log_file:
                log_writer.writerow(
                    [
                        step,
                        f"{elapsed:.2f}",
                        tokens_processed,
                        f"{lr:.8e}",
                        "",
                        f"{val_loss:.6f}",
                        "",
                        f"{tokens_per_second:.2f}",
                        f"{peak_memory_mib:.2f}" if is_cuda else "",
                    ]
                )
                log_f.flush()
            if args.early_stop_val_loss is not None and val_loss < args.early_stop_val_loss:
                path = os.path.join(args.checkpoint_dir, f"ckpt_val{val_loss:.4f}_step{step:06d}.pt")
                save_training_checkpoint(step, path, elapsed, tokens_processed)
                print(
                    f"val_loss {val_loss:.4f} < {args.early_stop_val_loss:.4f}; "
                    f"saved {path}"
                )
                break
            model.train()

        if step % args.save_interval == 0 and step > 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step:06d}.pt")
            save_training_checkpoint(step, path, elapsed, tokens_processed)
            print(f"saved checkpoint -> {path}")

    if args.log_file:
        log_f.close()

if __name__ == "__main__":
    train()
