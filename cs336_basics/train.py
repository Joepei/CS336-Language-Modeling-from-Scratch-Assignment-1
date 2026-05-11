import argparse
import csv
import numpy as np
import os
import time
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import AdamW, load_checkpoint, cross_entropy, learning_rate_schedule, gradient_clipping, data_loading, save_checkpoint
import torch

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
    # Logging / checkpointing
    p.add_argument("--log_interval",  type=int, default=100)
    p.add_argument("--val_interval",  type=int, default=500)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--resume",         type=str,  default=None)  # path to checkpoint
    p.add_argument("--overfit_batch",  action="store_true")
    p.add_argument("--log_file",       type=str, default=None)  # path to CSV log
    return p.parse_args()


def train():
    args = parse_args()
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
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer)

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if args.overfit_batch:
        fixed_inputs, fixed_targets = data_loading(train_data, args.batch_size, args.context_length, args.device)
        fixed_inputs, fixed_targets = fixed_inputs.long(), fixed_targets.long()

    if args.log_file:
        log_f = open(args.log_file, "w", newline="")
        log_writer = csv.writer(log_f)
        log_writer.writerow(["step", "wall_time", "train_loss", "val_loss"])

    t0 = time.time()
    for step in range(start_step, args.total_steps):

        # 1. LR schedule
        lr = learning_rate_schedule(step, args.lr_max, args.lr_min, args.warmup_steps, args.total_steps)
        for group in optimizer.param_groups:
            group['lr'] = lr

        # 2. Batch creation
        if args.overfit_batch:
            inputs, targets = fixed_inputs, fixed_targets
        else:
            inputs, targets = data_loading(train_data, args.batch_size, args.context_length, args.device)
            inputs, targets = inputs.long(), targets.long()

        # 3, 4. Forward pass & Loss
        logits = model(inputs)
        loss = cross_entropy(logits, targets)

        # 5-8. Backward & Update
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        # Logging, validation, checkpointing
        if step % args.log_interval == 0:
            elapsed = time.time() - t0
            print(f"step {step:6d} | {elapsed:8.1f}s | lr {lr:.2e} | train_loss {loss.item():.4f}")
            if args.log_file:
                log_writer.writerow([step, f"{elapsed:.2f}", f"{loss.item():.6f}", ""])

        if step % args.val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_losses = []
                for _ in range(20):
                    x, y = data_loading(val_data, args.batch_size, args.context_length, args.device)
                    val_losses.append(cross_entropy(model(x.long()), y.long()).item())

            val_loss = sum(val_losses) / len(val_losses)
            elapsed = time.time() - t0
            print(f"step {step:6d} | {elapsed:8.1f}s | val_loss {val_loss:.4f}")
            if args.log_file:
                log_writer.writerow([step, f"{elapsed:.2f}", "", f"{val_loss:.6f}"])
                log_f.flush()
            if val_loss < 1.45:
                path = os.path.join(args.checkpoint_dir, f"ckpt_val{val_loss:.4f}_step{step:06d}.pt")
                save_checkpoint(model, optimizer, step, path)
                print(f"val_loss {val_loss:.4f} < 1.45 — saved {path}")
                break
            model.train()

        if step % args.save_interval == 0 and step > 0:
            path = os.path.join(args.checkpoint_dir, f"ckpt_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, path)
            print(f"saved checkpoint → {path}")

    if args.log_file:
        log_f.close()

if __name__ == "__main__":
    train()
