from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_COLUMNS = (
    "run",
    "log_file",
    "status",
    "dataset",
    "model_parameters",
    "vocab_size",
    "context_length",
    "d_model",
    "num_layers",
    "num_heads",
    "d_ff",
    "batch_size",
    "lr_max",
    "total_steps",
    "max_step",
    "tokens_processed",
    "wall_time_seconds",
    "final_train_loss",
    "best_val_loss",
    "final_val_loss",
    "median_tokens_per_second",
    "peak_memory_mib",
)


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate CS336 experiment logs and render plots.")
    parser.add_argument("--log_dir", type=Path, default=Path("logs"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/experiments"))
    parser.add_argument("--pattern", default="*.csv", help="Recursive log filename pattern.")
    return parser.parse_args()


def optional_float(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def optional_int(value):
    number = optional_float(value)
    return int(number) if number is not None else None


def load_config(log_path: Path):
    config_path = Path(f"{log_path}.config.json")
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as config_f:
        return json.load(config_f)


def infer_from_name(name: str, pattern: str, converter):
    match = re.search(pattern, name, flags=re.IGNORECASE)
    if not match:
        return None
    raw_value = next((group for group in match.groups() if group is not None), None)
    try:
        return converter(raw_value)
    except (TypeError, ValueError):
        return None


def infer_dataset(config):
    train_path = str(config.get("train_path", "")).lower()
    if "owt" in train_path:
        return "OpenWebText"
    if "tiny" in train_path or "ts_" in train_path:
        return "TinyStories"
    return ""


def load_run(log_path: Path, log_dir: Path):
    with log_path.open(newline="", encoding="utf-8-sig") as log_f:
        rows = list(csv.DictReader(log_f))
    if not rows:
        return None

    relative_path = log_path.relative_to(log_dir)
    name = relative_path.with_suffix("").as_posix()
    config = load_config(log_path)
    batch_size = optional_int(config.get("batch_size"))
    if batch_size is None:
        batch_size = infer_from_name(name, r"(?:^|/)bs_(\d+)|(?:^|_)bs_(\d+)", int)
        if batch_size is None:
            # The alternation above can place the value in group two.
            match = re.search(r"(?:^|[/_])bs_(\d+)", name, flags=re.IGNORECASE)
            batch_size = int(match.group(1)) if match else None

    context_length = optional_int(config.get("context_length"))
    max_step = max((optional_int(row.get("step")) or 0 for row in rows), default=0)
    total_steps = optional_int(config.get("total_steps"))

    points = []
    for row in rows:
        step = optional_int(row.get("step"))
        tokens = optional_int(row.get("tokens_processed"))
        if tokens is None and step is not None and batch_size is not None and context_length is not None:
            tokens = (step + 1) * batch_size * context_length
        points.append(
            {
                "step": step,
                "tokens": tokens,
                "wall_time": optional_float(row.get("wall_time")),
                "train_loss": optional_float(row.get("train_loss")),
                "val_loss": optional_float(row.get("val_loss")),
                "tokens_per_second": optional_float(row.get("tokens_per_second")),
                "peak_memory_mib": optional_float(row.get("peak_memory_mib")),
            }
        )

    train_losses = [point["train_loss"] for point in points if point["train_loss"] is not None]
    val_losses = [point["val_loss"] for point in points if point["val_loss"] is not None]
    wall_times = [point["wall_time"] for point in points if point["wall_time"] is not None]
    token_counts = [point["tokens"] for point in points if point["tokens"] is not None]
    throughputs = [
        point["tokens_per_second"]
        for point in points
        if point["tokens_per_second"] is not None and point["tokens_per_second"] > 0
    ]
    memory_values = [point["peak_memory_mib"] for point in points if point["peak_memory_mib"] is not None]

    if total_steps is None:
        status = "legacy"
    elif max_step >= total_steps - 1:
        status = "complete"
    elif config.get("early_stop_val_loss") is not None:
        status = "early_stopped"
    else:
        status = "partial"

    summary = {
        "run": name,
        "log_file": relative_path.as_posix(),
        "status": status,
        "dataset": infer_dataset(config),
        "model_parameters": optional_int(config.get("model_parameters")),
        "vocab_size": optional_int(config.get("vocab_size")),
        "context_length": context_length,
        "d_model": optional_int(config.get("d_model")),
        "num_layers": optional_int(config.get("num_layers")),
        "num_heads": optional_int(config.get("num_heads")),
        "d_ff": optional_int(config.get("d_ff")),
        "batch_size": batch_size,
        "lr_max": optional_float(config.get("lr_max")),
        "total_steps": total_steps,
        "max_step": max_step,
        "tokens_processed": max(token_counts) if token_counts else None,
        "wall_time_seconds": max(wall_times) if wall_times else None,
        "final_train_loss": train_losses[-1] if train_losses else None,
        "best_val_loss": min(val_losses) if val_losses else None,
        "final_val_loss": val_losses[-1] if val_losses else None,
        "median_tokens_per_second": statistics.median(throughputs) if throughputs else None,
        "peak_memory_mib": max(memory_values) if memory_values else None,
    }
    return {"name": name, "config": config, "points": points, "summary": summary}


def format_summary_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return value


def write_summary(runs, output_path: Path):
    with output_path.open("w", newline="", encoding="utf-8") as output_f:
        writer = csv.DictWriter(output_f, fieldnames=SUMMARY_COLUMNS)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {key: format_summary_value(run["summary"].get(key)) for key in SUMMARY_COLUMNS}
            )


def save_validation_curves(runs, x_key, x_label, output_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    plotted = 0
    for run in runs:
        points = [
            (point[x_key], point["val_loss"])
            for point in run["points"]
            if point[x_key] is not None and point["val_loss"] is not None
        ]
        if not points:
            continue
        points.sort(key=lambda point: point[0])
        ax.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=3,
            linewidth=1.5,
            label=run["name"],
        )
        plotted += 1

    if not plotted:
        plt.close(fig)
        return False
    ax.set_xlabel(x_label)
    ax.set_ylabel("Validation loss (nats/token)")
    ax.set_title(f"Validation loss vs. {x_label.lower()}")
    ax.grid(True, alpha=0.25)
    if plotted <= 16:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def save_batch_plot(runs, metric, ylabel, title, output_path, select_min=False):
    grouped = defaultdict(list)
    for run in runs:
        batch_size = run["summary"].get("batch_size")
        value = run["summary"].get(metric)
        if batch_size is not None and value is not None:
            grouped[batch_size].append(value)
    if len(grouped) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for batch_size, values in sorted(grouped.items()):
        ax.scatter([batch_size] * len(values), values, alpha=0.45, color="tab:blue")
    batches = sorted(grouped)
    aggregate = [
        min(grouped[batch]) if select_min else statistics.median(grouped[batch])
        for batch in batches
    ]
    ax.plot(batches, aggregate, marker="o", linewidth=2, color="tab:orange")
    ax.set_xscale("log", base=2)
    ax.set_xticks(batches, [str(batch) for batch in batches])
    ax.set_xlabel("Training batch size")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def save_parameter_plot(runs, output_path):
    points = []
    for run in runs:
        parameters = run["summary"].get("model_parameters")
        val_loss = run["summary"].get("final_val_loss")
        if parameters is not None and val_loss is not None:
            points.append((parameters, val_loss, run["name"]))
    if len({point[0] for point in points}) < 2:
        return False

    fig, ax = plt.subplots(figsize=(8, 5))
    for parameters, val_loss, name in points:
        ax.scatter(parameters, val_loss, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("Trainable parameters")
    ax.set_ylabel("Final validation loss (nats/token)")
    ax.set_title("Validation loss vs. model size")
    ax.grid(True, alpha=0.25)
    if len(points) <= 16:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return True


def main():
    args = parse_args()
    log_dir = args.log_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not log_dir.is_dir():
        raise FileNotFoundError(f"Log directory does not exist: {log_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = []
    for log_path in sorted(log_dir.rglob(args.pattern)):
        run = load_run(log_path, log_dir)
        if run is not None:
            runs.append(run)
    if not runs:
        raise ValueError(f"No non-empty logs matching {args.pattern!r} found under {log_dir}")

    summary_path = output_dir / "summary.csv"
    write_summary(runs, summary_path)

    generated = [summary_path]
    plot_specs = (
        ("tokens", "Tokens processed", "validation_loss_vs_tokens.png"),
        ("wall_time", "Wall-clock time (seconds)", "validation_loss_vs_wall_time.png"),
    )
    for x_key, x_label, filename in plot_specs:
        output_path = output_dir / filename
        if save_validation_curves(runs, x_key, x_label, output_path):
            generated.append(output_path)

    throughput_path = output_dir / "throughput_vs_batch_size.png"
    if save_batch_plot(
        runs,
        "median_tokens_per_second",
        "Median tokens/second",
        "Training throughput vs. batch size",
        throughput_path,
    ):
        generated.append(throughput_path)
    else:
        throughput_path.unlink(missing_ok=True)

    batch_loss_path = output_dir / "validation_loss_vs_batch_size.png"
    if save_batch_plot(
        runs,
        "final_val_loss",
        "Final validation loss (nats/token)",
        "Best final validation loss vs. batch size",
        batch_loss_path,
        select_min=True,
    ):
        generated.append(batch_loss_path)
    else:
        batch_loss_path.unlink(missing_ok=True)

    parameter_path = output_dir / "validation_loss_vs_parameters.png"
    if save_parameter_plot(runs, parameter_path):
        generated.append(parameter_path)

    print(f"Aggregated {len(runs)} runs.")
    for path in generated:
        print(path)


if __name__ == "__main__":
    main()
