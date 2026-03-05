#!/usr/bin/env python3
"""Parse Edge Impulse-style training logs and plot loss curves.

Usage:
    python tools/plot_train_loss.py --input train.txt --output train_loss_curve.png
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


EPOCH_LINE_RE = re.compile(
    r"^\s*(\d+)\s+([0-9]*\.?[0-9]+)\s+([0-9]*\.?[0-9]+)\b"
)


def parse_loss_log(log_path: Path) -> tuple[list[int], list[float], list[float]]:
    """Extract epoch/train_loss/val_loss from a log file.

    If the same epoch appears multiple times, the last one is kept.
    """
    latest_by_epoch: dict[int, tuple[float, float]] = {}

    text = log_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        m = EPOCH_LINE_RE.match(line)
        if not m:
            continue

        epoch = int(m.group(1))
        train_loss = float(m.group(2))
        val_loss = float(m.group(3))
        latest_by_epoch[epoch] = (train_loss, val_loss)

    if not latest_by_epoch:
        raise ValueError(
            f"No epoch/loss rows found in '{log_path}'. Check log format."
        )

    epochs = sorted(latest_by_epoch.keys())
    train_losses = [latest_by_epoch[e][0] for e in epochs]
    val_losses = [latest_by_epoch[e][1] for e in epochs]
    return epochs, train_losses, val_losses


def plot_losses(
    epochs: list[int],
    train_losses: list[float],
    val_losses: list[float],
    output_path: Path,
    title: str,
) -> None:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(9, 5))
    plt.plot(epochs, train_losses, marker="o", linewidth=1.8, label="Train Loss")
    plt.plot(epochs, val_losses, marker="s", linewidth=1.8, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot training and validation loss from train log text"
    )
    parser.add_argument("--input", default="train.txt", help="Path to training log file")
    parser.add_argument(
        "--output",
        default="train_loss_curve.png",
        help="Output image path (.png recommended)",
    )
    parser.add_argument(
        "--title",
        default="Training vs Validation Loss",
        help="Chart title",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    epochs, train_losses, val_losses = parse_loss_log(input_path)
    plot_losses(epochs, train_losses, val_losses, output_path, args.title)

    print(f"Parsed {len(epochs)} unique epochs from: {input_path}")
    print(f"Saved loss curve to: {output_path}")


if __name__ == "__main__":
    main()
