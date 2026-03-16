#!/usr/bin/env python3
"""
Visualize Task 2(a) distributed training:
- Per-rank minibatch loss curves (4 ranks) + their per-step mean loss.
- Breakdown of compute vs communication latency.
- Report global mean loss across all ranks and steps.

Assumes run_glue.py saved, on rank 0:
  {output_dir}/train_loss_time_allranks.npz
with arrays:
  loss: [R, T]  per-rank minibatch losses
  time: [R, T]  per-step wall-clock time (forward + backward + communication)
  fwd:  [R, T]  per-step forward time
  bwd:  [R, T]  per-step backward + optimizer time
  comm: [R, T]  per-step communication time (gradient sync)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_rank_curves(out_dir: Path) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    path = out_dir / "train_loss_time_allranks.npz"
    if not path.exists():
        raise FileNotFoundError(f"Missing aggregated file: {path}")
    data = np.load(path)
    losses_arr = np.asarray(data["loss"], dtype=np.float32)  # [R, T]
    times_arr = np.asarray(data["time"], dtype=np.float32)   # [R, T]
    fwd_arr = np.asarray(data.get("fwd", np.zeros_like(times_arr)), dtype=np.float32)
    bwd_arr = np.asarray(data.get("bwd", np.zeros_like(times_arr)), dtype=np.float32)
    comm_arr = np.asarray(data.get("comm", np.zeros_like(times_arr)), dtype=np.float32)
    losses = [losses_arr[r] for r in range(losses_arr.shape[0])]
    times = [times_arr[r] for r in range(times_arr.shape[0])]
    fwds = [fwd_arr[r] for r in range(fwd_arr.shape[0])]
    bwds = [bwd_arr[r] for r in range(bwd_arr.shape[0])]
    comms = [comm_arr[r] for r in range(comm_arr.shape[0])]
    return losses, times, fwds, bwds, comms


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Task 2(a) distributed loss/time curves")
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path("/tmp/RTE_task2a"),
        help="Output dir used by run_glue.py (contains rank*_train_loss_time.npz files)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "task2a_loss_time.png",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    losses, times, fwds, bwds, comms = load_rank_curves(args.dir)

    # align lengths
    min_len = min(len(x) for x in losses)
    losses = [x[:min_len] for x in losses]
    times = [t[:min_len] for t in times]
    fwds = [c[:min_len] for c in fwds]
    bwds = [c[:min_len] for c in bwds]
    comms = [c[:min_len] for c in comms]

    steps = np.arange(min_len)
    losses_arr = np.stack(losses, axis=0)  # [R, T]
    mean_loss = losses_arr.mean(axis=0)    # [T]

    # time curves: average time of all ranks
    times_arr = np.stack(times, axis=0)
    mean_time = times_arr.mean(axis=0)
    fwd_arr = np.stack(fwds, axis=0)
    mean_fwd = fwd_arr.mean(axis=0)
    bwd_arr = np.stack(bwds, axis=0)
    mean_bwd = bwd_arr.mean(axis=0)
    comm_arr = np.stack(comms, axis=0)
    mean_comm = comm_arr.mean(axis=0)

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes

    # top: four loss curves + average loss curve
    world_size = len(losses)
    colors = plt.cm.tab10(np.linspace(0, 1, world_size))
    for r in range(world_size):
        ax0.plot(
            steps,
            losses[r],
            "-",
            color=colors[r],
            alpha=0.8,
            label=f"Rank {r}",
        )
    ax0.plot(
        steps,
        mean_loss,
        "k--",
        linewidth=2.0,
        label="Mean over 4 ranks",
    )
    ax0.set_xlabel("Optimization step")
    ax0.set_ylabel("Loss")
    ax0.set_title("Task 2(a): per-rank minibatch loss and mean loss")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    # bottom: time breakdown (x: step, y: time)
    ax1.plot(steps, mean_time, "k--", linewidth=2.0, label="Total (fwd + bwd + comm)")
    ax1.plot(steps, mean_fwd, "-", color="steelblue", linewidth=2.0, label="Forward")
    ax1.plot(steps, mean_bwd, "-", color="tab:green", linewidth=2.0, label="Backward + opt")
    ax1.plot(steps, mean_comm, "-", color="darkorange", linewidth=2.0, label="Communication (grad sync)")
    ax1.set_xlabel("Optimization step")
    ax1.set_ylabel("Time per step (s)")
    ax1.set_title("Average compute vs communication time per step")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")

    global_mean_loss = float(losses_arr.mean())
    print(f"Wrote {args.out}")
    print(f"Global mean loss over all ranks and steps: {global_mean_loss:.6f}")
    print(f"Global mean total time (except first step): {mean_time[1:].mean():.6f} seconds")
    print(f"Global mean forward time (except first step): {mean_fwd[1:].mean():.6f} seconds")
    print(f"Global mean backward time (except first step): {mean_bwd[1:].mean():.6f} seconds")
    print(f"Global mean communication time (except first step): {mean_comm[1:].mean():.6f} seconds")


if __name__ == "__main__":
    main()

