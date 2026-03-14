#!/usr/bin/env python3
"""
Plot Task 1 training loss per step, colored by epoch.
Uses embedded RTE run data, or parse a log with "Minibatch N loss: X" lines.

Usage:
  python3 plot_task1_loss.py
  python3 plot_task1_loss.py --log /path/to/task1.log --out loss.png
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Minibatch losses: 3 epochs x 39 steps (RTE, batch 64, bert-base-cased)
EPOCH_LOSSES = [0.7691709399223328, 0.7817338109016418, 0.6885838508605957, 0.7662752866744995, 0.7341869473457336, 0.6909624338150024, 0.6966149210929871, 0.7157525420188904, 0.7121194005012512, 0.7034932971000671, 0.7326527237892151, 0.6874642968177795, 0.7637413144111633, 0.7230794429779053, 0.7237532734870911, 0.6886762976646423, 0.6889545321464539, 0.6982187628746033, 0.6667585968971252, 0.6908092498779297, 0.7104145288467407, 0.7026562094688416, 0.6937975287437439, 0.6927496194839478, 0.7211198806762695, 0.7006632089614868, 0.7189897298812866, 0.6870840787887573, 0.7022961974143982, 0.7162225246429443, 0.6898196935653687, 0.7176743745803833, 0.7306923866271973, 0.6642029881477356, 0.6779599189758301, 0.6968522667884827, 0.690699577331543, 0.6722519993782043, 0.6678068041801453, 0.6644679307937622, 0.6756858229637146, 0.6787503361701965, 0.63865727186203, 0.6459658741950989, 0.6789073944091797, 0.6684646010398865, 0.6727049946784973, 0.6336624622344971, 0.6517199277877808, 0.657520055770874, 0.6750134229660034, 0.6474744081497192, 0.6285075545310974, 0.6034400463104248, 0.6607325077056885, 0.6847931146621704, 0.6960575580596924, 0.6231574416160583, 0.6709082126617432, 0.5783560872077942, 0.6403183341026306, 0.6562061309814453, 0.614274263381958, 0.6420649886131287, 0.6628763675689697, 0.6224581599235535, 0.6251125931739807, 0.6328685283660889, 0.6301849484443665, 0.6000886559486389, 0.569029688835144, 0.6436809301376343, 0.5598978996276855, 0.5408304333686829, 0.7601435780525208, 0.6590582728385925, 0.5825825929641724, 0.6008194088935852, 0.6291962265968323, 0.5876449942588806, 0.5566092729568481, 0.5800958871841431, 0.5509494543075562, 0.5781617164611816, 0.5673123002052307, 0.5689939856529236, 0.5461385250091553, 0.6265420913696289, 0.5412376523017883, 0.6300002336502075, 0.5234324932098389, 0.5420375466346741, 0.5733460187911987, 0.5417898893356323, 0.49535897374153137, 0.5823370814323425, 0.5453874468803406, 0.5494118928909302, 0.5505642890930176, 0.5734802484512329, 0.6134501695632935, 0.5761626958847046, 0.4549552798271179, 0.5112193822860718, 0.5078673362731934, 0.4595223665237427, 0.5613539814949036, 0.49411115050315857, 0.6303368806838989, 0.6403030157089233, 0.5234085321426392, 0.5231107473373413, 0.5916783213615417, 0.6037144660949707, 0.6672989130020142, 0.5231905579566956, 0.5975736975669861]
LIST_LEN=len(EPOCH_LOSSES)
assert LIST_LEN%3==0
LEN4EPOCH=LIST_LEN//3
EPOCH_LOSSES = [EPOCH_LOSSES[:LEN4EPOCH], EPOCH_LOSSES[LEN4EPOCH:2*LEN4EPOCH], EPOCH_LOSSES[2*LEN4EPOCH:]]
DEV_ACC = [0.628158844765343, 0.6498194945848376, 0.6209386281588448]
MEAN_LOSS_ALL = 0.6365272132759421


def parse_log(path: Path) -> list[list[float]]:
    """Group losses by epoch (new epoch when Minibatch 0 appears after non-empty cur)."""
    text = path.read_text(errors="ignore")
    epochs: list[list[float]] = []
    cur: list[float] = []
    for m in re.finditer(r"Minibatch\s+(\d+)\s+loss:\s*([0-9.eE+-]+)", text):
        mb = int(m.group(1))
        val = float(m.group(2))
        if mb == 0 and cur:
            epochs.append(cur)
            cur = []
        cur.append(val)
    if cur:
        epochs.append(cur)
    return epochs


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot Task 1 loss per epoch / step")
    ap.add_argument("--log", type=Path, help="training log with Minibatch N loss: lines")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "task1_loss.png",
    )
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if args.log and args.log.exists():
        epoch_losses = parse_log(args.log)
        if not epoch_losses:
            print("No losses in log; using embedded data")
            epoch_losses = EPOCH_LOSSES
    else:
        epoch_losses = EPOCH_LOSSES

    n_epochs = len(epoch_losses)
    dev_acc = DEV_ACC[:n_epochs]
    if len(dev_acc) < n_epochs:
        dev_acc = list(dev_acc) + [dev_acc[-1]] * (n_epochs - len(dev_acc))

    fig, axes = plt.subplots(2, 1, figsize=(9, 7), gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    colors = plt.cm.viridis(np.linspace(0.2, 0.85, n_epochs))
    global_step = 0
    for e, losses in enumerate(epoch_losses, start=1):
        steps = np.arange(global_step, global_step + len(losses))
        ax0.plot(
            steps,
            losses,
            "o-",
            color=colors[e - 1],
            label=f"Epoch {e} (mean={np.mean(losses):.4f})",
            ms=3,
        )
        global_step += len(losses)
    ax0.set_xlabel("Optimization step")
    ax0.set_ylabel("Loss")
    ax0.set_title("Task 1: RTE fine-tune loss (BERT-base, batch 64)")
    ax0.legend(loc="upper right", fontsize=8)
    ax0.grid(True, alpha=0.3)

    epochs_x = np.arange(1, n_epochs + 1)
    means = [float(np.mean(L)) for L in epoch_losses]
    w = 0.35
    ax1.bar(epochs_x - w / 2, means, width=w, label="Mean train loss", color="steelblue")
    ax1_twin = ax1.twinx()
    ax1_twin.plot(epochs_x, dev_acc[:n_epochs], "s-", color="darkorange", lw=2, ms=8, label="Dev accuracy")
    ax1.set_xticks(epochs_x)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Mean training loss")
    ax1_twin.set_ylabel("Dev accuracy")
    ax1.set_title("Per-epoch mean loss vs dev accuracy")
    ax1.legend(loc="upper left")
    ax1_twin.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"Wrote {args.out}")
    print(f"Global mean loss (reference): {MEAN_LOSS_ALL:.6f}")


if __name__ == "__main__":
    main()
