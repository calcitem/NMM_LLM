"""tools/plot_specialist_training.py — Live training dashboard for three v2 specialists.

Reads train_log.jsonl from each specialist's checkpoint folder and plots:
  Row 1: malom_win_move_rate         (50-game smoothed)
  Row 2: heuristic_top1_rate  vs  policy_top1_rate  (50-game smoothed)
  Row 3: best_win_rate  vs  win_rate_200  (50-game smoothed)
  Row 4: sentinel_chosen_mean vs sentinel_mean  +  gap shaded
         (gap > 0 = model preferring sentinel-favoured moves)
  Row 5: update_policy_loss  vs  update_value_loss
         (should both decrease; flat/rising value_loss = baseline not learning)

One column per specialist (open / mid / end).
Refreshes every 20 minutes.

Usage:
    .venv/bin/python tools/plot_specialist_training.py
    .venv/bin/python tools/plot_specialist_training.py --interval 5   # minutes
    .venv/bin/python tools/plot_specialist_training.py --no-loop      # single render
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")          # works headless-ish; fall back to Qt if needed
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
CKPT_BASE = ROOT / "learned_ai" / "checkpoints" / "scaffolded"

SPECIALISTS = [
    ("Opening",    CKPT_BASE / "s_open_v2"),
    ("Midgame",    CKPT_BASE / "s_mid_v2"),
    ("Endgame",    CKPT_BASE / "s_end_v2"),
    ("Generalist", CKPT_BASE / "s_gen_v2"),
]

SMOOTH = 50   # rolling window


def _load(path: Path) -> list[dict]:
    log = path / "train_log.jsonl"
    if not log.exists():
        return []
    rows = []
    with open(log, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _smooth(values: list[float], window: int) -> np.ndarray:
    if not values:
        return np.array([])
    arr = np.array(values, dtype=float)
    if len(arr) < 2:
        return arr
    kernel = np.ones(window) / window
    # 'same' mode, then trim edges where the window is incomplete
    out = np.convolve(arr, kernel, mode="full")[: len(arr)]
    # Fix leading values: use expanding mean until we have `window` points
    for i in range(min(window - 1, len(arr))):
        out[i] = arr[: i + 1].mean()
    return out


def _get(rows: list[dict], key: str) -> tuple[list[int], list[float]]:
    xs, ys = [], []
    for r in rows:
        v = r.get(key)
        if v is not None:
            xs.append(r.get("game", len(xs)))
            ys.append(float(v))
    return xs, ys


_WIN_OUTCOME  =  1.5
_LOSS_OUTCOME = -1.0

def _get_draw_rate(rows: list[dict], window: int = SMOOTH) -> tuple[list[int], list[float]]:
    """Rolling draw rate derived from per-game outcome.

    Draws are anything that is not a win (1.5) or loss (-1.0).
    Previously used == 0.0 but draw penalties changed to -0.15/-0.25.
    """
    xs, ys = [], []
    for r in rows:
        outcome = r.get("outcome")
        if outcome is not None:
            v = float(outcome)
            xs.append(r.get("game", len(xs)))
            ys.append(1.0 if (v != _WIN_OUTCOME and v != _LOSS_OUTCOME) else 0.0)
    return xs, ys


def _plot_series(ax, xs, ys, label, color, window=SMOOTH, alpha_raw=0.15):
    if not ys:
        return
    smoothed = _smooth(ys, window)
    ax.plot(xs, ys, color=color, alpha=alpha_raw, linewidth=0.6)
    ax.plot(xs, smoothed, color=color, linewidth=1.6, label=label)


def draw(fig, axes):
    for col, (name, ckpt_dir) in enumerate(SPECIALISTS):
        rows = _load(ckpt_dir)

        ax_malom = axes[0][col]
        ax_top1  = axes[1][col]
        ax_wr    = axes[2][col]
        ax_sent  = axes[3][col]
        ax_loss  = axes[4][col]

        for ax in (ax_malom, ax_top1, ax_wr, ax_sent, ax_loss):
            ax.cla()

        n = len(rows)
        subtitle = f"{name}  (n={n})"

        # ── Row 0: malom_win_move_rate ────────────────────────────────────────
        ax_malom.set_title(subtitle, fontsize=9, pad=3)
        xs, ys = _get(rows, "malom_win_move_rate")
        _plot_series(ax_malom, xs, ys, "malom win-move rate", "#2196F3")
        xs_u, ys_u = _get(rows, "malom_unknown_rate")
        _plot_series(ax_malom, xs_u, ys_u, "unknown rate", "#9E9E9E")
        ax_malom.set_ylim(0, 1.05)
        ax_malom.legend(fontsize=6, loc="lower right")
        ax_malom.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # ── Row 1: heuristic_top1_rate vs policy_top1_rate ────────────────────
        xs_h, ys_h = _get(rows, "heuristic_top1_rate")
        xs_p, ys_p = _get(rows, "policy_top1_rate")
        _plot_series(ax_top1, xs_h, ys_h, "heuristic top-1", "#FF9800")
        _plot_series(ax_top1, xs_p, ys_p, "policy top-1",    "#4CAF50")
        ax_top1.set_ylim(0, 1.05)
        ax_top1.legend(fontsize=6, loc="lower right")
        ax_top1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # ── Row 2: best_win_rate vs win_rate_200 + draw rate ─────────────────
        xs_b, ys_b = _get(rows, "best_win_rate")
        xs_w, ys_w = _get(rows, "win_rate_200")
        xs_d, ys_d = _get_draw_rate(rows)
        _plot_series(ax_wr, xs_b, ys_b, "best win rate",  "#E91E63")
        _plot_series(ax_wr, xs_w, ys_w, "win rate 200",   "#9C27B0")
        _plot_series(ax_wr, xs_d, ys_d, "draw rate",      "#FF9800")
        ax_wr.set_ylim(0, 1.05)
        ax_wr.legend(fontsize=6, loc="lower right")
        ax_wr.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))

        # ── Row 3: sentinel chosen vs mean — gap shows sentinel signal usage ──
        xs_sc, ys_sc = _get(rows, "sentinel_chosen_mean")
        xs_sm, ys_sm = _get(rows, "sentinel_mean")
        _plot_series(ax_sent, xs_sc, ys_sc, "chosen sentinel", "#00BCD4", alpha_raw=0.12)
        _plot_series(ax_sent, xs_sm, ys_sm, "mean sentinel",   "#607D8B", alpha_raw=0.12)
        # Shade the gap where both series overlap
        if ys_sc and ys_sm:
            common_len = min(len(xs_sc), len(xs_sm))
            xs_c  = xs_sc[:common_len]
            sm_s  = _smooth(ys_sm[:common_len], SMOOTH)
            sc_s  = _smooth(ys_sc[:common_len], SMOOTH)
            ax_sent.fill_between(xs_c, sm_s, sc_s,
                                 where=(sc_s >= sm_s),
                                 alpha=0.25, color="#00BCD4", label="gap (chosen > mean)")
            ax_sent.fill_between(xs_c, sm_s, sc_s,
                                 where=(sc_s < sm_s),
                                 alpha=0.25, color="#FF5722")
        ax_sent.set_ylim(0, 1.05)
        ax_sent.set_xlabel("game", fontsize=7)
        ax_sent.legend(fontsize=6, loc="lower right")

        # ── Row 4: policy loss vs value loss ─────────────────────────────────
        xs_pl, ys_pl = _get(rows, "update_policy_loss")
        xs_vl, ys_vl = _get(rows, "update_value_loss")
        _plot_series(ax_loss, xs_pl, ys_pl, "policy loss",  "#E91E63", alpha_raw=0.20)
        _plot_series(ax_loss, xs_vl, ys_vl, "value loss",   "#FF9800", alpha_raw=0.20)
        ax_loss.set_xlabel("game", fontsize=7)
        ax_loss.legend(fontsize=6, loc="upper right")

        for ax in (ax_malom, ax_top1, ax_wr, ax_sent, ax_loss):
            ax.tick_params(labelsize=6)
            ax.grid(True, alpha=0.3, linewidth=0.4)

    row_labels = ["Malom win-move rate", "Top-1 agreement", "Win rates", "Sentinel signal", "Policy / value loss"]
    for row, label in enumerate(row_labels):
        axes[row][0].set_ylabel(f"{label}\n(score)", fontsize=7)

    fig.suptitle(
        f"Specialist training  ·  {SMOOTH}-game smoothed  ·  {time.strftime('%H:%M:%S')}",
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.canvas.draw()
    fig.canvas.flush_events()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=float, default=20.0,
                        help="Refresh interval in minutes (default 20)")
    parser.add_argument("--no-loop", action="store_true",
                        help="Render once and exit")
    args = parser.parse_args()

    fig, axes = plt.subplots(
        5, 4,
        figsize=(18, 15),
        sharex=False,
    )

    plt.ion()
    draw(fig, axes)

    if args.no_loop:
        plt.ioff()
        plt.show()
        return

    interval_s = args.interval * 60
    try:
        while True:
            deadline = time.time() + interval_s
            while time.time() < deadline:
                plt.pause(1.0)   # keeps the window responsive
            draw(fig, axes)
    except KeyboardInterrupt:
        pass

    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
