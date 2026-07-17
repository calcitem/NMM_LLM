"""learned_ai/training/advance_stats.py — Sanmill-style advance criterion.

Mirrors the standard chess-testing statistic used in Sanmill's
`crates/tgf-cli/tests/head_to_head.rs` (lines 873–1020):

    p  = (W + 0.5·D) / (W + D + L)          # Score%
    SE = sqrt(p·(1-p) / n)                   # standard error of p
    z  = (target − p) / SE
    P(true score > target) = 1 − Φ(z)        # Φ = standard normal CDF

Advance decision:
  * `check_advance(...)` returns True when P(true score > target) ≥ 0.95
    on the rolling-window sample.

Target schedule (per-level):
  * Base ramp: 55% at level 1 → 60% at level 20.
  * Time-of-flight relaxation: after 1,000 games at a level with no advance,
    drop target 1% per additional 1,000 games until floor = 51%.  Matches the
    intuition "if we've been stuck at slightly-above-50% for a long time, that
    accumulated evidence is itself a small advantage worth advancing on".

Draws count as 0.5 wins inside p — a draw-heavy 60/20/20 (D/W/L) run gives
p = 0.5 and P(true > 55%) ≈ 0, so cannot advance.

Recovery (optional):
  * `check_recovery(...)` returns True when P(true score > 0.45) < 0.05
    (confidently WORSE than 45%) — reload best checkpoint pattern.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


# ── Standard-normal CDF via Abramowitz–Stegun 26.2.17 (matches Sanmill) ──────

def phi(x: float) -> float:
    """Standard normal CDF, Abramowitz-Stegun 26.2.17. Max abs error ~7.5e-8."""
    if x == 0.0:
        return 0.5
    sign = 1.0 if x >= 0.0 else -1.0
    ax = abs(x)
    # 26.2.17 constants
    p_a = 0.2316419
    b1 =  0.319381530
    b2 = -0.356563782
    b3 =  1.781477937
    b4 = -1.821255978
    b5 =  1.330274429
    t = 1.0 / (1.0 + p_a * ax)
    pdf = math.exp(-0.5 * ax * ax) / math.sqrt(2.0 * math.pi)
    poly = t * (b1 + t * (b2 + t * (b3 + t * (b4 + t * b5))))
    q = pdf * poly   # 1 - Φ(|x|)
    cdf = 1.0 - q
    return cdf if sign > 0 else 1.0 - cdf


# ── Sanmill statistic ─────────────────────────────────────────────────────────

def score_proportion(wins: int, draws: int, losses: int) -> float:
    """Score% — draws count 0.5.  Returns 0.5 when n == 0 (neutral prior)."""
    n = wins + draws + losses
    if n <= 0:
        return 0.5
    return (wins + 0.5 * draws) / n


def superiority_probability(wins: int, draws: int, losses: int, target: float) -> float:
    """P(true underlying Score% > target) via normal approximation."""
    n = wins + draws + losses
    if n <= 0:
        return 0.0
    p = score_proportion(wins, draws, losses)
    # Degenerate cases
    if p >= 1.0:
        return 1.0 if target < 1.0 else 0.0
    if p <= 0.0:
        return 0.0 if target > 0.0 else 1.0
    se = math.sqrt(p * (1.0 - p) / n)
    if se <= 0.0:
        return 1.0 if p > target else 0.0
    z = (target - p) / se
    return max(0.0, min(1.0, 1.0 - phi(z)))


# ── Target schedule with time-of-flight relaxation ────────────────────────────

BASE_TARGET_LOW   = 0.55   # target at level 1
BASE_TARGET_HIGH  = 0.60   # target at level 20
RELAX_START_AFTER = 1000   # games at a level before relaxation kicks in
RELAX_STEP        = 0.01   # 1% off per RELAX_STEP_GAMES beyond RELAX_START_AFTER
RELAX_STEP_GAMES  = 1000   # each additional 1000 games drops target 1%
RELAX_FLOOR       = 0.51   # never go below this


def advance_target(level: int,
                   games_at_level: int,
                   base_low: float = BASE_TARGET_LOW,
                   base_high: float = BASE_TARGET_HIGH,
                   floor: float = RELAX_FLOOR) -> float:
    """Effective target% for this level.

    Base: linear ramp from ``base_low`` at level 1 to ``base_high`` at level 20.
    Relaxation: after ``RELAX_START_AFTER`` games at the level, subtract
    ``RELAX_STEP`` per additional ``RELAX_STEP_GAMES`` games, floored at ``floor``.
    """
    # Base ramp — same shape as the old _check_advance.
    denom = 19.0
    frac = max(0, min(19, level - 1)) / denom
    base = base_low + (base_high - base_low) * frac
    if games_at_level <= RELAX_START_AFTER:
        return base
    excess = (games_at_level - RELAX_START_AFTER) // RELAX_STEP_GAMES
    relaxed = base - excess * RELAX_STEP
    return max(floor, relaxed)


# ── Advance / recovery checks ─────────────────────────────────────────────────

MIN_GAMES_FOR_ADVANCE  = 20   # window size floor for a decision
DEFAULT_PROB_THRESHOLD = 0.95   # confidence gate


@dataclass
class AdvanceDiag:
    """Explains why check_advance returned what it did."""
    should_advance: bool
    n:              int
    wins:           int
    draws:          int
    losses:         int
    score_pct:      float   # p
    target:         float   # target used
    p_super:        float   # P(true > target)
    reason:         str


def check_advance(win_history: deque,
                  difficulty: int,
                  games_at_level: int,
                  min_games: int = MIN_GAMES_FOR_ADVANCE,
                  prob_threshold: float = DEFAULT_PROB_THRESHOLD) -> AdvanceDiag:
    """Return diagnostic including advance decision.

    ``win_history`` — deque of per-game outcomes as floats:
        1.0 → win, 0.5 → draw, 0.0 → loss.
    ``difficulty`` — current level (1..DIFF_MAX).
    ``games_at_level`` — games played at this level (drives relaxation).
    """
    n = len(win_history)
    if n < min_games:
        return AdvanceDiag(False, n, 0, 0, 0, 0.5, 0.0, 0.0,
                           f"window too small ({n} < {min_games})")
    wins   = sum(1 for x in win_history if x == 1.0)
    draws  = sum(1 for x in win_history if x == 0.5)
    losses = sum(1 for x in win_history if x == 0.0)
    p      = score_proportion(wins, draws, losses)
    target = advance_target(difficulty, games_at_level)
    p_super = superiority_probability(wins, draws, losses, target)
    ok = p_super >= prob_threshold
    reason = (f"P={p_super:.3f} {'≥' if ok else '<'} {prob_threshold:.2f} "
              f"(target={target:.3f}, score={p:.3f})")
    return AdvanceDiag(ok, n, wins, draws, losses, p, target, p_super, reason)


RECOVERY_TARGET    = 0.45
RECOVERY_PROB_HI   = 0.95   # 1 - 0.05 → confidently below the target
MIN_GAMES_RECOVERY = 30


def check_recovery(win_history: deque,
                   min_games: int = MIN_GAMES_RECOVERY,
                   recovery_target: float = RECOVERY_TARGET,
                   prob_confidence: float = RECOVERY_PROB_HI) -> bool:
    """Return True when we are confidently WORSE than ``recovery_target`` (0.45).

    Equivalent to: P(true score ≤ 0.45) ≥ 0.95 → reload best checkpoint.
    """
    n = len(win_history)
    if n < min_games:
        return False
    wins   = sum(1 for x in win_history if x == 1.0)
    draws  = sum(1 for x in win_history if x == 0.5)
    losses = sum(1 for x in win_history if x == 0.0)
    # P(true > target) — if this is < (1 - prob_confidence) = 0.05, we are
    # confidently worse than the target.
    p_super = superiority_probability(wins, draws, losses, recovery_target)
    return p_super < (1.0 - prob_confidence)
