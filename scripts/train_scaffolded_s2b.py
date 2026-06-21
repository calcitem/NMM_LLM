"""scripts/train_scaffolded_s2b.py — Stage 2b: self-play with branched mid-game rollouts.

Extends s2_diagnostic with two additions:

1. SELF-PLAY: Half of main games pit the live model (temperature-sampled) against a
   periodically-frozen copy of itself.  The other half use the heuristic opponent from
   s2.  This provides trajectory diversity without the training-vs-inference gap of an
   undo mechanism.

2. BRANCHED ROLLOUTS: During every main game, the board state is snapshotted every
   BRANCH_EVERY learner turns.  After the main game ends, up to MAX_BRANCHES_PER_GAME
   of those snapshots are selected as starting points for fresh independent rollouts
   (model vs frozen copy).  Each branch is recorded as a completely separate trajectory
   — it never shares a gradient batch with the game it was spawned from, so there is no
   gradient contamination for shared positions.

GAME-STAGE DIVERSITY: Branch points are bucketed by phase:
   "opening"  — placement phase, fewer than 10 pieces placed in total
   "midgame"  — late placement or early movement (10+ placed, 12+ on board)
   "endgame"  — movement phase with fewer than 12 pieces on board

A rolling counter (BUCKET_WINDOW games) caps how many branches can come from any
single bucket (MAX_PER_BUCKET).  This prevents the training set from flooding with
one phase type while ensuring beginning, middle, and end-game positions all appear.

All other mechanics (reward shape, diagnostics, temperature schedule, LR backoff,
checkpoint logic) are identical to s2_diagnostic.

Checkpoints are saved to learned_ai/checkpoints/scaffolded/s2b/ by default.
Resume chain: explicit --resume → s2b/best.pt → s2/best.pt → s1b/best.pt → s1/best.pt

Usage
-----
# Quick smoke test (20 main games, no branches)
.venv/bin/python scripts/train_scaffolded_s2b.py --max-games 20 --max-branches-per-game 0

# Normal run from s2 checkpoint
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-s2

# Full run with PPO update
.venv/bin/python scripts/train_scaffolded_s2b.py --auto-resume-s2 --ppo --max-games 5000
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import sys
import time
from collections import deque, Counter
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.models.scaffolded_encoder import encode_position
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)

# ── Reward weights (same as s2) ───────────────────────────────────────────────

ALPHA   = 0.15   # sentinel relative score
BETA    = 0.10   # heuristic delta
GAMMA   = 0.25   # malom win quality
DELTA   = 0.15   # malom trap bonus
LAMBDA  = 0.50   # retro-active outcome weight
DECAY   = 0.98   # retro decay per ply remaining
VN_BETA = 0.10   # value-net delta

WIN_REWARD  =  1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.15   # draw in < 100 plies
DRAW_LONG   = -0.05   # draw by exhaustion

# ── Optimiser / schedule ──────────────────────────────────────────────────────

LR            = 1e-4
GAMMA_TD      = 0.99
TEMP_START    = 0.50
TEMP_MIN      = 0.45
TEMP_MAX      = 0.90
ENTROPY_COEF  = 0.01
UPDATE_EVERY  = 16
ROLLING_WIN   = 200
DIFF_START = 3    # already beating diff 3; begin here
DIFF_MAX   = 7
# Rolling win rate needed at each difficulty to advance to the next.
# Lower thresholds at higher difficulties — beating diff 6 at 40% is strong.
ADVANCE_THRESHOLDS = {3: 0.65, 4: 0.60, 5: 0.55, 6: 0.50}
EXIT_THRESHOLD = 0.40   # win rate vs diff 7 considered done
MAX_PLY       = 400
MAX_PLY_BRANCH = 250   # branch games start mid-game; cap shorter
TIME_BUDGET   = 0.05

LOG_EVERY         = 50
FREEZE_DROP       = 0.08
BACKOFF_DROP      = 0.12
LR_BACKOFF_FACTOR = 0.5
MIN_LR            = 1e-5
TEMP_RAMP_DENOM   = 4000

# ── s2b-specific knobs ────────────────────────────────────────────────────────

UPDATE_TARGET_EVERY    = 50    # games between frozen-model refreshes
SELF_PLAY_RATIO        = 0.5   # fraction of main games vs frozen model (rest vs heuristic)
BRANCH_EVERY           = 10    # save branch candidate every N learner moves in a main game
MAX_BRANCHES_PER_GAME  = 2     # max branch games spawned per main game
BUCKET_WINDOW          = 300   # rolling window size for saturation tracking
MAX_PER_BUCKET         = 80    # max branch games from any single bucket in that window

PHASE_BUCKETS = ("opening", "midgame", "endgame")


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class RewardBreakdown:
    total:      float = 0.0
    sentinel:   float = 0.0
    heuristic:  float = 0.0
    value_net:  float = 0.0
    malom_win:  float = 0.0
    malom_trap: float = 0.0
    retro:      float = 0.0


@dataclass
class StepDiag:
    reward:           RewardBreakdown
    legal_moves:      int
    chosen_idx:       int
    chosen_prob:      float
    entropy:          float
    top1_prob:        float
    sentinel_mean:    float
    sentinel_chosen:  float
    h_before:         float
    h_after:          float
    h_delta:          float
    vn_before:        float
    vn_after:         float
    vn_delta:         float
    malom_chosen_wdl: str
    malom_chosen_dtm: Optional[float]
    was_top1_policy:  int
    was_top1_heuristic: int


@dataclass
class GameDiag:
    game:                   int
    difficulty:             int
    learner_color:          str
    temperature:            float
    outcome:                float
    win_rate_200:           float
    ply:                    int
    steps:                  int
    update_policy_loss:     Optional[float]
    update_value_loss:      Optional[float]
    update_entropy:         Optional[float]
    reward_total_mean:      float
    reward_sentinel_mean:   float
    reward_heuristic_mean:  float
    reward_value_mean:      float
    reward_malom_win_mean:  float
    reward_malom_trap_mean: float
    reward_retro_mean:      float
    sentinel_mean:          float
    sentinel_chosen_mean:   float
    h_delta_mean:           float
    vn_delta_mean:          float
    chosen_prob_mean:       float
    entropy_mean:           float
    top1_prob_mean:         float
    legal_moves_mean:       float
    policy_top1_rate:       float
    heuristic_top1_rate:    float
    malom_win_move_rate:    float
    malom_unknown_rate:     float
    best_win_rate:          float
    temp_frozen:            int
    lr:                     float
    source_checkpoint:      str
    # s2b additions
    game_type:              str    # "vs_heuristic" | "vs_frozen" | "branch"
    phase_bucket:           str    # "opening" | "midgame" | "endgame" | "main"
    is_branch:              int    # 0 = main game, 1 = branch
    branch_ply_start:       int    # ply offset where branch was spawned (0 for main)
    target_age:             int    # games since last frozen-model update
    bucket_opening:         int    # current bucket counts in rolling window
    bucket_midgame:         int
    bucket_endgame:         int


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _safe_mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else 0.0


def _phase_bucket(board: BoardState) -> str:
    """Classify board into training phase bucket for saturation tracking."""
    total_placed = board.pieces_placed["W"] + board.pieces_placed["B"]
    total_on_board = board.pieces_on_board["W"] + board.pieces_on_board["B"]
    if board.phase == "place":
        return "opening" if total_placed < 10 else "midgame"
    # movement / fly phase
    return "endgame" if total_on_board < 12 else "midgame"


def _choose_resume_path(args: argparse.Namespace) -> tuple[Optional[Path], str]:
    if args.resume:
        p = Path(args.resume)
        if p.exists():
            return p, "explicit_resume"
    s2b_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b" / "best.pt"
    s2b_latest = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b" / "latest.pt"
    s2_best    = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2"  / "best.pt"
    s1b_best   = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1b" / "best.pt"
    s1_best    = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1"  / "best.pt"
    candidates = []
    if args.auto_resume_best:
        candidates.append((s2b_best,   "s2b_best"))
    if args.auto_resume_latest:
        candidates.append((s2b_latest, "s2b_latest"))
    if args.auto_resume_s2:
        candidates.append((s2_best,    "s2_best"))
    candidates += [(s1b_best, "s1b_best"), (s1_best, "s1_best")]
    for p, tag in candidates:
        if p.exists():
            return p, tag
    return None, "scratch"


def _load_model(device: torch.device, resume_path: Optional[Path]) -> tuple[ScaffoldedPolicyNet, int, float, int, str]:
    if resume_path is None:
        return ScaffoldedPolicyNet().to(device), 0, 0.0, DIFF_START, "scratch"
    ckpt = torch.load(resume_path, map_location=device, weights_only=False)
    cfg = ckpt.get("model_config", {})
    model = ScaffoldedPolicyNet.from_config(cfg).to(device)
    sd_key = "model" if "model" in ckpt else "state_dict"
    model.load_state_dict(ckpt[sd_key])
    start_game  = int(ckpt.get("game_count", 0))
    best_wr     = float(ckpt.get("best_win_rate", 0.0))
    difficulty  = int(ckpt.get("difficulty", DIFF_START))
    return model, start_game, best_wr, difficulty, str(resume_path)


def _compute_temperature(game_count: int, best_win_rate: float, current_win_rate: float, prev_temp: float) -> tuple[float, bool]:
    progress = min(1.0, game_count / max(TEMP_RAMP_DENOM, 1))
    target   = TEMP_START + (TEMP_MAX - TEMP_START) * progress
    target   = max(TEMP_MIN, min(TEMP_MAX, target))
    frozen   = False
    if best_win_rate > 0.0 and current_win_rate < best_win_rate - FREEZE_DROP:
        target = min(prev_temp, max(TEMP_START, prev_temp - 0.02))
        frozen = True
    return float(target), frozen


def _maybe_backoff_lr(opt: torch.optim.Optimizer, best_win_rate: float, current_win_rate: float) -> bool:
    if best_win_rate <= 0.0 or current_win_rate >= best_win_rate - BACKOFF_DROP:
        return False
    changed = False
    for g in opt.param_groups:
        old_lr = float(g["lr"])
        new_lr = max(MIN_LR, old_lr * LR_BACKOFF_FACTOR)
        if new_lr < old_lr:
            g["lr"] = new_lr
            changed = True
    return changed


def _compute_per_move_reward(enc, chosen_idx: int, enc_after, db_moves=None) -> tuple[float, RewardBreakdown, dict[str, Any]]:
    rb = RewardBreakdown()
    extra: dict[str, Any] = {"malom_chosen_wdl": "unknown", "malom_chosen_dtm": None}

    if getattr(enc, "sentinel_scores", None):
        mean_s   = float(sum(enc.sentinel_scores) / len(enc.sentinel_scores))
        played_s = float(enc.sentinel_scores[chosen_idx])
        rb.sentinel = ALPHA * (played_s - mean_s)

    if enc_after is not None:
        h_before = float(getattr(enc, "h_before", 0.0))
        h_after  = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
        rb.heuristic = BETA * math.tanh(h_after - h_before)

    if getattr(enc, "vn_scores_abs", None):
        vn_before = float(getattr(enc, "vn_before", 0.0))
        vn_after  = float(enc.vn_scores_abs[chosen_idx])
        rb.value_net = VN_BETA * math.tanh(vn_after - vn_before)

    if db_moves:
        mv_key   = _move_key(enc.legal_moves[chosen_idx])
        db_entry = next((m for m in db_moves if _move_key(m.get("move", {})) == mv_key), None)
        if db_entry:
            wdl = str(db_entry.get("wdl", "unknown"))
            dtm = db_entry.get("dtm")
            extra["malom_chosen_wdl"] = wdl
            extra["malom_chosen_dtm"] = dtm
            if wdl == "win":
                rb.malom_win = GAMMA * float(dtm_quality("win", dtm))

    rb.total = rb.sentinel + rb.heuristic + rb.value_net + rb.malom_win + rb.malom_trap + rb.retro
    return float(rb.total), rb, extra


def _retroactive_rescore(trajectory: list[ScaffoldedStep], step_diags: list[StepDiag], outcome: float) -> None:
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining  = n - t_idx - 1
        delta            = LAMBDA * outcome * (DECAY ** plies_remaining)
        step.reward     += delta
        step_diags[t_idx].reward.retro += float(delta)
        step_diags[t_idx].reward.total += float(delta)


# ── Frozen-model opponent ─────────────────────────────────────────────────────

class FrozenModelOpponent:
    """Plays argmax from a deep-copied, frozen snapshot of the live model."""

    def __init__(self, model: ScaffoldedPolicyNet, device: torch.device, sentinel=None, value_net=None):
        self._model     = copy.deepcopy(model).to(device)
        self._model.eval()
        self._device    = device
        self._sentinel  = sentinel
        self._value_net = value_net
        self.last_was_blunder = False
        self.last_thinking    = "frozen"

    def refresh(self, model: ScaffoldedPolicyNet) -> None:
        self._model.load_state_dict(copy.deepcopy(model).state_dict())
        self._model.eval()

    def choose_move(self, board: BoardState) -> dict:
        player = board.turn
        enc = encode_position(board, player,
                              sentinel_advisor=self._sentinel,
                              db=None,
                              value_net=self._value_net)
        if enc is None or not enc.legal_moves:
            return {}
        feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(self._device)
        with torch.no_grad():
            logits = self._model.policy_logits(feat_t)
            idx    = int(torch.argmax(logits).item())
        return enc.legal_moves[idx]


# ── Single-game rollout (shared by main and branch games) ─────────────────────

@dataclass
class RolloutResult:
    trajectory: list[ScaffoldedStep]
    step_diags: list[StepDiag]
    outcome:    float
    ply:        int
    branch_candidates: list[tuple[int, BoardState, str]]  # (ply, board, phase_bucket)


def _rollout(
    model:         ScaffoldedPolicyNet,
    device:        torch.device,
    start_board:   BoardState,
    learner_color: str,
    opponent,               # HeuristicAgent | FrozenModelOpponent
    opp_color:     str,
    sentinel,
    db,
    value_net,
    temperature:   float,
    max_ply:       int,
    record_branches: bool,
    branch_every:  int,
) -> RolloutResult:
    """
    Run a single game rollout from start_board.

    record_branches — if True, snapshot (ply, board, bucket) every branch_every
    learner turns for later branch-game spawning.
    """
    board              = start_board
    ply                = 0
    game_trajectory:   list[ScaffoldedStep] = []
    step_diags:        list[StepDiag]       = []
    branch_candidates: list[tuple[int, BoardState, str]] = []
    done               = False
    outcome            = 0.0
    learner_move_count = 0

    while ply < max_ply:
        terminal, winner = is_terminal(board)
        if terminal:
            if winner == learner_color:
                outcome = WIN_REWARD
            elif winner is not None:
                outcome = LOSS_REWARD
            else:
                outcome = DRAW_SHORT if ply < 100 else DRAW_LONG
            done = True
            break

        player = board.turn

        if player == learner_color:
            enc = encode_position(board, player, sentinel_advisor=sentinel, db=None, value_net=value_net)
            if enc is None or not enc.legal_moves:
                outcome = LOSS_REWARD
                done = True
                break

            feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
            with torch.no_grad():
                logits     = model.policy_logits(feat_t)
                scaled     = logits / max(temperature, 1e-6)
                log_probs  = F.log_softmax(scaled, dim=-1)
                probs      = log_probs.exp()
                if not torch.isfinite(probs).all():
                    probs  = torch.where(torch.isfinite(probs), probs, torch.zeros_like(probs))
                probs      = probs / probs.sum().clamp(min=1e-9)
                entropy    = float((-(probs * log_probs).sum()).item())
                chosen_idx = int(torch.multinomial(probs.cpu(), 1).item())
                chosen_prob = float(probs[chosen_idx].item())
                top1_prob  = float(probs.max().item())
                was_top1_policy = int(chosen_idx == int(torch.argmax(probs).item()))
                log_prob_old    = float(log_probs[chosen_idx].item())

            move       = enc.legal_moves[chosen_idx]
            board_after = board.apply_move(move)
            enc_after  = encode_position(board_after, opp_color, sentinel_advisor=sentinel, db=None, value_net=value_net)

            db_moves = []
            if db is not None:
                try:
                    db_moves = db.query_all_moves(board, player) or []
                except Exception:
                    pass

            reward, rb, extra = _compute_per_move_reward(enc, chosen_idx, enc_after, db_moves=db_moves)

            if db is not None:
                try:
                    opp_state_wdl = db.query_state(board_after)
                    if opp_state_wdl == "L":
                        reward      += DELTA
                        rb.malom_trap += DELTA
                        rb.total    += DELTA
                except Exception:
                    pass

            if enc_after is not None and enc_after.legal_moves:
                next_mf = enc_after.feat_matrix
                next_vi = enc_after.value_input
            else:
                next_mf = np.zeros((1, enc.feat_matrix.shape[1]), dtype=np.float32)
                next_vi = np.zeros(enc.value_input.shape, dtype=np.float32)

            terminal_next, _ = is_terminal(board_after)
            step = ScaffoldedStep(
                move_features=enc.feat_matrix,
                value_input=enc.value_input,
                chosen_idx=chosen_idx,
                log_prob_old=log_prob_old,
                reward=reward,
                next_move_features=next_mf,
                next_value_input=next_vi,
                done=terminal_next,
            )
            game_trajectory.append(step)

            sentinel_scores   = list(getattr(enc, "sentinel_scores", []) or [])
            sentinel_mean     = float(sum(sentinel_scores) / len(sentinel_scores)) if sentinel_scores else 0.0
            sentinel_chosen   = float(sentinel_scores[chosen_idx]) if sentinel_scores else 0.0
            h_before  = float(getattr(enc, "h_before", 0.0))
            h_after   = float(enc.h_scores_abs[chosen_idx]) if getattr(enc, "h_scores_abs", None) else h_before
            vn_before = float(getattr(enc, "vn_before", 0.0))
            vn_after  = float(enc.vn_scores_abs[chosen_idx]) if getattr(enc, "vn_scores_abs", None) else vn_before
            heuristic_top1 = 0
            if getattr(enc, "h_scores_abs", None):
                heuristic_top1 = int(chosen_idx == int(np.argmax(np.asarray(enc.h_scores_abs))))

            step_diags.append(StepDiag(
                reward=rb,
                legal_moves=len(enc.legal_moves),
                chosen_idx=chosen_idx,
                chosen_prob=chosen_prob,
                entropy=entropy,
                top1_prob=top1_prob,
                sentinel_mean=sentinel_mean,
                sentinel_chosen=sentinel_chosen,
                h_before=h_before,
                h_after=h_after,
                h_delta=h_after - h_before,
                vn_before=vn_before,
                vn_after=vn_after,
                vn_delta=vn_after - vn_before,
                malom_chosen_wdl=extra["malom_chosen_wdl"],
                malom_chosen_dtm=extra["malom_chosen_dtm"],
                was_top1_policy=was_top1_policy,
                was_top1_heuristic=heuristic_top1,
            ))

            # Record branch candidate every branch_every learner moves
            learner_move_count += 1
            if record_branches and branch_every > 0 and (learner_move_count % branch_every == 0):
                branch_candidates.append((ply, board, _phase_bucket(board)))

            board = board_after

        else:
            # Opponent's turn
            try:
                opp_move = opponent.choose_move(board)
            except Exception:
                opp_move = None
            if not opp_move:
                outcome = WIN_REWARD
                done    = True
                break
            board = board.apply_move(opp_move)

        ply += 1

    if not done:
        outcome = DRAW_LONG

    return RolloutResult(
        trajectory=game_trajectory,
        step_diags=step_diags,
        outcome=outcome,
        ply=ply,
        branch_candidates=branch_candidates,
    )


# ── Diagnostic logging ────────────────────────────────────────────────────────

def _build_game_diag(
    game_count:      int,
    difficulty:      int,
    learner_color:   str,
    temperature:     float,
    result:          RolloutResult,
    best_win_rate:   float,
    win_history:     deque,
    last_update_pl:  Optional[float],
    last_update_vl:  Optional[float],
    last_update_ent: Optional[float],
    opt:             torch.optim.Optimizer,
    temp_frozen:     bool,
    source_ckpt:     str,
    game_type:       str,
    phase_bucket:    str,
    is_branch:       bool,
    branch_ply_start: int,
    target_age:      int,
    bucket_counts:   Counter,
) -> GameDiag:
    sd = result.step_diags
    win_rate = sum(win_history) / max(len(win_history), 1)
    return GameDiag(
        game=game_count,
        difficulty=difficulty,
        learner_color=learner_color,
        temperature=round(temperature, 4),
        outcome=float(result.outcome),
        win_rate_200=round(win_rate, 4),
        ply=int(result.ply),
        steps=len(sd),
        update_policy_loss=None if last_update_pl is None else float(last_update_pl),
        update_value_loss=None if last_update_vl is None else float(last_update_vl),
        update_entropy=None if last_update_ent is None else float(last_update_ent),
        reward_total_mean=_safe_mean([d.reward.total for d in sd]),
        reward_sentinel_mean=_safe_mean([d.reward.sentinel for d in sd]),
        reward_heuristic_mean=_safe_mean([d.reward.heuristic for d in sd]),
        reward_value_mean=_safe_mean([d.reward.value_net for d in sd]),
        reward_malom_win_mean=_safe_mean([d.reward.malom_win for d in sd]),
        reward_malom_trap_mean=_safe_mean([d.reward.malom_trap for d in sd]),
        reward_retro_mean=_safe_mean([d.reward.retro for d in sd]),
        sentinel_mean=_safe_mean([d.sentinel_mean for d in sd]),
        sentinel_chosen_mean=_safe_mean([d.sentinel_chosen for d in sd]),
        h_delta_mean=_safe_mean([d.h_delta for d in sd]),
        vn_delta_mean=_safe_mean([d.vn_delta for d in sd]),
        chosen_prob_mean=_safe_mean([d.chosen_prob for d in sd]),
        entropy_mean=_safe_mean([d.entropy for d in sd]),
        top1_prob_mean=_safe_mean([d.top1_prob for d in sd]),
        legal_moves_mean=_safe_mean([float(d.legal_moves) for d in sd]),
        policy_top1_rate=_safe_mean([float(d.was_top1_policy) for d in sd]),
        heuristic_top1_rate=_safe_mean([float(d.was_top1_heuristic) for d in sd]),
        malom_win_move_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "win" else 0.0 for d in sd]),
        malom_unknown_rate=_safe_mean([1.0 if d.malom_chosen_wdl == "unknown" else 0.0 for d in sd]),
        best_win_rate=float(best_win_rate),
        temp_frozen=int(temp_frozen),
        lr=float(opt.param_groups[0]["lr"]),
        source_checkpoint=source_ckpt,
        game_type=game_type,
        phase_bucket=phase_bucket,
        is_branch=int(is_branch),
        branch_ply_start=branch_ply_start,
        target_age=target_age,
        bucket_opening=bucket_counts.get("opening", 0),
        bucket_midgame=bucket_counts.get("midgame", 0),
        bucket_endgame=bucket_counts.get("endgame", 0),
    )


# ── Main training loop ────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s2b] Device: {device}")
    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ── Load components ────────────────────────────────────────────────────────
    sentinel = None
    sent_path = args.sentinel or str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt")
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s2b] Sentinel loaded: {sent_path}")
        else:
            sentinel = None
    if sentinel is None:
        print("[s2b] Sentinel unavailable — sentinel reward = 0")

    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s2b] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s2b] Malom DB failed ({e})")
    if db is None:
        print("[s2b] Malom DB unavailable — Malom rewards = 0")

    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s2b] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s2b] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s2b] No value net — VN features will be 0")

    # ── Load model ─────────────────────────────────────────────────────────────
    resume_path, source_tag = _choose_resume_path(args)
    model, start_game, best_win_rate, difficulty, source_checkpoint = _load_model(device, resume_path)
    if resume_path is None:
        print("[s2b] No checkpoint found — starting from scratch")
    else:
        print(f"[s2b] Resuming from ({source_tag}): {resume_path}")

    # ── Frozen opponent (self-play target network) ─────────────────────────────
    frozen_opp = FrozenModelOpponent(model, device, sentinel=sentinel, value_net=value_net)
    games_since_target_update = 0

    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    opt       = torch.optim.Adam(model.parameters(), lr=args.lr)
    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    game_count          = start_game
    temperature         = args.temp_start
    win_history: deque[float] = deque(maxlen=args.rolling_win)
    ep_steps:  list[ScaffoldedStep] = []
    last_update_pl  = None
    last_update_vl  = None
    last_update_ent = None
    temp_frozen_last = False

    # Rolling bucket saturation tracker
    branch_bucket_history: deque[str] = deque(maxlen=args.bucket_window)

    log_path        = out_dir / "train_log.jsonl"
    update_log_path = out_dir / "update_log.jsonl"

    print(f"[s2b] Starting at game {game_count}, difficulty {difficulty}")
    print(f"[s2b] Self-play ratio {args.self_play_ratio:.0%}, "
          f"branch every {args.branch_every} turns, "
          f"max {args.max_branches_per_game} branches/game")

    diag_buffer: list[GameDiag] = []

    while game_count < args.max_games:
        current_wr = sum(win_history) / len(win_history) if win_history else 0.0
        temperature, temp_frozen_last = _compute_temperature(game_count, best_win_rate, current_wr, temperature)
        if _maybe_backoff_lr(opt, best_win_rate, current_wr):
            print(f"[s2b] LR backoff → lr={opt.param_groups[0]['lr']:.6g}")

        # Refresh frozen model periodically
        if games_since_target_update >= args.update_target_every:
            frozen_opp.refresh(model)
            games_since_target_update = 0
            print(f"[s2b] Frozen model updated at game {game_count}")

        learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color     = "B" if learner_color == "W" else "W"

        # Choose opponent type for this main game
        use_self_play = rng.random() < args.self_play_ratio
        if use_self_play:
            opponent = frozen_opp
            game_type = "vs_frozen"
        else:
            from learned_ai.agents.heuristic_agent import GameAI as _GA
            _h = HeuristicAgent(color=opp_color, difficulty=difficulty, game_ai=None)
            _h._inner = _GA(color=opp_color, difficulty=difficulty, override_time_budget=args.time_budget)
            opponent  = _h
            game_type = "vs_heuristic"

        # ── Main game rollout ──────────────────────────────────────────────────
        result = _rollout(
            model=model,
            device=device,
            start_board=BoardState.new_game(),
            learner_color=learner_color,
            opponent=opponent,
            opp_color=opp_color,
            sentinel=sentinel,
            db=db,
            value_net=value_net,
            temperature=temperature,
            max_ply=args.max_ply,
            record_branches=(args.max_branches_per_game > 0),
            branch_every=args.branch_every,
        )

        if result.trajectory:
            _retroactive_rescore(result.trajectory, result.step_diags, result.outcome)
        ep_steps.extend(result.trajectory)
        win_history.append(1.0 if result.outcome == WIN_REWARD else 0.0)
        game_count += 1
        games_since_target_update += 1

        bucket_counts = Counter(branch_bucket_history)
        diag_buffer.append(_build_game_diag(
            game_count, difficulty, learner_color, temperature, result,
            best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
            opt, temp_frozen_last, source_checkpoint,
            game_type=game_type, phase_bucket="main", is_branch=False,
            branch_ply_start=0, target_age=games_since_target_update,
            bucket_counts=bucket_counts,
        ))

        # ── Spawn branch games ─────────────────────────────────────────────────
        branches_spawned = 0
        # Shuffle candidates so we don't always pick early-game branches
        candidates = list(result.branch_candidates)
        rng.shuffle(candidates)
        # Try to pick one from each bucket first, then fill remaining slots
        seen_buckets: set[str] = set()
        ordered_candidates: list[tuple[int, BoardState, str]] = []
        for cand in candidates:
            if cand[2] not in seen_buckets:
                ordered_candidates.insert(0, cand)   # prioritise diverse buckets
                seen_buckets.add(cand[2])
            else:
                ordered_candidates.append(cand)

        for branch_ply, branch_board, bucket in ordered_candidates:
            if branches_spawned >= args.max_branches_per_game:
                break
            # Saturation check
            bucket_counts = Counter(branch_bucket_history)
            if bucket_counts.get(bucket, 0) >= args.max_per_bucket:
                continue

            # Branch game: model vs frozen copy from mid-game state
            # Learner color stays the same so reward signs are consistent
            branch_result = _rollout(
                model=model,
                device=device,
                start_board=branch_board,
                learner_color=learner_color,
                opponent=frozen_opp,
                opp_color=opp_color,
                sentinel=sentinel,
                db=db,
                value_net=value_net,
                temperature=temperature,
                max_ply=args.max_ply_branch,
                record_branches=False,   # no nested branching
                branch_every=0,
            )

            if branch_result.trajectory:
                _retroactive_rescore(branch_result.trajectory, branch_result.step_diags, branch_result.outcome)
                ep_steps.extend(branch_result.trajectory)
                branch_bucket_history.append(bucket)
                branches_spawned += 1
                game_count += 1
                games_since_target_update += 1
                win_history.append(1.0 if branch_result.outcome == WIN_REWARD else 0.0)

                bucket_counts = Counter(branch_bucket_history)
                diag_buffer.append(_build_game_diag(
                    game_count, difficulty, learner_color, temperature, branch_result,
                    best_win_rate, win_history, last_update_pl, last_update_vl, last_update_ent,
                    opt, temp_frozen_last, source_checkpoint,
                    game_type="branch", phase_bucket=bucket, is_branch=True,
                    branch_ply_start=branch_ply, target_age=games_since_target_update,
                    bucket_counts=bucket_counts,
                ))

        # ── Update ────────────────────────────────────────────────────────────
        if len(ep_steps) >= args.update_every:
            last_update_pl, last_update_vl, last_update_ent = update_fn(
                model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef
            )
            upd_entry = {
                "game":         game_count,
                "policy_loss":  None if last_update_pl  is None else float(last_update_pl),
                "value_loss":   None if last_update_vl  is None else float(last_update_vl),
                "entropy":      None if last_update_ent is None else float(last_update_ent),
                "lr":           float(opt.param_groups[0]["lr"]),
                "batch_steps":  len(ep_steps),
            }
            with open(update_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(upd_entry) + "\n")
            ep_steps.clear()

        # ── Periodic log + checkpoint ──────────────────────────────────────────
        if game_count % args.log_every == 0 and diag_buffer:
            win_rate     = sum(win_history) / max(len(win_history), 1)
            main_diags   = [d for d in diag_buffer if not d.is_branch]
            branch_diags = [d for d in diag_buffer if d.is_branch]
            bc = Counter(branch_bucket_history)

            # Write JSONL
            with open(log_path, "a", encoding="utf-8") as f:
                for d in diag_buffer:
                    f.write(json.dumps(asdict(d)) + "\n")
            diag_buffer.clear()

            # Console summary
            all_sd = []  # aggregate step diags from last window (skip — use last main game diag)
            last_main = next((d for d in reversed(main_diags) if main_diags), None)
            if last_main:
                print(
                    f"[s2b] game {game_count:6d} | diff {difficulty} | win-200={win_rate:.3f} | "
                    f"temp={temperature:.2f}{'F' if temp_frozen_last else ' '} | "
                    f"lr={opt.param_groups[0]['lr']:.5f} | "
                    f"branches={len(branch_diags)} "
                    f"[op={bc.get('opening',0)} mid={bc.get('midgame',0)} end={bc.get('endgame',0)}]"
                )

            ckpt = {
                "model":             model.state_dict(),
                "model_config":      model.get_config(),
                "stage":             "s2b",
                "game_count":        game_count,
                "best_win_rate":     best_win_rate,
                "difficulty":        difficulty,
                "source_checkpoint": source_checkpoint,
                "lr":                float(opt.param_groups[0]["lr"]),
                "temperature":       float(temperature),
            }
            torch.save(ckpt, out_dir / "latest.pt")

            if win_rate > best_win_rate and len(win_history) >= min(100, args.rolling_win):
                best_win_rate        = win_rate
                ckpt["best_win_rate"] = best_win_rate
                torch.save(ckpt, out_dir / "best.pt")
                print(f"[s2b]  → best win rate: {best_win_rate:.3f}")

        # ── Difficulty advancement ─────────────────────────────────────────────
        if len(win_history) >= args.rolling_win:
            win_rate = sum(win_history) / len(win_history)
            advance_thr = ADVANCE_THRESHOLDS.get(difficulty, args.advance_threshold)
            if difficulty >= args.diff_max:
                if win_rate >= args.exit_threshold:
                    print(f"[s2b] *** {win_rate:.3f} win rate vs difficulty {difficulty} — done! ***")
                    break
            elif win_rate >= advance_thr:
                difficulty += 1
                win_history.clear()
                print(f"[s2b] *** Advanced to difficulty {difficulty} (win rate was {win_rate:.3f}) ***")

    # ── Final flush ───────────────────────────────────────────────────────────
    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=args.gamma_td, entropy_coef=args.entropy_coef)
    if diag_buffer:
        with open(log_path, "a", encoding="utf-8") as f:
            for d in diag_buffer:
                f.write(json.dumps(asdict(d)) + "\n")

    ckpt = {
        "model":             model.state_dict(),
        "model_config":      model.get_config(),
        "stage":             "s2b",
        "game_count":        game_count,
        "best_win_rate":     best_win_rate,
        "difficulty":        difficulty,
        "source_checkpoint": source_checkpoint,
        "lr":                float(opt.param_groups[0]["lr"]),
        "temperature":       float(temperature),
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s2b] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s2b] Checkpoint: {out_dir / 'best.pt'}")
    print(f"[s2b] Logs: {log_path} and {update_log_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2b: self-play + branched mid-game rollouts")
    p.add_argument("--resume",              default="",    type=str, help="Explicit checkpoint path")
    p.add_argument("--auto-resume-best",    action="store_true", help="Prefer s2b/best.pt in resume chain")
    p.add_argument("--auto-resume-latest",  action="store_true", help="Prefer s2b/latest.pt in resume chain")
    p.add_argument("--auto-resume-s2",      action="store_true", help="Start from s2/best.pt")
    p.add_argument("--out-dir",  default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b"))
    p.add_argument("--sentinel", default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"))
    p.add_argument("--malom",    default="", type=str)
    p.add_argument("--value-net",default=str(_ROOT / "data" / "value_net.npz"), type=str)
    p.add_argument("--ppo",      action="store_true")
    p.add_argument("--max-games",           type=int,   default=5000)
    p.add_argument("--seed",                type=int,   default=42)
    p.add_argument("--lr",                  type=float, default=LR)
    p.add_argument("--gamma-td",            type=float, default=GAMMA_TD)
    p.add_argument("--entropy-coef",        type=float, default=ENTROPY_COEF)
    p.add_argument("--update-every",        type=int,   default=UPDATE_EVERY)
    p.add_argument("--rolling-win",         type=int,   default=ROLLING_WIN)
    p.add_argument("--diff-max",            type=int,   default=DIFF_MAX,
                   help="Highest difficulty to train against (default 7)")
    p.add_argument("--advance-threshold",   type=float, default=0.50,
                   help="Fallback win rate to advance difficulty (per-level defaults in ADVANCE_THRESHOLDS)")
    p.add_argument("--exit-threshold",      type=float, default=EXIT_THRESHOLD,
                   help="Win rate vs diff-max considered done (default 0.30)")
    p.add_argument("--temp-start",          type=float, default=TEMP_START)
    p.add_argument("--log-every",           type=int,   default=LOG_EVERY)
    p.add_argument("--max-ply",             type=int,   default=MAX_PLY)
    p.add_argument("--max-ply-branch",      type=int,   default=MAX_PLY_BRANCH)
    p.add_argument("--time-budget",         type=float, default=TIME_BUDGET)
    # s2b-specific
    p.add_argument("--self-play-ratio",     type=float, default=SELF_PLAY_RATIO,
                   help="Fraction of main games vs frozen model (default 0.5)")
    p.add_argument("--update-target-every", type=int,   default=UPDATE_TARGET_EVERY,
                   help="Games between frozen-model refreshes (default 50)")
    p.add_argument("--branch-every",        type=int,   default=BRANCH_EVERY,
                   help="Snapshot branch candidate every N learner moves (default 10)")
    p.add_argument("--max-branches-per-game", type=int, default=MAX_BRANCHES_PER_GAME,
                   help="Max branch games spawned per main game (default 2; 0 disables)")
    p.add_argument("--bucket-window",       type=int,   default=BUCKET_WINDOW,
                   help="Rolling window for bucket saturation (default 300)")
    p.add_argument("--max-per-bucket",      type=int,   default=MAX_PER_BUCKET,
                   help="Max branch games from any bucket in window (default 80)")
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
