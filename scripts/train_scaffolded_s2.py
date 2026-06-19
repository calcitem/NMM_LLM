"""scripts/train_scaffolded_s2.py — Stage 2: A2C with scaffolded inputs.

Trains ScaffoldedPolicyNet via A2C (or PPO) against the heuristic engine,
using a rich per-move reward structure:

Per-move shaped rewards (computed each learner turn):
  r_sentinel   = ALPHA * (sentinel_score_played - sentinel_mean)
                 Did we play above the average sentinel quality?
  r_heuristic  = BETA  * tanh(h_after - h_before)
                 Did the heuristic evaluation improve after our move?
  r_malom_win  = GAMMA * dtm_quality(move)   if Malom says move is winning
                 Winning Malom moves — more reward for shorter distance-to-win
  r_malom_trap = DELTA                       if resulting opp position is Malom "loss"
                 Opponent is now provably losing — big signal

Game-level retroactive rescoring (after game ends):
  Each move in the trajectory receives += LAMBDA * outcome * decay^(plies_remaining)
  where outcome = +1.0 (win) | -1.0 (loss) | +0.15 (short draw) | -0.05 (long draw)

This retroactive pass ensures a winning game's early moves get partial credit
for the eventual win without overriding the dense per-move signal.

Curriculum:
  Start vs diff 2; advance to diff 3 when rolling-200 win rate >= WIN_TARGET_1
  Exit when rolling-200 win rate >= WIN_TARGET_2 at diff 3

Usage
-----
    .venv/bin/python scripts/train_scaffolded_s2.py [options]

Options
-------
  --resume   PATH    Resume from a Stage 1 or Stage 2 checkpoint
  --out-dir  DIR     Checkpoint directory (default learned_ai/checkpoints/scaffolded/s2)
  --sentinel PATH    SentinelAdvisor checkpoint
  --malom    PATH    Malom DB directory (optional but recommended)
  --ppo              Use PPO instead of A2C
  --max-games N      Maximum number of games (default 10000)
  --seed      N      Random seed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import numpy as np
import torch

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.scaffolded_agent import ScaffoldedAgent
from learned_ai.models.scaffolded_encoder import encode_position
from learned_ai.models.scaffolded_net import ScaffoldedPolicyNet
from learned_ai.sentinel.infer import load_advisor
from learned_ai.sentinel.labels import dtm_quality
from learned_ai.training.scaffolded_a2c import (
    ScaffoldedStep,
    scaffolded_a2c_update,
    scaffolded_ppo_update,
)

# ── Reward hyperparameters ─────────────────────────────────────────────────────
ALPHA        = 0.15   # per-move: sentinel above-mean bonus
BETA         = 0.10   # per-move: heuristic improvement bonus
GAMMA        = 0.25   # per-move: Malom winning-move bonus
DELTA        = 0.15   # per-move: Malom trap reward (opp now loses)
LAMBDA       = 0.50   # game-level retroactive scale
DECAY        = 0.98   # retroactive reward decay (later moves get more credit)

WIN_REWARD   = 1.0
LOSS_REWARD  = -1.0
DRAW_SHORT   = 0.15   # draw in < 100 plies — close but not ideal
DRAW_LONG    = -0.05  # draw after 100+ plies

# ── Training hyperparameters ───────────────────────────────────────────────────
LR             = 1e-4
GAMMA_TD       = 0.99
TEMP_START     = 0.5
TEMP_END       = 1.2
ENTROPY_COEF   = 0.01
UPDATE_EVERY   = 16
MIN_BATCH      = 8
ROLLING_WIN    = 200
WIN_TARGET_1   = 0.60   # advance from diff 2 → diff 3
WIN_TARGET_2   = 0.60   # exit at diff 3
DIFF_START     = 2
DIFF_TARGET    = 3
MAX_PLY        = 400
TIME_BUDGET    = 0.05

MALOM_WDL_MAP  = {"win": 1.0, "draw": 0.5, "loss": 0.0}


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _compute_per_move_reward(
    enc,
    chosen_idx: int,
    enc_after,
    db,
    db_moves_after=None,
) -> float:
    """Compute the shaped per-move reward."""
    r = 0.0

    # Sentinel component: above-mean quality?
    if enc.sentinel_scores:
        mean_s = sum(enc.sentinel_scores) / len(enc.sentinel_scores)
        played_s = enc.sentinel_scores[chosen_idx]
        r += ALPHA * (played_s - mean_s)

    # Heuristic component: position improved?
    if enc_after is not None:
        h_delta = enc_after.h_before - enc.h_scores_abs[chosen_idx]
        # h_scores_abs[i] is h_after for move i; enc_after.h_before is the same thing
        h_delta_actual = enc.h_scores_abs[chosen_idx] - enc.h_before
        r += BETA * math.tanh(h_delta_actual)

    # Malom components
    if enc.db_moves:
        mv_key = _move_key(enc.legal_moves[chosen_idx])
        db_entry = next(
            (m for m in enc.db_moves if _move_key(m.get("move", {})) == mv_key),
            None,
        )
        if db_entry:
            wdl = db_entry.get("wdl", "unknown")
            dtm = db_entry.get("dtm")
            if wdl == "win":
                r += GAMMA * float(dtm_quality("win", dtm))

    # Trap reward: is the resulting position a Malom loss for the opponent?
    if db_moves_after is not None:
        # After our move, the board is enc_after; query the opponent's WDL
        # db_moves_after is already from the opponent's perspective
        # query_state returns "W"|"L"|"D" for the side to move in that pos
        pass  # populated by caller when available

    return float(r)


def _retroactive_rescore(
    trajectory: list[ScaffoldedStep],
    outcome: float,
    game_ply: int,
) -> list[ScaffoldedStep]:
    """Add game-outcome signal to all steps, decaying from the end."""
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining = n - t_idx - 1
        delta = LAMBDA * outcome * (DECAY ** plies_remaining)
        # Modify in-place (dataclass fields are mutable)
        step.reward += delta
    return trajectory


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s2] Device: {device}")

    rng = random.Random(args.seed)

    # ── Sentinel ───────────────────────────────────────────────────────────────
    sentinel = None
    sent_path = args.sentinel or str(
        _ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
    )
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s2] Sentinel loaded: {sent_path}")
        else:
            sentinel = None
    if sentinel is None:
        print("[s2] Sentinel unavailable — sentinel reward = 0")

    # ── Malom DB ───────────────────────────────────────────────────────────────
    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s2] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s2] Malom DB failed ({e})")
    if db is None:
        print("[s2] Malom DB unavailable — Malom rewards = 0")

    # ── Model ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and Path(args.resume).exists():
        print(f"[s2] Resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        cfg  = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg).to(device)
        sd_key = "model" if "model" in ckpt else "state_dict"
        model.load_state_dict(ckpt[sd_key])
        start_game = ckpt.get("game_count", 0)
    else:
        s1_path = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s1" / "best.pt"
        if s1_path.exists():
            print(f"[s2] Starting from Stage 1 checkpoint: {s1_path}")
            ckpt  = torch.load(s1_path, map_location=device, weights_only=False)
            cfg   = ckpt.get("model_config", {})
            model = ScaffoldedPolicyNet.from_config(cfg).to(device)
            sd_key = "model" if "model" in ckpt else "state_dict"
            model.load_state_dict(ckpt[sd_key])
        else:
            print("[s2] No Stage 1 checkpoint found — starting from scratch")
            model = ScaffoldedPolicyNet().to(device)
        start_game = 0

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Training state ────────────────────────────────────────────────────────
    difficulty      = DIFF_START
    temperature     = TEMP_START
    win_history:    deque[float] = deque(maxlen=ROLLING_WIN)
    ep_steps:       list[ScaffoldedStep] = []
    game_count      = start_game
    best_win_rate   = 0.0
    log_path        = out_dir / "train_log.jsonl"

    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    print(f"[s2] Starting at game {game_count}, difficulty {difficulty}")

    # ── Main loop ─────────────────────────────────────────────────────────────
    while game_count < args.max_games:
        # Anneal temperature
        progress = min(1.0, game_count / max(args.max_games * 0.8, 1))
        temperature = TEMP_START + (TEMP_END - TEMP_START) * progress

        # Colour randomisation
        learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color     = "B" if learner_color == "W" else "W"
        opponent      = HeuristicAgent(
            color=opp_color, difficulty=difficulty,
            game_ai=None,
        )
        # Build a fresh inner GameAI for the opponent each game
        from learned_ai.agents.heuristic_agent import GameAI as _GA
        opp_inner = _GA(color=opp_color, difficulty=difficulty,
                        override_time_budget=TIME_BUDGET)
        opponent._inner = opp_inner

        board   = BoardState.new_game()
        ply     = 0
        game_trajectory: list[ScaffoldedStep] = []
        done    = False
        outcome = 0.0

        while ply < MAX_PLY:
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
                # ── Learner turn ───────────────────────────────────────────────
                enc = encode_position(board, player, sentinel_advisor=sentinel, db=db)
                if enc is None or not enc.legal_moves:
                    outcome = LOSS_REWARD
                    done = True
                    break

                feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
                vi_t   = torch.tensor(enc.value_input,  dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits = model.policy_logits(feat_t)
                    import torch.nn.functional as F
                    scaled = logits / temperature
                    log_probs = F.log_softmax(scaled, dim=-1)
                    probs     = log_probs.exp()
                    if not torch.isfinite(probs).all():
                        probs = torch.where(
                            torch.isfinite(probs), probs, torch.zeros_like(probs)
                        )
                        probs = probs / probs.sum().clamp(min=1e-9)
                    chosen_idx = int(torch.multinomial(probs.cpu(), 1).item())
                    log_prob_old = float(log_probs[chosen_idx].item())

                move = enc.legal_moves[chosen_idx]
                board_after = board.apply_move(move)

                # Encode next state for value bootstrapping
                enc_after = encode_position(
                    board_after, opp_color,
                    sentinel_advisor=sentinel, db=db
                )

                # Per-move reward
                reward = _compute_per_move_reward(enc, chosen_idx, enc_after, db)

                # Malom trap: opponent is now in a losing position?
                if db is not None:
                    opp_state_wdl = None
                    try:
                        opp_state_wdl = db.query_state(board_after)
                    except Exception:
                        pass
                    # board_after is the opponent's turn; "L" for opp = good for us
                    if opp_state_wdl == "L":
                        reward += DELTA

                # Next state features for bootstrapping
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
                board = board_after

            else:
                # ── Opponent turn ──────────────────────────────────────────────
                opp_move = opponent.choose_move(board)
                if not opp_move:
                    outcome = WIN_REWARD
                    done = True
                    break
                board = board.apply_move(opp_move)

            ply += 1

        if not done:
            outcome = DRAW_LONG

        # ── Retroactive game-level rescoring ───────────────────────────────────
        if game_trajectory:
            _retroactive_rescore(game_trajectory, outcome, ply)

        ep_steps.extend(game_trajectory)
        win_history.append(1.0 if outcome == WIN_REWARD else 0.0)
        game_count += 1

        # ── A2C/PPO update ─────────────────────────────────────────────────────
        if len(ep_steps) >= UPDATE_EVERY:
            pl, vl, ent = update_fn(
                model, opt, ep_steps, device,
                gamma=GAMMA_TD, entropy_coef=ENTROPY_COEF,
            )
            ep_steps.clear()

        # ── Logging and checkpointing ──────────────────────────────────────────
        if game_count % 50 == 0:
            win_rate = sum(win_history) / max(len(win_history), 1)
            log_entry = {
                "game": game_count,
                "difficulty": difficulty,
                "win_rate_200": round(win_rate, 4),
                "temperature": round(temperature, 3),
                "outcome": outcome,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(
                f"[s2] game {game_count:6d} | diff {difficulty} | "
                f"win-200={win_rate:.3f} | temp={temperature:.2f} | "
                f"outcome={outcome:+.2f}"
            )

            ckpt = {
                "model":        model.state_dict(),
                "model_config": model.get_config(),
                "stage":        "s2",
                "game_count":   game_count,
                "best_win_rate": best_win_rate,
                "difficulty":   difficulty,
            }
            torch.save(ckpt, out_dir / "latest.pt")

            if win_rate > best_win_rate and len(win_history) >= 100:
                best_win_rate = win_rate
                torch.save(ckpt, out_dir / "best.pt")
                print(f"[s2]  → best win rate: {best_win_rate:.3f}")

        # ── Curriculum advancement ─────────────────────────────────────────────
        if len(win_history) >= ROLLING_WIN:
            win_rate = sum(win_history) / len(win_history)
            if difficulty == DIFF_START and win_rate >= WIN_TARGET_1:
                difficulty = DIFF_TARGET
                win_history.clear()
                print(f"[s2] *** Advanced to difficulty {difficulty} ***")
            elif difficulty == DIFF_TARGET and win_rate >= WIN_TARGET_2:
                print(f"[s2] *** Target win rate {WIN_TARGET_2} reached at diff {DIFF_TARGET}! ***")
                break

    # Final flush of remaining steps
    if ep_steps:
        update_fn(model, opt, ep_steps, device, gamma=GAMMA_TD)

    # Final save
    ckpt = {
        "model":         model.state_dict(),
        "model_config":  model.get_config(),
        "stage":         "s2",
        "game_count":    game_count,
        "best_win_rate": best_win_rate,
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s2] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s2] Checkpoint: {out_dir / 'best.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 2: scaffolded A2C self-play")
    p.add_argument("--resume",    default="", type=str)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2"),
    )
    p.add_argument(
        "--sentinel",
        default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"),
    )
    p.add_argument("--malom",     default="", type=str)
    p.add_argument("--ppo",       action="store_true")
    p.add_argument("--max-games", type=int,   default=10000)
    p.add_argument("--seed",      type=int,   default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
