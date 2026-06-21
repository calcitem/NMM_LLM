"""scripts/train_scaffolded_s3.py — Stage 3: Malom supervised fine-tuning.

Combines A2C self-play with a supervised cross-entropy loss toward the
Malom database's best move, weighted by DTM quality (faster wins get
stronger supervision weight).

The balance shifts from RL-dominated (Stage 2) to DB-guided:
  total_loss = RL_WEIGHT * a2c_loss + SL_WEIGHT * sl_loss

sl_loss is zero when the Malom DB has no entry for the position, so it
only fires in tractable endgame/midgame positions where the DB is populated.

Typical use: resume from Stage 2 best.pt, run at higher difficulty (3–4),
until rolling-200 win rate >= 40% vs diff 4.

Usage
-----
    .venv/bin/python scripts/train_scaffolded_s3.py [options]

Options
-------
  --resume   PATH    Stage 2 best.pt (required)
  --out-dir  DIR     Checkpoint directory
  --sentinel PATH    SentinelAdvisor checkpoint
  --malom    PATH    Malom DB directory (REQUIRED for SL signal)
  --ppo              Use PPO surrogate
  --max-games N      Max games (default 5000)
  --diff     D       Opponent difficulty to start at (default 3)
  --seed     N       Random seed
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
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
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

# ── Reward config (same as Stage 2) ───────────────────────────────────────────
ALPHA  = 0.15
BETA   = 0.10
GAMMA  = 0.25
DELTA  = 0.15
LAMBDA = 0.50
DECAY  = 0.98
VN_BETA = 0.10

WIN_REWARD  = 1.0
LOSS_REWARD = -1.0
DRAW_SHORT  =  0.15
DRAW_LONG   = -0.05

# ── Stage 3 specific ──────────────────────────────────────────────────────────
LR         = 3e-5   # lower than Stage 2 — fine-tuning
RL_WEIGHT  = 0.6    # fraction of total loss from A2C
SL_WEIGHT  = 0.4    # fraction from Malom supervised signal

GAMMA_TD   = 0.99
TEMP       = 1.0    # fixed temperature for fine-tuning
ROLLING_WIN = 200
WIN_TARGET  = 0.40  # exit threshold vs diff 4

UPDATE_EVERY = 16
MIN_BATCH    = 8
MAX_PLY      = 400
TIME_BUDGET  = 0.05


def _load_settings() -> dict:
    p = _ROOT / "data" / "settings.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def _move_key(mv: dict):
    return (mv.get("from"), mv.get("to"), mv.get("capture"))


def _retroactive_rescore(trajectory: list[ScaffoldedStep], outcome: float):
    n = len(trajectory)
    for t_idx, step in enumerate(trajectory):
        plies_remaining = n - t_idx - 1
        step.reward += LAMBDA * outcome * (DECAY ** plies_remaining)


def _per_move_reward(enc, chosen_idx: int, db, board_after) -> float:
    r = 0.0
    if enc.sentinel_scores:
        mean_s = sum(enc.sentinel_scores) / len(enc.sentinel_scores)
        r += ALPHA * (enc.sentinel_scores[chosen_idx] - mean_s)
    h_delta = enc.h_scores_abs[chosen_idx] - enc.h_before
    r += BETA * math.tanh(h_delta)
    # Value-net component
    if enc.vn_scores_abs:
        vn_delta = enc.vn_scores_abs[chosen_idx] - enc.vn_before
        r += VN_BETA * math.tanh(vn_delta)
    if enc.db_moves:
        mv_key = _move_key(enc.legal_moves[chosen_idx])
        entry = next(
            (m for m in enc.db_moves if _move_key(m.get("move", {})) == mv_key),
            None,
        )
        if entry and entry.get("wdl") == "win":
            r += GAMMA * float(dtm_quality("win", entry.get("dtm")))
    if db is not None and board_after is not None:
        try:
            opp_wdl = db.query_state(board_after)
            if opp_wdl == "L":
                r += DELTA
        except Exception:
            pass
    return float(r)


def _sl_loss(model, enc, db, device) -> Optional[torch.Tensor]:
    """Supervised loss toward Malom best move.

    Returns a scalar tensor when a DB entry is available, else None.
    The target distribution is a weighted softmax over moves:
      - Malom 'win' moves: weight = dtm_quality (faster wins = higher weight)
      - Malom 'draw' or missing: weight = 0.1
      - Malom 'loss': weight = 0.0
    """
    if not enc.db_moves:
        return None

    k = len(enc.legal_moves)
    weights = np.zeros(k, dtype=np.float32)
    has_known = False

    db_by_key = {_move_key(m.get("move", {})): m for m in enc.db_moves}

    for i, mv in enumerate(enc.legal_moves):
        entry = db_by_key.get(_move_key(mv))
        if entry is None:
            weights[i] = 0.1
            continue
        wdl = entry.get("wdl", "unknown")
        dtm = entry.get("dtm")
        if wdl == "win":
            q = float(dtm_quality("win", dtm))
            weights[i] = max(q, 0.05)
            has_known = True
        elif wdl == "draw":
            weights[i] = 0.15
            has_known = True
        elif wdl == "loss":
            weights[i] = 0.0
            has_known = True
        else:
            weights[i] = 0.1

    if not has_known:
        return None

    # Target: normalize weights to a probability distribution
    w_sum = weights.sum()
    if w_sum < 1e-9:
        return None
    target_probs = torch.tensor(weights / w_sum, dtype=torch.float32).to(device)

    feat = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
    logits = model.policy_logits(feat)
    log_probs = F.log_softmax(logits, dim=-1)
    # KL from target to learned: -sum(target * log_prob)
    return -(target_probs * log_probs).sum()


def run(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[s3] Device: {device}")
    rng = random.Random(args.seed)

    # ── Sentinel ───────────────────────────────────────────────────────────────
    sentinel = None
    sent_path = args.sentinel or str(
        _ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"
    )
    if Path(sent_path).exists():
        sentinel = load_advisor(sent_path)
        if sentinel and sentinel.is_loaded():
            print(f"[s3] Sentinel loaded: {sent_path}")
        else:
            sentinel = None

    # ── Malom DB ───────────────────────────────────────────────────────────────
    db = None
    malom_path = args.malom or _load_settings().get("malom_db_path", "")
    if malom_path and Path(malom_path).exists():
        try:
            from learned_ai.sentinel.db_teacher import ExternalSolvedDB
            db = ExternalSolvedDB(malom_path)
            if db.is_available():
                print(f"[s3] Malom DB loaded: {malom_path}")
            else:
                db = None
        except Exception as e:
            print(f"[s3] Malom DB failed ({e})")
    if db is None:
        print("[s3] WARNING: Malom DB unavailable — SL signal will be zero (not recommended)")

    # ── Value net ──────────────────────────────────────────────────────────────────
    value_net = None
    vn_path = args.value_net or str(_ROOT / "data" / "value_net.npz")
    if vn_path and Path(vn_path).exists():
        try:
            from ai.value_net import ValueNet as _ValueNet
            value_net = _ValueNet.load(vn_path)
            print(f"[s3] Value net loaded: {vn_path}")
        except Exception as e:
            print(f"[s3] Value net load failed ({e}) — VN features will be 0")
    else:
        print("[s3] No value net — VN features will be 0")

    # ── Model ─────────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _s2b_best = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2b" / "best.pt"
    _s2_best  = _ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s2"  / "best.pt"
    resume_path = args.resume or (
        str(_s2b_best) if _s2b_best.exists() else str(_s2_best)
    )
    if Path(resume_path).exists():
        print(f"[s3] Resuming from {resume_path}")
        ckpt  = torch.load(resume_path, map_location=device, weights_only=False)
        cfg   = ckpt.get("model_config", {})
        model = ScaffoldedPolicyNet.from_config(cfg).to(device)
        sd_key = "model" if "model" in ckpt else "state_dict"
        model.load_state_dict(ckpt[sd_key])
        start_game = ckpt.get("game_count", 0)
    else:
        print("[s3] ERROR: --resume checkpoint not found. Run Stage 2 first.")
        sys.exit(1)

    opt = torch.optim.Adam(model.parameters(), lr=LR)

    difficulty    = args.diff
    win_history:  deque[float] = deque(maxlen=ROLLING_WIN)
    ep_steps:     list[ScaffoldedStep] = []
    sl_loss_acc   = 0.0
    sl_count      = 0
    game_count    = start_game
    best_win_rate = 0.0
    log_path      = out_dir / "train_log.jsonl"

    update_fn = scaffolded_ppo_update if args.ppo else scaffolded_a2c_update

    print(f"[s3] Stage 3 fine-tuning at diff {difficulty} | RL={RL_WEIGHT} SL={SL_WEIGHT}")

    while game_count < start_game + args.max_games:
        learner_color = "W" if rng.random() < 0.5 else "B"
        opp_color     = "B" if learner_color == "W" else "W"

        from learned_ai.agents.heuristic_agent import GameAI as _GA
        opp_inner = _GA(color=opp_color, difficulty=difficulty,
                        override_time_budget=TIME_BUDGET)
        opponent = HeuristicAgent(color=opp_color, difficulty=difficulty)
        opponent._inner = opp_inner

        board     = BoardState.new_game()
        ply       = 0
        trajectory: list[ScaffoldedStep] = []
        sl_losses: list[torch.Tensor]    = []
        done      = False
        outcome   = 0.0

        while ply < MAX_PLY:
            terminal, winner = is_terminal(board)
            if terminal:
                outcome = (
                    WIN_REWARD if winner == learner_color
                    else LOSS_REWARD if winner is not None
                    else (DRAW_SHORT if ply < 100 else DRAW_LONG)
                )
                done = True
                break

            player = board.turn

            if player == learner_color:
                enc = encode_position(board, player, sentinel_advisor=sentinel, db=db, value_net=value_net)
                if enc is None or not enc.legal_moves:
                    outcome = LOSS_REWARD
                    done = True
                    break

                # SL loss (Malom supervision)
                sl = _sl_loss(model, enc, db, device)
                if sl is not None:
                    sl_losses.append(sl)
                    sl_loss_acc += float(sl.item())
                    sl_count += 1

                # Policy sampling
                feat_t = torch.tensor(enc.feat_matrix, dtype=torch.float32).to(device)
                vi_t   = torch.tensor(enc.value_input,  dtype=torch.float32).to(device)
                with torch.no_grad():
                    logits    = model.policy_logits(feat_t)
                    log_probs = F.log_softmax(logits / TEMP, dim=-1)
                    probs     = log_probs.exp()
                    if not torch.isfinite(probs).all():
                        probs = torch.where(
                            torch.isfinite(probs), probs, torch.zeros_like(probs)
                        )
                        probs = probs / probs.sum().clamp(min=1e-9)
                    chosen_idx   = int(torch.multinomial(probs.cpu(), 1).item())
                    log_prob_old = float(log_probs[chosen_idx].item())

                move       = enc.legal_moves[chosen_idx]
                board_after = board.apply_move(move)
                reward      = _per_move_reward(enc, chosen_idx, db, board_after)

                enc_after = encode_position(
                    board_after, opp_color,
                    sentinel_advisor=sentinel, db=db, value_net=value_net,
                )
                if enc_after is not None and enc_after.legal_moves:
                    next_mf = enc_after.feat_matrix
                    next_vi = enc_after.value_input
                else:
                    next_mf = np.zeros((1, enc.feat_matrix.shape[1]), dtype=np.float32)
                    next_vi = np.zeros(enc.value_input.shape, dtype=np.float32)

                t_next, _ = is_terminal(board_after)
                trajectory.append(ScaffoldedStep(
                    move_features=enc.feat_matrix,
                    value_input=enc.value_input,
                    chosen_idx=chosen_idx,
                    log_prob_old=log_prob_old,
                    reward=reward,
                    next_move_features=next_mf,
                    next_value_input=next_vi,
                    done=t_next,
                ))
                board = board_after
            else:
                opp_move = opponent.choose_move(board)
                if not opp_move:
                    outcome = WIN_REWARD
                    done = True
                    break
                board = board.apply_move(opp_move)

            ply += 1

        if not done:
            outcome = DRAW_LONG

        _retroactive_rescore(trajectory, outcome)
        ep_steps.extend(trajectory)
        win_history.append(1.0 if outcome == WIN_REWARD else 0.0)
        game_count += 1

        # ── Combined RL + SL update ────────────────────────────────────────────
        if len(ep_steps) >= UPDATE_EVERY:
            # A2C/PPO component
            if len(ep_steps) >= MIN_BATCH:
                model.train()
                pl, vl, ent = update_fn(
                    model, opt, ep_steps, device,
                    gamma=GAMMA_TD,
                )
            else:
                pl = vl = ent = 0.0
            ep_steps.clear()

            # SL component (accumulated from this game)
            if sl_losses:
                model.train()
                sl_total = torch.stack(sl_losses).mean() * SL_WEIGHT
                opt.zero_grad()
                sl_total.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                sl_losses.clear()

        # ── Logging ────────────────────────────────────────────────────────────
        if game_count % 50 == 0:
            win_rate = sum(win_history) / max(len(win_history), 1)
            mean_sl  = sl_loss_acc / max(sl_count, 1)
            log_entry = {
                "game":         game_count,
                "difficulty":   difficulty,
                "win_rate_200": round(win_rate, 4),
                "sl_loss_mean": round(mean_sl, 4),
                "outcome":      outcome,
            }
            with open(log_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            print(
                f"[s3] game {game_count:6d} | diff {difficulty} | "
                f"win-200={win_rate:.3f} | sl_loss={mean_sl:.4f}"
            )
            sl_loss_acc = sl_count = 0

            ckpt = {
                "model":        model.state_dict(),
                "model_config": model.get_config(),
                "stage":        "s3",
                "game_count":   game_count,
                "best_win_rate": best_win_rate,
            }
            torch.save(ckpt, out_dir / "latest.pt")
            if win_rate > best_win_rate and len(win_history) >= 100:
                best_win_rate = win_rate
                torch.save(ckpt, out_dir / "best.pt")
                print(f"[s3]  → best win rate: {best_win_rate:.3f}")

        if len(win_history) >= ROLLING_WIN:
            win_rate = sum(win_history) / len(win_history)
            if win_rate >= WIN_TARGET:
                print(f"[s3] *** Target {WIN_TARGET} reached at diff {difficulty}! ***")
                break

    # Final save
    ckpt = {
        "model":         model.state_dict(),
        "model_config":  model.get_config(),
        "stage":         "s3",
        "game_count":    game_count,
        "best_win_rate": best_win_rate,
    }
    torch.save(ckpt, out_dir / "latest.pt")
    print(f"\n[s3] Done. Games: {game_count}  Best win rate: {best_win_rate:.3f}")
    print(f"[s3] Checkpoint: {out_dir / 'best.pt'}")


def main() -> None:
    p = argparse.ArgumentParser(description="Stage 3: Malom supervised fine-tuning")
    p.add_argument("--resume",    default="", type=str)
    p.add_argument(
        "--out-dir",
        default=str(_ROOT / "learned_ai" / "checkpoints" / "scaffolded" / "s3"),
    )
    p.add_argument(
        "--sentinel",
        default=str(_ROOT / "learned_ai" / "sentinel" / "checkpoints" / "best.pt"),
    )
    p.add_argument("--malom",     default="", type=str)
    p.add_argument("--value-net", default=str(_ROOT / "data" / "value_net.npz"), type=str)
    p.add_argument("--ppo",       action="store_true")
    p.add_argument("--max-games", type=int, default=5000)
    p.add_argument("--diff",      type=int, default=3)
    p.add_argument("--seed",      type=int, default=42)
    args = p.parse_args()
    run(args)


if __name__ == "__main__":
    main()
