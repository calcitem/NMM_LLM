"""scripts/train_stage2.py — Stage 2: REINFORCE self-play vs weak heuristic.

Improvements over v1:
  - Sentinel warmup: no filter for the first --warmup-frac of games, so the
    model accumulates enough transitions to learn from before filtering kicks in.
  - Malom DB reward shaping (two signals, both Malom-exact):
      1. Move quality: query_move_quality delta after each learner move — rewards
         moves that directly improve the learner's own position.
      2. Trap reward: after each learner move, query the resulting position from
         the opponent's perspective.  If the opponent is now in a "L" (losing)
         position the learner gets a bonus on that transition — rewarding moves
         that constrain or trick the opponent into bad territory, not just moves
         that improve the learner's own evaluation.
    Both signals use the same --malom-weight scale and are active for the first
    --malom-frac of games.
  - Lower temperature (0.5) — less random, more intentional play.
  - Larger update batches (UPDATE_EVERY=16) — more stable gradient estimates.
  - Higher win reward (2.0) — stronger terminal signal on wins.
  - Lower sentinel threshold (0.1 after warmup) — only clear blunders filtered.

Curriculum:
  diff 2 (vn_blend=0) → diff 3 when rolling-200 win rate >= 65%.
  Exit when rolling-200 win rate >= 65% at diff 3.

Usage:
    .venv/bin/python scripts/train_stage2.py [--resume CKPT] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from game.board import BoardState
from game.rules import get_all_legal_moves, is_terminal
import learned_ai.agents.heuristic_agent as _ha_mod
from learned_ai.agents.heuristic_agent import HeuristicAgent
from learned_ai.agents.learned_agent import LearnedAgent
from learned_ai.models.backbone import NMMNet
from learned_ai.models.state_encoder import PHASE_NAMES
from learned_ai.sentinel.infer import SentinelAdvisor
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.training.replay_buffer import Transition

# ── Defaults ──────────────────────────────────────────────────────────────────

LR              = 1e-4
GAMMA           = 0.99
TEMPERATURE     = 0.5        # lower than v1 (was 1.0) — less random
ENTROPY_COEF    = 0.01
UPDATE_EVERY    = 16         # larger batches (was 4)
MIN_BATCH       = 32         # minimum transitions to run an update

WIN_REWARD      = 2.0        # stronger signal on wins (was implicitly 1.0)

SENTINEL_THRESHOLD  = 0.1   # after warmup (was 0.25)
WARMUP_FRAC         = 0.20  # fraction of max_games with no sentinel filter
MALOM_FRAC          = 0.30  # fraction of max_games with Malom reward shaping
MALOM_WEIGHT        = 0.3   # scale for per-move Malom delta (delta ∈ [-2,+2])

ROLLING_WINDOW      = 200
WIN_RATE_TARGET     = 0.65
DIFF_START          = 2
DIFF_TARGET         = 3
MAX_PLIES           = 400

DEFAULT_MALOM_DB = (
    "/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_opponent(difficulty: int, time_budget: float) -> HeuristicAgent:
    inner = _ha_mod.GameAI(color="B", difficulty=difficulty,
                           override_time_budget=time_budget)
    return HeuristicAgent(color="B", difficulty=difficulty, game_ai=inner)


# ── Episode runner ─────────────────────────────────────────────────────────────

def run_episode(
    model: NMMNet,
    learner_color: str,
    opponent: HeuristicAgent,
    sentinel: Optional[SentinelAdvisor],
    malom_db: Optional[ExternalSolvedDB],
    device: torch.device,
    game_idx: int,
    warmup_games: int,
    malom_games: int,
    temperature: float = TEMPERATURE,
    sentinel_threshold: float = SENTINEL_THRESHOLD,
    win_reward: float = WIN_REWARD,
    malom_weight: float = MALOM_WEIGHT,
    gamma: float = GAMMA,
    max_plies: int = MAX_PLIES,
) -> tuple[Optional[str], list[Transition], int, int]:
    """Play one game. Returns (winner, transitions, n_kept, n_filtered)."""
    use_sentinel = sentinel is not None and sentinel.is_loaded() and game_idx >= warmup_games
    use_malom    = malom_db is not None and malom_db.is_available() and game_idx < malom_games

    learner = LearnedAgent(
        color=learner_color, model=model, device=str(device),
        mode="sample", temperature=temperature,
    )

    board = BoardState.new_game()
    # (state, phase_id, primary_idx, legal_mask, keep, malom_bonus)
    steps: list[tuple] = []
    n_filtered = 0
    opp_moves = 0
    plies = 0
    learner_just_moved = False

    while plies < max_plies:
        terminal, winner = is_terminal(board)
        if terminal:
            break
        legal = get_all_legal_moves(board)
        if not legal:
            winner = "B" if board.turn == "W" else "W"
            break

        if board.turn == learner_color:
            move = learner.choose_move(board)
            if not move:
                winner = "B" if board.turn == "W" else "W"
                break
            d = learner.last_decision

            # Sentinel blunder filter (off during warmup)
            keep = True
            if use_sentinel:
                try:
                    adv = sentinel.advise(board, [move], board.turn, played_move_idx=0)
                    if adv is not None and adv.played_move_quality < sentinel_threshold:
                        keep = False
                        n_filtered += 1
                except Exception:
                    pass

            # Malom signal 1: move quality (how much did this move improve our position)
            malom_bonus = 0.0
            if use_malom:
                try:
                    q = malom_db.query_move_quality(board, move)
                    if q is not None:
                        malom_bonus = malom_weight * float(q)
                except Exception:
                    pass

            steps.append((d.state, d.phase_id, d.primary_index, d.legal_mask,
                          keep, malom_bonus))
            learner_just_moved = True
        else:
            if opp_moves == 0:
                move = random.choice(legal)   # random first move for variety
            else:
                move = opponent.choose_move(board)
            opp_moves += 1
            if not move:
                winner = learner_color
                break
            learner_just_moved = False

        board = board.apply_move(move)
        plies += 1

        # Malom signal 2: trap reward — after the learner's move, opponent is now
        # to move.  If Malom says the opponent is in a losing ("L") position, the
        # learner's last move created a trap and earns a bonus.
        if learner_just_moved and use_malom and steps:
            try:
                q_trap = malom_db.query(board)   # from opponent's (current mover's) perspective
                if q_trap == "L":                # opponent is losing → AI set a trap
                    s = steps[-1]
                    steps[-1] = (*s[:-1], s[-1] + malom_weight)
            except Exception:
                pass

    else:
        winner = None  # ply cap → draw

    # Assign discounted terminal reward + Malom shaped reward
    n = len(steps)
    transitions: list[Transition] = []
    for i, (state, phase_id, primary_idx, legal_mask, keep, malom_bonus) in enumerate(steps):
        if not keep:
            continue
        dist = n - 1 - i
        if winner is None:
            r_term = 0.0
        elif winner == learner_color:
            r_term = win_reward * (gamma ** dist)
        else:
            r_term = -win_reward * (gamma ** dist)
        transitions.append(Transition(
            state=state,
            legal_mask=legal_mask,
            primary_index=primary_idx,
            capture_index=None,
            reward=r_term + malom_bonus,
            phase_id=phase_id,
            side_to_move=learner_color,
            done=(i == n - 1),
        ))

    return winner, transitions, len(transitions), n_filtered


# ── REINFORCE update ──────────────────────────────────────────────────────────

def reinforce_update(
    model: NMMNet,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    device: torch.device,
) -> tuple[float, float]:
    """One REINFORCE step. Returns (policy_loss, value_loss) or (0, 0) if skipped."""
    if len(transitions) < MIN_BATCH:
        return 0.0, 0.0

    states        = torch.stack([t.state for t in transitions]).to(device)
    legal_masks   = torch.stack([t.legal_mask for t in transitions]).to(device)
    primary_indices = torch.tensor(
        [t.primary_index for t in transitions], device=device, dtype=torch.long)
    rewards       = torch.tensor(
        [t.reward for t in transitions], device=device, dtype=torch.float32)
    phase_ids     = [t.phase_id for t in transitions]

    model.train()
    feats = model.backbone(states)

    # Value-head baseline
    values = model.value_head(feats).squeeze(-1)
    advantages = rewards - values.detach()
    # Normalize only when there's real variance
    if advantages.std() > 1e-3:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    # Policy loss (phase-routed, log_probs recomputed from stored states/actions)
    pl_sum  = torch.zeros([], device=device)
    ent_sum = torch.zeros([], device=device)
    n_total = 0

    for ph in range(model.num_phases):
        idx = [i for i, p in enumerate(phase_ids) if p == ph]
        if not idx:
            continue
        idx_t    = torch.tensor(idx, device=device)
        logits_p = model.phase_heads[PHASE_NAMES[ph]](feats[idx_t])
        logits_p = logits_p.masked_fill(~legal_masks[idx_t], -1e9)
        log_probs_p = F.log_softmax(logits_p, dim=-1)
        sel_lp   = log_probs_p.gather(1, primary_indices[idx_t].unsqueeze(1)).squeeze(1)
        pl_sum   = pl_sum - (sel_lp * advantages[idx_t]).sum()
        probs_p  = log_probs_p.exp()
        ent_sum  = ent_sum + (-(probs_p * log_probs_p).sum(dim=-1)).sum()
        n_total += len(idx)

    policy_loss  = pl_sum  / max(n_total, 1)
    entropy_loss = ent_sum / max(n_total, 1)
    value_loss   = F.mse_loss(values, rewards)
    loss = policy_loss - ENTROPY_COEF * entropy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()

    return float(policy_loss.item()), float(value_loss.item())


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    pa = argparse.ArgumentParser(description="Stage 2: REINFORCE vs heuristic")
    pa.add_argument("--resume",   default=str(_ROOT / "learned_ai/checkpoints/stage1/best.pt"))
    pa.add_argument("--out-dir",  default=str(_ROOT / "learned_ai/checkpoints/stage2"))
    pa.add_argument("--sentinel", default=str(_ROOT / "learned_ai/sentinel/checkpoints/best.pt"))
    pa.add_argument("--malom-db", default=DEFAULT_MALOM_DB)
    pa.add_argument("--max-games",     type=int,   default=5_000)
    pa.add_argument("--time-budget",   type=float, default=0.05,
                    help="Seconds per opponent move (default 0.05)")
    pa.add_argument("--temperature",   type=float, default=TEMPERATURE)
    pa.add_argument("--win-reward",    type=float, default=WIN_REWARD)
    pa.add_argument("--warmup-frac",   type=float, default=WARMUP_FRAC,
                    help="Fraction of games with no sentinel filter (default 0.20)")
    pa.add_argument("--malom-frac",    type=float, default=MALOM_FRAC,
                    help="Fraction of games with Malom reward shaping (default 0.30)")
    pa.add_argument("--malom-weight",  type=float, default=MALOM_WEIGHT,
                    help="Scale for per-move Malom quality delta (default 0.30)")
    pa.add_argument("--diff-start",    type=int,   default=DIFF_START)
    pa.add_argument("--no-sentinel",   action="store_true")
    pa.add_argument("--no-malom",      action="store_true")
    args = pa.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Derived game counts
    warmup_games = int(args.max_games * args.warmup_frac)
    malom_games  = int(args.max_games * args.malom_frac)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = NMMNet()
    resume = Path(args.resume)
    if resume.exists():
        ckpt = torch.load(str(resume), map_location="cpu", weights_only=False)
        sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(sd)
        print(f"Resumed from {resume}")
    else:
        print(f"WARNING: no checkpoint at {resume} — using random weights")
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ── Sentinel ──────────────────────────────────────────────────────────────
    sentinel: Optional[SentinelAdvisor] = None
    if not args.no_sentinel:
        sp = Path(args.sentinel)
        if sp.exists():
            sentinel = SentinelAdvisor(checkpoint_path=str(sp), device="cpu")
            print(f"Sentinel loaded: {sp}  (threshold={SENTINEL_THRESHOLD} after game {warmup_games})")
        else:
            print(f"Sentinel not found at {sp} — running without filter")

    # ── Malom DB ──────────────────────────────────────────────────────────────
    malom_db: Optional[ExternalSolvedDB] = None
    if not args.no_malom:
        malom_db = ExternalSolvedDB(db_path=args.malom_db)
        if malom_db.is_available():
            print(f"Malom DB loaded: {args.malom_db}  (shaping for games 0–{malom_games})")
        else:
            print(f"Malom DB unavailable at {args.malom_db} — no reward shaping")

    # ── Curriculum ────────────────────────────────────────────────────────────
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    current_diff = args.diff_start
    opponent = make_opponent(current_diff, args.time_budget)

    results: deque[str] = deque(maxlen=ROLLING_WINDOW)
    accumulated: list[Transition] = []
    best_win_rate = 0.0
    total_filtered = 0
    total_kept = 0

    print(f"\nStage 2 v2  lr={LR}  γ={GAMMA}  T={args.temperature}"
          f"  win_reward={args.win_reward}  opp_budget={args.time_budget}s")
    print(f"  sentinel_thresh={SENTINEL_THRESHOLD} (warmup {warmup_games} games)"
          f"  malom_weight={args.malom_weight} (first {malom_games} games)")
    print(f"  update_every={UPDATE_EVERY}  min_batch={MIN_BATCH}")
    print(f"  diff {current_diff} → {DIFF_TARGET}  exit: {WIN_RATE_TARGET:.0%} rolling-{ROLLING_WINDOW}\n")

    t0 = time.time()
    for game in range(args.max_games):
        learner_color = "W" if game % 2 == 0 else "B"

        winner, transitions, n_kept, n_filt = run_episode(
            model, learner_color, opponent, sentinel, malom_db, device,
            game_idx=game,
            warmup_games=warmup_games,
            malom_games=malom_games,
            temperature=args.temperature,
            sentinel_threshold=SENTINEL_THRESHOLD,
            win_reward=args.win_reward,
            malom_weight=args.malom_weight,
        )

        total_kept     += n_kept
        total_filtered += n_filt
        accumulated.extend(transitions)

        r_str = "D" if winner is None else ("W" if winner == learner_color else "L")
        results.append(r_str)

        n_res    = len(results)
        win_rate = results.count("W") / n_res
        filt_pct = total_filtered / max(total_kept + total_filtered, 1)
        phase    = ("warmup" if game < warmup_games
                    else ("malom" if game < malom_games else "rl"))
        elapsed  = time.time() - t0

        print(f"  game {game+1:5d}  {r_str}  diff={current_diff}  "
              f"wr={win_rate:.1%} ({n_res:3d})  "
              f"filt={filt_pct:.0%}  "
              f"trans={n_kept:3d}  "
              f"[{phase}]  t={elapsed:.0f}s")

        # ── REINFORCE update ───────────────────────────────────────────────
        if (game + 1) % UPDATE_EVERY == 0 and accumulated:
            batch_size = len(accumulated)
            pl, vl = reinforce_update(model, optimizer, accumulated, device)
            accumulated.clear()
            if pl != 0.0:
                print(f"    → update  policy_loss={pl:.4f}  value_loss={vl:.4f}"
                      f"  batch={batch_size}")

        # ── Best checkpoint ────────────────────────────────────────────────
        if n_res >= 50 and win_rate > best_win_rate:
            best_win_rate = win_rate
            torch.save({"model": model.state_dict()}, out_dir / "best.pt")

        # ── Curriculum bump ────────────────────────────────────────────────
        if (current_diff < DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            current_diff += 1
            opponent = make_opponent(current_diff, args.time_budget)
            results.clear()
            total_filtered = 0
            total_kept = 0
            print(f"\n  ★ difficulty → {current_diff}  (win_rate was {win_rate:.1%})\n")

        # ── Exit criterion ─────────────────────────────────────────────────
        if (current_diff >= DIFF_TARGET
                and n_res >= ROLLING_WINDOW
                and win_rate >= WIN_RATE_TARGET):
            print(f"\n  ★ EXIT: {win_rate:.1%} win rate at diff {current_diff} (game {game+1})")
            break

    torch.save({"model": model.state_dict()}, out_dir / "latest.pt")
    n_res    = len(results)
    win_rate = results.count("W") / n_res if n_res else 0.0
    print(f"\nStage 2 done.  win_rate={win_rate:.1%}  diff={current_diff}"
          f"  best={best_win_rate:.1%}")
    print(f"Checkpoints → {out_dir}")


if __name__ == "__main__":
    main()
