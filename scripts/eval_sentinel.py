"""scripts/eval_sentinel.py — grade sentinel against Malom DB ground truth.

For every position in every replayed game, enumerates all legal moves, queries
the Malom DB for per-move WDL + DTM ground truth, runs the sentinel with DB
feature slots zeroed (matching live inference conditions), and reports alignment
metrics.

Metrics
-------
  win_acc         fraction of DB-win moves sentinel scores > 0.5
  loss_acc        fraction of DB-loss moves sentinel scores < 0.5
  overall_acc     combined (win + loss) direction accuracy
  top1_win_rate   positions with a DB-win available where sentinel ranks a win #1
  top1_exact      positions where sentinel's #1 move matches DB's best move exactly
  critical_miss   positions where DB-win exists but sentinel scores a loss highest
  bad_move_recall fraction of loss-in-≤10 moves sentinel scores < 0.4
  spearman_r      Spearman rank correlation (sentinel score vs DB quality)
  dtm_pearson_r   Pearson correlation of sentinel score and DB DTM quality
  score_by_wdl    mean/std sentinel score per DB category
  phase_breakdown win_acc and loss_acc split by game phase

Usage
-----
  python scripts/eval_sentinel.py \\
    --checkpoint learned_ai/sentinel/checkpoints/stage3/best.pt \\
    --game-dir data/games \\
    [--human-game-dir data/human_games] \\
    [--db-path "/mnt/windows/NMM_DB/Malom_Standard_Ultra-strong_1.1.0/Std_DD_89adjusted"] \\
    [--limit 200] \\
    [--output eval_results.json]
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from game.board import BoardState
from learned_ai.sentinel.config import SentinelConfig
from learned_ai.sentinel.dataset import (
    _board_from_fen_before,
    _enumerate_legal_moves,
    _heuristic_scores,
    _normalise_scores,
    _ranks_desc,
    _iter_game_records,
)
from learned_ai.sentinel.db_teacher import ExternalSolvedDB
from learned_ai.sentinel.feature_builder import FEATURE_DIM, build_move_features
from learned_ai.sentinel.labels import dtm_quality, WDL_QUALITY, BAD_MOVE_DTM_THRESHOLD
from learned_ai.sentinel.model import SentinelNet

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# DB-derived feature slots zeroed to replicate inference conditions.
_DB_FEATURE_SLOTS = list(range(41, 46)) + list(range(48, 58))
_DB_MASK: Optional[np.ndarray] = None  # built once in main()


def _build_db_mask() -> np.ndarray:
    mask = np.ones(FEATURE_DIM, dtype=np.float32)
    for s in _DB_FEATURE_SLOTS:
        mask[s] = 0.0
    return mask


def _db_quality(wdl: Optional[str], dtm: Optional[int]) -> Optional[float]:
    """Ground-truth quality in [0,1] for one move.  None when WDL unknown."""
    if not wdl or wdl == "unknown":
        return None
    if wdl in ("win", "loss") and dtm is not None:
        return float(dtm_quality(wdl, dtm))
    return WDL_QUALITY.get(wdl)


def _board_phase(board) -> str:
    try:
        if board.phase == "place":
            return "placement"
        pieces = sum(board.pieces_on_board.values())
        return "endgame" if pieces <= 8 else "midgame"
    except Exception:
        return "midgame"


def _spearman_r(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation without scipy."""
    if len(xs) < 2:
        return float("nan")
    a = np.array(xs, dtype=np.float64)
    b = np.array(ys, dtype=np.float64)
    ra = np.argsort(np.argsort(a)).astype(np.float64)
    rb = np.argsort(np.argsort(b)).astype(np.float64)
    denom = np.std(ra) * np.std(rb)
    if denom < 1e-12:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def _pearson_r(xs: List[float], ys: List[float]) -> float:
    if len(xs) < 2:
        return float("nan")
    a = np.array(xs, dtype=np.float64)
    b = np.array(ys, dtype=np.float64)
    denom = np.std(a) * np.std(b)
    if denom < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# ── per-position eval ──────────────────────────────────────────────────────────

def eval_position(
    model: SentinelNet,
    board: BoardState,
    player: str,
    db: ExternalSolvedDB,
    db_mask: np.ndarray,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Score all legal moves at one position and return metric primitives.

    Returns None when there are no legal moves or the DB has no data.
    """
    moves = _enumerate_legal_moves(board, player)
    if not moves:
        return None

    # DB ground truth for every move.
    all_db = db.query_all_moves(board, player) if db.is_available() else []
    if not all_db:
        return None

    db_by_key: Dict[tuple, Dict] = {}
    for entry in all_db:
        mv = entry.get("move", {})
        k = (mv.get("from"), mv.get("to"), mv.get("capture"))
        db_by_key[k] = entry

    # Heuristic context (same as training).
    raw_scores = _heuristic_scores(board, moves, player)
    norm_scores = _normalise_scores(raw_scores)
    ranks = _ranks_desc(raw_scores)
    n_legal = len(moves)

    # Build feature matrix — include full ctx so counterfactual slots fill
    # normally, then zero DB slots before the forward pass.
    feats = np.zeros((n_legal, FEATURE_DIM), dtype=np.float32)
    for i, mv in enumerate(moves):
        ctx = {
            "all_moves": all_db,
            "heuristic_rank": ranks[i],
            "n_legal": n_legal,
            "heuristic_score_norm": norm_scores[i],
        }
        try:
            feats[i] = build_move_features(board, mv, player, ctx)
        except Exception:
            feats[i] = 0.0
    feats = feats * db_mask  # zero oracle slots → inference conditions

    with torch.no_grad():
        x = torch.from_numpy(feats).to(device)
        scores = model(x).reshape(-1).cpu().numpy().tolist()

    # Per-move ground-truth qualities.
    db_qualities: List[Optional[float]] = []
    db_wdls: List[str] = []
    db_dtms: List[Optional[int]] = []
    for mv in moves:
        k = (mv.get("from"), mv.get("to"), mv.get("capture"))
        entry = db_by_key.get(k, {})
        wdl = entry.get("wdl", "unknown")
        dtm = entry.get("dtm")
        db_wdls.append(wdl)
        db_dtms.append(dtm)
        db_qualities.append(_db_quality(wdl, dtm))

    phase = _board_phase(board)

    return {
        "n_moves": n_legal,
        "scores": scores,
        "db_wdls": db_wdls,
        "db_dtms": db_dtms,
        "db_qualities": db_qualities,
        "phase": phase,
    }


# ── metric aggregation ─────────────────────────────────────────────────────────

class MetricAccumulator:
    def __init__(self):
        # direction accuracy
        self.win_correct = self.win_total = 0
        self.loss_correct = self.loss_total = 0
        self.draw_correct = self.draw_total = 0

        # top-1 position metrics
        self.top1_win_correct = self.top1_win_positions = 0  # win available, did sentinel pick one?
        self.top1_exact_match = self.top1_positions = 0      # sentinel top-1 == DB top-1 exact
        self.critical_miss = 0   # win available, sentinel scored a loss move highest

        # bad-move detection
        self.bad_move_flagged = self.bad_move_total = 0

        # correlation data
        self.corr_sentinel: List[float] = []
        self.corr_db: List[float] = []
        self.dtm_sentinel: List[float] = []
        self.dtm_db: List[float] = []

        # score distributions by WDL
        self.scores_by_wdl: Dict[str, List[float]] = defaultdict(list)

        # phase breakdown
        self.phase_win: Dict[str, List[int]] = defaultdict(lambda: [0, 0])   # [correct, total]
        self.phase_loss: Dict[str, List[int]] = defaultdict(lambda: [0, 0])

        self.positions = 0
        self.positions_with_db = 0

    def update(self, pos: Dict[str, Any]) -> None:
        scores = pos["scores"]
        wdls = pos["db_wdls"]
        dtms = pos["db_dtms"]
        quals = pos["db_qualities"]
        phase = pos["phase"]
        n = pos["n_moves"]
        self.positions += 1

        # ---- per-move metrics ----
        has_win = any(w == "win" for w in wdls)
        has_loss = any(w == "loss" for w in wdls)
        known_any = any(q is not None for q in quals)
        if known_any:
            self.positions_with_db += 1

        for i in range(n):
            s = scores[i]
            wdl = wdls[i]
            dtm = dtms[i]
            q = quals[i]

            self.scores_by_wdl[wdl].append(s)

            if wdl == "win":
                correct = int(s > 0.5)
                self.win_correct += correct
                self.win_total += 1
                self.phase_win[phase][0] += correct
                self.phase_win[phase][1] += 1
            elif wdl == "loss":
                correct = int(s < 0.5)
                self.loss_correct += correct
                self.loss_total += 1
                self.phase_loss[phase][0] += correct
                self.phase_loss[phase][1] += 1

                # bad-move (opponent wins quickly) detection
                if dtm is not None and abs(dtm) <= BAD_MOVE_DTM_THRESHOLD:
                    self.bad_move_total += 1
                    if s < 0.4:
                        self.bad_move_flagged += 1
            elif wdl == "draw":
                self.draw_correct += int(abs(s - 0.5) < 0.25)
                self.draw_total += 1

            # Correlation data (all moves with known DB quality)
            if q is not None:
                self.corr_sentinel.append(s)
                self.corr_db.append(q)
                # DTM-graded subset only (win or loss with DTM)
                if wdl in ("win", "loss") and dtm is not None:
                    self.dtm_sentinel.append(s)
                    self.dtm_db.append(float(dtm_quality(wdl, dtm)))

        # ---- top-1 position metrics ----
        if not known_any:
            return

        # DB best quality per move (None for unknown)
        db_q_list = [q if q is not None else -1.0 for q in quals]
        best_db_idx = int(np.argmax(db_q_list))
        best_db_wdl = wdls[best_db_idx]
        best_sentinel_idx = int(np.argmax(scores))

        self.top1_positions += 1
        # Exact match: sentinel's top move is the same index as DB's top move
        if best_sentinel_idx == best_db_idx:
            self.top1_exact_match += 1

        # Win-availability top-1: if a win exists, did sentinel pick a win?
        if has_win:
            self.top1_win_positions += 1
            if wdls[best_sentinel_idx] == "win":
                self.top1_win_correct += 1
            # Critical miss: win available, sentinel's top is a loss
            if has_loss and wdls[best_sentinel_idx] == "loss":
                self.critical_miss += 1

    def report(self) -> Dict[str, Any]:
        def safe_div(a, b):
            return round(a / b, 4) if b else None

        win_acc = safe_div(self.win_correct, self.win_total)
        loss_acc = safe_div(self.loss_correct, self.loss_total)
        total_dir = self.win_total + self.loss_total
        overall_acc = safe_div(self.win_correct + self.loss_correct, total_dir)

        top1_win_rate = safe_div(self.top1_win_correct, self.top1_win_positions)
        top1_exact = safe_div(self.top1_exact_match, self.top1_positions)
        critical_miss_rate = safe_div(self.critical_miss, self.top1_win_positions)
        bad_recall = safe_div(self.bad_move_flagged, self.bad_move_total)

        spearman = _spearman_r(self.corr_sentinel, self.corr_db)
        dtm_pearson = _pearson_r(self.dtm_sentinel, self.dtm_db)

        score_by_wdl = {}
        for wdl, vals in self.scores_by_wdl.items():
            arr = np.array(vals)
            score_by_wdl[wdl] = {"mean": round(float(arr.mean()), 4),
                                  "std": round(float(arr.std()), 4),
                                  "n": len(vals)}

        phase_breakdown = {}
        for phase in set(list(self.phase_win.keys()) + list(self.phase_loss.keys())):
            wc, wt = self.phase_win[phase]
            lc, lt = self.phase_loss[phase]
            phase_breakdown[phase] = {
                "win_acc": safe_div(wc, wt),
                "loss_acc": safe_div(lc, lt),
                "win_n": wt, "loss_n": lt,
            }

        return {
            "positions_evaluated": self.positions,
            "positions_with_db": self.positions_with_db,
            "win_acc": win_acc,
            "loss_acc": loss_acc,
            "overall_acc": overall_acc,
            "draw_acc": safe_div(self.draw_correct, self.draw_total),
            "top1_win_rate": top1_win_rate,
            "top1_exact_match": top1_exact,
            "critical_miss_rate": critical_miss_rate,
            "bad_move_recall": bad_recall,
            "bad_move_n": self.bad_move_total,
            "spearman_r": round(spearman, 4) if spearman == spearman else None,
            "dtm_pearson_r": round(dtm_pearson, 4) if dtm_pearson == dtm_pearson else None,
            "correlation_n": len(self.corr_sentinel),
            "dtm_correlation_n": len(self.dtm_sentinel),
            "score_by_wdl": score_by_wdl,
            "phase_breakdown": phase_breakdown,
            "counts": {
                "win_moves": self.win_total,
                "loss_moves": self.loss_total,
                "draw_moves": self.draw_total,
            },
        }


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> int:
    p = argparse.ArgumentParser(description="Grade sentinel against Malom DB")
    p.add_argument("--checkpoint", required=True, help="Path to sentinel .pt checkpoint")
    p.add_argument("--game-dir", default="data/games")
    p.add_argument("--human-game-dir", default=None)
    p.add_argument("--db-path", default="",
                   help="Malom DB path (defaults to config in checkpoint)")
    p.add_argument("--limit", type=int, default=None, help="Max game files to evaluate")
    p.add_argument("--output", default=None, help="Write JSON results to this path")
    p.add_argument("--device", default="cpu")
    p.add_argument("--quiet", action="store_true", help="Suppress per-epoch progress")
    args = p.parse_args()

    device = torch.device(args.device)

    # ── Load model ──────────────────────────────────────────────────────────────
    if not os.path.exists(args.checkpoint):
        print(f"ERROR: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 1

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg_dict = ckpt.get("config") if isinstance(ckpt, dict) else None
    config = SentinelConfig.from_dict(cfg_dict) if cfg_dict else SentinelConfig()
    aux_wdl = bool(ckpt.get("aux_wdl", False)) if isinstance(ckpt, dict) else False
    state_dict = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    model = SentinelNet(
        input_dim=config.input_dim,
        hidden_dims=config.hidden_dims,
        dropout=config.dropout,   # must match checkpoint architecture
        aux_wdl=aux_wdl,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()  # disables dropout at inference without changing architecture
    print(f"Loaded sentinel from {args.checkpoint}  (aux_wdl={aux_wdl})")

    # ── Load DB ─────────────────────────────────────────────────────────────────
    db_path = args.db_path or config.external_db_path
    db = ExternalSolvedDB(db_path=db_path, enabled=bool(db_path))
    if not db.is_available():
        print("ERROR: Malom DB not available — cannot evaluate without ground truth.",
              file=sys.stderr)
        print(f"  Tried: {db_path!r}", file=sys.stderr)
        return 1
    print(f"Malom DB available: {db_path}")

    # ── Feature mask ────────────────────────────────────────────────────────────
    db_mask = _build_db_mask()
    print(f"DB feature slots zeroed for inference-realistic eval: {_DB_FEATURE_SLOTS}")

    # ── Collect game files ───────────────────────────────────────────────────────
    import glob
    paths = sorted(glob.glob(os.path.join(args.game_dir, "**", "*.jsonl"), recursive=True))
    if args.human_game_dir:
        paths += sorted(glob.glob(os.path.join(args.human_game_dir, "**", "*.jsonl"), recursive=True))
    if args.limit:
        paths = paths[:args.limit]
    print(f"\nEvaluating {len(paths)} game files...")

    # ── Eval loop ────────────────────────────────────────────────────────────────
    acc = MetricAccumulator()
    t0 = time.time()
    games_done = 0
    positions_done = 0

    for path in paths:
        for record in _iter_game_records(path):
            moves_log = record.get("moves") or []
            for log_move in moves_log:
                fen = log_move.get("board_fen_before")
                if not fen:
                    continue
                board = _board_from_fen_before(fen)
                if board is None:
                    continue
                player = log_move.get("color") or getattr(board, "turn", "W")
                try:
                    result = eval_position(model, board, player, db, db_mask, device)
                except Exception as exc:
                    logger.debug("position failed: %s", exc)
                    continue
                if result is not None:
                    acc.update(result)
                    positions_done += 1
            games_done += 1
            if not args.quiet and games_done % 50 == 0:
                elapsed = time.time() - t0
                print(f"  {games_done}/{len(paths)} games, {positions_done} positions "
                      f"({elapsed:.0f}s)", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {games_done} games, {positions_done} positions in {elapsed:.1f}s\n")

    # ── Report ───────────────────────────────────────────────────────────────────
    results = acc.report()

    def pct(v):
        return f"{v * 100:.1f}%" if v is not None else "n/a"

    print("=" * 60)
    print("  SENTINEL EVALUATION REPORT")
    print("=" * 60)
    print(f"  Positions evaluated   : {results['positions_evaluated']:,}")
    print(f"  Positions with DB data: {results['positions_with_db']:,}")
    print()
    print("  DIRECTION ACCURACY")
    print(f"    Win  moves scored > 0.5 : {pct(results['win_acc'])}  (n={results['counts']['win_moves']:,})")
    print(f"    Loss moves scored < 0.5 : {pct(results['loss_acc'])}  (n={results['counts']['loss_moves']:,})")
    print(f"    Overall (win+loss)      : {pct(results['overall_acc'])}")
    print(f"    Draw moves near 0.5     : {pct(results['draw_acc'])}  (n={results['counts']['draw_moves']:,})")
    print()
    print("  TOP-1 POSITION METRICS")
    print(f"    Win available → sentinel picks a win  : {pct(results['top1_win_rate'])}")
    print(f"    Sentinel top-1 == DB top-1 (exact)    : {pct(results['top1_exact_match'])}")
    print(f"    Critical miss (win avail, scores loss) : {pct(results['critical_miss_rate'])}")
    print()
    print("  BAD-MOVE DETECTION")
    print(f"    Loss-in-≤{BAD_MOVE_DTM_THRESHOLD} moves scored < 0.4: {pct(results['bad_move_recall'])}  (n={results['bad_move_n']:,})")
    print()
    print("  RANK CORRELATION vs DB")
    print(f"    Spearman r (all moves with DB quality) : {results['spearman_r']}  (n={results['correlation_n']:,})")
    print(f"    Pearson r  (DTM-graded win+loss only)  : {results['dtm_pearson_r']}  (n={results['dtm_correlation_n']:,})")
    print()
    print("  MEAN SENTINEL SCORE BY DB CATEGORY")
    for wdl in ("win", "draw", "loss", "unknown"):
        d = results["score_by_wdl"].get(wdl)
        if d:
            print(f"    {wdl:8s}: {d['mean']:.3f} ± {d['std']:.3f}  (n={d['n']:,})")
    print()
    print("  PHASE BREAKDOWN")
    for phase in ("placement", "midgame", "endgame"):
        pb = results["phase_breakdown"].get(phase)
        if pb:
            print(f"    {phase:12s}: win_acc={pct(pb['win_acc'])} (n={pb['win_n']:,})"
                  f"  loss_acc={pct(pb['loss_acc'])} (n={pb['loss_n']:,})")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults written to {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
