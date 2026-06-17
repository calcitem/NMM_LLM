"""scripts/gen_stage1_data.py — Build Stage 1 imitation-learning dataset from HumanDB.

Queries the human_db.sqlite moves table (total >= 5), reconstructs each canonical board,
applies all 8 D4 symmetry augmentations, and emits (state_tensor, phase_id, primary_action,
weight) tuples.  weight = (wins + 0.5*draws) / total so CE loss trains toward high-quality
moves.

Output: learned_ai/data/stage1_imitation.npz
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))


# ── Board reconstruction from state_key ─────────────────────────────────────

def _board_from_state_key(state_key: str):
    """Reconstruct a BoardState from the canonical state_key string."""
    from game.board import BoardState, POSITIONS

    parts = state_key.split("|")
    canon24     = parts[0]          # 24-char '.', 'W', 'B'
    turn        = parts[1]          # 'W' or 'B'
    placed_w    = int(parts[3])
    placed_b    = int(parts[4])
    on_w        = int(parts[5])
    on_b        = int(parts[6])

    positions = {p: ('' if c == '.' else c) for p, c in zip(POSITIONS, canon24)}
    cap_by_white = max(0, placed_b - on_b)
    cap_by_black = max(0, placed_w - on_w)
    return BoardState(
        positions=positions,
        turn=turn,
        pieces_on_board={"W": on_w, "B": on_b},
        pieces_placed={"W": placed_w, "B": placed_b},
        pieces_captured={"W": cap_by_white, "B": cap_by_black},
        hash_key=0,
    )


def _augment_board(board, sym_idx: int):
    """Return a new BoardState with positions transformed by sym_idx (None on failure)."""
    if sym_idx == 0:
        return board
    from game.board import BoardState
    from ai.board_symmetry import transform_pos

    new_positions = {}
    for pos, val in board.positions.items():
        new_pos = transform_pos(pos, sym_idx)
        if new_pos is None:
            return None
        new_positions[new_pos] = val

    return BoardState(
        positions=new_positions,
        turn=board.turn,
        pieces_on_board=board.pieces_on_board.copy(),
        pieces_placed=board.pieces_placed.copy(),
        pieces_captured=board.pieces_captured.copy(),
        hash_key=0,
    )


def _parse_notation(notation: str) -> dict:
    """Parse a move notation string into a move dict."""
    notation = notation.replace("×", "x")
    capture = None
    if "x" in notation:
        main, capture = notation.split("x", 1)
    else:
        main = notation
    if "-" in main:
        from_sq, to_sq = main.split("-", 1)
        return {"from": from_sq, "to": to_sq, "capture": capture}
    return {"from": None, "to": main, "capture": capture}


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Generate Stage 1 imitation-learning dataset")
    p.add_argument("--db",        default=str(_ROOT / "data" / "human_db.sqlite"))
    p.add_argument("--out",       default=str(_ROOT / "learned_ai" / "data" / "stage1_imitation.npz"))
    p.add_argument("--min-total", type=int, default=5,
                   help="Minimum games per (position, move) pair to include")
    args = p.parse_args()

    from ai.board_symmetry import transform_notation
    from learned_ai.models.state_encoder import encode_state, detect_phase
    from learned_ai.models.action_encoder import encode_action

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        "SELECT state_key, notation, wins, losses, draws, total "
        "FROM moves WHERE total >= ?",
        (args.min_total,),
    ).fetchall()
    conn.close()

    print(f"Loaded {len(rows):,} (position, move) pairs from DB (total >= {args.min_total})")

    all_states:   list[np.ndarray] = []
    all_phases:   list[int]        = []
    all_actions:  list[int]        = []
    all_weights:  list[float]      = []

    skipped = 0

    for state_key, canon_notation, wins, losses, draws, total in rows:
        try:
            canon_board = _board_from_state_key(state_key)
        except Exception:
            skipped += 1
            continue

        weight = (wins + 0.5 * draws) / max(1, total)

        for sym_idx in range(8):
            aug_board = _augment_board(canon_board, sym_idx)
            if aug_board is None:
                continue

            aug_notation = transform_notation(canon_notation, sym_idx)
            if aug_notation is None:
                continue

            try:
                move_dict = _parse_notation(aug_notation)
                primary_idx, _cap_idx = encode_action(move_dict)
                state_tensor = encode_state(aug_board)
                phase_id = detect_phase(aug_board)
            except Exception:
                skipped += 1
                continue

            all_states.append(state_tensor.numpy())
            all_phases.append(phase_id)
            all_actions.append(primary_idx)
            all_weights.append(weight)

    print(f"Generated {len(all_states):,} augmented samples  (skipped {skipped})")
    if not all_states:
        print("ERROR: no samples generated")
        sys.exit(1)

    phase_counts: dict[int, int] = {}
    for ph in all_phases:
        phase_counts[ph] = phase_counts.get(ph, 0) + 1
    print(f"Phase distribution: {dict(sorted(phase_counts.items()))}")

    weights_arr = np.array(all_weights, dtype=np.float32)
    print(f"Weight range: [{weights_arr.min():.3f}, {weights_arr.max():.3f}]  "
          f"mean={weights_arr.mean():.3f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        states=np.array(all_states,   dtype=np.float32),
        phase_ids=np.array(all_phases, dtype=np.int8),
        primary_actions=np.array(all_actions, dtype=np.int32),
        weights=weights_arr,
    )
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
