//! Tactical scanning. Ports `ai/game_ai.py::_immediate_mill_threats` (closing
//! squares of opponent two-configs reachable in one move) used for the
//! mandatory-block rule and move ordering. See `docs/RUST_INTEGRATION_PLAN.md` §8.

use crate::board::{get_phase, ADJACENCY};
use crate::mills::{MILL_MASKS, SQUARE_MILLS};
use crate::types::{Board, Color, Phase, N_SQUARES};

/// Squares where the side-to-move's OPPONENT could close a mill next move.
/// Mirrors `_immediate_mill_threats(board)`: the opponent of `board.turn`.
/// In move phase only counts closing squares with an adjacent opponent piece
/// (excluding pieces inside the mill); in fly any empty closing square; in
/// place any empty closing square.
pub fn immediate_mill_threats(board: &Board) -> u32 {
    let opp = board.side_to_move.opponent();
    let opp_bits = board.bits(opp);
    let empty = board.empty();
    let opp_phase = get_phase(board, opp);
    let mut threats = 0u32;
    for &mm in &MILL_MASKS {
        if (opp_bits & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            let closing = empty & mm;
            let closing_idx = closing.trailing_zeros() as usize;
            let reachable = match opp_phase {
                Phase::Place => true,
                Phase::Fly => true,
                Phase::Move => (ADJACENCY[closing_idx] & opp_bits & !mm) != 0,
            };
            if reachable {
                threats |= closing;
            }
        }
    }
    threats
}

/// True if placing/moving to `to` for `color` forms a mill (used in ordering).
pub fn move_forms_mill(board: &Board, color: Color, from: Option<u8>, to: u8) -> bool {
    let mut bits = board.bits(color);
    if let Some(f) = from {
        bits &= !(1u32 << f);
    }
    bits |= 1u32 << to;
    // Each square belongs to exactly 2 mills — only check those.
    for &mi in &SQUARE_MILLS[to as usize] {
        let mm = MILL_MASKS[mi as usize];
        if (bits & mm) == mm {
            return true;
        }
    }
    false
}

#[allow(dead_code)]
fn _unused_marker() -> usize {
    N_SQUARES
}
