//! Legal move generation. Mirrors `game/board.py` (legal_placements,
//! legal_moves, legal_captures) + `game/rules.py::get_all_legal_moves`.
//! See `docs/RUST_INTEGRATION_PLAN.md` §6.

use crate::board::{get_phase, ADJACENCY};
use crate::mills::{MILL_MASKS, SQUARE_MILLS};
use crate::types::{Board, Color, Move, Phase, N_SQUARES};

/// True if `color` has at least one legal (partial) move in movement/fly phase.
/// Only meaningful for non-placement phases (used by is_blocked / terminal).
pub fn has_any_move(board: &Board, color: Color) -> bool {
    let phase = get_phase(board, color);
    let empty = board.empty();
    match phase {
        Phase::Place => empty != 0,
        Phase::Fly => board.count(color) > 0 && empty != 0,
        Phase::Move => {
            let own = board.bits(color);
            for sq in 0..N_SQUARES as u8 {
                if own & (1 << sq) != 0 && (ADJACENCY[sq as usize] & empty) != 0 {
                    return true;
                }
            }
            false
        }
    }
}

/// Does the partial move (ignoring capture) put a piece of `color` into a mill
/// containing `to`? Mirrors `does_form_mill`.
fn does_form_mill(board: &Board, color: Color, from: Option<u8>, to: u8) -> bool {
    let mut bits = board.bits(color);
    if let Some(f) = from {
        bits &= !(1u32 << f);
    }
    bits |= 1u32 << to;
    for &mi in &SQUARE_MILLS[to as usize] {
        let mm = MILL_MASKS[mi as usize];
        if (bits & mm) == mm {
            return true;
        }
    }
    false
}

/// Legal capture targets (POSITIONS order). Mirrors `legal_captures`: prefer
/// opponent pieces NOT in a mill; fall back to all if every opp piece is milled.
fn legal_captures(board: &Board, color: Color) -> Vec<u8> {
    let opp = color.opponent();
    let opp_bits = board.bits(opp);
    let mut non_mill: Vec<u8> = Vec::new();
    let mut all: Vec<u8> = Vec::new();
    // Iterate over set bits only instead of scanning all 24 squares.
    let mut bits = opp_bits;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        all.push(sq);
        let in_mill = SQUARE_MILLS[sq as usize]
            .iter()
            .any(|&mi| (opp_bits & MILL_MASKS[mi as usize]) == MILL_MASKS[mi as usize]);
        if !in_mill {
            non_mill.push(sq);
        }
    }
    if non_mill.is_empty() { all } else { non_mill }
}

/// All complete legal moves for `board.side_to_move`, in the same order Python's
/// `get_all_legal_moves` produces them.
pub fn legal_moves(board: &Board) -> Vec<Move> {
    let color = board.side_to_move;
    let phase = get_phase(board, color);
    let empty = board.empty();
    let own = board.bits(color);

    // Partial moves in POSITIONS order.
    let mut partial: Vec<(Option<u8>, u8)> = Vec::new();
    match phase {
        Phase::Place => {
            for sq in 0..N_SQUARES as u8 {
                if empty & (1 << sq) != 0 {
                    partial.push((None, sq));
                }
            }
        }
        Phase::Fly => {
            // own pieces in POSITIONS order, then every empty square in order.
            for src in 0..N_SQUARES as u8 {
                if own & (1 << src) == 0 {
                    continue;
                }
                for tgt in 0..N_SQUARES as u8 {
                    if empty & (1 << tgt) != 0 {
                        partial.push((Some(src), tgt));
                    }
                }
            }
        }
        Phase::Move => {
            for src in 0..N_SQUARES as u8 {
                if own & (1 << src) == 0 {
                    continue;
                }
                let adj = ADJACENCY[src as usize] & empty;
                for tgt in 0..N_SQUARES as u8 {
                    if adj & (1 << tgt) != 0 {
                        partial.push((Some(src), tgt));
                    }
                }
            }
        }
    }

    let mut complete: Vec<Move> = Vec::with_capacity(partial.len());
    for (from, to) in partial {
        if does_form_mill(board, color, from, to) {
            for cap in legal_captures(board, color) {
                complete.push(Move {
                    from,
                    to,
                    capture: Some(cap),
                });
            }
        } else {
            complete.push(Move {
                from,
                to,
                capture: None,
            });
        }
    }
    complete
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn placement_count_empty_board() {
        let bd = Board {
            white: 0,
            black: 0,
            white_placed: 0,
            black_placed: 0,
            side_to_move: Color::White,
        };
        assert_eq!(legal_moves(&bd).len(), 24);
    }

    #[test]
    fn mill_expands_captures() {
        // White has a7,d7 (idx 0,1); placing g7 (idx2) forms a mill.
        // Black has one piece at idx 5 (not in a mill) -> capturable.
        let bd = Board {
            white: (1 << 0) | (1 << 1),
            black: 1 << 5,
            white_placed: 2,
            black_placed: 1,
            side_to_move: Color::White,
        };
        let moves = legal_moves(&bd);
        let to_g7: Vec<_> = moves.iter().filter(|mv| mv.to == 2).collect();
        assert_eq!(to_g7.len(), 1);
        assert_eq!(to_g7[0].capture, Some(5));
    }
}
