//! Self-contained negamax + alpha-beta + iterative deepening search.
//!
//! Coarse-grained: Python calls `iterative_deepening` once via the PyO3 binding;
//! the whole tree is searched in Rust with its own Zobrist + TT. This engine is
//! NOT required to return the same move as the Python AI — only legal, sane
//! play. See `docs/RUST_INTEGRATION_PLAN.md` §10.

use std::time::Instant;

use crate::board::{make_move, terminal_winner, ADJACENCY, get_phase};
use crate::heuristics::{evaluate_v2, INF};
use crate::hash::{TranspositionTable, TtEntry, Zobrist, EXACT, LOWER_BOUND, UPPER_BOUND};
use crate::movegen::legal_moves;
use crate::tactics::move_forms_mill;
use crate::types::{Board, Color, Move, Phase, FULL_MASK};

pub struct SearchResult {
    pub best_move: Option<Move>,
    pub score: i64,
    pub nodes: u64,
    pub depth_reached: u8,
}

struct Searcher {
    zobrist: Zobrist,
    tt: TranspositionTable,
    nodes: u64,
    deadline: Instant,
    aborted: bool,
}

const ABORT_SCORE: i64 = i64::MIN + 1;

impl Searcher {
    fn ordered_moves(&self, board: &Board, tt_best: Option<u16>) -> Vec<Move> {
        let mut moves = legal_moves(board);
        let color = board.side_to_move;
        // Score: mill-forming + captures first, then a light static touch.
        moves.sort_by_key(|mv| {
            let mut s = 0i64;
            if mv.capture.is_some() {
                s -= 1000;
            }
            if move_forms_mill(board, color, mv.from, mv.to) {
                s -= 500;
            }
            s
        });
        if let Some(bi) = tt_best {
            let bi = bi as usize;
            if bi < moves.len() {
                let m = moves.remove(bi.min(moves.len() - 1));
                moves.insert(0, m);
            }
        }
        moves
    }

    // SE-11: extend by 1 ply for moves that form a mill at the first opponent ply.
    // `first_opp_ply` is true only when called directly from root(); all recursive
    // calls pass false, so the extension applies at most once per root move.
    fn negamax(&mut self, board: &Board, depth: u8, mut alpha: i64, beta: i64, first_opp_ply: bool) -> i64 {
        if self.aborted {
            return ABORT_SCORE;
        }
        self.nodes += 1;
        if self.nodes & 2047 == 0 && Instant::now() >= self.deadline {
            self.aborted = true;
            return ABORT_SCORE;
        }

        let color = board.side_to_move;
        if let Some(winner) = terminal_winner(board) {
            // From side-to-move perspective: losing if winner != stm.
            return if winner == color {
                INF - depth as i64
            } else {
                -(INF - depth as i64)
            };
        }

        let key = self.zobrist.hash(board);
        let mut tt_best: Option<u16> = None;
        if let Some(e) = self.tt.lookup(key) {
            if e.best_idx != u16::MAX {
                tt_best = Some(e.best_idx);
            }
            if e.depth >= depth {
                match e.flag {
                    EXACT => return e.score,
                    LOWER_BOUND if e.score >= beta => return e.score,
                    UPPER_BOUND if e.score <= alpha => return e.score,
                    _ => {}
                }
            }
        }

        if depth == 0 {
            return evaluate_v2(board, color);
        }

        let alpha_orig = alpha;
        let moves = self.ordered_moves(board, tt_best);
        if moves.is_empty() {
            // No moves: treat as loss for side-to-move (blocked).
            return -(INF - depth as i64);
        }

        let mut best_score = -INF * 4;
        let mut best_idx: u16 = u16::MAX;
        for (i, mv) in moves.iter().enumerate() {
            let nb = make_move(board, mv);
            // SE-11: extend by 1 for mill-forming moves at first opponent ply.
            let se11_ext: u8 = if first_opp_ply && move_forms_mill(board, color, mv.from, mv.to) {
                1
            } else {
                0
            };
            let score = -self.negamax(&nb, depth - 1 + se11_ext, -beta, -alpha, false);
            if self.aborted {
                return ABORT_SCORE;
            }
            if score > best_score {
                best_score = score;
                best_idx = i as u16;
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }

        let flag = if best_score <= alpha_orig {
            UPPER_BOUND
        } else if best_score >= beta {
            LOWER_BOUND
        } else {
            EXACT
        };
        self.tt.store(TtEntry {
            key,
            depth,
            score: best_score,
            flag,
            best_idx,
        });
        best_score
    }

    fn root(&mut self, board: &Board, depth: u8) -> (Option<Move>, i64) {
        let moves = self.ordered_moves(board, None);
        if moves.is_empty() {
            return (None, -INF);
        }
        let color = board.side_to_move;
        let in_placement = get_phase(board, color) == Phase::Place;
        let mut alpha = -INF * 4;
        let beta = INF * 4;
        let mut best_move = Some(moves[0]);
        for mv in moves.iter() {
            let nb = make_move(board, mv);
            // B-64: dead/near-dead placement penalty (mirrors Python tactical_move_bonus).
            // Only penalise non-mill placements with 0 or 1 free adjacent squares.
            let b64_penalty: i64 = if in_placement
                && mv.from.is_none()
                && !move_forms_mill(board, color, mv.from, mv.to)
            {
                let sq = mv.to as usize;
                let occupied_after = nb.white | nb.black;
                let free_after = (ADJACENCY[sq] & !occupied_after & FULL_MASK).count_ones();
                if free_after == 0 {
                    1500
                } else if free_after == 1 {
                    400
                } else {
                    0
                }
            } else {
                0
            };
            // SE-11: first_opp_ply=true so negamax extends mill-forming opponent replies.
            let score = -self.negamax(&nb, depth - 1, -beta, -alpha, true) - b64_penalty;
            if self.aborted {
                break;
            }
            if score > alpha {
                alpha = score;
                best_move = Some(*mv);
            }
        }
        (best_move, alpha)
    }
}

/// Iterative deepening with a wall-clock time limit. Returns the best move found
/// at the deepest fully (or partially) completed iteration.
pub fn iterative_deepening(board: &Board, max_depth: u8, time_limit_ms: u64) -> SearchResult {
    let deadline = Instant::now() + std::time::Duration::from_millis(time_limit_ms.max(1));
    let mut searcher = Searcher {
        zobrist: Zobrist::new(),
        tt: TranspositionTable::new(),
        nodes: 0,
        deadline,
        aborted: false,
    };

    let mut best = SearchResult {
        best_move: None,
        score: 0,
        nodes: 0,
        depth_reached: 0,
    };

    let cap = max_depth.max(1);
    for d in 1..=cap {
        let (mv, score) = searcher.root(board, d);
        if searcher.aborted {
            // Keep previous completed depth's result if we have one.
            if best.best_move.is_none() {
                best.best_move = mv;
                best.score = score;
                best.depth_reached = d;
            }
            break;
        }
        best.best_move = mv;
        best.score = score;
        best.depth_reached = d;
        best.nodes = searcher.nodes;
        if score.abs() >= INF - 100 {
            break; // forced mate found
        }
    }
    best.nodes = searcher.nodes;
    best
}

/// Convenience entry used by the PyO3 binding.
pub fn get_best_move(
    white: u32,
    black: u32,
    white_placed: u8,
    black_placed: u8,
    stm: Color,
    max_depth: u8,
    time_limit_ms: u64,
) -> SearchResult {
    let board = Board {
        white,
        black,
        white_placed,
        black_placed,
        side_to_move: stm,
    };
    iterative_deepening(&board, max_depth, time_limit_ms)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn finds_a_move_on_empty_board() {
        let r = get_best_move(0, 0, 0, 0, Color::White, 4, 200);
        assert!(r.best_move.is_some());
        assert!(r.nodes > 0);
    }

    #[test]
    fn takes_immediate_mill_capture() {
        // White a7,d7 (0,1) about to place g7 (2) forming a mill, capturing black.
        let r = get_best_move(
            (1 << 0) | (1 << 1),
            (1 << 5) | (1 << 13),
            2,
            2,
            Color::White,
            3,
            500,
        );
        let mv = r.best_move.unwrap();
        // The strongest move forms the mill at g7 with a capture.
        assert!(mv.capture.is_some() || mv.to == 2);
    }
}
