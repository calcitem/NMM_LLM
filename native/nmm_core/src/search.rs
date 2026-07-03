//! Self-contained negamax + alpha-beta + iterative deepening search.
//!
//! Coarse-grained: Python calls `iterative_deepening` once via the PyO3 binding;
//! the whole tree is searched in Rust with its own Zobrist + TT. This engine is
//! NOT required to return the same move as the Python AI — only legal, sane
//! play. See `docs/RUST_INTEGRATION_PLAN.md` §10.

use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use std::time::Instant;
use memmap2::Mmap;

use crate::board::{make_move, terminal_winner, ADJACENCY, get_phase};
use crate::db_probe;
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

pub struct RootMoveScore {
    pub mv: Move,
    pub score: i64,
}

pub struct SearchResultScored {
    pub scored_moves: Vec<RootMoveScore>,
    pub nodes: u64,
    pub depth_reached: u8,
}

struct Searcher {
    zobrist: Zobrist,
    tt: Arc<TranspositionTable>,
    nodes: u64,
    deadline: Instant,
    aborted: bool,
    killers: [[Option<Move>; 2]; MAX_PLY],
    history: [[i32; 24]; 25],  // [from_or_24][to]; from=24 for placements
    // T-C1: high-frequency opponent moves that earn a SE-11 depth extension.
    opp_ext_set: HashSet<(Option<u8>, u8, Option<u8>)>,
    // T-C2: mmap'd fullgame DB for in-search probe (probe between terminal check and TT).
    fullgame_db: Option<Arc<Mmap>>,
    // T-C3: mmap'd endgame solved tables, keyed by (nW, nB). O(1) WDL probe.
    endgame_solved_db: Option<Arc<HashMap<(u8, u8), Mmap>>>,
}

const ABORT_SCORE: i64 = i64::MIN + 1;
const MAX_PLY: usize = 64;
const ASP_MARGIN: i64 = 50;

impl Searcher {
    fn store_killer(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY || mv.capture.is_some() { return; }
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
    }

    fn ordered_moves(&self, board: &Board, tt_best: Option<u16>, ply: usize) -> Vec<Move> {
        let mut moves = legal_moves(board);
        let color = board.side_to_move;
        let k = if ply < MAX_PLY { self.killers[ply] } else { [None; 2] };
        moves.sort_by_key(|mv| {
            let mut s = 0i64;
            if mv.capture.is_some() { s -= 2000; }
            if move_forms_mill(board, color, mv.from, mv.to) { s -= 1000; }
            // T-B3: killer moves after captures/mills.
            if k[0] == Some(*mv) { s -= 600; }
            else if k[1] == Some(*mv) { s -= 500; }
            // T-B3: history heuristic as tiebreaker.
            let fi = mv.from.map_or(24usize, |f| f as usize);
            let ti = mv.to as usize;
            if ti < 24 { s -= self.history[fi][ti] as i64; }
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

    // T-B4: Quiescence search — extends depth-0 with captures and mill-forming moves
    // to avoid horizon-effect blunders at tactical boundaries.
    fn qsearch(&mut self, board: &Board, mut alpha: i64, beta: i64) -> i64 {
        if self.aborted { return ABORT_SCORE; }
        self.nodes += 1;
        if self.nodes & 2047 == 0 && Instant::now() >= self.deadline {
            self.aborted = true;
            return ABORT_SCORE;
        }
        let color = board.side_to_move;
        if let Some(winner) = terminal_winner(board) {
            return if winner == color { INF } else { -INF };
        }
        let stand_pat = evaluate_v2(board, color);
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }
        for mv in legal_moves(board).iter() {
            if mv.capture.is_none() && !move_forms_mill(board, color, mv.from, mv.to) {
                continue;
            }
            let nb = make_move(board, mv);
            let score = -self.qsearch(&nb, -beta, -alpha);
            if self.aborted { return ABORT_SCORE; }
            if score >= beta { return beta; }
            if score > alpha { alpha = score; }
        }
        alpha
    }

    // SE-11: extend by 1 ply for moves that form a mill at the first opponent ply.
    // `first_opp_ply` is true only when called directly from root/root_scored; all
    // recursive calls pass false, so the extension fires at most once per root move.
    //
    // T-B1: PVS — first move gets full window; subsequent moves use null window then re-search.
    // T-B2: LMR — late non-tactical moves searched at depth-1 first.
    // T-B3: killers + history updated on beta cutoff.
    // T-B4: qsearch at depth==0.
    // T-B5: null-move pruning at depth>=3 outside fly phase.
    fn negamax(&mut self, board: &Board, depth: u8, mut alpha: i64, beta: i64, first_opp_ply: bool, ply: u8) -> i64 {
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
            return if winner == color {
                INF - depth as i64
            } else {
                -(INF - depth as i64)
            };
        }

        // T-C2: FullGame DB probe (between terminal check and TT, matching Python ordering).
        if let Some(ref db) = self.fullgame_db {
            if let Some(white_score) = db_probe::probe_fullgame(db, board) {
                let stm_score = if color == Color::White { white_score as i64 } else { -(white_score as i64) };
                return if stm_score > 0 {
                    INF - depth as i64
                } else if stm_score < 0 {
                    -(INF - depth as i64)
                } else {
                    0
                };
            }
        }

        // T-C3: EndgameSolvedDB probe — O(1) WDL for post-placement positions ≤7 pieces each.
        // Only fires when both sides have fully placed (≥9 placed) and piece counts are in range.
        if board.white_placed >= 9 && board.black_placed >= 9 {
            let nw = board.white.count_ones() as u8;
            let nb = board.black.count_ones() as u8;
            if (3..=7).contains(&nw) && (3..=7).contains(&nb) {
                if let Some(ref tables) = self.endgame_solved_db {
                    if let Some(table) = tables.get(&(nw, nb)) {
                        if let Some(stm_result) = db_probe::probe_endgame_solved(table, board) {
                            return match stm_result {
                                1  => INF - depth as i64,
                                -1 => -(INF - depth as i64),
                                _  => 0,
                            };
                        }
                    }
                }
            }
        }

        let key = self.zobrist.hash(board);
        let mut tt_best: Option<u16> = None;
        if let Some(e) = self.tt.lookup(key) {
            if e.best_idx != u16::MAX {
                tt_best = Some(e.best_idx);
            }
            if e.depth >= depth {
                let s = e.score as i64;
                match e.flag {
                    EXACT => return s,
                    LOWER_BOUND if s >= beta => return s,
                    UPPER_BOUND if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        // T-B4: quiescence search at horizon.
        if depth == 0 {
            return self.qsearch(board, alpha, beta);
        }

        // T-B5: Null-move pruning (skip in fly phase to avoid zugzwang, and when
        // own side has ≤ 3 pieces where zugzwang risk is highest).
        let phase = get_phase(board, color);
        if depth >= 3
            && beta < INF / 2
            && phase != Phase::Fly
            && board.count(color) > 3
        {
            let null_board = Board { side_to_move: color.opponent(), ..*board };
            let null_score = -self.negamax(&null_board, depth - 3, -beta, -beta + 1, false, ply + 1);
            if !self.aborted && null_score >= beta {
                return beta;
            }
        }

        let alpha_orig = alpha;
        let moves = self.ordered_moves(board, tt_best, ply as usize);
        if moves.is_empty() {
            return -(INF - depth as i64);
        }

        let mut best_score = -INF * 4;
        let mut best_idx: u16 = u16::MAX;
        for (i, mv) in moves.iter().enumerate() {
            let nb = make_move(board, mv);
            let is_tactical = mv.capture.is_some() || move_forms_mill(board, color, mv.from, mv.to);
            // SE-11: extend by 1 at first opponent ply for tactical moves (original) or
            // high-frequency trajectory moves (T-C1: opp_ext_set from trajectory_db).
            let in_opp_ext = first_opp_ply
                && !self.opp_ext_set.is_empty()
                && self.opp_ext_set.contains(&(mv.from, mv.to, mv.capture));
            let se11_ext: u8 = if first_opp_ply && (is_tactical || in_opp_ext) { 1 } else { 0 };

            // T-B1 + T-B2: PVS with LMR for moves after the first.
            let score = if i == 0 {
                -self.negamax(&nb, depth - 1 + se11_ext, -beta, -alpha, false, ply + 1)
            } else {
                // T-B2: reduce late non-tactical moves at depth >= 3.
                let lmr: u8 = if depth >= 3 && i >= 3 && !is_tactical && se11_ext == 0 { 1 } else { 0 };
                // T-B1: null window at (possibly reduced) depth.
                let mut s = -self.negamax(
                    &nb, (depth - 1 + se11_ext).saturating_sub(lmr),
                    -alpha - 1, -alpha, false, ply + 1,
                );
                // Re-search at full depth+window if null window or LMR was wrong.
                if !self.aborted && s > alpha {
                    s = -self.negamax(&nb, depth - 1 + se11_ext, -beta, -alpha, false, ply + 1);
                }
                s
            };

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
                // T-B3: update killers and history on quiet beta cutoff.
                if !is_tactical {
                    self.store_killer(ply as usize, *mv);
                    let fi = mv.from.map_or(24usize, |f| f as usize);
                    let ti = mv.to as usize;
                    if ti < 24 {
                        self.history[fi][ti] = self.history[fi][ti]
                            .saturating_add((depth as i32) * (depth as i32));
                    }
                }
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
        self.tt.store(key, TtEntry {
            depth,
            score: best_score as i32,
            flag,
            best_idx,
        });
        best_score
    }

    fn root(&mut self, board: &Board, depth: u8, alpha_init: i64, beta_init: i64) -> (Option<Move>, i64) {
        let moves = self.ordered_moves(board, None, 0);
        if moves.is_empty() {
            return (None, -INF);
        }
        let color = board.side_to_move;
        let in_placement = get_phase(board, color) == Phase::Place;
        let mut alpha = alpha_init;
        let beta = beta_init;
        let mut best_move = Some(moves[0]);
        for mv in moves.iter() {
            let nb = make_move(board, mv);
            // B-64: dead/near-dead placement penalty (mirrors Python tactical_move_bonus).
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
            let score = -self.negamax(&nb, depth - 1, -beta, -alpha, true, 1) - b64_penalty;
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

    /// Full-window root search: every move gets an independent (-INF, INF) window
    /// so all returned scores are exact. Mirrors `root()` B-64 penalty exactly.
    /// `preferred` is an optional list of (from, to, capture) triples promoted to
    /// the front of the move list for better alpha-beta pruning (M3).
    fn root_scored(&mut self, board: &Board, depth: u8, preferred: &[(Option<u8>, u8, Option<u8>)]) -> Vec<RootMoveScore> {
        let mut moves = self.ordered_moves(board, None, 0);
        // M3: promote preferred moves to front (stable sort by priority tier).
        if !preferred.is_empty() {
            let preferred_set: std::collections::HashSet<(Option<u8>, u8, Option<u8>)> =
                preferred.iter().cloned().collect();
            moves.sort_by_key(|mv| {
                if preferred_set.contains(&(mv.from, mv.to, mv.capture)) { 0u8 } else { 1u8 }
            });
        }
        let color = board.side_to_move;
        let in_placement = get_phase(board, color) == Phase::Place;
        let mut result = Vec::with_capacity(moves.len());
        for mv in moves.iter() {
            let nb = make_move(board, mv);
            let b64_penalty: i64 = if in_placement
                && mv.from.is_none()
                && !move_forms_mill(board, color, mv.from, mv.to)
            {
                let sq = mv.to as usize;
                let occupied_after = nb.white | nb.black;
                let free_after = (ADJACENCY[sq] & !occupied_after & FULL_MASK).count_ones();
                if free_after == 0 { 1500 } else if free_after == 1 { 400 } else { 0 }
            } else {
                0
            };
            let score = -self.negamax(&nb, depth - 1, -INF * 4, INF * 4, true, 1) - b64_penalty;
            if self.aborted {
                break;
            }
            result.push(RootMoveScore { mv: *mv, score });
        }
        result
    }
}


fn new_searcher(deadline: Instant) -> Searcher {
    Searcher {
        zobrist: Zobrist::new(),
        tt: Arc::new(TranspositionTable::new()),
        nodes: 0,
        deadline,
        aborted: false,
        killers: [[None; 2]; MAX_PLY],
        history: [[0i32; 24]; 25],
        opp_ext_set: HashSet::new(),
        fullgame_db: None,
        endgame_solved_db: None,
    }
}

/// Iterative deepening with a wall-clock time limit. Returns the best move found
/// at the deepest fully (or partially) completed iteration.
/// T-B2: aspiration windows seeded from the previous depth's best score.
pub fn iterative_deepening(board: &Board, max_depth: u8, time_limit_ms: u64) -> SearchResult {
    let deadline = Instant::now() + std::time::Duration::from_millis(time_limit_ms.max(1));
    let mut searcher = new_searcher(deadline);

    let mut best = SearchResult {
        best_move: None,
        score: 0,
        nodes: 0,
        depth_reached: 0,
    };

    let cap = max_depth.max(1);
    let mut last_score = 0i64;
    for d in 1..=cap {
        // T-B2: aspiration windows after depth 1.
        let (a_init, b_init) = if d > 1 {
            (last_score - ASP_MARGIN, last_score + ASP_MARGIN)
        } else {
            (-INF * 4, INF * 4)
        };
        let (mv, score) = searcher.root(board, d, a_init, b_init);
        if searcher.aborted {
            if best.best_move.is_none() {
                best.best_move = mv;
                best.score = score;
                best.depth_reached = d;
            }
            break;
        }
        // Re-search with full window on aspiration fail.
        let (mv, score) = if score <= a_init || score >= b_init {
            searcher.root(board, d, -INF * 4, INF * 4)
        } else {
            (mv, score)
        };
        if searcher.aborted {
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
        last_score = score;
        if score.abs() >= INF - 100 {
            break; // forced mate found
        }
    }
    best.nodes = searcher.nodes;
    best
}

/// Iterative deepening returning scores for all root moves. Each move is
/// evaluated with a full (-INF, INF) window so every score is exact.
/// `preferred` moves are promoted to the front of root ordering (M3 hint).
/// `tt` (T-C4, T-E1): shared Arc TT — persists across turns via RustTtHandle, thread-safe.
/// `opp_ext_set` (T-C1): high-frequency opponent moves that earn SE-11 extension.
/// `fullgame_db` (T-C2): mmap'd DB for in-search binary-search probe.
/// `endgame_solved_db` (T-C3): mmap'd .wdl tables for O(1) endgame WDL probe.
/// Sorted descending by score.
pub fn iterative_deepening_scored(
    board: &Board,
    max_depth: u8,
    time_limit_ms: u64,
    preferred: &[(Option<u8>, u8, Option<u8>)],
    tt: Arc<TranspositionTable>,
    opp_ext_set: HashSet<(Option<u8>, u8, Option<u8>)>,
    fullgame_db: Option<Arc<Mmap>>,
    endgame_solved_db: Option<Arc<HashMap<(u8, u8), Mmap>>>,
) -> SearchResultScored {
    let deadline = Instant::now() + std::time::Duration::from_millis(time_limit_ms.max(1));
    let mut searcher = Searcher {
        zobrist: Zobrist::new(),
        tt,
        nodes: 0,
        deadline,
        aborted: false,
        killers: [[None; 2]; MAX_PLY],
        history: [[0i32; 24]; 25],
        opp_ext_set,
        fullgame_db,
        endgame_solved_db,
    };

    let mut best = SearchResultScored {
        scored_moves: Vec::new(),
        nodes: 0,
        depth_reached: 0,
    };

    let cap = max_depth.max(1);
    for d in 1..=cap {
        let scored = searcher.root_scored(board, d, preferred);
        if searcher.aborted {
            if best.scored_moves.is_empty() {
                best.scored_moves = scored;
                best.depth_reached = d;
            }
            break;
        }
        best.scored_moves = scored;
        best.depth_reached = d;
        best.nodes = searcher.nodes;
        if best.scored_moves.iter().any(|rm| rm.score.abs() >= INF - 100) {
            break;
        }
    }
    best.nodes = searcher.nodes;
    best.scored_moves.sort_by(|a, b| b.score.cmp(&a.score));
    best
}

/// T-E2b: Lazy SMP — run `n_threads - 1` helper threads that all share the same
/// Arc<TT>. Each helper starts at a different depth spread across [1..max_depth]
/// so they reach higher depths before the main thread, pre-warming the TT there.
/// Thread i starts at depth (max_depth * i / n_threads).max(1), so helpers
/// cover deep nodes early and the main thread benefits from their TT entries.
/// Falls back to single-threaded when n_threads <= 1.
pub fn iterative_deepening_scored_smp(
    board: &Board,
    max_depth: u8,
    time_limit_ms: u64,
    preferred: &[(Option<u8>, u8, Option<u8>)],
    tt: Arc<TranspositionTable>,
    opp_ext_set: HashSet<(Option<u8>, u8, Option<u8>)>,
    fullgame_db: Option<Arc<Mmap>>,
    endgame_solved_db: Option<Arc<HashMap<(u8, u8), Mmap>>>,
    n_threads: usize,
) -> SearchResultScored {
    if n_threads <= 1 {
        return iterative_deepening_scored(board, max_depth, time_limit_ms, preferred, tt, opp_ext_set, fullgame_db, endgame_solved_db);
    }

    let board_copy = *board;
    let duration = std::time::Duration::from_millis(time_limit_ms.max(1));

    let helpers: Vec<_> = (1..n_threads)
        .map(|i| {
            let tt_c = Arc::clone(&tt);
            let db_c = fullgame_db.clone();
            let esdb_c = endgame_solved_db.clone();
            let opp_c = opp_ext_set.clone();
            // Spread helpers across [1..max_depth]: helper i starts at max_depth*i/n_threads.
            let start_depth = ((max_depth as usize * i) / n_threads).max(1) as u8;
            std::thread::spawn(move || {
                let deadline = Instant::now() + duration;
                let mut helper = Searcher {
                    zobrist: Zobrist::new(),
                    tt: tt_c,
                    nodes: 0,
                    deadline,
                    aborted: false,
                    killers: [[None; 2]; MAX_PLY],
                    history: [[0i32; 24]; 25],
                    opp_ext_set: opp_c,
                    fullgame_db: db_c,
                    endgame_solved_db: esdb_c,
                };
                for d in start_depth..=max_depth {
                    helper.root(&board_copy, d, -INF * 4, INF * 4);
                    if helper.aborted {
                        break;
                    }
                }
            })
        })
        .collect();

    let result = iterative_deepening_scored(board, max_depth, time_limit_ms, preferred, tt, opp_ext_set, fullgame_db, endgame_solved_db);

    drop(helpers);
    result
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

    #[test]
    fn root_scored_returns_all_placement_moves() {
        let board = Board {
            white: 0,
            black: 0,
            white_placed: 0,
            black_placed: 0,
            side_to_move: Color::White,
        };
        let r = iterative_deepening_scored(&board, 3, 5000, &[], Arc::new(TranspositionTable::new()), HashSet::new(), None);
        assert_eq!(r.scored_moves.len(), 24, "expected 24 moves on empty board");
        assert!(r.nodes > 0);
        for rm in &r.scored_moves {
            assert!(rm.score.abs() < INF * 2, "score {} out of range", rm.score);
        }
        for pair in r.scored_moves.windows(2) {
            assert!(pair[0].score >= pair[1].score, "not sorted descending");
        }
    }
}
