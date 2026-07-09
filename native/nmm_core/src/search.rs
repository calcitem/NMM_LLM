//! Self-contained negamax + alpha-beta + iterative deepening search.
//!
//! Coarse-grained: Python calls `iterative_deepening` once via the PyO3 binding;
//! the whole tree is searched in Rust with its own Zobrist + TT. This engine is
//! NOT required to return the same move as the Python AI — only legal, sane
//! play. See `docs/RUST_INTEGRATION_PLAN.md` §10.

use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use memmap2::Mmap;

/// Global counter: incremented each time Phase-2 forcing extension fires in qsearch.
/// Exposed via py_get_forcing_ext_count() / py_reset_forcing_ext_count() FFI.
pub static FORCING_EXT_COUNT: AtomicU64 = AtomicU64::new(0);

use crate::board::{make_move, terminal_winner, ADJACENCY, get_phase};
use crate::db_probe;
use crate::heuristics::{evaluate_v2, EvalScale, INF};
use crate::hash::{TranspositionTable, TtEntry, Zobrist, EXACT, LOWER_BOUND, UPPER_BOUND};
use crate::movegen::legal_moves;
use crate::tactics::{immediate_mill_threats, move_forms_mill};
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
    // Countermove table: given the previous move (from, to), store the refutation
    // that caused a beta cutoff.  Indexed [from_or_24][to] (0..25 × 0..24).
    countermoves: Box<[[Option<Move>; 24]; 25]>,
    // Per-personality eval scale factors (mill / mobility / blocked_opp), default 100%.
    eval_scale: EvalScale,
    // T-C1: high-frequency opponent moves that earn a SE-11 depth extension.
    opp_ext_set: HashSet<(Option<u8>, u8, Option<u8>)>,
    // T-C2: mmap'd fullgame DB for in-search probe (probe between terminal check and TT).
    fullgame_db: Option<Arc<Mmap>>,
    // T-C3: mmap'd endgame solved tables, keyed by (nW, nB). O(1) WDL probe.
    endgame_solved_db: Option<Arc<HashMap<(u8, u8), Mmap>>>,
    // FGOP: AI's root color — used to detect opponent-to-move nodes inside negamax.
    ai_color: Color,
}

const ABORT_SCORE: i64 = i64::MIN + 1;
const MAX_PLY: usize = 64;
// LMR fires for move index >= this (same threshold as the lmr condition below).
const LMR_LATE_IDX: usize = 3;
// FGOP: frequency-gated opponent pruning constants.
const FGOP_DEPTH: u8  = 5;   // only prune when remaining depth ≤ this
const FGOP_MARGIN: i64 = 150; // eval margin below best opp move to trigger gate 1
// Phase 2: max extra plies added by forcing qsearch extension (two-config / forced-block).
// Tactical moves (captures, mill closures) are always searched and don't count against this.
const QS_FORCING_CAP: u8 = 6;
// Mate-score boundary: any |score| above this is a terminal win/loss, not an eval.
const MATE_THRESHOLD: i64 = INF / 2;
// V4-C: star squares — the 12 nodes at odd indices (2 mill memberships each).
const STAR_SQUARES: u32 =
    (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7)
    | (1 << 9) | (1 << 11) | (1 << 13) | (1 << 15)
    | (1 << 17) | (1 << 19) | (1 << 21) | (1 << 23);

/// Convert a ply-relative mate score to an absolute mate-in-N for TT storage.
/// Non-mate scores pass through unchanged.
/// Wins (score > MATE_THRESHOLD): stored as score + ply = INF - (ply_terminal - ply)
///   which equals INF - mate_in_N (position-independent).
/// Losses: symmetric.
#[inline]
fn score_to_tt(score: i64, ply: u8) -> i64 {
    if score > MATE_THRESHOLD {
        score + ply as i64
    } else if score < -MATE_THRESHOLD {
        score - ply as i64
    } else {
        score
    }
}

/// Recover ply-relative mate score from the absolute mate-in-N stored in TT.
#[inline]
fn score_from_tt(score: i64, ply: u8) -> i64 {
    if score > MATE_THRESHOLD {
        score - ply as i64
    } else if score < -MATE_THRESHOLD {
        score + ply as i64
    } else {
        score
    }
}

/// FGOP Gate 2b: true if `mv` is structurally rare for `color` to play.
///
/// Checks: the piece is being moved FROM a square that is part of an own
/// two-config (2-of-3 in a mill), AND the destination does not complete
/// any mill for `color`. Such moves voluntarily destroy own setup without
/// gaining a mill — humans almost never make them.
///
/// Uses SQUARE_MILLS per square (≤3 mills each) → O(6) total.
fn is_structurally_rare(mv: &Move, board: &Board, color: Color) -> bool {
    use crate::mills::{MILL_MASKS, SQUARE_MILLS};
    let Some(from_sq) = mv.from else { return false; };
    let to_sq = mv.to as usize;
    let from_bit = 1u32 << from_sq;
    let to_bit   = 1u32 << to_sq;
    let own   = board.bits(color);
    let empty = board.empty();

    // Gate A: from_sq is in an own two-config (will be broken by this move).
    let from_in_two_cfg = SQUARE_MILLS[from_sq as usize]
        .iter()
        .any(|&mi| {
            let mm = MILL_MASKS[mi as usize];
            (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1
        });
    if !from_in_two_cfg { return false; }

    // Gate B: to_sq does NOT complete any mill (move isn't compensating).
    let own_after = (own | to_bit) & !from_bit;
    !SQUARE_MILLS[to_sq]
        .iter()
        .any(|&mi| (own_after & MILL_MASKS[mi as usize]) == MILL_MASKS[mi as usize])
}

/// Phase 2: true if `mv` for `color` creates a reachable two-config after it lands.
///
/// A two-config is 2 own pieces + 1 empty closing square in a mill line.
/// "Reachable" means the closing square can be occupied by own next turn:
///   - Placement phase (still pieces to place after this move): any empty closing sq.
///   - Fly phase (own_after ≤ 3 pieces): any empty closing sq (can jump anywhere).
///   - Move phase: an own piece not already in the mill line is adjacent to the closing sq.
///
/// Only checks SQUARE_MILLS[mv.to] — lines affected by the landing square — which
/// is exactly where new two-configs can be created by this move. O(2×3) = O(6).
fn creates_reachable_two_config(board: &Board, color: Color, mv: &Move) -> bool {
    use crate::mills::{MILL_MASKS, SQUARE_MILLS};
    let to_sq = mv.to as usize;
    let from_bit = mv.from.map_or(0u32, |f| 1u32 << f);
    let to_bit   = 1u32 << to_sq;
    let own      = board.bits(color);
    let own_after = (own | to_bit) & !from_bit;

    // Opponent bits after any capture — needed for computing occupied squares.
    let cap_bit  = mv.capture.map_or(0u32, |c| 1u32 << c);
    let opp_after = board.bits(color.opponent()) & !cap_bit;
    let occupied_after = own_after | opp_after;

    // Will we still be in placement phase AFTER this move completes?
    let placed_after = match color {
        Color::White => board.white_placed + if mv.from.is_none() { 1 } else { 0 },
        Color::Black => board.black_placed + if mv.from.is_none() { 1 } else { 0 },
    };
    let still_placing = placed_after < 9;
    // Fly phase: own has ≤ 3 pieces (can jump to any empty square).
    let fly = own_after.count_ones() <= 3;

    for &mi in &SQUARE_MILLS[to_sq] {
        let mm = MILL_MASKS[mi as usize];
        if (own_after & mm).count_ones() == 2 {
            let closing = mm & !occupied_after;
            if closing.count_ones() != 1 { continue; }
            let closing_sq = closing.trailing_zeros() as usize;
            // Check reachability for the next own move.
            let reachable = still_placing
                || fly
                || (ADJACENCY[closing_sq] & own_after & !mm) != 0;
            if reachable { return true; }
        }
    }
    false
}

impl Searcher {
    fn store_killer(&mut self, ply: usize, mv: Move) {
        if ply >= MAX_PLY || mv.capture.is_some() { return; }
        if self.killers[ply][0] != Some(mv) {
            self.killers[ply][1] = self.killers[ply][0];
            self.killers[ply][0] = Some(mv);
        }
    }

    fn ordered_moves(&self, board: &Board, tt_best: Option<u16>, ply: usize, prev_move: Option<Move>) -> Vec<Move> {
        let mut moves = legal_moves(board);
        let color = board.side_to_move;
        let k = if ply < MAX_PLY { self.killers[ply] } else { [None; 2] };
        // Look up countermove for the previous ply's move.
        let cm: Option<Move> = prev_move.and_then(|pm| {
            let pf = pm.from.map_or(24usize, |f| f as usize);
            let pt = pm.to as usize;
            if pt < 24 { self.countermoves[pf][pt] } else { None }
        });
        moves.sort_by_key(|mv| {
            let mut s = 0i64;
            if mv.capture.is_some() { s -= 2000; }
            if move_forms_mill(board, color, mv.from, mv.to) { s -= 1000; }
            // V4-C: star square placement bonus (non-mill placements only).
            if mv.from.is_none()
                && !move_forms_mill(board, color, mv.from, mv.to)
                && (STAR_SQUARES & (1u32 << mv.to)) != 0
            { s -= 300; }
            // T-B3: killer moves after captures/mills.
            if k[0] == Some(*mv) { s -= 600; }
            else if k[1] == Some(*mv) { s -= 500; }
            // Countermove: refutation of the previous move, between killers and history.
            if cm == Some(*mv) { s -= 450; }
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

    // T-B4 / Phase 2: Quiescence search extended with forcing moves.
    //
    // Always extends: captures, mill-closing moves (original qsearch).
    // Phase 2 extension (up to QS_FORCING_CAP extra plies):
    //   - Reachable two-config creators: mv lands so own now has 2 pieces in a mill
    //     line with a reachable closing square → threatens a mill next move.
    //   - Forced blocks: mv.to is a square where the opponent can close a mill
    //     (pre-computed as opp_threats via immediate_mill_threats). Playing there
    //     blocks the threat; not playing there would let opp form a mill next move.
    //
    // Tactical moves (capture/mill) never count against QS_FORCING_CAP.
    // Forcing-only moves increment qs_ply → cap fires after 6 forcing extensions.
    fn qsearch(&mut self, board: &Board, mut alpha: i64, beta: i64, qs_ply: u8) -> i64 {
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
        let stand_pat = evaluate_v2(board, color, self.eval_scale);
        if stand_pat >= beta { return beta; }
        if stand_pat > alpha { alpha = stand_pat; }

        // Pre-compute opponent mill threats once — used for forced-block filter.
        // Only pay this cost when the forcing cap has not been reached.
        let can_force = qs_ply < QS_FORCING_CAP;
        let opp_threats: u32 = if can_force { immediate_mill_threats(board) } else { 0 };

        for mv in legal_moves(board).iter() {
            let is_tactical = mv.capture.is_some()
                || move_forms_mill(board, color, mv.from, mv.to);
            // Phase 2: also extend reachable two-config creators and forced blocks.
            let is_forcing = !is_tactical && can_force && (
                (opp_threats & (1u32 << mv.to)) != 0
                || creates_reachable_two_config(board, color, mv)
            );
            if !is_tactical && !is_forcing {
                continue;
            }
            let nb = make_move(board, mv);
            // Tactical moves don't consume the forcing budget; forcing-only moves do.
            let next_qs = if is_tactical { qs_ply } else {
                FORCING_EXT_COUNT.fetch_add(1, Ordering::Relaxed);
                qs_ply + 1
            };
            let score = -self.qsearch(&nb, -beta, -alpha, next_qs);
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
    // T-B3: killers + history + countermoves updated on beta cutoff.
    // T-B4: qsearch at depth==0.
    // T-B5: null-move pruning at depth>=3 outside fly phase.
    // `prev_move`: the move played to reach this node (None at root / after null move).
    fn negamax(&mut self, board: &Board, depth: u8, mut alpha: i64, beta: i64, first_opp_ply: bool, ply: u8, prev_move: Option<Move>) -> i64 {
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
            // Prefer shorter wins (INF - ply decreases as ply increases) and
            // delay losses (-(INF - ply) increases as ply increases → less bad).
            return if winner == color {
                INF - ply as i64
            } else {
                -(INF - ply as i64)
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
                let s = score_from_tt(e.score as i64, ply);
                match e.flag {
                    EXACT => return s,
                    LOWER_BOUND if s >= beta => return s,
                    UPPER_BOUND if s <= alpha => return s,
                    _ => {}
                }
            }
        }

        // T-B4: quiescence search at horizon (qs_ply=0 → fresh forcing budget).
        if depth == 0 {
            return self.qsearch(board, alpha, beta, 0);
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
            let null_score = -self.negamax(&null_board, depth - 3, -beta, -beta + 1, false, ply + 1, None);
            if !self.aborted && null_score >= beta {
                return beta;
            }
        }

        let alpha_orig = alpha;
        let mut moves = self.ordered_moves(board, tt_best, ply as usize, prev_move);
        if moves.is_empty() {
            return -(INF - depth as i64);
        }

        let mut best_score = -INF * 4;
        let mut best_idx: u16 = u16::MAX;
        // FGOP: track best static eval from opponent's POV to compute eval gate.
        let is_opp_ply = color != self.ai_color;
        let fgop_active = is_opp_ply && depth <= FGOP_DEPTH && ply > 0;
        let mut best_opp_static: i64 = -INF * 4;
        for (i, mv) in moves.iter().enumerate() {
            let nb = make_move(board, mv);
            let is_tactical = mv.capture.is_some() || move_forms_mill(board, color, mv.from, mv.to);
            // FGOP dual-gate: skip clearly-bad opponent moves that are also structurally rare.
            // Gate 1: static eval from opponent's POV is far below the best seen so far.
            // Gate 2: this move voluntarily breaks own two-config without completing a mill.
            // Never prune the first move (i==0) or tactical moves (captures/mills).
            if fgop_active && i > 0 && !is_tactical {
                let opp_static = evaluate_v2(&nb, color, self.eval_scale);
                if opp_static < best_opp_static - FGOP_MARGIN
                    && is_structurally_rare(mv, board, color)
                {
                    continue;
                }
                best_opp_static = best_opp_static.max(opp_static);
            }
            // SE-11: extend by 1 at first opponent ply for tactical moves (original) or
            // high-frequency trajectory moves (T-C1: opp_ext_set from trajectory_db).
            let in_opp_ext = first_opp_ply
                && !self.opp_ext_set.is_empty()
                && self.opp_ext_set.contains(&(mv.from, mv.to, mv.capture));
            let se11_ext: u8 = if first_opp_ply && (is_tactical || in_opp_ext) { 1 } else { 0 };

            // T-B1 + T-B2: PVS with LMR for moves after the first.
            let score = if i == 0 {
                -self.negamax(&nb, depth - 1 + se11_ext, -beta, -alpha, false, ply + 1, Some(*mv))
            } else {
                // T-B2: reduce late non-tactical moves at depth >= 3.
                let lmr: u8 = if depth >= 3 && i >= LMR_LATE_IDX && !is_tactical && se11_ext == 0 { 1 } else { 0 };
                // T-B1: null window at (possibly reduced) depth.
                let mut s = -self.negamax(
                    &nb, (depth - 1 + se11_ext).saturating_sub(lmr),
                    -alpha - 1, -alpha, false, ply + 1, Some(*mv),
                );
                // Re-search at full depth+window if null window or LMR was wrong.
                if !self.aborted && s > alpha {
                    s = -self.negamax(&nb, depth - 1 + se11_ext, -beta, -alpha, false, ply + 1, Some(*mv));
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
                        // Countermove: record this refutation against the previous move.
                        if let Some(pm) = prev_move {
                            let pf = pm.from.map_or(24usize, |f| f as usize);
                            let pt = pm.to as usize;
                            if pt < 24 {
                                self.countermoves[pf][pt] = Some(*mv);
                            }
                        }
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
            score: score_to_tt(best_score, ply) as i32,
            flag,
            best_idx,
        });
        best_score
    }

    fn root(&mut self, board: &Board, depth: u8, alpha_init: i64, beta_init: i64) -> (Option<Move>, i64) {
        let moves = self.ordered_moves(board, None, 0, None);
        if moves.is_empty() {
            return (None, -INF);
        }
        let color = board.side_to_move;
        let in_placement = get_phase(board, color) == Phase::Place;
        let mut alpha = alpha_init;
        let beta = beta_init;
        let mut best_move = Some(moves[0]);
        let mut best_score = -INF * 4;
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
            let score = -self.negamax(&nb, depth - 1, -beta, -alpha, true, 1, Some(*mv)) - b64_penalty;
            if self.aborted {
                break;
            }
            if score > best_score {
                best_score = score;
                best_move = Some(*mv);
            }
            if score > alpha {
                alpha = score;
            }
            if alpha >= beta {
                break;
            }
        }
        (best_move, best_score)
    }

    // V4-B: MTD(f) — Memory-enhanced Test Driver with zero-window passes.
    // Requires root() to be fail-soft (returns best_score, not clamped to beta).
    fn mtdf(&mut self, board: &Board, depth: u8, f: i64) -> (Option<Move>, i64) {
        let mut lower = -INF * 4;
        let mut upper =  INF * 4;
        let mut best_mv: Option<Move> = None;
        let mut score = f;
        while lower < upper {
            let beta = if score == lower { score + 1 } else { score };
            let (mv, s) = self.root(board, depth, beta - 1, beta);
            if self.aborted {
                return (mv, s);
            }
            best_mv = mv;
            score = s;
            if score < beta { upper = score; } else { lower = score; }
        }
        (best_mv, score)
    }

    /// Full-window root search: every move gets an independent (-INF, INF) window
    /// so all returned scores are exact. Mirrors `root()` B-64 penalty exactly.
    /// `preferred` is an optional list of (from, to, capture) triples promoted to
    /// the front of the move list for better alpha-beta pruning (M3).
    fn root_scored(&mut self, board: &Board, depth: u8, preferred: &[(Option<u8>, u8, Option<u8>)]) -> Vec<RootMoveScore> {
        let mut moves = self.ordered_moves(board, None, 0, None);
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
            let score = -self.negamax(&nb, depth - 1, -INF * 4, INF * 4, true, 1, Some(*mv)) - b64_penalty;
            if self.aborted {
                break;
            }
            result.push(RootMoveScore { mv: *mv, score });
        }
        result
    }
}


fn new_searcher(deadline: Instant, ai_color: Color) -> Searcher {
    Searcher {
        zobrist: Zobrist::new(),
        tt: Arc::new(TranspositionTable::new()),
        nodes: 0,
        deadline,
        aborted: false,
        killers: [[None; 2]; MAX_PLY],
        history: [[0i32; 24]; 25],
        countermoves: Box::new([[None; 24]; 25]),
        eval_scale: EvalScale::default(),
        ai_color,
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
    let mut searcher = new_searcher(deadline, board.side_to_move);

    let mut best = SearchResult {
        best_move: None,
        score: 0,
        nodes: 0,
        depth_reached: 0,
    };

    let cap = max_depth.max(1);
    let mut last_score = 0i64;
    for d in 1..=cap {
        // V4-B: MTD(f) after depth 1; seed from previous iteration's score.
        let (mv, score) = if d > 1 {
            searcher.mtdf(board, d, last_score)
        } else {
            searcher.root(board, d, -INF * 4, INF * 4)
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
    eval_scale: EvalScale,
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
        countermoves: Box::new([[None; 24]; 25]),
        eval_scale,
        ai_color: board.side_to_move,
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
    eval_scale: EvalScale,
) -> SearchResultScored {
    if n_threads <= 1 {
        return iterative_deepening_scored(board, max_depth, time_limit_ms, preferred, tt, opp_ext_set, fullgame_db, endgame_solved_db, eval_scale);
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
                    countermoves: Box::new([[None; 24]; 25]),
                    eval_scale,
                    ai_color: board_copy.side_to_move,
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

    let result = iterative_deepening_scored(board, max_depth, time_limit_ms, preferred, tt, opp_ext_set, fullgame_db, endgame_solved_db, eval_scale);

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
        let r = iterative_deepening_scored(&board, 3, 5000, &[], Arc::new(TranspositionTable::new()), HashSet::new(), None, None, EvalScale::default());
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
