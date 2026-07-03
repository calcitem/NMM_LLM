//! PyO3 module entry for `nmm_core`. Exposes the Rust-accelerated NMM engine
//! primitives + coarse-grained search + DB/opening key generation.
//! See `docs/RUST_INTEGRATION_PLAN.md` §12 for the surface contract.

use std::collections::{HashSet, HashMap};
use std::sync::Arc;
use std::fs::File;
use memmap2::Mmap;
use pyo3::prelude::*;

mod board;
mod db_probe;
mod heuristics;
mod hash;
mod mills;
mod movegen;
mod opening_probe;
mod phases;
mod search;
mod symmetry;
mod tactics;
mod types;

use crate::hash::TranspositionTable;
use crate::types::{Board, Color};

/// T-C4 + T-E1: Persistent Rust transposition table held across `choose_move` calls.
/// Cleared only on new game via `reset_game_bans()`. Arc<TT> (atomic slots) is Sync,
/// so no Mutex needed — threads share it directly via Arc::clone.
#[pyclass]
struct RustTtHandle {
    inner: Arc<TranspositionTable>,
}

#[pymethods]
impl RustTtHandle {
    #[new]
    fn new() -> Self {
        RustTtHandle { inner: Arc::new(TranspositionTable::new()) }
    }

    fn clear(&self) {
        self.inner.clear();
    }
}

/// T-C2: Read-only mmap of a fullgame DB binary file for in-search probing.
#[pyclass]
struct FullgameDbHandle {
    mmap: Arc<Mmap>,
}

#[pymethods]
impl FullgameDbHandle {
    #[staticmethod]
    fn open(path: &str) -> PyResult<Self> {
        let file = File::open(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(FullgameDbHandle { mmap: Arc::new(mmap) })
    }
}

/// T-C3: Collection of mmap'd endgame solved (.wdl) files, one per (nW, nB) pair.
/// Keys are (white_piece_count, black_piece_count); values are 3..=7 on each axis.
pub type EndgameSolvedMap = HashMap<(u8, u8), Mmap>;

#[pyclass]
struct EndgameSolvedDbHandle {
    tables: Arc<EndgameSolvedMap>,
}

#[pymethods]
impl EndgameSolvedDbHandle {
    /// Open all endgame_W_B.wdl files found in `dir_path`.
    /// Missing files are silently skipped — probe returns None for those piece counts.
    #[staticmethod]
    fn open(dir_path: &str) -> PyResult<Self> {
        let dir = std::path::Path::new(dir_path);
        let mut tables = HashMap::new();
        for nw in 3u8..=7 {
            for nb in 3u8..=7 {
                let path = dir.join(format!("endgame_{}_{}.wdl", nw, nb));
                if !path.exists() { continue; }
                let file = File::open(&path)
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                let mmap = unsafe { Mmap::map(&file) }
                    .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
                tables.insert((nw, nb), mmap);
            }
        }
        Ok(EndgameSolvedDbHandle { tables: Arc::new(tables) })
    }

    fn table_count(&self) -> usize {
        self.tables.len()
    }
}

fn mk_board(white: u32, black: u32, wp: u8, bp: u8, stm: u8) -> Board {
    Board {
        white,
        black,
        white_placed: wp,
        black_placed: bp,
        side_to_move: Color::from_u8(stm),
    }
}

/// Apply D4 transform `idx` (0..8) to a 24-bit board mask.
#[pyfunction]
fn py_apply_transform(bits: u32, idx: usize) -> PyResult<u32> {
    if idx >= 8 {
        return Err(pyo3::exceptions::PyValueError::new_err("transform idx must be 0..8"));
    }
    Ok(symmetry::apply_transform(bits, idx))
}

/// Canonical (white, black) bitboard pair under D4 (lex-min). For search/TT.
#[pyfunction]
fn py_canonical_key(white: u32, black: u32) -> (u32, u32) {
    symmetry::canonical_key(white, black)
}

/// Canonical 24-char board string + sym_idx (matches `canonical_board_str`).
#[pyfunction]
fn py_canonical_board_str(board24: &str) -> (String, usize) {
    symmetry::canonical_board_str(board24)
}

/// Legal moves as a list of (from|None, to, capture|None) tuples.
#[pyfunction]
fn py_legal_moves(
    white: u32,
    black: u32,
    wp: u8,
    bp: u8,
    stm: u8,
) -> Vec<(Option<u8>, u8, Option<u8>)> {
    let board = mk_board(white, black, wp, bp, stm);
    movegen::legal_moves(&board)
        .into_iter()
        .map(|m| (m.from, m.to, m.capture))
        .collect()
}

/// True if a mill line through `square` is fully owned by `color` (0=W,1=B) in
/// the CURRENT bitboard. Static check over existing bits — `square` is NOT added
/// to `color`; callers wanting a hypothetical placement should set the bit first.
#[pyfunction]
fn py_forms_mill(white: u32, black: u32, square: u8, color: u8) -> bool {
    let board = mk_board(white, black, 0, 0, 0);
    mills::forms_mill(&board, square, Color::from_u8(color))
}

#[pyfunction]
fn py_count_mills(white: u32, black: u32, color: u8) -> u32 {
    let board = mk_board(white, black, 0, 0, 0);
    mills::count_mills(&board, Color::from_u8(color))
}

/// Phase for `color` (0=W,1=B): 0=place, 1=move, 2=fly.
#[pyfunction]
fn py_detect_phase(wp: u8, bp: u8, won: u32, bon: u32, color: u8) -> u8 {
    // Build a board with the given on-board counts on arbitrary squares.
    let white = if won >= 32 { u32::MAX } else { (1u32 << won) - 1 };
    let black_full = if bon >= 32 { 0 } else { ((1u32 << bon) - 1) << 12 };
    let board = mk_board(white, black_full, wp, bp, 0);
    phases::detect_phase_u8(&board, Color::from_u8(color))
}

/// Integer base evaluation from `stm`'s perspective.
#[pyfunction]
fn py_evaluate(white: u32, black: u32, wp: u8, bp: u8, stm: u8) -> i64 {
    let board = mk_board(white, black, wp, bp, stm);
    heuristics::evaluate_base(&board, Color::from_u8(stm))
}

/// Immediate mill-threat closing squares (as a 24-bit mask) the opponent of stm
/// can close next move.
#[pyfunction]
fn py_immediate_threats(white: u32, black: u32, wp: u8, bp: u8, stm: u8) -> u32 {
    let board = mk_board(white, black, wp, bp, stm);
    tactics::immediate_mill_threats(&board)
}

/// Coarse-grained best move: (from|None, to, capture|None).
#[pyfunction]
#[pyo3(signature = (white, black, white_placed, black_placed, side_to_move, max_depth=6, time_limit_ms=5000))]
fn py_get_best_move(
    white: u32,
    black: u32,
    white_placed: u8,
    black_placed: u8,
    side_to_move: u8,
    max_depth: u8,
    time_limit_ms: u64,
) -> (Option<u8>, Option<u8>, Option<u8>) {
    let r = search::get_best_move(
        white,
        black,
        white_placed,
        black_placed,
        Color::from_u8(side_to_move),
        max_depth,
        time_limit_ms,
    );
    match r.best_move {
        Some(m) => (m.from, Some(m.to), m.capture),
        None => (None, None, None),
    }
}

/// Like `py_get_best_move` but also returns search stats for benchmarking:
/// (from|None, to|None, capture|None, nodes, depth_reached).
#[pyfunction]
#[pyo3(signature = (white, black, white_placed, black_placed, side_to_move, max_depth=6, time_limit_ms=5000))]
fn py_search_stats(
    white: u32,
    black: u32,
    white_placed: u8,
    black_placed: u8,
    side_to_move: u8,
    max_depth: u8,
    time_limit_ms: u64,
) -> (Option<u8>, Option<u8>, Option<u8>, u64, u8) {
    let r = search::get_best_move(
        white,
        black,
        white_placed,
        black_placed,
        Color::from_u8(side_to_move),
        max_depth,
        time_limit_ms,
    );
    match r.best_move {
        Some(m) => (m.from, Some(m.to), m.capture, r.nodes, r.depth_reached),
        None => (None, None, None, r.nodes, r.depth_reached),
    }
}

/// Per-move scored root search. Returns `(nodes, depth_reached, moves)` where
/// each move is `(from|None, to, capture|None, score)`, sorted best-first.
/// Every score is exact (full-window search — no root alpha-beta pruning).
/// `preferred_root`: optional list of (from|None, to, cap|None) triples promoted
/// to the front of move ordering for better alpha-beta pruning (M3 trajectory hint).
/// `tt_handle` (T-C4/T-E1): Arc TT persisted across turns; shared across SMP threads.
/// `db_handle` (T-C2): mmap'd fullgame DB for in-search binary-search probe.
/// `endgame_db_handle` (T-C3): mmap'd endgame solved (.wdl) files for O(1) WDL probe.
/// `opp_ext_moves` (T-C1): high-freq opponent moves that earn SE-11 depth extension.
/// `threads` (T-E3): Lazy-SMP thread count; default = 1 (single-threaded).
#[pyfunction]
#[pyo3(signature = (white, black, white_placed, black_placed, side_to_move, max_depth=6, time_limit_ms=5000, preferred_root=None, tt_handle=None, db_handle=None, endgame_db_handle=None, opp_ext_moves=None, threads=None))]
fn py_search_root_scored(
    py: Python<'_>,
    white: u32,
    black: u32,
    white_placed: u8,
    black_placed: u8,
    side_to_move: u8,
    max_depth: u8,
    time_limit_ms: u64,
    preferred_root: Option<Vec<(Option<u8>, u8, Option<u8>)>>,
    tt_handle: Option<Py<RustTtHandle>>,
    db_handle: Option<Py<FullgameDbHandle>>,
    endgame_db_handle: Option<Py<EndgameSolvedDbHandle>>,
    opp_ext_moves: Option<Vec<(Option<u8>, u8, Option<u8>)>>,
    threads: Option<usize>,
) -> (u64, u8, Vec<(Option<u8>, u8, Option<u8>, i64)>) {
    let board = Board {
        white,
        black,
        white_placed,
        black_placed,
        side_to_move: Color::from_u8(side_to_move),
    };
    let preferred = preferred_root.unwrap_or_default();

    // T-C1: build HashSet of high-frequency opponent moves for SE-11 extension.
    let opp_ext_set: HashSet<(Option<u8>, u8, Option<u8>)> =
        opp_ext_moves.unwrap_or_default().into_iter().collect();

    // T-C2: borrow mmap Arc from the DB handle (cheap clone of Arc pointer).
    let fullgame_db: Option<Arc<Mmap>> = db_handle.as_ref().map(|h| {
        h.bind(py).borrow().mmap.clone()
    });

    // T-C3: borrow Arc<EndgameSolvedMap> from the handle.
    let endgame_solved_db: Option<Arc<EndgameSolvedMap>> = endgame_db_handle.as_ref().map(|h| {
        h.bind(py).borrow().tables.clone()
    });

    // T-C4/T-E1: clone Arc<TT> from handle (cheap pointer clone; TT persists for GameAI lifetime).
    let tt: Arc<TranspositionTable> = tt_handle
        .as_ref()
        .map(|h| h.bind(py).borrow().inner.clone())
        .unwrap_or_else(|| Arc::new(TranspositionTable::new()));

    // T-E3: resolve thread count. Default = 1 (single-threaded); pass threads>1 to enable Lazy SMP.
    let n_threads = threads.unwrap_or(1);

    let r = search::iterative_deepening_scored_smp(
        &board, max_depth, time_limit_ms, &preferred,
        tt, opp_ext_set, fullgame_db, endgame_solved_db, n_threads,
    );

    let moves = r
        .scored_moves
        .into_iter()
        .map(|rm| (rm.mv.from, rm.mv.to, rm.mv.capture, rm.score))
        .collect();
    (r.nodes, r.depth_reached, moves)
}

/// FullGame DB 9-byte key (byte-identical to Python `_encode_canonical`).
#[pyfunction]
fn py_db_key(white: u32, black: u32, turn: u8, placed_w: u8, placed_b: u8) -> Vec<u8> {
    db_probe::fullgame_key(white, black, turn, placed_w, placed_b)
}

/// Endgame DB string key "<canonical board24>|<turn>".
#[pyfunction]
fn py_endgame_key(white: u32, black: u32, turn: u8) -> String {
    db_probe::endgame_key(white, black, turn)
}

/// Opening/trajectory key: (pipe-joined canonical sequence, sym_idx).
#[pyfunction]
#[pyo3(signature = (notations, depth=None))]
fn py_opening_key(notations: Vec<String>, depth: Option<usize>) -> (String, usize) {
    let d = depth.unwrap_or(notations.len());
    opening_probe::opening_key(&notations, d)
}

/// Transform a move notation by D4 sym_idx; None if unmapped.
#[pyfunction]
fn py_transform_notation(notation: &str, sym_idx: usize) -> Option<String> {
    if sym_idx >= 8 {
        return None;
    }
    opening_probe::transform_notation(notation, sym_idx)
}

#[pymodule]
fn nmm_core(_py: Python, m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RustTtHandle>()?;
    m.add_class::<FullgameDbHandle>()?;
    m.add_class::<EndgameSolvedDbHandle>()?;
    m.add_function(wrap_pyfunction!(py_apply_transform, m)?)?;
    m.add_function(wrap_pyfunction!(py_canonical_key, m)?)?;
    m.add_function(wrap_pyfunction!(py_canonical_board_str, m)?)?;
    m.add_function(wrap_pyfunction!(py_legal_moves, m)?)?;
    m.add_function(wrap_pyfunction!(py_forms_mill, m)?)?;
    m.add_function(wrap_pyfunction!(py_count_mills, m)?)?;
    m.add_function(wrap_pyfunction!(py_detect_phase, m)?)?;
    m.add_function(wrap_pyfunction!(py_evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(py_immediate_threats, m)?)?;
    m.add_function(wrap_pyfunction!(py_get_best_move, m)?)?;
    m.add_function(wrap_pyfunction!(py_search_stats, m)?)?;
    m.add_function(wrap_pyfunction!(py_search_root_scored, m)?)?;
    m.add_function(wrap_pyfunction!(py_db_key, m)?)?;
    m.add_function(wrap_pyfunction!(py_endgame_key, m)?)?;
    m.add_function(wrap_pyfunction!(py_opening_key, m)?)?;
    m.add_function(wrap_pyfunction!(py_transform_notation, m)?)?;
    Ok(())
}
