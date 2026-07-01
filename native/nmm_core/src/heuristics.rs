//! Integer feature helpers + base evaluate. Ports the integer terms of
//! `ai/heuristics.py::evaluate`. See `docs/RUST_INTEGRATION_PLAN.md` §7.
//!
//! The full Python heuristic adds many float-scaled phase-conditional terms;
//! this Rust evaluate implements the integer BASE formula used by the
//! self-contained Rust search. Python remains the default evaluator.

use crate::board::{get_phase, terminal_winner, ADJACENCY};
use crate::mills::{MILL_MASKS, SQUARE_MILLS};
use crate::types::{Board, Color, Phase};

pub const INF: i64 = 10_000_000;
const FLY_MOBILITY_CAP: i64 = 5;

// Cardinal nodes (4-conn): b4,d2,d6,f4 = idx 15,13,9,11
const CARDINAL_MASK: u32 = (1 << 9) | (1 << 11) | (1 << 13) | (1 << 15);
// Cross nodes (3-conn): d7,g4,d1,a4,d5,e4,d3,c4 = 1,3,5,7,17,19,21,23
const CROSS3_MASK: u32 =
    (1 << 1) | (1 << 3) | (1 << 5) | (1 << 7) | (1 << 17) | (1 << 19) | (1 << 21) | (1 << 23);

// Phase weight tuples (mill, block, piece_diff, two_cfg, dbl_mill, win_cfg).
fn weights(phase: Phase) -> [i64; 6] {
    match phase {
        Phase::Place => [30, 12, 12, 5, 0, 0],
        Phase::Move => [30, 48, 12, 5, 50, 0],
        Phase::Fly => [32, 350, 2, 0, 90, 1190],
    }
}

fn mob_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 3,
        Phase::Move => 8,
        Phase::Fly => 20,
    }
}
fn threat_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 15,
        Phase::Move => 18,
        Phase::Fly => 80,
    }
}
fn cycle_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 8,
        Phase::Move => 22,
        Phase::Fly => 80,
    }
}
fn fork_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 6,
        Phase::Move => 14,
        Phase::Fly => 55,
    }
}
fn herd_weight(p: Phase) -> i64 {
    match p {
        Phase::Place => 6,
        Phase::Move => 18,
        Phase::Fly => 0,
    }
}
fn near_blocked_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 30,
        _ => 0,
    }
}
fn wrap_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 40,
        Phase::Fly => 60,
        _ => 0,
    }
}
fn fly_asym_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 80,
        _ => 0,
    }
}
fn domination_weight(p: Phase) -> i64 {
    match p {
        Phase::Move => 150,
        Phase::Fly => 80,
        _ => 0,
    }
}

pub fn closed_mills(board: &Board, color: Color) -> i64 {
    let bits = board.bits(color);
    let mut n = 0;
    for &mm in &MILL_MASKS {
        if (bits & mm) == mm {
            n += 1;
        }
    }
    n
}

pub fn blocked_count(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    let mut bits = own;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if (ADJACENCY[sq] & empty) == 0 {
            count += 1;
        }
    }
    count
}

pub fn two_configs(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for &mm in &MILL_MASKS {
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            count += 1;
        }
    }
    count
}

pub fn double_mills(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let mut count = 0;
    let mut bits = own;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let n = SQUARE_MILLS[sq as usize]
            .iter()
            .filter(|&&mi| (own & MILL_MASKS[mi as usize]) == MILL_MASKS[mi as usize])
            .count();
        if n >= 2 {
            count += 1;
        }
    }
    count
}

pub fn win_config(board: &Board, opp: Color) -> i64 {
    if board.placed(opp) == 9 && board.count(opp) <= 3 {
        1
    } else {
        0
    }
}

pub fn mobility(board: &Board, color: Color) -> i64 {
    let phase = get_phase(board, color);
    if phase == Phase::Fly {
        let empty = board.empty().count_ones() as i64;
        return FLY_MOBILITY_CAP.min(empty);
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    let mut bits = own;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        count += (ADJACENCY[sq] & empty).count_ones() as i64;
    }
    count
}

pub fn mill_threats(board: &Board, color: Color) -> i64 {
    let phase = get_phase(board, color);
    let can_place = board.placed(color) < 9;
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for &mm in &MILL_MASKS {
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            let empty_sq = (empty & mm).trailing_zeros() as usize;
            let reachable = match phase {
                Phase::Place => can_place,
                Phase::Fly => true,
                Phase::Move => {
                    let adj_own = ADJACENCY[empty_sq] & own & !mm;
                    adj_own != 0
                }
            };
            if reachable {
                count += 1;
            }
        }
    }
    count
}

pub fn position_value(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let mut total = 0;
    total += (own & CARDINAL_MASK).count_ones() as i64 * 5;
    total += (own & CROSS3_MASK).count_ones() as i64 * 3;
    let corners = own & !CARDINAL_MASK & !CROSS3_MASK;
    total += corners.count_ones() as i64 * 2;
    total
}

pub fn mill_cycle_ready(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    for &mm in &MILL_MASKS {
        if (own & mm) != mm {
            continue;
        }
        // Any piece in the mill with a free adjacent square — iterate set bits of mm.
        let mut mill_bits = mm;
        let mut ready = false;
        while mill_bits != 0 {
            let sq = mill_bits.trailing_zeros() as usize;
            mill_bits &= mill_bits - 1;
            if (ADJACENCY[sq] & empty) != 0 {
                ready = true;
                break;
            }
        }
        if ready {
            count += 1;
        }
    }
    count
}

pub fn fork_threats(board: &Board, color: Color) -> i64 {
    let own = board.bits(color);
    let empty = board.empty();
    // Collect bitmask of open-mill mask indices as a u16 bitset (16 mills fit in u16).
    let mut open_mask_bits: u16 = 0;
    for (i, &mm) in MILL_MASKS.iter().enumerate() {
        if (own & mm).count_ones() == 2 && (empty & mm).count_ones() == 1 {
            open_mask_bits |= 1 << i;
        }
    }
    if open_mask_bits.count_ones() < 2 {
        return 0;
    }
    // For each own piece, count how many open mills contain it via SQUARE_MILLS.
    let mut count = 0;
    let mut bits = own;
    while bits != 0 {
        let sq = bits.trailing_zeros() as u8;
        bits &= bits - 1;
        let n = SQUARE_MILLS[sq as usize]
            .iter()
            .filter(|&&mi| open_mask_bits & (1 << mi) != 0)
            .count();
        if n >= 2 {
            count += 1;
        }
    }
    count
}

pub fn encirclement(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let opp = board.bits(color.opponent());
    let own = board.bits(color);
    let mut count = 0;
    let mut bits = opp;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        count += (ADJACENCY[sq] & own).count_ones() as i64;
    }
    count
}

pub fn squeeze_count(board: &Board, color: Color) -> i64 {
    if get_phase(board, color) == Phase::Fly {
        return 0;
    }
    let own = board.bits(color);
    let empty = board.empty();
    let mut count = 0;
    let mut bits = own;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        if (ADJACENCY[sq] & empty).count_ones() == 1 {
            count += 1;
        }
    }
    count
}

pub fn mill_wrapping_pressure(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    if get_phase(board, opp) == Phase::Fly {
        return 0;
    }
    let opp_bits = board.bits(opp);
    let own = board.bits(color);
    let mut total = 0;
    for &mm in &MILL_MASKS {
        if (opp_bits & mm) != mm {
            continue;
        }
        let mut covered = 0u32;
        let mut mill_bits = mm;
        while mill_bits != 0 {
            let sq = mill_bits.trailing_zeros() as usize;
            mill_bits &= mill_bits - 1;
            covered |= ADJACENCY[sq] & own & !mm;
        }
        total += covered.count_ones() as i64;
    }
    total
}

pub fn fly_asymmetry(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    let color_fly = board.placed(color) >= 9 && board.count(color) == 3;
    let opp_fly = board.placed(opp) >= 9 && board.count(opp) == 3;
    if color_fly && !opp_fly {
        return 1;
    }
    if opp_fly && !color_fly && board.count(color) <= 5 {
        return -1;
    }
    0
}

pub fn open_mill_domination(board: &Board, color: Color) -> i64 {
    let opp = color.opponent();
    let own_pieces = board.count(color) as i64;
    let opp_pieces = board.count(opp) as i64;
    if own_pieces < 6 || opp_pieces > 5 {
        return 0;
    }
    (two_configs(board, color) - (opp_pieces - 1)).max(0)
}

/// Simple leaf evaluator matching Python `evaluate_v2` exactly.
/// Two passes: O(24) board scan + O(16) mill scan. No helper calls beyond
/// `closed_mills` and `two_configs` which are already O(16).
///
/// Weights:
///   Place: piece(1) mob(1) blocked(8) mill(30) threat(15)
///   Move:  piece(12) mob(1) opp_blocked(48) mill(30) threat(18) zugzwang(600)
///   Fly:   piece(2) mill(32) threat(80) surplus(900)
pub fn evaluate_v2(board: &Board, color: Color) -> i64 {
    if let Some(winner) = terminal_winner(board) {
        return if winner == color { INF } else { -INF };
    }
    let opp = color.opponent();
    let own_p = board.count(color) as i64;
    let opp_p = board.count(opp) as i64;
    // Piece-loss terminal (mirrors Python's fast check).
    if board.placed(color) >= 9 && own_p < 3 {
        return -INF;
    }
    if board.placed(opp) >= 9 && opp_p < 3 {
        return INF;
    }
    let phase = get_phase(board, color);
    let own_bits = board.bits(color);
    let opp_bits = board.bits(opp);
    let empty = board.empty();

    // Mobility + blocked: iterate only over occupied squares via set-bit loop.
    let mut own_mob: i64 = 0;
    let mut opp_mob: i64 = 0;
    let mut own_blocked: i64 = 0;
    let mut opp_blocked: i64 = 0;
    let mut bits = own_bits;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let free = (ADJACENCY[sq] & empty).count_ones() as i64;
        own_mob += free;
        if free == 0 { own_blocked += 1; }
    }
    let mut bits = opp_bits;
    while bits != 0 {
        let sq = bits.trailing_zeros() as usize;
        bits &= bits - 1;
        let free = (ADJACENCY[sq] & empty).count_ones() as i64;
        opp_mob += free;
        if free == 0 { opp_blocked += 1; }
    }

    // Blockade: if own side cannot move it's a loss.
    if phase == Phase::Move && own_mob == 0 {
        return -INF;
    }

    // Single pass over 16 mill masks: compute closed mills + two-configs for both sides.
    let mut own_mills: i64 = 0;
    let mut opp_mills: i64 = 0;
    let mut own_thr: i64 = 0;
    let mut opp_thr: i64 = 0;
    for &mm in &MILL_MASKS {
        let own_in = (own_bits & mm).count_ones();
        let opp_in = (opp_bits & mm).count_ones();
        let emp_in = (empty & mm).count_ones();
        match own_in {
            3 => own_mills += 1,
            2 if emp_in == 1 => own_thr += 1,
            _ => {}
        }
        match opp_in {
            3 => opp_mills += 1,
            2 if emp_in == 1 => opp_thr += 1,
            _ => {}
        }
    }
    match phase {
        Phase::Place => {
            (own_p - opp_p)
                + (own_mob - opp_mob)
                + 8 * (opp_blocked - own_blocked)
                + 30 * (own_mills - opp_mills)
                + 15 * (own_thr - opp_thr)
        }
        Phase::Move => {
            let mut score = 12 * (own_p - opp_p)
                + (own_mob - opp_mob)
                + 48 * opp_blocked
                + 30 * (own_mills - opp_mills)
                + 18 * (own_thr - opp_thr);
            if opp_mob < 3 {
                score += 600 * (3 - opp_mob);
            }
            score
        }
        Phase::Fly => {
            let own_surp = (own_thr - 1).max(0);
            let opp_surp = (opp_thr - 1).max(0);
            2 * (own_p - opp_p)
                + 32 * (own_mills - opp_mills)
                + 80 * (own_thr - opp_thr)
                + 900 * (own_surp - opp_surp)
        }
    }
}

/// Integer base evaluate from `color`'s perspective. Mirrors the base formula in
/// `ai/heuristics.py::evaluate` (terminal + base term sum). Float-scaled
/// extras are intentionally omitted (see module doc).
pub fn evaluate_base(board: &Board, color: Color) -> i64 {
    if let Some(winner) = terminal_winner(board) {
        return if winner == color { INF } else { -INF };
    }
    let opp = color.opponent();
    let phase = get_phase(board, color);
    let w = weights(phase);

    let our_mills = closed_mills(board, color);
    let opp_mills = closed_mills(board, opp);
    let blocked = blocked_count(board, opp);
    let piece_diff = board.count(color) as i64 - board.count(opp) as i64;
    let our_two = two_configs(board, color);
    let opp_two = two_configs(board, opp);
    let our_dbl = double_mills(board, color);
    let opp_dbl = double_mills(board, opp);
    let win_cfg = win_config(board, opp);
    let our_mob = mobility(board, color);
    let opp_mob = mobility(board, opp);
    let our_thr = mill_threats(board, color);
    let opp_thr = mill_threats(board, opp);
    let our_pos = position_value(board, color);
    let opp_pos = position_value(board, opp);
    let our_cycle = mill_cycle_ready(board, color);
    let opp_cycle = mill_cycle_ready(board, opp);
    let our_fork = fork_threats(board, color);
    let opp_fork = fork_threats(board, opp);
    let our_herd = encirclement(board, color);
    let opp_herd = encirclement(board, opp);
    let our_squeeze = squeeze_count(board, opp); // opponent near-blocked (good)
    let opp_squeeze = squeeze_count(board, color); // own near-blocked (bad)
    let our_wrap = mill_wrapping_pressure(board, color);
    let opp_wrap = mill_wrapping_pressure(board, opp);
    let fly_asym = fly_asymmetry(board, color);
    let our_dom = open_mill_domination(board, color);
    let opp_dom = open_mill_domination(board, opp);

    w[0] * (our_mills - opp_mills)
        + w[1] * blocked
        + w[2] * piece_diff
        + w[3] * (our_two - opp_two)
        + w[4] * (our_dbl - opp_dbl)
        + w[5] * win_cfg
        + mob_weight(phase) * (our_mob - opp_mob)
        + threat_weight(phase) * (our_thr - opp_thr)
        + 4 * (our_pos - opp_pos)
        + cycle_weight(phase) * (our_cycle - opp_cycle)
        + fork_weight(phase) * (our_fork - opp_fork)
        + herd_weight(phase) * (our_herd - opp_herd)
        + near_blocked_weight(phase) * (our_squeeze - opp_squeeze)
        + wrap_weight(phase) * (our_wrap - opp_wrap)
        + fly_asym_weight(phase) * fly_asym
        + domination_weight(phase) * (our_dom - opp_dom)
}
