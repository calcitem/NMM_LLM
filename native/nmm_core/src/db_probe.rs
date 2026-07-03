//! DB key generation + in-process binary-search probe for T-C2.
//! Key format must be byte-identical to Python. See `docs/RUST_INTEGRATION_PLAN.md` §9.

use crate::board::board24_string;
use crate::symmetry::canonical_board_str;
use crate::types::Board;

const FGDB_HEADER_SIZE: usize = 32;
const FGDB_RECORD_SIZE: usize = 36;
const FGDB_KEY_SIZE: usize = 9;
const FGDB_OUTCOME_OFFSET: usize = 9;

/// `_PIECE_BITS`: '.'=0, 'W'=1, 'B'=2.
#[inline]
fn piece_bits(ch: u8) -> u64 {
    match ch {
        b'W' => 1,
        b'B' => 2,
        _ => 0,
    }
}

/// FullGame DB 9-byte key. Mirrors `ai/fullgame_db.py::_encode_canonical`
/// applied to the canonical board string:
///   6-byte LE packed 2-bit/square + turn(0/1) + placed_w + placed_b
pub fn fullgame_key(white: u32, black: u32, turn: u8, placed_w: u8, placed_b: u8) -> Vec<u8> {
    let board24 = board24_string(white, black);
    let (canon, _sym) = canonical_board_str(&board24);
    let bytes = canon.as_bytes();
    let mut val: u64 = 0;
    for (i, &ch) in bytes.iter().enumerate() {
        val |= piece_bits(ch) << (i * 2);
    }
    // 6 little-endian bytes of val, then turn, placed_w, placed_b.
    let le = val.to_le_bytes();
    let mut out = Vec::with_capacity(9);
    out.extend_from_slice(&le[..6]);
    out.push(if turn == 0 { 0 } else { 1 });
    out.push(placed_w);
    out.push(placed_b);
    out
}

/// Inline 9-byte key for T-C2 (no Vec allocation; uses Board directly).
pub fn fullgame_key_inline(board: &Board) -> [u8; 9] {
    let board24 = board24_string(board.white, board.black);
    let (canon, _sym) = canonical_board_str(&board24);
    let bytes = canon.as_bytes();
    let mut val: u64 = 0;
    for (i, &ch) in bytes.iter().enumerate() {
        val |= piece_bits(ch) << (i * 2);
    }
    let le = val.to_le_bytes();
    let mut out = [0u8; 9];
    out[..6].copy_from_slice(&le[..6]);
    // turn: 0=White, 1=Black (matches Python's turn encoding)
    out[6] = match board.side_to_move {
        crate::types::Color::White => 0,
        crate::types::Color::Black => 1,
    };
    out[7] = board.white_placed;
    out[8] = board.black_placed;
    out
}

/// Binary-search the mmap'd fullgame DB; returns outcome from White's perspective:
/// +1=W win, -1=B win, 0=draw, None=not found or unknown.
pub fn probe_fullgame(data: &[u8], board: &Board) -> Option<i8> {
    if data.len() < FGDB_HEADER_SIZE + 4 {
        return None;
    }
    // Record count at header bytes 10-13 (u32 LE).
    let n_records = u32::from_le_bytes(
        data[10..14].try_into().ok()?
    ) as usize;
    let key = fullgame_key_inline(board);
    let mut lo = 0usize;
    let mut hi = n_records;
    while lo < hi {
        let mid = lo + (hi - lo) / 2;
        let off = FGDB_HEADER_SIZE + mid * FGDB_RECORD_SIZE;
        if off + FGDB_RECORD_SIZE > data.len() {
            break;
        }
        let rec_key = &data[off..off + FGDB_KEY_SIZE];
        match rec_key.cmp(&key[..]) {
            std::cmp::Ordering::Equal => {
                return match data[off + FGDB_OUTCOME_OFFSET] {
                    1 => Some(1),   // W_win
                    2 => Some(-1),  // B_win
                    3 => Some(0),   // draw
                    _ => None,      // unknown
                };
            }
            std::cmp::Ordering::Less => lo = mid + 1,
            std::cmp::Ordering::Greater => hi = mid,
        }
    }
    None
}

/// Endgame DB key string: "<canonical board24>|<turn>" (turn as 'W'/'B').
pub fn endgame_key(white: u32, black: u32, turn: u8) -> String {
    let board24 = board24_string(white, black);
    let (canon, _sym) = canonical_board_str(&board24);
    let t = if turn == 0 { 'W' } else { 'B' };
    format!("{canon}|{t}")
}

// ── T-C3: EndgameSolvedDB probe ───────────────────────────────────────────────
//
// Mirrors ai/endgame_solved_db.py exactly.
// File format: endgame_{nW}_{nB}.wdl — 2 bits per position, 4 packed per byte.
// WDL values: 0=unknown, 1=Win (stm), 2=Loss (stm), 3=Draw.
// Position ID: combinatorial number system over white piece indices,
// then remapped black piece indices, times 2 for turn.

/// C(n, k): exact integer binomial coefficient. Returns 0 for k > n.
pub fn comb(n: u64, k: u64) -> u64 {
    if k > n { return 0; }
    let k = k.min(n - k);
    if k == 0 { return 1; }
    (1..=k).fold(1u64, |acc, i| acc * (n - k + i) / i)
}

/// Combinatorial rank of a sorted ascending slice of indices.
/// rank = Σ_i C(sorted_indices[i], i+1)
fn combo_rank(sorted_indices: &[u8]) -> u64 {
    sorted_indices.iter().enumerate()
        .map(|(i, &c)| comb(c as u64, (i + 1) as u64))
        .sum()
}

/// Encode (white_mask, black_mask, stm) to a direct position ID for a .wdl file.
/// Returns None if piece counts are outside the [3..=7] range supported by the DB.
pub fn encode_endgame_pos_id(board: &crate::types::Board) -> Option<u64> {
    let nw = board.white.count_ones() as u64;
    let nb = board.black.count_ones() as u64;
    if !(3..=7).contains(&nw) || !(3..=7).contains(&nb) { return None; }

    // Sorted piece indices for white (0..23).
    let mut w_indices: Vec<u8> = (0u8..24).filter(|&i| board.white & (1 << i) != 0).collect();
    w_indices.sort_unstable();

    // Sorted piece indices for black.
    let mut b_raw: Vec<u8> = (0u8..24).filter(|&i| board.black & (1 << i) != 0).collect();
    b_raw.sort_unstable();

    // Remaining squares after removing white pieces (in order).
    let w_mask = board.white;
    let remaining: Vec<u8> = (0u8..24).filter(|&i| w_mask & (1 << i) == 0).collect();

    // Remap black indices into the compressed remaining-squares space.
    let b_remapped: Vec<u8> = b_raw.iter()
        .map(|&b| remaining.iter().position(|&r| r == b).unwrap() as u8)
        .collect();

    let white_rank = combo_rank(&w_indices);
    let black_rank = combo_rank(&b_remapped);
    let black_combinations = comb(24 - nw, nb);
    let turn_bit = if board.side_to_move == crate::types::Color::White { 0u64 } else { 1u64 };

    Some(white_rank * black_combinations * 2 + black_rank * 2 + turn_bit)
}

/// Read the 2-bit WDL value for pos_id from a packed .wdl byte table.
#[inline]
fn get_wdl(data: &[u8], pos_id: u64) -> u8 {
    let byte_idx = (pos_id >> 2) as usize;
    let shift = ((pos_id & 3) << 1) as u32;
    if byte_idx >= data.len() { return 0; }
    (data[byte_idx] >> shift) & 3
}

/// Probe a single mmap'd .wdl table for the given board.
/// Returns: Some(+1) = stm wins, Some(-1) = stm loses, Some(0) = draw, None = unknown/OOB.
pub fn probe_endgame_solved(data: &[u8], board: &crate::types::Board) -> Option<i8> {
    let pos_id = encode_endgame_pos_id(board)?;
    match get_wdl(data, pos_id) {
        1 => Some(1),   // stm wins
        2 => Some(-1),  // stm loses
        3 => Some(0),   // draw
        _ => None,      // unknown
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_is_nine_bytes() {
        let k = fullgame_key(0, 0, 0, 0, 0);
        assert_eq!(k.len(), 9);
        assert_eq!(k, vec![0, 0, 0, 0, 0, 0, 0, 0, 0]);
    }

    #[test]
    fn turn_and_counts_encoded() {
        let k = fullgame_key(0, 0, 1, 5, 7);
        assert_eq!(k[6], 1);
        assert_eq!(k[7], 5);
        assert_eq!(k[8], 7);
    }
}
