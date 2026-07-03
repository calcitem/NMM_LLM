//! Zobrist hashing + transposition table for the Rust-internal search.
//!
//! T-E1: TT slots are now two AtomicU64 per entry (xor_key + data), enabling
//! shared Arc<TranspositionTable> across Lazy-SMP helper threads. Xor-key trick
//! (Stockfish-style) gives integrity checking of torn reads for free.
//!
//! Data layout (u64):
//!   bits  0-31: score (i32 stored as u32 — scores are bounded by ±INF=10M)
//!   bits 32-39: depth (u8)
//!   bits 40-41: flag (u8, 2 bits; 0=EXACT, 1=LOWER, 2=UPPER)
//!   bits 42-57: best_idx (u16; u16::MAX = no move)
//!   bits 58-63: zero
//!
//! data=0 is the empty sentinel: depth=0 → treated as miss (negamax never stores depth=0).

use std::sync::atomic::{AtomicU64, Ordering};
use crate::types::{Board, Color, N_SQUARES};

pub const EXACT: u8 = 0;
pub const LOWER_BOUND: u8 = 1;
pub const UPPER_BOUND: u8 = 2;

const TT_BITS: usize = 20; // 2^20 slots
const TT_SIZE: usize = 1 << TT_BITS;
const TT_MASK: u64 = (TT_SIZE as u64) - 1;

/// SplitMix64 for deterministic key generation (fixed seed).
struct SplitMix64(u64);
impl SplitMix64 {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }
}

pub struct Zobrist {
    piece: [[u64; N_SQUARES]; 2],
    placed_done: [u64; 2],
    side: u64,
}

impl Zobrist {
    pub fn new() -> Self {
        let mut rng = SplitMix64(0x9E3779B97F4A7C15);
        let mut piece = [[0u64; N_SQUARES]; 2];
        for c in 0..2 {
            for s in 0..N_SQUARES {
                piece[c][s] = rng.next();
            }
        }
        let placed_done = [rng.next(), rng.next()];
        let side = rng.next();
        Zobrist {
            piece,
            placed_done,
            side,
        }
    }

    pub fn hash(&self, board: &Board) -> u64 {
        let mut h = 0u64;
        let mut w = board.white;
        while w != 0 {
            let i = w.trailing_zeros() as usize;
            h ^= self.piece[0][i];
            w &= w - 1;
        }
        let mut b = board.black;
        while b != 0 {
            let i = b.trailing_zeros() as usize;
            h ^= self.piece[1][i];
            b &= b - 1;
        }
        if board.white_placed >= 9 {
            h ^= self.placed_done[0];
        }
        if board.black_placed >= 9 {
            h ^= self.placed_done[1];
        }
        if board.side_to_move == Color::Black {
            h ^= self.side;
        }
        h
    }
}

#[derive(Clone, Copy)]
pub struct TtEntry {
    pub depth: u8,
    pub score: i32,
    pub flag: u8,
    pub best_idx: u16, // u16::MAX = no move
}

#[inline]
fn pack(score: i32, depth: u8, flag: u8, best_idx: u16) -> u64 {
    (score as u32 as u64)
        | ((depth as u64) << 32)
        | ((flag as u64 & 0x03) << 40)
        | ((best_idx as u64) << 42)
}

#[inline]
fn unpack(data: u64) -> (i32, u8, u8, u16) {
    let score = data as u32 as i32;
    let depth = (data >> 32) as u8;
    let flag = ((data >> 40) & 0x03) as u8;
    let best_idx = ((data >> 42) & 0xFFFF) as u16;
    (score, depth, flag, best_idx)
}

/// T-E1: lock-free TT using two AtomicU64 slots per entry (xor-key trick).
/// layout: table[2*i] = xor_key = key ^ data, table[2*i+1] = data.
/// Sync (AtomicU64 is Sync) — safe to share via Arc<TranspositionTable>.
pub struct TranspositionTable {
    table: Vec<AtomicU64>,
}

impl TranspositionTable {
    pub fn new() -> Self {
        let table: Vec<AtomicU64> = (0..TT_SIZE * 2).map(|_| AtomicU64::new(0)).collect();
        TranspositionTable { table }
    }

    pub fn clear(&self) {
        for slot in self.table.iter() {
            slot.store(0, Ordering::Relaxed);
        }
    }

    pub fn lookup(&self, key: u64) -> Option<TtEntry> {
        let idx = (key & TT_MASK) as usize;
        let xor_key = self.table[idx * 2].load(Ordering::Relaxed);
        let data = self.table[idx * 2 + 1].load(Ordering::Relaxed);
        if xor_key ^ data != key {
            return None;
        }
        let (score, depth, flag, best_idx) = unpack(data);
        if depth == 0 {
            return None;
        }
        Some(TtEntry { depth, score, flag, best_idx })
    }

    pub fn store(&self, key: u64, entry: TtEntry) {
        let idx = (key & TT_MASK) as usize;
        // Depth-preferred replacement (racy but benign in SMP context).
        let existing_data = self.table[idx * 2 + 1].load(Ordering::Relaxed);
        if existing_data != 0 {
            let (_, existing_depth, _, _) = unpack(existing_data);
            if existing_depth > entry.depth {
                return;
            }
        }
        let data = pack(entry.score, entry.depth, entry.flag, entry.best_idx);
        // Write data before xor_key so a torn read sees a failed XOR check.
        self.table[idx * 2 + 1].store(data, Ordering::Relaxed);
        self.table[idx * 2].store(key ^ data, Ordering::Relaxed);
    }
}
