"""ai/malom_db.py — Read-only adapter for the Malom ultra-strong NMM database.

The Malom database (ggevay/malom, GPL-3, by Gabor E. Gevay and Gabor Danner)
provides a solved endgame for Nine Men's Morris.  The database lives outside
the repo (typically /mnt/windows/NMM_DB/strong/).

File format  (DD / version-2 .sec2 files)
------------------------------------------
Bytes 0–63     : 64-byte header
    int32 version   (= 2)
    int32 esize     (= 3, bytes per entry)
    int32 f2off     (= 12, bit offset of field2 within 3-byte entry)
    int32 sdflag    (= 0, stone-diff flag; must be 0)
    remaining bytes : zero-padded to 64 bytes

Bytes 64 … 64+N*3-1 : N×3-byte entries, little-endian 24-bit words.
    raw24   = b[0] | (b[1]<<8) | (b[2]<<16)
    key1    = sign_extend(raw24 & 0xFFF, 12)   # sector-relative WDL value
    key2    = sign_extend(raw24 >> 12,   12)   # depth / DTW (signed)

Bytes 64+N*3 … end:
    int32 em_set_size   (number of overflow entries)
    em_set_size × (int32 key, int32 val)   # 8 bytes per pair

Entry semantics
---------------
key1 == virt_win_val  (+299) → position is a WIN  for the querying side
key1 == virt_loss_val (-299) → position is a LOSS for the querying side
key1 == 0                    → DRAW
key1 == 0 AND key2 > 0       → still in retrograde progress (Count state);
                               should not appear in the finished database.

Turn-side convention
--------------------
The Malom board type stores: low 24 bits = White piece bits, high 24 bits = Black
piece bits.  There is NO explicit side-to-move bit.  Instead:
  • To query for White-to-move: call hash(W_bits | (B_bits<<24), W, B, WF, BF).
  • To query for Black-to-move: swap the white and black piece bitboards AND swap
    the W/B and WF/BF counts before calling hash.
The returned key1 is always from the perspective of the player whose pieces occupy
the *low 24 bits* of the board — i.e. the current mover.

Bit numbering (Malom positions 0–23)
--------------------------------------
The Malom source uses a specific bit order for the 24 board positions.  This
mapping was verified against the `rot90`, `tt_fuggoleges` symmetry functions and
the `millpos` constants in movegen.cpp.

    Bit  0 = a4   Bit  1 = a7   Bit  2 = d7   Bit  3 = g7
    Bit  4 = g4   Bit  5 = g1   Bit  6 = d1   Bit  7 = a1
    Bit  8 = b4   Bit  9 = b6   Bit 10 = d6   Bit 11 = f6
    Bit 12 = f4   Bit 13 = f2   Bit 14 = d2   Bit 15 = b2
    Bit 16 = c4   Bit 17 = c5   Bit 18 = d5   Bit 19 = e5
    Bit 20 = e4   Bit 21 = e3   Bit 22 = d3   Bit 23 = c3

Hash function (translated from Malom C++ hash.cpp)
---------------------------------------------------
The hash is a two-part combinatorial index:
    h = f_lookup[W_bits] * C(24-W, B) + g_lookup[collapse(board)]

where:
  f_lookup[W_bits] = canonical orbit index of the White piece configuration
                     under the 16 board symmetries (4 rotations × 4 reflections,
                     plus the outer/inner ring swap).
  collapse(board)  = black piece bits compressed into (24-W) bit positions by
                     removing the slots occupied by White pieces.
  g_lookup[x]      = combinatorial rank of the compressed Black configuration.

The hash is initialised once per (W, B) pair and cached in _HASH_CACHE.

Public surface
--------------
    MalomDB(db_dir)
        .is_available()  → True when .sec2 files are found in db_dir
        .query(board)    → {"outcome": "W"|"L"|"D", "dtw": int} | None
        .close()

    parse_secval(path)       → (virt_win, virt_loss, {(W,B,WF,BF): int})
    board_to_wbf(board)      → (wb_bits, bb_bits, wf, bf)
    read_sector(path)        → (data, hash_count, em_set, virt_win, virt_loss)
    decode_entry(data, idx, em_set, virt_win, virt_loss) → "W"|"L"|"D"|None
"""

from __future__ import annotations

import logging
import struct
from math import comb
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Board position mapping ─────────────────────────────────────────────────────
#
# Maps Malom bit index (0–23) to our algebraic position names.
# Verified against millpos constants in Malom movegen.cpp (GPL-3):
#   millpos[0]=14=bits{1,2,3}=a7,d7,g7  (top side)
#   millpos[1]=56=bits{3,4,5}=g7,g4,g1  (right side)
#   millpos[2]=224=bits{5,6,7}=g1,d1,a1 (bottom side)
#   millpos[3]=131=bits{0,1,7}=a4,a7,a1 (left side)
#
MALOM_BITS_TO_POS: list[str] = [
    # Outer ring (bits 0–7)
    "a4", "a7", "d7", "g7", "g4", "g1", "d1", "a1",
    # Middle ring (bits 8–15)
    "b4", "b6", "d6", "f6", "f4", "f2", "d2", "b2",
    # Inner ring (bits 16–23)
    "c4", "c5", "d5", "e5", "e4", "e3", "d3", "c3",
]

# Reverse mapping: algebraic position name → Malom bit index
_POS_TO_MALOM_BIT: dict[str, int] = {pos: i for i, pos in enumerate(MALOM_BITS_TO_POS)}

# ── Header constants ───────────────────────────────────────────────────────────

_HEADER_SIZE = 64        # bytes
_ESIZE = 3               # bytes per entry
_FIELD2_OFFSET = 12      # bits (for STANDARD variant)
_FIELD1_SIZE = 12        # bits
_FIELD2_SIZE = 12        # bits  (= 8*3 - 12)

# ── Symmetry operations (translated from symmetries_slow.cpp, GPL-3) ──────────
#
# The 16 symmetry operations of the square extended with outer/inner ring swap.
# Each is a permutation of bit positions 0–23.
# Order matches the `slow[]` array in symmetries.cpp:
#   {rot90, rot180, rot270, tt_fugg, tt_vizs, tt_bslash, tt_slash,
#    swap, swap_rot90, ..., id}
#
# The `inv[]` array from symmetries.cpp:
#   int inv[] = {2,1,0,3,4,5,6,7,10,9,8,11,12,13,14,15};
# gives the inverse symmetry index for each operation.

def _make_perm_table(perm: list[int]) -> list[int]:
    """Build a precomputed 256-entry lookup for 8-bit inputs under `perm`."""
    t = [0] * 256
    for pat in range(256):
        r = 0
        for bit in range(8):
            if (pat >> bit) & 1:
                r |= 1 << perm[bit]
        t[pat] = r
    return t

# Permutations for each of the 8 base positions within one 8-position ring:
#   ring: 0=left-mid, 1=top-left, 2=top-mid, 3=top-right,
#         4=right-mid, 5=bot-right, 6=bot-mid, 7=bot-left
# (indices are relative within the ring, i.e. add ring_offset before use)
_ROT90_RING   = [2,3,4,5,6,7,0,1]  # 90° clockwise
_ROT180_RING  = [4,5,6,7,0,1,2,3]  # 180°
_ROT270_RING  = [6,7,0,1,2,3,4,5]  # 270° clockwise
_TTFUGG_RING  = [4,3,2,1,0,7,6,5]  # vertical flip
_TTVIZS_RING  = [0,7,6,5,4,3,2,1]  # horizontal flip
_TTBSLASH_RING= [2,1,0,7,6,5,4,3]  # diagonal flip (\)
_TTSLASH_RING = [6,5,4,3,2,1,0,7]  # diagonal flip (/)

def _ring_perm_to_full(ring_perm: list[int]) -> list[int]:
    """Expand a ring-local 8-permutation to a full 24-position permutation
    (same permutation applied independently to each of the three rings)."""
    full = [0] * 24
    for ring in range(3):
        off = ring * 8
        for i in range(8):
            full[off + i] = off + ring_perm[i]
    return full

def _swap_perm() -> list[int]:
    """Swap outer ring (bits 0–7) with inner ring (bits 16–23)."""
    perm = list(range(24))
    for i in range(8):
        perm[i], perm[16+i] = 16+i, i
    return perm

def _compose(p1: list[int], p2: list[int]) -> list[int]:
    """p2 after p1: new_pos[i] = p2[p1[i]]."""
    return [p2[p1[i]] for i in range(24)]


def _sym24_from_perm(perm: list[int], a: int) -> int:
    """Apply a 24-position permutation to a 24-bit board integer."""
    r = 0
    for i in range(24):
        if (a >> i) & 1:
            r |= 1 << perm[i]
    return r


# Pre-build the 16 full permutations
_ROT90_FULL   = _ring_perm_to_full(_ROT90_RING)
_ROT180_FULL  = _ring_perm_to_full(_ROT180_RING)
_ROT270_FULL  = _ring_perm_to_full(_ROT270_RING)
_TTFUGG_FULL  = _ring_perm_to_full(_TTFUGG_RING)
_TTVIZS_FULL  = _ring_perm_to_full(_TTVIZS_RING)
_TTBSLASH_FULL= _ring_perm_to_full(_TTBSLASH_RING)
_TTSLASH_FULL = _ring_perm_to_full(_TTSLASH_RING)
_SWAP_FULL    = _swap_perm()
_ID_FULL      = list(range(24))

# Order matches Malom's slow[] array (indices 0–15)
_SYM_PERMS: list[list[int]] = [
    _ROT90_FULL,                              # 0
    _ROT180_FULL,                             # 1
    _ROT270_FULL,                             # 2
    _TTFUGG_FULL,                             # 3
    _TTVIZS_FULL,                             # 4
    _TTBSLASH_FULL,                           # 5
    _TTSLASH_FULL,                            # 6
    _SWAP_FULL,                               # 7
    _compose(_SWAP_FULL, _ROT90_FULL),        # 8  swap_rot90
    _compose(_SWAP_FULL, _ROT180_FULL),       # 9  swap_rot180
    _compose(_SWAP_FULL, _ROT270_FULL),       # 10 swap_rot270
    _compose(_SWAP_FULL, _TTFUGG_FULL),       # 11 swap_tt_fugg
    _compose(_SWAP_FULL, _TTVIZS_FULL),       # 12 swap_tt_vizs
    _compose(_SWAP_FULL, _TTBSLASH_FULL),     # 13 swap_tt_bslash
    _compose(_SWAP_FULL, _TTSLASH_FULL),      # 14 swap_tt_slash
    _ID_FULL,                                 # 15 identity (last)
]

# Inverse symmetry table from Malom symmetries.cpp:
# int inv[] = {2,1,0,3,4,5,6,7,10,9,8,11,12,13,14,15};
_SYM_INV: list[int] = [2,1,0,3,4,5,6,7,10,9,8,11,12,13,14,15]

# ── Hash cache (one HashState per (W,B) sector pair) ──────────────────────────

class _HashState:
    """Pre-built lookup tables for hashing W white + B black pieces."""

    __slots__ = ("W", "B", "f_lookup", "f_sym_lookup", "g_lookup", "hash_count")

    def __init__(self, W: int, B: int) -> None:
        self.W = W
        self.B = B
        self._build(W, B)

    # ── combinatorial helpers ──────────────────────────────────────────────

    @staticmethod
    def _next_choose(x: int) -> int:
        """Return the next k-combination in bit representation (Gosper's hack)."""
        if x == 0:
            return 1 << 24
        c = x & (-x)
        r = x + c
        return (((r ^ x) >> 2) // c) | r

    # ── build ──────────────────────────────────────────────────────────────

    def _build(self, W: int, B: int) -> None:
        f_lookup: dict[int, int] = {}   # W_bits → canonical orbit index
        f_sym_lookup: dict[int, int] = {} # W_bits → symmetry op that brings to canonical

        c = 0
        w = (1 << W) - 1
        while w < (1 << 24):
            if w not in f_lookup:
                for i in range(16):
                    sw = _sym24_from_perm(_SYM_PERMS[i], w)
                    f_lookup[sw] = c
                    f_sym_lookup[sw] = _SYM_INV[i]
                c += 1
            w = self._next_choose(w)

        # g_lookup: compressed-black-bits → rank
        g_lookup: dict[int, int] = {}
        gc = 0
        b = (1 << B) - 1
        while b < (1 << (24 - W)):
            g_lookup[b] = gc
            gc += 1
            b = self._next_choose(b)

        self.f_lookup = f_lookup
        self.f_sym_lookup = f_sym_lookup
        self.g_lookup = g_lookup
        self.hash_count = c * comb(24 - W, B)

    # ── hash ──────────────────────────────────────────────────────────────

    def hash(self, w_bits: int, b_bits: int) -> int:
        """Return the hash index for a (white_bits, black_bits) board.

        Applies the canonical symmetry to white first, then uses the same
        operation on black, then computes collapse + g_lookup.
        """
        sym_op = self.f_sym_lookup[w_bits]
        cw = _sym24_from_perm(_SYM_PERMS[sym_op], w_bits)
        cb = _sym24_from_perm(_SYM_PERMS[sym_op], b_bits)

        collapsed = _collapse(cw, cb, self.W)

        return self.f_lookup[cw] * comb(24 - self.W, self.B) + self.g_lookup[collapsed]


# Module-level hash state cache: (W, B) → _HashState
_HASH_CACHE: dict[tuple[int, int], _HashState] = {}


def _get_hash_state(W: int, B: int) -> _HashState:
    key = (W, B)
    if key not in _HASH_CACHE:
        _HASH_CACHE[key] = _HashState(W, B)
    return _HASH_CACHE[key]


# ── Collapse ───────────────────────────────────────────────────────────────────

def _collapse(w_bits: int, b_bits: int, W: int) -> int:
    """Compress b_bits into (24-W) bits by removing the slots where w_bits is set.

    Translated from collapse() in hash.cpp (GPL-3).
    """
    r = 0
    j = 1
    for i in range(24):
        bit = 1 << i
        if not (w_bits & bit):
            if b_bits & bit:
                r |= j
            j <<= 1
    return r


# ── Section value parsing ──────────────────────────────────────────────────────

def parse_secval(path: str | Path) -> tuple[int, int, dict[tuple[int,int,int,int], int]]:
    """Parse std.secval and return (virt_win, virt_loss, sector_vals_dict).

    The secval file format (text):
        virt_loss_val: <int>
        virt_win_val: <int>
        <count>
        <W> <B> <WF> <BF>  <sec_val>
        ...

    Returns
    -------
    virt_win  : int   (positive; currently 299)
    virt_loss : int   (negative; currently -299)
    secvals   : dict mapping (W, B, WF, BF) → sec_val (int)
    """
    path = Path(path)
    virt_win = 299
    virt_loss = -299
    secvals: dict[tuple[int,int,int,int], int] = {}

    with open(path, "r", encoding="ascii", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith("virt_loss_val:"):
                virt_loss = int(line.split(":")[1].strip())
            elif line.startswith("virt_win_val:"):
                virt_win = int(line.split(":")[1].strip())
            else:
                parts = line.split()
                if len(parts) == 5:
                    try:
                        W, B, WF, BF = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                        sv = int(parts[4])
                        secvals[(W, B, WF, BF)] = sv
                    except ValueError:
                        pass

    return virt_win, virt_loss, secvals


# ── Board conversion ───────────────────────────────────────────────────────────

def board_to_wbf(board) -> tuple[int, int, int, int]:
    """Convert a BoardState to Malom (wb_bits, bb_bits, wf, bf).

    Returns
    -------
    wb_bits : 24-bit int with a 1 for each White piece (Malom bit numbering)
    bb_bits : 24-bit int with a 1 for each Black piece (Malom bit numbering)
    wf      : stones still to be placed by White (= max(0, 9 - pieces_placed["W"]))
    bf      : stones still to be placed by Black
    """
    wb_bits = 0
    bb_bits = 0
    for pos, color in board.positions.items():
        bit = _POS_TO_MALOM_BIT.get(pos)
        if bit is None:
            continue
        if color == "W":
            wb_bits |= 1 << bit
        elif color == "B":
            bb_bits |= 1 << bit

    wf = max(0, 9 - board.pieces_placed["W"])
    bf = max(0, 9 - board.pieces_placed["B"])
    return wb_bits, bb_bits, wf, bf


# ── Sector file reading ────────────────────────────────────────────────────────

def read_sector(path: str | Path,
                virt_win: int = 299,
                virt_loss: int = -299
                ) -> tuple[bytes, int, dict[int, int], int, int]:
    """Open and validate a .sec2 file.

    Returns
    -------
    data       : raw entry bytes (N × 3 bytes)
    hash_count : number of entries N
    em_set     : overflow dict {index → key2_value}
    virt_win   : from header validation (passed through)
    virt_loss  : from header validation (passed through)

    Raises ValueError if the header is invalid.
    """
    path = Path(path)
    with open(path, "rb") as f:
        header = f.read(_HEADER_SIZE)
        if len(header) < _HEADER_SIZE:
            raise ValueError(f"Header too short in {path}")

        version    = struct.unpack_from("<i", header, 0)[0]
        esize      = struct.unpack_from("<i", header, 4)[0]
        f2off      = struct.unpack_from("<i", header, 8)[0]
        sdflag     = struct.unpack_from("<i", header, 12)[0]

        if version != 2:
            raise ValueError(f"Unexpected version {version} in {path}")
        if esize != 3:
            raise ValueError(f"Unexpected esize {esize} in {path} (expected 3)")
        if f2off != 12:
            raise ValueError(f"Unexpected field2_offset {f2off} in {path} (expected 12)")
        if sdflag != 0:
            raise ValueError(f"stone_diff_flag={sdflag} not supported (expected 0)")

        # Read entry data
        file_size = path.stat().st_size
        # em_set footer: 4-byte count + count*8 bytes of (int32,int32) pairs
        # Read the last 4 bytes to get em_set_size first
        f.seek(-4, 2)
        raw_tail = f.read(4)
        # Can't determine em_set_size from tail alone; read from after data.
        # Compute: data_size = file_size - header_size - 4 - em_set_size*8
        # Try em_set_size = 0 first, then scan if size doesn't work out
        data_size_candidate = file_size - _HEADER_SIZE - 4
        if data_size_candidate % 3 == 0:
            # em_set_size = 0 case: data runs from header to (file_size - 4)
            f.seek(_HEADER_SIZE)
            data = f.read(data_size_candidate)
            hash_count = data_size_candidate // 3
            em_set_size = struct.unpack_from("<i", f.read(4))[0]
            if em_set_size != 0:
                # There are overflow entries; re-read with proper layout
                # data_size = file_size - header - 4 - em_set_size*8
                data_size = file_size - _HEADER_SIZE - 4 - em_set_size * 8
                if data_size % 3 != 0:
                    raise ValueError(f"Cannot determine layout in {path}: data_size={data_size}")
                f.seek(_HEADER_SIZE)
                data = f.read(data_size)
                hash_count = data_size // 3
                raw_em_size = f.read(4)
                em_set_size = struct.unpack_from("<i", raw_em_size)[0]
                em_set: dict[int, int] = {}
                for _ in range(em_set_size):
                    kv = f.read(8)
                    k, v = struct.unpack_from("<ii", kv)
                    em_set[k] = v
            else:
                em_set = {}
        else:
            # Try to find a valid layout with em_set_size > 0
            found = False
            for em_sz in range(1, 1000):
                ds = file_size - _HEADER_SIZE - 4 - em_sz * 8
                if ds >= 0 and ds % 3 == 0:
                    f.seek(_HEADER_SIZE)
                    data = f.read(ds)
                    hash_count = ds // 3
                    raw_em_size = f.read(4)
                    em_set_size = struct.unpack_from("<i", raw_em_size)[0]
                    if em_set_size == em_sz:
                        em_set = {}
                        for _ in range(em_set_size):
                            kv = f.read(8)
                            k, v = struct.unpack_from("<ii", kv)
                            em_set[k] = v
                        found = True
                        break
            if not found:
                raise ValueError(f"Cannot determine layout in {path}")

    return data, hash_count, em_set, virt_win, virt_loss


# ── Entry decoding ─────────────────────────────────────────────────────────────

def _sign_extend_12(val: int) -> int:
    if val & 0x800:
        return val - 0x1000
    return val


def decode_entry(data: bytes,
                 idx: int,
                 em_set: dict[int, int],
                 virt_win: int,
                 virt_loss: int):
    """Decode entry at index `idx` from raw data bytes.

    Returns
    -------
    "W"                : position is won  for the querying side (mover)
    "L"                : position is lost for the querying side (mover)
    "D"                : draw
    ("SYM", sym_op)    : Sym redirect — caller must apply sym_op to the
                         already-canonicalized board, rehash, and call again.
                         sym_op is an integer 0–15 into _SYM_PERMS.
    None               : Count state (in-progress; should not appear in
                         finished DB) or em_set overflow with unexpected key2.

    The em_set maps an index to an overflowed key2 value when the 12-bit field
    in the 3-byte entry is insufficient.

    Sym state semantics (from eval_elem.cpp, GPL-3):
        cas() == Sym  when  key1 == 0  and  key2 < 0
        sym_op = -(key2 + 1)
    The Sym redirect is guaranteed non-Sym by a C++ assert in hash.cpp.
    """
    off = idx * _ESIZE
    b0, b1, b2 = data[off], data[off + 1], data[off + 2]
    raw24 = b0 | (b1 << 8) | (b2 << 16)

    key1 = _sign_extend_12(raw24 & 0xFFF)
    key2_raw = _sign_extend_12(raw24 >> 12)

    # Check for em_set overflow (key2 field has a sentinel value)
    # Sentinel: key2 = spec_field2 = -(1 << (field2_size-1)) = -2048
    _SPEC_FIELD2 = -(1 << (_FIELD2_SIZE - 1))  # = -2048
    if key2_raw == _SPEC_FIELD2:
        if idx in em_set:
            key2 = em_set[idx]
        else:
            # Unexpected: spec_field2 without em_set entry; treat as unknown
            return None
    else:
        key2 = key2_raw

    # Determine entry type via cas() semantics from eval_elem.cpp (GPL-3):
    #   key1 != 0  → Val   (Win or Loss)
    #   key1 == 0, key2 > 0  → Count  (retrograde in progress)
    #   key1 == 0, key2 == 0 → Draw
    #   key1 == 0, key2 < 0  → Sym    (redirect to another entry)
    if key1 == 0:
        if key2 > 0:
            # Count state: retrograde not finished; should not be in released DB
            return None
        if key2 < 0:
            # Sym redirect: the canonical representative is at a different hash.
            # sym_op is the symmetry to apply to the already-canonicalized board.
            sym_op = -(key2 + 1)
            return ("SYM", sym_op)
        # key2 == 0 → Draw
        return "D"

    if key1 == virt_win:
        return "W"
    if key1 == virt_loss:
        return "L"

    # Non-virtual values may appear in intermediate sectors; map by sign
    if key1 > 0:
        return "W"
    return "L"


# ── MalomDB ────────────────────────────────────────────────────────────────────

class MalomDB:
    """Read-only adapter for the Malom ultra-strong NMM database.

    Usage::

        db = MalomDB("/mnt/windows/NMM_DB/strong")
        if db.is_available():
            result = db.query(board_state)
            # result = {"outcome": "W"|"L"|"D", "dtw": int} or None
    """

    def __init__(self, db_dir: str | Path) -> None:
        self._db_dir = Path(db_dir)
        self._virt_win = 299
        self._virt_loss = -299
        self._secvals: dict[tuple[int,int,int,int], int] = {}
        # Cache: sector key → (data, hash_count, em_set)
        self._cache: dict[tuple[int,int,int,int], tuple[bytes,int,dict[int,int]]] = {}
        self._available = False
        self._warned = False
        self._load_secval()

    # ── Initialisation ────────────────────────────────────────────────────

    def _load_secval(self) -> None:
        secval_path = self._db_dir / "std.secval"
        if not secval_path.exists():
            return
        try:
            vw, vl, secvals = parse_secval(secval_path)
            self._virt_win = vw
            self._virt_loss = vl
            self._secvals = secvals
            self._available = any(self._db_dir.glob("std_*.sec2"))
        except Exception as exc:
            logger.warning("[MalomDB] failed to load secval: %s", exc)

    # ── Availability ──────────────────────────────────────────────────────

    def is_available(self) -> bool:
        """True when .sec2 files exist in the configured database directory."""
        return self._available

    # ── Internal helpers ──────────────────────────────────────────────────

    def _get_sector(self, path: Path, sector: tuple[int,int,int,int]
                    ) -> Optional[tuple[bytes, int, dict[int,int]]]:
        if sector in self._cache:
            return self._cache[sector]
        if not path.exists():
            return None
        try:
            data, hash_count, em_set, _, _ = read_sector(
                path, self._virt_win, self._virt_loss
            )
            self._cache[sector] = (data, hash_count, em_set)
            return self._cache[sector]
        except Exception as exc:
            logger.warning("[MalomDB] failed to read sector %s: %s", path.name, exc)
            return None

    # ── Public query ──────────────────────────────────────────────────────

    def query(self, board) -> Optional[dict]:
        """Return the WDL outcome for the side to move, or None.

        Parameters
        ----------
        board : BoardState — the position to query

        Returns
        -------
        {"outcome": "W"|"L"|"D", "dtw": int}  on success
        None  if the position cannot be looked up (DB unavailable,
              sector file missing, or hash failure)

        The "outcome" is from the perspective of the CURRENT MOVER (board.turn).
        "dtw" is the depth-to-win (positive) or depth-to-loss (negative);
        it comes from field2 of the entry.  Its exact semantics depend on which
        sector the position belongs to; treat it as a relative quality hint.
        """
        if not self._available:
            if not self._warned:
                self._warned = True
                logger.info("[MalomDB] DB unavailable — all queries return None")
            return None

        wb, bb, wf, bf = board_to_wbf(board)
        W = bin(wb).count("1")
        B = bin(bb).count("1")

        # Side-to-move convention:
        # The Malom hash treats the first argument's pieces as "White" (low bits).
        # To get an outcome from the current mover's perspective, we place the
        # current mover's pieces in the "White" (low) position and query the
        # appropriately swapped sector.
        if board.turn == "W":
            qw, qb, qW, qB, qWF, qBF = wb, bb, W, B, wf, bf
        else:
            # Black to move: treat Black as "White" for the hash query
            qw, qb, qW, qB, qWF, qBF = bb, wb, B, W, bf, wf

        sector = (qW, qB, qWF, qBF)
        sec_fname = f"std_{qW}_{qB}_{qWF}_{qBF}.sec2"
        sec_path = self._db_dir / sec_fname

        cached = self._get_sector(sec_path, sector)
        if cached is None:
            return None

        data, hash_count, em_set = cached

        # Step 1: canonicalize white bits via f_sym_lookup, then compute h1
        hs = _get_hash_state(qW, qB)
        try:
            sym_op_canon = hs.f_sym_lookup[qw]
        except KeyError:
            return None
        cw = _sym24_from_perm(_SYM_PERMS[sym_op_canon], qw)
        cb = _sym24_from_perm(_SYM_PERMS[sym_op_canon], qb)

        collapsed = _collapse(cw, cb, qW)
        try:
            idx = hs.f_lookup[cw] * comb(24 - qW, qB) + hs.g_lookup[collapsed]
        except KeyError:
            return None
        if idx < 0 or idx >= hash_count:
            return None

        entry = decode_entry(data, idx, em_set, self._virt_win, self._virt_loss)

        # Step 2: Sym redirect — apply a second symmetry to the already-canonical
        # board, recompute h2, and read the guaranteed-non-Sym entry.
        # Translated from Hash::hash() in Malom hash.cpp (GPL-3).
        if isinstance(entry, tuple) and entry[0] == "SYM":
            sym_op2 = entry[1]
            cw2 = _sym24_from_perm(_SYM_PERMS[sym_op2], cw)
            cb2 = _sym24_from_perm(_SYM_PERMS[sym_op2], cb)
            # After a second symmetry the canonical White position may differ;
            # look it up directly (it must already be in f_lookup).
            try:
                collapsed2 = _collapse(cw2, cb2, qW)
                idx = hs.f_lookup[cw2] * comb(24 - qW, qB) + hs.g_lookup[collapsed2]
            except KeyError:
                return None
            if idx < 0 or idx >= hash_count:
                return None
            entry = decode_entry(data, idx, em_set, self._virt_win, self._virt_loss)
            # Per C++ assert this must not be another Sym; if it is, bail out.
            if isinstance(entry, tuple):
                logger.warning("[MalomDB] unexpected double Sym redirect at idx=%d", idx)
                return None

        if entry is None:
            return None

        outcome = entry

        # Extract dtw from field2
        off = idx * _ESIZE
        b0, b1, b2 = data[off], data[off + 1], data[off + 2]
        raw24 = b0 | (b1 << 8) | (b2 << 16)
        dtw = _sign_extend_12(raw24 >> 12)

        return {"outcome": outcome, "dtw": dtw}

    def close(self) -> None:
        """Release cached sector data."""
        self._cache.clear()

    def __repr__(self) -> str:
        return (
            f"MalomDB(db_dir={str(self._db_dir)!r}, "
            f"available={self._available}, "
            f"virt_win={self._virt_win}, virt_loss={self._virt_loss})"
        )
