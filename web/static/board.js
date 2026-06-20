/**
 * board.js — SVG Nine Men's Morris board rendering.
 *
 * All 24 node positions use centred coordinates where d4 is (0,0),
 * columns a–g map to –3…+3, rows 1–7 map to –3…+3.
 * In SVG y increases downward, so row = –svgY.
 */

const CX = 300, CY = 300, SCALE = 80;
const PIECE_R = 22, NODE_R = 7, HINT_R = 17;

const NODE_COORDS = {
  a7:[-3, 3], d7:[0, 3], g7:[3, 3],
  g4:[3, 0],  g1:[3,-3], d1:[0,-3], a1:[-3,-3], a4:[-3,0],
  b6:[-2, 2], d6:[0, 2], f6:[2, 2],
  f4:[2, 0],  f2:[2,-2], d2:[0,-2], b2:[-2,-2], b4:[-2,0],
  c5:[-1, 1], d5:[0, 1], e5:[1, 1],
  e4:[1, 0],  e3:[1,-1], d3:[0,-1], c3:[-1,-1], c4:[-1,0],
};

// Board lines: each entry is an array of node names forming a continuous path.
const BOARD_LINES = [
  // Outer square
  ["a7","d7","g7","g4","g1","d1","a1","a4","a7"],
  // Middle square
  ["b6","d6","f6","f4","f2","d2","b2","b4","b6"],
  // Inner square
  ["c5","d5","e5","e4","e3","d3","c3","c4","c5"],
  // Cross connections (top, right, bottom, left)
  ["d7","d6","d5"],
  ["g4","f4","e4"],
  ["d1","d2","d3"],
  ["a4","b4","c4"],
];

// The 16 valid mills for flash detection
const MILLS = [
  ["a7","d7","g7"],["g7","g4","g1"],["g1","d1","a1"],["a1","a4","a7"],
  ["b6","d6","f6"],["f6","f4","f2"],["f2","d2","b2"],["b2","b4","b6"],
  ["c5","d5","e5"],["e5","e4","e3"],["e3","d3","c3"],["c3","c4","c5"],
  ["d7","d6","d5"],["g4","f4","e4"],["d1","d2","d3"],["a4","b4","c4"],
];

function nodeXY(name) {
  const [col, row] = NODE_COORDS[name];
  return [CX + col * SCALE, CY - row * SCALE];
}

export class Board {
  constructor(svgEl, onNodeClick) {
    this.svg        = svgEl;
    this.onNodeClick = onNodeClick;
    this.grid       = {};       // position → "W"|"B"|null
    this.legalDests = new Set();
    this.legalSrcs  = new Set();
    this.selected   = null;     // currently selected source node
    this.phase      = "place";
    this.isHuman    = true;
    this.capMode    = false;    // awaiting capture click
    this.legalCaps  = new Set();
    this._millNodes = new Set();
    this._init();
  }

  _init() {
    const svg = this.svg;
    svg.innerHTML = "";
    // Extra space at left (30px) and bottom (30px) for coordinate labels
    svg.setAttribute("viewBox", "-30 0 630 630");

    // Arrow-marker defs for DB overlay
    const defs = _el("defs");
    for (const [id, col] of [["arr-green","#4caf50"],["arr-red","#e05050"],["arr-grey","#666"],["arr-blue","#7bbfff"]]) {
      const mk = _el("marker", { id, markerWidth:"7", markerHeight:"5", refX:"5", refY:"2.5", orient:"auto" });
      const poly = _el("polygon", { points:"0 0,7 2.5,0 5", fill:col });
      mk.appendChild(poly);
      defs.appendChild(mk);
    }
    svg.appendChild(defs);

    // Board background
    const bg = _el("rect", { x:20, y:20, width:560, height:560, rx:8,
      fill:"#c8a96e", stroke:"#7a5230", "stroke-width":2 });
    svg.appendChild(bg);

    // Lines
    const lineGroup = _el("g", { stroke:"#5c3318", "stroke-width":3, "stroke-linecap":"round" });
    for (const path of BOARD_LINES) {
      for (let i = 0; i < path.length - 1; i++) {
        const [x1,y1] = nodeXY(path[i]);
        const [x2,y2] = nodeXY(path[i+1]);
        lineGroup.appendChild(_el("line", { x1, y1, x2, y2 }));
      }
    }
    svg.appendChild(lineGroup);

    // Coordinate labels
    const labelStyle = { fill:"#5c3318", "font-size":"18", "font-weight":"bold",
      "font-family":"monospace", "text-anchor":"middle", "dominant-baseline":"middle" };
    const labelGroup = _el("g");
    const cols = ["a","b","c","d","e","f","g"];
    const colX  = [-3,-2,-1,0,1,2,3].map(c => CX + c * SCALE);
    const rows = ["7","6","5","4","3","2","1"];
    const rowY  = [3,2,1,0,-1,-2,-3].map(r => CY - r * SCALE);
    cols.forEach((lbl, i) => {
      const t = _el("text", { x:colX[i], y:600, ...labelStyle });
      t.textContent = lbl;
      labelGroup.appendChild(t);
    });
    rows.forEach((lbl, i) => {
      const t = _el("text", { x:-10, y:rowY[i], ...labelStyle });
      t.textContent = lbl;
      labelGroup.appendChild(t);
    });
    svg.appendChild(labelGroup);

    // Node markers
    this._nodeEls = {};
    const nodeGroup = _el("g");
    for (const name of Object.keys(NODE_COORDS)) {
      const [x, y] = nodeXY(name);
      const circle = _el("circle", { cx:x, cy:y, r:NODE_R,
        fill:"#b8935a", stroke:"#5c3318", "stroke-width":2, cursor:"pointer" });
      circle.dataset.node = name;
      circle.addEventListener("click", () => this.onNodeClick(name));
      this._nodeEls[name] = circle;
      nodeGroup.appendChild(circle);
    }
    svg.appendChild(nodeGroup);

    // Piece layer
    this._pieceGroup = _el("g");
    svg.appendChild(this._pieceGroup);

    // Hint layer above pieces so capture/hint rings intercept clicks correctly
    this._hintGroup = _el("g");
    svg.appendChild(this._hintGroup);

    // Mill flash overlay
    this._millGroup = _el("g", { opacity:0, "pointer-events":"none" });
    svg.appendChild(this._millGroup);

    // Hint overlay — temporary move suggestion rings, above everything
    this._hintOverlay = _el("g", { "pointer-events":"none" });
    svg.appendChild(this._hintOverlay);
    this._hintTimer = null;

    // DB overlay — trajectory/fullgame/endgame arrows and halos; below score labels
    this._dbGroup = _el("g", { "pointer-events":"none" });
    svg.appendChild(this._dbGroup);

    // Diagnostic overlay — score labels; topmost, never intercepts clicks
    this._diagGroup = _el("g", { "pointer-events":"none" });
    svg.appendChild(this._diagGroup);
    this._diagSelected = null;  // source square selected for movement diag
  }

  render(state) {
    const newGrid = state.board || {};
    this.grid        = newGrid;
    this.phase       = state.phase;
    this.isHuman     = state.is_human_turn;
    this.legalDests  = new Set(state.legal_dests || []);
    this.legalSrcs   = new Set(state.legal_sources || []);
    this.capMode     = false;
    this.legalCaps   = new Set();
    this.selected    = null;
    this.clearHint();
    this._drawPieces();
    this._drawHints();
  }

  enterCapture(legalCaps) {
    this.capMode  = true;
    this.legalCaps = new Set(legalCaps);
    this._drawHints();
  }

  _drawPieces() {
    this._pieceGroup.innerHTML = "";
    for (const [name, color] of Object.entries(this.grid)) {
      if (!color) continue;
      const [x, y] = nodeXY(name);
      const g = _el("g", { "data-node": name });

      // Shadow
      g.appendChild(_el("circle", { cx:x+2, cy:y+2, r:PIECE_R,
        fill:"rgba(0,0,0,0.25)" }));

      // Piece body
      const fill   = color === "W" ? "#f2ede0" : "#1e1a2e";
      const stroke = color === "W" ? "#888" : "#444";
      const body   = _el("circle", { cx:x, cy:y, r:PIECE_R,
        fill, stroke, "stroke-width":2 });
      g.appendChild(body);

      // Highlight ring for selected
      if (name === this.selected) {
        g.appendChild(_el("circle", { cx:x, cy:y, r:PIECE_R+4,
          fill:"none", stroke:"#f4c542", "stroke-width":3,
          opacity:0.9 }));
      }

      // Mill glow
      if (this._millNodes.has(name)) {
        g.appendChild(_el("circle", { cx:x, cy:y, r:PIECE_R+6,
          fill:"none", stroke:"#ff4444", "stroke-width":3,
          opacity:0.7 }));
      }

      this._pieceGroup.appendChild(g);
    }
  }

  _drawHints() {
    this._hintGroup.innerHTML = "";
    if (!this.isHuman) return;

    if (this.capMode) {
      // Red rings on capturable opponent pieces
      for (const name of this.legalCaps) {
        const [x, y] = nodeXY(name);
        const ring = _el("circle", { cx:x, cy:y, r:PIECE_R+5,
          fill:"rgba(220,50,50,0.25)", stroke:"#e03030", "stroke-width":3,
          cursor:"pointer" });
        ring.dataset.node = name;
        ring.addEventListener("click", () => this.onNodeClick(name));
        this._hintGroup.appendChild(ring);
      }
      return;
    }

    if (this.phase === "place") {
      // Green dots on empty legal destinations
      for (const name of this.legalDests) {
        if (this.grid[name]) continue;
        const [x, y] = nodeXY(name);
        const dot = _el("circle", { cx:x, cy:y, r:HINT_R,
          fill:"rgba(60,180,80,0.30)", stroke:"rgba(60,180,80,0.7)",
          "stroke-width":2, cursor:"pointer" });
        dot.dataset.node = name;
        dot.addEventListener("click", () => this.onNodeClick(name));
        this._hintGroup.appendChild(dot);
      }
    } else {
      // Movement: dim own-piece dots on legal sources; bright destination dots
      if (!this.selected) {
        for (const name of this.legalSrcs) {
          const [x, y] = nodeXY(name);
          const dot = _el("circle", { cx:x, cy:y, r:PIECE_R+5,
            fill:"rgba(244,197,66,0.25)", stroke:"#f4c542",
            "stroke-width":2, cursor:"pointer" });
          dot.dataset.node = name;
          dot.addEventListener("click", () => this.onNodeClick(name));
          this._hintGroup.appendChild(dot);
        }
      } else {
        // Show destinations for selected piece
        const legal = this._legalDestsFor(this.selected);
        for (const name of legal) {
          const [x, y] = nodeXY(name);
          const dot = _el("circle", { cx:x, cy:y, r:HINT_R,
            fill:"rgba(60,180,80,0.30)", stroke:"rgba(60,180,80,0.7)",
            "stroke-width":2, cursor:"pointer" });
          dot.dataset.node = name;
          dot.addEventListener("click", () => this.onNodeClick(name));
          this._hintGroup.appendChild(dot);
        }
        // Cancel: clicking selected piece again deselects
        const [sx, sy] = nodeXY(this.selected);
        const cancel = _el("circle", { cx:sx, cy:sy, r:PIECE_R+5,
          fill:"rgba(244,197,66,0.25)", stroke:"#f4c542",
          "stroke-width":2, cursor:"pointer" });
        cancel.addEventListener("click", () => {
          this.selected = null;
          this._drawPieces();
          this._drawHints();
        });
        this._hintGroup.appendChild(cancel);
      }
    }
  }

  _legalDestsFor(src) {
    // Ask the server-provided legal_dests for moves FROM src.
    // We store legalMoves as a flat list set; we need to expose src→dests.
    // The state message sends legal_dests for the CURRENT turn, but without
    // per-source pairing — we re-compute from what we have.
    // For now, expose this.legalMovePairs set (set by render if provided).
    return (this._movePairs || [])
      .filter(([f]) => f === src)
      .map(([, t]) => t);
  }

  setMovePairs(pairs) {
    // pairs: [[from, to], ...]
    this._movePairs = pairs;
  }

  selectSource(name) {
    this.selected = name;
    this._drawPieces();
    this._drawHints();
  }

  showHint(from_pos, to_pos) {
    this.clearHint();
    if (from_pos) {
      // Movement: amber pulse on the piece to move
      const [fx, fy] = nodeXY(from_pos);
      this._hintOverlay.appendChild(_el("circle", {
        cx:fx, cy:fy, r:PIECE_R + 8,
        fill:"rgba(244,197,66,0.30)", stroke:"#f4c542", "stroke-width":3,
      }));
    }
    if (to_pos) {
      // Destination: blue pulse
      const [tx, ty] = nodeXY(to_pos);
      this._hintOverlay.appendChild(_el("circle", {
        cx:tx, cy:ty, r:PIECE_R + 8,
        fill:"rgba(60,150,255,0.30)", stroke:"#4099ff", "stroke-width":3,
      }));
    }
    this._hintTimer = setTimeout(() => this.clearHint(), 4000);
  }

  clearHint() {
    clearTimeout(this._hintTimer);
    this._hintTimer = null;
    this._hintOverlay.innerHTML = "";
  }

  // Render board positions from a FEN string (for move replay).
  // FEN format: "<24chars>|<turn>|<W_placed>|<B_placed>"
  // Positions are in canonical order matching board.py POSITIONS list.
  renderFromFen(fen) {
    const POSITIONS = [
      "a7","d7","g7","g4","g1","d1","a1","a4",
      "b6","d6","f6","f4","f2","d2","b2","b4",
      "c5","d5","e5","e4","e3","d3","c3","c4",
    ];
    const chars = fen.split("|")[0] || "";
    this.grid = {};
    for (let i = 0; i < POSITIONS.length; i++) {
      const ch = chars[i];
      this.grid[POSITIONS[i]] = ch === "W" ? "W" : ch === "B" ? "B" : null;
    }
    this._millNodes = new Set();
    this._hintGroup.innerHTML = "";
    this._hintOverlay.innerHTML = "";
    this._drawPieces();
  }

  flashMills(boardGrid, color) {
    // Find any mills belonging to `color` in the current grid and flash them.
    const mills = MILLS.filter(m => m.every(n => boardGrid[n] === color));
    const nodes = new Set(mills.flat());
    this._millNodes = nodes;
    this._drawPieces();
    setTimeout(() => {
      this._millNodes = new Set();
      this._drawPieces();
    }, 800);
  }

  clearDiag() {
    this._diagGroup.innerHTML = "";
    this._dbGroup.innerHTML   = "";
    this._diagSelected = null;
  }

  // ── DB overlay (trajectory / fullgame / endgame) ──────────────────────────
  // moves: [{from, to, capture, traj_freq, db_delta, eg_flag}, ...]
  // opts: {phase, selectedSrc, showTraj, showDB}
  renderDiagDB(moves, opts = {}) {
    this._dbGroup.innerHTML = "";
    if (!moves || !moves.length) return;

    const phase        = opts.phase || "move";
    const selSrc       = opts.selectedSrc || null;
    const showTraj     = opts.showTraj !== false;
    const showDB       = opts.showDB  !== false;
    const showSentinel = opts.showSentinel || false;
    const showOverseer = opts.showOverseer || false;
    const visFrac      = opts.visibilityFraction != null ? opts.visibilityFraction : 1.0;

    // Deterministic per-move hash for stable visibility thinning (no flicker on redraws)
    const _mvHash = mv => {
      const s = (mv.from || "") + (mv.to || "");
      let h = 0;
      for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) | 0;
      return Math.abs(h);
    };

    // Helper: arrow/halo color from db_delta or eg_flag only.
    // Sentinel scores no longer drive circle/arrow colours — they appear as
    // S:XX% text labels only, and can co-exist with the DB/Malom overlay.
    const dbColor = (delta, egFlag) => {
      if (egFlag === "W") return "#4caf50";
      if (egFlag === "L") return "#e05050";
      if (egFlag === "D") return "#888";    // endgame draw — keep grey
      if (delta == null)  return null;
      if (delta > 0.1)    return "#4caf50";
      if (delta < -0.1)   return "#e05050";
      return null;   // neutral FullGame delta — no display (not enough info)
    };
    const markerId = col =>
      col === "#4caf50" ? "arr-green" : col === "#e05050" ? "arr-red" : "arr-grey";

    // Helper: sentinel score label text ("S:63%"), or null when not applicable
    const sentinelLabel = (sentinelScore) => {
      if (!showSentinel || sentinelScore == null) return null;
      return `S:${Math.round(sentinelScore * 100)}%`;
    };
    // Helper: overseer pick-probability label ("O:45%"), or null when not applicable.
    // Suppress when prob < 1% (noise) — k==1 always hits 100% and is also suppressed.
    const overseerLabel = (prob) => {
      if (!showOverseer || prob == null) return null;
      const pct = Math.round(prob * 100);
      if (pct < 1) return null;
      return `O:${pct}%`;
    };

    if (phase === "place" || phase === "capture") {
      // Halos on destination squares — no arrows needed (no source)
      for (const mv of moves) {
        if (visFrac < 1.0 && (_mvHash(mv) % 100) >= Math.round(visFrac * 100)) continue;
        const pos = mv.to;
        if (!pos) continue;
        const col  = dbColor(showDB ? mv.db_delta : null, showDB ? mv.eg_flag : null);
        const freq = showTraj ? (mv.traj_freq || 0) : 0;
        const slbl = sentinelLabel(mv.sentinel_score);
        const olbl = overseerLabel(mv.overseer_prob);

        if (col) {
          const [x, y] = nodeXY(pos);
          this._dbGroup.appendChild(_el("circle", { cx:x, cy:y, r: PIECE_R + 9,
            fill: "none", stroke: col, "stroke-width": 2.5, opacity: 0.7 }));
        }
        if (slbl || olbl || freq > 0) {
          const [x, y] = nodeXY(pos);
          const hasPiece = !!this.grid[pos];
          let ty = hasPiece ? y - PIECE_R - 5 : y - NODE_R - 5;
          if (freq > 0) {
            const t = _el("text", { x, y: ty, "font-size":"9", fill:"#5a10c0",
              "text-anchor":"middle", "font-family":"monospace",
              stroke:"white", "stroke-width":"2.5", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = `T:${Math.round(freq * 100)}%`;
            this._dbGroup.appendChild(t);
            ty -= 11;
          }
          if (slbl) {
            const sentCol = (mv.sentinel_score >= 0.55) ? "#66bb6a"
                          : (mv.sentinel_score <= 0.45) ? "#ef5350" : "#bdbdbd";
            const t = _el("text", { x, y: ty, "font-size":"10", "font-weight":"bold",
              fill: sentCol, "text-anchor":"middle", "font-family":"monospace",
              stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = slbl;
            this._dbGroup.appendChild(t);
            ty -= 11;
          }
          if (olbl) {
            const t = _el("text", { x, y: ty, "font-size":"10", "font-weight":"bold",
              fill: "#f5a623", "text-anchor":"middle", "font-family":"monospace",
              stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = olbl;
            this._dbGroup.appendChild(t);
          }
        }
      }
    } else {
      // Movement / fly phase
      const toRender = selSrc
        ? moves.filter(m => m.from === selSrc)
        : moves;

      for (const mv of toRender) {
        if (visFrac < 1.0 && (_mvHash(mv) % 100) >= Math.round(visFrac * 100)) continue;
        if (!mv.from || !mv.to) continue;
        const col  = dbColor(showDB ? mv.db_delta : null, showDB ? mv.eg_flag : null);
        const freq = showTraj ? (mv.traj_freq || 0) : 0;
        const slbl = selSrc ? sentinelLabel(mv.sentinel_score) : null;
        const olbl = selSrc ? overseerLabel(mv.overseer_prob)  : null;
        if (!col && freq === 0 && !slbl && !olbl) continue;

        const [x1, y1] = nodeXY(mv.from);
        const [x2, y2] = nodeXY(mv.to);

        if (col) {
          // Shorten line so arrowhead doesn't overlap pieces
          const dx = x2 - x1, dy = y2 - y1;
          const dist = Math.sqrt(dx*dx + dy*dy) || 1;
          const shrink = 28;
          const sx = x1 + dx/dist * PIECE_R;
          const ex = x2 - dx/dist * shrink;
          const sy = y1 + dy/dist * PIECE_R;
          const ey = y2 - dy/dist * shrink;
          if (dist > PIECE_R * 2) {
            this._dbGroup.appendChild(_el("line", {
              x1: sx, y1: sy, x2: ex, y2: ey,
              stroke: col, "stroke-width": 2,
              "marker-end": `url(#${markerId(col)})`,
              opacity: 0.75,
            }));
          }
        }
        if (slbl || olbl || freq > 0) {
          const [x, y] = nodeXY(mv.to);
          let ty = y - PIECE_R - 4;
          if (freq > 0 && (!selSrc || mv.from === selSrc)) {
            const t = _el("text", { x: x + 1, y: ty, "font-size":"8", fill:"#5a10c0",
              "text-anchor":"middle", "font-family":"monospace",
              stroke:"white", "stroke-width":"2.5", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = `T:${Math.round(freq * 100)}%`;
            this._dbGroup.appendChild(t);
            ty -= 10;
          }
          if (slbl) {
            const sentCol = (mv.sentinel_score >= 0.55) ? "#66bb6a"
                          : (mv.sentinel_score <= 0.45) ? "#ef5350" : "#bdbdbd";
            const t = _el("text", { x: x + 1, y: ty, "font-size":"10", "font-weight":"bold",
              fill: sentCol, "text-anchor":"middle", "font-family":"monospace",
              stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = slbl;
            this._dbGroup.appendChild(t);
            ty -= 11;
          }
          if (olbl) {
            const t = _el("text", { x: x + 1, y: ty, "font-size":"10", "font-weight":"bold",
              fill: "#f5a623", "text-anchor":"middle", "font-family":"monospace",
              stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
              "paint-order":"stroke" });
            t.textContent = olbl;
            this._dbGroup.appendChild(t);
          }
        }
      }

      // If no source selected, ring sources with the best DB/Malom colour.
      // Sentinel-only mode draws no rings (sentinel appears as labels only).
      if (!selSrc && showDB) {
        const srcDB = new Map();  // source → best db color
        for (const mv of moves) {
          const col = dbColor(mv.db_delta, mv.eg_flag);
          if (col && col !== "#888" && mv.from) {
            if (!srcDB.has(mv.from) || col === "#4caf50")
              srcDB.set(mv.from, col);
          }
        }
        for (const [src, col] of srcDB) {
          const [x, y] = nodeXY(src);
          this._dbGroup.appendChild(_el("circle", { cx:x, cy:y, r: PIECE_R + 7,
            fill:"none", stroke: col, "stroke-width":2, opacity:0.55, "stroke-dasharray":"4 3" }));
        }
      }

      // Per-source best sentinel label when no piece is selected.
      // Shows the highest-quality move available for each piece as S:XX% above it.
      if (!selSrc && showSentinel) {
        const srcBest = new Map();  // source → highest sentinel_score across its moves
        for (const mv of moves) {
          if (mv.from && mv.sentinel_score != null) {
            const prev = srcBest.get(mv.from);
            if (prev == null || mv.sentinel_score > prev)
              srcBest.set(mv.from, mv.sentinel_score);
          }
        }
        for (const [src, score] of srcBest) {
          const [x, y] = nodeXY(src);
          const sentCol = (score >= 0.55) ? "#66bb6a"   // bright green
                        : (score <= 0.45) ? "#ef5350"   // bright red
                        : "#bdbdbd";                    // light grey — neutral
          const t = _el("text", { x, y: y - PIECE_R - 3, "font-size":"10", "font-weight":"bold",
            fill: sentCol, "text-anchor":"middle", "font-family":"monospace",
            stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
            "paint-order":"stroke" });
          t.textContent = `S:${Math.round(score * 100)}%`;
          this._dbGroup.appendChild(t);
        }
      }

      // Per-source best Overseer pick-probability when no piece is selected.
      // Shows the highest prob move for each piece as O:XX% above it.
      // Stacks above the sentinel label when both overlays are active.
      if (!selSrc && showOverseer) {
        // Build source → highest prob map and source → has_sentinel_label map in one pass.
        const srcBestOv   = new Map();  // source → highest overseer_prob
        const srcHasSent  = new Set();  // sources that have a non-null sentinel_score
        for (const mv of moves) {
          if (!mv.from) continue;
          if (mv.overseer_prob != null) {
            const prev = srcBestOv.get(mv.from);
            if (prev == null || mv.overseer_prob > prev)
              srcBestOv.set(mv.from, mv.overseer_prob);
          }
          if (showSentinel && mv.sentinel_score != null)
            srcHasSent.add(mv.from);
        }
        for (const [src, prob] of srcBestOv) {
          const olbl = overseerLabel(prob);
          if (!olbl) continue;
          const [x, y] = nodeXY(src);
          // Shift up by 11px when a sentinel label is also shown for this source.
          const sentOffset = srcHasSent.has(src) ? -11 : 0;
          const t = _el("text", { x, y: y - PIECE_R - 3 + sentOffset,
            "font-size":"10", "font-weight":"bold",
            fill: "#f5a623", "text-anchor":"middle", "font-family":"monospace",
            stroke:"#1a1208", "stroke-width":"3", "stroke-linejoin":"round",
            "paint-order":"stroke" });
          t.textContent = olbl;
          this._dbGroup.appendChild(t);
        }
      }
    }
  }

  // ── Diagnostic overlay ────────────────────────────────────────────────────
  // diagData: {mode, color, eval_w, eval_b, moves: [{from, to, capture, score, tac_total?, tac_terms?, eval_score?}]}
  // opts: {mode2?: {moves: [...]}, selectedSrc?: string, phase: "place"|"move"|"fly"|"capture"}
  renderDiag(diagData, opts = {}) {
    this._diagGroup.innerHTML = "";
    if (!diagData || !diagData.moves || !diagData.moves.length) return;

    const { moves, mode } = diagData;
    const phase   = opts.phase || diagData.phase || "move";
    const selSrc  = opts.selectedSrc || null;
    const moves2  = (opts.mode2 && opts.mode2.moves) ? opts.mode2.moves : null;

    // Build score lookup: pos → best score for that square (or source piece)
    // For movement phase (no selection): per-source aggregation = best outgoing score
    // For movement phase (selection active): per-destination scores
    // For placement / capture: per-destination (= to) scores

    // Score normalisation helpers
    const scores = moves.map(m => m.score);
    const minS = Math.min(...scores), maxS = Math.max(...scores);
    const scoreColor = s => {
      if (maxS === minS) return "#aaa";
      const t = (s - minS) / (maxS - minS);  // 0=worst, 1=best
      if (t > 0.65) return "#4caf50";          // green
      if (t < 0.35) return "#e05050";          // red
      return "#aaa";
    };

    const scores2 = moves2 ? moves2.map(m => m.score) : null;
    const min2 = scores2 ? Math.min(...scores2) : 0;
    const max2 = scores2 ? Math.max(...scores2) : 0;
    const scoreColor2 = s => {
      if (!scores2 || max2 === min2) return "#7bbfff";
      const t = (s - min2) / (max2 - min2);
      if (t > 0.65) return "#5bc8f5";
      if (t < 0.35) return "#f59b5b";
      return "#7bbfff";
    };

    const fmt = n => (n >= 0 ? `+${n}` : `${n}`);

    const renderLabel = (pos, label1, col1, label2, col2) => {
      const [x, y] = nodeXY(pos);
      const hasPiece = !!this.grid[pos];
      const ly = hasPiece ? y + PIECE_R + 11 : y + NODE_R + 11;

      const g = _el("g");
      // Background rect — width depends on whether we have two numbers
      const tw = label2 ? 74 : 44;
      g.appendChild(_el("rect", {
        x: x - tw/2, y: ly - 8, width: tw, height: 14, rx: 3,
        fill: "rgba(10,8,4,0.82)",
      }));
      if (label2) {
        const t1 = _el("text", { x: x - 4, y: ly + 3,
          "font-size": "9", fill: col1,
          "text-anchor": "end", "font-family": "monospace" });
        t1.textContent = label1;
        g.appendChild(t1);
        // separator
        const sep = _el("text", { x: x, y: ly + 3, "font-size": "9", fill: "#555",
          "text-anchor": "middle", "font-family": "monospace" });
        sep.textContent = "|";
        g.appendChild(sep);
        const t2 = _el("text", { x: x + 4, y: ly + 3,
          "font-size": "9", fill: col2,
          "text-anchor": "start", "font-family": "monospace" });
        t2.textContent = label2;
        g.appendChild(t2);
      } else {
        const t = _el("text", { x, y: ly + 3,
          "font-size": "9.5", fill: col1,
          "text-anchor": "middle", "font-family": "monospace" });
        t.textContent = label1;
        g.appendChild(t);
      }
      this._diagGroup.appendChild(g);
    };

    // Build lookup for mode2 scores keyed by "from|to"
    const mk2 = m => `${m.from||''}|${m.to||''}`;
    const map2 = moves2 ? new Map(moves2.map(m => [mk2(m), m])) : null;

    if (phase === "place" || phase === "capture") {
      // One label per destination square
      for (const mv of moves) {
        const pos = mv.to || mv.capture;
        if (!pos) continue;
        const col1 = scoreColor(mv.score);
        const s2 = map2 ? map2.get(mk2(mv)) : null;
        renderLabel(pos, fmt(mv.score), col1,
          s2 ? fmt(s2.score) : null,
          s2 ? scoreColor2(s2.score) : null);
      }
    } else {
      // Movement / fly phase
      if (selSrc) {
        // Selected: show destination scores for moves FROM selSrc
        const destMoves = moves.filter(m => m.from === selSrc);
        for (const mv of destMoves) {
          const col1 = scoreColor(mv.score);
          const s2 = map2 ? map2.get(mk2(mv)) : null;
          renderLabel(mv.to, fmt(mv.score), col1,
            s2 ? fmt(s2.score) : null,
            s2 ? scoreColor2(s2.score) : null);
        }
      } else {
        // No selection: aggregate best score per source piece
        const srcBest = new Map();    // pos → {score, mv}
        const srcBest2 = new Map();   // pos → {score}
        for (const mv of moves) {
          const src = mv.from;
          if (!src) continue;
          if (!srcBest.has(src) || mv.score > srcBest.get(src).score)
            srcBest.set(src, mv);
        }
        if (moves2) {
          for (const mv of moves2) {
            const src = mv.from;
            if (!src) continue;
            if (!srcBest2.has(src) || mv.score > srcBest2.get(src).score)
              srcBest2.set(src, mv);
          }
        }
        for (const [src, {score}] of srcBest) {
          const col1 = scoreColor(score);
          const s2 = srcBest2.get(src);
          renderLabel(src, fmt(score), col1,
            s2 ? fmt(s2.score) : null,
            s2 ? scoreColor2(s2.score) : null);
        }
      }
    }
  }
}

function _el(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
