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
    this._animFrom  = null;     // pending move-anim source (set before render)
    this._animTo    = null;     // pending move-anim destination
    this._movedPiece = null;    // { pos, dx, dy } for current render cycle
    this.legalDests = new Set();
    this.legalSrcs  = new Set();
    this.selected   = null;     // currently selected source node
    this.phase      = "place";
    this.isHuman    = true;
    this.capMode    = false;    // awaiting capture click
    this.legalCaps  = new Set();
    this._millNodes = new Set();
    this._newPieces = new Set(); // positions that gained a piece since last render
    this._init();
  }

  _init() {
    const svg = this.svg;
    svg.innerHTML = "";
    // Extra space at left (30px) and bottom (30px) for coordinate labels
    svg.setAttribute("viewBox", "-30 0 630 630");

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
  }

  setNextMoveAnim(from, to) {
    this._animFrom = from;
    this._animTo   = to;
  }

  animateMoveOptimistic(from, to) {
    if (!from || !to) { this._movedPiece = null; return; }
    const [ox, oy] = nodeXY(from);
    const [nx, ny] = nodeXY(to);
    this._movedPiece = { pos: to, dx: ox - nx, dy: oy - ny };
  }

  render(state) {
    const newGrid = state.board || {};
    this._newPieces  = new Set();
    this._movedPiece = null;
    for (const [pos, color] of Object.entries(newGrid)) {
      if (color && !this.grid[pos]) {
        if (this._animTo === pos && this._animFrom) {
          const [ox, oy] = nodeXY(this._animFrom);
          const [nx, ny] = nodeXY(pos);
          this._movedPiece = { pos, dx: ox - nx, dy: oy - ny };
        } else {
          this._newPieces.add(pos);
        }
      }
    }
    this._animFrom = null;
    this._animTo   = null;
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
    const animating  = this._newPieces;
    const movedPiece = this._movedPiece;
    this._newPieces  = new Set();
    this._movedPiece = null;
    this._pieceGroup.innerHTML = "";
    for (const [name, color] of Object.entries(this.grid)) {
      if (!color) continue;
      const [x, y] = nodeXY(name);
      const g = _el("g", { "data-node": name });
      if (animating.has(name)) {
        g.setAttribute("class", "piece-arrive");
        g.style.transformOrigin = `${x}px ${y}px`;
      } else if (movedPiece && movedPiece.pos === name) {
        const dx = movedPiece.dx, dy = movedPiece.dy, sg = g;
        g.setAttribute("transform", `translate(${dx},${dy})`);
        const t0 = performance.now(), dur = 250;
        const slide = ts => {
          const p    = Math.min((ts - t0) / dur, 1);
          const ease = 1 - Math.pow(1 - p, 3);
          if (sg.parentNode) {
            sg.setAttribute("transform", `translate(${dx*(1-ease)},${dy*(1-ease)})`);
            if (p < 1) requestAnimationFrame(slide);
            else sg.removeAttribute("transform");
          }
        };
        requestAnimationFrame(slide);
      }

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
}

function _el(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  return el;
}
