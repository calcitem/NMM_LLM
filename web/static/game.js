/**
 * game.js — WebSocket game controller for Nine Men's Morris.
 */
import { Board } from "./board.js";

const $ = id => document.getElementById(id);

// ── State ─────────────────────────────────────────────────────────────────────

let ws              = null;
let board           = null;
let gameState       = null;
let phase           = "idle";
let evalHistory     = [];     // [{move: n, score: f}] — history for the graph
let hintsLeft       = 3;      // server-tracked cap; synced via hint messages
let drawUnlocked    = false;  // true once 40 post-placement half-moves have passed
let forceAggressive = false;  // when true, AI ignores fly-sacrifice heuristic
let thinkingInterval  = null; // setInterval handle while AI is thinking
let thinkingStarted   = 0;    // Date.now() when thinking began
let thinkingExpected  = 0;    // expected seconds from server
let canMarkBad        = false; // true only between ai_move and the next human move commit

// ── AI weight defaults (Stage 5.13) ──────────────────────────────────────────

const WEIGHT_DEFAULTS = [
  // ── Tactical urgency ─────────────────────────────────────────────────
  { key: "close_mill",           group: "Tactical",   label: "Mill closure urgency",        def: 500, min: 100, max: 1000, step: 25,
    tip: "Bonus when the AI closes one of its own mills this move" },
  { key: "cycling_mill",         group: "Tactical",   label: "Cycling mill setup",          def: 300, min: 50,  max: 800,  step: 25,
    tip: "Bonus for building a cycling mill: two 2-configs whose empty closing squares are adjacent, so a single pivot piece shuttles between them forcing a capture every two turns. Also rewards disrupting the opponent's cycling setups." },
  { key: "block_opponent_mill",  group: "Tactical",   label: "Block immediate mill threat", def: 400, min: 100, max: 900,  step: 25,
    tip: "Bonus for moves that neutralise an opponent mill the opponent could close next turn" },
  { key: "stop_opponent_mills",  group: "Tactical",   label: "Disrupt opponent 2-configs",  def: 450, min: 100, max: 900,  step: 25,
    tip: "Bonus for breaking up any opponent 2-piece mill setup, even if not immediately closeable" },
  { key: "feeder_diamond",       group: "Tactical",   label: "Feeder diamond creation",     def: 200, min: 50,  max: 600,  step: 25,
    tip: "Bonus for building a diamond/fork structure: four pieces all adjacent to one empty square, forming two simultaneous mill threats. If one anchor is captured, another piece slides in to close the remaining mill." },
  { key: "mill_wrapping",        group: "Tactical",   label: "Mill wrapping",               def: 150, min: 0,   max: 500,  step: 25,
    tip: "Bonus for occupying exit squares around opponent closed mills — wrapping the mill so the opponent's pivot piece has nowhere to slide. High values let the AI accept an opponent mill if it can surround it." },
  { key: "cardinal_block",       group: "Tactical",   label: "Block cardinal mills",        def: 400, min: 0,   max: 500,  step: 25,
    tip: "Bonus for occupying or evicting opponent pieces from cross-node (d-row/column) squares" },
  { key: "scatter_placement",    group: "Tactical",   label: "Early spread placement",      def: 100, min: 0,   max: 500,  step: 25,
    tip: "Bonus for placing pieces not adjacent to existing own pieces in the first 6 placements" },
  // ── Positional base weights ───────────────────────────────────────────
  { key: "long_term_position",   group: "Positional", label: "Positional weight %",         def: 100, min: 10,  max: 200,  step: 5,
    tip: "Overall multiplier on non-tactical positional scoring (100 = normal)" },
  { key: "mill_count_scale",     group: "Positional", label: "Mill count weight %",         def: 100, min: 0,   max: 300,  step: 5,
    tip: "Scales how much each closed mill contributes to the static evaluation" },
  { key: "mobility_scale",       group: "Positional", label: "Mobility weight %",           def: 100, min: 0,   max: 400,  step: 5,
    tip: "Scales how much having more legal moves than the opponent is valued" },
  { key: "blocked_scale",        group: "Positional", label: "Blocked pieces weight %",     def: 100, min: 0,   max: 500,  step: 5,
    tip: "Scales the bonus for having opponent pieces with no legal moves" },
  // ── Behaviour ─────────────────────────────────────────────────────────
  { key: "make_mistakes",        group: "Behaviour",  label: "Make mistakes %",             def: 0,   min: 0,   max: 100,  step: 5,
    tip: "Probability (%) of playing a deliberately bad move each turn" },
  { key: "opening_adherence",    group: "Behaviour",  label: "Opening book adherence %",    def: 50,  min: 0,   max: 100,  step: 5,
    tip: "How strongly the AI follows its chosen opening line. 0 = ignores the book entirely; 100 = always prefers the book destination over tactical moves." },
];

// ── Personality presets ───────────────────────────────────────────────────────

const PERSONALITIES = [
  { value: "balanced",   label: "Balanced"                      },
  { value: "aggressive", label: "Aggressive — The Crusher"      },
  { value: "defensive",  label: "Defensive — The Blocker"       },
  { value: "positional", label: "Positional — The Strategist"   },
  { value: "scholar",    label: "Scholar — The Bookworm"        },
  { value: "chaos",      label: "Chaos — The Trickster"         },
];

const PERSONALITY_PRESETS = {
  balanced: {
    close_mill: 500, cycling_mill: 50, block_opponent_mill: 400,
    stop_opponent_mills: 450, feeder_diamond: 200, mill_wrapping: 150,
    cardinal_block: 400, scatter_placement: 100, long_term_position: 100,
    mill_count_scale: 100, mobility_scale: 100, blocked_scale: 100,
    make_mistakes: 0, opening_adherence: 30,
  },
  // Hunts mills relentlessly; ignores cycling in favour of immediate mill closure.
  aggressive: {
    close_mill: 900, cycling_mill: 75, block_opponent_mill: 150,
    stop_opponent_mills: 150, feeder_diamond: 350, mill_wrapping: 50,
    cardinal_block: 500, scatter_placement: 25, long_term_position: 70,
    mill_count_scale: 180, mobility_scale: 50, blocked_scale: 80,
    make_mistakes: 0, opening_adherence: 15,
  },
  // Smothers every opponent threat; wraps opponent mills; builds resilient diamond setups.
  defensive: {
    close_mill: 300, cycling_mill: 25, block_opponent_mill: 850,
    stop_opponent_mills: 800, feeder_diamond: 350, mill_wrapping: 350,
    cardinal_block: 275, scatter_placement: 100, long_term_position: 150,
    mill_count_scale: 75, mobility_scale: 200, blocked_scale: 250,
    make_mistakes: 0, opening_adherence: 25,
  },
  // Spreads out, controls cross nodes, builds long-term structures.
  positional: {
    close_mill: 400, cycling_mill: 60, block_opponent_mill: 350,
    stop_opponent_mills: 350, feeder_diamond: 300, mill_wrapping: 250,
    cardinal_block: 500, scatter_placement: 450, long_term_position: 200,
    mill_count_scale: 80, mobility_scale: 300, blocked_scale: 150,
    make_mistakes: 0, opening_adherence: 40,
  },
  // Methodical opening, solid diamond structures, balanced wrapping awareness.
  scholar: {
    close_mill: 450, cycling_mill: 50, block_opponent_mill: 400,
    stop_opponent_mills: 400, feeder_diamond: 250, mill_wrapping: 200,
    cardinal_block: 450, scatter_placement: 400, long_term_position: 175,
    mill_count_scale: 100, mobility_scale: 200, blocked_scale: 125,
    make_mistakes: 0, opening_adherence: 50,
  },
  // Scatters pieces randomly, ignores strategy, makes frequent blunders.
  chaos: {
    close_mill: 150, cycling_mill: 25, block_opponent_mill: 150,
    stop_opponent_mills: 150, feeder_diamond: 75, mill_wrapping: 25,
    cardinal_block: 0, scatter_placement: 500, long_term_position: 10,
    mill_count_scale: 50, mobility_scale: 50, blocked_scale: 50,
    make_mistakes: 45, opening_adherence: 0,
  },
};

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  board = new Board($("board-svg"), onNodeClick);

  // Render AI weight sliders then load saved weights for the active personality.
  _buildWeightSliders();
  fetch("/api/weights").then(r => r.json()).then(saved => {
    const personality = (saved && _matchPersonality(saved)) ?? "balanced";
    _loadPersonality(personality);
  }).catch(() => _loadPersonality("balanced"));

  $("btn-reset-weights").addEventListener("click", () => {
    const ps = $("sel-personality");
    const name = ps?.value ?? "balanced";
    _applyPersonality(name !== "custom" ? name : "balanced");
    if (ps && name === "custom") ps.value = "balanced";
  });
  $("btn-save-weights").addEventListener("click", () => {
    const ps    = $("sel-personality");
    const name  = ps?.value ?? "custom";
    const body  = _getWeights();
    fetch(`/api/personalities/${name}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    }).then(r => r.json()).then(() => {
      addCommentary("Game", `Settings saved for "${name}" — applied from next new game.`);
    }).catch(() => addCommentary("Error", "Could not save settings."));
  });

  $("btn-new-game").addEventListener("click", startNewGame);
  $("toggle-settings").addEventListener("click", () => {
    const p = $("settings-panel");
    p.hidden = !p.hidden;
    $("toggle-settings").classList.toggle("btn-active", !p.hidden);
  });
  $("toggle-ai-tuning").addEventListener("click", () => {
    const p = $("ai-tuning-panel");
    p.hidden = !p.hidden;
    $("toggle-ai-tuning").classList.toggle("btn-active", !p.hidden);
  });
  $("toggle-moves").addEventListener("click", () => {
    const p = $("moves-panel");
    p.hidden = !p.hidden;
    $("toggle-moves").classList.toggle("btn-active", !p.hidden);
  });
  $("btn-undo").addEventListener("click", () => {
    if (!ws || phase === "idle") return;
    ws.send(JSON.stringify({ type: "undo" }));
  });
  $("btn-hint").addEventListener("click", () => {
    if (!ws || phase === "idle" || phase === "game_over") return;
    if (!gameState || !gameState.is_human_turn || hintsLeft <= 0) return;
    $("btn-hint").disabled = true;
    ws.send(JSON.stringify({ type: "hint_request" }));
  });
  $("btn-draw").addEventListener("click", () => {
    if (!ws || !drawUnlocked || phase !== "playing") return;
    $("btn-draw").disabled = true;
    ws.send(JSON.stringify({ type: "draw_offer" }));
  });
  $("btn-force-cap").addEventListener("click", () => {
    if (!ws || phase === "idle") return;
    forceAggressive = !forceAggressive;
    $("btn-force-cap").classList.toggle("btn-active", forceAggressive);
    ws.send(JSON.stringify({ type: "force_aggressive", active: forceAggressive }));
    addCommentary("Game", forceAggressive
      ? "Force Capture ON — AI will capture aggressively even in 4v4."
      : "Force Capture OFF — AI returns to fly-sacrifice strategy."
    );
  });
  $("btn-force-move").addEventListener("click", () => {
    if (!ws) return;
    ws.send(JSON.stringify({ type: "force_move" }));
    stopThinkingTimer();
    $("btn-force-move").hidden = true;
  });
  $("btn-bad-move").addEventListener("click", () => {
    if (!ws || !canMarkBad) return;
    canMarkBad = false;
    $("btn-bad-move").hidden = true;
    ws.send(JSON.stringify({ type: "bad_move" }));
  });
  $("player-chat-send").addEventListener("click", sendPlayerMessage);
  $("player-chat-input").addEventListener("keydown", e => {
    if (e.key === "Enter") sendPlayerMessage();
  });

  $("settings-panel").hidden  = false;
  $("ai-tuning-panel").hidden = true;
  renderIdle();
});

function renderIdle() {
  setStatus("Configure a game and click New Game.");
  setTurnBadge(null, null);
}

// ── New game ──────────────────────────────────────────────────────────────────

function startNewGame() {
  const hc     = $("sel-human-color").value;
  const diff   = parseInt($("sel-difficulty").value);
  const vs     = $("sel-opponent").value === "human";
  const useLlm = $("chk-llm").checked;

  clearCommentary();
  setStatus("Starting…");
  phase = "idle";
  evalHistory = [];
  hintsLeft = 3;
  drawUnlocked = false;
  forceAggressive = false;
  $("btn-force-cap").classList.remove("btn-active");
  $("btn-force-cap").disabled = true;
  drawEvalGraph();
  renderMoves([]);
  $("btn-undo").disabled = true;
  $("btn-force-move").hidden = true;
  $("btn-bad-move").hidden = true;
  canMarkBad = false;
  stopThinkingTimer();
  updateHintButton();
  updateDrawButton();

  if (ws) { ws.close(); ws = null; }

  const wsUrl = `ws://${location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type:       "new_game",
      human_color: hc,
      difficulty:  diff,
      vs_human:    vs,
      use_llm:     useLlm,
      ai_weights:  _getWeights(),
    }));
    $("settings-panel").hidden = true;
  };

  ws.onmessage = evt => handleMessage(JSON.parse(evt.data));
  ws.onerror   = () => setStatus("Connection error.");
  ws.onclose   = () => {
    if (phase !== "game_over") setStatus("Disconnected.");
  };
}

// ── Message handling ──────────────────────────────────────────────────────────

function handleMessage(msg) {
  switch (msg.type) {

    case "state":
      gameState = msg;
      phase = msg.finished ? "game_over" : "playing";
      stopThinkingTimer();
      $("btn-force-move").hidden = true;
      board.render(msg);
      if (msg.move_pairs) board.setMovePairs(msg.move_pairs);
      updateInfoPanel(msg);
      if (msg.eval_score !== undefined) {
        evalHistory.push(msg.eval_score);
        drawEvalGraph();
      }
      if (msg.moves) renderMoves(msg.moves);
      $("btn-undo").disabled = (phase === "idle" || phase === "game_over");
      updateHintButton(msg.is_human_turn && phase !== "game_over");
      if ((msg.post_placement_moves ?? 0) >= 40) drawUnlocked = true;
      updateDrawButton();
      $("btn-force-cap").disabled = (phase === "idle" || phase === "game_over");
      $("btn-bad-move").hidden = !canMarkBad;
      if (msg.is_human_turn) {
        setStatus(
          msg.phase === "place"
            ? "Your turn — click a green node to place."
            : "Your turn — select a piece, then its destination."
        );
      }
      break;

    case "capture_required":
      phase = "capture";
      board.isHuman = true;   // ensure _drawHints() draws capture rings
      if (msg.projected_board) board.grid = msg.projected_board;
      board._drawPieces();
      board.enterCapture(msg.legal_captures);
      setStatus("Mill! Click an opponent piece to capture.");
      break;

    case "thinking":
      startThinkingTimer(msg.color, msg.expected_seconds ?? 0, ws);
      $("btn-force-move").hidden = false;
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      break;

    case "ai_move": {
      const from    = msg.from ? msg.from : "—";
      const to      = msg.to;
      const cap     = msg.capture ? ` × ${msg.capture}` : "";
      const blunder = msg.was_blunder ? " ← deliberate mistake!" : "";
      addCommentary("GameAI", `Played ${from === "—" ? to : from + "→" + to}${cap}${blunder}`);
      if (msg.can_mark_bad) {
        canMarkBad = true;
        $("btn-bad-move").hidden = false;
      }
      break;
    }

    case "bad_move_ack":
      addCommentary("[Training]", `"${msg.bad_notation}" marked bad — AI retrying.`);
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      break;

    case "commentary":
      addCommentary("MillsAI", msg.text);
      break;

    case "hint":
      board.showHint(msg.from, msg.to);
      hintsLeft = msg.hints_left;
      updateHintButton(true);
      if (msg.explanation) {
        addCommentary("[Hint]", msg.explanation);
      } else {
        const dest = msg.from ? `${msg.from} → ${msg.to}` : msg.to;
        addCommentary("[Hint]", `Suggested move: ${dest}`);
      }
      break;

    case "game_over": {
      phase = "game_over";
      stopThinkingTimer();
      $("btn-force-move").hidden = true;
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      const isResign = msg.result === "ai_resignation";
      const statusText = isResign
        ? `${msg.winner === "W" ? "White" : "Black"} wins — AI resigns!`
        : msg.message;
      setStatus(statusText);
      setTurnBadge(null, msg.winner);
      addCommentary("Game", msg.message);
      $("btn-undo").disabled = true;
      $("btn-force-cap").disabled = true;
      updateHintButton(false);
      updateDrawButton();
      break;
    }

    case "draw_accepted":
      addCommentary("Game", "Draw offer accepted.");
      break;

    case "draw_rejected":
      addCommentary("Game", "Draw offer declined — the AI believes it can win.");
      updateDrawButton();
      break;

    case "error":
      addCommentary("Error", msg.message);
      break;
  }
}

// ── Click handling ────────────────────────────────────────────────────────────

function onNodeClick(name) {
  if (!ws || phase === "idle" || phase === "game_over" || !gameState) return;
  if (!gameState.is_human_turn) return;

  if (phase === "capture") {
    canMarkBad = false;
    $("btn-bad-move").hidden = true;
    ws.send(JSON.stringify({ type: "capture", position: name }));
    return;
  }

  if (gameState.phase === "place") {
    if (gameState.legal_dests.includes(name) && !gameState.board[name]) {
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      // Optimistic render: show piece immediately before server confirms.
      board.grid = { ...gameState.board, [name]: gameState.turn };
      board.legalDests = new Set();
      board.legalSrcs  = new Set();
      board._drawPieces();
      board._drawHints();
      ws.send(JSON.stringify({ type: "move", from: null, to: name }));
    }
    return;
  }

  // Movement phase
  if (!board.selected) {
    if (gameState.legal_sources.includes(name) &&
        gameState.board[name] === gameState.turn) {
      board.selectSource(name);
    }
  } else {
    const src = board.selected;
    if (name === src) {
      board.selected = null;
      board._drawPieces();
      board._drawHints();
      return;
    }
    const pairs = board._movePairs || [];
    const valid = pairs.some(([f, t]) => f === src && t === name);
    if (valid) {
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      // Optimistic render: move piece to destination immediately.
      const newGrid = { ...gameState.board };
      newGrid[name] = newGrid[src];
      delete newGrid[src];
      board.grid     = newGrid;
      board.selected = null;
      board.legalDests = new Set();
      board.legalSrcs  = new Set();
      board._drawPieces();
      board._drawHints();
      ws.send(JSON.stringify({ type: "move", from: src, to: name }));
    } else if (gameState.legal_sources.includes(name) &&
               gameState.board[name] === gameState.turn) {
      board.selectSource(name);
    }
  }
}

// ── Player chat ───────────────────────────────────────────────────────────────

function sendPlayerMessage() {
  const input = $("player-chat-input");
  const text  = input.value.trim();
  if (!text || !ws || phase === "idle") return;
  ws.send(JSON.stringify({ type: "player_message", text }));
  input.value = "";
}

// ── UI helpers ────────────────────────────────────────────────────────────────

function updateInfoPanel(state) {
  const color = state.turn;
  const name  = color === "W" ? "White" : "Black";
  setTurnBadge(name, null);

  $("info-phase").textContent    = state.phase;
  $("info-w-placed").textContent = state.pieces_placed?.W ?? 0;
  $("info-b-placed").textContent = state.pieces_placed?.B ?? 0;
  // pieces_captured[W] = pieces White has taken from Black
  $("info-w-taken").textContent  = state.pieces_captured?.W ?? 0;
  $("info-b-taken").textContent  = state.pieces_captured?.B ?? 0;
}

function setStatus(text) {
  $("status-bar").textContent = text;
}

function startThinkingTimer(color, expectedSec, socket) {
  stopThinkingTimer();
  thinkingStarted  = Date.now();
  thinkingExpected = expectedSec;
  const colorName  = color === "W" ? "White" : "Black";
  let autoFired    = false;

  function tick() {
    if (!thinkingInterval) return;
    const elapsed    = (Date.now() - thinkingStarted) / 1000;
    const remaining  = Math.max(0, expectedSec - elapsed);
    if (remaining > 0) {
      setStatus(`AI (${colorName}) thinking… ${remaining.toFixed(1)}s`);
    } else {
      setStatus(`AI (${colorName}) finalizing…`);
      if (!autoFired && socket && socket.readyState === WebSocket.OPEN) {
        autoFired = true;
        socket.send(JSON.stringify({ type: "force_move" }));
      }
    }
  }
  tick();
  thinkingInterval = setInterval(tick, 200);
}

function stopThinkingTimer() {
  if (thinkingInterval !== null) {
    clearInterval(thinkingInterval);
    thinkingInterval = null;
  }
}

function updateDrawButton() {
  const btn = $("btn-draw");
  btn.disabled = !drawUnlocked || phase !== "playing";
}

function updateHintButton(isHumanTurn = false) {
  const btn = $("btn-hint");
  if (hintsLeft <= 0) {
    btn.textContent = "No hints left";
    btn.disabled = true;
  } else {
    btn.textContent = `Hint (${hintsLeft})`;
    btn.disabled = !isHumanTurn || phase === "idle" || phase === "game_over";
  }
}

// ── Eval graph ────────────────────────────────────────────────────────────────

function drawEvalGraph() {
  const svg = $("eval-graph");
  if (!svg) return;

  const W = 800, H = 64;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("width", "100%");
  svg.setAttribute("height", H);
  svg.innerHTML = "";

  const ns = "http://www.w3.org/2000/svg";
  const mk = (tag, attrs) => {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    return el;
  };

  // Background
  svg.appendChild(mk("rect", { x:0, y:0, width:W, height:H, fill:"#1e1a12", rx:4 }));

  // Centre line (equal position)
  svg.appendChild(mk("line", { x1:0, y1:H/2, x2:W, y2:H/2, stroke:"#3d3325", "stroke-width":1 }));

  // Label extremes
  const labelAttrs = { fill:"#5a5040", "font-size":"9", "font-family":"monospace" };
  const tw = mk("text", { ...labelAttrs, x:3, y:10, "dominant-baseline":"hanging" });
  tw.textContent = "White";
  svg.appendChild(tw);
  const tb = mk("text", { ...labelAttrs, x:3, y:H-2 });
  tb.textContent = "Black";
  svg.appendChild(tb);

  const n = evalHistory.length;
  if (n < 2) return;

  const mid    = H / 2;
  const xScale = (W - 2) / Math.max(n - 1, 1);
  const pts    = evalHistory.map((s, i) => ({
    x: 1 + i * xScale,
    y: mid - s * (mid - 4),   // 4px padding from edges
  }));

  // Build area path (filled between line and centre)
  let area = `M ${pts[0].x},${mid}`;
  for (const p of pts) area += ` L ${p.x},${p.y}`;
  area += ` L ${pts[pts.length-1].x},${mid} Z`;

  // Fill: white when White leading (positive), black when Black leading
  // Use gradient-like split: positive fill = white-tan, negative = dark
  const lastScore = evalHistory[n - 1];
  const fillCol   = lastScore > 0.05 ? "rgba(242,237,224,0.18)"
                  : lastScore < -0.05 ? "rgba(30,26,46,0.5)"
                  : "rgba(100,90,70,0.15)";
  svg.appendChild(mk("path", { d: area, fill: fillCol }));

  // Line
  let linePath = `M ${pts[0].x},${pts[0].y}`;
  for (const p of pts.slice(1)) linePath += ` L ${p.x},${p.y}`;
  svg.appendChild(mk("path", { d: linePath, stroke:"#c8a96e", "stroke-width":1.5, fill:"none" }));

  // Current value dot
  const last = pts[n - 1];
  svg.appendChild(mk("circle", { cx: last.x, cy: last.y, r: 3,
    fill: lastScore > 0 ? "#f2ede0" : "#4040a0", stroke:"#c8a96e", "stroke-width":1 }));

  // Score label
  const pct = Math.round(Math.abs(lastScore) * 100);
  const who = lastScore > 0.05 ? `+${pct} W` : lastScore < -0.05 ? `+${pct} B` : "=";
  const lbl = mk("text", { x: Math.min(last.x + 4, W - 32), y: Math.max(last.y - 3, 10),
    fill:"#c8a96e", "font-size":"9", "font-family":"monospace" });
  lbl.textContent = who;
  svg.appendChild(lbl);
}

// ── Moves list ────────────────────────────────────────────────────────────────

function renderMoves(moves) {
  const list = $("moves-list");
  if (!list) return;
  list.innerHTML = "";

  if (!moves || moves.length === 0) return;

  // Pair moves into rows: [white_notation, black_notation]
  const rows = [];
  let i = 0;
  // White always moves first; handle the case where the player is Black
  // and the AI (White) may have already moved once before the first state.
  while (i < moves.length) {
    const w = moves[i].color === "W" ? moves[i] : null;
    const b = moves[i].color === "B" && !w ? moves[i] : null;
    if (w) {
      const bNext = moves[i + 1] && moves[i + 1].color === "B" ? moves[i + 1] : null;
      rows.push([w.notation, bNext ? bNext.notation : ""]);
      i += bNext ? 2 : 1;
    } else {
      rows.push(["—", b ? b.notation : moves[i].notation]);
      i += 1;
    }
  }

  // Header row
  const hdr = document.createElement("div");
  hdr.className = "move-row move-row-hdr";
  hdr.innerHTML = `<span class="move-num">#</span>
    <span class="move-w">⬜</span>
    <span class="move-b">⬛</span>`;
  list.appendChild(hdr);

  rows.forEach((pair, idx) => {
    const row = document.createElement("div");
    row.className = "move-row" + (idx === rows.length - 1 ? " move-row-last" : "");
    const num  = document.createElement("span");
    num.className   = "move-num";
    num.textContent = `${idx + 1}.`;
    const wm = document.createElement("span");
    wm.className   = "move-w";
    wm.textContent = pair[0] || "";
    const bm = document.createElement("span");
    bm.className   = "move-b";
    bm.textContent = pair[1] || "";
    row.appendChild(num);
    row.appendChild(wm);
    row.appendChild(bm);
    list.appendChild(row);
  });

  // Auto-scroll to bottom
  list.scrollTop = list.scrollHeight;
}

function setTurnBadge(name, winner) {
  const el = $("turn-badge");
  if (winner) {
    el.textContent = winner === "W" ? "⬜ White wins" : "⬛ Black wins";
    el.className   = "badge " + (winner === "W" ? "badge-white" : "badge-black");
  } else if (name) {
    el.textContent = name === "White" ? "⬜ White to move" : "⬛ Black to move";
    el.className   = "badge " + (name === "White" ? "badge-white" : "badge-black");
  } else {
    el.textContent = "";
    el.className   = "badge";
  }
}

function addCommentary(speaker, text) {
  if (!text) return;
  const feed  = $("commentary-feed");
  const div   = document.createElement("div");
  div.className = "commentary-line";
  const label = document.createElement("span");
  label.className   = "speaker";
  label.textContent = speaker + ": ";
  div.appendChild(label);
  div.appendChild(document.createTextNode(text));
  feed.appendChild(div);
  feed.scrollTop = feed.scrollHeight;
}

function clearCommentary() {
  $("commentary-feed").innerHTML = "";
}

// ── AI weight sliders (Stage 5.13) ────────────────────────────────────────────

function _buildWeightSliders() {
  const container = $("ai-weight-sliders");
  if (!container) return;
  container.innerHTML = "";

  // ── Personality selector ──────────────────────────────────────────────
  const pRow = document.createElement("div");
  pRow.className = "form-row";
  pRow.style.marginBottom = "10px";

  const pLabel = document.createElement("label");
  pLabel.textContent = "Personality";
  pLabel.htmlFor = "sel-personality";

  const pSelect = document.createElement("select");
  pSelect.id = "sel-personality";

  const customOpt = document.createElement("option");
  customOpt.value = "custom";
  customOpt.textContent = "Custom (saved separately)";
  pSelect.appendChild(customOpt);

  PERSONALITIES.forEach(p => {
    const opt = document.createElement("option");
    opt.value = p.value;
    opt.textContent = p.label;
    pSelect.appendChild(opt);
  });
  pSelect.value = "balanced";

  pSelect.addEventListener("change", () => {
    if (pSelect.value !== "custom") _loadPersonality(pSelect.value);
  });

  pRow.appendChild(pLabel);
  pRow.appendChild(pSelect);
  container.appendChild(pRow);

  // ── Sliders (grouped) ─────────────────────────────────────────────────
  let currentGroup = null;
  WEIGHT_DEFAULTS.forEach(w => {
    if (w.group !== currentGroup) {
      currentGroup = w.group;
      const hdr = document.createElement("div");
      hdr.className = "slider-group-hdr";
      hdr.textContent = w.group;
      container.appendChild(hdr);
    }

    const row = document.createElement("div");
    row.className = "slider-row";

    const labelRow = document.createElement("div");
    labelRow.className = "slider-label";
    labelRow.title = w.tip;

    const name = document.createElement("span");
    name.textContent = w.label;
    name.style.color = "var(--text-dim)";

    const val = document.createElement("span");
    val.id = `slider-val-${w.key}`;
    val.textContent = w.def;

    labelRow.appendChild(name);
    labelRow.appendChild(val);

    const input = document.createElement("input");
    input.type  = "range";
    input.id    = `slider-${w.key}`;
    input.min   = w.min;
    input.max   = w.max;
    input.value = w.def;
    input.step  = w.step ?? 25;
    input.addEventListener("input", () => {
      _updateSliderLabel(w.key, parseInt(input.value));
      // Mark as custom when the user manually drags a slider
      const ps = $("sel-personality");
      if (ps) { ps.value = "custom"; }
    });

    row.appendChild(labelRow);
    row.appendChild(input);
    container.appendChild(row);
  });
}

function _applyPersonality(name) {
  const preset = PERSONALITY_PRESETS[name];
  if (!preset) return;
  WEIGHT_DEFAULTS.forEach(w => {
    const val = preset[w.key] ?? w.def;
    const el  = $(`slider-${w.key}`);
    if (el) { el.value = val; _updateSliderLabel(w.key, val); }
  });
}

function _loadPersonality(name) {
  fetch(`/api/personalities/${name}`)
    .then(r => r.json())
    .then(saved => {
      if (saved && Object.keys(saved).length > 0) {
        WEIGHT_DEFAULTS.forEach(w => {
          if (w.key in saved) {
            const el = $(`slider-${w.key}`);
            if (el) { el.value = saved[w.key]; _updateSliderLabel(w.key, saved[w.key]); }
          }
        });
      } else if (name !== "custom") {
        _applyPersonality(name);
      }
      const ps = $("sel-personality");
      if (ps) ps.value = name;
    })
    .catch(() => { if (name !== "custom") _applyPersonality(name); });
}

function _updateSliderLabel(key, value) {
  const el = $(`slider-val-${key}`);
  if (el) el.textContent = value;
}

function _getWeights() {
  const weights = {};
  WEIGHT_DEFAULTS.forEach(w => {
    const el = $(`slider-${w.key}`);
    weights[w.key] = el ? parseInt(el.value) : w.def;
  });
  return weights;
}

function _matchPersonality(weights) {
  for (const name of Object.keys(PERSONALITY_PRESETS)) {
    const preset = PERSONALITY_PRESETS[name];
    const matches = WEIGHT_DEFAULTS.every(w =>
      (weights[w.key] ?? w.def) === (preset[w.key] ?? w.def)
    );
    if (matches) return name;
  }
  return null;
}
