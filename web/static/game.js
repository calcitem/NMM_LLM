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

// ── Boot ──────────────────────────────────────────────────────────────────────

document.addEventListener("DOMContentLoaded", () => {
  board = new Board($("board-svg"), onNodeClick);

  $("btn-new-game").addEventListener("click", startNewGame);
  $("toggle-settings").addEventListener("click", () => {
    const p = $("settings-panel");
    p.hidden = !p.hidden;
    $("toggle-settings").classList.toggle("btn-active", !p.hidden);
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
  $("player-chat-send").addEventListener("click", sendPlayerMessage);
  $("player-chat-input").addEventListener("keydown", e => {
    if (e.key === "Enter") sendPlayerMessage();
  });

  $("settings-panel").hidden = false;
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
  stopThinkingTimer();
  updateHintButton();
  updateDrawButton();

  if (ws) { ws.close(); ws = null; }

  const wsUrl = `ws://${location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type: "new_game",
      human_color: hc,
      difficulty: diff,
      vs_human: vs,
      use_llm: useLlm,
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
      startThinkingTimer(msg.color, msg.expected_seconds ?? 0);
      $("btn-force-move").hidden = false;
      break;

    case "ai_move": {
      const from    = msg.from ? msg.from : "—";
      const to      = msg.to;
      const cap     = msg.capture ? ` × ${msg.capture}` : "";
      const blunder = msg.was_blunder ? " ← deliberate mistake!" : "";
      addCommentary("GameAI", `Played ${from === "—" ? to : from + "→" + to}${cap}${blunder}`);
      break;
    }

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

    case "game_over":
      phase = "game_over";
      stopThinkingTimer();
      $("btn-force-move").hidden = true;
      setStatus(msg.message);
      setTurnBadge(null, msg.winner);
      addCommentary("Game", msg.message);
      $("btn-undo").disabled = true;
      $("btn-force-cap").disabled = true;
      updateHintButton(false);
      updateDrawButton();
      break;

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
    ws.send(JSON.stringify({ type: "capture", position: name }));
    return;
  }

  if (gameState.phase === "place") {
    if (gameState.legal_dests.includes(name) && !gameState.board[name]) {
      // Optimistic render: show piece immediately before server confirms.
      // Keep isHuman=true so _drawHints() still runs if capture_required follows.
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
      // Optimistic render: move piece to destination immediately.
      // Keep isHuman=true so _drawHints() still runs if capture_required follows.
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

function startThinkingTimer(color, expectedSec) {
  stopThinkingTimer();
  thinkingStarted  = Date.now();
  thinkingExpected = expectedSec;
  const colorName  = color === "W" ? "White" : "Black";
  const maxStr     = expectedSec > 0 ? ` / ~${Math.round(expectedSec)}s` : "";

  function tick() {
    const elapsed = ((Date.now() - thinkingStarted) / 1000).toFixed(1);
    setStatus(`AI (${colorName}) thinking… ${elapsed}s${maxStr}`);
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
