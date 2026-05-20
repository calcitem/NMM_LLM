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
let _hintCountdown    = null; // setInterval handle for hint countdown
let canMarkBad        = false; // true only between ai_move and the next human move commit
let lastAiBadMoveDesc = "";   // move notation shown in the ban confirmation dialog
let canMarkGoodGame   = false; // true after a draw ends (AI vs human)
let isVsHuman         = false; // true when current game is human vs human (handoff buttons visible)
let replayMoves       = [];   // moves with FEN data, populated when game ends
let replayIdx         = -1;   // -1 = not replaying; 0..n-1 = ply index
let _openingsData     = [];   // cached openings from /api/openings
let _currentMoves     = [];   // latest moves array, kept for copyMoveNotation()

// ── Setup mode state ──────────────────────────────────────────────────────────
let setupMode       = false;  // true while the position editor is open
let setupGrid       = {};     // pos → "W"|"B"|"" for the editor board
let setupBrush      = "";     // currently selected palette piece: ""|"W"|"B"
let sessionGames    = 0;      // games finished this session
const QUALIFY_GAMES = 0;      // no qualification required — tournament always available

// ── Player profile state ──────────────────────────────────────────────────────
let playerName = localStorage.getItem("nmm_player_name") || "";

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
  { key: "cardinal_block",       group: "Tactical",   label: "Block cardinal mills",        def: 200, min: 0,   max: 500,  step: 25,
    tip: "Bonus for occupying or evicting opponent pieces from cross-node (d-row/column) squares" },
  { key: "scatter_placement",    group: "Tactical",   label: "Early spread placement",      def: 75,  min: 0,   max: 500,  step: 25,
    tip: "Bonus for placing pieces not adjacent to existing own pieces in the first 6 placements" },
  { key: "setup_mill",          group: "Tactical",   label: "Setup mill bonus",            def: 100, min: 0,   max: 500,  step: 25,
    tip: "Bonus per new two-config (open mill setup) gained this move during placement — rewards building toward future mills" },
  { key: "mill_opening",        group: "Tactical",   label: "Mill opening bonus",          def: 200, min: 0,   max: 600,  step: 25,
    tip: "Bonus for deliberately opening a closed mill when another cycling mill remains — enables recapture next turn" },
  // ── Positional base weights ───────────────────────────────────────────
  { key: "long_term_position",   group: "Positional", label: "Positional weight %",         def: 100, min: 10,  max: 200,  step: 5,
    tip: "Overall multiplier on non-tactical positional scoring (100 = normal)" },
  { key: "mill_count_scale",     group: "Positional", label: "Mill count weight %",         def: 100, min: 0,   max: 300,  step: 5,
    tip: "Scales how much each closed mill contributes to the static evaluation" },
  { key: "mobility_scale",       group: "Positional", label: "Mobility weight %",           def: 100, min: 0,   max: 400,  step: 5,
    tip: "Scales how much having more legal moves than the opponent is valued" },
  { key: "blocked_scale",        group: "Positional", label: "Blocked pieces weight %",     def: 100, min: 0,   max: 500,  step: 5,
    tip: "Scales the bonus for having opponent pieces with no legal moves" },
  // ── Tactical (continued) — Defensive additions ───────────────────────
  { key: "fork_anticipation",    group: "Tactical",   label: "Fork anticipation block",     def: 90,  min: 0,   max: 300,  step: 10,
    tip: "Bonus for blocking squares the opponent could use within 2 moves to create a double mill threat (fork)" },
  { key: "locked_mill_escape",   group: "Tactical",   label: "Locked mill escape",          def: 160, min: 0,   max: 400,  step: 10,
    tip: "Bonus for moving a piece out of a locked mill (all exits blocked by opponent) toward a new 2-config" },
  { key: "redirected_pin",       group: "Tactical",   label: "Redirected pin creation",     def: 140, min: 0,   max: 400,  step: 10,
    tip: "Bonus when a move forces an opponent piece to simultaneously guard two own mill threats (double-pin)" },
  // ── Positional (continued) ─────────────────────────────────────────────
  { key: "defer_for_chain",      group: "Positional", label: "Defer mill for chain bonus",  def: 300, min: 0,   max: 600,  step: 25,
    tip: "Extra bonus (pieces 7-9 only) for skipping an available mill to execute a 4-step forcing sequence ending with a mill" },
  { key: "block_cycling_priority", group: "Positional", label: "Block cycling fork arm",   def: 120, min: 0,   max: 300,  step: 10,
    tip: "Bonus for blocking the fork arm with higher cycling freedom — surrendering the arm the opponent cannot easily exploit" },
  // ── Behaviour ─────────────────────────────────────────────────────────
  { key: "make_mistakes",        group: "Behaviour",  label: "Make mistakes %",             def: 0,   min: 0,   max: 100,  step: 5,
    tip: "Probability (%) of playing a deliberately bad move each turn" },
  { key: "opening_adherence",    group: "Behaviour",  label: "Opening book adherence %",    def: 50,  min: 0,   max: 100,  step: 5,
    tip: "How strongly the AI follows its chosen opening line. 0 = ignores the book entirely; 100 = always prefers the book destination over tactical moves." },
  { key: "loss_exploit",         group: "Behaviour",  label: "Exploit opponent losing lines %", def: 150, min: 0, max: 300, step: 10,
    tip: "How strongly to follow game lines where the opponent historically loses. 150 = 1.5× weight on opponent-loss trajectory hints." },
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
    cardinal_block: 200, scatter_placement: 75, setup_mill: 100, mill_opening: 200,
    long_term_position: 100, mill_count_scale: 100, mobility_scale: 100, blocked_scale: 100,
    fork_anticipation: 90, locked_mill_escape: 160, redirected_pin: 140,
    defer_for_chain: 300, block_cycling_priority: 120,
    make_mistakes: 0, opening_adherence: 30, loss_exploit: 150,
  },
  // Hunts mills relentlessly; ignores cycling in favour of immediate mill closure.
  aggressive: {
    close_mill: 900, cycling_mill: 75, block_opponent_mill: 150,
    stop_opponent_mills: 150, feeder_diamond: 350, mill_wrapping: 50,
    cardinal_block: 300, scatter_placement: 25, setup_mill: 200, mill_opening: 350,
    long_term_position: 70, mill_count_scale: 180, mobility_scale: 50, blocked_scale: 80,
    fork_anticipation: 50, locked_mill_escape: 100, redirected_pin: 80,
    defer_for_chain: 200, block_cycling_priority: 60,
    make_mistakes: 0, opening_adherence: 15, loss_exploit: 200,
  },
  // Smothers every opponent threat; wraps opponent mills; builds resilient diamond setups.
  defensive: {
    close_mill: 300, cycling_mill: 25, block_opponent_mill: 850,
    stop_opponent_mills: 800, feeder_diamond: 350, mill_wrapping: 350,
    cardinal_block: 150, scatter_placement: 75, setup_mill: 100, mill_opening: 100,
    long_term_position: 150, mill_count_scale: 75, mobility_scale: 200, blocked_scale: 250,
    fork_anticipation: 150, locked_mill_escape: 220, redirected_pin: 200,
    defer_for_chain: 350, block_cycling_priority: 200,
    make_mistakes: 0, opening_adherence: 25, loss_exploit: 100,
  },
  // Spreads out, controls cross nodes, builds long-term structures.
  positional: {
    close_mill: 400, cycling_mill: 60, block_opponent_mill: 350,
    stop_opponent_mills: 350, feeder_diamond: 300, mill_wrapping: 250,
    cardinal_block: 400, scatter_placement: 350, setup_mill: 175, mill_opening: 175,
    long_term_position: 200, mill_count_scale: 80, mobility_scale: 300, blocked_scale: 150,
    fork_anticipation: 120, locked_mill_escape: 180, redirected_pin: 160,
    defer_for_chain: 320, block_cycling_priority: 180,
    make_mistakes: 0, opening_adherence: 40, loss_exploit: 180,
  },
  // Methodical opening, solid diamond structures, balanced wrapping awareness.
  scholar: {
    close_mill: 450, cycling_mill: 50, block_opponent_mill: 400,
    stop_opponent_mills: 400, feeder_diamond: 250, mill_wrapping: 200,
    cardinal_block: 300, scatter_placement: 300, setup_mill: 150, mill_opening: 225,
    long_term_position: 175, mill_count_scale: 100, mobility_scale: 200, blocked_scale: 125,
    fork_anticipation: 100, locked_mill_escape: 160, redirected_pin: 150,
    defer_for_chain: 300, block_cycling_priority: 140,
    make_mistakes: 0, opening_adherence: 50, loss_exploit: 180,
  },
  // Scatters pieces randomly, ignores strategy, makes frequent blunders.
  chaos: {
    close_mill: 150, cycling_mill: 25, block_opponent_mill: 150,
    stop_opponent_mills: 150, feeder_diamond: 75, mill_wrapping: 25,
    cardinal_block: 0, scatter_placement: 500, setup_mill: 50, mill_opening: 75,
    long_term_position: 10, mill_count_scale: 50, mobility_scale: 50, blocked_scale: 50,
    fork_anticipation: 20, locked_mill_escape: 50, redirected_pin: 30,
    defer_for_chain: 100, block_cycling_priority: 30,
    make_mistakes: 45, opening_adherence: 0, loss_exploit: 50,
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
      addCommentary("Game", `Settings saved for "${name}" — applied from next new game.`, "ai");
    }).catch(() => addCommentary("Error", "Could not save settings.", "ai"));
  });

  // Show/hide personality row based on opponent type
  function _updatePersonalityRow() {
    const row = $("row-personality");
    if (row) row.hidden = $("sel-opponent").value === "human";
  }
  $("sel-opponent").addEventListener("change", _updatePersonalityRow);
  _updatePersonalityRow();

  // Bidirectional sync: header personality picker ↔ settings panel picker
  const _hdrP = $("hdr-personality");
  const _sidP = $("sel-game-personality");
  if (_hdrP && _sidP) {
    _hdrP.value = _sidP.value;
    _hdrP.addEventListener("change", () => { _sidP.value = _hdrP.value; });
    _sidP.addEventListener("change", () => { _hdrP.value = _sidP.value; });
  }

  $("btn-new-game").addEventListener("click", startNewGame);

  // Setup position controls
  $("btn-setup-toggle").addEventListener("click", enterSetupMode);
  $("btn-setup-cancel").addEventListener("click", exitSetupMode);
  $("btn-setup-clear").addEventListener("click", () => {
    setupGrid = {};
    _renderSetupBoard();
    _updateSetupUI();
  });
  $("btn-setup-start").addEventListener("click", startSetupGame);
  document.querySelectorAll(".setup-swatch").forEach(btn => {
    btn.addEventListener("click", () => {
      setupBrush = btn.dataset.piece;
      document.querySelectorAll(".setup-swatch").forEach(b =>
        b.classList.toggle("setup-swatch-active", b === btn));
    });
  });

  $("sel-setup-phase").addEventListener("change", _updateSetupUI);
  $("sel-setup-turn").addEventListener("change", _updateSetupUI);

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
  $("toggle-openings").addEventListener("click", () => {
    const p = $("openings-panel");
    p.hidden = !p.hidden;
    $("toggle-openings").classList.toggle("btn-active", !p.hidden);
  });
  $("rng-replay-speed").addEventListener("input", () => {
    const ms = parseInt($("rng-replay-speed").value);
    $("lbl-replay-speed").textContent = (ms / 1000).toFixed(1) + "s";
  });
  $("btn-replay-opening").addEventListener("click", startReplayOpening);
  $("sel-opening").addEventListener("change", _showOpeningInfo);
  $("btn-undo").addEventListener("click", () => {
    if (!ws || phase === "idle") return;
    ws.send(JSON.stringify({ type: "undo" }));
  });
  $("copy-moves-btn").addEventListener("click", copyMoveNotation);
  $("btn-hint").addEventListener("click", () => {
    if (!ws || phase === "idle" || phase === "game_over") return;
    if (!gameState || !gameState.is_human_turn || hintsLeft <= 0) return;
    ws.send(JSON.stringify({ type: "hint_request" }));
    startHintCountdown();
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
      : "Force Capture OFF — AI returns to fly-sacrifice strategy.",
    "ai");
  });
  $("btn-force-move").addEventListener("click", () => {
    if (!ws) return;
    ws.send(JSON.stringify({ type: "force_move" }));
    stopThinkingTimer();
    $("btn-force-move").hidden = true;
  });
  $("btn-bad-move").addEventListener("click", () => {
    if (!ws || !canMarkBad) return;
    const desc = lastAiBadMoveDesc ? `"${lastAiBadMoveDesc}"` : "this move";
    const confirmed = window.confirm(
      `Ban ${desc} from this position?\n\n` +
      `The AI will avoid it for the rest of this game and in future games ` +
      `with the same move sequence. This is saved to disk and cannot be easily undone.`
    );
    if (!confirmed) return;
    canMarkBad = false;
    $("btn-bad-move").hidden = true;
    ws.send(JSON.stringify({ type: "bad_move" }));
  });
  $("btn-good-game").addEventListener("click", () => {
    if (!ws || !canMarkGoodGame) return;
    canMarkGoodGame = false;
    $("btn-good-game").hidden = true;
    ws.send(JSON.stringify({ type: "good_game" }));
  });
  $("player-chat-send").addEventListener("click", sendPlayerMessage);
  $("player-chat-input").addEventListener("keydown", e => {
    if (e.key === "Enter") sendPlayerMessage();
  });

  // Replay controls
  $("btn-replay-first").addEventListener("click", () => replayGo(0));
  $("btn-replay-prev").addEventListener("click",  () => replayGo(replayIdx - 1));
  $("btn-replay-next").addEventListener("click",  () => replayGo(replayIdx + 1));
  $("btn-replay-last").addEventListener("click",  () => replayGo(replayMoves.length));
  $("btn-replay-live").addEventListener("click",  exitReplay);

  $("settings-panel").hidden  = false;
  $("ai-tuning-panel").hidden = true;
  $("moves-panel").hidden     = false;   // show moves by default
  _setReplayButtonsDisabled(true);
  renderIdle();
  _loadOpenings();

  // ── Left column tab toggle ────────────────────────────────────────────
  $("tab-chat").addEventListener("click", () => _switchLeftTab("chat"));
  // Clicking the active profile tab closes it (returns to chat).
  $("tab-profile").addEventListener("click", () =>
    _switchLeftTab($("profile-view").hidden ? "profile" : "chat"));

  // ── Player profile ────────────────────────────────────────────────────
  if (playerName) {
    $("player-name-input").value = playerName;
    _fetchAndRenderProfile(playerName);
  }
  $("btn-save-profile").addEventListener("click", () => {
    const name = $("player-name-input").value.trim();
    if (!name) return;
    playerName = name;
    localStorage.setItem("nmm_player_name", name);
    $("profile-empty-msg").hidden = true;
    _fetchAndRenderProfile(name);
    addCommentary("Game", `Profile saved for "${name}".`, "ai");
  });

  // Header "New Game" button mirrors sidebar button
  $("btn-new-game-header").addEventListener("click", () => $("btn-new-game").click());

  // Header "Setup" toggle
  $("toggle-setup").addEventListener("click", () => {
    if (setupMode) exitSetupMode();
    else enterSetupMode();
  });

  // Header "Tournament" toggle
  $("toggle-tournament").addEventListener("click", () => {
    const p = $("tournament-panel");
    p.hidden = !p.hidden;
    $("toggle-tournament").classList.toggle("btn-active", !p.hidden);
  });
  $("btn-tournament-start").addEventListener("click", () => {
    if (ws) ws.send(JSON.stringify({ type: "tournament_start" }));
  });
  $("btn-tournament-restart").addEventListener("click", () => {
    $("tournament-complete-info").hidden = true;
    $("tournament-rows").innerHTML = "";
    $("tournament-active").hidden = true;
    $("btn-tournament-start").hidden = false;
    $("tournament-intro").hidden = false;
    if (ws) ws.send(JSON.stringify({ type: "tournament_start" }));
  });
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

  // Apply personality for this game (overrides current sliders unless "current")
  const gamePSelect = $("sel-game-personality");
  if (gamePSelect && gamePSelect.value !== "current") {
    let chosenPersonality = gamePSelect.value;
    if (chosenPersonality === "random") {
      const opts = PERSONALITIES.map(p => p.value);
      chosenPersonality = opts[Math.floor(Math.random() * opts.length)];
    }
    _loadPersonality(chosenPersonality);
  }

  clearCommentary();
  setStatus("Starting…");
  phase = "idle";
  evalHistory = [];
  hintsLeft = 3;
  drawUnlocked = false;
  forceAggressive = false;
  replayMoves = [];
  replayIdx   = -1;
  _updateReplayLabel();
  _setReplayButtonsDisabled(true);
  $("btn-force-cap").classList.remove("btn-active");
  $("btn-force-cap").disabled = true;
  drawEvalGraph();
  renderMoves([]);
  $("btn-undo").disabled = true;
  $("btn-force-move").hidden = true;
  $("btn-bad-move").hidden = true;
  canMarkBad = false;
  $("btn-good-game").hidden = true;
  canMarkGoodGame = false;
  stopThinkingTimer();
  updateHintButton();
  updateDrawButton();

  isVsHuman = vs;
  _updateHandoffButtons();

  if (ws) { ws.close(); ws = null; }

  const wsUrl = `ws://${location.host}/ws`;
  ws = new WebSocket(wsUrl);

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type:        "new_game",
      human_color:  hc,
      difficulty:   diff,
      vs_human:     vs,
      use_llm:      useLlm,
      ai_weights:   _getWeights(),
      player_name:  playerName,
    }));
    $("settings-panel").hidden = true;
  };

  ws.onmessage = evt => handleMessage(JSON.parse(evt.data));
  ws.onerror   = () => setStatus("Connection error.");
  ws.onclose   = () => {
    if (phase !== "game_over") setStatus("Disconnected.");
  };
}

// ── Handoff to AI ─────────────────────────────────────────────────────────────

function _updateHandoffButtons() {
  const active = isVsHuman && phase === "playing";
  $("btn-handoff-w").hidden = !active;
  $("btn-handoff-b").hidden = !active;
}

function _handoffToAI(color) {
  if (!ws || phase !== "playing") return;
  const diff   = parseInt($("sel-difficulty").value);
  const useLlm = $("chk-llm").checked;
  ws.send(JSON.stringify({
    type:       "handoff_to_ai",
    color,
    difficulty: diff,
    use_llm:    useLlm,
    ai_weights: _getWeights(),
  }));
}

$("btn-handoff-w").addEventListener("click", () => _handoffToAI("W"));
$("btn-handoff-b").addEventListener("click", () => _handoffToAI("B"));

// ── Position setup ────────────────────────────────────────────────────────────

const ALL_POSITIONS = [
  "a7","d7","g7","g4","g1","d1","a1","a4",
  "b6","d6","f6","f4","f2","d2","b2","b4",
  "c5","d5","e5","e4","e3","d3","c3","c4",
];

function enterSetupMode() {
  setupMode = true;
  setupGrid = {};
  // Seed from current live board if a game is in progress
  if (gameState && gameState.board) {
    for (const [pos, v] of Object.entries(gameState.board)) {
      if (v) setupGrid[pos] = v;
    }
  }
  // Inherit human colour from the main settings selector
  const mainHc = $("sel-human-color").value;
  const setupHc = $("sel-setup-human-color");
  if (setupHc && mainHc !== "R") setupHc.value = mainHc;

  setupBrush = "";  // default: eraser
  document.querySelectorAll(".setup-swatch").forEach(b =>
    b.classList.toggle("setup-swatch-active", b.dataset.piece === ""));

  $("settings-panel").hidden = true;
  $("setup-panel").hidden    = false;
  $("toggle-settings").classList.remove("btn-active");
  $("toggle-setup").classList.add("btn-active");

  _renderSetupBoard();
  _updateSetupUI();
  setStatus("Setup mode — click nodes to place pieces.");
}

function exitSetupMode() {
  setupMode = false;
  $("setup-panel").hidden = true;
  $("toggle-setup").classList.remove("btn-active");
  // Restore piece layer click interception (disabled during setup for erase to work)
  board._pieceGroup.setAttribute("pointer-events", "");
  // Restore live board if game is running
  if (gameState) {
    board.render(gameState);
    if (gameState.move_pairs) board.setMovePairs(gameState.move_pairs);
    setStatus(phase === "playing" ? "Setup cancelled — game continues." : "");
  } else {
    renderIdle();
  }
}

function _renderSetupBoard() {
  const grid = {};
  for (const pos of ALL_POSITIONS) grid[pos] = setupGrid[pos] || null;
  board.grid        = grid;
  board.legalDests  = new Set();
  board.legalSrcs   = new Set();
  board.selected    = null;
  board._millNodes  = new Set();
  board._hintGroup.innerHTML   = "";
  board._hintOverlay.innerHTML = "";
  board._drawPieces();
  // Pieces sit above node circles in the SVG z-order. Setting pointer-events:none
  // on the piece layer lets clicks fall through to the node circles, which carry
  // the click→onNodeClick listener needed for the erase/place brush to work.
  board._pieceGroup.setAttribute("pointer-events", "none");
}

function _setupValidation() {
  const w = ALL_POSITIONS.filter(p => setupGrid[p] === "W").length;
  const b = ALL_POSITIONS.filter(p => setupGrid[p] === "B").length;
  const phase = $("sel-setup-phase").value;
  const errors = [];

  if (w < 1 || b < 1) errors.push("Each side needs at least 1 piece.");
  if (w > 9)          errors.push("White cannot have more than 9 pieces.");
  if (b > 9)          errors.push("Black cannot have more than 9 pieces.");
  if (phase === "move") {
    if (w < 3) errors.push("Movement phase: White needs at least 3 pieces.");
    if (b < 3) errors.push("Movement phase: Black needs at least 3 pieces.");
  }
  return { w, b, errors };
}

function _updateSetupUI() {
  const { w, b, errors } = _setupValidation();
  $("setup-counts").innerHTML = `White: ${w} &nbsp;|&nbsp; Black: ${b}`;
  $("setup-error").textContent = errors[0] || "";
  $("btn-setup-start").disabled = errors.length > 0;
}

function startSetupGame() {
  const { errors } = _setupValidation();
  if (errors.length) return;

  const hc     = $("sel-setup-human-color").value;
  const diff   = parseInt($("sel-difficulty").value);
  const vs     = $("sel-opponent").value === "human";
  const useLlm = $("chk-llm").checked;

  const gamePSelect = $("sel-game-personality");
  if (gamePSelect && gamePSelect.value !== "current") {
    let p = gamePSelect.value;
    if (p === "random") {
      const opts = PERSONALITIES.map(x => x.value);
      p = opts[Math.floor(Math.random() * opts.length)];
    }
    _loadPersonality(p);
  }

  clearCommentary();
  setStatus("Starting setup game…");
  phase = "idle";
  evalHistory = [];
  hintsLeft = 3;
  drawUnlocked = false;
  forceAggressive = false;
  replayMoves = [];
  replayIdx   = -1;
  _updateReplayLabel();
  _setReplayButtonsDisabled(true);
  $("btn-force-cap").classList.remove("btn-active");
  $("btn-force-cap").disabled = true;
  drawEvalGraph();
  renderMoves([]);
  $("btn-undo").disabled = true;
  $("btn-force-move").hidden = true;
  $("btn-bad-move").hidden = true;
  canMarkBad = false;
  $("btn-good-game").hidden = true;
  canMarkGoodGame = false;
  stopThinkingTimer();
  updateHintButton();
  updateDrawButton();

  if (ws) { ws.close(); ws = null; }

  const wsUrl = `ws://${location.host}/ws`;
  ws = new WebSocket(wsUrl);

  const positions = {};
  for (const pos of ALL_POSITIONS) positions[pos] = setupGrid[pos] || "";

  ws.onopen = () => {
    ws.send(JSON.stringify({
      type:        "setup_game",
      human_color:  hc,
      difficulty:   diff,
      vs_human:     vs,
      use_llm:      useLlm,
      ai_weights:   _getWeights(),
      positions:    positions,
      phase:        $("sel-setup-phase").value,
      turn:         $("sel-setup-turn").value,
    }));
    setupMode = false;
    $("setup-panel").hidden = true;
    $("toggle-setup").classList.remove("btn-active");
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
      _updateHandoffButtons();
      if (replayIdx === -1) {
        board.render(msg);
        if (msg.move_pairs) board.setMovePairs(msg.move_pairs);
      }
      updateInfoPanel(msg);
      if (msg.eval_score !== undefined) {
        evalHistory.push(msg.eval_score);
        drawEvalGraph();
      }
      if (msg.moves) {
        renderMoves(msg.moves);
        if (msg.moves.length > 0) {
          replayMoves = msg.moves;
          _setReplayButtonsDisabled(false);
          _updateReplayLabel();
        }
      }
      $("btn-undo").disabled = (phase === "idle" || phase === "game_over");
      if (msg.hints_left !== undefined) hintsLeft = msg.hints_left;
      updateHintButton(msg.is_human_turn && phase !== "game_over");
      if ((msg.post_placement_moves ?? 0) >= 40) drawUnlocked = true;
      updateDrawButton();
      {
        // B-1: Force Capture only makes sense when human has exactly 4 pieces
        // (the AI's fly-sacrifice hesitation only applies at that count).
        const humanColor = msg.human_color;
        const humanPieces = humanColor
          ? Object.values(msg.board || {}).filter(c => c === humanColor).length
          : 0;
        const capDisable = (phase === "idle" || phase === "game_over" || humanPieces !== 4);
        $("btn-force-cap").disabled = capDisable;
        if (capDisable && forceAggressive) {
          forceAggressive = false;
          $("btn-force-cap").classList.remove("btn-active");
          ws.send(JSON.stringify({ type: "force_aggressive", active: false }));
        }
      }
      $("btn-bad-move").hidden = !canMarkBad;
      if (msg.is_human_turn && replayIdx === -1) {
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
      board.selected = null;
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
      addCommentary("GameAI", `Played ${from === "—" ? to : from + "→" + to}${cap}${blunder}`, "ai");
      if (msg.can_mark_bad) {
        canMarkBad = true;
        lastAiBadMoveDesc = msg.from + "→" + msg.to + (msg.capture ? "×" + msg.capture : "");
        $("btn-bad-move").hidden = false;
      }
      break;
    }

    case "bad_move_ack":
      addCommentary("[Training]", `"${msg.bad_notation}" marked bad — AI retrying.`, "ai");
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      break;

    case "good_game_ack":
      addCommentary("[Training]", "Good game noted — AI's moves reinforced as a win in the trajectory.", "ai");
      break;

    case "commentary":
      addCommentary(msg.speaker ?? "MillsAI", msg.text, msg.section);
      break;

    case "hint":
      board.showHint(msg.from, msg.to);
      hintsLeft = msg.hints_left;
      updateHintButton(true);
      if (msg.explanation) {
        addCommentary("[Hint]", msg.explanation, "human");
      } else {
        const dest = msg.from ? `${msg.from} → ${msg.to}` : msg.to;
        addCommentary("[Hint]", `Suggested move: ${dest}`, "human");
      }
      break;

    case "game_over": {
      phase = "game_over";
      stopThinkingTimer();
      $("btn-force-move").hidden = true;
      canMarkBad = false;
      $("btn-bad-move").hidden = true;
      // Show Good Game after a draw in AI vs human (reinforces strong AI play)
      canMarkGoodGame = !msg.winner && !isVsHuman;
      $("btn-good-game").hidden = !canMarkGoodGame;
      isVsHuman = false;
      _updateHandoffButtons();
      const isResign = msg.result === "ai_resignation";
      const statusText = isResign
        ? `${msg.winner === "W" ? "White" : "Black"} wins — AI resigns!`
        : msg.message;
      setStatus(statusText);
      setTurnBadge(null, msg.winner);
      addCommentary("Game", msg.message, "ai");
      $("btn-undo").disabled = true;
      $("btn-force-cap").disabled = true;
      updateHintButton(false);
      updateDrawButton();

      // Adaptive difficulty feedback
      if (msg.adaptive) {
        const ad = msg.adaptive;
        if (ad.action === "softened") {
          addCommentary("Adaptive", `After ${AdaptiveTracker.SOFTEN_AFTER} losses I've dropped to difficulty ${ad.difficulty} and will make more deliberate mistakes. Keep playing — you'll improve!`, "ai");
          setAdaptiveBadge(ad.difficulty, true);
        } else if (ad.action === "restored") {
          addCommentary("Adaptive", `Great improvement! Restoring difficulty to ${ad.difficulty}.`, "ai");
          setAdaptiveBadge(ad.difficulty, false);
        } else if (ad.action === "suggest_harder") {
          addCommentary("Adaptive", `You're on a ${AdaptiveTracker.HARDEN_SUGGEST}-game win streak! Consider trying difficulty ${ad.difficulty} for a tougher challenge.`, "ai");
        }
      }

      // Session games counter — unlocks tournament button
      sessionGames++;
      if (sessionGames >= QUALIFY_GAMES) {
        const tb = $("toggle-tournament");
        tb.disabled = false;
        tb.title = "Open Tournament Mode";
      }
      break;
    }

    case "handoff_ack":
      isVsHuman = false;
      _updateHandoffButtons();
      addCommentary("Game",
        `${msg.ai_color === "W" ? "White" : "Black"} handed to AI — continuing from current position.`,
        "ai");
      break;

    case "draw_accepted":
      addCommentary("Game", "Draw offer accepted.", "ai");
      break;

    case "draw_rejected":
      addCommentary("Game", "Draw offer declined — the AI believes it can win.", "ai");
      updateDrawButton();
      break;

    case "tournament_init":
      _renderTournamentInit(msg);
      break;

    case "tournament_next":
      _handleTournamentNext(msg);
      break;

    case "tournament_update":
      _updateTournamentScoreboard(msg);
      break;

    case "tournament_complete":
      _handleTournamentComplete(msg);
      break;

    case "profile_update":
      _renderProfile(msg);
      break;

    case "library_reload":
      addCommentary("Game",
        `Library updated: ${msg.game_count} games, ${msg.traj_entries} trajectory entries, ` +
        `${msg.endgame_positions} endgame positions.`, "ai");
      break;

    case "error":
      addCommentary("Error", msg.message, "ai");
      break;
  }
}

// ── Click handling ────────────────────────────────────────────────────────────

function onNodeClick(name) {
  // Setup mode: cycle the clicked node through empty→W→B→empty (or place brush)
  if (setupMode) {
    const cur = setupGrid[name] || "";
    if (setupBrush !== undefined && setupBrush !== null) {
      // Palette-driven: erase sets "", W/B sets that colour
      setupGrid[name] = setupBrush;
    } else {
      // Cycle mode (fallback)
      setupGrid[name] = cur === "" ? "W" : cur === "W" ? "B" : "";
    }
    _renderSetupBoard();
    _updateSetupUI();
    return;
  }

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
      setStatus("Move sent — AI calculating…");
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

  const ef = state.early_families;
  if (ef && state.phase === "place") {
    const wFam = ef.white_family || "";
    const bFam = ef.black_family || "";
    $("info-opening-family").textContent = wFam === bFam ? wFam : `W:${wFam} / B:${bFam}`;
    $("opening-family-row").hidden = false;
  } else {
    $("opening-family-row").hidden = true;
  }

  const ad = state.adaptive;
  if (ad && ad.softened) {
    setAdaptiveBadge(ad.difficulty, true);
  } else if (ad && !ad.softened && ad.difficulty === ad.base) {
    setAdaptiveBadge(null, false);
  }
}

function setStatus(text) {
  $("status-bar").textContent = text;
}

// ── Adaptive difficulty badge ─────────────────────────────────────────────────

// Mirror of server constants — used in message text only.
const AdaptiveTracker = { SOFTEN_AFTER: 3, HARDEN_SUGGEST: 3 };

function setAdaptiveBadge(difficulty, softened) {
  let badge = document.getElementById("adaptive-badge");
  if (!badge) {
    badge = document.createElement("span");
    badge.id = "adaptive-badge";
    const sb = $("status-bar");
    sb.parentNode.insertBefore(badge, sb.nextSibling);
  }
  if (difficulty === null) {
    badge.hidden = true;
    return;
  }
  badge.hidden = false;
  badge.className = softened ? "adaptive-badge adaptive-softened" : "adaptive-badge";
  badge.textContent = softened
    ? `Adaptive: Diff ${difficulty} (easing)`
    : `Adaptive: Diff ${difficulty}`;
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

function startHintCountdown() {
  stopHintCountdown();
  const btn = $("btn-hint");
  btn.disabled = true;
  let secs = 0;
  btn.textContent = "Calculating hint…";
  _hintCountdown = setInterval(() => {
    secs++;
    btn.textContent = `Calculating hint… ${secs}s`;
  }, 1000);
}

function stopHintCountdown() {
  if (_hintCountdown) { clearInterval(_hintCountdown); _hintCountdown = null; }
}

function updateHintButton(isHumanTurn = false) {
  stopHintCountdown();
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
  _currentMoves = moves || [];
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

function copyMoveNotation() {
  const moves = _currentMoves;
  if (!moves || moves.length === 0) return;

  // Pair moves into rows the same way renderMoves does
  const rows = [];
  let i = 0;
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

  const text = rows.map((pair, idx) => `${idx + 1}.${pair[0]}${pair[1] ? " " + pair[1] : ""}`).join("\n");

  navigator.clipboard.writeText(text).then(() => {
    const btn = $("copy-moves-btn");
    if (btn) {
      const prev = btn.textContent;
      btn.textContent = "Copied!";
      setTimeout(() => { btn.textContent = prev; }, 1500);
    }
  }).catch(() => _showMoveTextBox(text));
}

function _showMoveTextBox(text) {
  let overlay = document.getElementById("move-copy-overlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "move-copy-overlay";
    overlay.style.cssText = "position:fixed;inset:0;background:rgba(0,0,0,.6);z-index:999;display:flex;align-items:center;justify-content:center";
    overlay.addEventListener("click", e => { if (e.target === overlay) overlay.remove(); });
    document.body.appendChild(overlay);
    const box = document.createElement("div");
    box.style.cssText = "background:#1e1a12;border:1px solid var(--border);border-radius:6px;padding:14px;min-width:260px;max-width:420px;width:90%";
    const lbl = document.createElement("p");
    lbl.style.cssText = "margin:0 0 8px;font-size:.8rem;color:var(--text-dim)";
    lbl.textContent = "Select all and copy (Ctrl+A, Ctrl+C):";
    const ta = document.createElement("textarea");
    ta.style.cssText = "width:100%;height:220px;background:#0e0c08;border:1px solid var(--border);color:var(--text);font-family:monospace;font-size:.78rem;padding:6px;border-radius:4px;box-sizing:border-box;resize:none";
    ta.readOnly = true;
    ta.value = text;
    const close = document.createElement("button");
    close.textContent = "Close";
    close.className = "btn-small";
    close.style.cssText = "margin-top:8px;width:100%";
    close.addEventListener("click", () => overlay.remove());
    box.append(lbl, ta, close);
    overlay.appendChild(box);
    // Auto-select all text
    ta.focus();
    ta.select();
  } else {
    overlay.remove();
  }
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

// ── Move replay ───────────────────────────────────────────────────────────────
// replayIdx: -1 = live; 0 = initial board; k = board after move k.
// move[k].fen is the board BEFORE move k, so:
//   idx=0 → replayMoves[0].fen (position before move 0 = initial board)
//   idx=k → replayMoves[k].fen (position before move k = after move k-1)
//   idx=n → gameState.board (after the last move)

function replayGo(idx) {
  if (!replayMoves.length) return;
  // idx=0 → initial board (before any moves)
  // idx=k → board after move k  (1 ≤ k ≤ replayMoves.length)
  idx = Math.max(0, Math.min(replayMoves.length, idx));
  replayIdx = idx;

  if (idx === 0) {
    // Initial board: FEN stored in move[0] is the board BEFORE that move
    const fen = replayMoves[0].fen;
    if (fen) board.renderFromFen(fen);
  } else if (idx < replayMoves.length) {
    // Board after move idx-1: FEN of move[idx] = board before move idx = after move idx-1
    const fen = replayMoves[idx].fen;
    if (fen) board.renderFromFen(fen);
  } else {
    // After last move: use the live/final board from gameState
    if (gameState) {
      board.grid = Object.assign({}, gameState.board);
      board._millNodes = new Set();
      board._hintGroup.innerHTML  = "";
      board._hintOverlay.innerHTML = "";
      board._drawPieces();
    }
  }

  _updateReplayLabel();
  _highlightReplayMove(idx);
}

function exitReplay() {
  replayIdx = -1;
  if (gameState) {
    board.render(gameState);
    if (gameState.move_pairs) board.setMovePairs(gameState.move_pairs);
  }
  _updateReplayLabel();
  _highlightReplayMove(-1);
}

function _setReplayButtonsDisabled(disabled) {
  ["btn-replay-first","btn-replay-prev","btn-replay-next",
   "btn-replay-last","btn-replay-live"].forEach(id => {
    const el = $(id);
    if (el) el.disabled = disabled;
  });
}

function _updateReplayLabel() {
  const lbl = $("replay-ply-label");
  if (!lbl) return;
  const total = replayMoves.length;
  if (replayIdx === -1) {
    lbl.textContent = total ? `— / ${total}` : "0 / 0";
  } else {
    // idx=0 is the start position; idx=total is after the last move
    lbl.textContent = `${replayIdx} / ${total}`;
  }
}

function _highlightReplayMove(idx) {
  const list = $("moves-list");
  if (!list) return;
  list.querySelectorAll(".move-row").forEach(r => r.classList.remove("move-row-replay"));
  if (idx <= 0) return;  // 0 = initial board, nothing to highlight
  // idx=1 is after move 0 (White's first move); each display row = 2 half-moves
  const rows = list.querySelectorAll(".move-row:not(.move-row-hdr)");
  const rowIdx = Math.floor((idx - 1) / 2);
  if (rows[rowIdx]) {
    rows[rowIdx].classList.add("move-row-replay");
    rows[rowIdx].scrollIntoView({ block: "nearest" });
  }
}

// ── Commentary routing ────────────────────────────────────────────────────────

// Speakers that belong to the AI discussion box (bottom).
// All others go to the human-facing box (top).
const _AI_SPEAKERS = new Set(["GameAI", "Game", "Error", "MillsLLM"]);

function addCommentary(speaker, text, section) {
  if (!text) return;
  // Determine target box: explicit section override, or classify by speaker
  const isAi  = section === "ai"  || (!section && _AI_SPEAKERS.has(speaker));
  const feedId = isAi ? "commentary-ai" : "commentary-human";
  const feed   = $(feedId);
  if (!feed) return;

  const div = document.createElement("div");
  div.className = "commentary-line";
  const label = document.createElement("span");
  label.className   = "speaker";
  label.textContent = speaker + ": ";
  div.appendChild(label);
  div.appendChild(document.createTextNode(text));
  // Prepend so newest appears at top
  feed.insertBefore(div, feed.firstChild);
}

function clearCommentary() {
  const h = $("commentary-human");
  const a = $("commentary-ai");
  if (h) h.innerHTML = "";
  if (a) a.innerHTML = "";
}

// ── Openings panel ────────────────────────────────────────────────────────────

function _loadOpenings() {
  fetch("/api/openings")
    .then(r => r.json())
    .then(openings => {
      _openingsData = openings;
      const sel = $("sel-opening");
      sel.innerHTML = "";
      if (!openings.length) {
        sel.innerHTML = '<option value="">No openings found</option>';
        return;
      }
      // Group by family
      const families = {};
      for (const op of openings) {
        const fam = op.family || "Other";
        if (!families[fam]) families[fam] = [];
        families[fam].push(op);
      }
      for (const [fam, ops] of Object.entries(families).sort()) {
        const grp = document.createElement("optgroup");
        grp.label = fam;
        for (const op of ops) {
          const opt = document.createElement("option");
          opt.value = op.id;
          opt.textContent = `${op.name} (${op.n_moves} moves)`;
          grp.appendChild(opt);
        }
        sel.appendChild(grp);
      }
      _showOpeningInfo();
    })
    .catch(() => {
      $("sel-opening").innerHTML = '<option value="">Could not load openings</option>';
    });
}

function _showOpeningInfo() {
  const sel  = $("sel-opening");
  const info = $("opening-info");
  if (!sel || !info) return;
  const op = _openingsData.find(o => o.id === sel.value);
  if (!op) { info.textContent = ""; return; }
  const total = op.total_games;
  const stats = total
    ? `W ${op.w_wins} / B ${op.b_wins} / D ${op.draws}  (${total} games)`
    : "No games recorded yet";
  const side = op.side === "W" ? "White" : op.side === "B" ? "Black" : "Both sides";
  info.innerHTML = `<b>${op.n_moves} moves</b> · ${side} · ${stats}` +
    (op.notes ? `<br><em>${op.notes.slice(0, 120)}</em>` : "");
}

function startReplayOpening() {
  if (!ws) { setStatus("Connect first — start a new game."); return; }
  const id      = $("sel-opening").value;
  if (!id) return;
  const speedMs = parseInt($("rng-replay-speed").value);
  const mode    = $("sel-continue-mode").value;
  ws.send(JSON.stringify({
    type:          "replay_opening",
    opening_id:    id,
    speed_ms:      speedMs,
    continue_mode: mode,
  }));
  setStatus("Replaying opening…");
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

// ── Tournament helpers ────────────────────────────────────────────────────────

function _renderTournamentInit(msg) {
  $("tournament-intro").hidden  = true;
  $("tournament-active").hidden = false;
  $("tournament-complete-info").hidden = true;
  $("btn-tournament-start").hidden = true;
  $("tournament-rows").innerHTML = "";
  $("tournament-elo").textContent   = msg.player_elo;
  $("tournament-total").textContent = "0";
  $("tournament-max").textContent   = msg.roster.length * 2;
  // Show panel
  $("tournament-panel").hidden = false;
  $("toggle-tournament").classList.add("btn-active");
}

function _setTournamentBadge(text) {
  let badge = document.getElementById("tournament-badge");
  if (!badge) {
    badge = document.createElement("div");
    badge.id = "tournament-badge";
    badge.className = "tournament-badge";
    const sb = $("status-bar");
    sb.parentNode.insertBefore(badge, sb.nextSibling);
  }
  if (text === null) { badge.hidden = true; return; }
  badge.hidden = false;
  badge.textContent = text;
}

function _handleTournamentNext(msg) {
  const colorName = msg.human_color === "W" ? "White" : "Black";
  $("tournament-opponent-info").innerHTML =
    `Game ${msg.game_idx + 1} of 6: <strong>${msg.label}</strong><br>` +
    `You play as <strong>${colorName}</strong>`;
  _setTournamentBadge(`🏆 Round ${msg.game_idx + 1}/6 — ${msg.label} · You play ${colorName}`);
  addCommentary("Tournament", `Game ${msg.game_idx + 1}: ${msg.label} — you play as ${colorName}`, "ai");
  // Auto-start the tournament game over the existing WebSocket
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type:           "new_game",
      tournament_game: true,
      use_llm:        $("chk-llm").checked,
    }));
  }
}

function _updateTournamentScoreboard(msg) {
  $("tournament-elo").textContent   = msg.player_elo;
  $("tournament-total").textContent = msg.points;
  $("tournament-rows").innerHTML = (msg.results || []).map(r => {
    const cls = r.result === "W" ? "t-win" : r.result === "L" ? "t-loss" : "t-draw";
    const sym = r.result === "W" ? "Win" : r.result === "L" ? "Loss" : "Draw";
    const wp  = r.white_personality || "—";
    const bp  = r.black_personality || "—";
    return `<tr class="${cls}">` +
      `<td>${r.label}</td>` +
      `<td style="text-align:center;font-size:.8em">${wp}<br><span style="color:var(--text-dim)">vs</span><br>${bp}</td>` +
      `<td style="text-align:center">${sym}</td>` +
      `<td style="text-align:center">${r.points}</td>` +
      `</tr>`;
  }).join("");
}

function _handleTournamentComplete(msg) {
  _updateTournamentScoreboard(msg);
  _setTournamentBadge(null);
  $("tournament-opponent-info").textContent = "Tournament complete!";
  $("tournament-complete-info").hidden = false;
  $("tournament-rank").textContent      = msg.rank_label;
  $("tournament-final-elo").textContent = `Final Elo: ${msg.player_elo}`;
  addCommentary("Tournament",
    `Tournament complete! Rank: ${msg.rank_label}  |  ` +
    `Points: ${msg.points}/${msg.max_points}  |  Elo: ${msg.player_elo}`, "ai");
}

// ── Left column tab toggle ────────────────────────────────────────────────────

function _switchLeftTab(tab) {
  const isChat = tab === "chat";
  $("chat-view").hidden    = !isChat;
  $("profile-view").hidden = isChat;
  $("tab-chat").classList.toggle("left-tab-active", isChat);
  $("tab-profile").classList.toggle("left-tab-active", !isChat);
}

// ── Player profile helpers ────────────────────────────────────────────────────

function _fetchAndRenderProfile(name) {
  fetch(`/api/profile/${encodeURIComponent(name)}`)
    .then(r => r.json())
    .then(p => { if (!p.error) _renderProfile(p); })
    .catch(() => {});
}

function _renderProfile(p) {
  const stats = $("profile-stats");
  const empty = $("profile-empty-msg");
  if (!stats) return;
  stats.hidden = false;
  if (empty) empty.hidden = true;

  $("profile-elo").textContent     = p.elo ?? 1000;
  $("profile-games").textContent   = p.games_played ?? 0;
  $("profile-wins").textContent    = p.wins ?? 0;
  $("profile-losses").textContent  = p.losses ?? 0;
  $("profile-draws").textContent   = p.draws ?? 0;

  const gp = p.games_played ?? 0;
  const wr = gp > 0 ? Math.round(((p.wins ?? 0) / gp) * 100) + "%" : "—";
  $("profile-winrate").textContent    = wr;
  $("profile-difficulty").textContent = p.current_difficulty ?? 3;
  $("profile-last-played").textContent = p.last_played ?? "—";
  $("profile-created").textContent     = p.created_at ?? "—";
}
