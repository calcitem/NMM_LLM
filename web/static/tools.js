/* tools.js — Tools management page */
"use strict";

// ── Status refresh ────────────────────────────────────────────────────────────

async function refreshStatus() {
  let data;
  try {
    const r = await fetch("/api/tool_status");
    data = await r.json();
  } catch (e) {
    console.error("tool_status fetch failed", e);
    return;
  }

  const fmt = (n) => n >= 1e6 ? (n / 1e6).toFixed(1) + "M"
                   : n >= 1e3 ? (n / 1e3).toFixed(1) + "K"
                   : String(n);

  // FullGame DB
  const fg = data.fullgame_db || {};
  setText("fg-status",    fg.exists ? "Available" : "Not found", fg.exists ? "status-ok" : "status-missing");
  setText("fg-positions", fg.positions != null ? fmt(fg.positions) : "—");
  setText("fg-resolved",  fg.resolved  != null ? fmt(fg.resolved)  : "—");
  setText("fg-size",      fg.size_mb   != null ? fg.size_mb + " MB" : "—");
  setText("fg-mtime",     fg.mtime || "—");

  // Endgame solved DB
  const es = data.endgame_solved || {};
  setText("es-status",    es.exists ? "Available" : "Not found", es.exists ? "status-ok" : "status-missing");
  setText("es-positions", es.positions != null ? fmt(es.positions) : "—");
  setText("es-size",      es.size_mb   != null ? es.size_mb + " MB" : "—");
  setText("es-mtime",     es.mtime || "—");

  // Trajectory DB
  const tdb = data.trajectory_db || {};
  setText("tdb-games",   fmt(tdb.games   || 0));
  setText("tdb-entries", fmt(tdb.entries || 0));

  // Games
  const gm = data.games || {};
  setText("gm-count",    fmt(gm.count    || 0));
  setText("gm-earliest", gm.earliest || "—");
  setText("gm-latest",   gm.latest   || "—");

  // Weights
  const wt = data.weights || {};
  setText("wt-status",  wt.exists ? "Found" : "Not found", wt.exists ? "status-ok" : "status-missing");
  setText("wt-size",    wt.size_mb != null ? wt.size_mb + " MB" : "—");
  setText("wt-mtime",   wt.mtime || "—");

  // Opening book
  const ob = data.opening_book || {};
  setText("ob-total", fmt(ob.total || 0));
  setText("ob-named", fmt(ob.named || 0));

  // Auto-evolve status
  const ae = data.auto_evolve || {};
  if (ae.after_games > 0) {
    document.getElementById("ae-threshold").value = ae.after_games;
    document.getElementById("ae-status").textContent =
      `${ae.games_since} / ${ae.after_games} games since last run`;
  }

  // Busy badge
  document.getElementById("busy-badge").style.display = data.busy ? "" : "none";

  setBusy(data.busy);
}

function setText(id, text, extraClass) {
  const el = document.getElementById(id);
  if (!el) return;
  el.textContent = text;
  el.className = "";
  if (extraClass) el.classList.add(extraClass);
}

// ── WebSocket tool runner ─────────────────────────────────────────────────────

let _ws = null;
let _currentTool = null;

function setBusy(busy) {
  document.querySelectorAll(".btn-run").forEach(b => b.disabled = busy);
  document.getElementById("btn-stop").style.display = busy ? "" : "none";
  document.getElementById("busy-badge").style.display = busy ? "" : "none";
}

function logLine(text, cls) {
  const el = document.getElementById("log-output");
  const span = document.createElement("div");
  span.textContent = text;
  if (cls) span.className = "log-" + cls;
  el.appendChild(span);
  if (document.getElementById("chk-autoscroll").checked) {
    el.scrollTop = el.scrollHeight;
  }
}

function runTool(tool, args, confirmed) {
  if (_ws) { _ws.close(); _ws = null; }

  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  _ws = new WebSocket(`${proto}//${location.host}/ws/tools`);
  _currentTool = tool;
  document.getElementById("log-tool-label").textContent = `— ${tool}`;
  setBusy(true);

  _ws.onopen = () => {
    _ws.send(JSON.stringify({ tool, args, confirmed: !!confirmed }));
  };

  _ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "confirm") {
      // Server needs confirmation (should not reach here — handled before open)
      return;
    }
    if (msg.type === "done") {
      logLine(msg.text, "done");
      setBusy(false);
      refreshStatus();
      _ws = null;
    } else if (msg.type === "error") {
      logLine(msg.text, "error");
      setBusy(false);
      refreshStatus();
      _ws = null;
    } else if (msg.type === "cmd") {
      logLine(msg.text, "cmd");
    } else {
      logLine(msg.text);
    }
  };

  _ws.onerror = (e) => {
    logLine("[WebSocket error]", "error");
    setBusy(false);
    _ws = null;
  };

  _ws.onclose = () => {
    if (_currentTool) {
      setBusy(false);
      _currentTool = null;
    }
  };
}

// ── Build FullGame DB ─────────────────────────────────────────────────────────

document.getElementById("btn-build-fg").addEventListener("click", () => {
  const dbdir = document.getElementById("fg-dbdir").value.trim();
  const args = ["--expand-from-games", "data/games"];
  args.push("--min-seed-frequency", document.getElementById("fg-minseed").value);
  args.push("--expand-depth",       document.getElementById("fg-expdepth").value);
  args.push("--early-expand-depth", document.getElementById("fg-expdepth-early").value);
  const maxexpand = document.getElementById("fg-maxexpand").value;
  if (parseInt(maxexpand) > 0) args.push("--max-expand-positions", maxexpand);
  if (dbdir) args.push("--db-dir", dbdir);
  runTool("build_fullgame_db", args);
});

// ── Build Endgame DB ──────────────────────────────────────────────────────────

document.getElementById("btn-build-eg").addEventListener("click", () => {
  const outdir = document.getElementById("eg-outdir").value.trim() || "data/endgame";
  runTool("build_endgame_db", ["--out-dir", outdir]);
});

// ── Self-Play ─────────────────────────────────────────────────────────────────

document.getElementById("btn-selfplay").addEventListener("click", () => {
  const args = [
    "--games",    document.getElementById("sp-games").value,
    "--white",    document.getElementById("sp-white").value,
    "--black",    document.getElementById("sp-black").value,
    "--parallel", document.getElementById("sp-parallel").value,
  ];
  if (document.getElementById("sp-nollm").checked) args.push("--no-llm");
  runTool("self_play", args);
});

// ── Evolve Weights ────────────────────────────────────────────────────────────

document.getElementById("ew-gauntlet").addEventListener("change", (e) => {
  document.getElementById("ew-gauntlet-note").style.display = e.target.checked ? "" : "none";
});

document.getElementById("btn-evolve").addEventListener("click", () => {
  const args = [
    "--generations", document.getElementById("ew-gens").value,
    "--games-per-gen", document.getElementById("ew-gpg").value,
    "--difficulty", document.getElementById("ew-diff").value,
    "--parallel",  document.getElementById("ew-parallel").value,
    "--sigma",     document.getElementById("ew-sigma").value,
  ];
  if (document.getElementById("ew-gauntlet").checked) args.push("--gauntlet");
  const subset = parseInt(document.getElementById("ew-subset").value);
  if (subset > 0) args.push("--subset-size", subset);
  runTool("evolve_weights_v2", args);
});

// ── Auto-Evolve setting ───────────────────────────────────────────────────────

async function loadAutoEvolve() {
  try {
    const r = await fetch("/api/auto_evolve");
    const d = await r.json();
    document.getElementById("ae-threshold").value = d.after_games || 0;
    const status = d.after_games > 0
      ? `${d.games_since} / ${d.after_games} games since last run`
      : "disabled";
    document.getElementById("ae-status").textContent = status;
  } catch (e) {
    console.error("auto_evolve fetch failed", e);
  }
}

document.getElementById("btn-ae-save").addEventListener("click", async () => {
  const after = parseInt(document.getElementById("ae-threshold").value) || 0;
  try {
    await fetch("/api/auto_evolve", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({after_games: after}),
    });
    document.getElementById("ae-status").textContent = after > 0
      ? `Set — triggers every ${after} games`
      : "disabled";
  } catch (e) {
    console.error("auto_evolve save failed", e);
  }
});

// ── Name Openings ─────────────────────────────────────────────────────────────

document.getElementById("btn-nameopenings").addEventListener("click", () => {
  const args = ["--min-common", document.getElementById("no-mincommon").value];
  if (document.getElementById("no-dryrun").checked)   args.push("--dry-run");
  if (document.getElementById("no-mergeonly").checked) args.push("--merge-only");
  runTool("name_openings", args);
});

// ── Purge AI Learning ─────────────────────────────────────────────────────────

document.getElementById("btn-purge").addEventListener("click", () => {
  if (document.getElementById("purge-dryrun").checked) {
    // Dry run is read-only — no confirmation dialog needed
    runTool("purge_ai_learning", ["--dry-run", "--yes"], /*confirmed=*/true);
    return;
  }
  // Real purge — show confirmation modal
  document.getElementById("modal-confirm").style.display = "flex";
});

document.getElementById("modal-cancel").addEventListener("click", () => {
  document.getElementById("modal-confirm").style.display = "none";
});

document.getElementById("modal-confirm-ok").addEventListener("click", () => {
  document.getElementById("modal-confirm").style.display = "none";
  runTool("purge_ai_learning", ["--yes"], true);
});

// ── Log controls ──────────────────────────────────────────────────────────────

document.getElementById("btn-clear-log").addEventListener("click", () => {
  document.getElementById("log-output").innerHTML = "";
});

document.getElementById("btn-stop").addEventListener("click", () => {
  if (_ws) { _ws.close(); _ws = null; }
  logLine("[stopped by user]", "error");
  setBusy(false);
});

document.getElementById("btn-refresh-status").addEventListener("click", refreshStatus);

// ── Init ──────────────────────────────────────────────────────────────────────

refreshStatus();
loadAutoEvolve();
// Auto-refresh status every 15s
setInterval(refreshStatus, 15_000);
