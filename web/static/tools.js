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
  setText("es-status",     es.exists ? "Available" : "Not found", es.exists ? "status-ok" : "status-missing");
  setText("es-tablecount", es.table_count != null ? String(es.table_count) : "0");
  setText("es-positions",  es.positions != null ? fmt(es.positions) : "—");
  setText("es-size",       es.size_mb   != null ? es.size_mb + " MB" : "—");
  setText("es-mtime",      es.mtime || "—");

  // WDL tables list
  const tablesBody = document.getElementById("es-tables-body");
  if (tablesBody) {
    const tables = es.tables || [];
    if (tables.length === 0) {
      tablesBody.innerHTML = '<div class="stat-row"><span style="color:var(--text-dim)">No tables built yet</span></div>';
    } else {
      tablesBody.innerHTML = tables.map(t =>
        `<div class="stat-row"><span>${t.name}</span><span>${t.size_mb} MB</span></div>`
      ).join("");
    }
  }

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

  // Value network
  const vn = data.value_net || {};
  setText("vn-status", vn.exists ? "Found" : "Not found", vn.exists ? "status-ok" : "status-missing");
  setText("vn-size",   vn.size_mb != null ? vn.size_mb + " MB" : "—");
  setText("vn-mtime",  vn.mtime || "—");

  // Opening book
  const ob = data.opening_book || {};
  setText("ob-total", fmt(ob.total || 0));
  setText("ob-named", fmt(ob.named || 0));

  // Malom perfect DB
  const ml = data.malom_db || {};
  setText("ml-status", ml.status === "loaded" ? "Loaded" : "Not loaded",
          ml.status === "loaded" ? "status-ok" : "status-missing");
  setText("ml-path", ml.path || "—");

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

  _ws.onerror = () => {
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
  const gamesdir = document.getElementById("fg-gamesdir").value.trim();
  const dbdir    = document.getElementById("fg-dbdir").value.trim();
  const args = [];
  if (gamesdir) args.push("--expand-from-games", gamesdir);
  if (dbdir)    args.push("--db-dir", dbdir);
  args.push("--min-seed-frequency",  document.getElementById("fg-minseed").value);
  args.push("--expand-depth",        document.getElementById("fg-expdepth").value);
  args.push("--early-expand-depth",  document.getElementById("fg-expdepth-early").value);
  args.push("--passes",              document.getElementById("fg-passes").value);
  args.push("--max-gb",              document.getElementById("fg-maxgb").value);
  args.push("--max-db-gb",           document.getElementById("fg-maxdbgb").value);
  const maxexpand = parseInt(document.getElementById("fg-maxexpand").value);
  if (maxexpand > 0) args.push("--max-expand-positions", maxexpand);
  if (document.getElementById("fg-quiet").checked)  args.push("--quiet");
  if (document.getElementById("fg-dryrun").checked) args.push("--dry-run");
  runTool("build_fullgame_db", args);
});

// ── Build Endgame Solved DB ───────────────────────────────────────────────────

document.getElementById("btn-build-eg").addEventListener("click", () => {
  const outdir = document.getElementById("eg-outdir").value.trim() || "data/endgame";
  const args = ["--out-dir", outdir];
  if (document.getElementById("eg-buildall").checked) {
    args.push("--build-all");
    args.push("--max-sum", document.getElementById("eg-maxsum").value);
  } else {
    const nw = parseInt(document.getElementById("eg-nw").value);
    const nb = parseInt(document.getElementById("eg-nb").value);
    if (nw > 0) args.push("--nW", nw);
    if (nb > 0) args.push("--nB", nb);
  }
  if (document.getElementById("eg-skip").checked)  args.push("--skip-existing");
  if (document.getElementById("eg-quiet").checked) args.push("--quiet");
  runTool("build_endgame_db", args);
});

// ── Endgame Self-Play ─────────────────────────────────────────────────────────

document.getElementById("btn-endgame-play").addEventListener("click", () => {
  const args = [
    "--positions",  document.getElementById("ep-positions").value,
    "--difficulty", document.getElementById("ep-diff").value,
    "--parallel",   document.getElementById("ep-parallel").value,
    "--min-pieces", document.getElementById("ep-minpc").value,
    "--max-pieces", document.getElementById("ep-maxpc").value,
  ];
  const pers = document.getElementById("ep-personalities").value.trim();
  if (pers) args.push("--personalities", pers);
  if (document.getElementById("ep-seedgames").checked) args.push("--seed-from-games");
  runTool("endgame_play", args);
});

// ── Self-Play ─────────────────────────────────────────────────────────────────

document.getElementById("btn-selfplay").addEventListener("click", () => {
  const args = [
    "--games",    document.getElementById("sp-games").value,
    "--white",    document.getElementById("sp-white").value,
    "--black",    document.getElementById("sp-black").value,
    "--parallel", document.getElementById("sp-parallel").value,
    "--blunder",  document.getElementById("sp-blunder").value,
  ];
  const whitePers = document.getElementById("sp-white-pers").value;
  const blackPers = document.getElementById("sp-black-pers").value;
  const persPool  = document.getElementById("sp-personalities").value.trim();
  if (whitePers) args.push("--white-personality", whitePers);
  if (blackPers) args.push("--black-personality", blackPers);
  if (persPool)  args.push("--personalities", persPool);
  if (document.getElementById("sp-nollm").checked)       args.push("--no-llm");
  if (document.getElementById("sp-swap").checked)        args.push("--swap");
  if (document.getElementById("sp-nameopenings").checked) args.push("--name-openings");
  runTool("self_play", args);
});

// ── Train Value Network ───────────────────────────────────────────────────────

document.getElementById("btn-train-vnet").addEventListener("click", () => {
  const args = [
    "--games-dir",  document.getElementById("vn-gamesdir").value.trim() || "data/games",
    "--output",     document.getElementById("vn-output").value.trim()   || "data/value_net.npz",
    "--epochs",     document.getElementById("vn-epochs").value,
    "--lr",         document.getElementById("vn-lr").value,
    "--batch-size", document.getElementById("vn-batch").value,
  ];
  runTool("train_value_net", args);
});

// ── Evolve Weights ────────────────────────────────────────────────────────────

document.getElementById("ew-gauntlet").addEventListener("change", (e) => {
  document.getElementById("ew-gauntlet-note").style.display = e.target.checked ? "" : "none";
});

document.getElementById("btn-evolve").addEventListener("click", () => {
  const args = [
    "--generations",        document.getElementById("ew-gens").value,
    "--games-per-gen",      document.getElementById("ew-gpg").value,
    "--difficulty",         document.getElementById("ew-diff").value,
    "--parallel",           document.getElementById("ew-parallel").value,
    "--sigma",              document.getElementById("ew-sigma").value,
    "--threshold",          document.getElementById("ew-threshold").value,
    "--gauntlet-threshold", document.getElementById("ew-g-threshold").value,
    "--era-size",           document.getElementById("ew-erasize").value,
    "--era-top-k",          document.getElementById("ew-topk").value,
    "--bias-strength",      document.getElementById("ew-bias").value,
    "--warm-blend",         document.getElementById("ew-warmblend").value,
  ];
  if (document.getElementById("ew-gauntlet").checked) args.push("--gauntlet");
  const pers = document.getElementById("ew-personalities").value.trim();
  if (pers) args.push("--personalities", pers);
  const subset = parseInt(document.getElementById("ew-subset").value);
  if (subset > 0) args.push("--subset-size", subset);
  const seed = document.getElementById("ew-seed").value.trim();
  if (seed) args.push("--seed", seed);
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

// ── DB Path Settings ──────────────────────────────────────────────────────────

async function loadDbSettings() {
  try {
    const r = await fetch("/api/db_settings");
    const d = await r.json();
    document.getElementById("dbs-fullgame").value = d.fullgame_db_path || "";
    document.getElementById("dbs-endgame").value  = d.endgame_solved_dir || "";
    document.getElementById("dbs-malom").value    = d.malom_db_path || "";
  } catch (e) {
    console.error("db_settings fetch failed", e);
  }
}

document.getElementById("btn-dbs-save").addEventListener("click", async () => {
  const statusEl = document.getElementById("dbs-status");
  try {
    await fetch("/api/db_settings", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        fullgame_db_path:   document.getElementById("dbs-fullgame").value.trim(),
        endgame_solved_dir: document.getElementById("dbs-endgame").value.trim(),
        malom_db_path:      document.getElementById("dbs-malom").value.trim(),
      }),
    });
    statusEl.textContent = "Saved — restart server to apply";
    statusEl.style.color = "var(--accent, #4caf50)";
  } catch (e) {
    statusEl.textContent = "Save failed";
    statusEl.style.color = "var(--danger, #e53935)";
    console.error("db_settings save failed", e);
  }
});

// ── Name Openings ─────────────────────────────────────────────────────────────

document.getElementById("btn-nameopenings").addEventListener("click", () => {
  const args = [
    "--min-common", document.getElementById("no-mincommon").value,
    "--ollama-url", document.getElementById("no-ollamaurl").value.trim(),
    "--model",      document.getElementById("no-model").value.trim(),
  ];
  if (document.getElementById("no-dryrun").checked)   args.push("--dry-run");
  if (document.getElementById("no-mergeonly").checked) args.push("--merge-only");
  runTool("name_openings", args);
});

// ── Purge AI Learning ─────────────────────────────────────────────────────────

document.getElementById("btn-purge").addEventListener("click", () => {
  if (document.getElementById("purge-dryrun").checked) {
    runTool("purge_ai_learning", ["--dry-run", "--yes"], /*confirmed=*/true);
    return;
  }
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
loadDbSettings();
setInterval(refreshStatus, 15_000);
