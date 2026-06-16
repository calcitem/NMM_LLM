/**
 * explorer.js — 3D NMM position explorer (Three.js ES module)
 *
 * Board positions are laid out in 3D space matching the NMM coordinate grid.
 * Win-% bars rise above each candidate next-move square.
 * Color: green = Malom says opponent loses (good move), red = opponent wins (bad move),
 *        amber = draw, grey = no Malom annotation (brightness = human win%).
 * Click a bar to advance the board. Back button rewinds.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// ── Board geometry data ───────────────────────────────────────────────────────

const POS_COORDS = {
  a7: [-3, 0, -3], d7: [0, 0, -3], g7: [3, 0, -3],
  a4: [-3, 0,  0],                  g4: [3, 0,  0],
  a1: [-3, 0,  3], d1: [0, 0,  3], g1: [3, 0,  3],
  b6: [-2, 0, -2], d6: [0, 0, -2], f6: [2, 0, -2],
  b4: [-2, 0,  0],                  f4: [2, 0,  0],
  b2: [-2, 0,  2], d2: [0, 0,  2], f2: [2, 0,  2],
  c5: [-1, 0, -1], d5: [0, 0, -1], e5: [1, 0, -1],
  c4: [-1, 0,  0],                  e4: [1, 0,  0],
  c3: [-1, 0,  1], d3: [0, 0,  1], e3: [1, 0,  1],
};

const EDGES = [
  ['a7','d7'],['d7','g7'],['g7','g4'],['g4','g1'],['g1','d1'],['d1','a1'],['a1','a4'],['a4','a7'],
  ['b6','d6'],['d6','f6'],['f6','f4'],['f4','f2'],['f2','d2'],['d2','b2'],['b2','b4'],['b4','b6'],
  ['c5','d5'],['d5','e5'],['e5','e4'],['e4','e3'],['e3','d3'],['d3','c3'],['c3','c4'],['c4','c5'],
  ['d7','d6'],['d6','d5'],
  ['g4','f4'],['f4','e4'],
  ['d1','d2'],['d2','d3'],
  ['a4','b4'],['b4','c4'],
];

// ── Colors ────────────────────────────────────────────────────────────────────

const C = {
  board:   0x6b4f22,
  pad:     0x4a3518,
  padHov:  0x7a5f2a,
  lineWd:  0x8b6b3a,
  white:   0xf5f0dc,
  black:   0x1a1a1a,
  barGood: 0x22c55e,   // Malom: opp loses → good for mover
  barBad:  0xef4444,   // Malom: opp wins  → bad for mover
  barDraw: 0xf59e0b,
  barNone: 0x6b8eae,   // no Malom data
  barHov:  0xffd700,
  highlight: 0xffd700,
};

function winPctColor(pct) {
  // No Malom: interpolate red(0%) → amber(50%) → green(100%)
  const t = Math.max(0, Math.min(1, pct));
  if (t < 0.5) {
    const u = t * 2;
    return new THREE.Color().setRGB(0.94, 0.24 + 0.38 * u, 0.07);
  }
  const u = (t - 0.5) * 2;
  return new THREE.Color().setRGB(0.94 - 0.8 * u, 0.62 + 0.14 * u, 0.07);
}

function barColor(moveData) {
  if (moveData.malom_wdl_after === 'L') return new THREE.Color(C.barGood);
  if (moveData.malom_wdl_after === 'W') return new THREE.Color(C.barBad);
  if (moveData.malom_wdl_after === 'D') return new THREE.Color(C.barDraw);
  return winPctColor(moveData.win_pct);
}

// ── Scene setup ───────────────────────────────────────────────────────────────

const canvas  = document.getElementById('board-canvas');
const wrap    = document.getElementById('canvas-wrap');
const tooltip = document.getElementById('tooltip');
const loading = document.getElementById('loading-overlay');

const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x1a1612);
scene.fog = new THREE.Fog(0x1a1612, 18, 30);

const camera = new THREE.PerspectiveCamera(45, 1, 0.1, 100);
camera.position.set(0, 8, 9);
camera.lookAt(0, 0, 0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor  = 0.08;
controls.minDistance    = 4;
controls.maxDistance    = 20;
controls.maxPolarAngle  = Math.PI / 2.1;

// Lights
const ambient = new THREE.AmbientLight(0xfff8e7, 0.6);
scene.add(ambient);
const dirLight = new THREE.DirectionalLight(0xffe8b0, 1.2);
dirLight.position.set(5, 10, 8);
dirLight.castShadow = true;
dirLight.shadow.mapSize.set(2048, 2048);
scene.add(dirLight);
const fillLight = new THREE.DirectionalLight(0x8fb3d4, 0.3);
fillLight.position.set(-5, 3, -5);
scene.add(fillLight);

// ── Static board geometry ─────────────────────────────────────────────────────

function buildStaticBoard() {
  // Ground plane
  const planeGeo = new THREE.PlaneGeometry(9, 9);
  const planeMat = new THREE.MeshLambertMaterial({ color: C.board });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI / 2;
  plane.receiveShadow = true;
  scene.add(plane);

  // Position pads
  const padGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.06, 16);
  for (const [pos, [x,, z]] of Object.entries(POS_COORDS)) {
    const mat = new THREE.MeshLambertMaterial({ color: C.pad });
    const mesh = new THREE.Mesh(padGeo, mat);
    mesh.position.set(x, 0.03, z);
    mesh.receiveShadow = true;
    mesh.userData.pos = pos;
    mesh.userData.isPad = true;
    scene.add(mesh);
  }

  // Edges
  const lineMat = new THREE.MeshLambertMaterial({ color: C.lineWd });
  for (const [a, b] of EDGES) {
    const [ax,, az] = POS_COORDS[a];
    const [bx,, bz] = POS_COORDS[b];
    const mid = new THREE.Vector3((ax+bx)/2, 0.02, (az+bz)/2);
    const dir = new THREE.Vector3(bx-ax, 0, bz-az);
    const len = dir.length();
    const geo = new THREE.CylinderGeometry(0.04, 0.04, len, 6);
    const mesh = new THREE.Mesh(geo, lineMat);
    mesh.position.copy(mid);
    mesh.rotation.z = Math.PI / 2;
    const angle = Math.atan2(bz - az, bx - ax);
    mesh.rotation.y = -angle;
    scene.add(mesh);
  }
}

buildStaticBoard();

// ── Dynamic layers ────────────────────────────────────────────────────────────

const pieceGroup = new THREE.Group();
const barGroup   = new THREE.Group();
scene.add(pieceGroup);
scene.add(barGroup);

const pieceGeoW = new THREE.CylinderGeometry(0.26, 0.26, 0.4, 20);
const pieceGeoB = new THREE.CylinderGeometry(0.26, 0.26, 0.4, 20);
const matWhite  = new THREE.MeshLambertMaterial({ color: C.white });
const matBlack  = new THREE.MeshLambertMaterial({ color: C.black });

function rebuildPieces(boardDict) {
  pieceGroup.clear();
  for (const [pos, piece] of Object.entries(boardDict)) {
    if (!piece) continue;
    const [x,, z] = POS_COORDS[pos];
    const geo  = piece === 'W' ? pieceGeoW : pieceGeoB;
    const mat  = piece === 'W' ? matWhite  : matBlack;
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, 0.23, z);
    mesh.castShadow = true;
    pieceGroup.add(mesh);
  }
}

const barMeshMap = new Map(); // notation → { mesh, data }

function rebuildBars(movesArray) {
  barGroup.clear();
  barMeshMap.clear();

  for (const mv of movesArray) {
    const toSq = mv.to_sq;
    if (!toSq || !POS_COORDS[toSq]) continue;

    const height = Math.max(0.15, mv.win_pct * 4);
    const geo  = new THREE.BoxGeometry(0.38, height, 0.38);
    const col  = barColor(mv);
    const mat  = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 0.88 });
    const mesh = new THREE.Mesh(geo, mat);

    const [x,, z] = POS_COORDS[toSq];
    mesh.position.set(x, height / 2 + 0.07, z);
    mesh.castShadow = true;
    mesh.userData.notation  = mv.notation;
    mesh.userData.moveData  = mv;
    mesh.userData.baseColor = col.clone();

    barGroup.add(mesh);
    barMeshMap.set(mv.notation, { mesh, data: mv });
  }
}

// ── Raycasting / hover / click ────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
let   hoveredBar = null;

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  mouse.x = ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(barGroup.children);

  if (hoveredBar) {
    hoveredBar.material.color.copy(hoveredBar.userData.baseColor);
    hoveredBar.material.opacity = 0.88;
    hoveredBar = null;
    tooltip.style.display = 'none';
  }

  if (hits.length > 0) {
    const mesh = hits[0].object;
    hoveredBar = mesh;
    mesh.material.color.set(C.barHov);
    mesh.material.opacity = 1.0;
    showTooltip(e.clientX, e.clientY, mesh.userData.moveData);

    // Highlight corresponding list item
    document.querySelectorAll('.move-item').forEach(el => {
      el.classList.toggle('highlighted', el.dataset.notation === mesh.userData.notation);
    });
  } else {
    document.querySelectorAll('.move-item.highlighted').forEach(el => el.classList.remove('highlighted'));
  }
}

canvas.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('click', e => {
  if (hoveredBar) {
    const notation = hoveredBar.userData.notation;
    applyMove(notation);
  }
});

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTooltip(cx, cy, mv) {
  const wdlText = mv.malom_wdl_after
    ? `${mv.malom_wdl_after}${mv.malom_dtw_after != null ? ` (${mv.malom_dtw_after > 0 ? '+' : ''}${mv.malom_dtw_after} DTW)` : ''}`
    : '—';
  tooltip.innerHTML = `
    <div class="tt-notation">${mv.notation}</div>
    <div class="tt-row"><span class="tt-label">Win%</span><span>${(mv.win_pct*100).toFixed(1)}%</span></div>
    <div class="tt-row"><span class="tt-label">W/D/L</span><span>${mv.wins}/${mv.draws}/${mv.losses}</span></div>
    <div class="tt-row"><span class="tt-label">Games</span><span>${mv.total}</span></div>
    <div class="tt-row"><span class="tt-label">Avg plies left</span><span>${mv.avg_moves_to_end.toFixed(0)}</span></div>
    <div class="tt-row"><span class="tt-label">Malom (after)</span><span>${wdlText}</span></div>
  `;
  const rect = wrap.getBoundingClientRect();
  let tx = cx - rect.left + 14;
  let ty = cy - rect.top  - 10;
  if (tx + 180 > rect.width)  tx = cx - rect.left - 180;
  if (ty + 140 > rect.height) ty = cy - rect.top  - 140;
  tooltip.style.left    = tx + 'px';
  tooltip.style.top     = ty + 'px';
  tooltip.style.display = 'block';
}

// ── Side-panel updates ────────────────────────────────────────────────────────

function updatePanel(data) {
  // Turn badge
  const badge = document.getElementById('turn-badge');
  badge.textContent = data.turn === 'W' ? 'White' : 'Black';
  badge.className   = 'turn-badge ' + (data.turn === 'W' ? 'white' : 'black');

  // Position stats
  const posEl = document.getElementById('pos-stats');
  const ps    = data.position_stats;
  if (!ps) {
    posEl.innerHTML = '<div id="no-data-notice">No HumanDB data for this position.</div>';
  } else {
    const tot = Math.max(1, ps.total_games);
    const wp  = (ps.wins  / tot * 100).toFixed(1);
    const dp  = (ps.draws / tot * 100).toFixed(1);
    const lp  = (ps.losses/ tot * 100).toFixed(1);
    const malomHtml = ps.malom_wdl
      ? `<span class="malom-badge malom-${ps.malom_wdl.toLowerCase()}">${ps.malom_wdl}${ps.malom_dtw != null ? ` DTW ${ps.malom_dtw}` : ''}</span>`
      : '';
    posEl.innerHTML = `
      <div class="stat-row"><span class="stat-label">Games</span><span class="stat-val">${ps.total_games.toLocaleString()}</span></div>
      <div class="wdl-bar">
        <div class="w" style="width:${wp}%"></div>
        <div class="d" style="width:${dp}%"></div>
        <div class="l" style="width:${lp}%"></div>
      </div>
      <div class="stat-row"><span class="stat-label">Win%</span><span class="stat-val" style="color:#22c55e">${wp}%</span></div>
      <div class="stat-row"><span class="stat-label">Draw%</span><span class="stat-val" style="color:#f59e0b">${dp}%</span></div>
      <div class="stat-row"><span class="stat-label">Loss%</span><span class="stat-val" style="color:#ef4444">${lp}%</span></div>
      ${malomHtml}
    `;
  }

  // Move list
  const listEl = document.getElementById('move-list');
  listEl.innerHTML = '';
  if (data.moves && data.moves.length > 0) {
    for (const mv of data.moves) {
      const col   = barColor(mv);
      const colHex = '#' + col.getHexString();
      const wdl   = mv.malom_wdl_after
        ? `<span class="move-malom" style="background:${mv.malom_wdl_after==='L'?'#16532a':mv.malom_wdl_after==='W'?'#7f1d1d':'#78350f'};color:${mv.malom_wdl_after==='L'?'#4ade80':mv.malom_wdl_after==='W'?'#fca5a5':'#fcd34d'}">${mv.malom_wdl_after}${mv.malom_dtw_after!=null?' '+mv.malom_dtw_after:''}</span>`
        : '';
      const item = document.createElement('div');
      item.className = 'move-item';
      item.dataset.notation = mv.notation;
      item.innerHTML = `
        <div class="move-bar-swatch" style="background:${colHex}"></div>
        <span class="move-notation">${mv.notation}</span>
        <span class="move-sub">${mv.total}</span>
        ${wdl}
        <span class="move-pct" style="color:${colHex}">${(mv.win_pct*100).toFixed(1)}%</span>
      `;
      item.addEventListener('click', () => applyMove(mv.notation));
      item.addEventListener('mouseenter', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) { entry.mesh.material.color.set(C.barHov); entry.mesh.material.opacity = 1; }
      });
      item.addEventListener('mouseleave', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) { entry.mesh.material.color.copy(entry.mesh.userData.baseColor); entry.mesh.material.opacity = 0.88; }
      });
      listEl.appendChild(item);
    }
  } else {
    listEl.innerHTML = '<div style="padding:0.5rem 0.75rem;color:#8a7a5a;font-size:0.8rem">No move data.</div>';
  }

  // Winning line
  const lineEl = document.getElementById('winning-line');
  lineEl.textContent = data.winning_line && data.winning_line.length > 0
    ? data.winning_line.join(' → ') : '—';
}

// ── Navigation ────────────────────────────────────────────────────────────────

const history = [];  // stack of FEN strings
let   currentData = null;

async function loadPosition(fen) {
  loading.style.display = 'flex';
  try {
    const res  = await fetch('/api/explorer/position?fen=' + encodeURIComponent(fen));
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    currentData = data;
    document.getElementById('fen-input').value = data.fen;
    rebuildPieces(data.board);
    rebuildBars(data.moves || []);
    updatePanel(data);
    document.getElementById('btn-back').disabled = history.length === 0;
  } catch (err) {
    alert('Failed to load position: ' + err.message);
  } finally {
    loading.style.display = 'none';
  }
}

async function applyMove(notation) {
  if (!currentData) return;
  history.push(currentData.fen);
  await loadPosition(await fenAfterMove(currentData.fen, notation));
}

async function fenAfterMove(fen, move) {
  const res  = await fetch(`/api/explorer/move?fen=${encodeURIComponent(fen)}&move=${encodeURIComponent(move)}`);
  const data = await res.json();
  return data.fen || fen;
}

document.getElementById('btn-back').addEventListener('click', () => {
  if (history.length === 0) return;
  const prev = history.pop();
  loadPosition(prev);
});

document.getElementById('btn-best').addEventListener('click', async () => {
  if (!currentData || !currentData.position_stats) return;
  const best = currentData.position_stats.canonical_winning_move;
  if (!best) { alert('No winning move in DB for this position.'); return; }
  await applyMove(best);
});

document.getElementById('btn-go').addEventListener('click', () => {
  const fen = document.getElementById('fen-input').value.trim();
  if (fen) { history.length = 0; loadPosition(fen); }
});
document.getElementById('fen-input').addEventListener('keydown', e => {
  if (e.key === 'Enter') document.getElementById('btn-go').click();
});

// ── Resize ────────────────────────────────────────────────────────────────────

function resize() {
  const w = wrap.clientWidth;
  const h = wrap.clientHeight;
  renderer.setSize(w, h);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(wrap);
resize();

// ── Render loop ───────────────────────────────────────────────────────────────

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();

// ── Boot ──────────────────────────────────────────────────────────────────────

loadPosition('........................|W|0|0');
