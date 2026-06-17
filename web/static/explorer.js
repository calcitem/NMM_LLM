/**
 * explorer.js — 3D NMM position explorer (Three.js ES module)
 *
 * Bar height = how often humans played that move (most-played = tallest).
 * Bar color  = human win-rate gradient: green (winning) → orange → red (losing).
 * Arrows     = cylinder+cone from the piece's current square to its destination.
 * Malom overlay (toggle) = colored rings + DTW numbers on candidate squares.
 * Click a bar to advance; Back rewinds.
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { CSS2DRenderer, CSS2DObject } from 'three/addons/renderers/CSS2DRenderer.js';

// ── Sprite text helper (for DTW numbers) ─────────────────────────────────────
function makeDtwSprite(text, hexColor) {
  const W = 64, H = 32;
  const canvas = document.createElement('canvas');
  canvas.width  = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  ctx.font = 'bold 20px monospace';
  ctx.textBaseline = 'middle';
  ctx.textAlign    = 'center';
  // dark outline
  ctx.strokeStyle = 'rgba(0,0,0,0.85)';
  ctx.lineWidth   = 4;
  ctx.strokeText(text, W / 2, H / 2);
  ctx.fillStyle = hexColor;
  ctx.fillText(text, W / 2, H / 2);
  const tex = new THREE.CanvasTexture(canvas);
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const sprite = new THREE.Sprite(mat);
  sprite.scale.set(0.7, 0.35, 1);
  return sprite;
}

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
  board:  0x6b4f22,
  pad:    0x4a3518,
  lineWd: 0x8b6b3a,
  white:  0xf5f0dc,
  black:  0x1a1a1a,
  barHov: 0xffd700,
};

// Green (high win%) → orange → red (high loss%)
function winPctColor(pct) {
  const t = Math.max(0, Math.min(1, pct));
  if (t < 0.5) {
    const u = t * 2;
    return new THREE.Color().setRGB(0.94, 0.24 + 0.38 * u, 0.07);
  }
  const u = (t - 0.5) * 2;
  return new THREE.Color().setRGB(0.94 - 0.8 * u, 0.62 + 0.14 * u, 0.07);
}

// Sentinel score [0,1] → green (good) → red (poor)
function sentinelColor(score) {
  // Blue gradient: dark blue (low quality) → bright cyan-blue (high quality)
  const t = Math.max(0, Math.min(1, score));
  return new THREE.Color().setHSL(0.58 + t * 0.08, 0.75, 0.28 + t * 0.35);
}

function barColor(moveData) {
  if (moveData.has_db_data) return winPctColor(moveData.win_pct);
  if (moveData.sentinel_score != null) return sentinelColor(moveData.sentinel_score);
  return new THREE.Color(0x555555);
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

// CSS2D renderer — overlaid for coordinate labels and Malom DTW text
const labelRenderer = new CSS2DRenderer();
labelRenderer.domElement.style.cssText = 'position:absolute;top:0;left:0;pointer-events:none;';
wrap.appendChild(labelRenderer.domElement);

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
  const planeGeo = new THREE.PlaneGeometry(9, 9);
  const planeMat = new THREE.MeshLambertMaterial({ color: C.board });
  const plane = new THREE.Mesh(planeGeo, planeMat);
  plane.rotation.x = -Math.PI / 2;
  plane.receiveShadow = true;
  scene.add(plane);

  const padGeo = new THREE.CylinderGeometry(0.28, 0.28, 0.06, 16);
  for (const [pos, [x,, z]] of Object.entries(POS_COORDS)) {
    const mat  = new THREE.MeshLambertMaterial({ color: C.pad });
    const mesh = new THREE.Mesh(padGeo, mat);
    mesh.position.set(x, 0.03, z);
    mesh.receiveShadow = true;
    mesh.userData.pos   = pos;
    mesh.userData.isPad = true;
    scene.add(mesh);
  }

  const lineMat = new THREE.MeshLambertMaterial({ color: C.lineWd });
  for (const [a, b] of EDGES) {
    const [ax,, az] = POS_COORDS[a];
    const [bx,, bz] = POS_COORDS[b];
    const mid = new THREE.Vector3((ax+bx)/2, 0.02, (az+bz)/2);
    const len = new THREE.Vector3(bx-ax, 0, bz-az).length();
    const geo = new THREE.CylinderGeometry(0.04, 0.04, len, 6);
    const mesh = new THREE.Mesh(geo, lineMat);
    mesh.position.copy(mid);
    mesh.rotation.z = Math.PI / 2;
    mesh.rotation.y = -Math.atan2(bz - az, bx - ax);
    scene.add(mesh);
  }
}

buildStaticBoard();

// ── Coordinate labels (CSS2D) ─────────────────────────────────────────────────

function buildCoordLabels() {
  // Column letters a–g along south edge (z = 4.2)
  ['a','b','c','d','e','f','g'].forEach((letter, i) => {
    const el = document.createElement('div');
    el.className = 'coord-label';
    el.textContent = letter;
    const obj = new CSS2DObject(el);
    obj.position.set(i - 3, 0.5, 4.5);
    scene.add(obj);
  });

  // Row numbers 7–1 along west edge (x = -4.2)
  ['7','6','5','4','3','2','1'].forEach((num, i) => {
    const el = document.createElement('div');
    el.className = 'coord-label';
    el.textContent = num;
    const obj = new CSS2DObject(el);
    obj.position.set(-4.5, 0.5, i - 3);
    scene.add(obj);
  });
}

buildCoordLabels();

// ── Dynamic layers ────────────────────────────────────────────────────────────

const pieceGroup = new THREE.Group();
const barGroup   = new THREE.Group();
const arrowGroup = new THREE.Group();
const malomGroup = new THREE.Group();
malomGroup.visible = false;
scene.add(pieceGroup, barGroup, arrowGroup, malomGroup);

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

const barMeshMap = new Map();
const MAX_BAR_HEIGHT = 2.0;

function rebuildBars(movesArray) {
  barGroup.clear();
  barMeshMap.clear();

  // DB moves: height based on total games
  const dbMoves    = movesArray.filter(m => m.has_db_data);
  const nonDbMoves = movesArray.filter(m => !m.has_db_data);
  const maxTotal   = Math.max(1, ...dbMoves.map(m => m.total));

  // Non-DB moves: height based on heuristic score (shift+normalize)
  const hScores   = nonDbMoves.map(m => m.heuristic_score);
  const minH      = hScores.length ? Math.min(...hScores) : 0;
  const maxH      = hScores.length ? Math.max(...hScores) : 1;
  const hRange    = Math.max(1, maxH - minH);

  for (const mv of movesArray) {
    const toSq = mv.to_sq;
    if (!toSq || !POS_COORDS[toSq]) continue;

    let height;
    if (mv.has_db_data) {
      height = Math.max(0.12, (mv.total / maxTotal) * MAX_BAR_HEIGHT);
    } else {
      // Normalize heuristic into [0.12, MAX_BAR_HEIGHT * 0.5]
      const norm = (mv.heuristic_score - minH) / hRange;
      height = 0.12 + norm * (MAX_BAR_HEIGHT * 0.5);
    }

    const col  = barColor(mv);
    const mat  = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: mv.has_db_data ? 0.88 : 0.55 });
    const mesh = new THREE.Mesh(new THREE.BoxGeometry(0.38, height, 0.38), mat);
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

// ── Move arrows (from_sq → to_sq) ────────────────────────────────────────────

const _up = new THREE.Vector3(0, 1, 0);

function rebuildArrows(movesArray) {
  arrowGroup.clear();
  const dbMoves  = movesArray.filter(m => m.has_db_data);
  const maxTotal = Math.max(1, ...dbMoves.map(m => m.total));

  for (const mv of movesArray) {
    if (!mv.from_sq || !POS_COORDS[mv.from_sq] || !POS_COORDS[mv.to_sq]) continue;
    const [fx,, fz] = POS_COORDS[mv.from_sq];
    const [tx,, tz] = POS_COORDS[mv.to_sq];

    const from3 = new THREE.Vector3(fx, 0.58, fz);
    const to3   = new THREE.Vector3(tx, 0.58, tz);
    const dir   = new THREE.Vector3().subVectors(to3, from3);
    const len   = dir.length();
    if (len < 0.01) continue;
    const dirN = dir.clone().normalize();
    const q    = new THREE.Quaternion().setFromUnitVectors(_up, dirN);

    let col, opacity, shaftRadius, headRadius;
    if (mv.has_db_data) {
      col         = barColor(mv).getHex();
      opacity     = 0.22 + 0.65 * (mv.total / maxTotal);
      shaftRadius = 0.038;
      headRadius  = 0.115;
    } else {
      // Non-DB moves: thin grey arrows, low opacity
      col         = 0x555555;
      opacity     = 0.18;
      shaftRadius = 0.020;
      headRadius  = 0.065;
    }

    const mat      = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity });
    const headLen  = Math.min(0.38, len * 0.28);
    const shaftLen = len - headLen - 0.04;

    const shaft = new THREE.Mesh(new THREE.CylinderGeometry(shaftRadius, shaftRadius, shaftLen, 6), mat);
    shaft.position.copy(from3).addScaledVector(dirN, shaftLen / 2);
    shaft.setRotationFromQuaternion(q);

    const head = new THREE.Mesh(new THREE.ConeGeometry(headRadius, headLen, 8), mat.clone());
    head.position.copy(from3).addScaledVector(dirN, shaftLen + headLen / 2);
    head.setRotationFromQuaternion(q);

    arrowGroup.add(shaft, head);
  }
}

// ── Malom overlay (rings + DTW labels) ───────────────────────────────────────

const malomRingGeo = new THREE.RingGeometry(0.26, 0.41, 24);
malomRingGeo.rotateX(-Math.PI / 2);   // lay flat on board plane

function rebuildMalomOverlay(movesArray) {
  malomGroup.clear();
  for (const mv of movesArray) {
    if (!mv.malom_wdl_after || !mv.to_sq || !POS_COORDS[mv.to_sq]) continue;
    const col = mv.malom_wdl_after === 'L' ? 0x22c55e
               : mv.malom_wdl_after === 'W' ? 0xef4444
               : 0xf59e0b;
    const mat  = new THREE.MeshLambertMaterial({ color: col, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
    const ring = new THREE.Mesh(malomRingGeo, mat);
    const [x,, z] = POS_COORDS[mv.to_sq];
    ring.position.set(x, 0.08, z);

    if (mv.malom_dtw_after != null) {
      const col = mv.malom_wdl_after === 'L' ? '#4ade80'
                : mv.malom_wdl_after === 'W' ? '#fca5a5'
                : '#fcd34d';
      const sprite = makeDtwSprite(String(Math.abs(mv.malom_dtw_after)), col);
      sprite.position.set(0, 0.6, 0);
      ring.add(sprite);
    }
    malomGroup.add(ring);
  }
}

const malomToggle = document.getElementById('malom-toggle');
if (malomToggle) {
  malomToggle.addEventListener('change', () => {
    malomGroup.visible = malomToggle.checked;
  });
}

// ── Raycasting / hover / click ────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
let   hoveredBar = null;

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  const barHits = raycaster.intersectObjects(barGroup.children);
  if (hoveredBar) {
    hoveredBar.material.color.copy(hoveredBar.userData.baseColor);
    hoveredBar.material.opacity = 0.88;
    hoveredBar = null;
    tooltip.style.display = 'none';
  }
  if (barHits.length > 0) {
    const mesh = barHits[0].object;
    hoveredBar = mesh;
    mesh.material.color.set(C.barHov);
    mesh.material.opacity = 1.0;
    showTooltip(e.clientX, e.clientY, mesh.userData.moveData);
    document.querySelectorAll('.move-item').forEach(el =>
      el.classList.toggle('highlighted', el.dataset.notation === mesh.userData.notation));
  } else {
    document.querySelectorAll('.move-item.highlighted').forEach(el => el.classList.remove('highlighted'));
  }
}

canvas.addEventListener('mousemove', onMouseMove);
canvas.addEventListener('click', () => {
  if (hoveredBar) applyMove(hoveredBar.userData.notation);
});

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTooltip(cx, cy, mv) {
  const sentText = mv.sentinel_score != null
    ? `${(mv.sentinel_score * 100).toFixed(1)}%`
    : '—';
  const heurText = mv.heuristic_score != null
    ? (mv.heuristic_score >= 0 ? '+' : '') + mv.heuristic_score
    : '—';

  let dbRows = '';
  if (mv.has_db_data) {
    const wdlText = mv.malom_wdl_after
      ? `${mv.malom_wdl_after}${mv.malom_dtw_after != null ? ` (${mv.malom_dtw_after > 0 ? '+' : ''}${mv.malom_dtw_after} DTW)` : ''}`
      : '—';
    dbRows = `
    <div class="tt-row"><span class="tt-label">Win%</span><span>${(mv.win_pct*100).toFixed(1)}%</span></div>
    <div class="tt-row"><span class="tt-label">W/D/L</span><span>${mv.wins}/${mv.draws}/${mv.losses}</span></div>
    <div class="tt-row"><span class="tt-label">Games</span><span>${mv.total}</span></div>
    <div class="tt-row"><span class="tt-label">Avg plies left</span><span>${mv.avg_moves_to_end.toFixed(0)}</span></div>
    <div class="tt-row"><span class="tt-label">Malom (after)</span><span>${wdlText}</span></div>`;
  }

  tooltip.innerHTML = `
    <div class="tt-notation">${mv.notation}</div>
    ${dbRows}
    <div class="tt-row"><span class="tt-label">Sentinel</span><span>${sentText}</span></div>
    <div class="tt-row"><span class="tt-label">Heuristic</span><span>${heurText}</span></div>
  `;
  const wr = wrap.getBoundingClientRect();
  let tx = cx - wr.left + 14;
  let ty = cy - wr.top  - 10;
  if (tx + 180 > wr.width)  tx = cx - wr.left - 180;
  if (ty + 160 > wr.height) ty = cy - wr.top  - 160;
  tooltip.style.left    = tx + 'px';
  tooltip.style.top     = ty + 'px';
  tooltip.style.display = 'block';
}

// ── Side-panel updates ────────────────────────────────────────────────────────

function updatePanel(data) {
  const badge = document.getElementById('turn-badge');
  badge.textContent = data.turn === 'W' ? 'White' : 'Black';
  badge.className   = 'turn-badge ' + (data.turn === 'W' ? 'white' : 'black');

  const posEl = document.getElementById('pos-stats');
  const ps    = data.position_stats;
  if (!ps) {
    posEl.innerHTML = '<div id="no-data-notice">No HumanDB data for this position.</div>';
  } else {
    const tot = Math.max(1, ps.total_games);
    const wp  = (ps.wins   / tot * 100).toFixed(1);
    const dp  = (ps.draws  / tot * 100).toFixed(1);
    const lp  = (ps.losses / tot * 100).toFixed(1);
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

  const listEl = document.getElementById('move-list');
  listEl.innerHTML = '';
  if (data.moves && data.moves.length > 0) {
    for (const mv of data.moves) {
      const col    = barColor(mv);
      const colHex = '#' + col.getHexString();

      let rightContent = '';
      if (mv.has_db_data) {
        const wdl = mv.malom_wdl_after
          ? `<span class="move-malom" style="background:${mv.malom_wdl_after==='L'?'#16532a':mv.malom_wdl_after==='W'?'#7f1d1d':'#78350f'};color:${mv.malom_wdl_after==='L'?'#4ade80':mv.malom_wdl_after==='W'?'#fca5a5':'#fcd34d'}">${mv.malom_wdl_after}${mv.malom_dtw_after!=null?' '+mv.malom_dtw_after:''}</span>`
          : '';
        rightContent = `
          <span class="move-sub">${mv.total}</span>
          ${wdl}
          <span class="move-pct" style="color:${colHex}">${(mv.win_pct*100).toFixed(1)}%</span>
        `;
      } else {
        const heurStr = (mv.heuristic_score >= 0 ? '+' : '') + mv.heuristic_score;
        const sentStr = mv.sentinel_score != null
          ? `<span class="move-sentinel">${(mv.sentinel_score*100).toFixed(0)}%</span>`
          : '';
        rightContent = `
          <span class="move-heuristic">${heurStr}</span>
          ${sentStr}
        `;
      }

      const item = document.createElement('div');
      item.className = 'move-item' + (mv.has_db_data ? '' : ' move-item-no-db');
      item.dataset.notation = mv.notation;
      item.innerHTML = `
        <div class="move-bar-swatch" style="background:${colHex}"></div>
        <span class="move-notation">${mv.notation}</span>
        ${rightContent}
      `;
      item.addEventListener('click', () => applyMove(mv.notation));
      item.addEventListener('mouseenter', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) { entry.mesh.material.color.set(C.barHov); entry.mesh.material.opacity = 1; }
      });
      item.addEventListener('mouseleave', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) {
          entry.mesh.material.color.copy(entry.mesh.userData.baseColor);
          entry.mesh.material.opacity = mv.has_db_data ? 0.88 : 0.55;
        }
      });
      listEl.appendChild(item);
    }
  } else {
    listEl.innerHTML = '<div style="padding:0.5rem 0.75rem;color:#8a7a5a;font-size:0.8rem">No move data.</div>';
  }

  const lineEl = document.getElementById('winning-line');
  lineEl.textContent = data.winning_line && data.winning_line.length > 0
    ? data.winning_line.join(' → ') : '—';
}

// ── Navigation ────────────────────────────────────────────────────────────────

const history = [];
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
    rebuildArrows(data.moves || []);
    rebuildMalomOverlay(data.moves || []);
    updatePanel(data);
    document.getElementById('btn-back').disabled = history.length === 0;
    const backLink = document.querySelector('a.btn-back');
    if (backLink && data.fen) backLink.href = '/?setup_fen=' + encodeURIComponent(data.fen);
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
  loadPosition(history.pop());
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
  labelRenderer.setSize(w, h);
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
  labelRenderer.render(scene, camera);
}
animate();

// ── Boot ──────────────────────────────────────────────────────────────────────

const _urlFen = new URLSearchParams(window.location.search).get('fen');
loadPosition(_urlFen || '........................|W|0|0');
