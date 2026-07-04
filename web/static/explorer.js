/**
 * explorer.js — 3D NMM position explorer (Three.js ES module)
 *
 * Bar height = how often humans played that move (most-played = tallest).
 * Bar color  = human win-rate gradient: green (winning) → orange → red (losing).
 * Arrows     = cylinder+cone from the piece's current square to its destination.
 * Malom overlay (toggle) = colored rings + DTW numbers on candidate squares.
 *
 * Interaction state machine:
 *   Place phase : click bar → if all variants need capture enter capture mode, else apply.
 *   Move/fly    : click own piece → piece_selected; click destination bar → apply or capture.
 *   Capture mode: click red opponent piece → complete move; click empty / Esc → cancel.
 *
 * Hint rings on board:
 *   Gold  — HumanDB best destination (highest win%)
 *   Blue  — Sentinel best destination (highest sentinel_score)
 *   Red   — Capturable opponent piece in capture mode
 *   Gold  — Sentinel's recommended capture (overrides red for that piece)
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

function winPctColor(pct) {
  const t = Math.max(0, Math.min(1, pct));
  if (t < 0.5) {
    const u = t * 2;
    return new THREE.Color().setRGB(0.94, 0.24 + 0.38 * u, 0.07);
  }
  const u = (t - 0.5) * 2;
  return new THREE.Color().setRGB(0.94 - 0.8 * u, 0.62 + 0.14 * u, 0.07);
}

function sentinelColor(score) {
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
  ['a','b','c','d','e','f','g'].forEach((letter, i) => {
    const el = document.createElement('div');
    el.className = 'coord-label';
    el.textContent = letter;
    const obj = new CSS2DObject(el);
    obj.position.set(i - 3, 0.5, 4.5);
    scene.add(obj);
  });

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
const hintGroup  = new THREE.Group();
malomGroup.visible = false;
scene.add(pieceGroup, barGroup, arrowGroup, malomGroup, hintGroup);

const pieceGeoW = new THREE.CylinderGeometry(0.26, 0.26, 0.4, 20);
const pieceGeoB = new THREE.CylinderGeometry(0.26, 0.26, 0.4, 20);

// ── Interaction state machine ─────────────────────────────────────────────────

let selectionState      = 'idle'; // 'idle' | 'piece_selected' | 'capture'
let selectedPieceSq     = null;   // own piece square when piece_selected
let pendingCaptureMoves = [];     // move variants waiting for capture-sq pick
let captureReturnState  = 'idle'; // state to restore on capture cancel
let currentPhase        = 'place';
let currentTurn         = 'W';

// ── Piece rebuild — per-piece materials for individual highlighting ────────────

function rebuildPieces(boardDict) {
  pieceGroup.clear();
  for (const [pos, piece] of Object.entries(boardDict)) {
    if (!piece || !POS_COORDS[pos]) continue;
    const [x,, z] = POS_COORDS[pos];
    const geo  = piece === 'W' ? pieceGeoW : pieceGeoB;
    const base = new THREE.Color(piece === 'W' ? C.white : C.black);
    const mat  = new THREE.MeshLambertMaterial({ color: base.clone(), transparent: true, opacity: 1.0 });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(x, 0.23, z);
    mesh.castShadow = true;
    mesh.userData.pos       = pos;
    mesh.userData.color     = piece;
    mesh.userData.baseColor = base;
    pieceGroup.add(mesh);
  }
}

// ── Bar rebuild — deduplicated by to_sq ───────────────────────────────────────

const barMeshMap  = new Map(); // notation → { mesh, data }
const barGroupMap = new Map(); // toSq → [meshes]  (all segments for a destination)
const MAX_BAR_HEIGHT = 0.55;
const BAR_W          = 0.14;
const BAR_OFFSET_X   = 0.38;  // trajectory bar: beside piece to the right
const SENT_OFFSET_X  = 0.62;  // sentinel bar: further right
const SENT_W         = 0.10;

function _addBarMesh(barX, z, segH, yBot, colHex, opacity, rep, mvsForSq, needsCapture, toSq) {
  const mat  = new THREE.MeshLambertMaterial({ color: colHex, transparent: true, opacity });
  const geom = new THREE.BoxGeometry(BAR_W, segH, BAR_W);
  const mesh = new THREE.Mesh(geom, mat);
  mesh.position.set(barX, yBot + segH / 2, z);
  mesh.castShadow = true;
  mesh.userData.notation     = rep.notation;
  mesh.userData.moveData     = rep;
  mesh.userData.allMoves     = mvsForSq;
  mesh.userData.needsCapture = needsCapture;
  mesh.userData.toSq         = toSq;
  mesh.userData.baseColor    = new THREE.Color(colHex);
  mesh.userData.baseOpacity  = opacity;
  barGroup.add(mesh);
  return mesh;
}

function rebuildBars(movesArray) {
  barGroup.clear();
  barMeshMap.clear();
  barGroupMap.clear();
  if (!movesArray || movesArray.length === 0) return;

  // Group by to_sq so mill-closing positions show one bar per destination
  const byToSq = new Map();
  for (const mv of movesArray) {
    if (!mv.to_sq || !POS_COORDS[mv.to_sq]) continue;
    if (!byToSq.has(mv.to_sq)) byToSq.set(mv.to_sq, []);
    byToSq.get(mv.to_sq).push(mv);
  }

  const dbMoves  = movesArray.filter(m => m.has_db_data);
  const maxTotal = Math.max(1, ...dbMoves.map(m => m.total || 0));
  const hScores  = movesArray.filter(m => !m.has_db_data).map(m => m.heuristic_score);
  const minH     = hScores.length ? Math.min(...hScores) : 0;
  const maxH     = hScores.length ? Math.max(...hScores) : 1;
  const hRange   = Math.max(1, maxH - minH);

  for (const [toSq, mvsForSq] of byToSq) {
    const rep          = mvsForSq.find(m => !m.capture_sq) || mvsForSq[0];
    const needsCapture = mvsForSq.every(m => m.capture_sq != null);
    const [x,, z]      = POS_COORDS[toSq];
    const baseY        = 0.07;
    const segMeshes    = [];

    // ── Trajectory bar (W/D/L stacked segments, beside piece) ──
    const barX = x + BAR_OFFSET_X;
    if (rep.has_db_data) {
      const totalForSq = mvsForSq.reduce((s, m) => s + (m.total || 0), 0);
      const barH  = Math.max(0.06, (totalForSq / maxTotal) * MAX_BAR_HEIGHT);
      const wins   = rep.wins   || 0;
      const draws  = rep.draws  || 0;
      const losses = rep.losses || 0;
      const total  = Math.max(1, wins + draws + losses);

      const lossH = barH * (losses / total);
      const drawH = barH * (draws  / total);
      const winH  = barH * (wins   / total);

      const segs = [
        { h: lossH, col: 0xef4444, yBot: baseY },
        { h: drawH, col: 0xa06040, yBot: baseY + lossH },
        { h: winH,  col: 0x4ade80, yBot: baseY + lossH + drawH },
      ];
      let primaryMesh = null;
      for (const seg of segs) {
        if (seg.h < 0.005) continue;
        const m = _addBarMesh(barX, z, seg.h, seg.yBot, seg.col, 0.88, rep, mvsForSq, needsCapture, toSq);
        segMeshes.push(m);
        if (!primaryMesh) primaryMesh = m;
      }
      if (primaryMesh) {
        for (const mv of mvsForSq) barMeshMap.set(mv.notation, { mesh: primaryMesh, data: mv });
      }
    } else {
      const norm  = (rep.heuristic_score - minH) / hRange;
      const barH  = 0.06 + norm * (MAX_BAR_HEIGHT * 0.5);
      const m = _addBarMesh(barX, z, barH, baseY, 0x888888, 0.55, rep, mvsForSq, needsCapture, toSq);
      segMeshes.push(m);
      for (const mv of mvsForSq) barMeshMap.set(mv.notation, { mesh: m, data: mv });
    }

    // ── Sentinel bar (blue, separate column further right) ──
    const sentScore = rep.sentinel_score;
    if (sentScore != null && sentScore > 0.01) {
      const sentH = Math.max(0.04, sentScore * MAX_BAR_HEIGHT);
      const sm = new THREE.Mesh(
        new THREE.BoxGeometry(SENT_W, sentH, SENT_W),
        new THREE.MeshLambertMaterial({ color: 0x5595d4, transparent: true, opacity: 0.85 }),
      );
      sm.position.set(x + SENT_OFFSET_X, baseY + sentH / 2, z);
      sm.castShadow = true;
      sm.userData.notation     = rep.notation;
      sm.userData.moveData     = rep;
      sm.userData.allMoves     = mvsForSq;
      sm.userData.needsCapture = needsCapture;
      sm.userData.toSq         = toSq;
      sm.userData.baseColor    = new THREE.Color(0x5595d4);
      sm.userData.baseOpacity  = 0.85;
      barGroup.add(sm);
      segMeshes.push(sm);
    }

    barGroupMap.set(toSq, segMeshes);
  }
}

// ── Move arrows (from_sq → to_sq) ────────────────────────────────────────────

const _up = new THREE.Vector3(0, 1, 0);

function rebuildArrows(movesArray) {
  arrowGroup.clear();
  if (!movesArray || movesArray.length === 0) return;
  const dbMoves  = movesArray.filter(m => m.has_db_data);
  const maxTotal = Math.max(1, ...dbMoves.map(m => m.total || 0));

  // Deduplicate arrows by (from_sq, to_sq)
  const seen = new Set();
  for (const mv of movesArray) {
    if (!mv.from_sq || !POS_COORDS[mv.from_sq] || !POS_COORDS[mv.to_sq]) continue;
    const key = `${mv.from_sq}-${mv.to_sq}`;
    if (seen.has(key)) continue;
    seen.add(key);

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
malomRingGeo.rotateX(-Math.PI / 2);

function rebuildMalomOverlay(movesArray) {
  malomGroup.clear();
  if (!movesArray) return;
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
      const labelCol = mv.malom_wdl_after === 'L' ? '#4ade80'
                     : mv.malom_wdl_after === 'W' ? '#fca5a5'
                     : '#fcd34d';
      const sprite = makeDtwSprite(String(Math.abs(mv.malom_dtw_after)), labelCol);
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

// ── Hint rings (HumanDB gold, Sentinel blue, capture red/gold) ────────────────

const hintRingGeo = new THREE.RingGeometry(0.34, 0.52, 24);
hintRingGeo.rotateX(-Math.PI / 2);

function makeHintRing(sq, hexColor) {
  if (!POS_COORDS[sq]) return null;
  const mat  = new THREE.MeshBasicMaterial({ color: hexColor, transparent: true, opacity: 0.9, side: THREE.DoubleSide });
  const ring = new THREE.Mesh(hintRingGeo, mat);
  const [x,, z] = POS_COORDS[sq];
  ring.position.set(x, 0.12, z);
  return ring;
}

function rebuildHints(allMoves) {
  hintGroup.clear();
  if (!allMoves) return;

  if (selectionState === 'capture') {
    // Red rings on all capturable squares; gold on sentinel's top capture pick
    const capSqs = new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean));
    const bestSent = [...pendingCaptureMoves]
      .filter(m => m.capture_sq && m.sentinel_score != null)
      .sort((a, b) => b.sentinel_score - a.sentinel_score)[0];
    const sentCapSq = bestSent?.capture_sq ?? null;
    for (const sq of capSqs) {
      const ring = makeHintRing(sq, sq === sentCapSq ? 0xffd700 : 0xff3333);
      if (ring) hintGroup.add(ring);
    }
    return;
  }

  // idle / piece_selected — HumanDB best (gold) and Sentinel best (blue)
  const visibleMoves = (selectionState === 'piece_selected' && selectedPieceSq)
    ? allMoves.filter(m => m.from_sq === selectedPieceSq)
    : allMoves;
  if (visibleMoves.length === 0) return;

  const dbMoves = visibleMoves.filter(m => m.has_db_data && m.win_pct != null);
  let humanBestSq = null;
  if (dbMoves.length > 0) {
    humanBestSq = [...dbMoves].sort((a, b) => b.win_pct - a.win_pct)[0].to_sq;
    const ring = makeHintRing(humanBestSq, 0xffd700);
    if (ring) hintGroup.add(ring);
  }

  const sentMoves = visibleMoves.filter(m => m.sentinel_score != null);
  if (sentMoves.length > 0) {
    const sentBestSq = [...sentMoves].sort((a, b) => b.sentinel_score - a.sentinel_score)[0].to_sq;
    if (sentBestSq !== humanBestSq) {
      const ring = makeHintRing(sentBestSq, 0x4488ff);
      if (ring) hintGroup.add(ring);
    }
  }
}

// ── Piece colour highlights ───────────────────────────────────────────────────

function updatePieceHighlights() {
  const capSqs = (selectionState === 'capture')
    ? new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean))
    : new Set();

  let sentCapSq = null;
  if (selectionState === 'capture') {
    const best = [...pendingCaptureMoves]
      .filter(m => m.capture_sq && m.sentinel_score != null)
      .sort((a, b) => b.sentinel_score - a.sentinel_score)[0];
    sentCapSq = best?.capture_sq ?? null;
  }

  for (const mesh of pieceGroup.children) {
    const { pos, color: pc, baseColor } = mesh.userData;
    if (selectionState === 'capture') {
      if (capSqs.has(pos)) {
        mesh.material.color.setHex(pos === sentCapSq ? 0xffd700 : 0xee2222);
        mesh.material.opacity = 1.0;
      } else {
        mesh.material.color.copy(baseColor);
        mesh.material.opacity = 0.45;
      }
    } else if (selectionState === 'piece_selected' && pos === selectedPieceSq) {
      mesh.material.color.setHex(0xffd700);
      mesh.material.opacity = 1.0;
    } else {
      mesh.material.color.copy(baseColor);
      mesh.material.opacity = 1.0;
    }
  }
}

// ── Selection state helpers ───────────────────────────────────────────────────

function filterMovesForState() {
  if (!currentData) return [];
  const all = currentData.moves || [];
  if (selectionState === 'capture') return [];
  if (selectionState === 'piece_selected' && selectedPieceSq) {
    return all.filter(mv => mv.from_sq === selectedPieceSq);
  }
  // idle + move/fly phase: no bars until a piece is selected
  if (currentPhase !== 'place' && selectionState === 'idle') return [];
  return all;
}

function updateStatusIndicator() {
  const el = document.getElementById('selection-status');
  if (!el) return;
  if (currentPhase === 'place' && selectionState === 'idle') {
    el.style.display = 'none';
    return;
  }
  let msg;
  if (selectionState === 'capture') {
    msg = '▶ Click an opponent piece to capture  ·  Esc or click empty to cancel';
  } else if (selectionState === 'piece_selected') {
    msg = `▶ ${selectedPieceSq} selected — click a destination bar`;
  } else {
    msg = '▶ Click one of your pieces to move';
  }
  el.textContent = msg;
  el.style.display = 'block';
}

function _refreshAfterStateChange() {
  const filtered = filterMovesForState();
  rebuildBars(filtered);
  rebuildArrows(filtered);
  rebuildHints(currentData?.moves || []);
  updatePieceHighlights();
  updateStatusIndicator();
}

// ── Raycasting / hover / click ────────────────────────────────────────────────

const raycaster = new THREE.Raycaster();
const mouse     = new THREE.Vector2();
let   hoveredBar   = null;
let   hoveredPiece = null;

function onMouseMove(e) {
  const rect = canvas.getBoundingClientRect();
  mouse.x =  ((e.clientX - rect.left) / rect.width)  * 2 - 1;
  mouse.y = -((e.clientY - rect.top)  / rect.height) * 2 + 1;
  raycaster.setFromCamera(mouse, camera);

  // ── Bar hover ──
  if (hoveredBar) {
    for (const m of (barGroupMap.get(hoveredBar.userData.toSq) || [hoveredBar])) {
      m.material.color.copy(m.userData.baseColor);
      m.material.opacity = m.userData.baseOpacity ?? (m.userData.moveData?.has_db_data ? 0.88 : 0.55);
    }
    hoveredBar = null;
    tooltip.style.display = 'none';
  }
  const barHits = raycaster.intersectObjects(barGroup.children);
  if (barHits.length > 0) {
    const mesh = barHits[0].object;
    hoveredBar = mesh;
    for (const m of (barGroupMap.get(mesh.userData.toSq) || [mesh])) {
      m.material.color.set(C.barHov);
      m.material.opacity = 1.0;
    }
    showTooltip(e.clientX, e.clientY, mesh.userData.moveData);
    document.querySelectorAll('.move-item').forEach(el =>
      el.classList.toggle('highlighted', el.dataset.notation === mesh.userData.notation));
  } else {
    document.querySelectorAll('.move-item.highlighted').forEach(el => el.classList.remove('highlighted'));
  }

  // ── Piece hover ──
  const prevHoveredPiece = hoveredPiece;
  hoveredPiece = null;

  const pieceTargets = [];
  if (selectionState === 'capture') {
    const capSqs = new Set(pendingCaptureMoves.map(m => m.capture_sq).filter(Boolean));
    for (const mesh of pieceGroup.children) {
      if (capSqs.has(mesh.userData.pos)) pieceTargets.push(mesh);
    }
  } else if (currentPhase !== 'place') {
    // Own pieces are hoverable in move/fly phase
    for (const mesh of pieceGroup.children) {
      if (mesh.userData.color === currentTurn) pieceTargets.push(mesh);
    }
  }

  if (pieceTargets.length > 0) {
    const hits = raycaster.intersectObjects(pieceTargets);
    if (hits.length > 0) hoveredPiece = hits[0].object;
  }

  if (prevHoveredPiece !== hoveredPiece) {
    updatePieceHighlights();
    if (hoveredPiece) {
      hoveredPiece.material.color.setHex(0xffd700);
      hoveredPiece.material.opacity = 1.0;
    }
  }

  canvas.style.cursor = (hoveredBar || hoveredPiece) ? 'pointer' : 'default';
}

canvas.addEventListener('mousemove', onMouseMove);

canvas.addEventListener('click', () => {
  // ── Capture mode ──
  if (selectionState === 'capture') {
    if (hoveredPiece) {
      const sq = hoveredPiece.userData.pos;
      const mv = pendingCaptureMoves.find(m => m.capture_sq === sq);
      if (mv) {
        selectionState      = 'idle';
        selectedPieceSq     = null;
        pendingCaptureMoves = [];
        applyMove(mv.notation);
        return;
      }
    }
    // Cancel: click on non-capturable area
    selectionState = captureReturnState;
    if (selectionState !== 'piece_selected') selectedPieceSq = null;
    pendingCaptureMoves = [];
    _refreshAfterStateChange();
    return;
  }

  // ── Piece selected ──
  if (selectionState === 'piece_selected') {
    if (hoveredBar) {
      const { allMoves, needsCapture } = hoveredBar.userData;
      if (needsCapture) {
        pendingCaptureMoves = allMoves;
        captureReturnState  = 'piece_selected';
        selectionState      = 'capture';
        _refreshAfterStateChange();
      } else {
        const notation = allMoves.find(m => !m.capture_sq)?.notation ?? allMoves[0].notation;
        selectionState  = 'idle';
        selectedPieceSq = null;
        applyMove(notation);
      }
      return;
    }
    if (hoveredPiece) {
      const sq = hoveredPiece.userData.pos;
      if (sq === selectedPieceSq) {
        // Deselect
        selectionState  = 'idle';
        selectedPieceSq = null;
      } else if (hoveredPiece.userData.color === currentTurn) {
        // Switch to different own piece
        selectedPieceSq = sq;
      }
      _refreshAfterStateChange();
      return;
    }
    // Click on empty space — deselect
    selectionState  = 'idle';
    selectedPieceSq = null;
    _refreshAfterStateChange();
    return;
  }

  // ── Idle ──
  if (currentPhase === 'place') {
    if (hoveredBar) {
      const { allMoves, needsCapture } = hoveredBar.userData;
      if (needsCapture) {
        pendingCaptureMoves = allMoves;
        captureReturnState  = 'idle';
        selectionState      = 'capture';
        _refreshAfterStateChange();
      } else {
        const notation = allMoves.find(m => !m.capture_sq)?.notation ?? allMoves[0].notation;
        applyMove(notation);
      }
    }
  } else {
    // move / fly phase — require piece click first
    if (hoveredPiece && hoveredPiece.userData.color === currentTurn) {
      selectedPieceSq = hoveredPiece.userData.pos;
      selectionState  = 'piece_selected';
      _refreshAfterStateChange();
    }
  }
});

// Escape key cancels any active selection
window.addEventListener('keydown', e => {
  if (e.key === 'Escape' && selectionState !== 'idle') {
    if (selectionState === 'capture') {
      selectionState = captureReturnState;
      if (selectionState !== 'piece_selected') selectedPieceSq = null;
      pendingCaptureMoves = [];
    } else {
      selectionState  = 'idle';
      selectedPieceSq = null;
    }
    _refreshAfterStateChange();
  }
});

// ── Tooltip ───────────────────────────────────────────────────────────────────

function showTooltip(cx, cy, mv) {
  if (!mv) return;
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
        if (entry) {
          for (const m of (barGroupMap.get(entry.mesh.userData.toSq) || [entry.mesh])) {
            m.material.color.set(C.barHov); m.material.opacity = 1;
          }
        }
      });
      item.addEventListener('mouseleave', () => {
        const entry = barMeshMap.get(mv.notation);
        if (entry) {
          for (const m of (barGroupMap.get(entry.mesh.userData.toSq) || [entry.mesh])) {
            m.material.color.copy(m.userData.baseColor);
            m.material.opacity = m.userData.baseOpacity ?? (mv.has_db_data ? 0.88 : 0.55);
          }
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
  // Reset all selection state for new position
  selectionState      = 'idle';
  selectedPieceSq     = null;
  pendingCaptureMoves = [];
  captureReturnState  = 'idle';
  hoveredBar          = null;
  hoveredPiece        = null;
  tooltip.style.display = 'none';

  try {
    const res  = await fetch('/api/explorer/position?fen=' + encodeURIComponent(fen));
    const data = await res.json();
    if (data.error) { alert('Error: ' + data.error); return; }
    currentData  = data;
    currentTurn  = data.turn;
    currentPhase = data.phase || 'move';
    document.getElementById('fen-input').value = data.fen;
    rebuildPieces(data.board);
    const initialMoves = filterMovesForState();
    rebuildBars(initialMoves);
    rebuildArrows(initialMoves);
    rebuildMalomOverlay(data.moves || []);
    rebuildHints(data.moves || []);
    updatePanel(data);
    updatePieceHighlights();
    updateStatusIndicator();
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
