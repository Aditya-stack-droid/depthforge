"use strict";
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

const state = { sample: null, rgb: null, depth: null, scale: 4, cond: "hard", method: "hybrid", kind: "gt",
  cfg: null, last: null, busy: false };

/* ---------- toasts ---------- */
function toast(msg, ok = false) {
  const t = document.createElement("div");
  t.className = "toast" + (ok ? " ok" : ""); t.textContent = msg;
  $("#toasts").append(t); setTimeout(() => t.remove(), 6500);
}

/* ---------- segmented controls ---------- */
function seg(root, onPick) {
  root.addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b || !root.contains(b)) return;
    $$("button", root).forEach(x => x.classList.toggle("on", x === b)); onPick(b.dataset.v);
  });
}
function setSeg(root, v) { $$("button", root).forEach(b => b.classList.toggle("on", b.dataset.v === String(v))); }

/* ---------- boot ---------- */
async function boot() {
  const [cfg, res] = await Promise.all([fetch("/api/config").then(r => r.json()), fetch("/api/results").then(r => r.json()).catch(() => ({}))]);
  state.cfg = cfg;

  const box = $("#samples");
  cfg.samples.forEach(name => {
    const b = document.createElement("button");
    b.dataset.name = name; b.title = name.replace(/_/g, " "); b.setAttribute("role", "option");
    b.innerHTML = `<img alt="${esc(name)}" loading="lazy" src="/api/thumb/${encodeURIComponent(name)}">`;
    b.onclick = () => pickSample(name);
    box.append(b);
  });

  const sm = $("#segMethod");
  cfg.methods.forEach(m => { const b = document.createElement("button"); b.dataset.v = m; b.textContent = m; if (m === state.method) b.classList.add("on"); sm.append(b); });
  seg(sm, v => state.method = v);
  seg($("#segScale"), v => state.scale = +v);
  seg($("#segKind"), v => state.kind = v);

  const sel = $("#cond");
  cfg.conditions.forEach(c => sel.add(new Option(c.key, c.key)));
  sel.value = state.cond;
  const info = () => $("#condInfo").textContent = (cfg.conditions.find(c => c.key === state.cond) || {}).info || "";
  sel.onchange = () => { state.cond = sel.value; info(); }; info();

  cfg.stages.forEach(s => {
    const l = document.createElement("label"); l.className = "tg";
    l.innerHTML = `<span>${esc(s.label)}</span><input type="checkbox" data-k="${s.key}" data-def="${s.default ? 1 : 0}" ${s.default ? "checked" : ""}>`;
    $("#stages").append(l);
  });

  $("#presets").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    const [sc, cd, me] = b.dataset.p.split(",");
    state.scale = +sc; state.cond = cd; state.method = me;
    setSeg($("#segScale"), sc); setSeg(sm, me); sel.value = cd; info();
  });

  dropzone("#dropRgb", "rgb"); dropzone("#dropDepth", "depth");
  $("#run").onclick = run;
  $("#cmpAll").onclick = compareAll;

  const mark = () => $$("#themes button").forEach(b => b.classList.toggle("on", b.dataset.t === document.documentElement.dataset.theme));
  $("#themes").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    document.documentElement.dataset.theme = b.dataset.t;
    try { localStorage.setItem("df-theme", b.dataset.t); } catch { /* private mode */ }
    mark();
  });
  mark();
  tabs(); slider(); viewer3d(); results(res);
  const first = cfg.samples.find(s => s.startsWith("nyu_test_1")) || cfg.samples[0];
  if (first) { pickSample(first, true); run(); }
}

function pickSample(name) {
  state.sample = name; state.rgb = state.depth = null;
  $$("#samples button").forEach(b => b.classList.toggle("on", b.dataset.name === name));
  $$(".drop").forEach(d => { d.classList.remove("has"); const s = $(".drop-s", d); s.textContent = s.dataset.empty; });
}

function dropzone(sel, key) {
  const z = $(sel), inp = $("input", z);
  const take = f => {
    if (!f) return; state[key] = f; state.sample = null;
    $$("#samples button").forEach(b => b.classList.remove("on"));
    z.classList.add("has"); $(".drop-s", z).textContent = f.name;
  };
  inp.onchange = () => take(inp.files[0]);
  ["dragenter", "dragover"].forEach(ev => z.addEventListener(ev, e => { e.preventDefault(); z.classList.add("over"); }));
  ["dragleave", "drop"].forEach(ev => z.addEventListener(ev, e => { e.preventDefault(); z.classList.remove("over"); }));
  z.addEventListener("drop", e => take(e.dataTransfer.files[0]));
}

/* ---------- request ---------- */
function formData(extra = {}) {
  const fd = new FormData();
  if (state.sample) fd.append("sample", state.sample);
  else if (state.rgb && state.depth) { fd.append("rgb", state.rgb); fd.append("depth", state.depth); }
  else throw new Error("Choose a sample, or upload both an RGB image and a depth map.");
  fd.append("scale", state.scale); fd.append("condition", state.cond);
  Object.entries(extra).forEach(([k, v]) => fd.append(k, v));
  return fd;
}
async function post(url, fd) {
  const r = await fetch(url, { method: "POST", body: fd });
  let j = {}; try { j = await r.json(); } catch { /* non-JSON error */ }
  if (!r.ok) throw new Error(j.error || j.detail && JSON.stringify(j.detail) || `Server error (${r.status})`);
  return j;
}
function busy(on, msg) {
  state.busy = on; $("#run").classList.toggle("busy", on); $("#run").disabled = on; $("#loadbar").classList.toggle("on", on);
  if (msg) $("#status").textContent = msg;
}

async function run() {
  if (state.busy) return;
  let fd;
  try {
    const off = $$("#stages input").filter(i => i.dataset.def === "1" && !i.checked).map(i => i.dataset.k);
    const on = $$("#stages input").filter(i => i.dataset.def === "0" && i.checked).map(i => i.dataset.k);
    fd = formData({ method: state.method, kind: state.kind, off: off.join(","), on: on.join(",") });
  } catch (e) { toast(e.message); return; }
  busy(true, "Running… the first run after idle can take about 10 s");
  try {
    const d = await post("/api/run", fd);
    state.last = d; render(d);
    $("#status").textContent = `Done in ${(d.latency_ms / 1000).toFixed(1)} s · output ${d.size[0]}×${d.size[1]}`;
  } catch (e) { $("#status").textContent = "Failed."; toast(e.message); }
  finally { busy(false); }
}

/* ---------- render ---------- */
const NAMES = { rgb: "RGB guide", lr: "Low-res input", bicubic: "Bicubic", gt: "Ground truth", result: "DepthForge", error: "Error (DepthForge)", bic_error: "Error (bicubic)" };
function render(d) {
  const L = ["bicubic", "lr", "rgb", ...(d.has_gt ? ["gt", "bic_error"] : [])];
  const R = ["result", ...(d.has_gt ? ["gt", "error"] : []), "rgb"];
  fill($("#selLeft"), L, "bicubic"); fill($("#selRight"), R, "result");
  $("#cmp").style.aspectRatio = d.size[0] / d.size[1];
  setCompare(); $("#empty").classList.add("off");

  const m = d.metrics, chips = [];
  if (d.has_gt) {
    const gain = 100 * (1 - m.rmse / m.rmse_bicubic);
    chips.push(["RMSE", m.rmse.toFixed(4) + " m"], ["vs bicubic", (gain >= 0 ? "−" : "+") + Math.abs(gain).toFixed(1) + "%", gain > 0],
      ["MAE", m.mae.toFixed(4) + " m"], ["PSNR", m.psnr.toFixed(1) + " dB"], ["SSIM", m.ssim.toFixed(3)],
      ["Edge RMSE", m.edge_rmse.toFixed(4) + " m"], ["Boundary F1", m.boundary_f1.toFixed(3)]);
  }
  chips.push(["Latency", (d.latency_ms / 1000).toFixed(2) + " s"], ["Output", `${d.size[0]}×${d.size[1]}`]);
  $("#mchips").innerHTML = chips.map(([k, v, g]) => `<div class="mc${g ? " good" : ""}"><span>${k}</span><b>${esc(v)}</b></div>`).join("") +
    (d.has_gt ? "" : `<div class="mc" style="grid-column:1/-1"><span>Note</span><b style="font-size:13px;font-weight:500">No ground truth, so accuracy metrics are unavailable for real low-res input.</b></div>`);
  $("#decisions").innerHTML = d.decisions.map(x => `<li>${esc(x)}</li>`).join("");
  const dl = $("#dl"); dl.href = d.download; dl.hidden = false;

  $("#stageGrid").innerHTML = d.stages.map((s, i) => `<figure class="fig"><img src="${s.img}" alt="${esc(s.label)}" data-i="${i}"><figcaption>${esc(s.label)}</figcaption></figure>`).join("");
  load3d(d.view3d);
}
function fill(sel, keys, def) {
  const cur = sel.value; sel.innerHTML = "";
  keys.forEach(k => sel.add(new Option(NAMES[k], k))); sel.value = keys.includes(cur) ? cur : def;
}

/* ---------- tabs + lightbox ---------- */
function tabs() {
  $(".tabs").addEventListener("click", e => {
    const b = e.target.closest("button"); if (!b) return;
    $$(".tabs button").forEach(x => x.classList.toggle("on", x === b));
    $$(".pane").forEach(p => p.classList.toggle("on", p.dataset.pane === b.dataset.tab));
    if (b.dataset.tab === "view3d") V3.show();
  });
  $("#stageGrid").addEventListener("click", e => {
    const i = e.target.closest("img"); if (!i) return;
    const o = document.createElement("div"); o.className = "lb"; o.innerHTML = `<img src="${i.src}" alt="">`;
    o.onclick = () => o.remove(); document.body.append(o);
  });
  document.addEventListener("keydown", e => { if (e.key === "Escape") $$(".lb").forEach(x => x.remove()); });
}

/* ---------- before/after slider ---------- */
let pos = 0.5;
function slider() {
  const c = $("#cmp");
  const set = x => { pos = Math.max(0, Math.min(1, x)); $("#imgL").style.clipPath = `inset(0 ${(1 - pos) * 100}% 0 0)`; $("#handle").style.left = pos * 100 + "%"; };
  const at = e => { const r = c.getBoundingClientRect(); set((e.clientX - r.left) / r.width); };
  let drag = false;
  c.addEventListener("pointerdown", e => { drag = true; c.setPointerCapture(e.pointerId); at(e); });
  c.addEventListener("pointermove", e => { if (drag) at(e); });
  c.addEventListener("pointerup", () => drag = false);
  c.addEventListener("keydown", e => { if (e.key === "ArrowLeft") set(pos - .03); if (e.key === "ArrowRight") set(pos + .03); });
  $("#selLeft").onchange = $("#selRight").onchange = setCompare;
  set(.5);
}
function setCompare() {
  const d = state.last; if (!d) return;
  const l = $("#selLeft").value, r = $("#selRight").value;
  $("#imgL").src = d.images[l]; $("#imgR").src = d.images[r];
  $("#capL").textContent = NAMES[l]; $("#capR").textContent = NAMES[r];
}

/* ---------- all methods ---------- */
async function compareAll() {
  const b = $("#cmpAll"); b.disabled = true; b.textContent = "Running…";
  try {
    if (state.kind !== "gt") throw new Error("Switch Advanced → Depth input type to “Simulate sensor” to compare against ground truth.");
    const d = await post("/api/compare", formData());
    const rows = d.rows.filter(r => r.rmse !== undefined), best = Math.min(...rows.map(r => r.rmse));
    $("#cmpOut").innerHTML = `<div class="tw"><table class="t"><thead><tr><th>Method</th><th>RMSE</th><th>MAE</th><th>PSNR</th><th>SSIM</th><th>Edge RMSE</th><th>Boundary F1</th><th>ms</th></tr></thead><tbody>` +
      rows.map(r => `<tr class="${r.rmse === best ? "best" : ""} ${r.method === "hybrid" ? "ours" : ""}"><td>${r.method}</td><td>${r.rmse.toFixed(4)}</td><td>${r.mae.toFixed(4)}</td><td>${r.psnr.toFixed(1)}</td><td>${r.ssim.toFixed(3)}</td><td>${r.edge_rmse.toFixed(4)}</td><td>${r.boundary_f1.toFixed(3)}</td><td>${Math.round(r.ms)}</td></tr>`).join("") +
      `</tbody></table></div><div class="grid" style="margin-top:16px">` +
      d.rows.map(r => `<figure class="fig"><img src="${r.img}" alt="${esc(r.method)}"><figcaption>${esc(r.method)}${r.rmse !== undefined ? " · RMSE " + r.rmse.toFixed(4) : ""}</figcaption></figure>`).join("") + `</div>`;
  } catch (e) { toast(e.message); }
  finally { b.disabled = false; b.textContent = "Compare all methods"; }
}

/* ---------- results chart ---------- */
function results(res) {
  const sec = $("#results"), conds = Object.keys(res).filter(c => res[c].bicubic && res[c].hybrid);
  if (!conds.length) { sec.hidden = true; $$('a[href="#results"]').forEach(a => a.hidden = true); return; }
  const cv_ = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const METH = [["bicubic", "Bicubic", "var(--c-bicubic)"], ["classical", "Classical", "var(--c-classical)"], ["dl", "Deep learning", "var(--c-dl)"], ["hybrid", "Hybrid (ours)", "var(--acc)"]];
  $("#legend").innerHTML = METH.map(([, n, c]) => `<span><i style="background:${c}"></i>${n}</span>`).join("");
  const sc = $("#segResCond");
  conds.forEach((c, i) => { const b = document.createElement("button"); b.dataset.v = c; b.textContent = c === "clean" ? "Clean input" : c === "hard" ? "Hard sensor (noise + holes)" : c; if (!i) b.classList.add("on"); sc.append(b); });
  let cur = conds[0]; seg(sc, c => draw(cur = c)); draw(cur);
  function draw(cond) {
    const scales = ["2", "4", "8"], W = 900, H = 300, L = 52, B = 34, T = 22, gw = (W - L) / 3;
    const val = (m, s) => (res[cond][m] && res[cond][m][s] || {}).rmse;
    const max = Math.max(...METH.flatMap(([m]) => scales.map(s => val(m, s) || 0))) * 1.12;
    const y = v => T + (H - T - B) * (1 - v / max), bw = Math.min(46, (gw - 40) / METH.length);
    let s = `<svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="xMidYMid meet">`;
    for (let i = 0; i <= 4; i++) { const v = max * i / 4; s += `<line x1="${L}" x2="${W}" y1="${y(v)}" y2="${y(v)}" style="stroke:var(--line)"/><text x="${L - 8}" y="${y(v) + 4}" text-anchor="end">${v.toFixed(3)}</text>`; }
    scales.forEach((sk, gi) => {
      const x0 = L + gi * gw + (gw - bw * METH.length - 6 * (METH.length - 1)) / 2;
      METH.forEach(([m, , col], k) => {
        const v = val(m, sk); if (v === undefined) return;
        const x = x0 + k * (bw + 6);
        s += `<rect x="${x}" y="${y(v)}" width="${bw}" height="${H - B - y(v)}" rx="4" style="fill:${col}"><title>${m} ${sk}×: ${v.toFixed(4)} m</title></rect><text class="v" x="${x + bw / 2}" y="${y(v) - 6}" text-anchor="middle">${v.toFixed(3)}</text>`;
      });
      s += `<text x="${L + gi * gw + gw / 2}" y="${H - 10}" text-anchor="middle" style="fill:var(--soft);font-size:13px">${sk}× upsampling</text>`;
    });
    $("#chart").innerHTML = s + "</svg>";
    const g = scales.map(sk => { const b = val("bicubic", sk), h = val("hybrid", sk); return b && h ? `${sk}×: ${(100 * (1 - h / b)).toFixed(0)}% lower` : ""; }).filter(Boolean).join(" · ");
    $("#resNote").textContent = `RMSE in metres. Hybrid vs bicubic: ${g}.`;
  }
}

/* ---------- 3D viewer (raw WebGL2, no libraries) ---------- */
const V3 = { gl: null, pay: null, D: [], dirty: false };
function viewer3d() {
  const cv = $("#v3");
  $("#v3src").onchange = $("#v3cut").oninput = () => V3.build && V3.build();
  let drag = 0, lx = 0, ly = 0;
  cv.onpointerdown = e => { drag = 1; lx = e.clientX; ly = e.clientY; $("#v3sway").checked = false; cv.setPointerCapture(e.pointerId); cv.style.cursor = "grabbing"; };
  cv.onpointerup = () => { drag = 0; cv.style.cursor = "grab"; };
  cv.onpointermove = e => { if (drag) { V3.yaw += (e.clientX - lx) * .008; V3.pitch = Math.max(-1.2, Math.min(1.2, V3.pitch + (e.clientY - ly) * .008)); lx = e.clientX; ly = e.clientY; } };
  cv.onwheel = e => { e.preventDefault(); V3.zoom = Math.max(.15, Math.min(3, V3.zoom * (e.deltaY > 0 ? 1.08 : .92))); };
  Object.assign(V3, { yaw: 0, pitch: 0, zoom: 1, t0: performance.now() });
}
function load3d(p) {
  V3.pay = p;
  V3.D = p.blobs.map(b => { const s = atob(b), u = new Uint8Array(s.length); for (let i = 0; i < s.length; i++) u[i] = s.charCodeAt(i); return new Uint16Array(u.buffer); });
  const sel = $("#v3src"); sel.innerHTML = ""; p.names.forEach((n, i) => sel.add(new Option(n, i)));
  V3.dirty = true; if ($('.pane[data-pane="view3d"]').classList.contains("on")) V3.show();
}
V3.show = function () {
  if (!V3.pay) return;
  const cv = $("#v3");
  if (!V3.gl) if (!initGL(cv)) { $("#v3wrap").innerHTML = '<div class="empty">WebGL2 is not available in this browser.</div>'; return; }
  if (V3.dirty) { V3.setup(); V3.dirty = false; }
  if (!V3.running) { V3.running = true; requestAnimationFrame(V3.loop); }
};
function initGL(cv) {
  const gl = cv.getContext("webgl2"); if (!gl) return false; V3.gl = gl;
  const vs = `#version 300 es
in vec3 p;in vec2 uv;uniform mat4 M;out vec2 vuv;out float vd;out vec3 vp;
void main(){vuv=uv;vd=-p.z;vp=p;gl_Position=M*vec4(p,1.);}`;
  const fs = `#version 300 es
precision highp float;in vec2 vuv;in float vd;in vec3 vp;uniform sampler2D t;uniform int mode;uniform vec2 rng;out vec4 o;
vec3 ramp(float x){x=clamp(x,0.,1.);return vec3(clamp(1.5-abs(4.*x-3.),0.,1.),clamp(1.5-abs(4.*x-2.),0.,1.),clamp(1.5-abs(4.*x-1.),0.,1.));}
void main(){if(mode==0)o=vec4(texture(t,vuv).rgb,1.);
else if(mode==1)o=vec4(ramp(1.-(vd-rng.x)/(rng.y-rng.x)),1.);
else{vec3 n=normalize(cross(dFdx(vp),dFdy(vp)));float l=abs(dot(n,normalize(vec3(.3,.5,1.))));o=vec4(vec3(.15+.85*l),1.);}}`;
  const sh = (t, s) => { const x = gl.createShader(t); gl.shaderSource(x, s); gl.compileShader(x); return x; };
  const pr = gl.createProgram(); gl.attachShader(pr, sh(gl.VERTEX_SHADER, vs)); gl.attachShader(pr, sh(gl.FRAGMENT_SHADER, fs)); gl.linkProgram(pr); gl.useProgram(pr);
  V3.pr = pr; V3.vao = gl.createVertexArray(); V3.vbo = gl.createBuffer(); V3.ibo = gl.createBuffer(); V3.uvb = gl.createBuffer(); V3.tx = gl.createTexture();
  V3.u = n => gl.getUniformLocation(pr, n);
  return true;
}
V3.setup = function () {
  const gl = V3.gl, { w: W, h: H, tex } = V3.pay; V3.W = W; V3.H = H; V3.fpx = .9 * W; V3.pos = new Float32Array(W * H * 3);
  gl.bindVertexArray(V3.vao);
  const uv = new Float32Array(W * H * 2); for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { uv[(j * W + i) * 2] = i / (W - 1); uv[(j * W + i) * 2 + 1] = j / (H - 1); }
  gl.bindBuffer(gl.ARRAY_BUFFER, V3.uvb); gl.bufferData(gl.ARRAY_BUFFER, uv, gl.STATIC_DRAW);
  const al = gl.getAttribLocation(V3.pr, "uv"); gl.enableVertexAttribArray(al); gl.vertexAttribPointer(al, 2, gl.FLOAT, false, 0, 0);
  gl.bindBuffer(gl.ARRAY_BUFFER, V3.vbo); const ap = gl.getAttribLocation(V3.pr, "p"); gl.enableVertexAttribArray(ap); gl.vertexAttribPointer(ap, 3, gl.FLOAT, false, 0, 0);
  const im = new Image(); im.onload = () => { gl.bindTexture(gl.TEXTURE_2D, V3.tx); gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, im); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR); }; im.src = tex;
  V3.build = function () {
    const W = V3.W, H = V3.H, d = V3.D[+$("#v3src").value || 0], cut = +$("#v3cut").value / 100, pos = V3.pos, fpx = V3.fpx, v = [];
    for (let j = 0; j < H; j++) for (let i = 0; i < W; i++) { const z = d[j * W + i] / 1000 || .01, k = (j * W + i) * 3; pos[k] = (i - W / 2) / fpx * z; pos[k + 1] = -(j - H / 2) / fpx * z; pos[k + 2] = -z; if (d[j * W + i]) v.push(z); }
    v.sort((a, b) => a - b); V3.dmed = v.length ? v[v.length >> 1] : 1;
    const idx = [];
    for (let j = 0; j < H - 1; j++) for (let i = 0; i < W - 1; i++) {
      const a = j * W + i, b = a + 1, c = a + W, e = c + 1, za = -pos[a * 3 + 2], zb = -pos[b * 3 + 2], zc = -pos[c * 3 + 2], ze = -pos[e * 3 + 2];
      const mx = Math.max(za, zb, zc, ze), mn = Math.min(za, zb, zc, ze);
      if (d[a] && d[b] && d[c] && d[e] && (mx - mn) < cut * mn) idx.push(a, c, b, b, c, e);
    }
    const I = new Uint32Array(idx); V3.n = I.length;
    gl.bindVertexArray(V3.vao); gl.bindBuffer(gl.ARRAY_BUFFER, V3.vbo); gl.bufferData(gl.ARRAY_BUFFER, pos, gl.DYNAMIC_DRAW);
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, V3.ibo); gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, I, gl.DYNAMIC_DRAW);
  };
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, V3.ibo);
  V3.yaw = 0; V3.pitch = 0; V3.zoom = 1; V3.build();
};
const mul = (a, b) => { const r = new Float32Array(16); for (let c = 0; c < 4; c++) for (let l = 0; l < 4; l++) { let s = 0; for (let k = 0; k < 4; k++) s += a[k * 4 + l] * b[c * 4 + k]; r[c * 4 + l] = s; } return r; };
const T = (x, y, z) => new Float32Array([1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, x, y, z, 1]);
const RY = a => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1]); };
const RX = a => { const c = Math.cos(a), s = Math.sin(a); return new Float32Array([1, 0, 0, 0, 0, c, s, 0, 0, -s, c, 0, 0, 0, 0, 1]); };
V3.loop = function (t) {
  const pane = $('.pane[data-pane="view3d"]');
  if (!pane.classList.contains("on")) { V3.running = false; return; }
  const gl = V3.gl, cv = $("#v3"), dpr = devicePixelRatio || 1;
  if ($("#v3sway").checked) { V3.yaw = Math.sin((t - V3.t0) / 1500) * .45; V3.pitch = Math.sin((t - V3.t0) / 2300) * .12; }
  cv.width = cv.clientWidth * dpr; cv.height = cv.clientHeight * dpr; gl.viewport(0, 0, cv.width, cv.height);
  gl.clearColor(.024, .035, .05, 1); gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT); gl.enable(gl.DEPTH_TEST);
  const asp = cv.width / cv.height, fy = 1 / Math.tan(Math.atan((V3.H / 2) / V3.fpx) * .95), n = .05, f = 200;
  const P = new Float32Array([fy / asp, 0, 0, 0, 0, fy, 0, 0, 0, 0, (f + n) / (n - f), -1, 0, 0, 2 * f * n / (n - f), 0]);
  const Vw = mul(T(0, 0, -V3.dmed * V3.zoom), mul(RX(V3.pitch), mul(RY(V3.yaw), T(0, 0, V3.dmed))));
  gl.uniformMatrix4fv(V3.u("M"), false, mul(P, Vw)); gl.uniform1i(V3.u("mode"), +$("#v3mode").value);
  gl.uniform2f(V3.u("rng"), V3.pay.dmin, V3.pay.dmax); gl.bindVertexArray(V3.vao); gl.drawElements(gl.TRIANGLES, V3.n, gl.UNSIGNED_INT, 0);
  requestAnimationFrame(V3.loop);
};

boot().catch(e => toast("Could not start: " + e.message));
