"""Interactive 3D depth viewer (raw WebGL2, no external libraries, works offline).

Back-projects each depth map to a textured mesh with a pinhole camera model and lets
you orbit it, switch between result / bicubic / ground truth, and shade the geometry
flat-grey. Smeared edges and "flying pixels" that are almost invisible in a colour-mapped
depth image are obvious in 3D, which is the quickest way to *see* edge preservation.
Triangles that span a large depth jump are dropped (slider), so bicubic's smeared edges
show up as stretched sheets rather than being hidden.
"""
from __future__ import annotations

import base64
import html as _html

import cv2
import numpy as np


def _b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def payload(rgb: np.ndarray, depths: dict[str, np.ndarray], max_side: int = 480) -> dict:
    """Mesh data for a viewer: texture (JPEG data URL), depth maps as base64 uint16 millimetres
    (nearest-neighbour resized so edges stay hard), and a depth range for colouring."""
    H, W = rgb.shape[:2]
    f = min(1.0, max_side / max(H, W))
    h, w = max(2, int(round(H * f))), max(2, int(round(W * f)))
    tex = cv2.resize(rgb, (w, h), interpolation=cv2.INTER_AREA)
    ok, jpg = cv2.imencode(".jpg", cv2.cvtColor(tex, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
    names, blobs = [], []
    for k, d in depths.items():
        dd = cv2.resize(np.nan_to_num(d.astype(np.float32)), (w, h), interpolation=cv2.INTER_NEAREST)
        names.append(k)
        blobs.append(_b64(np.clip(dd * 1000, 0, 65535).astype("<u2").tobytes()))
    ref = next(iter(depths.values()))
    v = ref[ref > 0]
    dmin, dmax = (float(np.percentile(v, 1)), float(np.percentile(v, 99))) if v.size else (0.5, 5.0)
    return dict(w=w, h=h, tex="data:image/jpeg;base64," + _b64(jpg.tobytes()), names=names, blobs=blobs,
                dmin=dmin, dmax=dmax)


def viewer_html(rgb: np.ndarray, depths: dict[str, np.ndarray], max_side: int = 480) -> str:
    """rgb: HxWx3 uint8; depths: {label: HxW float metres}. Returns a standalone HTML page."""
    import json
    d = payload(rgb, depths, max_side)
    return (_PAGE.replace("__W__", str(d["w"])).replace("__H__", str(d["h"])).replace("__NAMES__", json.dumps(d["names"]))
            .replace("__BLOBS__", json.dumps(d["blobs"])).replace("__TEX__", d["tex"])
            .replace("__DMIN__", f"{d['dmin']:.4f}").replace("__DMAX__", f"{d['dmax']:.4f}"))


def iframe(page: str, height: int = 460) -> str:
    return (f'<iframe srcdoc="{_html.escape(page, quote=True)}" style="width:100%;height:{height}px;'
            f'border:0;border-radius:14px;background:#0b0f17" sandbox="allow-scripts"></iframe>')


_PAGE = r"""<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0b0f17;color:#cbd5e1;font:12px ui-sans-serif,system-ui,sans-serif;overflow:hidden}
canvas{width:100%;height:100%;display:block;cursor:grab}
#ui{position:absolute;left:10px;top:8px;display:flex;gap:10px;flex-wrap:wrap;align-items:center;background:rgba(15,23,42,.8);
border:1px solid #232b3b;border-radius:10px;padding:6px 10px}
select,input{accent-color:#ff7a1a;background:#121826;color:#e2e8f0;border:1px solid #232b3b;border-radius:6px}
#hint{position:absolute;right:12px;bottom:8px;color:#64748b}
</style></head><body><canvas id="c"></canvas>
<div id="ui"><label>Depth <select id="src"></select></label>
<label>Colour <select id="mode"><option value="0">RGB texture</option><option value="2" selected>Flat shaded geometry</option><option value="1">Depth colours</option></select></label>
<label>Tear edges <input id="cut" type="range" min="1" max="60" value="12"></label>
<label><input id="sway" type="checkbox" checked> auto-orbit</label></div>
<div id="hint">drag = orbit | wheel = zoom</div>
<script>
const W=__W__,H=__H__,NAMES=__NAMES__,BL=__BLOBS__,DMIN=__DMIN__,DMAX=__DMAX__;
const D=BL.map(b=>{const s=atob(b),u=new Uint8Array(s.length);for(let i=0;i<s.length;i++)u[i]=s.charCodeAt(i);return new Uint16Array(u.buffer)});
const cv=document.getElementById('c'),gl=cv.getContext('webgl2');
const sel=document.getElementById('src');NAMES.forEach((n,i)=>sel.add(new Option(n,i)));
const vs=`#version 300 es
in vec3 p;in vec2 uv;uniform mat4 M;out vec2 vuv;out float vd;out vec3 vp;
void main(){vuv=uv;vd=-p.z;vp=p;gl_Position=M*vec4(p,1.);}`;
const fs=`#version 300 es
precision highp float;in vec2 vuv;in float vd;in vec3 vp;uniform sampler2D t;uniform int mode;uniform vec2 rng;out vec4 o;
vec3 ramp(float x){x=clamp(x,0.,1.);return vec3(clamp(1.5-abs(4.*x-3.),0.,1.),clamp(1.5-abs(4.*x-2.),0.,1.),clamp(1.5-abs(4.*x-1.),0.,1.));}
void main(){if(mode==0)o=vec4(texture(t,vuv).rgb,1.);
else if(mode==1)o=vec4(ramp(1.-(vd-rng.x)/(rng.y-rng.x)),1.);
else{vec3 n=normalize(cross(dFdx(vp),dFdy(vp)));float l=abs(dot(n,normalize(vec3(.3,.5,1.))));o=vec4(vec3(.15+.85*l),1.);}}`;
function sh(t,s){const x=gl.createShader(t);gl.shaderSource(x,s);gl.compileShader(x);return x}
const pr=gl.createProgram();gl.attachShader(pr,sh(gl.VERTEX_SHADER,vs));gl.attachShader(pr,sh(gl.FRAGMENT_SHADER,fs));gl.linkProgram(pr);gl.useProgram(pr);
const vbo=gl.createBuffer(),ibo=gl.createBuffer(),uvb=gl.createBuffer(),vao=gl.createVertexArray();
const uv=new Float32Array(W*H*2);for(let j=0;j<H;j++)for(let i=0;i<W;i++){uv[(j*W+i)*2]=i/(W-1);uv[(j*W+i)*2+1]=j/(H-1)}
gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,uvb);gl.bufferData(gl.ARRAY_BUFFER,uv,gl.STATIC_DRAW);
const al=gl.getAttribLocation(pr,'uv');gl.enableVertexAttribArray(al);gl.vertexAttribPointer(al,2,gl.FLOAT,false,0,0);
gl.bindBuffer(gl.ARRAY_BUFFER,vbo);const ap=gl.getAttribLocation(pr,'p');gl.enableVertexAttribArray(ap);gl.vertexAttribPointer(ap,3,gl.FLOAT,false,0,0);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ibo);
const tx=gl.createTexture(),im=new Image();im.onload=()=>{gl.bindTexture(gl.TEXTURE_2D,tx);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,im);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.LINEAR)};im.src="__TEX__";
const fpx=0.9*W,pos=new Float32Array(W*H*3);let nidx=0,dmed=1;
function build(){const d=D[sel.value],cut=+document.getElementById('cut').value/100;const v=[];
for(let j=0;j<H;j++)for(let i=0;i<W;i++){const z=d[j*W+i]/1000||0.01,k=(j*W+i)*3;pos[k]=(i-W/2)/fpx*z;pos[k+1]=-(j-H/2)/fpx*z;pos[k+2]=-z;if(d[j*W+i])v.push(z)}
v.sort((a,b)=>a-b);dmed=v.length?v[v.length>>1]:1;const idx=[];
for(let j=0;j<H-1;j++)for(let i=0;i<W-1;i++){const a=j*W+i,b=a+1,c=a+W,e=c+1,za=-pos[a*3+2],zb=-pos[b*3+2],zc=-pos[c*3+2],ze=-pos[e*3+2];
const mx=Math.max(za,zb,zc,ze),mn=Math.min(za,zb,zc,ze);if(d[a]&&d[b]&&d[c]&&d[e]&&(mx-mn)<cut*mn){idx.push(a,c,b,b,c,e)}}
const I=new Uint32Array(idx);nidx=I.length;gl.bindVertexArray(vao);gl.bindBuffer(gl.ARRAY_BUFFER,vbo);gl.bufferData(gl.ARRAY_BUFFER,pos,gl.DYNAMIC_DRAW);
gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER,ibo);gl.bufferData(gl.ELEMENT_ARRAY_BUFFER,I,gl.DYNAMIC_DRAW)}
let yaw=0,pitch=0,zoom=1,drag=0,lx=0,ly=0,t0=performance.now();
function mul(a,b){const r=new Float32Array(16);for(let c=0;c<4;c++)for(let l=0;l<4;l++){let s=0;for(let k=0;k<4;k++)s+=a[k*4+l]*b[c*4+k];r[c*4+l]=s}return r}
function T(x,y,z){return new Float32Array([1,0,0,0,0,1,0,0,0,0,1,0,x,y,z,1])}
function RY(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([c,0,-s,0,0,1,0,0,s,0,c,0,0,0,0,1])}
function RX(a){const c=Math.cos(a),s=Math.sin(a);return new Float32Array([1,0,0,0,0,c,s,0,0,-s,c,0,0,0,0,1])}
function draw(){cv.width=cv.clientWidth*devicePixelRatio;cv.height=cv.clientHeight*devicePixelRatio;gl.viewport(0,0,cv.width,cv.height);
gl.clearColor(.043,.059,.09,1);gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);gl.enable(gl.DEPTH_TEST);
const asp=cv.width/cv.height,fy=1/Math.tan(Math.atan((H/2)/fpx)*0.95),n=.05,f=200;
const P=new Float32Array([fy/asp,0,0,0,0,fy,0,0,0,0,(f+n)/(n-f),-1,0,0,2*f*n/(n-f),0]);
const V=mul(T(0,0,-dmed*zoom),mul(RX(pitch),mul(RY(yaw),T(0,0,dmed))));
gl.uniformMatrix4fv(gl.getUniformLocation(pr,'M'),false,mul(P,V));gl.uniform1i(gl.getUniformLocation(pr,'mode'),+document.getElementById('mode').value);
gl.uniform2f(gl.getUniformLocation(pr,'rng'),DMIN,DMAX);gl.bindVertexArray(vao);gl.drawElements(gl.TRIANGLES,nidx,gl.UNSIGNED_INT,0)}
function loop(t){if(document.getElementById('sway').checked&&!drag){yaw=Math.sin((t-t0)/1500)*.45;pitch=Math.sin((t-t0)/2300)*.12}draw();requestAnimationFrame(loop)}
cv.onmousedown=e=>{drag=1;lx=e.clientX;ly=e.clientY;document.getElementById('sway').checked=false;cv.style.cursor='grabbing'};
onmouseup=()=>{drag=0;cv.style.cursor='grab'};onmousemove=e=>{if(drag){yaw+=(e.clientX-lx)*.008;pitch=Math.max(-1.2,Math.min(1.2,pitch+(e.clientY-ly)*.008));lx=e.clientX;ly=e.clientY}};
cv.onwheel=e=>{e.preventDefault();zoom=Math.max(.15,Math.min(3,zoom*(e.deltaY>0?1.08:.92)))};
sel.onchange=document.getElementById('cut').oninput=build;build();requestAnimationFrame(loop);
</script></body></html>"""
