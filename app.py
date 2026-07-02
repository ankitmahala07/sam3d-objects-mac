"""
SAM-3D Objects — Sci-Fi Gradio UI  v3
Step-by-step pipeline execution, live intermediate 3D views, animated UI
"""

import sys, os, threading, queue, datetime, shutil, time, json, traceback, glob, struct
import numpy as np
import cv2
from PIL import Image as PILImage

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "notebook"))
os.environ.setdefault("SPARSE_BACKEND", "native")

# ── MPS monkey-patches ───────────────────────────────────────────────────────
import torch
if torch.backends.mps.is_available() and not torch.cuda.is_available():
    def _mps_cuda(self, *a, **kw): return self.to("mps")
    torch.Tensor.cuda = _mps_cuda
    import torch.nn as nn
    def _mps_module_cuda(self, device=None): return self.to("mps")
    nn.Module.cuda = _mps_module_cuda

import gradio as gr

# ── shared state ─────────────────────────────────────────────────────────────
_EVENTS: list[dict] = []
_EV_LOCK = threading.Lock()

_ASSETS: dict = {}          # "voxel_N", "splat_N", "glb_N" → file path
_STATS:  dict = {           # live stats for UI
    "stage": "idle", "obj": 0, "n_obj": 0,
    "start": None, "stage_start": None,
    "stage_times": {},
}
_ASSET_LOCK = threading.Lock()

# ── tqdm → event bridge ───────────────────────────────────────────────────────
import io as _io
class _TqdmCapture(_io.StringIO):
    _last = ""
    def write(self, s):
        s = s.strip("\r\n ")
        if not s or s == self._last:
            return
        self._last = s
        if any(t in s for t in ["%|", "it/s", "step", "Step", "Render", "Sampling"]):
            stage = ("render" if "Render" in s else
                     "latent" if "latent" in s.lower() else "sparse")
            with _EV_LOCK:
                _EVENTS.append({"t": _ts(), "level": "INFO",
                                 "msg": f"  ⟳ {s}", "stage": stage})
    def flush(self): pass
sys.stderr = _TqdmCapture()

# ── loguru tap ────────────────────────────────────────────────────────────────
PIPELINE_STAGES = [
    ("init",      "SYSTEM INIT",      "Initialising neural substrate"),
    ("condition", "CONDITION EMBED",  "Encoding visual condition vectors"),
    ("sparse",    "SPARSE STRUCTURE", "Sampling voxel lattice  [stage 1]"),
    ("latent",    "STRUCTURED LATENT","Sampling SLAT manifold  [stage 2]"),
    ("decode",    "DECODE SLAT",      "Decoding latent representations"),
    ("postproc",  "POST-PROCESS",     "Mesh extraction & texture baking"),
    ("render",    "MULTIVIEW RENDER", "Rasterising 30-view projection"),
    ("done",      "OUTPUT READY",     "3D asset serialised to disk"),
]
STAGE_KW = {
    "init":      ["Loading model","Loading checkpoint","Loaded DINO","model weights","GPU name"],
    "condition": ["Condition embedder","Running condition","compute_pointmap","preprocess"],
    "sparse":    ["Sampling sparse structure","sparse structure"],
    "latent":    ["Sampling sparse latent","Condition embedder finishes","sparse latent"],
    "decode":    ["Decoding sparse latent","decode"],
    "postproc":  ["Postprocessing mesh","texture baking","Mesh extraction"],
    "render":    ["Rendering","render_multiview","Rasterising"],
    "done":      ["Finished!","saved to","Complete","OUTPUT READY"],
}

def _ts(): return datetime.datetime.now().strftime("%H:%M:%S")

def _push(msg: str, stage: str = "init", level: str = "INFO"):
    with _EV_LOCK:
        _EVENTS.append({"t": _ts(), "level": level, "msg": msg, "stage": stage})

def _log_sink(message):
    record = message.record
    text   = record["message"]
    stage  = "init"
    for s, kws in STAGE_KW.items():
        if any(kw.lower() in text.lower() for kw in kws):
            stage = s; break
    _push(text, stage, record["level"].name)

from loguru import logger as _loguru
_loguru.add(_log_sink, level="INFO")

# ── pipeline loader ───────────────────────────────────────────────────────────
_pipeline      = None
_pipeline_lock = threading.Lock()

def get_pipeline(config_path: str):
    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _push("Loading pipeline checkpoints…", "init")
            from inference import Inference
            _pipeline = Inference(config_path, compile=False)
        return _pipeline

# ── object-selection helpers ──────────────────────────────────────────────────
MASK_COLORS = [(0,255,180),(255,80,40),(160,80,255),(255,220,0)]
N_SLOTS = 4

def _empty_sel():
    return {"masks": [None]*N_SLOTS,
            "last_clicks": [[] for _ in range(N_SLOTS)],
            "lasso": [[] for _ in range(N_SLOTS)]}

# ── rembg cache (keyed by image shape+hash so re-runs are instant) ───────────
_rembg_cache: dict = {}

def _rembg_mask(img_rgb):
    from rembg import remove
    return np.array(remove(PILImage.fromarray(img_rgb)))[..., 3] > 30

def _fg_components(img_rgb):
    """rembg foreground → cleaned connected components, largest first."""
    key = (img_rgb.shape, img_rgb[::8, ::8].tobytes())  # fast hash
    if key not in _rembg_cache:
        _rembg_cache.clear()          # keep only the current image
        fg = _rembg_mask(img_rgb).astype(np.uint8)
        # light opening to remove noise pixels
        k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k3)
        n, labels = cv2.connectedComponents(fg)
        comps = []
        for lbl in range(1, n):
            region = labels == lbl
            if region.sum() > 300:
                comps.append(region.astype(bool))
        _rembg_cache[key] = sorted(comps, key=lambda m: -m.sum())
    return _rembg_cache[key]

def _auto_detect_objects(img_rgb):
    """Return up to N_SLOTS largest foreground components."""
    return _fg_components(img_rgb)

import base64 as _b64, io as _bio

def _img_to_b64(img_np):
    buf = _bio.BytesIO()
    PILImage.fromarray(img_np.astype(np.uint8)).save(buf, format="PNG")
    return _b64.b64encode(buf.getvalue()).decode()

def _make_canvas_html(img_np):
    """Zoom/pan image viewer. Lock button enables capture-phase event interception."""
    uid = str(int(time.time()*1000))[-7:]
    if img_np is None:
        return ("<div style='background:#010609;border:1px solid #0f2038;border-radius:2px;"
                "height:340px;display:flex;flex-direction:column;align-items:center;"
                "justify-content:center;gap:10px;color:#2a4060;font-family:monospace;font-size:.8em'>"
                "<div>◈ Drop / click the upload box above</div>"
                "<div style='font-size:.7em;color:#1a3050'>then lock &amp; interact here</div>"
                "</div>")
    b64 = _img_to_b64(img_np)
    iw  = img_np.shape[1]
    ih  = img_np.shape[0]
    # Use stable IDs (not uid-based) so state persists across overlay re-renders
    return f"""
<div style='position:relative'>
  <div style='display:flex;align-items:center;gap:8px;margin-bottom:4px'>
    <button id='sam3d-lockbtn'
      style='font-family:monospace;font-size:.7em;padding:3px 10px;
             background:#010609;border:1px solid #00f5d4;color:#00f5d4;
             border-radius:2px;cursor:pointer;letter-spacing:1px;
             box-shadow:0 0 8px #00f5d430;transition:all .2s'>
      🔒 LOCK TO INTERACT
    </button>
    <span style='font-family:monospace;font-size:.62em;color:#1a3050'>
      scroll=zoom · middle-drag=pan · left-click=select
    </span>
  </div>
  <div id='sam3d-wrap' style='background:#010609;border:1px solid #0f2038;
       border-radius:2px;overflow:hidden;position:relative;height:320px;
       user-select:none;cursor:crosshair'>
    <img id='sam3d-img' src='data:image/png;base64,{b64}'
         style='position:absolute;top:0;left:0;transform-origin:0 0;
                max-width:none;max-height:none;pointer-events:none'
         draggable='false'>
    <div id='sam3d-crd' style='position:absolute;bottom:4px;right:8px;
         font-family:monospace;font-size:.63em;color:#1a3050;pointer-events:none'>x:— y:—</div>
    <div id='sam3d-lockedov' style='display:none;position:absolute;top:5px;left:8px;
         font-family:monospace;font-size:.63em;color:#ff5e20;pointer-events:none;
         text-shadow:0 0 8px #ff5e2099'>⬡ LOCKED</div>
  </div>
</div>
<script>
(function(){{
  // ── guard: only init once; on re-render just update the image src ────────
  const IW={iw}, IH={ih};
  const wrap= document.getElementById('sam3d-wrap');
  const img = document.getElementById('sam3d-img');
  const crd = document.getElementById('sam3d-crd');
  const ov  = document.getElementById('sam3d-lockedov');
  const btn = document.getElementById('sam3d-lockbtn');
  if(!wrap||!img||!btn) return;

  // If already initialised (re-render of overlay), just update image src
  if(window._sam3dReady){{
    img.src='data:image/png;base64,{b64}';
    return;
  }}
  window._sam3dReady=true;

  let tx=0,ty=0,sc=1;
  let dragging=false,dsx=0,dsy=0,dtx=0,dty=0;
  let locked=false, _tog=false;

  function applyT(){{ img.style.transform='translate('+tx+'px,'+ty+'px) scale('+sc+')'; }}
  function fit(){{
    const cw=wrap.clientWidth,ch=wrap.clientHeight;
    if(!cw||!ch) return;
    sc=Math.min(cw/IW,ch/IH)*0.97;
    tx=(cw-IW*sc)/2; ty=(ch-IH*sc)/2; applyT();
  }}
  function tryFit(n){{ if(wrap.clientWidth>0) fit(); else if(n>0) requestAnimationFrame(()=>tryFit(n-1)); }}
  tryFit(40);

  function imgCoords(cx,cy){{
    const r=wrap.getBoundingClientRect();
    return [Math.round((cx-r.left-tx)/sc), Math.round((cy-r.top-ty)/sc)];
  }}
  function inImg(ix,iy){{ return ix>=0&&iy>=0&&ix<IW&&iy<IH; }}

  function firePick(ix,iy){{
    const tb=document.querySelector('#zclick textarea');
    if(!tb){{ console.warn('SAM3D: #zclick textarea not found'); return; }}
    _tog=!_tog;
    const nv=ix+','+iy+(_tog?'!':'');
    const setter=Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype,'value').set;
    setter.call(tb,nv);
    tb.dispatchEvent(new InputEvent('input',{{bubbles:true}}));
    tb.dispatchEvent(new Event('change',{{bubbles:true}}));
  }}

  // ── capture handlers ────────────────────────────────────────────────────
  function onWheel(e){{
    const r=wrap.getBoundingClientRect();
    if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const f=e.deltaY<0?1.15:1/1.15;
    const ns=Math.max(0.1,Math.min(40,sc*f));
    const mx=e.clientX-r.left, my=e.clientY-r.top;
    tx=mx-(mx-tx)*ns/sc; ty=my-(my-ty)*ns/sc; sc=ns; applyT();
  }}
  function onMdown(e){{
    const r=wrap.getBoundingClientRect();
    const inside=e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom;
    if(e.button===1&&inside){{
      e.preventDefault(); e.stopImmediatePropagation();
      dragging=true; dsx=e.clientX; dsy=e.clientY; dtx=tx; dty=ty;
      wrap.style.cursor='grabbing';
    }}
  }}
  function onMmove(e){{
    if(dragging){{ tx=dtx+(e.clientX-dsx); ty=dty+(e.clientY-dsy); applyT(); }}
    const [ix,iy]=imgCoords(e.clientX,e.clientY);
    if(inImg(ix,iy)) crd.textContent='x:'+ix+' y:'+iy;
  }}
  function onMup(e){{
    if(e.button===1&&dragging){{ dragging=false; wrap.style.cursor='crosshair'; }}
  }}
  function onClick(e){{
    const r=wrap.getBoundingClientRect();
    const inside=e.clientX>=r.left&&e.clientX<=r.right&&e.clientY>=r.top&&e.clientY<=r.bottom;
    if(!inside||e.button!==0) return;
    e.preventDefault(); e.stopImmediatePropagation();
    const [ix,iy]=imgCoords(e.clientX,e.clientY);
    if(inImg(ix,iy)) firePick(ix,iy);
  }}

  function doLock(){{
    locked=true; document.body.style.overflow='hidden';
    btn.textContent='🔓 UNLOCK'; btn.style.borderColor='#ff5e20';
    btn.style.color='#ff5e20'; btn.style.boxShadow='0 0 12px #ff5e2066';
    wrap.style.borderColor='#ff5e20'; if(ov) ov.style.display='block';
    window.addEventListener('wheel',    onWheel, {{capture:true,passive:false}});
    window.addEventListener('mousedown',onMdown, {{capture:true}});
    window.addEventListener('mousemove',onMmove, {{capture:true}});
    window.addEventListener('mouseup',  onMup,   {{capture:true}});
    window.addEventListener('click',    onClick, {{capture:true}});
  }}
  function doUnlock(){{
    locked=false; dragging=false; document.body.style.overflow='';
    btn.textContent='🔒 LOCK TO INTERACT'; btn.style.borderColor='#00f5d4';
    btn.style.color='#00f5d4'; btn.style.boxShadow='0 0 8px #00f5d430';
    wrap.style.borderColor='#0f2038'; if(ov) ov.style.display='none';
    window.removeEventListener('wheel',    onWheel, {{capture:true}});
    window.removeEventListener('mousedown',onMdown, {{capture:true}});
    window.removeEventListener('mousemove',onMmove, {{capture:true}});
    window.removeEventListener('mouseup',  onMup,   {{capture:true}});
    window.removeEventListener('click',    onClick, {{capture:true}});
  }}

  btn.addEventListener('click', function(e){{
    e.stopPropagation();
    locked ? doUnlock() : doLock();
  }});
}})();
</script>
"""

def _comp_at(img_rgb, x, y):
    """Return the foreground component (bool mask) that contains pixel (x,y).
    If (x,y) is in background, return the nearest component."""
    h, w = img_rgb.shape[:2]
    comps = _fg_components(img_rgb)
    if not comps:
        return None
    # direct hit
    if 0 <= y < h and 0 <= x < w:
        for comp in comps:
            if comp[y, x]:
                return comp
    # nearest centroid
    best, best_d = comps[0], float("inf")
    for comp in comps:
        pts = np.argwhere(comp)
        cy, cx = pts[:,0].mean(), pts[:,1].mean()
        d = (cx-x)**2 + (cy-y)**2
        if d < best_d:
            best_d, best = d, comp
    return best

def _sel_add(img_rgb, existing, x, y):
    """Add the foreground component under (x,y) to existing mask."""
    h, w = img_rgb.shape[:2]
    region = _comp_at(img_rgb, x, y)
    if region is None:
        return existing if existing is not None else np.zeros((h,w), bool)
    if existing is not None and np.any(existing):
        return existing | region
    return region

def _sel_remove(existing, x, y):
    """Remove the connected component of existing mask at/nearest to (x,y)."""
    if existing is None or not np.any(existing):
        h, w = existing.shape if existing is not None else (1,1)
        return np.zeros((h,w), bool)
    h, w = existing.shape
    n, labels = cv2.connectedComponents(existing.astype(np.uint8))
    if n <= 1:
        return np.zeros((h,w), bool)
    lbl = labels[y, x] if (0 <= y < h and 0 <= x < w) else 0
    if lbl == 0:
        best, best_d = 1, float("inf")
        for l in range(1, n):
            pts = np.argwhere(labels == l)
            cy, cx = pts[:,0].mean(), pts[:,1].mean()
            d = (cx-x)**2 + (cy-y)**2
            if d < best_d: best_d, best = d, l
        lbl = best
    result = existing.copy()
    result[labels == lbl] = False
    return result

def _apply_lasso(existing, pts, h, w, mode="add"):
    """Fill the polygon defined by pts and add/subtract from existing mask."""
    if len(pts) < 3:
        return existing
    poly = np.array(pts, dtype=np.int32).reshape((-1,1,2))
    fill = np.zeros((h,w),np.uint8)
    cv2.fillPoly(fill,[poly],1)
    region = fill.astype(bool)
    if mode == "remove":
        if existing is None: return np.zeros((h,w),bool)
        return existing & ~region
    if existing is not None and existing.any():
        return existing | region
    return region

def _overlay_masks(img_rgb, sel, active):
    out = img_rgb.copy().astype(np.float32)
    masks = sel["masks"]
    for i in range(N_SLOTS):
        m = masks[i]
        if m is None or not m.any(): continue
        c = MASK_COLORS[i]
        ov = np.zeros_like(out); ov[m] = c
        alpha = 0.42 if i==active else 0.28
        out = np.where(m[...,None], out*(1-alpha)+ov*alpha, out)
        canvas = out.clip(0,255).astype(np.uint8)
        ctrs,_ = cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if i==active:
            cv2.drawContours(canvas,ctrs,-1,(255,255,255),4)
            cv2.drawContours(canvas,ctrs,-1,c,2)
        else:
            cv2.drawContours(canvas,ctrs,-1,c,1)
        out = canvas.astype(np.float32)
        if ctrs:
            all_pts = np.concatenate([ct.reshape(-1,2) for ct in ctrs])
            cx,cy = int(all_pts[:,0].mean()),int(all_pts[:,1].mean())
            cv2.putText(canvas,f"OBJ {i+1}",(cx-22,cy+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,(0,0,0),4)
            cv2.putText(canvas,f"OBJ {i+1}",(cx-22,cy+6),cv2.FONT_HERSHEY_SIMPLEX,0.55,c,2)
            out = canvas.astype(np.float32)
    canvas = out.clip(0,255).astype(np.uint8)
    for i in range(N_SLOTS):
        c = MASK_COLORS[i]
        for entry in sel.get("last_clicks",[[] for _ in range(N_SLOTS)])[i]:
            px,py = entry[0],entry[1]
            em = entry[2] if len(entry)>2 else "add"
            outer = 10 if i==active else 6
            if em=="remove":
                cv2.circle(canvas,(px,py),outer,(255,255,255),-1)
                cv2.circle(canvas,(px,py),outer-2,(0,40,180),-1)
                d=outer-3
                cv2.line(canvas,(px-d,py-d),(px+d,py+d),(255,80,80),2)
                cv2.line(canvas,(px+d,py-d),(px-d,py+d),(255,80,80),2)
            else:
                cv2.circle(canvas,(px,py),outer,(255,255,255),-1)
                cv2.circle(canvas,(px,py),outer-2,c,-1)
                if i==active: cv2.circle(canvas,(px,py),outer+3,c,1)
        # draw lasso polygon for this slot
        lasso_pts = sel.get("lasso",[[] for _ in range(N_SLOTS)])[i]
        if lasso_pts:
            lc = c if i==active else tuple(x//2 for x in c)
            for k,(lx,ly) in enumerate(lasso_pts):
                cv2.circle(canvas,(lx,ly),5,(255,255,255),-1)
                cv2.circle(canvas,(lx,ly),4,lc,-1)
                if k==0:
                    cv2.circle(canvas,(lx,ly),8,lc,2)  # first dot = open indicator
                if k>0:
                    px2,py2 = lasso_pts[k-1]
                    cv2.line(canvas,(px2,py2),(lx,ly),(255,255,255),3)
                    cv2.line(canvas,(px2,py2),(lx,ly),lc,1)
            # dashed closing line preview (last → first)
            if len(lasso_pts)>2:
                x1l,y1l = lasso_pts[-1]; x0l,y0l = lasso_pts[0]
                for t in np.linspace(0,1,12):
                    tx=int(x1l+(x0l-x1l)*t); ty=int(y1l+(y0l-y1l)*t)
                    if int(t*12)%2==0:
                        cv2.circle(canvas,(tx,ty),2,lc,-1)
    return canvas

def _slot_label_html(sel, active):
    parts = []
    for i in range(N_SLOTS):
        chex = "#{:02x}{:02x}{:02x}".format(*MASK_COLORS[i])
        npts = len(sel.get("last_clicks",[[] for _ in range(N_SLOTS)])[i])
        has  = sel["masks"][i] is not None and sel["masks"][i].any()
        bdr  = f"2px solid {chex};box-shadow:0 0 8px {chex}88" if i==active else "1px solid #1a3350"
        bg   = f"{chex}22" if i==active else "#0d1626"
        icon = "◉" if i==active else ("◎" if has else "○")
        st   = f"{npts}pt" if npts else ("auto" if has else "—")
        parts.append(
            f"<div style='padding:4px 8px;border:{bdr};background:{bg};"
            f"border-radius:3px;font-family:monospace;font-size:.75em'>"
            f"<span style='color:{chex};font-weight:700'>{icon} OBJ {i+1}</span> "
            f"<span style='color:#3a6080'>[{st}]</span></div>"
        )
    ahex = "#{:02x}{:02x}{:02x}".format(*MASK_COLORS[active])
    n_lasso = len(sel.get("lasso",[[] for _ in range(N_SLOTS)])[active])
    lasso_hint = (f" · <span style='color:#ffcc44'>LASSO: {n_lasso} pts — APPLY or CLEAR</span>"
                  if n_lasso else "")
    hdr  = (f"<div style='font-family:monospace;font-size:.7em;color:#3a6080;margin-bottom:4px'>"
            f"ACTIVE → <span style='color:{ahex};font-weight:700'>OBJ {active+1}</span>"
            f" · ADD/REMOVE: click → auto-detect region · LASSO: click dots → APPLY{lasso_hint}</div>")
    return hdr+"<div style='display:flex;gap:5px;flex-wrap:wrap'>"+"".join(parts)+"</div>"

# ── output-folder helpers ─────────────────────────────────────────────────────
def _find(folder, pattern):
    m = glob.glob(os.path.join(folder, pattern))
    return m[0] if m else None

def scan_output_folder(folder):
    if not folder or not os.path.isdir(folder): return []
    runs = []
    for e in sorted(os.scandir(folder), key=lambda x: x.name, reverse=True):
        if not e.is_dir(): continue
        d = e.path
        run = {"name":e.name,"path":d,
               "original":_find(d,"original.png"),
               "extracted":_find(d,"extracted*.png"),
               "voxel":_find(d,"voxel.ply"),
               "ply":_find(d,"splat*.ply"),
               "glb":_find(d,"output*.glb")}
        run["sub_objects"] = []
        try:
            for sub in sorted(os.scandir(d), key=lambda x: x.name):
                if sub.is_dir():
                    run["sub_objects"].append({
                        "name":sub.name,"path":sub.path,
                        "original":_find(sub.path,"original.png"),
                        "extracted":_find(sub.path,"extracted*.png"),
                        "voxel":_find(sub.path,"voxel.ply"),
                        "ply":_find(sub.path,"splat*.ply"),
                        "glb":_find(sub.path,"output*.glb"),
                    })
        except: pass
        runs.append(run)
    return runs

def _gallery_items(runs):
    items = []
    for r in runs:
        img = r["extracted"] or r["original"]
        if img:
            tags = " ".join(t for t,k in [("✓VOX",r["voxel"]),("✓PLY",r["ply"]),("✓GLB",r["glb"])] if k)
            items.append((img, f"{r['name']}  {tags}".strip()))
    return items

def _run_detail_html(run):
    if not run:
        return "<p style='color:#3a6080;font-family:monospace'>← Select a run</p>"
    def row(lbl,val,link=False):
        if not val:
            return f"<tr><td style='color:#3a6080'>{lbl}</td><td style='color:#3a6080'>—</td></tr>"
        fn = os.path.basename(val)
        if link:
            return (f"<tr><td style='color:#00f5d4'>{lbl}</td>"
                    f"<td><a href='file={val}' target='_blank' "
                    f"style='color:#a0f0d0;font-family:monospace'>{fn}</a></td></tr>")
        return f"<tr><td style='color:#00f5d4'>{lbl}</td><td style='color:#cde8ff;font-family:monospace'>{val}</td></tr>"
    lines = [f"<h4 style='color:#00f5d4;font-family:monospace'>▸ {run['name']}</h4>",
             "<table style='width:100%;border-collapse:collapse;font-size:.78em'>",
             row("PATH",run["path"]),row("ORIGINAL",run["original"],True),
             row("EXTRACTED",run["extracted"],True),row("VOXEL PLY",run["voxel"],True),
             row("SPLAT PLY",run["ply"],True),row("MESH GLB",run["glb"],True),"</table>"]
    for so in run.get("sub_objects",[]):
        lines += [f"<h5 style='color:#a855f7;font-family:monospace'>◈ {so['name']}</h5>",
                  "<table style='width:100%;border-collapse:collapse;font-size:.76em'>",
                  row("ORIGINAL",so["original"],True),row("EXTRACTED",so["extracted"],True),
                  row("VOXEL",so.get("voxel"),True),row("PLY",so["ply"],True),
                  row("GLB",so["glb"],True),"</table>"]
    return "\n".join(lines)

# ── PLY helpers ───────────────────────────────────────────────────────────────
def _gr_serve(path):
    """Copy file into Gradio's upload temp dir so it gets a proper served URL."""
    import tempfile, gradio.processing_utils as gpu
    try:
        tmp = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(path)[1],
            dir=None,
        )
        tmp.close()
        shutil.copy2(path, tmp.name)
        return tmp.name
    except Exception:
        return path  # fall back to original path


def _save_voxel_ply(coords_np, path):
    """Save Nx3 float32 point cloud as binary PLY."""
    pts = coords_np.astype(np.float32)
    n   = len(pts)
    hdr = (f"ply\nformat binary_little_endian 1.0\n"
           f"element vertex {n}\nproperty float x\nproperty float y\n"
           f"property float z\nend_header\n").encode()
    with open(path,"wb") as f:
        f.write(hdr); f.write(pts.tobytes())

# ── progress helpers ──────────────────────────────────────────────────────────
TLINE_TMPL = """<style>
.sr{{display:flex;align-items:center;gap:8px;padding:3px 2px;
     font-family:'Courier New',monospace;font-size:.72em}}
.sd{{width:8px;height:8px;border-radius:50%;flex-shrink:0;transition:all .3s}}
.done{{background:#00f5d4;box-shadow:0 0 6px #00f5d4}}
.active{{background:#ff5e20;box-shadow:0 0 14px #ff5e20;animation:p 0.8s infinite}}
.pend{{background:#1a3350;border:1px solid #3a6080}}
.sl{{flex:1;color:#cde8ff}}.sl.a{{color:#ff5e20}}.sl.d{{color:#00f5d4}}.sl.p{{color:#3a6080}}
.sc{{color:#2a4560;font-size:.85em}}
.st{{color:#00f5d455;font-size:.8em;margin-left:auto}}
@keyframes p{{0%,100%{{opacity:1;transform:scale(1)}}50%{{opacity:.3;transform:scale(0.7)}}}}
</style>{rows}"""

def _tline(active):
    ids = [s[0] for s in PIPELINE_STAGES]
    ai  = ids.index(active) if active in ids else -1
    rows = []
    for i,(sid,lbl,desc) in enumerate(PIPELINE_STAGES):
        dc,lc = (("done","d") if i<ai else ("active","a") if i==ai else ("pend","p"))
        t = _STATS["stage_times"].get(sid,"")
        tstr = f"<span class='st'>{t}</span>" if t else ""
        rows.append(f'<div class="sr"><div class="sd {dc}"></div>'
                    f'<span class="sl {lc}">[{i+1:02}] {lbl}</span>'
                    f'<span class="sc"> // {desc}</span>{tstr}</div>')
    return TLINE_TMPL.format(rows="\n".join(rows))

def _build_log(n=20):
    with _EV_LOCK: evs = list(_EVENTS[-n:])
    pfx = {"INFO":"›","WARNING":"⚠","ERROR":"✗"}
    lines = []
    for e in evs:
        p = pfx.get(e["level"],"·")
        col = ("#ff5e20" if e["level"]=="ERROR" else
               "#ffcc44" if e["level"]=="WARNING" else "#9dffd6")
        lines.append(f'<span style="color:#3a6080">{e["t"]}</span> '
                     f'<span style="color:{col}">{p} {e["msg"]}</span>')
    inner = "<br>".join(lines)
    # fixed-height scrollable terminal; JS auto-scrolls to bottom
    return (f"<div id='log-inner' style='height:320px;overflow-y:auto;"
            f"background:#010609;border:1px solid #00f5d4;"
            f"box-shadow:0 0 20px #00f5d420,inset 0 0 40px #00f5d408;"
            f"border-radius:2px;font-family:monospace;font-size:.72em;"
            f"line-height:1.7;padding:8px'>{inner}"
            f"<div id='log-anchor'></div></div>"
            f"<script>(function(){{"
            f"var a=document.getElementById('log-anchor');"
            f"if(a)a.scrollIntoView({{behavior:'instant'}});"
            f"}})();</script>")

def _cur_stage():
    with _EV_LOCK: evs = list(_EVENTS)
    for e in reversed(evs):
        if e.get("stage"): return e["stage"]
    return _STATS.get("stage","init")

def _elapsed_str(start):
    if start is None: return "—"
    s = int(time.time()-start)
    return f"{s//3600:02d}:{(s%3600)//60:02d}:{s%60:02d}"

def _stats_html():
    s      = _STATS
    elapsed= _elapsed_str(s.get("start"))
    stage  = s.get("stage","idle").upper()
    obj    = s.get("obj",0); n_obj = s.get("n_obj",0)
    try:
        import psutil; mem = f"{psutil.Process().memory_info().rss/1024**3:.1f} GB"
    except: mem = "—"
    return (
        f"<div style='font-family:monospace;font-size:.72em;display:flex;gap:20px;"
        f"color:#3a6080;padding:4px 0;border-top:1px solid #1a3350;margin-top:4px'>"
        f"<span>⏱ <b style='color:#00f5d4'>{elapsed}</b></span>"
        f"<span>🧱 STAGE <b style='color:#ff5e20'>{stage}</b></span>"
        f"<span>🗂 OBJ <b style='color:#a855f7'>{obj}/{n_obj}</b></span>"
        f"<span>💾 RAM <b style='color:#cde8ff'>{mem}</b></span>"
        f"</div>"
    )

# ── inference worker (step-by-step) ──────────────────────────────────────────
_gen_thread: threading.Thread | None = None
_result_q:   queue.Queue = queue.Queue()
_is_running  = False

def _gen_worker(img_rgb, masks, config_path, out_base, seed, s1, s2):
    global _is_running
    try:
        pipeline = get_pipeline(config_path)
        pipe     = pipeline._pipeline

        _STATS.update({"start": time.time(), "n_obj": len(masks),
                        "stage_times": {}, "stage": "condition"})

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        results   = []

        import torch as _t
        use_tex = _t.backends.mps.is_available() or _t.cuda.is_available()

        for idx, mask in enumerate(masks):
            _STATS["obj"] = idx+1
            obj_dir = os.path.join(out_base, f"{timestamp}_obj{idx+1}")
            os.makedirs(obj_dir, exist_ok=True)

            PILImage.fromarray(img_rgb).save(os.path.join(obj_dir,"original.png"))
            rgba = np.concatenate([img_rgb,(mask*255).astype(np.uint8)[...,None]],-1)
            PILImage.fromarray(rgba).save(os.path.join(obj_dir,"extracted.png"))

            def _stg(name, msg):
                t0 = time.time()
                _STATS["stage"] = name
                _push(msg, name)
                return t0

            def _done(name, t0, detail=""):
                dt = time.time()-t0
                _STATS["stage_times"][name] = f"{dt:.0f}s"
                _push(f"✓ {PIPELINE_STAGES[[s[0] for s in PIPELINE_STAGES].index(name)][1]}"
                      f" complete  [{dt:.1f}s]{' — '+detail if detail else ''}", name)

            with pipe.device:
                # ── CONDITION ───────────────────────────────────────────────
                t0 = _stg("condition", f"[OBJ {idx+1}] Computing depth pointmap…")
                merged = pipeline.merge_mask_to_rgba(img_rgb, mask)
                pmap_dict = pipe.compute_pointmap(merged, None)
                pointmap  = pmap_dict["pointmap"]
                _push(f"  Pointmap shape: {list(pointmap.shape)}", "condition")

                _push(f"  Preprocessing image (ss + slat)…", "condition")
                ss_input   = pipe.preprocess_image(merged, pipe.ss_preprocessor, pointmap=pointmap)
                slat_input = pipe.preprocess_image(merged, pipe.slat_preprocessor)
                if seed is not None: _t.manual_seed(seed)
                _done("condition", t0)

                # ── SPARSE STRUCTURE ─────────────────────────────────────────
                t0 = _stg("sparse", f"[OBJ {idx+1}] Stage 1 diffusion — {s1} denoising steps…")
                ss_ret = pipe.sample_sparse_structure(ss_input, inference_steps=s1)
                pm_scale = ss_input.get("pointmap_scale")
                pm_shift = ss_input.get("pointmap_shift")
                ss_ret.update(pipe.pose_decoder(ss_ret, scene_scale=pm_scale, scene_shift=pm_shift))
                ss_ret["scale"] = ss_ret["scale"] * ss_ret["downsample_factor"]
                coords = ss_ret["coords"]                          # [N, 4]
                voxels = coords[:,1:].float() / 64.0 - 0.5        # normalise
                n_vox  = len(coords)
                voxel_ply = os.path.join(obj_dir,"voxel.ply")
                _save_voxel_ply(voxels.cpu().numpy(), voxel_ply)
                with _ASSET_LOCK:
                    _ASSETS[f"voxel_{idx}"] = _gr_serve(voxel_ply)
                    _ASSETS[f"splat_{idx}"] = None
                    _ASSETS[f"glb_{idx}"]   = None
                _done("sparse", t0, f"{n_vox:,} active voxels → voxel.ply")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # ── STRUCTURED LATENT ────────────────────────────────────────
                t0 = _stg("latent", f"[OBJ {idx+1}] Stage 2 diffusion — {s2} denoising steps…")
                slat = pipe.sample_slat(slat_input, coords, inference_steps=s2)
                _done("latent", t0)
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # ── DECODE GAUSSIAN ──────────────────────────────────────────
                t0 = _stg("decode", f"[OBJ {idx+1}] Decoding to Gaussian splat…")
                slat_dec = pipe.decode_slat(slat, ["gaussian"])
                gs = slat_dec["gaussian"][0]
                ply_path = os.path.join(obj_dir,"splat.ply")
                gs.save_ply(ply_path)
                n_gs = gs.get_xyz.shape[0]
                with _ASSET_LOCK:
                    _ASSETS[f"splat_{idx}"] = _gr_serve(ply_path)
                _done("decode", t0, f"{n_gs:,} gaussians → splat.ply")
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()

                # ── MESH + TEXTURE ───────────────────────────────────────────
                glb_path = None
                if use_tex:
                    t0 = _stg("postproc", f"[OBJ {idx+1}] Mesh extraction + texture baking…")
                    slat_dec.update(pipe.decode_slat(slat, ["mesh"]))
                    outputs = pipe.postprocess_slat_output(slat_dec, False, True, False)

                    t0r = _stg("render", f"[OBJ {idx+1}] Multiview rasterisation…")
                    if outputs.get("glb"):
                        glb_path = os.path.join(obj_dir,"output.glb")
                        outputs["glb"].export(glb_path)
                        with _ASSET_LOCK:
                            _ASSETS[f"glb_{idx}"] = _gr_serve(glb_path)
                        _done("render", t0r)
                        _done("postproc", t0, "output.glb")
                else:
                    _push("[OBJ {idx+1}] Skipping texture bake (no GPU)", "postproc","WARNING")

            results.append({"obj":idx+1,"dir":obj_dir,
                             "voxel":voxel_ply,"ply":ply_path,"glb":glb_path})
            _push(f"━━ Object {idx+1} complete → {obj_dir}", "done")

        _STATS["stage"] = "done"
        _result_q.put({"ok":True,"results":results,"out_base":out_base})
    except Exception as e:
        _push(f"✗ {e}", "done", "ERROR")
        _result_q.put({"ok":False,"error":str(e),"tb":traceback.format_exc()})
    finally:
        _is_running = False

# ── CSS ───────────────────────────────────────────────────────────────────────
CSS = """
/* ── variables ── */
:root{
  --bg:#04080f;--panel:#080f1e;--border:#0f2038;
  --cyan:#00f5d4;--cyan2:#00bfa0;--orange:#ff5e20;--violet:#a855f7;
  --dim:#2a4060;--text:#bde0ff;--muted:#3a6080;
  --mono:'JetBrains Mono','Fira Code','Courier New',monospace;
}

/* ── base ── */
body,.gradio-container{
  background:var(--bg)!important;
  color:var(--text)!important;
  font-family:var(--mono)!important;
}

/* ── scan-line overlay ── */
body::after{
  content:'';position:fixed;top:0;left:0;right:0;bottom:0;
  background:repeating-linear-gradient(
    0deg,transparent,transparent 2px,rgba(0,0,0,.07) 2px,rgba(0,0,0,.07) 4px);
  pointer-events:none;z-index:9998;
}


/* ── panels ── */
.gr-box,.block{
  background:var(--panel)!important;
  border:1px solid var(--border)!important;
  border-radius:2px!important;
  box-shadow:inset 0 0 30px rgba(0,245,212,.03)!important;
}

/* ── buttons ── */
button.primary{
  background:linear-gradient(135deg,#003344,#005533)!important;
  border:1px solid var(--cyan)!important;color:var(--cyan)!important;
  font-family:var(--mono)!important;font-weight:700!important;
  letter-spacing:2px!important;text-transform:uppercase!important;
  box-shadow:0 0 16px #00f5d430,inset 0 0 8px #00f5d410!important;
  transition:all .2s!important;
}
button.primary:hover{
  box-shadow:0 0 30px #00f5d466,inset 0 0 16px #00f5d420!important;
  letter-spacing:3px!important;
}
button.secondary{
  background:#060e1a!important;border:1px solid var(--dim)!important;
  color:var(--text)!important;font-family:var(--mono)!important;
  transition:border-color .2s!important;
}
button.secondary:hover{border-color:var(--cyan)!important;}

/* ── typography ── */
label,.label-wrap span,h1,h2,h3,p,span{color:var(--text)!important;font-family:var(--mono)!important}
h1{color:var(--cyan)!important;letter-spacing:4px;text-shadow:0 0 20px #00f5d466}
h3{color:var(--cyan)!important;letter-spacing:2px;font-size:.75em;opacity:.7;
   text-transform:uppercase}

/* ── inputs ── */
input[type=number],input[type=text],select,textarea{
  background:#020810!important;border:1px solid var(--dim)!important;
  color:var(--text)!important;font-family:var(--mono)!important;
}
input[type=range]{accent-color:var(--cyan)}

/* ── log terminal (styled inline in _build_log) ── */
#log-inner{ scrollbar-width:thin; scrollbar-color:#1a3350 #010609; }

/* ── hide click-bridge textbox but keep in DOM ── */
#zclick{ position:fixed!important; top:-999px!important; left:-999px!important;
         width:1px!important; height:1px!important; opacity:0!important;
         pointer-events:none!important; }

/* ── timeline ── */
#tline-box{
  background:#010609!important;border:1px solid var(--border)!important;
  padding:6px 8px!important;border-radius:2px!important;
}

/* ── tabs ── */
.tab-nav button{
  background:var(--panel)!important;border:1px solid var(--border)!important;
  color:var(--muted)!important;font-family:var(--mono)!important;
  font-size:.7em!important;letter-spacing:1px!important;text-transform:uppercase!important;
}
.tab-nav button.selected{
  border-bottom:2px solid var(--cyan)!important;color:var(--cyan)!important;
  text-shadow:0 0 8px var(--cyan);
}

/* ── 3D viewer frame ── */
canvas{background:#010609!important}
.model3D-container,.model-3d-wrap{
  border:1px solid var(--cyan)!important;
  box-shadow:0 0 20px #00f5d420!important;
}

/* ── gallery ── */
.gallery-item img{border:1px solid var(--border)!important;border-radius:1px!important}
.gallery-item:hover img{border-color:var(--cyan)!important}

/* ── radio ── */
input[type=radio]{accent-color:var(--cyan)}

/* ── glowing header decoration ── */
.hdr-deco{
  border-left:3px solid var(--cyan);padding-left:8px;
  box-shadow:-4px 0 12px #00f5d440;
}
"""

# ── waveform + timer HTML (self-animating, not updated by poll) ───────────────
WAVEFORM_HTML = """
<div style='background:#010609;border:1px solid #0f2038;border-radius:2px;
            padding:6px;margin:4px 0'>
  <div style='font-family:monospace;font-size:.65em;color:#2a4060;
              margin-bottom:4px;letter-spacing:1px'>
    ◈ NEURAL ACTIVITY MONITOR
  </div>
  <canvas id='nwave' height='40'
          style='width:100%;display:block'></canvas>
</div>
<script>
(function(){
  const c=document.getElementById('nwave');
  if(!c)return;
  const ctx=c.getContext('2d');
  let p=0,running=true;
  function resize(){c.width=c.parentElement.offsetWidth-12;}
  resize();window.addEventListener('resize',resize);
  function draw(){
    if(!running)return;
    const W=c.width,H=c.height;
    ctx.fillStyle='#010609';ctx.fillRect(0,0,W,H);
    // secondary dim line
    ctx.strokeStyle='#0f2038';ctx.lineWidth=1;
    ctx.beginPath();
    for(let x=0;x<W;x++){
      const y=H/2+Math.sin(x*.03+p*.5)*8+Math.sin(x*.09+p*.9)*4;
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }ctx.stroke();
    // primary glow line
    ctx.strokeStyle='#00f5d4';ctx.lineWidth=1.5;
    ctx.shadowColor='#00f5d4';ctx.shadowBlur=6;
    ctx.beginPath();
    for(let x=0;x<W;x++){
      const noise=(Math.random()-.5)*1.5;
      const y=H/2+Math.sin(x*.05+p)*12+Math.sin(x*.13+p*1.7)*5+noise;
      x===0?ctx.moveTo(x,y):ctx.lineTo(x,y);
    }ctx.stroke();
    ctx.shadowBlur=0;
    p+=0.055;requestAnimationFrame(draw);
  }
  draw();
  // stop when element removed
  new MutationObserver(()=>{if(!document.contains(c))running=false;})
    .observe(document.body,{childList:true,subtree:true});
})();
</script>
"""

STATS_PANEL_STATIC = """
<div id='stats-wrap' style='font-family:monospace;font-size:.68em;
     display:flex;gap:16px;color:#2a4060;padding:4px 0 2px;
     border-top:1px solid #0f2038;margin-top:2px;flex-wrap:wrap'>
  <span>⏱ ELAPSED <b id='elapsed-val' style='color:#00f5d4'>00:00:00</b></span>
  <span>🧠 STAGE <b id='stage-val' style='color:#ff5e20'>IDLE</b></span>
  <span>📦 OBJECT <b id='obj-val' style='color:#a855f7'>0 / 0</b></span>
</div>
<script>
(function(){
  let start=null,interval=null;
  function pad(n){return String(n).padStart(2,'0');}
  function tick(){
    if(!start)return;
    const s=Math.floor((Date.now()-start)/1000);
    const el=document.getElementById('elapsed-val');
    if(el)el.textContent=pad(Math.floor(s/3600))+':'+pad(Math.floor((s%3600)/60))+':'+pad(s%60);
  }
  // Listen for custom events fired by poll()
  window.addEventListener('sam3d-start',e=>{
    start=Date.now();
    if(interval)clearInterval(interval);
    interval=setInterval(tick,1000);
    const sv=document.getElementById('stage-val');
    const ov=document.getElementById('obj-val');
    if(sv)sv.textContent='INIT';
    if(ov)ov.textContent=`1 / ${e.detail.n_obj}`;
  });
  window.addEventListener('sam3d-update',e=>{
    const sv=document.getElementById('stage-val');
    const ov=document.getElementById('obj-val');
    if(sv)sv.textContent=e.detail.stage.toUpperCase();
    if(ov)ov.textContent=`${e.detail.obj} / ${e.detail.n_obj}`;
  });
})();
</script>
"""

CONFIG_PATH = os.path.join(ROOT,"checkpoints","hf","checkpoints","pipeline.yaml")
DEFAULT_OUT  = os.path.join(ROOT,"outputs")

# ── build app ─────────────────────────────────────────────────────────────────
def build_app():
    with gr.Blocks(title="SAM-3D // NEURAL RECONSTRUCTION") as demo:

        gr.HTML("""
<div style='padding:16px 0 8px'>
  <div style='font-family:"Courier New",monospace;color:#00f5d4;
              font-size:1.6em;letter-spacing:5px;font-weight:700;
              text-shadow:0 0 30px #00f5d4aa'>
    ◈ SAM-3D // NEURAL RECONSTRUCTION SYSTEM
  </div>
  <div style='font-family:"Courier New",monospace;color:#2a4060;
              font-size:.72em;letter-spacing:3px;margin-top:4px'>
    ▸ VOLUMETRIC INFERENCE ENGINE  ·  APPLE SILICON MPS  ·  STEP-BY-STEP PIPELINE
  </div>
</div>""")

        sel_state      = gr.State(_empty_sel())
        active_obj     = gr.State(0)
        orig_img_state = gr.State(None)
        runs_state     = gr.State([])

        with gr.Tabs():

            # ═══ TAB: GENERATE ═══════════════════════════════════════════════
            with gr.Tab("⬡  GENERATE"):
                with gr.Row():

                    # ── LEFT ─────────────────────────────────────────────────
                    with gr.Column(scale=5):
                        gr.Markdown("### ▸ INPUT FRAME  +  OBJECT SELECTION")
                        upload_img = gr.Image(
                            label="SOURCE IMAGE  [ click to select · drag to different slot ]",
                            type="numpy", height=360, interactive=True,
                        )
                        img_height_slider = gr.Slider(
                            label="IMAGE HEIGHT", minimum=200, maximum=700,
                            step=50, value=360, interactive=True,
                        )
                        active_slot_radio = gr.Radio(
                            choices=["OBJ 1","OBJ 2","OBJ 3","OBJ 4"],
                            value="OBJ 1", label="ACTIVE SLOT", type="index",
                        )
                        click_mode_radio = gr.Radio(
                            choices=["➕ ADD REGION","➖ REMOVE REGION","✏ LASSO"],
                            value="➕ ADD REGION", label="CLICK MODE",
                        )
                        slot_status_html = gr.HTML(_slot_label_html(_empty_sel(),0))
                        with gr.Row():
                            btn_auto         = gr.Button("⟳ AUTO",        variant="secondary", size="sm")
                            btn_apply_lasso  = gr.Button("✔ APPLY LASSO", variant="secondary", size="sm")
                            btn_clear_lasso  = gr.Button("✕ LASSO",       variant="secondary", size="sm")
                        with gr.Row():
                            btn_clr_slot  = gr.Button("✕ SLOT",        variant="secondary", size="sm")
                            btn_clr_all   = gr.Button("✕ ALL",         variant="secondary", size="sm")

                        gr.Markdown("### ▸ OUTPUT FOLDER")
                        with gr.Row():
                            out_folder_box = gr.Textbox(
                                label="DIRECTORY  [ drag folder here ]",
                                value=DEFAULT_OUT, scale=4,
                                elem_id="out-folder-box")
                            btn_mkdir = gr.Button("✚", variant="secondary", size="sm", scale=1)
                        folder_status = gr.Textbox(
                            label="", value="", interactive=False, max_lines=1, visible=False)
                        gr.HTML("""<script>
(function(){
  function _wire(){
    var wrap = document.querySelector('#out-folder-box textarea');
    if(!wrap){ setTimeout(_wire,400); return; }
    var box = wrap;
    ['dragover','dragenter'].forEach(function(ev){
      box.addEventListener(ev, function(e){
        e.preventDefault(); e.stopPropagation();
        box.style.outline='2px solid #00f5d4';
      });
    });
    ['dragleave','drop'].forEach(function(ev){
      box.addEventListener(ev, function(e){
        box.style.outline='';
      });
    });
    box.addEventListener('drop', function(e){
      e.preventDefault(); e.stopPropagation();
      var items = e.dataTransfer.items || [];
      for(var i=0;i<items.length;i++){
        var entry = items[i].webkitGetAsEntry ? items[i].webkitGetAsEntry() : null;
        if(entry && entry.isDirectory){ box.value=entry.fullPath; box.dispatchEvent(new Event('input',{bubbles:true})); return; }
      }
      // fallback: files have a path on Electron/desktop
      var files = e.dataTransfer.files;
      if(files && files.length){
        var p = files[0].path || files[0].name;
        // strip filename to get directory if it looks like a file
        if(p && p.indexOf('.')>-1 && !p.endsWith('/')) p=p.replace(/\\/[^\\/]+$/,'');
        box.value=p; box.dispatchEvent(new Event('input',{bubbles:true}));
      }
    });
  }
  _wire();
})();
</script>""")

                        gr.Markdown("### ▸ RENDERING PARAMETERS")
                        with gr.Row():
                            seed = gr.Number(label="SEED", value=42, precision=0, minimum=0, scale=1)
                            quality_preset = gr.Radio(
                                ["FAST","STANDARD","HIGH"],
                                label="QUALITY", value="STANDARD", scale=2)
                        with gr.Row():
                            stage1_steps = gr.Slider(
                                label="STAGE-1 STEPS  (sparse structure)",
                                minimum=5,maximum=50,step=5,value=25)
                            stage2_steps = gr.Slider(
                                label="STAGE-2 STEPS  (SLAT diffusion)",
                                minimum=5,maximum=50,step=5,value=25)

                        with gr.Row():
                            run_btn      = gr.Button("⬡  INITIATE 3D RECONSTRUCTION", variant="primary", scale=3)
                            extract_btn  = gr.Button("⬡  EXTRACT ONLY", variant="secondary", scale=1)
                        status_box = gr.Textbox(
                            label="STATUS", value="Idle", interactive=False, max_lines=1)

                    # ── RIGHT ────────────────────────────────────────────────
                    with gr.Column(scale=5):
                        gr.Markdown("### ▸ PIPELINE STAGE MONITOR")
                        timeline_html = gr.HTML(_tline("init"), elem_id="tline-box")
                        gr.HTML(WAVEFORM_HTML)
                        gr.HTML(STATS_PANEL_STATIC)

                        gr.Markdown("### ▸ SYSTEM LOG  [ live telemetry ]")
                        log_box = gr.HTML(
                            value="",
                            elem_id="prog-box",
                        )

                        render_badge_html = gr.HTML("", elem_id="render-badge")
                        gr.Markdown("### ▸ LIVE 3D OUTPUT  [ updates at each milestone ]")
                        # obj_viewers[i] = (voxel, splat, glb, dl_ply, dl_glb)
                        obj_viewers = []
                        with gr.Tabs():
                            for _oi in range(N_SLOTS):
                                _chex = "#{:02x}{:02x}{:02x}".format(*MASK_COLORS[_oi])
                                with gr.Tab(f"OBJ {_oi+1}"):
                                    with gr.Tabs():
                                        with gr.Tab("VOXEL CLOUD"):
                                            gr.HTML(f"<div style='font-family:monospace;font-size:.65em;"
                                                    f"color:{_chex};padding:2px 0'>◈ Sparse voxel structure"
                                                    f" — ready after Stage 1</div>")
                                            _v = gr.Model3D(label="", height=220)
                                        with gr.Tab("GAUSSIAN SPLAT"):
                                            gr.HTML(f"<div style='font-family:monospace;font-size:.65em;"
                                                    f"color:{_chex};padding:2px 0'>◈ Gaussian splat PLY"
                                                    f" — ready after decode</div>")
                                            _s = gr.Model3D(label="", height=220)
                                            _dp = gr.File(label="↓ PLY", visible=False)
                                        with gr.Tab("TEXTURED MESH"):
                                            gr.HTML(f"<div style='font-family:monospace;font-size:.65em;"
                                                    f"color:{_chex};padding:2px 0'>◈ Textured GLB mesh"
                                                    f" — ready after texture bake</div>")
                                            _g = gr.Model3D(label="", height=220)
                                            _dg = gr.File(label="↓ GLB", visible=False)
                                    obj_viewers.append((_v, _s, _g, _dp, _dg))

            # ═══ TAB: OUTPUT ARCHIVE ═════════════════════════════════════════
            with gr.Tab("📂  OUTPUT ARCHIVE"):
                with gr.Row():
                    archive_folder = gr.Textbox(label="BROWSE FOLDER",
                                                value=DEFAULT_OUT, scale=5)
                    btn_refresh = gr.Button("⟳ SCAN",variant="secondary",scale=1)
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("### ▸ RUN GALLERY")
                        archive_gallery = gr.Gallery(
                            label="",show_label=False,columns=3,
                            height=400,object_fit="contain",allow_preview=True)
                    with gr.Column(scale=5):
                        gr.Markdown("### ▸ RUN DETAILS")
                        run_detail_html = gr.HTML(
                            "<p style='color:#2a4060;font-family:monospace'>← Select a run</p>")
                        archive_extracted = gr.Image(
                            label="Extracted",type="filepath",height=180,interactive=False)
                        with gr.Row():
                            archive_voxel = gr.Model3D(label="Voxel",height=200)
                            archive_splat = gr.Model3D(label="Splat",height=200)
                        archive_glb_view = gr.Model3D(label="Textured Mesh",height=220)
                        with gr.Row():
                            arc_dl_ply = gr.File(label="PLY")
                            arc_dl_glb = gr.File(label="GLB")

        # ── live timer ────────────────────────────────────────────────────────
        timer = gr.Timer(value=1.5)
        _prev_assets = {}

        # Flatten viewer components into ordered list for poll() outputs
        # Order: [voxel_0, splat_0, glb_0, dlply_0, dlglb_0,  voxel_1, ...]
        _viewer_outputs = []
        for _v, _s, _g, _dp, _dg in obj_viewers:
            _viewer_outputs += [_v, _s, _g, _dp, _dg]

        # Map pipeline stage → which sub-tab is active
        _STAGE_TO_SUBTAB = {
            "sparse":  "VOXEL CLOUD",
            "latent":  "GAUSSIAN SPLAT",
            "decode":  "GAUSSIAN SPLAT",
            "postproc":"TEXTURED MESH",
            "render":  "TEXTURED MESH",
        }

        def _render_badge(stage, obj_i):
            """HTML badge + JS that auto-clicks the right OBJ tab and sub-tab."""
            if not _is_running or stage in ("idle","done","init","condition"):
                return ""
            obj_label  = f"OBJ {obj_i}"
            subtab     = _STAGE_TO_SUBTAB.get(stage, "VOXEL CLOUD")
            chex       = "#{:02x}{:02x}{:02x}".format(*MASK_COLORS[max(0,obj_i-1)])
            stage_name = next((s[1] for s in PIPELINE_STAGES if s[0]==stage), stage.upper())
            badge = (
                f"<div style='font-family:monospace;font-size:.7em;padding:4px 8px;"
                f"background:#010609;border:1px solid {chex};border-radius:2px;"
                f"display:flex;gap:14px;align-items:center;margin-bottom:4px;"
                f"box-shadow:0 0 10px {chex}44'>"
                f"<span style='color:#3a6080'>▸ LIVE</span>"
                f"<span style='color:{chex};font-weight:700;animation:lbp .8s infinite'>"
                f"⬡ {obj_label}</span>"
                f"<span style='color:#ff5e20'>◉ {stage_name}</span>"
                f"<span style='color:#3a6080'>→ {subtab}</span>"
                f"</div>"
                f"<style>@keyframes lbp{{0%,100%{{opacity:1}}50%{{opacity:.3}}}}</style>"
            )
            js = f"""<script>
(function(){{
  function clickTab(txt){{
    for(const b of document.querySelectorAll('button[role="tab"]')){{
      if(b.textContent.trim()===txt && !b.classList.contains('selected')){{
        b.click(); return true;
      }}
    }}
    return false;
  }}
  // slight delay so Gradio finishes its own rendering
  setTimeout(()=>{{
    clickTab('{obj_label}');
    setTimeout(()=>clickTab('{subtab}'),80);
  }},120);
}})();
</script>"""
            return badge + js

        def poll():
            stage  = _cur_stage()
            tl     = _tline(stage)
            log    = _build_log()
            status = "⟳ Running…" if _is_running else ("✓ Done" if _EVENTS else "Idle")
            obj_i  = _STATS.get("obj", 1)
            badge  = _render_badge(stage, obj_i)

            # Build update list matching _viewer_outputs (5 per object × N_SLOTS)
            updates = [gr.update()] * (N_SLOTS * 5)

            with _ASSET_LOCK:
                assets_snap = dict(_ASSETS)

            for i in range(N_SLOTS):
                base = i * 5
                v = assets_snap.get(f"voxel_{i}")
                s = assets_snap.get(f"splat_{i}")
                g = assets_snap.get(f"glb_{i}")
                if v and _prev_assets.get(f"v{i}") != v:
                    updates[base]   = gr.update(value=v)
                    _prev_assets[f"v{i}"] = v
                if s and _prev_assets.get(f"s{i}") != s:
                    updates[base+1] = gr.update(value=s)
                    updates[base+3] = gr.update(value=s, visible=True)
                    _prev_assets[f"s{i}"] = s
                if g and _prev_assets.get(f"g{i}") != g:
                    updates[base+2] = gr.update(value=g)
                    updates[base+4] = gr.update(value=g, visible=True)
                    _prev_assets[f"g{i}"] = g

            try:
                res = _result_q.get_nowait()
                if res.get("ok"):
                    rs = res["results"]
                    status = f"✓ Complete — {len(rs)} object(s) → {res.get('out_base','')}"
                    tl = _tline("done")
                else:
                    status = f"✗ Error: {res.get('error','?')}"
            except queue.Empty:
                pass

            return [tl, log, status, badge] + updates

        timer.tick(poll, outputs=[timeline_html, log_box, status_box,
                                   render_badge_html] + _viewer_outputs)

        # ── quality preset ────────────────────────────────────────────────────
        def apply_q(p):
            v={"FAST":10,"STANDARD":25,"HIGH":50}.get(p,25)
            return gr.update(value=v),gr.update(value=v)
        quality_preset.change(apply_q, quality_preset, [stage1_steps,stage2_steps])

        # ── mkdir ─────────────────────────────────────────────────────────────
        def make_dir(p):
            try:
                os.makedirs(p,exist_ok=True)
                return gr.update(value=f"✓ {p}",visible=True)
            except Exception as e:
                return gr.update(value=f"✗ {e}",visible=True)
        btn_mkdir.click(make_dir, out_folder_box, folder_status)

        # ── image resize ──────────────────────────────────────────────────────
        img_height_slider.change(
            lambda h: gr.update(height=int(h)),
            img_height_slider, upload_img)

        # ── overlay helper ────────────────────────────────────────────────────
        def _ov(orig, sel, active):
            if orig is None: return None
            return _overlay_masks(np.array(orig)[..., :3], sel, active)

        # ── image upload ──────────────────────────────────────────────────────
        def on_upload(img):
            sel = _empty_sel()
            if img is None:
                return sel, None, None, _slot_label_html(sel,0), _tline("init"), ""
            rgb  = np.array(img)[..., :3]
            objs = _auto_detect_objects(rgb)
            for i,m in enumerate(objs[:N_SLOTS]):
                sel["masks"][i] = m
            return sel, img, _overlay_masks(rgb,sel,0), \
                   _slot_label_html(sel,0), _tline("init"), ""

        upload_img.upload(on_upload, upload_img,
            [sel_state, orig_img_state, upload_img,
             slot_status_html, timeline_html, log_box])

        # ── click ─────────────────────────────────────────────────────────────
        def on_click(orig, sel, active, mode_str, evt: gr.SelectData):
            if orig is None: return sel, None, _slot_label_html(sel,active)
            import copy; sel = copy.deepcopy(sel)
            # ensure keys exist (defensive, state may be from older session)
            if "lasso" not in sel: sel["lasso"] = [[] for _ in range(N_SLOTS)]
            if "last_clicks" not in sel: sel["last_clicks"] = [[] for _ in range(N_SLOTS)]
            x, y = int(evt.index[0]), int(evt.index[1])
            rgb  = np.array(orig)[..., :3]
            if "LASSO" in mode_str:
                sel["lasso"][active] = sel["lasso"][active] + [(x,y)]
            elif "REMOVE" in mode_str:
                sel["masks"][active] = _sel_remove(sel["masks"][active], x, y)
                sel["last_clicks"][active] = (sel["last_clicks"][active] + [(x,y,"remove")])[-5:]
            else:
                sel["masks"][active] = _sel_add(rgb, sel["masks"][active], x, y)
                sel["last_clicks"][active] = (sel["last_clicks"][active] + [(x,y,"add")])[-5:]
            return sel, _overlay_masks(rgb,sel,active), _slot_label_html(sel,active)

        upload_img.select(on_click,
            [orig_img_state, sel_state, active_obj, click_mode_radio],
            [sel_state, upload_img, slot_status_html])

        # ── slot change ───────────────────────────────────────────────────────
        def on_slot(orig, sel, idx):
            active = int(idx)
            return active, _slot_label_html(sel,active), _ov(orig,sel,active)

        active_slot_radio.change(on_slot,
            [orig_img_state, sel_state, active_slot_radio],
            [active_obj, slot_status_html, upload_img])

        # ── auto-detect ───────────────────────────────────────────────────────
        def on_auto(orig, sel, active):
            import copy; sel = copy.deepcopy(sel)
            if orig is None: return sel, None, _slot_label_html(sel,active)
            rgb  = np.array(orig)[..., :3]
            objs = _auto_detect_objects(rgb)
            for i in range(N_SLOTS): sel["masks"][i]=None; sel["last_clicks"][i]=[]
            for i,m in enumerate(objs[:N_SLOTS]): sel["masks"][i]=m
            return sel, _overlay_masks(rgb,sel,active), _slot_label_html(sel,active)

        btn_auto.click(on_auto, [orig_img_state, sel_state, active_obj],
                       [sel_state, upload_img, slot_status_html])

        # ── lasso apply / clear ───────────────────────────────────────────────
        def on_apply_lasso(orig, sel, active, mode_str):
            import copy; sel = copy.deepcopy(sel)
            if orig is None: return sel, None, _slot_label_html(sel,active)
            rgb = np.array(orig)[..., :3]
            pts = sel["lasso"][active]
            if len(pts) >= 3:
                mode = "remove" if "REMOVE" in mode_str else "add"
                h,w  = rgb.shape[:2]
                sel["masks"][active] = _apply_lasso(sel["masks"][active], pts, h, w, mode)
            sel["lasso"][active] = []
            return sel, _overlay_masks(rgb,sel,active), _slot_label_html(sel,active)

        btn_apply_lasso.click(on_apply_lasso,
            [orig_img_state, sel_state, active_obj, click_mode_radio],
            [sel_state, upload_img, slot_status_html])

        def on_clear_lasso(orig, sel, active):
            import copy; sel = copy.deepcopy(sel)
            sel["lasso"][active] = []
            return sel, _ov(orig,sel,active), _slot_label_html(sel,active)

        btn_clear_lasso.click(on_clear_lasso,
            [orig_img_state, sel_state, active_obj],
            [sel_state, upload_img, slot_status_html])

        # ── clear slot / all ──────────────────────────────────────────────────
        def on_clr_slot(orig, sel, active):
            import copy; sel = copy.deepcopy(sel)
            sel["masks"][active] = None
            sel["last_clicks"][active] = []
            sel["lasso"][active] = []
            return sel, _ov(orig,sel,active), _slot_label_html(sel,active)

        btn_clr_slot.click(on_clr_slot, [orig_img_state, sel_state, active_obj],
                           [sel_state, upload_img, slot_status_html])

        def on_clr_all(orig, sel):
            new_sel = _empty_sel()
            return new_sel, _ov(orig,new_sel,0), _slot_label_html(new_sel,0)

        btn_clr_all.click(on_clr_all, [orig_img_state, sel_state],
                          [sel_state, upload_img, slot_status_html])

        # ── RUN ───────────────────────────────────────────────────────────────
        def start_gen(img, sel, active, out_folder, seed, s1, s2):
            global _gen_thread, _is_running
            if _is_running:
                return "⚠ Already running", gr.update(), gr.update()
            if img is None:
                return "⚠ Upload an image first", gr.update(), gr.update()
            masks = [m for m in sel["masks"] if m is not None and m.any()]
            if not masks:
                if img is None:
                    return "⚠ Upload an image first", gr.update(), gr.update()
                rgb  = np.array(img)[..., :3]
                objs = _auto_detect_objects(rgb)
                masks = [objs[0]] if objs else []
            if not masks:
                return "⚠ No objects selected", gr.update(), gr.update()
            out_folder = (out_folder or "").strip() or DEFAULT_OUT
            os.makedirs(out_folder, exist_ok=True)
            with _EV_LOCK: _EVENTS.clear()
            with _ASSET_LOCK: _ASSETS.clear()
            _prev_assets.clear()
            _STATS.update({"stage":"init","obj":0,"n_obj":len(masks),"start":None})
            _is_running = True
            _gen_thread = threading.Thread(
                target=_gen_worker,
                args=(np.array(img)[..., :3], masks, CONFIG_PATH,
                      out_folder, int(seed), int(s1), int(s2)),
                daemon=True)
            _gen_thread.start()
            return (f"⚡ {len(masks)} object(s) → {out_folder}",
                    gr.update(value=_tline("init")),
                    gr.update(value=""))

        run_btn.click(start_gen,
            inputs=[orig_img_state,sel_state,active_obj,out_folder_box,
                    seed,stage1_steps,stage2_steps],
            outputs=[status_box,timeline_html,log_box])

        # ── EXTRACT ONLY ──────────────────────────────────────────────────────
        def extract_only(img, sel, out_folder):
            if img is None:
                return "⚠ Upload an image first", gr.update()
            rgb = np.array(img)[..., :3]
            masks = [m for m in sel["masks"] if m is not None and m.any()]
            if not masks:
                masks = _auto_detect_objects(rgb)
            if not masks:
                return "⚠ No objects detected", gr.update()
            out_folder = (out_folder or "").strip() or DEFAULT_OUT
            os.makedirs(out_folder, exist_ok=True)
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            saved = []
            for idx, mask in enumerate(masks):
                obj_dir = os.path.join(out_folder, f"{timestamp}_extract_obj{idx+1}")
                os.makedirs(obj_dir, exist_ok=True)
                PILImage.fromarray(rgb).save(os.path.join(obj_dir, "original.png"))
                rgba = np.concatenate([rgb, (mask * 255).astype(np.uint8)[..., None]], -1)
                p = os.path.join(obj_dir, "extracted.png")
                PILImage.fromarray(rgba).save(p)
                saved.append(p)
            msg = f"✓ Extracted {len(saved)} object(s) → {out_folder}"
            # show first extracted image in the upload slot for inspection
            first = np.array(PILImage.open(saved[0]).convert("RGB"))
            return msg, gr.update(value=first)

        extract_btn.click(extract_only,
            inputs=[orig_img_state, sel_state, out_folder_box],
            outputs=[status_box, upload_img])

        # ── archive ───────────────────────────────────────────────────────────
        def refresh_arc(folder):
            runs  = scan_output_folder(folder)
            items = _gallery_items(runs)
            return runs, items

        btn_refresh.click(refresh_arc, archive_folder, [runs_state,archive_gallery])
        out_folder_box.change(lambda v: gr.update(value=v), out_folder_box, archive_folder)

        def on_arc_select(runs, evt: gr.SelectData):
            idx = evt.index
            if not runs or idx >= len(runs):
                return "<p>No data</p>", None, None, None, None, None, None
            r   = runs[idx]
            glb = r["glb"] or next((s["glb"] for s in r.get("sub_objects",[]) if s.get("glb")),None)
            ply = r["ply"] or next((s["ply"] for s in r.get("sub_objects",[]) if s.get("ply")),None)
            vox = r.get("voxel") or next((s.get("voxel") for s in r.get("sub_objects",[]) if s.get("voxel")),None)
            return (_run_detail_html(r),
                    r["extracted"] or r["original"],
                    vox, ply, glb, ply, glb)

        archive_gallery.select(on_arc_select, [runs_state],
            [run_detail_html,archive_extracted,
             archive_voxel,archive_splat,archive_glb_view,
             arc_dl_ply,arc_dl_glb])

        demo.load(lambda: (scan_output_folder(DEFAULT_OUT),
                           _gallery_items(scan_output_folder(DEFAULT_OUT))),
                  outputs=[runs_state,archive_gallery])

    return demo


if __name__ == "__main__":
    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        css=CSS,
        allowed_paths=[DEFAULT_OUT, ROOT],
    )
