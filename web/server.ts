import { Database } from "bun:sqlite";
import { mkdirSync, unlinkSync } from "node:fs";
import { drizzle } from "drizzle-orm/bun-sqlite";
import { eq, desc } from "drizzle-orm";
import { events } from "./src/schema";

const DB_PATH = process.env.DB_PATH ?? "/srv/tracker/tracker.db";
const CAPTURES_DIR = process.env.CAPTURES_DIR ?? "/srv/tracker/captures";
const TOKEN = process.env.TRACKER_TOKEN ?? "";
const DASH_USER = process.env.DASH_USER ?? "";
const DASH_PASS = process.env.DASH_PASS ?? "";
const PORT = Number(process.env.PORT ?? 3010);
const HOST = process.env.HOST ?? "127.0.0.1";

mkdirSync(`${CAPTURES_DIR}/thumb`, { recursive: true });

const sqlite = new Database(DB_PATH, { create: true });
sqlite.exec(`
CREATE TABLE IF NOT EXISTS events (
  id          TEXT PRIMARY KEY,
  track_id    INTEGER,
  class_id    INTEGER,
  label       TEXT,
  confidence  REAL,
  direction   TEXT,
  crossed_at  TEXT,
  bbox        TEXT,
  line        TEXT,
  image_path  TEXT,
  thumb_path  TEXT,
  created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS camera_config (
  id         INTEGER PRIMARY KEY CHECK (id = 1),
  data       TEXT NOT NULL,
  updated_at TEXT NOT NULL
);`);
const db = drizzle(sqlite);

// ── Camera config (normalized 0..1 coords; set in dashboard, pulled by Pi) ──
type CamConfig = {
  line: number[][] | null;
  roi: number[][] | null;
  flip_sides: boolean;
  enabled_classes: number[];
  source_w: number | null;
  source_h: number | null;
};
const DEFAULT_CONFIG: CamConfig = {
  line: null, roi: null, flip_sides: false,
  enabled_classes: [2, 3, 5, 7], source_w: null, source_h: null,
};

function getConfig(): CamConfig & { updated_at: string | null } {
  const row = sqlite
    .query("SELECT data, updated_at FROM camera_config WHERE id = 1")
    .get() as { data: string; updated_at: string } | null;
  if (!row) return { ...DEFAULT_CONFIG, updated_at: null };
  try {
    return { ...DEFAULT_CONFIG, ...JSON.parse(row.data), updated_at: row.updated_at };
  } catch {
    return { ...DEFAULT_CONFIG, updated_at: row.updated_at };
  }
}

function saveConfig(c: CamConfig) {
  const now = new Date().toISOString();
  sqlite
    .query(`INSERT INTO camera_config (id, data, updated_at) VALUES (1, ?, ?)
            ON CONFLICT(id) DO UPDATE SET data = excluded.data, updated_at = excluded.updated_at`)
    .run(JSON.stringify(c), now);
  return now;
}

const num = (v: FormDataEntryValue | null) => (v == null || v === "" ? null : Number(v));
const str = (v: FormDataEntryValue | null) => (v == null ? null : String(v));
const esc = (s: unknown) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]!));

// ── Live frame relay (Pi POSTs /api/live; browsers read /live.mjpg) ──────
let liveFrame: Uint8Array | null = null;
let liveAt = 0;
let liveSeq = 0;
let liveViewers = 0;   // active /live.mjpg clients
let liveJpgAt = 0;     // last /live.jpg request (setup editor snapshot)

function bearerOk(req: Request) {
  return !TOKEN || req.headers.get("authorization") === `Bearer ${TOKEN}`;
}

function basicOk(req: Request) {
  if (!DASH_USER) return true;
  const h = req.headers.get("authorization") ?? "";
  if (!h.startsWith("Basic ")) return false;
  const [u, p] = Buffer.from(h.slice(6), "base64").toString().split(":");
  return u === DASH_USER && p === DASH_PASS;
}

function unauthorized() {
  return new Response("Authentication required", {
    status: 401,
    headers: { "WWW-Authenticate": 'Basic realm="Vehicle Tracker", charset="UTF-8"' },
  });
}

async function serveImage(id: string, thumb: boolean) {
  const row = db.select().from(events).where(eq(events.id, id)).get();
  const rel = thumb ? row?.thumbPath : row?.imagePath;
  if (!rel) return new Response("not found", { status: 404 });
  const file = Bun.file(`${CAPTURES_DIR}/${rel}`);
  if (!(await file.exists())) return new Response("not found", { status: 404 });
  return new Response(file);
}

type EventRow = {
  id: string; label: string | null; direction: string | null;
  track_id: number | null; crossed_at: string | null; created_at: string;
};

function dashboard(initialDay: string | null) {
  const dayExpr = "date(COALESCE(crossed_at, created_at))";
  const counts = sqlite
    .query("SELECT direction, COUNT(*) AS c FROM events GROUP BY direction")
    .all() as { direction: string; c: number }[];
  const cIn = counts.find((r) => r.direction === "in")?.c ?? 0;
  const cOut = counts.find((r) => r.direction === "out")?.c ?? 0;

  const days = sqlite
    .query(`SELECT ${dayExpr} AS d, SUM(direction='in') AS inn, SUM(direction='out') AS outt
            FROM events GROUP BY d`)
    .all() as { d: string; inn: number; outt: number }[];
  const dayStats = Object.fromEntries(days.map((d) => [d.d, { in: d.inn ?? 0, out: d.outt ?? 0 }]));

  const rows = sqlite
    .query(`SELECT id,label,direction,track_id,crossed_at,created_at FROM events
            ORDER BY created_at DESC LIMIT 500`)
    .all() as EventRow[];
  const events = rows.map((r) => ({
    id: r.id, label: r.label ?? "", dir: r.direction ?? "", track: r.track_id,
    ts: r.crossed_at || r.created_at || "",
  }));
  const init = initialDay && /^\d{4}-\d{2}-\d{2}$/.test(initialDay) ? initialDay : null;
  const J = (o: unknown) => JSON.stringify(o).replace(/</g, "\\u003c");

  return `<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#ffffff">
<title>Vehicle Tracker</title>
<style>
:root{--bg:#f4f5f7;--card:#fff;--line:#e6e8ec;--text:#111827;--muted:#6b7280;--green:#16a34a;--red:#dc2626;--blue:#2563eb}
*{box-sizing:border-box}
body{margin:0;font:15px/1.45 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif;background:var(--bg);color:var(--text);
  padding-bottom:calc(64px + env(safe-area-inset-bottom));-webkit-text-size-adjust:100%}
button{font:inherit;color:inherit}
a{color:var(--blue);text-decoration:none}
header{position:sticky;top:0;z-index:20;background:rgba(255,255,255,.94);backdrop-filter:blur(10px);border-bottom:1px solid var(--line)}
.bar{display:flex;align-items:center;gap:10px;padding:10px 14px 4px}
.title{font-weight:700;font-size:16px}
.live{margin-left:auto;display:flex;align-items:center;gap:6px;font-size:12px;color:var(--muted)}
.dot{width:8px;height:8px;border-radius:50%;background:#9ca3af}
.dot.on{background:var(--green)}
.icon{width:42px;height:42px;border:1px solid var(--line);background:#fff;border-radius:12px;font-size:18px;display:flex;align-items:center;justify-content:center;cursor:pointer}
.todaylbl{padding:2px 15px 0;font-size:12px;font-weight:700;color:var(--muted);letter-spacing:.02em}
.stats{display:flex;gap:10px;padding:6px 14px 12px}
.stat{flex:1;background:#fbfcfe;border:1px solid var(--line);border-radius:14px;padding:10px 12px;display:flex;align-items:baseline}
.stat .lbl{font-size:12px;font-weight:700;letter-spacing:.06em}
.stat .num{margin-left:auto;font-size:26px;font-weight:800;font-variant-numeric:tabular-nums}
.stat.in .lbl,.stat.in .num{color:var(--green)}
.stat.out .lbl,.stat.out .num{color:var(--red)}
main{max-width:860px;margin:0 auto;padding:12px}
.view.hidden{display:none}
.player{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:16px;overflow:hidden}
.player img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:translateZ(0)}
.sub{margin:8px 2px 0;color:var(--muted);font-size:12px;text-align:center}
.calhead{display:flex;align-items:center;gap:8px;margin-bottom:10px}
#calTitle{font-weight:700;font-size:16px}
.pill{border:1px solid var(--line);background:#fff;border-radius:999px;padding:9px 16px;cursor:pointer}
.calgrid{display:grid;grid-template-columns:repeat(7,1fr);gap:5px}
.calgrid .dow{font-size:11px;color:var(--muted);text-align:center;padding:2px 0}
.cell{min-height:54px;border:1px solid var(--line);border-radius:12px;background:#fff;padding:5px 6px;display:flex;flex-direction:column;justify-content:space-between;cursor:pointer}
.cell.empty{border:none;background:transparent;cursor:default}
.cell.today{border-color:var(--blue)}
.cell.sel{background:#dbeafe;border-color:var(--blue)}
.cell .n{font-size:12px;font-weight:700}
.cell.future{opacity:.35;cursor:default}
.cell .c{font-size:11px;line-height:1.1;display:flex;gap:8px}
.cell .ci{color:var(--green);font-weight:800}
.cell .co{color:var(--red);font-weight:800}
.icon:disabled{opacity:.35;cursor:default}
.histhead{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap}
.histhead .d{font-weight:700}
.histhead .c{font-size:12px;color:var(--muted)}
.histhead .pill{margin-left:auto}
.gallery{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:10px}
figure{margin:0;background:#fff;border:1px solid var(--line);border-radius:14px;overflow:hidden}
figure img{width:100%;display:block;aspect-ratio:16/9;object-fit:cover;background:#e5e7eb}
figcaption{padding:7px 9px;font-size:12px;display:flex;justify-content:space-between;gap:6px}
figcaption .t{color:var(--muted);font-size:11px}
.empty2{color:var(--muted);text-align:center;padding:48px 20px}
.tabbar{position:fixed;left:0;right:0;bottom:0;z-index:30;display:flex;background:rgba(255,255,255,.97);backdrop-filter:blur(10px);border-top:1px solid var(--line);padding-bottom:env(safe-area-inset-bottom)}
.tabbar button{flex:1;border:none;background:none;padding:14px 0;font-size:13px;color:var(--muted);cursor:pointer}
.tabbar button.active{color:var(--blue);font-weight:700}
.sheet{position:fixed;inset:0;z-index:40;background:var(--bg);overflow:auto;padding:calc(12px + env(safe-area-inset-top)) 12px calc(12px + env(safe-area-inset-bottom))}
.sheet.hidden{display:none}
.sheetbar{display:flex;align-items:center;margin-bottom:10px}
.sheetbar b{font-size:16px}
.sheetbar .icon{margin-left:auto}
.editor{position:relative;width:100%;aspect-ratio:16/9;background:#000;border-radius:14px;overflow:hidden;cursor:crosshair}
.editor img{position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:translateZ(0)}
.editor svg{position:absolute;inset:0;width:100%;height:100%;transform:translateZ(0)}
.tools{display:flex;flex-wrap:wrap;gap:8px;margin-top:12px}
.tools button{border:1px solid var(--line);background:#fff;border-radius:12px;padding:11px 15px;cursor:pointer}
.tools button.active{background:var(--blue);border-color:var(--blue);color:#fff}
.tools button.primary{background:var(--green);border-color:var(--green);color:#fff}
.msg{font-size:12px;color:var(--muted);align-self:center}
.hint{margin-top:12px;color:var(--muted);font-size:12px}
</style></head><body>
<header>
  <div class="bar">
    <div class="title">Vehicle Tracker</div>
    <div class="live"><span class="dot" id="dot"></span><span id="ltxt">…</span></div>
    <button class="icon" id="setupBtn" title="Setup">&#9881;</button>
  </div>
  <div class="todaylbl" id="todayLbl">Today</div>
  <div class="stats">
    <div class="stat in"><span class="lbl">IN</span><span class="num" id="sIn">0</span></div>
    <div class="stat out"><span class="lbl">OUT</span><span class="num" id="sOut">0</span></div>
  </div>
</header>
<main>
  <section id="v-live" class="view">
    <div class="player"><img id="snap" alt="live"></div>
    <div class="sub">Tap the gear to edit the counting line / scan area</div>
  </section>
  <section id="v-cal" class="view hidden">
    <div class="calhead">
      <button class="icon" id="prevMonth">&#8249;</button>
      <div id="calTitle"></div>
      <button class="icon" id="nextMonth">&#8250;</button>
      <button class="pill" id="todayBtn">Today</button>
    </div>
    <div class="calgrid" id="calGrid"></div>
    <div class="hint"><b style="color:var(--green)">IN</b> / <b style="color:var(--red)">OUT</b> per day &middot; &#8592; / &#8594; change day &middot; tap a day for captures</div>
  </section>
  <section id="v-hist" class="view hidden">
    <div class="histhead" id="histHead"></div>
    <div class="gallery" id="hist"></div>
  </section>
</main>
<nav class="tabbar">
  <button data-tab="v-live" class="active">Live</button>
  <button data-tab="v-cal">Calendar</button>
  <button data-tab="v-hist">History</button>
</nav>
<div class="sheet hidden" id="setup">
  <div class="sheetbar"><b>Camera setup</b><button class="icon" id="closeSetup">&#10005;</button></div>
  <div class="editor" id="editor">
    <img id="snap2" alt="snapshot">
    <svg id="ov" viewBox="0 0 1 1" preserveAspectRatio="none"></svg>
  </div>
  <div class="tools">
    <button data-mode="line">Line</button>
    <button data-mode="area">Area</button>
    <button data-mode="in">IN side</button>
    <button id="reset">Reset</button>
    <button id="save" class="primary">Save</button>
    <span class="msg" id="msg"></span>
  </div>
  <p class="hint">Line: tap 2 points. Area: tap points, tap the first point (or Save) to close. IN side: tap where traffic enters. Then Save.</p>
</div>
<script>
const DAYS=${J(dayStats)};
const EVENTS=${J(events)};
const INIT=${J(init)};
</script>
<script>
(function(){
var S=document.getElementById('snap'),dot=document.getElementById('dot'),ltxt=document.getElementById('ltxt');
var tk=todayKey(),tc=DAYS[tk]||{};
document.getElementById('sIn').textContent=tc['in']||0;
document.getElementById('sOut').textContent=tc.out||0;
function clock(){var n=new Date();document.getElementById('todayLbl').textContent='Today · '+
  n.toLocaleDateString(undefined,{weekday:'short',day:'numeric',month:'short'})+' · '+
  n.toLocaleTimeString(undefined,{hour:'2-digit',minute:'2-digit',second:'2-digit'});}
clock();setInterval(clock,1000);
// Keep the IN/OUT header counts live (they used to only render on page load).
function refreshCounts(){fetch('/api/config',{cache:'no-store'})
  .then(function(r){return r.ok?r.json():null;})
  .then(function(d){if(!d||!d.today)return;
    document.getElementById('sIn').textContent=d.today['in']||0;
    document.getElementById('sOut').textContent=d.today.out||0;
    if(DAYS[d.today.date]){DAYS[d.today.date]['in']=d.today['in'];DAYS[d.today.date].out=d.today.out;}})
  .catch(function(){});}
refreshCounts();setInterval(refreshCounts,5000);

S.onload=function(){dot.classList.add('on');ltxt.textContent='LIVE';};
S.onerror=function(){dot.classList.remove('on');ltxt.textContent='idle';};
// Resilient live view: poll /live.jpg snapshots instead of a one-shot MJPEG
// connection (which never reconnects after a server restart). Polling also keeps
// the server's live_wanted flag alive so the camera keeps uploading.
var liveOn=false,pumpTimer=null,lastURL=null;
function liveTick(){
  if(!liveOn)return;
  fetch('/live.jpg',{cache:'no-store'}).then(function(r){if(!r.ok)throw 0;return r.blob();})
    .then(function(b){var u=URL.createObjectURL(b);S.src=u;
      if(lastURL)URL.revokeObjectURL(lastURL);lastURL=u;})
    .catch(function(){});
}
function setLive(on){if(on===liveOn)return;liveOn=on;
  if(on){liveTick();pumpTimer=setInterval(liveTick,250);}
  else if(pumpTimer){clearInterval(pumpTimer);pumpTimer=null;}}
var views=[].slice.call(document.querySelectorAll('.view'));
function showTab(id){views.forEach(function(v){v.classList.toggle('hidden',v.id!==id);});
  [].slice.call(document.querySelectorAll('.tabbar button')).forEach(function(b){b.classList.toggle('active',b.dataset.tab===id);});
  setLive(id==='v-live');
  window.scrollTo(0,0);}
[].slice.call(document.querySelectorAll('.tabbar button')).forEach(function(b){b.onclick=function(){showTab(b.dataset.tab);};});
showTab('v-live');
var selDay=INIT;
var view=INIT?new Date(INIT+'T00:00:00'):new Date();
function pad(n){return String(n).padStart(2,'0');}
function ymd(y,m,d){return y+'-'+pad(m+1)+'-'+pad(d);}
var calGrid=document.getElementById('calGrid'),calTitle=document.getElementById('calTitle');
function todayKey(){var n=new Date();return ymd(n.getFullYear(),n.getMonth(),n.getDate());}
function atOrAfterCurrentMonth(){var n=new Date();return view.getFullYear()>n.getFullYear()||
  (view.getFullYear()===n.getFullYear()&&view.getMonth()>=n.getMonth());}
function renderCal(){
  var y=view.getFullYear(),m=view.getMonth();
  calTitle.textContent=view.toLocaleString(undefined,{month:'long',year:'numeric'});
  var today=todayKey();
  var h='';['Mo','Tu','We','Th','Fr','Sa','Su'].forEach(function(d){h+='<div class="dow">'+d+'</div>';});
  var start=(new Date(y,m,1).getDay()+6)%7,dim=new Date(y,m+1,0).getDate();
  for(var i=0;i<start;i++)h+='<div class="cell empty"></div>';
  for(var d=1;d<=dim;d++){var k=ymd(y,m,d),c=DAYS[k]||{},future=k>today;
    var cls='cell'+(future?' future':'')+(k===today?' today':'')+(k===selDay?' sel':'');
    h+='<div class="'+cls+'"'+(future?'':' data-day="'+k+'"')+'><span class="n">'+d+'</span><span class="c">'+
      '<b class="ci">'+(c['in']||0)+'</b><b class="co">'+(c.out||0)+'</b></span></div>';}
  calGrid.innerHTML=h;
  [].slice.call(calGrid.querySelectorAll('[data-day]')).forEach(function(el){
    el.onclick=function(){selDay=el.dataset.day;view=new Date(selDay+'T00:00:00');renderCal();renderHist();showTab('v-hist');};});
  document.getElementById('nextMonth').disabled=atOrAfterCurrentMonth();
}
function shiftDay(n){var b=selDay?new Date(selDay+'T00:00:00'):new Date();b.setDate(b.getDate()+n);
  var k=ymd(b.getFullYear(),b.getMonth(),b.getDate());if(k>todayKey())return;
  selDay=k;view=new Date(selDay+'T00:00:00');renderCal();renderHist();}
document.getElementById('prevMonth').onclick=function(){view.setMonth(view.getMonth()-1);renderCal();};
document.getElementById('nextMonth').onclick=function(){if(atOrAfterCurrentMonth())return;view.setMonth(view.getMonth()+1);renderCal();};
document.getElementById('todayBtn').onclick=function(){var t=new Date();selDay=ymd(t.getFullYear(),t.getMonth(),t.getDate());view=new Date(selDay+'T00:00:00');renderCal();renderHist();};
document.addEventListener('keydown',function(e){if(e.target&&/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;
  if(e.key==='ArrowLeft'){shiftDay(-1);e.preventDefault();}else if(e.key==='ArrowRight'){shiftDay(1);e.preventDefault();}});
var hist=document.getElementById('hist'),histHead=document.getElementById('histHead');
function fmt(ts){return (ts||'').replace('T',' ').slice(0,19);}
function renderHist(){
  var list=selDay?EVENTS.filter(function(e){return (e.ts||'').slice(0,10)===selDay;}):EVENTS;
  var c=selDay?(DAYS[selDay]||{in:0,out:0}):{in:${cIn},out:${cOut}};
  var ctrls=selDay?'<button class="pill" id="clearDay" style="margin-left:auto">All</button>':'';
  histHead.innerHTML='<span class="d">'+(selDay?selDay:'All recent')+'</span>'+
    '<span class="c"><b style="color:var(--green)">'+c['in']+' IN</b> &middot; <b style="color:var(--red)">'+c.out+' OUT</b></span>'+ctrls;
  if(selDay)document.getElementById('clearDay').onclick=function(){selDay=null;renderCal();renderHist();};
  hist.innerHTML=list.length?list.map(function(e){
    var col=e.dir.toLowerCase()==='in'?'var(--green)':'var(--red)';
    return '<figure><a href="/img/'+encodeURIComponent(e.id)+'" target="_blank">'+
      '<img loading="lazy" src="/thumb/'+encodeURIComponent(e.id)+'" alt=""></a>'+
      '<figcaption><span style="color:'+col+';font-weight:700">'+(e.dir||'').toUpperCase()+' '+e.label+'</span>'+
      '<span class="t">'+fmt(e.ts).slice(5,16)+'</span></figcaption></figure>';
  }).join(''):'<div class="empty2">No captures</div>';
}
renderCal();renderHist();
var setup=document.getElementById('setup'),S2=document.getElementById('snap2'),O=document.getElementById('ov'),msg=document.getElementById('msg');
document.getElementById('setupBtn').onclick=function(){setup.classList.remove('hidden');S2.src='/live.jpg?'+Date.now();};
document.getElementById('closeSetup').onclick=function(){setup.classList.add('hidden');};
var cfg={line:null,roi:null,flip_sides:false,enabled_classes:[2,3,5,7]};
var mode=null,dl=[],da=[];
var NS='http://www.w3.org/2000/svg';
function el(t,a){var e=document.createElementNS(NS,t);for(var k in a)e.setAttribute(k,a[k]);
  e.setAttribute('vector-effect','non-scaling-stroke');return e;}
var elPoly=el('polygon',{fill:'rgba(100,180,255,.15)',stroke:'#64b4ff','stroke-width':2});
var elDraft=el('polyline',{fill:'none',stroke:'#f59e0b','stroke-width':2,'stroke-dasharray':'4'});
var elLine=el('line',{stroke:'#3b82f6','stroke-width':3});
var elIn=el('text',{fill:'#16a34a','font-size':'0.05','font-weight':'700','text-anchor':'middle'});elIn.textContent='IN';
var elOut=el('text',{fill:'#dc2626','font-size':'0.05','font-weight':'700','text-anchor':'middle'});elOut.textContent='OUT';
var elPts=el('g',{fill:'#f59e0b'});
O.append(elPoly,elDraft,elLine,elIn,elOut,elPts);
function tp(a){return a.map(function(p){return p[0]+','+p[1];}).join(' ');}
function sh(e,v){e.style.display=v?'':'none';}
function draw(){
  var hasRoi=cfg.roi&&cfg.roi.length>=3;
  sh(elPoly,hasRoi);if(hasRoi)elPoly.setAttribute('points',tp(cfg.roi));
  sh(elDraft,da.length>=2);if(da.length>=2)elDraft.setAttribute('points',tp(da));
  var hasLine=cfg.line&&cfg.line.length===2;
  sh(elLine,hasLine);sh(elIn,hasLine);sh(elOut,hasLine);
  if(hasLine){var A=cfg.line[0],B=cfg.line[1];
    elLine.setAttribute('x1',A[0]);elLine.setAttribute('y1',A[1]);
    elLine.setAttribute('x2',B[0]);elLine.setAttribute('y2',B[1]);
    var dx=B[0]-A[0],dy=B[1]-A[1],L=Math.hypot(dx,dy)||1;
    var ux=-dy/L,uy=dx/L;if(cfg.flip_sides){ux=-ux;uy=-uy;}
    elIn.setAttribute('x',(A[0]+B[0])/2+ux*0.06);elIn.setAttribute('y',(A[1]+B[1])/2+uy*0.06);
    elOut.setAttribute('x',(A[0]+B[0])/2-ux*0.06);elOut.setAttribute('y',(A[1]+B[1])/2-uy*0.06);}
  while(elPts.firstChild)elPts.removeChild(elPts.firstChild);
  dl.forEach(function(p){elPts.appendChild(el('circle',{cx:p[0],cy:p[1],r:0.008}));});
}
function xy(ev){var r=O.getBoundingClientRect();return [(ev.clientX-r.left)/r.width,(ev.clientY-r.top)/r.height];}
function side(line,p){var A=line[0],B=line[1];return (B[0]-A[0])*(p[1]-A[1])-(B[1]-A[1])*(p[0]-A[0]);}
function setMode(m){mode=m;[].slice.call(document.querySelectorAll('button[data-mode]')).forEach(function(b){b.classList.toggle('active',b.dataset.mode===m);});}
O.addEventListener('click',function(ev){var p=xy(ev);
  if(mode==='line'){dl.push(p);if(dl.length===2){cfg.line=dl.slice();dl=[];setMode(null);}}
  else if(mode==='area'){if(da.length>=3&&Math.hypot(p[0]-da[0][0],p[1]-da[0][1])<0.05){cfg.roi=da.slice();da=[];setMode(null);}else da.push(p);}
  else if(mode==='in'&&cfg.line){cfg.flip_sides=side(cfg.line,p)<0;setMode(null);}
  draw();});
[].slice.call(document.querySelectorAll('button[data-mode]')).forEach(function(b){b.onclick=function(){if(b.dataset.mode==='line')dl=[];setMode(b.dataset.mode);draw();};});
document.getElementById('reset').onclick=function(){cfg.line=null;cfg.roi=null;dl=[];da=[];setMode(null);draw();};
document.getElementById('save').onclick=async function(){
  if(da.length>=3){cfg.roi=da.slice();da=[];setMode(null);draw();}
  msg.textContent='saving…';
  try{var r=await fetch('/api/config',{method:'POST',headers:{'content-type':'application/json'},
    body:JSON.stringify({line:cfg.line,roi:cfg.roi,flip_sides:cfg.flip_sides,enabled_classes:cfg.enabled_classes,source_w:S2.naturalWidth,source_h:S2.naturalHeight})});
    msg.textContent=r.ok?'saved ✓':'save failed';}catch(e){msg.textContent='save failed';}
};
fetch('/api/config').then(function(r){return r.json();}).then(function(c){cfg=Object.assign({},cfg,c);draw();}).catch(function(){});
})();
</script>
</body></html>`;
}

// ── Retention: events (and their images) older than RETENTION_DAYS auto-expire ──
const RETENTION_DAYS = Number(process.env.RETENTION_DAYS ?? 30);
function pruneOldEvents() {
  const cutoff = new Date(Date.now() - RETENTION_DAYS * 86400000).toISOString().slice(0, 10);
  const old = sqlite
    .query(`SELECT image_path, thumb_path FROM events
            WHERE date(COALESCE(crossed_at, created_at)) < ?`)
    .all(cutoff) as { image_path: string | null; thumb_path: string | null }[];
  if (!old.length) return 0;
  for (const r of old) {
    for (const rel of [r.image_path, r.thumb_path]) {
      if (rel) { try { unlinkSync(`${CAPTURES_DIR}/${rel}`); } catch {} }
    }
  }
  sqlite.query(`DELETE FROM events WHERE date(COALESCE(crossed_at, created_at)) < ?`).run(cutoff);
  console.log(`[tracker-web] retention: pruned ${old.length} events older than ${cutoff}`);
  return old.length;
}
setInterval(pruneOldEvents, 3_600_000);
pruneOldEvents();

Bun.serve({
  port: PORT,
  hostname: HOST,
  async fetch(req) {
    const url = new URL(req.url);

    // Machine API — Bearer token
    if (url.pathname === "/api/ingest" && req.method === "POST") {
      if (!bearerOk(req)) return new Response("unauthorized", { status: 401 });
      const form = await req.formData();
      const id = String(form.get("id") ?? "");
      if (!id) return new Response("missing id", { status: 400 });
      const image = form.get("image");
      const thumb = form.get("thumb");
      const imagePath = image instanceof File ? `${id}.jpg` : "";
      const thumbPath = thumb instanceof File ? `thumb/${id}.jpg` : "";
      if (image instanceof File) await Bun.write(`${CAPTURES_DIR}/${imagePath}`, image);
      if (thumb instanceof File) await Bun.write(`${CAPTURES_DIR}/${thumbPath}`, thumb);
      db.insert(events)
        .values({
          id,
          trackId: num(form.get("track_id")),
          classId: num(form.get("class_id")),
          label: str(form.get("label")),
          confidence: num(form.get("confidence")),
          direction: str(form.get("direction")),
          crossedAt: str(form.get("crossed_at")),
          bbox: str(form.get("bbox")),
          line: str(form.get("line")),
          imagePath,
          thumbPath,
          createdAt: new Date().toISOString(),
        })
        .onConflictDoNothing()
        .run();
      return Response.json({ ok: true, id });
    }

    // Live frame push — Bearer token (Pi → VPS)
    if (url.pathname === "/api/live" && req.method === "POST") {
      if (!bearerOk(req)) return new Response("unauthorized", { status: 401 });
      const buf = await req.arrayBuffer();
      if (buf.byteLength > 0) {
        liveFrame = new Uint8Array(buf);
        liveAt = Date.now();
        liveSeq++;
      }
      return Response.json({ ok: true, bytes: buf.byteLength, at: liveAt, seq: liveSeq });
    }

    // Camera config — Pi pulls with Bearer, dashboard reads/writes with Basic
    if (url.pathname === "/api/config" && req.method === "GET") {
      if (!bearerOk(req) && !basicOk(req)) return unauthorized();
      // The Pi uses live_wanted to only encode/upload while someone is watching.
      const live_wanted = liveViewers > 0 || Date.now() - liveJpgAt < 10_000;
      const cnt = sqlite
        .query("SELECT direction, COUNT(*) AS c FROM events GROUP BY direction")
        .all() as { direction: string; c: number }[];
      const allIn = cnt.find((r) => r.direction === "in")?.c ?? 0;
      const allOut = cnt.find((r) => r.direction === "out")?.c ?? 0;
      // Today's totals so the dashboard header can refresh without a reload.
      const dayExpr = "date(COALESCE(crossed_at, created_at))";
      const tcnt = sqlite
        .query(`SELECT direction, COUNT(*) AS c FROM events WHERE ${dayExpr} = date('now') GROUP BY direction`)
        .all() as { direction: string; c: number }[];
      const tIn = tcnt.find((r) => r.direction === "in")?.c ?? 0;
      const tOut = tcnt.find((r) => r.direction === "out")?.c ?? 0;
      return Response.json({
        ...getConfig(), viewers: liveViewers, live_wanted,
        counts: { in: allIn, out: allOut },
        today: { date: new Date().toISOString().slice(0, 10), in: tIn, out: tOut },
      });
    }
    if (url.pathname === "/api/config" && req.method === "POST") {
      if (!basicOk(req)) return unauthorized();
      let body: Partial<CamConfig>;
      try {
        body = (await req.json()) as Partial<CamConfig>;
      } catch {
        return new Response("bad json", { status: 400 });
      }
      const norm = (pts: unknown): number[][] | null => {
        if (!Array.isArray(pts) || pts.length < 2) return null;
        const out = pts
          .filter((p) => Array.isArray(p) && p.length === 2)
          .map((p) => [Math.min(1, Math.max(0, Number(p[0]))), Math.min(1, Math.max(0, Number(p[1])))]);
        return out.length >= 2 ? out : null;
      };
      const cfg: CamConfig = {
        line: norm(body.line),
        roi: norm(body.roi),
        flip_sides: Boolean(body.flip_sides),
        enabled_classes: Array.isArray(body.enabled_classes)
          ? body.enabled_classes.map(Number).filter((n) => Number.isFinite(n))
          : DEFAULT_CONFIG.enabled_classes,
        source_w: Number.isFinite(body.source_w) ? Number(body.source_w) : null,
        source_h: Number.isFinite(body.source_h) ? Number(body.source_h) : null,
      };
      const updated_at = saveConfig(cfg);
      return Response.json({ ok: true, updated_at, config: cfg });
    }

    // Browser routes — Basic auth
    if (!basicOk(req)) return unauthorized();

    if (url.pathname === "/live.jpg") {
      liveJpgAt = Date.now();
      if (!liveFrame || Date.now() - liveAt > 10_000) return new Response("no live feed", { status: 404 });
      return new Response(liveFrame, {
        headers: {
          "content-type": "image/jpeg",
          "cache-control": "no-store",
          "x-live-seq": String(liveSeq),
          "x-live-age": String(Date.now() - liveAt),
        },
      });
    }

    if (url.pathname === "/live.mjpg") {
      const encoder = new TextEncoder();
      const boundary = "frame";
      liveViewers++;
      let released = false;
      let timer: ReturnType<typeof setInterval> | null = null;
      const release = () => {
        if (released) return;
        released = true;
        liveViewers = Math.max(0, liveViewers - 1);
        if (timer) clearInterval(timer);
      };
      const stream = new ReadableStream({
        start(controller) {
          let sentSeq = -1;
          timer = setInterval(() => {
            if (!liveFrame || Date.now() - liveAt > 10_000) return;
            // Best-effort: only push genuinely new frames, and drop them when
            // this client is falling behind instead of queueing latency.
            if (liveSeq === sentSeq) return;
            if (controller.desiredSize !== null && controller.desiredSize <= 0) return;
            sentSeq = liveSeq;
            const head = encoder.encode(
              `--${boundary}\r\nContent-Type: image/jpeg\r\nContent-Length: ${liveFrame.length}\r\n\r\n`);
            const tail = encoder.encode("\r\n");
            const part = new Uint8Array(head.length + liveFrame.length + tail.length);
            part.set(head, 0);
            part.set(liveFrame, head.length);
            part.set(tail, head.length + liveFrame.length);
            try {
              controller.enqueue(part);
            } catch {
              release();
            }
          }, 100);
          const stop = () => { release(); try { controller.close(); } catch {} };
          req.signal.addEventListener("abort", stop);
        },
        cancel() { release(); },
      });
      return new Response(stream, {
        headers: {
          "content-type": `multipart/x-mixed-replace; boundary=${boundary}`,
          "cache-control": "no-cache, no-store, must-revalidate",
          "x-accel-buffering": "no",
        },
      });
    }

    if (url.pathname === "/")
      return new Response(dashboard(url.searchParams.get("day")), {
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "no-store, no-cache, must-revalidate",
          "pragma": "no-cache",
        },
      });
    if (url.pathname.startsWith("/img/")) return serveImage(decodeURIComponent(url.pathname.slice(5)), false);
    if (url.pathname.startsWith("/thumb/")) return serveImage(decodeURIComponent(url.pathname.slice(7)), true);
    return new Response("not found", { status: 404 });
  },
});

console.log(`[tracker-web] :${PORT} db=${DB_PATH} captures=${CAPTURES_DIR} auth=${DASH_USER ? "on" : "off"}`);
