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
// A browser shell into the edge is powerful and risky, so /terminal is opt-in:
// set ENABLE_TERMINAL=1 in /etc/tracker/web.env to turn it on.
const ENABLE_TERMINAL = process.env.ENABLE_TERMINAL === "1";
// Second (raw) camera on the live page. The Pi relays its substream as JPEGs;
// this only serves them. Enable with CAM2=1 in web.env.
const CAM2 = process.env.CAM2 === "1";

// Dashboard display + day-bucketing timezone: IST (UTC+5:30). Timestamps are
// stored in UTC; shift only for grouping/labels so calendar days and "today"
// totals line up with local (IST) days. SQLite parses the ISO offset then adds
// the shift.
const IST_SHIFT = "+330 minutes";
const istDay = (col: string) => `date(${col}, '${IST_SHIFT}')`;
const istNow = () => new Date(Date.now() + 330 * 60_000).toISOString().slice(0, 10);

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
  enabled_classes: [2, 5, 7], source_w: null, source_h: null,
};
const TWO_WHEELER_CLASSES = new Set([1, 3]); // COCO bicycle and motorcycle

function getConfig(): CamConfig & { updated_at: string | null } {
  const row = sqlite
    .query("SELECT data, updated_at FROM camera_config WHERE id = 1")
    .get() as { data: string; updated_at: string } | null;
  if (!row) return { ...DEFAULT_CONFIG, updated_at: null };
  try {
    const stored = JSON.parse(row.data);
    return {
      ...DEFAULT_CONFIG,
      ...stored,
      enabled_classes: Array.isArray(stored.enabled_classes)
        ? stored.enabled_classes.map(Number).filter((n: number) => Number.isFinite(n) && !TWO_WHEELER_CLASSES.has(n))
        : DEFAULT_CONFIG.enabled_classes,
      updated_at: row.updated_at,
    };
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

// Camera 2: raw substream relayed by the Pi with no detection (optional).
let live2Frame: Uint8Array | null = null;
let live2At = 0;
let live2Seq = 0;
let live2Viewers = 0;
let live2JpgAt = 0;

// ── Web terminal (reverse WebSocket: Pi dials out, browser drives it) ───
// The Pi is behind NAT and only makes outbound connections, so the edge agent
// registers itself here (`/api/term?role=agent`, Bearer token); browsers attach
// with a short-lived one-time token minted by the Basic-auth /terminal page.
// Browser keystrokes and agent PTY output are relayed verbatim; resize control
// frames are \x01-prefixed JSON so they can't be confused with typed input.
type TermRole = "agent" | "browser";
type TermData = { role: TermRole };
let agentSocket: ServerWebSocket<TermData> | null = null;
const browserSockets = new Set<ServerWebSocket<TermData>>();
const termTokens = new Map<string, number>();

function issueTermToken() {
  const t = crypto.randomUUID().replace(/-/g, "");
  const now = Date.now();
  for (const [k, exp] of termTokens) if (exp < now) termTokens.delete(k);
  termTokens.set(t, now + 120_000);
  return t;
}
function termTokenOk(t: string | null) {
  if (!t) return false;
  const exp = termTokens.get(t);
  return !!exp && exp > Date.now();
}

// xterm.js is self-hosted from node_modules so the page has no CDN dependency.
const TERM_ASSETS: Record<string, string> = {
  "xterm.js": "@xterm/xterm/lib/xterm.js",
  "xterm.css": "@xterm/xterm/css/xterm.css",
  "addon-fit.js": "@xterm/addon-fit/lib/addon-fit.js",
};
function serveTermAsset(name: string) {
  const rel = TERM_ASSETS[name];
  if (!rel) return new Response("not found", { status: 404 });
  return new Response(Bun.file(`${import.meta.dir}/node_modules/${rel}`));
}

function terminalPage(token: string) {
  return `<!doctype html><html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="theme-color" content="#0b0f14">
<title>Terminal · Vehicle Tracker</title>
<link rel="stylesheet" href="/terminal/xterm.css">
<style>
html,body{margin:0;height:100%;background:#0b0f14;color:#e5e7eb;font:14px system-ui,sans-serif;overflow:hidden}
#top{display:flex;align-items:center;gap:10px;padding:9px 12px;background:#111827;border-bottom:1px solid #1f2937}
#top b{font-size:14px}
#top a{color:#60a5fa;text-decoration:none;font-size:12px}
#st{margin-left:auto;font-size:12px;color:#9ca3af;display:flex;align-items:center;gap:6px}
.dot{width:8px;height:8px;border-radius:50%;background:#6b7280}
.dot.on{background:#16a34a}
#term{position:absolute;inset:42px 0 0 0;padding:6px}
</style></head><body>
<div id="top"><b>tracker-pi</b><a href="/live">&larr; dashboard</a>
  <div id="st"><span class="dot" id="dot"></span><span id="txt">connecting&hellip;</span></div></div>
<div id="term"></div>
<script src="/terminal/xterm.js"></script>
<script src="/terminal/addon-fit.js"></script>
<script>
(function(){
  var term=new Terminal({cursorBlink:true,fontSize:14,scrollback:5000,
    theme:{background:'#0b0f14',foreground:'#e5e7eb'}});
  var fit=new FitAddon.FitAddon();
  term.loadAddon(fit);
  term.open(document.getElementById('term'));
  var dot=document.getElementById('dot'),txt=document.getElementById('txt');
  var ws=null,stopped=false;
  var ctrl=function(o){return '\\x01'+JSON.stringify(o);};
  function resize(){try{fit.fit();}catch(e){}}
  window.addEventListener('resize',resize);
  function setSt(on,s){dot.classList.toggle('on',on);txt.textContent=s;}
  function connect(){
    if(stopped)return;
    setSt(false,'connecting\\u2026');
    fetch('/api/term-token',{cache:'no-store'}).then(function(r){return r.ok?r.json():null;})
      .then(function(d){
        if(!d||!d.token){setSt(false,'auth failed');setTimeout(connect,3000);return;}
        var proto=location.protocol==='https:'?'wss:':'ws:';
        ws=new WebSocket(proto+'//'+location.host+'/api/term?role=browser&t='+encodeURIComponent(d.token));
        ws.binaryType='arraybuffer';
        ws.onopen=function(){setSt(true,'connected');resize();
          if(ws.readyState===1)ws.send(ctrl({c:term.cols,r:term.rows}));};
        ws.onmessage=function(ev){term.write(typeof ev.data==='string'?ev.data:new Uint8Array(ev.data));};
        ws.onclose=function(){setSt(false,'disconnected');if(!stopped)setTimeout(connect,2000);};
        ws.onerror=function(){};
      }).catch(function(){setSt(false,'offline');setTimeout(connect,3000);});
  }
  term.onData(function(d){if(ws&&ws.readyState===1)ws.send(d);});
  term.onResize(function(s){if(ws&&ws.readyState===1)ws.send(ctrl({c:s.cols,r:s.rows}));});
  connect();
})();
</script>
</body></html>`;
}

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

// Shared by the initial page render and GET /api/events so the calendar/history
// views can auto-refresh without a reload.
function loadEventData() {
  const dayExpr = istDay("COALESCE(crossed_at, created_at)");
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
  return { cIn, cOut, dayStats, events };
}

function istTodayCounts() {
  const tcnt = sqlite
    .query(`SELECT direction, COUNT(*) AS c FROM events WHERE ${istDay("COALESCE(crossed_at, created_at)")} = ${istDay("'now'")} GROUP BY direction`)
    .all() as { direction: string; c: number }[];
  return {
    date: istNow(),
    in: tcnt.find((r) => r.direction === "in")?.c ?? 0,
    out: tcnt.find((r) => r.direction === "out")?.c ?? 0,
  };
}

function dashboard(view: "live" | "calendar" | "history", initialDay: string | null) {
  const { cIn, cOut, dayStats, events } = loadEventData();
  const init = initialDay && /^\d{4}-\d{2}-\d{2}$/.test(initialDay) ? initialDay : null;
  const J = (o: unknown) => JSON.stringify(o).replace(/</g, "\\u003c");
  const cam2Panel = CAM2
    ? `<div class="player" id="cam2" style="margin-top:12px"><img id="snap_cam2" alt="camera 2"></div>`
    : "";

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
figcaption{padding:7px 9px;font-size:12px;display:flex;justify-content:space-between;align-items:center;gap:6px}
figcaption .t{color:var(--text);font-size:16px;font-weight:800;font-variant-numeric:tabular-nums;letter-spacing:.01em;white-space:nowrap}
.empty2{color:var(--muted);text-align:center;padding:48px 20px}
.tabbar{position:fixed;left:0;right:0;bottom:0;z-index:30;display:flex;background:rgba(255,255,255,.97);backdrop-filter:blur(10px);border-top:1px solid var(--line);padding-bottom:env(safe-area-inset-bottom)}
.tabbar a{flex:1;text-align:center;text-decoration:none;padding:14px 0;font-size:13px;color:var(--muted);cursor:pointer}
.tabbar a.active{color:var(--blue);font-weight:700}
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
  <section id="v-live" class="view${view === "live" ? "" : " hidden"}">
    <div class="player"><img id="snap" alt="live"></div>
    ${cam2Panel}
    <div class="sub">Tap the gear to edit the counting line / scan area</div>
  </section>
  <section id="v-cal" class="view${view === "calendar" ? "" : " hidden"}">
    <div class="calhead">
      <button class="icon" id="prevMonth">&#8249;</button>
      <div id="calTitle"></div>
      <button class="icon" id="nextMonth">&#8250;</button>
      <button class="pill" id="todayBtn">Today</button>
    </div>
    <div class="calgrid" id="calGrid"></div>
    <div class="hint"><b style="color:var(--green)">IN</b> / <b style="color:var(--red)">OUT</b> per day &middot; &#8592; / &#8594; change day &middot; tap a day for captures</div>
    <div class="histhead" id="calHistHead" style="margin-top:16px"></div>
    <div class="gallery" id="calDay"></div>
  </section>
  <section id="v-hist" class="view${view === "history" ? "" : " hidden"}">
    <div class="histhead" id="histHead"></div>
    <div class="gallery" id="hist"></div>
  </section>
</main>
<nav class="tabbar">
  <a href="/live" class="tab${view === "live" ? " active" : ""}">Live</a>
  <a href="/calendar" class="tab${view === "calendar" ? " active" : ""}">Calendar</a>
  <a href="/history" class="tab${view === "history" ? " active" : ""}">History</a>
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
let DAYS=${J(dayStats)};
let EVENTS=${J(events)};
const INIT=${J(init)};
const VIEW=${J(view)};
let TOTALS={in:${cIn},out:${cOut}};
</script>
<script>
(function(){
var S=document.getElementById('snap'),dot=document.getElementById('dot'),ltxt=document.getElementById('ltxt');
var SC2=document.getElementById('snap_cam2');
var tk=todayKey(),tc=DAYS[tk]||{};
document.getElementById('sIn').textContent=tc['in']||0;
document.getElementById('sOut').textContent=tc.out||0;
function clock(){var n=new Date();document.getElementById('todayLbl').textContent='Today · '+
  n.toLocaleDateString('en-IN',{timeZone:'Asia/Kolkata',weekday:'short',day:'numeric',month:'short'})+' · '+
  n.toLocaleTimeString('en-GB',{timeZone:'Asia/Kolkata',hour:'2-digit',minute:'2-digit',second:'2-digit'})+' IST';}
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
var live2On=false,pumpTimer2=null,lastURL2=null;
function liveTick(){
  if(!liveOn)return;
  fetch('/live.jpg',{cache:'no-store'}).then(function(r){if(!r.ok)throw 0;return r.blob();})
    .then(function(b){var u=URL.createObjectURL(b);S.src=u;
      if(lastURL)URL.revokeObjectURL(lastURL);lastURL=u;})
    .catch(function(){});
}
// Camera 2 (raw substream, no detection) polls on a slower cadence; polling it
// flips the server's live2_wanted so the Pi starts relaying.
function live2Tick(){
  if(!live2On||!SC2)return;
  fetch('/live2.jpg',{cache:'no-store'}).then(function(r){if(!r.ok)throw 0;return r.blob();})
    .then(function(b){var u=URL.createObjectURL(b);SC2.src=u;
      if(lastURL2)URL.revokeObjectURL(lastURL2);lastURL2=u;})
    .catch(function(){});
}
function setLive(on){if(on===liveOn)return;liveOn=on;
  if(on){liveTick();pumpTimer=setInterval(liveTick,250);
    if(SC2){live2On=true;live2Tick();pumpTimer2=setInterval(live2Tick,500);}}
  else{if(pumpTimer){clearInterval(pumpTimer);pumpTimer=null;}
    live2On=false;if(pumpTimer2){clearInterval(pumpTimer2);pumpTimer2=null;}}}
var views=[].slice.call(document.querySelectorAll('.view'));
// Each view is its own route (/live, /calendar, /history); the nav links do a
// normal navigation, so we only reveal the route's view and toggle the live pump.
function showTab(id){views.forEach(function(v){v.classList.toggle('hidden',v.id!==id);});
  setLive(id==='v-live');
  window.scrollTo(0,0);}
showTab({live:'v-live',calendar:'v-cal',history:'v-hist'}[VIEW]||'v-live');
var selDay=INIT;
var view=INIT?new Date(INIT+'T00:00:00'):new Date();
function pad(n){return String(n).padStart(2,'0');}
function ymd(y,m,d){return y+'-'+pad(m+1)+'-'+pad(d);}
var calGrid=document.getElementById('calGrid'),calTitle=document.getElementById('calTitle');
// All day keys on the dashboard are IST (UTC+5:30), matching the server.
function todayKey(){return new Date().toLocaleDateString('sv-SE',{timeZone:'Asia/Kolkata'});}
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
    // Stay on /calendar: reveal the day's captures inline instead of jumping to /history.
    el.onclick=function(){selDay=el.dataset.day;renderCal();renderCalDay();};});
  document.getElementById('nextMonth').disabled=atOrAfterCurrentMonth();
}
function selectDay(k){selDay=k;if(k)view=new Date(k+'T00:00:00');renderCal();renderHist();renderCalDay();}
function shiftDay(n){var b=selDay?new Date(selDay+'T00:00:00'):new Date();b.setDate(b.getDate()+n);
  var k=ymd(b.getFullYear(),b.getMonth(),b.getDate());if(k>todayKey())return;
  selectDay(k);}
document.getElementById('prevMonth').onclick=function(){view.setMonth(view.getMonth()-1);renderCal();};
document.getElementById('nextMonth').onclick=function(){if(atOrAfterCurrentMonth())return;view.setMonth(view.getMonth()+1);renderCal();};
document.getElementById('todayBtn').onclick=function(){selectDay(todayKey());};
document.addEventListener('keydown',function(e){if(e.target&&/INPUT|TEXTAREA|SELECT/.test(e.target.tagName))return;
  if(e.key==='ArrowLeft'){shiftDay(-1);e.preventDefault();}else if(e.key==='ArrowRight'){shiftDay(1);e.preventDefault();}});
var hist=document.getElementById('hist'),histHead=document.getElementById('histHead');
var calDay=document.getElementById('calDay'),calHistHead=document.getElementById('calHistHead');
function istDay(ts){try{return new Date(ts).toLocaleDateString('sv-SE',{timeZone:'Asia/Kolkata'});}catch(e){return (ts||'').slice(0,10);}}
function fmt(ts){try{return new Date(ts).toLocaleString('sv-SE',{timeZone:'Asia/Kolkata'});}catch(e){return (ts||'').replace('T',' ').slice(0,19);}}
// Render a capture list into any (head, grid) pair. A day filters to one IST day.
function renderList(day, headEl, gridEl){
  var list=day?EVENTS.filter(function(e){return istDay(e.ts)===day;}):EVENTS;
  var c=day?(DAYS[day]||{in:0,out:0}):{in:TOTALS.in,out:TOTALS.out};
  var ctrls=day?'<button class="pill clearBtn" style="margin-left:auto">All</button>':'';
  headEl.innerHTML='<span class="d">'+(day?day:'All recent')+'</span>'+
    '<span class="c"><b style="color:var(--green)">'+c['in']+' IN</b> &middot; <b style="color:var(--red)">'+c.out+' OUT</b></span>'+ctrls;
  if(day){var clr=headEl.querySelector('.clearBtn');if(clr)clr.onclick=function(){selectDay(null);};}
  gridEl.innerHTML=list.length?list.map(function(e){
    var col=e.dir.toLowerCase()==='in'?'var(--green)':'var(--red)';
    return '<figure><a href="/img/'+encodeURIComponent(e.id)+'" target="_blank">'+
      '<img loading="lazy" src="/thumb/'+encodeURIComponent(e.id)+'" alt=""></a>'+
      '<figcaption><span style="color:'+col+';font-weight:700">'+(e.dir||'').toUpperCase()+' '+e.label+'</span>'+
      '<span class="t">'+fmt(e.ts).slice(11,19)+'</span></figcaption></figure>';
  }).join(''):'<div class="empty2">No captures</div>';
}
function renderHist(){renderList(selDay,histHead,hist);}
function renderCalDay(){
  if(!selDay){calHistHead.innerHTML='';calDay.innerHTML='';return;}
  renderList(selDay,calHistHead,calDay);
}
renderCal();renderHist();renderCalDay();
// Calendar/History auto-refresh: poll the event feed and re-render only when
// something actually changed (so images/scroll don't flicker). The live view
// doesn't need it.
if(VIEW!=='live'){
  var lastTotal=EVENTS.length;
  function refreshData(){
    fetch('/api/events',{cache:'no-store'}).then(function(r){return r.ok?r.json():null;})
      .then(function(d){if(!d)return;
        if(d.events)EVENTS=d.events;
        if(d.days)DAYS=d.days;
        if(d.counts)TOTALS=d.counts;
        if(d.today){document.getElementById('sIn').textContent=d.today['in']||0;
          document.getElementById('sOut').textContent=d.today.out||0;}
        if(EVENTS.length!==lastTotal){lastTotal=EVENTS.length;renderCal();renderHist();renderCalDay();}})
      .catch(function(){});
  }
  setInterval(refreshData,8000);
}
var setup=document.getElementById('setup'),S2=document.getElementById('snap2'),O=document.getElementById('ov'),msg=document.getElementById('msg');
document.getElementById('setupBtn').onclick=function(){setup.classList.remove('hidden');S2.src='/live.jpg?'+Date.now();};
document.getElementById('closeSetup').onclick=function(){setup.classList.add('hidden');};
var cfg={line:null,roi:null,flip_sides:false,enabled_classes:[2,5,7]};
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
  websocket: {
    open(ws) {
      if (ws.data.role === "agent") {
        if (agentSocket && agentSocket !== ws) {
          try { agentSocket.close(); } catch {}
        }
        agentSocket = ws;
        console.log("[tracker-web] terminal agent connected");
      } else {
        browserSockets.add(ws);
      }
    },
    message(ws, message) {
      // Relay verbatim: agent output → all browsers; browser input/resize → agent.
      if (ws.data.role === "agent") {
        for (const b of browserSockets) { try { b.send(message); } catch {} }
      } else if (agentSocket) {
        try { agentSocket.send(message); } catch {}
      }
    },
    close(ws) {
      if (ws.data.role === "agent") {
        if (agentSocket === ws) agentSocket = null;
        console.log("[tracker-web] terminal agent disconnected");
      } else {
        browserSockets.delete(ws);
      }
    },
  },
  async fetch(req, server) {
    const url = new URL(req.url);

    // Reverse-terminal WebSocket: the agent authenticates with the Bearer
    // token, browsers with a short-lived token from the Basic-auth page.
    if (url.pathname === "/api/term") {
      if (!ENABLE_TERMINAL) return new Response("not found", { status: 404 });
      const role = url.searchParams.get("role");
      if (role === "agent") {
        if (!bearerOk(req)) return new Response("unauthorized", { status: 401 });
      } else if (role === "browser") {
        if (!termTokenOk(url.searchParams.get("t"))) return new Response("unauthorized", { status: 401 });
      } else {
        return new Response("bad role", { status: 400 });
      }
      const data: TermData = { role: role as TermRole };
      if (server.upgrade(req, { data })) return;
      return new Response("upgrade failed", { status: 400 });
    }

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

    // Camera-2 raw frame push — Bearer token (Pi → VPS, no detection)
    if (url.pathname === "/api/live2" && req.method === "POST") {
      if (!bearerOk(req)) return new Response("unauthorized", { status: 401 });
      const buf = await req.arrayBuffer();
      if (buf.byteLength > 0) {
        live2Frame = new Uint8Array(buf);
        live2At = Date.now();
        live2Seq++;
      }
      return Response.json({ ok: true, bytes: buf.byteLength, at: live2At, seq: live2Seq });
    }

    // Camera config — Pi pulls with Bearer, dashboard reads/writes with Basic
    if (url.pathname === "/api/config" && req.method === "GET") {
      if (!bearerOk(req) && !basicOk(req)) return unauthorized();
      // The Pi uses live_wanted to only encode/upload while someone is watching.
      const live_wanted = liveViewers > 0 || Date.now() - liveJpgAt < 10_000;
      const live2_wanted = live2Viewers > 0 || Date.now() - live2JpgAt < 10_000;
      const cnt = sqlite
        .query("SELECT direction, COUNT(*) AS c FROM events GROUP BY direction")
        .all() as { direction: string; c: number }[];
      const allIn = cnt.find((r) => r.direction === "in")?.c ?? 0;
      const allOut = cnt.find((r) => r.direction === "out")?.c ?? 0;
      // Today's totals so the dashboard header can refresh without a reload.
      const dayExpr = istDay("COALESCE(crossed_at, created_at)");
      const tcnt = sqlite
        .query(`SELECT direction, COUNT(*) AS c FROM events WHERE ${dayExpr} = ${istDay("'now'")} GROUP BY direction`)
        .all() as { direction: string; c: number }[];
      const tIn = tcnt.find((r) => r.direction === "in")?.c ?? 0;
      const tOut = tcnt.find((r) => r.direction === "out")?.c ?? 0;
      return Response.json({
        ...getConfig(), viewers: liveViewers, live_wanted,
        live2_wanted, cam2: CAM2,
        counts: { in: allIn, out: allOut },
        today: { date: istNow(), in: tIn, out: tOut },
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
          ? body.enabled_classes.map(Number).filter((n) => Number.isFinite(n) && !TWO_WHEELER_CLASSES.has(n))
          : DEFAULT_CONFIG.enabled_classes,
        source_w: Number.isFinite(body.source_w) ? Number(body.source_w) : null,
        source_h: Number.isFinite(body.source_h) ? Number(body.source_h) : null,
      };
      const updated_at = saveConfig(cfg);
      return Response.json({ ok: true, updated_at, config: cfg });
    }

    // Browser routes — Basic auth
    if (!basicOk(req)) return unauthorized();

    if (url.pathname === "/terminal") {
      if (!ENABLE_TERMINAL) return new Response("terminal disabled", { status: 404 });
      return new Response(terminalPage(issueTermToken()), {
        headers: { "content-type": "text/html; charset=utf-8", "cache-control": "no-store" },
      });
    }
    if (url.pathname === "/api/term-token") {
      if (!ENABLE_TERMINAL) return new Response("terminal disabled", { status: 404 });
      return Response.json({ token: issueTermToken(), agent: agentSocket !== null });
    }
    if (url.pathname.startsWith("/terminal/") && !ENABLE_TERMINAL) {
      return new Response("terminal disabled", { status: 404 });
    }
    if (url.pathname.startsWith("/terminal/")) {
      return serveTermAsset(decodeURIComponent(url.pathname.slice("/terminal/".length)));
    }

    // Event feed for the calendar/history auto-refresh.
    if (url.pathname === "/api/events") {
      const { cIn, cOut, dayStats, events } = loadEventData();
      return Response.json({
        events, days: dayStats, counts: { in: cIn, out: cOut },
        today: istTodayCounts(),
      });
    }

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

    if (url.pathname === "/live2.jpg") {
      live2JpgAt = Date.now();
      if (!live2Frame || Date.now() - live2At > 10_000) return new Response("no live feed", { status: 404 });
      return new Response(live2Frame, {
        headers: {
          "content-type": "image/jpeg",
          "cache-control": "no-store",
          "x-live-seq": String(live2Seq),
          "x-live-age": String(Date.now() - live2At),
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

    if (url.pathname === "/") return Response.redirect("/live", 302);
    if (url.pathname === "/live" || url.pathname === "/calendar" || url.pathname === "/history") {
      const view = url.pathname.slice(1) as "live" | "calendar" | "history";
      const day = url.searchParams.get("day") ?? url.searchParams.get("d");
      return new Response(dashboard(view, day), {
        headers: {
          "content-type": "text/html; charset=utf-8",
          "cache-control": "no-store, no-cache, must-revalidate",
          "pragma": "no-cache",
        },
      });
    }
    if (url.pathname.startsWith("/img/")) return serveImage(decodeURIComponent(url.pathname.slice(5)), false);
    if (url.pathname.startsWith("/thumb/")) return serveImage(decodeURIComponent(url.pathname.slice(7)), true);
    return new Response("not found", { status: 404 });
  },
});

console.log(`[tracker-web] :${PORT} db=${DB_PATH} captures=${CAPTURES_DIR} auth=${DASH_USER ? "on" : "off"}`);
