// Day/time helpers. All day keys are IST (UTC+5:30), matching the server.
const IST = { timeZone: "Asia/Kolkata" } as const;

export function pad(n: number) { return String(n).padStart(2, "0"); }
export function ymd(y: number, m: number, d: number) { return `${y}-${pad(m + 1)}-${pad(d)}`; }
export function todayKey() { return new Date().toLocaleDateString("sv-SE", IST); }

export function dayLabel(k: string) {
  if (k === todayKey()) return "Today";
  const y = new Date(Date.now() - 86_400_000).toLocaleDateString("sv-SE", IST);
  if (k === y) return "Yesterday";
  try {
    return new Date(`${k}T00:00:00`).toLocaleDateString("en-IN",
      { weekday: "short", day: "numeric", month: "short", year: "numeric", ...IST });
  } catch {
    return k;
  }
}

export function fmt(ts: string) {
  try {
    return new Date(ts).toLocaleString("sv-SE", IST);
  } catch {
    return (ts || "").replace("T", " ").slice(0, 19);
  }
}

// Hours dominate, minutes are secondary.
export function fmtTimeParts(ts: string): { h: string; m: string } {
  const s = fmt(ts).slice(11, 16);
  return { h: s.slice(0, 2), m: s.slice(3, 5) };
}

export function clockText() {
  const n = new Date();
  return "Today · " +
    n.toLocaleDateString("en-IN", { ...IST, weekday: "short", day: "numeric", month: "short" }) + " · " +
    n.toLocaleTimeString("en-GB", { ...IST, hour: "2-digit", minute: "2-digit" }) + " IST";
}
