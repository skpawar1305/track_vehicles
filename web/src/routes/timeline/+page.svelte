<script lang="ts">
  import { onMount } from "svelte";
  import { get } from "svelte/store";
  import { page } from "$app/state";
  import { days, todayCounts } from "#lib/stores.js";
  import { dayLabel, fmtTimeParts, todayKey, ymd } from "#lib/format.js";
  import type { DayEvent } from "#lib/types.js";

  type DayData = { day: string; counts: { in: number; out: number }; events: DayEvent[] };

  function initDay() {
    const d = page.url.searchParams.get("day") ?? page.url.searchParams.get("d");
    return d && /^\d{4}-\d{2}-\d{2}$/.test(d) ? d : todayKey();
  }

  let selDay = $state(initDay());
  let dir = $state<"in" | "out">("out");
  let cache = $state<Record<string, DayData>>({});
  let lastDays = "";

  const current = $derived(cache[selDay] ?? null);
  const inCount = $derived(current ? current.events.filter((e) => !e.ign && e.dir.toLowerCase() === "in").length : 0);
  const outCount = $derived(current ? current.events.filter((e) => !e.ign && e.dir.toLowerCase() === "out").length : 0);
  const items = $derived(
    current ? current.events.filter((e) => e.dir.toLowerCase() === dir) : [],
  );

  // Day squares: the last 30 days (retention window) plus any day with data.
  const strip = $derived.by(() => {
    const set: Record<string, 1> = {};
    const today = todayKey();
    const b = new Date(today + "T00:00:00");
    for (let i = 0; i < 30; i++) {
      const d = new Date(b.getTime() - i * 86_400_000);
      set[ymd(d.getFullYear(), d.getMonth(), d.getDate())] = 1;
    }
    Object.keys($days).forEach((k) => { if (k <= today) set[k] = 1; });
    set[selDay] = 1;
    return Object.keys(set).sort();
  });

  function dayNum(k: string) { return new Date(k + "T00:00:00").getDate(); }
  function dayWd(k: string) {
    const d = new Date(k + "T00:00:00");
    return d.getDate() === 1
      ? d.toLocaleDateString("en-IN", { month: "short" })
      : d.toLocaleDateString("en-IN", { weekday: "short" });
  }

  async function loadDay(k: string) {
    const cached = cache[k];
    try {
      const r = await fetch("/api/day?d=" + encodeURIComponent(k), { cache: "no-store" });
      if (!r.ok) return;
      const d: DayData = await r.json();
      const changed = !cached || JSON.stringify(d.events) !== JSON.stringify(cached.events);
      if (changed) cache = { ...cache, [k]: d };
    } catch {}
  }

  function selectDay(k: string) { selDay = k; loadDay(k); }
  function shiftDay(n: number) {
    const b = new Date((selDay || todayKey()) + "T00:00:00");
    b.setDate(b.getDate() + n);
    const k = ymd(b.getFullYear(), b.getMonth(), b.getDate());
    if (k > todayKey()) return;
    selectDay(k);
  }

  function findEvent(id: string): DayEvent | null {
    for (const k in cache) {
      const e = cache[k].events.find((x) => x.id === id);
      if (e) return e;
    }
    return null;
  }

  async function refreshCounts() {
    try {
      const x = await fetch("/api/events", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null));
      if (!x) return;
      const ds = JSON.stringify(x.days);
      if (ds !== lastDays) {
        lastDays = ds;
        days.set(x.days);
        todayCounts.set({ in: x.today.in, out: x.today.out });
        loadDay(selDay);
      }
    } catch {}
  }

  async function toggleIgnore(id: string) {
    const e = findEvent(id);
    if (!e) return;
    const on = !e.ign;
    e.ign = on; // optimistic (deep $state → tab counts update instantly)
    try {
      const r = await fetch("/api/ignore", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({ id, ignored: on }),
      });
      if (!r.ok) throw 0;
      const x = await fetch("/api/events", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null));
      if (x) { days.set(x.days); todayCounts.set({ in: x.today.in, out: x.today.out }); lastDays = JSON.stringify(x.days); }
    } catch {
      e.ign = !on; // revert
    }
  }

  function onKey(e: KeyboardEvent) {
    const t = e.target as HTMLElement | null;
    if (t && /INPUT|TEXTAREA|SELECT/.test(t.tagName)) return;
    if (e.key === "ArrowLeft") { shiftDay(-1); e.preventDefault(); }
    else if (e.key === "ArrowRight") { shiftDay(1); e.preventDefault(); }
  }

  $effect(() => {
    selDay;
    const el = document.querySelector(".daycard.sel");
    if (el && el.scrollIntoView) el.scrollIntoView({ block: "nearest", inline: "center" });
  });

  onMount(() => {
    lastDays = JSON.stringify(get(days));
    loadDay(selDay);
    const t = setInterval(refreshCounts, 8000);
    window.addEventListener("keydown", onKey);
    return () => { clearInterval(t); window.removeEventListener("keydown", onKey); };
  });
</script>

<section>
  <div class="daystrip">
    {#each strip as k (k)}
      {@const c = $days[k] ?? { in: 0, out: 0 }}
      <button class="daycard" class:sel={k === selDay} class:today={k === todayKey()} onclick={() => selectDay(k)}>
        <span class="dn">{dayNum(k)}</span>
        <span class="wd">{dayWd(k)}</span>
        <span class="cnt">
          {#if c.in || c.out}
            <span class="b bi">{c.in}</span><span class="b bo">{c.out}</span>
          {:else}
            <span class="z">&middot;</span>
          {/if}
        </span>
      </button>
    {/each}
  </div>
  <div class="hint">Swipe for older days &middot; &#8592; / &#8594; step a day</div>
  <div class="histhead" style="margin-top:14px"><span class="d">{dayLabel(selDay)}</span></div>
  <div>
    {#if current}
      <div class="grp">
        <div class="seg">
          <button class="in" class:on={dir === "in"} onclick={() => (dir = "in")}>IN <span class="n">{inCount}</span></button>
          <button class="out" class:on={dir === "out"} onclick={() => (dir = "out")}>OUT <span class="n">{outCount}</span></button>
        </div>
        {#if items.length === 0}
          <div class="empty2">No {dir.toUpperCase()} captures</div>
        {:else}
          <div class="gallery">
            {#each items as e (e.id)}
              <figure class:ign={e.ign}>
                <div class="thumbwrap">
                  <a href={"/img/" + encodeURIComponent(e.id)} target="_blank"><img loading="lazy" src={"/thumb/" + encodeURIComponent(e.id)} alt=""></a>
                  <button class="ov ig" title={e.ign ? "Include this capture in the counts" : "Exclude this capture from the counts"} aria-pressed={e.ign} onclick={() => toggleIgnore(e.id)}>
                    {#if e.ign}
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>
                    {:else}
                      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"/><line x1="1" y1="1" x2="23" y2="23"/></svg>
                    {/if}
                  </button>
                  <a class="ov sub" href={"/thumb/" + encodeURIComponent(e.id)} target="_blank" title="Open the crossing thumbnail (substream)">
                    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
                  </a>
                </div>
                <!-- svelte-ignore a11y_figcaption_index -->
                <figcaption>
                  <span class="lab">{e.label || "vehicle"}</span>
                  <span class="t"><b>{fmtTimeParts(e.ts).h}</b><span class="mm">{fmtTimeParts(e.ts).m}</span></span>
                </figcaption>
              </figure>
            {/each}
          </div>
        {/if}
      </div>
    {:else}
      <div class="empty2">Loading&hellip;</div>
    {/if}
  </div>
</section>
