<script lang="ts">
  import "../app.css";
  import { onMount } from "svelte";
  import { page } from "$app/stores";
  import { days, todayCounts, edge, liveOk, cam2, setupOpen } from "$lib/stores";
  import { clockText } from "$lib/format";
  import SetupSheet from "$lib/components/SetupSheet.svelte";

  let { children } = $props();

  let now = $state(new Date());
  let path = $derived($page.url.pathname);
  let isLive = $derived(path === "/live");
  let isTimeline = $derived(path === "/timeline");
  let isTerm = $derived(path === "/terminal");

  let clockLbl = $derived.by(() => { now; return clockText(); });
  let edgeFps = $derived.by(() => {
    if (!$edge.stats || ($edge.age != null && $edge.age >= 30)) return null;
    // Prefer the true stream rate: the detect rate dips when the NVR repeats
    // frames on a static scene, which made the header look broken when idle.
    const v = $edge.stats.stream_fps ?? $edge.stats.detect_fps;
    return v != null ? Math.round(Number(v)) : "?";
  });
  let edgeTemp = $derived.by(() => {
    if (!$edge.stats || ($edge.age != null && $edge.age >= 30)) return null;
    return $edge.stats.temp_c != null ? `${Math.round(Number($edge.stats.temp_c))}\u00b0C` : null;
  });

  onMount(() => {
    const clock = setInterval(() => (now = new Date()), 1000);
    const poll = async () => {
      try {
        const r = await fetch("/api/config", { cache: "no-store" });
        if (!r.ok) return;
        const d = await r.json();
        if (d.today) todayCounts.set({ in: d.today.in, out: d.today.out });
        edge.set({ stats: d.edge, age: d.edge_age });
        cam2.set(!!d.cam2);
      } catch {}
    };
    const loadDays = async () => {
      try {
        const x = await fetch("/api/events", { cache: "no-store" }).then((r) => (r.ok ? r.json() : null));
        if (x) days.set(x.days);
      } catch {}
    };
    poll();
    loadDays();
    const timer = setInterval(poll, 2000);
    // Refresh immediately when the tab is refocused, so a count that landed
    // while the tab was backgrounded shows up the moment you look back.
    const onVis = () => { if (document.visibilityState === "visible") poll(); };
    document.addEventListener("visibilitychange", onVis);
    window.addEventListener("focus", poll);
    return () => {
      clearInterval(clock); clearInterval(timer);
      document.removeEventListener("visibilitychange", onVis);
      window.removeEventListener("focus", poll);
    };
  });

  // The header LIVE dot only reflects the active /live feed.
  $effect(() => { if (!isLive) liveOk.set(false); });
</script>

{#if isTerm}
  {@render children()}
{:else}
<header>
  <div class="bar">
    <div class="brand">
      <div class="logo"><svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round"><circle cx="6" cy="18" r="2.4"/><circle cx="18" cy="6" r="2.4"/><path d="M8.4 18H15a4 4 0 0 0 0-8H9a4 4 0 0 1 0-8h.6"/></svg></div>
      <div><div class="title">Vehicle Tracker</div><div class="subt">{isLive ? "Live view" : "Capture timeline"}</div></div>
    </div>
    <div class="live"><span class="dot" class:on={$liveOk}></span><span>{$liveOk ? "LIVE" : "idle"}</span></div>
    {#if edgeFps != null}<span class="edge"><span class="e">{edgeFps}</span>&nbsp;fps{#if edgeTemp != null} · {edgeTemp}{/if}</span>{/if}
    <button class="icon" title="Setup" onclick={() => setupOpen.set(true)}>&#9881;</button>
  </div>
  <div class="todaylbl">{clockLbl}</div>
  {#if isLive}
    <div class="stats">
      <div class="stat in"><span class="lbl">IN</span><span class="num">{$todayCounts.in}</span></div>
      <div class="stat out"><span class="lbl">OUT</span><span class="num">{$todayCounts.out}</span></div>
    </div>
  {/if}
</header>

<main>{@render children()}</main>

<nav class="tabbar">
  <a href="/live" class:active={isLive}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="6" width="14" height="12" rx="2"/><path d="m22 8-6 4 6 4V8z"/></svg>
    <span>Live</span>
  </a>
  <a href="/timeline" class:active={isTimeline}>
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M3 12h18"/><circle cx="7" cy="12" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="17" cy="12" r="2.2"/></svg>
    <span>Timeline</span>
  </a>
</nav>

<SetupSheet />
{/if}
