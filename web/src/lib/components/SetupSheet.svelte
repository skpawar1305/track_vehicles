<script lang="ts">
  import { setupOpen } from "$lib/stores";

  type Pt = [number, number];
  type Cfg = { line: Pt[] | null; roi: Pt[] | null; flip_sides: boolean; enabled_classes: number[] };

  let cfg = $state<Cfg>({ line: null, roi: null, flip_sides: false, enabled_classes: [2, 5, 7] });
  let mode = $state<null | "line" | "area" | "in">(null);
  let dl = $state<Pt[]>([]);
  let da = $state<Pt[]>([]);
  let msg = $state("");
  let snapSrc = $state("");
  let imgEl = $state<HTMLImageElement | undefined>();
  let editorEl = $state<HTMLDivElement | undefined>();

  let hasLine = $derived(!!(cfg.line && cfg.line.length === 2));
  let hasRoi = $derived(!!(cfg.roi && cfg.roi.length >= 3));
  let roiPoints = $derived(hasRoi ? cfg.roi!.map((p) => p.join(",")).join(" ") : "");
  let daPoints = $derived(da.map((p) => p.join(",")).join(" "));
  let ioPos = $derived.by(() => {
    if (!cfg.line || cfg.line.length !== 2) return { i: [0, 0] as Pt, o: [0, 0] as Pt };
    const A = cfg.line[0], B = cfg.line[1];
    const dx = B[0] - A[0], dy = B[1] - A[1], L = Math.hypot(dx, dy) || 1;
    let ux = -dy / L, uy = dx / L;
    if (cfg.flip_sides) { ux = -ux; uy = -uy; }
    return {
      i: [(A[0] + B[0]) / 2 + ux * 0.06, (A[1] + B[1]) / 2 + uy * 0.06] as Pt,
      o: [(A[0] + B[0]) / 2 - ux * 0.06, (A[1] + B[1]) / 2 - uy * 0.06] as Pt,
    };
  });

  $effect(() => {
    if (!$setupOpen) return;
    snapSrc = "/live.jpg?" + Date.now();
    fetch("/api/config").then((r) => r.json()).then((c) => { cfg = { ...cfg, ...c }; }).catch(() => {});
  });

  function side(line: Pt[], p: Pt) {
    const A = line[0], B = line[1];
    return (B[0] - A[0]) * (p[1] - A[1]) - (B[1] - A[1]) * (p[0] - A[0]);
  }

  function setMode(m: typeof mode, clearDraft = false) {
    mode = m;
    if (clearDraft) dl = [];
  }

  function onClick(ev: MouseEvent) {
    if (!editorEl) return;
    const r = editorEl.getBoundingClientRect();
    const p: Pt = [(ev.clientX - r.left) / r.width, (ev.clientY - r.top) / r.height];
    if (mode === "line") {
      dl = [...dl, p];
      if (dl.length === 2) { cfg.line = dl.slice(); dl = []; mode = null; }
    } else if (mode === "area") {
      if (da.length >= 3 && Math.hypot(p[0] - da[0][0], p[1] - da[0][1]) < 0.05) {
        cfg.roi = da.slice(); da = []; mode = null;
      } else da = [...da, p];
    } else if (mode === "in" && cfg.line) {
      cfg.flip_sides = side(cfg.line, p) < 0; mode = null;
    }
  }

  function reset() { cfg.line = null; cfg.roi = null; dl = []; da = []; mode = null; }

  async function save() {
    if (da.length >= 3) { cfg.roi = da.slice(); da = []; mode = null; }
    msg = "saving…";
    try {
      const r = await fetch("/api/config", {
        method: "POST", headers: { "content-type": "application/json" },
        body: JSON.stringify({
          line: cfg.line, roi: cfg.roi, flip_sides: cfg.flip_sides, enabled_classes: cfg.enabled_classes,
          source_w: imgEl?.naturalWidth, source_h: imgEl?.naturalHeight,
        }),
      });
      msg = r.ok ? "saved ✓" : "save failed";
    } catch { msg = "save failed"; }
  }
</script>

{#if $setupOpen}
  <div class="sheet">
    <div class="sheetbar"><b>Camera setup</b><button class="icon" onclick={() => setupOpen.set(false)}>&#10005;</button></div>
    <!-- svelte-ignore a11y_no_static_element_interactions, a11y_click_events_have_key_events -->
    <div class="editor" bind:this={editorEl} onclick={onClick}>
      <img bind:this={imgEl} src={snapSrc} alt="snapshot">
      <svg viewBox="0 0 1 1" preserveAspectRatio="none">
        {#if hasRoi}
          <polygon points={roiPoints} fill="rgba(100,180,255,.15)" stroke="#64b4ff" stroke-width="2" vector-effect="non-scaling-stroke" />
        {/if}
        {#if da.length >= 2}
          <polyline points={daPoints} fill="none" stroke="#f59e0b" stroke-width="2" stroke-dasharray="4" vector-effect="non-scaling-stroke" />
        {/if}
        {#if hasLine}
          <line x1={cfg.line![0][0]} y1={cfg.line![0][1]} x2={cfg.line![1][0]} y2={cfg.line![1][1]} stroke="#3b82f6" stroke-width="3" vector-effect="non-scaling-stroke" />
          <text x={ioPos.i[0]} y={ioPos.i[1]} fill="#16a34a" font-size="0.05" font-weight="700" text-anchor="middle">IN</text>
          <text x={ioPos.o[0]} y={ioPos.o[1]} fill="#dc2626" font-size="0.05" font-weight="700" text-anchor="middle">OUT</text>
        {/if}
        {#each dl as p}
          <circle cx={p[0]} cy={p[1]} r="0.008" fill="#f59e0b" vector-effect="non-scaling-stroke" />
        {/each}
      </svg>
    </div>
    <div class="tools">
      <button class:active={mode === "line"} onclick={() => setMode("line", true)}>Line</button>
      <button class:active={mode === "area"} onclick={() => setMode("area")}>Area</button>
      <button class:active={mode === "in"} onclick={() => setMode("in")}>IN side</button>
      <button onclick={reset}>Reset</button>
      <button class="primary" onclick={save}>Save</button>
      <span class="msg">{msg}</span>
    </div>
    <p class="hint">Line: tap 2 points. Area: tap points, tap the first point (or Save) to close. IN side: tap where traffic enters. Then Save.</p>
  </div>
{/if}
