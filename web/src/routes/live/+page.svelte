<script lang="ts">
  import { onMount } from "svelte";
  import { cam2, liveOk } from "#lib/stores.js";

  let active = $state<"main" | "cam2">("main");
  let mainEl = $state<HTMLImageElement | undefined>();
  let cam2El = $state<HTMLImageElement | undefined>();

  let urlMain: string | null = null;
  let urlCam2: string | null = null;
  let timer: ReturnType<typeof setInterval> | null = null;

  const feedImg = (k: "main" | "cam2") => (k === "cam2" ? cam2El : mainEl);

  // Poll snapshots instead of a one-shot MJPEG connection (which never
  // reconnects after a server restart). Polling also keeps the server's
  // live_wanted flag alive so the Pi keeps uploading. Only ONE feed is pumped
  // at a time: the idle feed's flag goes stale and the Pi stops sending it.
  async function pump(k: "main" | "cam2") {
    const img = feedImg(k);
    if (!img) return;
    try {
      const r = await fetch(k === "cam2" ? "/live2.jpg" : "/live.jpg", { cache: "no-store" });
      if (!r.ok) throw 0;
      const b = await r.blob();
      const u = URL.createObjectURL(b);
      img.src = u;
      if (k === "cam2") { if (urlCam2) URL.revokeObjectURL(urlCam2); urlCam2 = u; }
      else { if (urlMain) URL.revokeObjectURL(urlMain); urlMain = u; }
      if (active === k) liveOk.set(true);
    } catch {
      if (active === k) liveOk.set(false);
    }
  }

  function stop() { if (timer) { clearInterval(timer); timer = null; } }
  function show(k: "main" | "cam2") {
    active = k;
    liveOk.set(false);
    stop();
    pump(k);
    timer = setInterval(() => pump(k), k === "cam2" ? 500 : 250);
  }
  function flip() { if (!$cam2) return; show(active === "cam2" ? "main" : "cam2"); }
  function onKey(e: KeyboardEvent) {
    if ($cam2 && (e.key === "ArrowLeft" || e.key === "ArrowRight")) flip();
  }

  onMount(() => {
    show("main");
    window.addEventListener("keydown", onKey);
    return () => {
      window.removeEventListener("keydown", onKey);
      stop();
      liveOk.set(false);
      if (urlMain) URL.revokeObjectURL(urlMain);
      if (urlCam2) URL.revokeObjectURL(urlCam2);
    };
  });
</script>

<section>
  <div id="feedsWrap">
    <div class="feeds">
      <div class="feed" class:hidden={active !== "main"}>
        <div class="feedtag"><span class="d"></span>Main · counting</div>
        <div class="player"><img bind:this={mainEl} alt="live"></div>
      </div>
      {#if $cam2}
        <div class="feed" class:hidden={active !== "cam2"}>
          <div class="feedtag"><span class="d"></span>Camera 2</div>
          <div class="player"><img bind:this={cam2El} alt="camera 2"></div>
        </div>
      {/if}
    </div>
    {#if $cam2}
      <button class="nav prev" aria-label="Previous feed" onclick={flip}>&#8249;</button>
      <button class="nav next" aria-label="Next feed" onclick={flip}>&#8250;</button>
    {/if}
  </div>
  <div class="sub">Tap the gear to edit the counting line / scan area</div>
</section>
