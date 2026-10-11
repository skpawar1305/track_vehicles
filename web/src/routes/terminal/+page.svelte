<script lang="ts">
  import { onMount } from "svelte";
  import { Terminal } from "@xterm/xterm";
  import { FitAddon } from "@xterm/addon-fit";
  import "@xterm/xterm/css/xterm.css";

  let el = $state<HTMLDivElement | undefined>();
  let status = $state("connecting…");
  let ok = $state(false);

  onMount(() => {
    const term = new Terminal({
      cursorBlink: true, fontSize: 14, scrollback: 5000,
      theme: { background: "#0b0f14", foreground: "#e5e7eb" },
    });
    const fit = new FitAddon();
    term.loadAddon(fit);
    term.open(el!);

    let ws: WebSocket | null = null;
    let stopped = false;
    const ctrl = (o: unknown) => "\x01" + JSON.stringify(o);
    const resize = () => { try { fit.fit(); } catch {} };
    window.addEventListener("resize", resize);

    function connect() {
      if (stopped) return;
      ok = false; status = "connecting…";
      fetch("/api/term-token", { cache: "no-store" })
        .then((r) => (r.ok ? r.json() : null))
        .then((d) => {
          if (!d || !d.token) { status = "auth failed"; setTimeout(connect, 3000); return; }
          const proto = location.protocol === "https:" ? "wss:" : "ws:";
          ws = new WebSocket(proto + "//" + location.host + "/api/term?role=browser&t=" + encodeURIComponent(d.token));
          ws.binaryType = "arraybuffer";
          ws.onopen = () => {
            ok = true; status = "connected"; resize();
            if (ws && ws.readyState === 1) ws.send(ctrl({ c: term.cols, r: term.rows }));
          };
          ws.onmessage = (ev) => term.write(typeof ev.data === "string" ? ev.data : new Uint8Array(ev.data));
          ws.onclose = () => { ok = false; status = "disconnected"; if (!stopped) setTimeout(connect, 2000); };
          ws.onerror = () => {};
        })
        .catch(() => { status = "offline"; setTimeout(connect, 3000); });
    }

    term.onData((d) => { if (ws && ws.readyState === 1) ws.send(d); });
    term.onResize((s) => { if (ws && ws.readyState === 1) ws.send(ctrl({ c: s.cols, r: s.rows })); });
    connect();

    return () => {
      stopped = true;
      window.removeEventListener("resize", resize);
      try { ws?.close(); } catch {}
      term.dispose();
    };
  });
</script>

<div class="twrap">
  <div class="tbar">
    <b>tracker-pi</b><a href="/live">&larr; dashboard</a>
    <div class="tst"><span class="tdot" class:on={ok}></span><span>{status}</span></div>
  </div>
  <div class="tterm" bind:this={el}></div>
</div>

<style>
  .twrap {
    position: fixed; inset: 0; z-index: 50; display: flex; flex-direction: column;
    background: #0b0f14; color: #e5e7eb; font: 14px system-ui, sans-serif;
  }
  .tbar {
    display: flex; align-items: center; gap: 10px; padding: 9px 12px;
    background: #111827; border-bottom: 1px solid #1f2937;
  }
  .tbar b { font-size: 14px; }
  .tbar a { color: #60a5fa; text-decoration: none; font-size: 12px; }
  .tst { margin-left: auto; font-size: 12px; color: #9ca3af; display: flex; align-items: center; gap: 6px; }
  .tdot { width: 8px; height: 8px; border-radius: 50%; background: #6b7280; }
  .tdot.on { background: #16a34a; }
  .tterm { flex: 1; min-height: 0; padding: 6px; }
  .tterm :global(.xterm) { height: 100%; }
</style>
