/* ============================================================================
   backend.js  (v2 — supersedes v1)

   Changes from v1, both from the slice-1 report:
     • STREAMING RESTORED. v1 buffered every token and only called addMsg()
       at `done`, so nothing appeared until the whole answer landed. That was
       a regression against the skeleton. Now the reply element is opened on
       the first token and appended into as tokens arrive — no rendering code
       in dashboard.html is touched, because addMsg() already creates the
       element and we only write into its .cmsg-body.
     • `inputMode` is a lexical `let` in dashboard.html, NOT a window property.
       v1 read window.inputMode (always undefined) so the orb fell to idle even
       in voice mode. restState() now resolves it safely.

   Real contract (server.py, verified):
     client -> {"text": "<user message>"}
     server -> one JSON object per orchestrator event, verbatim:
               thinking | token | delegation | recovery | error | done
     field names:  token.content   delegation.status   error.message
   plus `telemetry` (slice 2), the one frame that arrives with no turn in
   flight: gpu / cpu / ram percentages and vram_used_gb / vram_total_gb, each
   key present only when the host could actually measure it.
   plus `memory` (slice 6): the real profile — facts_total, added_today,
   recall_p95_ms (null: nothing measures it), categories[{key,count}] and
   recent[{ts,text}]. Edge-triggered, not on a clock.

   THROUGHPUT is computed here (slice 3) from the token stream itself — the
   wire carries no rate. The server now gives every connection its own
   conversation, so a reload starts fresh; this file needs nothing for that.

   Load AFTER dashboard's inline script:
       <script src="backend.js"></script>
   ========================================================================== */
(() => {
  "use strict";

  const CONFIG = {
    LIVE: true,                    // false -> mockup runs untouched
    URL: "ws://" + (location.hostname || "localhost") + ":8765",
    RETRY_MS: 1200,
    RETRY_MAX: 15000,
  };
  if (!CONFIG.LIVE) return;

  /* --- seams (already global in dashboard.html) -------------------------- */
  const need = (n) => {
    const f = window[n];
    if (typeof f !== "function") {
      console.error(`[backend] missing seam ${n}() — is backend.js loaded after dashboard's script?`);
      return () => {};
    }
    return f;
  };
  const setOrbState     = need("setOrbState");
  const setSystemStatus = need("setSystemStatus");
  const logEvent        = need("logEvent");
  const markToolActive  = need("markToolActive");
  const addMsg          = need("addMsg");

  const chatLog = document.getElementById("chatlog");

  /* `inputMode` is a lexical binding in dashboard.html, not on window.
     Resolve it defensively so a rename there degrades instead of breaking. */
  function restState() {
    let mode;
    try { mode = inputMode; } catch { mode = undefined; }   // eslint-disable-line
    return mode === "voice" ? "armed" : "idle";
  }

  /* --- transport --------------------------------------------------------- */
  let ws = null, retry = CONFIG.RETRY_MS;
  window.__live = true;            // dashboard's mock loops check this

  function connect() {
    setSystemStatus("degraded", "CONNECTING");
    try { ws = new WebSocket(CONFIG.URL); }
    catch { return scheduleRetry(); }

    ws.onopen = () => {
      retry = CONFIG.RETRY_MS;
      setSystemStatus("ok");
      logEvent("SYS", "Backend link established.");
    };

    ws.onmessage = (ev) => {
      let m;
      try { m = JSON.parse(ev.data); }
      catch { return logEvent("SYS", "Malformed frame discarded.", true); }
      handle(m);
    };

    ws.onclose = () => {
      setSystemStatus("down");
      abortTurn();
      scheduleRetry();
    };
    ws.onerror = () => { try { ws.close(); } catch {} };
  }

  function scheduleRetry() {
    setTimeout(connect, retry);
    retry = Math.min(retry * 1.7, CONFIG.RETRY_MAX);
  }

  /* --- streaming reply ---------------------------------------------------
     openReply() uses addMsg() to build the element, then holds a reference to
     its body so tokens can be appended incrementally. Nothing in
     dashboard.html changes; we only write text into what it created.        */
  let bodyEl = null;               // .cmsg-body of the open reply, or null

  function openReply() {
    if (bodyEl) return;
    addMsg("", "j");
    const last = chatLog && chatLog.lastElementChild;
    bodyEl = last ? last.querySelector(".cmsg-body") : null;
  }

  function pushToken(text) {
    if (!text) return;
    openReply();
    if (!bodyEl) return;
    bodyEl.textContent += text;
    chatLog.scrollTop = chatLog.scrollHeight;   // stay pinned to the newest line
  }

  function closeReply() {
    // an empty reply element would be a blank bubble — drop it
    if (bodyEl && !bodyEl.textContent.trim()) {
      const el = bodyEl.closest(".cmsg");
      if (el && el.parentNode) el.parentNode.removeChild(el);
    }
    bodyEl = null;
  }

  /* --- throughput (slice 3) ----------------------------------------------
     Client-side only: nothing on the wire carries a rate, so count what we
     can see. UNITS ARE CHUNKS/SEC, not model tokens/sec — one `token` event
     may carry several characters. Ollama's true eval_count / eval_duration
     would need a backend change and is a separate decision.

     The clock starts at the FIRST chunk, not at `thinking`: reasoning runs
     for seconds before any content arrives (it streams in a separate field
     and never reaches this stream), and counting that dead time in the
     denominator would report a rate the model is not running at.          */
  const TPS_TICK_MS = 250;
  let chunks = 0, firstChunkAt = 0, tpsTimer = null;

  function setTps(v) { if (window.setThroughput) window.setThroughput(v); }

  function startThroughput() {
    chunks = 0;
    firstChunkAt = 0;
    if (tpsTimer) return;
    tpsTimer = setInterval(() => {
      if (!firstChunkAt) return;
      const secs = (performance.now() - firstChunkAt) / 1000;
      if (secs >= 0.25) setTps(chunks / secs);
    }, TPS_TICK_MS);
  }

  function countChunk() {
    if (!firstChunkAt) firstChunkAt = performance.now();
    chunks++;
  }

  function stopThroughput() {
    if (tpsTimer) { clearInterval(tpsTimer); tpsTimer = null; }
    setTps(0);                 // target 0 — the dial eases down to IDLE itself
  }

  function abortTurn() {
    stopThroughput();
    if (bodyEl) {
      // keep whatever arrived; mark it so a truncated answer isn't mistaken
      // for a complete one
      if (bodyEl.textContent.trim()) bodyEl.textContent += " …";
      logEvent("SYS", "Turn aborted — link lost.", true);
    }
    closeReply();
    setOrbState(restState());
  }

  /* --- the six real events ------------------------------------------------ */
  function handle(m) {
    switch (m.type) {

      case "thinking":
        closeReply();
        setOrbState("thinking");
        startThroughput();         // also re-armed after a tool call's second pass
        break;

      case "token":
        countChunk();
        pushToken(m.content);      // .content — verified against the skeleton
        break;

      case "delegation":
        markToolActive(m.status || "tool", 1100);
        logEvent("TOOL", String(m.status || "delegating"));
        break;

      case "telemetry":
        // ambient, arrives outside a turn — never touches the reply stream
        if (window.setTelemetry) window.setTelemetry(m);
        break;

      case "memory":
        // the real profile (slice 6): on connect, then on every digest write
        if (window.setMemory) window.setMemory(m);
        break;

      case "recovery":
        logEvent("SYS", "Recovered — retrying the turn.", true);
        break;

      case "error":
        logEvent("SYS", m.message || "Turn failed.", true);
        closeReply();
        stopThroughput();
        setOrbState(restState());
        break;

      case "done":
        closeReply();
        stopThroughput();
        setOrbState(restState());
        break;

      default:
        console.warn("[backend] unhandled event:", m.type, m);
    }
  }

  /* --- outbound ----------------------------------------------------------- */
  window.sendToBackend = (text) => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ text }));
      return true;
    }
    logEvent("SYS", "Not connected — message not sent.", true);
    return false;
  };

  connect();
})();