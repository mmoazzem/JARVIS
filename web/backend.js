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
   plus `weather` (slice 8): the service's held reading, pushed on connect and
   on a 10-minute poll shared by every open dashboard. current / units /
   forecast (all ten days, one rendered) / place / lat / lon / observed_at,
   plus is_current + age_minutes so a reading that could not be refreshed is
   shown as old rather than passed off as now. Units are Open-Meteo's words
   (fahrenheit, kn, inch) — the tile maps them to glyphs, and converts
   wind_direction degrees to a compass point. Absent key = unknown, not zero.
   plus `memory` (slice 6): the real profile — facts_total, added_today,
   recall_p95_ms (null: nothing measures it), categories[{key,count}] and
   recent[{ts,text}]. Edge-triggered, not on a clock.
   plus the speech bracket (slice 7). Audio PLAYS ON THE HOST — none of it
   crosses this socket; these frames only say what is being heard:
     speech_pending  this turn claimed the sound card and will speak
     speech_start    the first word is audible NOW (not the first token)
     audio_level     {rms} real RMS of what is sounding, ~20 Hz
     speech_end      the bracket closes — always, including on failure
   `speech_pending` exists because `done` cannot answer "is speech coming?".
   For a short reply the last sentence is still synthesizing when `done` lands,
   so resting on `done` would flick STANDBY -> SPEAKING -> STANDBY. A turn that
   opened the bracket rests on `speech_end` instead; one with no bracket rests
   on `done` exactly as before.

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
  const setAudioLevel   = need("setAudioLevel");

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

     This is a ROLLING WINDOW, not a running average. It used to be
     `chunks / (now - firstChunkAt)` with firstChunkAt fixed at the turn's
     first chunk — total over total elapsed, which converges in the first
     couple of seconds and then barely moves: ten seconds in, one new chunk
     shifts it by a tenth of its deviation. That reads as a frozen dial. The
     window forgets, so the number is what the stream is doing NOW.

     The denominator is the WINDOW, not the age of the oldest sample, so a
     short reply reports a low rate instead of dividing a handful of chunks by
     a few milliseconds and spiking to an absurd one.

     Timing still begins at the FIRST CHUNK and never at `thinking`: reasoning
     runs for seconds before any content arrives (it streams in a separate
     field and never reaches this stream), and counting that dead time would
     report a rate the model is not running at. A window of arrival stamps
     keeps that for free — dead time simply contributes no samples.        */
  const TPS_TICK_MS = 250;
  // ~1 s: shorter is jittery on a bursty stream, longer reintroduces the same
  // inertia in miniature.
  const TPS_WINDOW_MS = 1000;
  let chunkTimes = [], tpsTimer = null;

  function setTps(v) { if (window.setThroughput) window.setThroughput(v); }

  function startThroughput() {
    // Re-armed after a tool call's second pass, so the window is emptied here:
    // stamps from before the tool ran describe a stream that has since stopped.
    chunkTimes = [];
    if (tpsTimer) return;
    tpsTimer = setInterval(() => {
      const cutoff = performance.now() - TPS_WINDOW_MS;
      while (chunkTimes.length && chunkTimes[0] < cutoff) chunkTimes.shift();
      // An empty window is a STALL and reports 0, rather than holding the last
      // rate and claiming the model is still producing.
      setTps(chunkTimes.length / (TPS_WINDOW_MS / 1000));
    }, TPS_TICK_MS);
  }

  function countChunk() {
    chunkTimes.push(performance.now());
  }

  function stopThroughput() {
    if (tpsTimer) { clearInterval(tpsTimer); tpsTimer = null; }
    chunkTimes = [];
    setTps(0);                 // target 0 — the dial eases down to IDLE itself
  }

  /* --- the speech bracket -------------------------------------------------
     Two flags, not one. `expected` spans the whole bracket and is what holds
     `done` off; `audible` spans only real playback and is what gates the level
     meter, so a stray reading outside a burst can never move the waveform.   */
  let speechExpected = false, speechAudible = false;

  function endSpeech() {
    speechExpected = false;
    speechAudible = false;
    // null, not 0: nothing is being measured once the bracket closes. A hard 0
    // would read as a permanent measured silence and pin AUDIO IN there forever.
    setAudioLevel(null);
  }

  function abortTurn() {
    stopThroughput();
    endSpeech();
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

      case "sessions":
        // real conversations (slice 8): on connect and after every turn
        if (window.setSessions) window.setSessions(m.sessions);
        break;

      case "weather":
        // the held reading (slice 8): on connect, on the shared 10-minute
        // poll, and whenever a unit preference changes what it means
        if (window.setWeather) window.setWeather(m);
        break;

      case "speech_pending":
        // the turn holds the sound card; `done` must not rest ahead of it
        speechExpected = true;
        break;

      case "speech_start":
        speechAudible = true;
        setOrbState("speaking");
        break;

      case "audio_level":
        if (speechAudible) setAudioLevel(Math.max(0, Math.min(1, +m.rms || 0)));
        break;

      case "speech_end":
        endSpeech();
        setOrbState(restState());
        break;

      case "recovery":
        logEvent("SYS", "Recovered — retrying the turn.", true);
        break;

      case "error":
        logEvent("SYS", m.message || "Turn failed.", true);
        closeReply();
        stopThroughput();
        if (!speechExpected) setOrbState(restState());
        break;

      case "done":
        closeReply();
        stopThroughput();
        // A speaking turn rests on speech_end instead — the answer is still
        // being read out loud long after its last token arrived.
        if (!speechExpected) setOrbState(restState());
        break;

      default:
        console.warn("[backend] unhandled event:", m.type, m);
    }
  }

  /* --- outbound ----------------------------------------------------------- */
  function sendOp(payload) {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify(payload));
      return true;
    }
    logEvent("SYS", "Not connected — message not sent.", true);
    return false;
  }

  window.sendToBackend = (text) => sendOp({ text });

  /* DISMISS, not delete: the server hides the id and every record stays on
     disk. Nothing is spliced client-side — the next `sessions` frame is the
     answer, so a refusal (the live session) leaves the list as it was. */
  window.dismissSession = (id) => sendOp({ op: "dismiss_session", id });

  connect();
})();