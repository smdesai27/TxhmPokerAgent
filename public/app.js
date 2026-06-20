(function () {
  "use strict";

  const API_BASE = window.POKER_API_BASE || "";
  const GITHUB_URL = window.POKER_GITHUB_URL || "";
  const WRITEUP_URL = window.POKER_WRITEUP_URL || GITHUB_URL;

  let sessionId = null;
  let autoContinueTimer = null;
  let busy = false;

  const $ = (sel) => document.querySelector(sel);

  const btnNewSession = $("#btn-new-session");
  const btnNewHand = $("#btn-new-hand");
  const chkAutoContinue = $("#chk-auto-continue");
  const seatSelect = $("#seat-select");
  const actionsBar = $("#actions-bar");
  const placeholder = $("#placeholder");
  const resultOverlay = $("#result-overlay");
  const resultChips = $("#result-chips");
  const resultDetails = $("#result-details");
  const resultNext = $("#result-next");
  const historyLog = $("#history-log");
  const toast = $("#toast");
  const turnIndicator = $("#turn-indicator");
  const statusDot = $("#status-dot");
  const badgeText = $("#agent-badge-text");

  // ---- wire portfolio links ----
  if (GITHUB_URL) $("#link-github").href = GITHUB_URL;
  if (WRITEUP_URL) $("#link-writeup").href = WRITEUP_URL;

  // ---- API ----
  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    let res;
    try {
      res = await fetch(API_BASE + path, opts);
    } catch {
      throw new Error(
        API_BASE
          ? "Can't reach the agent backend. Free-tier servers sleep when idle — give it ~30s and retry."
          : "Backend not connected. Set POKER_API_BASE in public/config.js."
      );
    }
    if (!res.ok) {
      let msg;
      try { msg = (await res.json()).detail || res.statusText; } catch { msg = res.statusText; }
      throw new Error(msg);
    }
    return res.json();
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 4000);
  }

  // ---- agent status badge (handles Render free-tier cold start) ----
  async function refreshHealth(attempt) {
    try {
      const h = await api("GET", "/api/v1/health");
      const live = h.status === "ok" && h.model_loaded;
      statusDot.className = "status-dot " + (live ? "online" : "offline");
      const mode = (h.game_mode || "").toUpperCase();
      badgeText.innerHTML = live
        ? `<strong>Agent online</strong>${mode ? " · " + mode : ""}`
        : "agent loading…";
      const badge = $("#agent-badge");
      if (h.checkpoint_label) badge.title = h.checkpoint_label;
    } catch {
      statusDot.className = "status-dot offline";
      badgeText.textContent = attempt ? "agent asleep — click New session to wake it" : "waking agent… (free tier)";
      if (!attempt) setTimeout(() => refreshHealth(1), 3500);
    }
  }

  // ---- card rendering (corner pips + center suit) ----
  const RED_SUITS = new Set(["♥", "♦"]); // ♥ ♦
  function renderCard(cardInfo, hidden) {
    const el = document.createElement("span");
    el.className = "card";
    if (hidden || !cardInfo) { el.classList.add("face-down"); return el; }
    const disp = cardInfo.display || "";
    const suit = disp.slice(-1);
    const rank = disp.slice(0, -1) || disp;
    el.classList.add("face-up", RED_SUITS.has(suit) ? "red" : "black");
    el.innerHTML =
      `<span class="card-corner tl">${rank}<span class="suit">${suit}</span></span>` +
      `<span class="card-suit-center">${suit}</span>` +
      `<span class="card-corner br">${rank}<span class="suit">${suit}</span></span>`;
    return el;
  }

  function renderCards(container, cards, hidden, count) {
    container.innerHTML = "";
    if (hidden && count) {
      for (let i = 0; i < count; i++) container.appendChild(renderCard(null, true));
      return;
    }
    for (const c of cards) container.appendChild(renderCard(c, false));
  }

  function cleanActionName(raw) {
    let name = raw;
    const moveIdx = name.indexOf("move=");
    if (moveIdx !== -1) name = name.slice(moveIdx + 5);
    const lower = name.toLowerCase().trim();
    if (lower === "fold") return "Fold";
    if (lower === "check") return "Check";
    if (lower === "call") return "Call";
    if (lower === "allin" || lower === "all-in" || lower === "all in") return "All In";
    const raiseMatch = name.match(/^RaiseTo\s+(\d+)/i);
    if (raiseMatch) return "Raise to " + raiseMatch[1];
    if (lower.startsWith("raise") || lower.startsWith("bet")) return name.charAt(0).toUpperCase() + name.slice(1);
    return name;
  }

  function classifyAction(name) {
    const n = name.toLowerCase();
    if (n.includes("fold")) return "fold";
    if (n.includes("check")) return "check";
    if (n.includes("call")) return "call";
    if (n.includes("all-in") || n.includes("allin") || n.includes("all in")) return "allin";
    if (n.includes("raise") || n.includes("bet")) return "raise";
    return "default";
  }

  function setTurn(text, thinking) {
    if (!text) { turnIndicator.classList.add("hidden"); return; }
    turnIndicator.textContent = text;
    turnIndicator.className = "turn-indicator" + (thinking ? " thinking" : "");
  }

  function setBusy(state) {
    busy = state;
    btnNewSession.disabled = state;
    btnNewHand.disabled = state || !sessionId;
    for (const b of actionsBar.querySelectorAll("button")) b.disabled = state;
    if (state) setTurn("Agent is thinking…", true);
  }

  // ---- main render ----
  function renderState(state, stats) {
    placeholder.style.display = "none";

    const botCards = $("#bot-cards");
    if (state.is_terminal && state.bot_cards.length > 0) renderCards(botCards, state.bot_cards, false);
    else renderCards(botCards, [], true, 2);
    $("#bot-stack").textContent = state.bot_stack.toFixed(0);

    renderCards($("#board-cards"), state.board_cards, false);
    $("#pot-label").textContent = "Pot " + state.pot_size.toFixed(0);
    $("#street-label").textContent = state.street;

    renderCards($("#hero-cards"), state.hero_cards, false);
    $("#hero-stack").textContent = state.human_stack.toFixed(0);

    actionsBar.innerHTML = "";
    const yourMove = !state.is_terminal && state.current_player === "human";
    if (yourMove) {
      for (const la of state.legal_actions) {
        const btn = document.createElement("button");
        btn.className = "action-btn " + classifyAction(la.name);
        btn.textContent = cleanActionName(la.name);
        btn.addEventListener("click", () => doAction(la.action_id));
        actionsBar.appendChild(btn);
      }
      setTurn("Your move", false);
    } else if (!state.is_terminal) {
      setTurn("Agent is thinking…", true);
    } else {
      setTurn("", false);
    }

    btnNewHand.disabled = !state.is_terminal;

    if (autoContinueTimer) { clearTimeout(autoContinueTimer); autoContinueTimer = null; }
    if (state.is_terminal && state.hand_result) {
      const hr = state.hand_result;
      if (chkAutoContinue.checked && sessionId) {
        resultOverlay.classList.add("hidden");
        autoContinueTimer = setTimeout(() => { autoContinueTimer = null; doNewHand(); }, 700);
      } else {
        resultChips.textContent = (hr.human_chips >= 0 ? "+" : "") + hr.human_chips.toFixed(0) + " chips";
        resultChips.className = "chips " + (hr.human_chips >= 0 ? "positive" : "negative");
        resultDetails.textContent = "(" + (hr.human_bb >= 0 ? "+" : "") + hr.human_bb.toFixed(2) + " bb)";
        resultOverlay.classList.remove("hidden");
      }
    } else {
      resultOverlay.classList.add("hidden");
    }

    renderStats(stats);
    appendHistory(state.events_since_last_action, state.is_terminal, state.hand_result);
  }

  function renderStats(stats) {
    $("#stat-hands").textContent = stats.hands_played;
    $("#stat-human-chips").textContent = stats.total_human_chips.toFixed(0);
    $("#stat-human-bb").textContent = stats.total_human_bb.toFixed(2);
    $("#stat-bot-chips").textContent = stats.total_bot_chips.toFixed(0);
    $("#stat-bot-bb").textContent = stats.total_bot_bb.toFixed(2);
  }

  function appendHistory(events, isTerminal, handResult) {
    for (const ev of events) {
      const div = document.createElement("div");
      div.className = "event " + ev.actor;
      div.textContent = `[${ev.actor}] ${ev.action_name}`;
      historyLog.appendChild(div);
    }
    if (isTerminal && handResult) {
      const div = document.createElement("div");
      div.className = "event system";
      div.textContent = `Result: ${handResult.human_chips >= 0 ? "+" : ""}${handResult.human_chips.toFixed(0)} chips`;
      historyLog.appendChild(div);
    }
    historyLog.scrollTop = historyLog.scrollHeight;
  }

  function sysLine(text) {
    const div = document.createElement("div");
    div.className = "event system";
    div.textContent = text;
    historyLog.appendChild(div);
  }

  // ---- actions ----
  async function doNewSession() {
    if (busy) return;
    setBusy(true);
    try {
      const seat = parseInt(seatSelect.value, 10);
      const data = await api("POST", "/api/v1/session", { human_seat: seat });
      sessionId = data.game_state.session_id;
      historyLog.innerHTML = "";
      sysLine(`--- New session (you are P${seat}) ---`);
      statusDot.className = "status-dot online";
      badgeText.innerHTML = "<strong>Agent online</strong>";
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast(e.message);
      setTurn("", false);
    } finally {
      setBusy(false);
    }
  }

  async function doAction(actionId) {
    if (busy) return;
    setBusy(true);
    try {
      const data = await api("POST", `/api/v1/session/${sessionId}/action`, { action_id: actionId });
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast(e.message);
      setTurn("Your move", false);
    } finally {
      setBusy(false);
    }
  }

  async function doNewHand() {
    if (!sessionId || busy) return;
    setBusy(true);
    try {
      resultOverlay.classList.add("hidden");
      const data = await api("POST", `/api/v1/session/${sessionId}/new_hand`);
      sysLine(`--- Hand #${data.game_state.hand_number} ---`);
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast(e.message);
    } finally {
      setBusy(false);
    }
  }

  btnNewSession.addEventListener("click", doNewSession);
  btnNewHand.addEventListener("click", doNewHand);
  resultNext.addEventListener("click", doNewHand);

  refreshHealth(0);
})();
