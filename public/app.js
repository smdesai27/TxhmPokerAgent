(function () {
  "use strict";

  const API_BASE = window.POKER_API_BASE || "";

  let sessionId = null;
  let autoContinueTimer = null;

  const $ = (sel) => document.querySelector(sel);

  // DOM refs
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

  // API helpers
  async function api(method, path, body) {
    const opts = { method, headers: { "Content-Type": "application/json" } };
    if (body !== undefined) opts.body = JSON.stringify(body);
    const res = await fetch(API_BASE + path, opts);
    if (!res.ok) {
      let msg;
      try {
        const err = await res.json();
        msg = err.detail || res.statusText;
      } catch {
        msg = res.statusText;
      }
      throw new Error(msg);
    }
    return res.json();
  }

  function showToast(msg) {
    toast.textContent = msg;
    toast.classList.add("visible");
    setTimeout(() => toast.classList.remove("visible"), 3000);
  }

  // Card rendering
  function renderCard(cardInfo, hidden) {
    const el = document.createElement("span");
    el.classList.add("card");
    if (hidden || !cardInfo) {
      el.classList.add("face-down");
      el.textContent = "?";
      return el;
    }
    el.classList.add("face-up");
    const suit = cardInfo.display.slice(-1);
    if (suit === "\u2665" || suit === "\u2666") {
      el.classList.add("red");
    } else {
      el.classList.add("black");
    }
    el.textContent = cardInfo.display;
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

  // Strip OpenSpiel prefix ("player=X move=...") and produce clean button label
  function cleanActionName(raw) {
    let name = raw;
    // Strip "player=N move=" prefix if present
    const moveIdx = name.indexOf("move=");
    if (moveIdx !== -1) {
      name = name.slice(moveIdx + 5);
    }
    // Normalize specific patterns
    const lower = name.toLowerCase().trim();
    if (lower === "fold") return "Fold";
    if (lower === "check") return "Check";
    if (lower === "call") return "Call";
    if (lower === "allin" || lower === "all-in" || lower === "all in") return "All In";
    // "RaiseTo 200" -> "Raise 200"
    const raiseMatch = name.match(/^RaiseTo\s+(\d+)/i);
    if (raiseMatch) return "Raise " + raiseMatch[1];
    // Fallback: simple names like "raise half pot"
    if (lower.startsWith("raise") || lower.startsWith("bet")) {
      return name.charAt(0).toUpperCase() + name.slice(1);
    }
    return name;
  }

  // Action button classification
  function classifyAction(name) {
    const n = name.toLowerCase();
    if (n.includes("fold")) return "fold";
    if (n.includes("check")) return "check";
    if (n.includes("call")) return "call";
    if (n.includes("all-in") || n.includes("allin") || n.includes("all in")) return "allin";
    if (n.includes("raise") || n.includes("bet")) return "raise";
    return "default";
  }

  // Main render
  function renderState(state, stats) {
    placeholder.style.display = "none";

    // Bot cards
    const botCards = $("#bot-cards");
    if (state.is_terminal && state.bot_cards.length > 0) {
      renderCards(botCards, state.bot_cards, false);
    } else {
      renderCards(botCards, [], true, 2);
    }
    $("#bot-stack").textContent = state.bot_stack.toFixed(0);

    // Board
    renderCards($("#board-cards"), state.board_cards, false);
    $("#pot-label").textContent = "Pot: " + state.pot_size.toFixed(0);
    $("#street-label").textContent = state.street;

    // Hero cards
    renderCards($("#hero-cards"), state.hero_cards, false);
    $("#hero-stack").textContent = state.human_stack.toFixed(0);

    // Actions
    actionsBar.innerHTML = "";
    if (!state.is_terminal && state.current_player === "human") {
      for (const la of state.legal_actions) {
        const btn = document.createElement("button");
        btn.classList.add("action-btn", classifyAction(la.name));
        btn.textContent = cleanActionName(la.name);
        btn.addEventListener("click", () => doAction(la.action_id));
        actionsBar.appendChild(btn);
      }
    }

    btnNewHand.disabled = !state.is_terminal;

    // Result overlay / auto-continue
    if (autoContinueTimer) {
      clearTimeout(autoContinueTimer);
      autoContinueTimer = null;
    }
    if (state.is_terminal && state.hand_result) {
      if (chkAutoContinue.checked && sessionId) {
        // Skip overlay, auto-deal next hand after brief pause
        resultOverlay.classList.add("hidden");
        autoContinueTimer = setTimeout(() => {
          autoContinueTimer = null;
          doNewHand();
        }, 600);
      } else {
        const hr = state.hand_result;
        resultChips.textContent = (hr.human_chips >= 0 ? "+" : "") + hr.human_chips.toFixed(0) + " chips";
        resultChips.className = "chips " + (hr.human_chips >= 0 ? "positive" : "negative");
        resultDetails.textContent =
          "(" + (hr.human_bb >= 0 ? "+" : "") + hr.human_bb.toFixed(2) + " bb)";
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
      div.classList.add("event", ev.actor);
      div.textContent = `[${ev.actor}] ${ev.action_name}`;
      historyLog.appendChild(div);
    }
    if (isTerminal && handResult) {
      const div = document.createElement("div");
      div.classList.add("event", "system");
      div.textContent = `Result: ${handResult.human_chips >= 0 ? "+" : ""}${handResult.human_chips.toFixed(0)} chips`;
      historyLog.appendChild(div);
    }
    historyLog.scrollTop = historyLog.scrollHeight;
  }

  // Actions
  async function doNewSession() {
    try {
      const seat = parseInt(seatSelect.value, 10);
      const data = await api("POST", "/api/v1/session", { human_seat: seat });
      sessionId = data.game_state.session_id;
      historyLog.innerHTML = "";
      const sysDiv = document.createElement("div");
      sysDiv.classList.add("event", "system");
      sysDiv.textContent = `--- New Session (seat P${seat}) ---`;
      historyLog.appendChild(sysDiv);
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast("Error: " + e.message);
    }
  }

  async function doAction(actionId) {
    try {
      const data = await api("POST", `/api/v1/session/${sessionId}/action`, { action_id: actionId });
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast("Error: " + e.message);
    }
  }

  async function doNewHand() {
    if (!sessionId) return;
    try {
      resultOverlay.classList.add("hidden");
      const data = await api("POST", `/api/v1/session/${sessionId}/new_hand`);
      const sysDiv = document.createElement("div");
      sysDiv.classList.add("event", "system");
      sysDiv.textContent = `--- Hand #${data.game_state.hand_number} ---`;
      historyLog.appendChild(sysDiv);
      renderState(data.game_state, data.stats);
    } catch (e) {
      showToast("Error: " + e.message);
    }
  }

  // Events
  btnNewSession.addEventListener("click", doNewSession);
  btnNewHand.addEventListener("click", doNewHand);
  resultNext.addEventListener("click", doNewHand);
})();
