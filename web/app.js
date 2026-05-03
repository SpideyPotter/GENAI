/**
 * AgriChat — POST JSON to your inference API, or use demo mode when empty.
 * Expected response: { "reply": "..." } or { "message": "..." } or { "choices": [{ "message": { "content": "..." } }] }
 */
(function () {
  const SYSTEM_CONTEXT =
    "You are an experienced agronomist assistant. You help with farming techniques, crop management, soil health, pests, sustainable practices, and regional agriculture guidance. Be clear, practical, and safety-conscious when discussing chemicals or livestock.";

  const els = {
    chat: document.getElementById("chat-messages"),
    form: document.getElementById("chat-form"),
    input: document.getElementById("message-input"),
    send: document.getElementById("btn-send"),
    model: document.getElementById("model-select"),
    apiUrl: document.getElementById("api-url"),
    clear: document.getElementById("btn-clear"),
    status: document.getElementById("connection-status"),
  };

  let history = [];

  function loadHistory() {
    try {
      const raw = localStorage.getItem("agrichat_history");
      if (raw) history = JSON.parse(raw);
    } catch {
      history = [];
    }
  }

  function saveHistory() {
    try {
      localStorage.setItem("agrichat_history", JSON.stringify(history.slice(-40)));
    } catch {
      /* ignore */
    }
  }

  function scrollToBottom() {
    els.chat.scrollTop = els.chat.scrollHeight;
  }

  function appendMessage(role, text, className = role) {
    const div = document.createElement("div");
    div.className = `msg ${className}`;
    if (role === "assistant") {
      const roleEl = document.createElement("div");
      roleEl.className = "role";
      roleEl.textContent = "Agronomist";
      div.appendChild(roleEl);
    }
    const body = document.createElement("div");
    body.textContent = text;
    div.appendChild(body);
    els.chat.appendChild(div);
    scrollToBottom();
    return div;
  }

  function showTyping() {
    const wrap = document.createElement("div");
    wrap.className = "typing";
    wrap.id = "typing-indicator";
    wrap.innerHTML = "<span></span><span></span><span></span>";
    els.chat.appendChild(wrap);
    scrollToBottom();
    return wrap;
  }

  function hideTyping() {
    document.getElementById("typing-indicator")?.remove();
  }

  function parseReply(data) {
    if (typeof data === "string") return data;
    if (data.reply) return data.reply;
    if (data.message) return data.message;
    if (data.response) return data.response;
    if (data.choices?.[0]?.message?.content) return data.choices[0].message.content;
    if (data.choices?.[0]?.text) return data.choices[0].text;
    return JSON.stringify(data);
  }

  function demoReply(userText) {
    const lower = userText.toLowerCase();
    if (lower.includes("soil") || lower.includes("nitrogen"))
      return "For soil health, start with a recent soil test (pH, organic matter, N-P-K). Match nitrogen applications to crop stage and use split applications where runoff risk is high. Cover crops between seasons help build organic matter and reduce erosion.";
    if (lower.includes("pest") || lower.includes("insect"))
      return "Integrated pest management (IPM) works best: scout regularly, identify the pest correctly, and use thresholds before spraying. Rotate modes of action and preserve beneficial insects where possible. Always follow label rates and pre-harvest intervals.";
    if (lower.includes("water") || lower.includes("irrigation"))
      return "Match irrigation to soil moisture and crop growth stage. Tools like tensiometers or soil probes reduce over-watering. Early morning irrigation often cuts disease pressure compared to evening wet foliage.";
    return (
      "Thanks for your question about agriculture. In a full setup, this answer would come from your fine-tuned model. " +
      "Set the API URL in the sidebar to your inference server (POST JSON with `message` and optional `model`). " +
      "Meanwhile: be specific about your crop, region, and season for the most useful agronomic advice."
    );
  }

  async function getAssistantReply(userText) {
    const url = (els.apiUrl.value || "").trim();
    const model = els.model.value;

    if (!url) {
      await new Promise((r) => setTimeout(r, 450 + Math.random() * 400));
      return { text: demoReply(userText), demo: true };
    }

    const body = {
      message: userText,
      model,
      system: SYSTEM_CONTEXT,
      history: history.map((h) => ({ role: h.role, content: h.content })),
    };

    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });

    if (!res.ok) {
      const errText = await res.text().catch(() => res.statusText);
      throw new Error(errText || `HTTP ${res.status}`);
    }

    const data = await res.json();
    return { text: parseReply(data), demo: false };
  }

  function setStatus(demo) {
    els.status.textContent = demo ? "Demo mode" : "API connected";
    els.status.className = "status-pill" + (demo ? " demo" : "");
  }

  function renderHistory() {
    els.chat.innerHTML = "";
    history.forEach((h) => {
      appendMessage(h.role, h.content, h.role);
    });
    scrollToBottom();
  }

  async function onSubmit(e) {
    e.preventDefault();
    const text = els.input.value.trim();
    if (!text) return;

    els.send.disabled = true;
    els.input.value = "";

    appendMessage("user", text, "user");
    history.push({ role: "user", content: text });
    saveHistory();

    showTyping();
    try {
      const { text: reply, demo } = await getAssistantReply(text);
      hideTyping();
      appendMessage("assistant", reply, "assistant");
      history.push({ role: "assistant", content: reply });
      saveHistory();
      setStatus(demo);
    } catch (err) {
      hideTyping();
      const msg = err instanceof Error ? err.message : String(err);
      appendMessage("assistant", "Could not reach the API: " + msg, "error");
      setStatus(!(els.apiUrl.value || "").trim());
    } finally {
      els.send.disabled = false;
      els.input.focus();
    }
  }

  els.form.addEventListener("submit", onSubmit);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      els.form.requestSubmit();
    }
  });

  els.clear.addEventListener("click", () => {
    history = [];
    saveHistory();
    els.chat.innerHTML = "";
    welcome();
    setStatus(!(els.apiUrl.value || "").trim());
  });

  els.apiUrl.addEventListener("change", () => {
    setStatus(!(els.apiUrl.value || "").trim());
  });

  function welcome() {
    const intro =
      "Hello — I’m your agriculture assistant. Ask about crops, soil, nutrients, pests, equipment, or sustainable practices. " +
      "Connect your model API in the sidebar for live answers from your fine-tuned Llama checkpoints.";
    appendMessage("assistant", intro, "assistant");
  }

  const savedUrl = localStorage.getItem("agrichat_api_url");
  if (savedUrl) els.apiUrl.value = savedUrl;

  loadHistory();
  if (history.length) {
    renderHistory();
    setStatus(!(els.apiUrl.value || "").trim());
  } else {
    welcome();
    setStatus(!(els.apiUrl.value || "").trim());
  }

  els.apiUrl.addEventListener("change", () => {
    localStorage.setItem("agrichat_api_url", els.apiUrl.value.trim());
    setStatus(!(els.apiUrl.value || "").trim());
  });

  els.input.focus();
})();
