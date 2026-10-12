// Guided Clinical Screening Assistant client logic
(function() {
  let sessionId = null;
  let currentPromptType = null;

  // DOM elements
  const tabQuick = document.getElementById("tab-quick");
  const tabAssistant = document.getElementById("tab-assistant");
  const quickFlow = document.getElementById("quick-flow-container");
  const assistantFlow = document.getElementById("assistant-flow-container");

  const chatHistory = document.getElementById("assistant-chat-history");
  const inputBar = document.getElementById("assistant-input-bar");
  const textInput = document.getElementById("assistant-text-input");
  const sendBtn = document.getElementById("assistant-send-btn");
  const activeChips = document.getElementById("assistant-chips-container");
  const uploaderArea = document.getElementById("assistant-uploader-area");
  const fileInput = document.getElementById("assistant-file-input");
  const restartBtn = document.getElementById("assistant-restart-btn");

  // Tab switching
  if (tabQuick && tabAssistant) {
    tabQuick.addEventListener("click", () => switchTab("quick"));
    tabAssistant.addEventListener("click", () => switchTab("assistant"));
  }

  function switchTab(mode) {
    if (mode === "quick") {
      tabQuick.classList.add("active");
      tabQuick.setAttribute("aria-selected", "true");
      tabAssistant.classList.remove("active");
      tabAssistant.setAttribute("aria-selected", "false");
      quickFlow.hidden = false;
      assistantFlow.hidden = true;
    } else {
      tabAssistant.classList.add("active");
      tabAssistant.setAttribute("aria-selected", "true");
      tabQuick.classList.remove("active");
      tabQuick.setAttribute("aria-selected", "false");
      quickFlow.hidden = true;
      assistantFlow.hidden = false;
      if (!sessionId) {
        initSession();
      }
    }
  }

  // All text is escaped BEFORE the tiny markdown conversion below, so neither user input nor model output
  // can inject HTML.
  const escapeHtml = (v) => String(v).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

  function appendMessage(role, text, isAlert = false) {
    const row = document.createElement("div");
    row.className = `msg-row ${role}`;

    const bubble = document.createElement("div");
    bubble.className = `msg-bubble ${isAlert ? "alert-bubble" : ""}`;

    // Handle basic markdown bullets and linebreaks
    const htmlText = escapeHtml(text)
      .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\n\n/g, "</p><p>")
      .replace(/\n•\s/g, "<br>&bull;&nbsp;")
      .replace(/\n/g, "<br>");

    bubble.innerHTML = `<p>${htmlText}</p>`;
    row.appendChild(bubble);
    chatHistory.appendChild(row);
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }

  function renderChips(chips) {
    activeChips.innerHTML = "";
    if (!chips || chips.length === 0) {
      activeChips.hidden = true;
      return;
    }
    activeChips.hidden = false;
    chips.forEach(chipText => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `chip-btn ${chipText.toLowerCase() === "skip" ? "chip-skip" : ""}`;
      btn.textContent = chipText;
      btn.addEventListener("click", () => {
        sendUserMessage(chipText);
      });
      activeChips.appendChild(btn);
    });
  }

  async function initSession() {
    chatHistory.innerHTML = "";
    activeChips.innerHTML = "";
    activeChips.hidden = true;
    uploaderArea.hidden = true;
    inputBar.hidden = true;
    if (restartBtn) restartBtn.hidden = true;

    appendMessage("assistant", "Starting session...");

    try {
      const res = await fetch("/api/session", { method: "POST" });
      const data = await res.getJson ? await res.getJson() : await res.json();
      if (!res.ok) {
        appendMessage("assistant", data.error || "Unable to start session.");
        return;
      }

      sessionId = data.session_id;
      currentPromptType = data.type;

      // Clear "Starting session..." placeholder
      chatHistory.innerHTML = "";
      appendMessage("assistant", data.message);
      renderChips(data.chips);
      if (restartBtn) restartBtn.hidden = false;
    } catch (err) {
      appendMessage("assistant", "Network error starting session. Please try again.");
    }
  }

  async function sendUserMessage(text) {
    if (!text || !sessionId) return;

    appendMessage("user", text);
    activeChips.innerHTML = "";
    activeChips.hidden = true;
    inputBar.hidden = true;

    // Temporary thinking bubble
    const thinkingRow = document.createElement("div");
    thinkingRow.className = "msg-row assistant";
    thinkingRow.id = "assistant-thinking";
    thinkingRow.innerHTML = '<div class="msg-bubble"><span class="spinner"></span>Processing…</div>';
    chatHistory.appendChild(thinkingRow);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    try {
      const res = await fetch(`/api/session/${sessionId}/message`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ text })
      });
      const data = await res.json();

      const thinkingEl = document.getElementById("assistant-thinking");
      if (thinkingEl) thinkingEl.remove();

      if (!res.ok) {
        appendMessage("assistant", data.error || "An error occurred.");
        return;
      }

      if (data.status === "in_progress") {
        currentPromptType = data.type;
        appendMessage("assistant", data.message);
        if (data.type === "image_upload") {
          showImageUploader();
        } else {
          renderChips(data.chips);
          inputBar.hidden = false;
          textInput.value = "";
          textInput.focus();
        }
      } else if (data.status === "declined") {
        appendMessage("assistant", "You declined consent. You can start a new screening session anytime.");
        if (restartBtn) restartBtn.hidden = false;
      } else if (data.report_ready) {
        loadReport();
      } else {
        appendMessage("assistant", data.message || "Session ended.");
      }
    } catch (err) {
      const thinkingEl = document.getElementById("assistant-thinking");
      if (thinkingEl) thinkingEl.remove();
      appendMessage("assistant", "Network error communicating with the assistant.");
    }
  }

  function showImageUploader() {
    uploaderArea.hidden = false;
    inputBar.hidden = true;
    activeChips.hidden = true;
  }

  // Handle image upload inside assistant
  if (uploaderArea && fileInput) {
    uploaderArea.addEventListener("click", () => fileInput.click());
    uploaderArea.addEventListener("dragover", e => {
      e.preventDefault();
      uploaderArea.classList.add("dragover");
    });
    uploaderArea.addEventListener("dragleave", () => uploaderArea.classList.remove("dragover"));
    uploaderArea.addEventListener("drop", e => {
      e.preventDefault();
      uploaderArea.classList.remove("dragover");
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        uploadAssistantImage(e.dataTransfer.files[0]);
      }
    });

    fileInput.addEventListener("change", () => {
      if (fileInput.files && fileInput.files[0]) {
        uploadAssistantImage(fileInput.files[0]);
      }
    });
  }

  async function uploadAssistantImage(file) {
    if (!sessionId) return;
    uploaderArea.hidden = true;

    // Show image preview bubble
    const previewUrl = URL.createObjectURL(file);
    const row = document.createElement("div");
    row.className = "msg-row user";
    row.innerHTML = `<div class="msg-bubble"><img src="${previewUrl}" style="max-width:200px;border-radius:4px;display:block" alt="Uploaded photo" />Photo uploaded</div>`;
    chatHistory.appendChild(row);

    const thinkingRow = document.createElement("div");
    thinkingRow.className = "msg-row assistant";
    thinkingRow.id = "assistant-vision-thinking";
    thinkingRow.innerHTML = '<div class="msg-bubble"><span class="spinner"></span>Analysing photo with 3-model ensemble &amp; TTA…</div>';
    chatHistory.appendChild(thinkingRow);
    chatHistory.scrollTop = chatHistory.scrollHeight;

    const fd = new FormData();
    fd.append("image", file);

    try {
      const res = await fetch(`/api/session/${sessionId}/image`, {
        method: "POST",
        body: fd
      });
      const data = await res.json();

      const thinkingEl = document.getElementById("assistant-vision-thinking");
      if (thinkingEl) thinkingEl.remove();

      if (!res.ok) {
        appendMessage("assistant", data.error || "Image analysis failed.");
        showImageUploader();
        return;
      }

      if (data.status === "in_progress" && data.type === "image_upload") {
        appendMessage("assistant", data.message || "Quality check failed. Please retake the photo.");
        showImageUploader();
      } else if (data.report_ready) {
        loadReport();
      }
    } catch (err) {
      const thinkingEl = document.getElementById("assistant-vision-thinking");
      if (thinkingEl) thinkingEl.remove();
      appendMessage("assistant", "Network error during image analysis.");
      showImageUploader();
    }
  }

  async function loadReport() {
    try {
      const res = await fetch(`/api/session/${sessionId}/report`);
      const data = await res.json();
      if (!res.ok) {
        appendMessage("assistant", "Report generation encountered an issue.");
        return;
      }

      const rep = data.report;
      const riskClass = rep.final_risk.toLowerCase();

      let topClassesHtml = "";
      (rep.top_classes || []).forEach(c => {
        const p = (c.calibrated_probability || c.probability) * 100;
        topClassesHtml += `<li><strong>${escapeHtml(c.name)}</strong> (${escapeHtml(c.code)}): ${p.toFixed(1)}%</li>`;
      });

      let redFlagsHtml = "";
      if (rep.red_flags && rep.red_flags.length > 0) {
        redFlagsHtml = `<div style="margin-top:12px;color:#c62828"><strong>Acute Concerns Noted:</strong><ul>` +
          rep.red_flags.map(f => `<li>${escapeHtml(f)}</li>`).join("") + `</ul></div>`;
      }

      const reportBox = document.createElement("div");
      reportBox.className = "msg-row assistant";
      reportBox.innerHTML = `
        <div class="report-summary-box" style="width:100%">
          <h3>Clinical Screening Summary</h3>
          <div>Overall Assessment Level: <span class="risk-badge ${riskClass}">${escapeHtml(rep.final_risk)}</span></div>
          <div>Model score on melanoma / basal cell / actinic keratosis classes (uncalibrated, not a risk estimate): <strong>${(rep.concern_probability * 100).toFixed(1)}%</strong></div>
          ${redFlagsHtml}
          <div style="margin-top:12px">
            <strong>Closest image matches (not a diagnosis):</strong>
            <ul style="margin:4px 0 0 0;padding-left:20px">${topClassesHtml}</ul>
          </div>
          <div class="report-actions">
            <a href="/api/session/${sessionId}/report/html" target="_blank" class="btn" style="text-decoration:none">View &amp; Print Full Report (PDF)</a>
            <button type="button" class="btn outline" id="assistant-restart-inner">Start New Assessment</button>
          </div>
        </div>
      `;
      chatHistory.appendChild(reportBox);
      chatHistory.scrollTop = chatHistory.scrollHeight;

      document.getElementById("assistant-restart-inner")?.addEventListener("click", initSession);
    } catch (err) {
      appendMessage("assistant", "Error rendering final report.");
    }
  }

  // Form submit for text input
  if (inputBar && textInput) {
    inputBar.addEventListener("submit", e => {
      e.preventDefault();
      const text = textInput.value.trim();
      if (text) {
        sendUserMessage(text);
      }
    });
  }

  if (restartBtn) {
    restartBtn.addEventListener("click", () => {
      if (confirm("Restart guided screening session? Current session progress will be lost.")) {
        initSession();
      }
    });
  }
})();
