const $ = (id) => document.getElementById(id);
let current = null; // File chosen by the user or loaded from a sample
let lastSummary = "";
let lastCodes = [];
let heatmapUrl = null;
let originalUrl = null;

function setFile(file) {
  current = file;
  $("go").disabled = !file;
  $("error").textContent = "";
  if (!file) return;
  $("preview").src = URL.createObjectURL(file);
  $("preview").hidden = false;
  $("drop-empty").hidden = true;
}

const drop = $("drop");
drop.addEventListener("click", () => $("file").click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("file").click(); } });
$("file").addEventListener("change", (e) => setFile(e.target.files[0]));
["dragenter", "dragover"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.add("over"); }));
["dragleave", "drop"].forEach((t) => drop.addEventListener(t, (e) => { e.preventDefault(); drop.classList.remove("over"); }));
drop.addEventListener("drop", (e) => setFile(e.dataTransfer.files[0]));

document.querySelectorAll(".samples button").forEach((b) =>
  b.addEventListener("click", async () => {
    const blob = await (await fetch(b.dataset.src)).blob();
    setFile(new File([blob], "sample.jpg", { type: blob.type }));
  })
);

function show(view) {
  $("upload-view").hidden = view !== "upload";
  $("result-view").hidden = view !== "result";
  window.scrollTo(0, 0);
}

function render(d) {
  const top = d.predictions[0];
  originalUrl = $("preview").src;
  heatmapUrl = d.heatmap || null;
  $("result-img").src = originalUrl;
  $("cam-row").hidden = !heatmapUrl;
  $("cam-toggle").textContent = "Show where the model looked";
  $("cam-toggle").setAttribute("aria-pressed", "false");
  $("cam-note").hidden = true;
  $("pred").textContent = d.info ? d.info.title : top.name;
  $("conf").textContent = Math.round(top.probability * 100) + "%";
  $("risk").textContent = { low: "not flagged", elevated: "review advised", high: "see a doctor" }[d.risk] || d.risk;
  $("risk").className = "risk " + (d.risk === "low" ? "elevated" : d.risk);  // never styled as "all clear" green
  $("explanation").textContent = d.explanation;
  $("about-title").textContent = "More About " + (d.info ? d.info.title : top.name);
  $("about-text").textContent = d.info ? d.info.summary : "";
  $("bars").replaceChildren(...d.predictions.slice(0, 3).map((p) => {
    const pct = (p.probability * 100).toFixed(1);
    const row = document.createElement("div");
    const head = document.createElement("div");
    head.className = "bar-row";
    const a = document.createElement("span"); a.textContent = p.name;
    const b = document.createElement("span"); b.textContent = pct + "%";
    head.append(a, b);
    const bar = document.createElement("div"); bar.className = "bar";
    const fill = document.createElement("span"); fill.style.width = pct + "%";
    bar.append(fill); row.append(head, bar);
    return row;
  }));
  lastCodes = d.predictions.slice(0, 3).map((p) => p.code);
  lastSummary = `Skin lesion screening (research prototype): closest match ${top.name} ` +
    `(${Math.round(top.probability * 100)}% confidence), flag: ${$("risk").textContent}. Not a diagnosis; not reassurance.`;
}

$("go").addEventListener("click", async () => {
  if (!current) return;
  $("go").disabled = true; $("go").textContent = "Analysing...";
  $("error").textContent = "";
  const body = new FormData();
  body.append("image", current);
  try {
    const r = await fetch("/api/predict", { method: "POST", body });
    const d = await r.json();
    if (!r.ok) throw new Error(d.error || "Request failed");
    render(d);
    show("result");
  } catch (err) {
    $("error").textContent = err.message;
  }
  $("go").disabled = false; $("go").textContent = "Get Results";
});

$("back").addEventListener("click", () => show("upload"));
$("share").addEventListener("click", async () => {
  try { await navigator.clipboard.writeText(lastSummary); $("share").textContent = "Copied!"; }
  catch { window.prompt("Copy your results summary:", lastSummary); }
  setTimeout(() => ($("share").textContent = "Share Results"), 1800);
});

const askForm = $("ask-form");
if (askForm) {
  askForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    const out = $("ask-a");
    out.textContent = "Thinking...";
    try {
      const r = await fetch("/api/chat", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: $("ask-q").value, codes: lastCodes }),
      });
      const d = await r.json();
      out.textContent = r.ok ? d.answer : d.error || "Request failed";
    } catch { out.textContent = "Could not reach the server."; }
  });
}

$("cam-toggle").addEventListener("click", () => {
  const on = $("cam-toggle").getAttribute("aria-pressed") !== "true";
  $("result-img").src = on ? heatmapUrl : originalUrl;
  $("cam-toggle").setAttribute("aria-pressed", String(on));
  $("cam-toggle").textContent = on ? "Show original image" : "Show where the model looked";
  $("cam-note").hidden = !on;
});
