const DIMS = ["truthful", "vulnerable", "faithful", "respected"];

let items = [];       // blinded result records
let scores = {};      // { index: { truthful: N, vulnerable: N, faithful: N, respected: N } }
let current = 0;
let runMeta = {};

// ── File loading ──────────────────────────────────────────────────────────────

document.getElementById("file-input").addEventListener("change", (e) => {
  const file = e.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (ev) => {
    try {
      const data = JSON.parse(ev.target.result);
      items = data.results;
      runMeta = { timestamp: data.timestamp, model: data.model, filename: file.name };
      scores = {};
      current = 0;
      show("score-screen");
      render();
    } catch {
      alert("Could not parse file. Make sure you selected a *_blinded.json file.");
    }
  };
  reader.readAsText(file);
});

// ── Rendering ─────────────────────────────────────────────────────────────────

function render() {
  if (current >= items.length) {
    showDone();
    return;
  }

  const item = items[current];
  const saved = scores[current] || {};

  document.getElementById("case-id").textContent = item.case_id;
  document.getElementById("case-group").textContent = `Group ${item.group}`;
  document.getElementById("arm-code").textContent = `Arm ${item.arm_code}`;
  document.getElementById("case-title").textContent = item.title;
  document.getElementById("case-prompt").textContent = item.prompt;
  document.getElementById("answer-text").textContent = item.answer;

  DIMS.forEach((dim) => {
    const container = document.querySelector(`.stars[data-dim="${dim}"]`);
    container.innerHTML = "";
    for (let i = 1; i <= 5; i++) {
      const star = document.createElement("span");
      star.className = "star" + (i <= (saved[dim] || 0) ? " active" : "");
      star.textContent = "★";
      star.dataset.value = i;
      star.addEventListener("click", () => setStar(dim, i));
      star.addEventListener("mouseover", () => highlightStars(container, i));
      star.addEventListener("mouseout", () => highlightStars(container, saved[dim] || 0));
      container.appendChild(star);
    }
  });

  const pct = Math.round((current / items.length) * 100);
  document.getElementById("progress-bar").style.width = pct + "%";
  document.getElementById("progress-label").textContent =
    `${current + 1} of ${items.length} answers`;
  document.getElementById("nav-label").textContent =
    `${current + 1} / ${items.length}`;

  document.getElementById("btn-prev").disabled = current === 0;
  document.getElementById("btn-next").textContent =
    current === items.length - 1 ? "Finish →" : "Next →";
}

function highlightStars(container, upTo) {
  container.querySelectorAll(".star").forEach((s) => {
    s.classList.toggle("active", parseInt(s.dataset.value) <= upTo);
  });
}

function setStar(dim, value) {
  if (!scores[current]) scores[current] = {};
  scores[current][dim] = value;
  render();
}

// ── Navigation ────────────────────────────────────────────────────────────────

document.getElementById("btn-prev").addEventListener("click", () => {
  if (current > 0) { current--; render(); scrollToTop(); }
});

document.getElementById("btn-next").addEventListener("click", () => {
  const saved = scores[current] || {};
  const missing = DIMS.filter((d) => !saved[d]);
  if (missing.length) {
    alert(`Please score all 4 dimensions before continuing.\nMissing: ${missing.join(", ")}`);
    return;
  }
  current++;
  render();
  scrollToTop();
});

function scrollToTop() {
  window.scrollTo({ top: 0, behavior: "smooth" });
}

// ── Done / Export ─────────────────────────────────────────────────────────────

function showDone() {
  const scored = Object.keys(scores).length;
  document.getElementById("done-summary").textContent =
    `${scored} of ${items.length} answers scored. Model: ${runMeta.model || "unknown"}`;
  show("done-screen");
  document.getElementById("progress-bar").style.width = "100%";
  document.getElementById("progress-label").textContent = "Complete";
}

document.getElementById("btn-export").addEventListener("click", exportCSV);

function exportCSV() {
  const header = ["case_id", "group", "title", "arm_code", ...DIMS];
  const rows = items.map((item, i) => {
    const s = scores[i] || {};
    return [
      item.case_id,
      item.group,
      `"${item.title.replace(/"/g, '""')}"`,
      item.arm_code,
      ...DIMS.map((d) => s[d] || ""),
    ];
  });

  const csv = [header, ...rows].map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `scores_${runMeta.timestamp || Date.now()}.csv`;
  a.click();
  URL.revokeObjectURL(url);
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function show(id) {
  ["load-screen", "score-screen", "done-screen"].forEach((s) => {
    document.getElementById(s).classList.toggle("hidden", s !== id);
  });
}
