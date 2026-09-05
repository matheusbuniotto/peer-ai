async function main() {
  const report = await fetch("results.json").then((r) => r.json());

  document.getElementById("generated-at").textContent =
    "generated " + report.generated_at;

  renderSummary(report);
  renderByFlaw(report);
  renderModels(report);
  renderCases(report);
  renderPublishedFailures(report);
}

function pct(x) {
  return (x * 100).toFixed(1) + "%";
}

function renderSummary(report) {
  const stats = [
    ["accuracy", report.accuracy],
    ["false ships", report.false_ships],
    ["false blocks", report.false_blocks],
  ];
  const el = document.getElementById("summary");
  el.innerHTML = stats
    .map(
      ([label, value]) =>
        `<div class="stat"><div class="value">${pct(value)}</div><div class="label">${label}</div></div>`
    )
    .join("");
}

function renderByFlaw(report) {
  const el = document.getElementById("by-flaw");
  const flaws = Object.entries(report.by_flaw).sort(([a], [b]) => a.localeCompare(b));
  el.innerHTML = flaws
    .map(
      ([flaw, score]) => `
      <div class="bar-row">
        <div class="name">${flaw}</div>
        <div class="bar-track"><div class="bar-fill" style="width:${score * 100}%"></div></div>
        <div class="pct">${pct(score)}</div>
      </div>`
    )
    .join("");
}

function renderModels(report) {
  const tbody = document.querySelector("#models tbody");
  tbody.innerHTML = report.models
    .map((m) => `<tr><td>${m.name}</td><td>${pct(m.accuracy)}</td></tr>`)
    .join("");
}

function renderCases(report) {
  const tbody = document.querySelector("#cases tbody");
  tbody.innerHTML = "";
  for (const c of report.cases) {
    const row = document.createElement("tr");
    row.innerHTML = `
      <td>${c.correct ? '<span class="ok">✓</span>' : '<span class="fail">✗</span>'}</td>
      <td>${c.id}</td>
      <td>${c.flaw}</td>
      <td>${c.truth}</td>
      <td>${c.verdict ?? "—"}</td>
    `;

    const detail = document.createElement("tr");
    detail.className = "trajectory";
    const cell = document.createElement("td");
    cell.colSpan = 5;
    cell.innerHTML = c.trajectory
      .map(
        (t) =>
          `<div class="turn"><span class="act">[${t.turn}] ${t.act}</span>\n${escapeHtml(
            JSON.stringify(t.body, null, 2)
          )}</div>`
      )
      .join("");
    detail.appendChild(cell);

    row.addEventListener("click", () => detail.classList.toggle("open"));

    tbody.appendChild(row);
    tbody.appendChild(detail);
  }
}

function renderPublishedFailures(report) {
  const el = document.getElementById("published");
  if (!report.published_failures.length) {
    el.textContent = "";
    return;
  }
  const ids = report.published_failures.map((f) => f.case_id).join(", ");
  el.textContent = `Known failing cases: ${ids}`;
}

function escapeHtml(s) {
  return s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" })[c]);
}

main();
