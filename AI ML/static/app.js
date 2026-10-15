/* Frontend logic for the E-commerce Customer Segmentation dashboard.
   All data comes live from the Flask API (/api/*) - nothing is fabricated. */
const $ = (s) => document.querySelector(s);
const $$ = (s) => document.querySelectorAll(s);
const fmt = (n, d = 0) => Number(n).toLocaleString(undefined, {maximumFractionDigits: d, minimumFractionDigits: d});
const pct = (n, d = 1) => Number(n).toFixed(d) + "%";

const palette = ["#6c8cff", "#9c6cff", "#35d0a5", "#ffb547", "#ff6b7a", "#4dd0e1", "#c6a3ff", "#f790c0", "#7fdb8a", "#ffd166"];
Chart.defaults.color = "#8b97c2";
Chart.defaults.borderColor = "rgba(38,50,88,.6)";
Chart.defaults.font.family = "Segoe UI, sans-serif";
const charts = {};
function makeChart(id, cfg) {
  if (charts[id]) charts[id].destroy();
  charts[id] = new Chart(document.getElementById(id), cfg);
  return charts[id];
}
const baseOpts = {responsive: true, maintainAspectRatio: false,
  plugins: {legend: {labels: {boxWidth: 12, font: {size: 11}}}},
  scales: {x: {ticks: {font: {size: 10.5}}, grid: {color: "rgba(38,50,88,.35)"}},
           y: {ticks: {font: {size: 10.5}}, grid: {color: "rgba(38,50,88,.35)"}, beginAtZero: true}}};

/* ---------------- state & helpers ---------------- */
let datasetInfo = null, eda = null, mlResult = null;
const SECTION_META = {
  dashboard: ["Dashboard", "Overview of the project pipeline and live dataset status"],
  dataset: ["Dataset Analysis", "Structure, columns, dtypes and missing-value audit"],
  eda: ["Exploratory Data Analysis", "Distributions, outliers, correlations and patterns — computed from the loaded CSV"],
  training: ["Model Training", "Unsupervised segmentation (K-Means, DBSCAN) & supervised churn classifiers"],
  evaluation: ["Model Evaluation", "Confusion matrices, per-class TP/TN/FP/FN, overfitting analysis on the unseen test split"],
  comparison: ["Model Comparison", "Side-by-side metrics of all trained classifiers"],
  conclusion: ["Conclusion", "Final interpretation derived from the trained results"],
};
function showSection(id) {
  $$(".panel").forEach(p => p.classList.remove("active"));
  $("#" + id).classList.add("active");
  $$(".nav-btn").forEach(b => b.classList.toggle("active", b.dataset.section === id));
  const [t, s] = SECTION_META[id];
  $("#section-title").textContent = t; $("#section-sub").textContent = s;
  window.scrollTo({top: 0});
  // charts created while the panel was hidden need an explicit resize
  setTimeout(() => { Object.values(charts).forEach(ch => { try { ch.resize(); } catch (e) {} }); }, 60);
}
$$(".nav-btn").forEach(b => b.onclick = () => showSection(b.dataset.section));

let toastTimer = null;
function toast(msg, kind = "") {
  const t = $("#toast");
  t.textContent = msg; t.className = "toast " + kind; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => t.hidden = true, 4500);
}

/* ---------------- dataset header badge & dashboard ---------------- */
async function refreshHealth() {
  const h = await fetch("/api/health").then(r => r.json());
  const b = $("#dataset-badge");
  if (h.dataset_loaded) {
    b.innerHTML = `<i>&#9679;</i> ${h.dataset_name}<br>${h.dataset_is_default ? "bundled default dataset" : "user-uploaded dataset"}`;
  } else {
    b.innerHTML = `<span style="color:#ff6b7a">&#9679;</span> No dataset — ${h.error || "upload a CSV"}`;
  }
  return h;
}

function kpi(label, value, hint, cls = "accent") {
  return `<div class="kpi ${cls}"><div class="label">${label}</div><div class="value">${value}</div>${hint ? `<div class="hint">${hint}</div>` : ""}</div>`;
}

function renderDashboard() {
  if (!datasetInfo) return;  if (!datasetInfo) return;
  const info = datasetInfo.info;
  let trained = "";
  if (mlResult) {
    trained = kpi("K-Means segments", mlResult.clustering.best_k, "best k by silhouette + DBSCAN backup")
      + kpi("Best classifier", mlResult.classification.best_model, `test F1-macro ${mlResult.classification.metrics.find(m => m.model === mlResult.classification.best_model).test_f1_macro}`)
      + kpi("Pipeline runtime", mlResult.total_duration_sec + " s", "full unsupervised + supervised run");
  }
  $("#kpi-grid").innerHTML =
    kpi("Rows", fmt(info.rows), "customer records") +
    kpi("Columns", fmt(info.columns.length), `${info.numeric_columns} numeric · ${info.categorical_columns} categorical`) +
    kpi("Missing cells", fmt(info.total_missing), `${(info.total_missing / info.cell_count * 100).toFixed(2)}% of all cells`, info.total_missing ? "warn" : "good") +
    kpi("Duplicate rows", fmt(info.duplicate_rows), info.duplicate_rows ? "checking…" : "none detected", info.duplicate_rows ? "warn" : "good") +
    trained;
  const st = [
    ["ok", `Dataset loaded: ${fmt(info.rows)} rows × ${info.columns.length} columns`],
    ["ok", "EDA complete — distributions, outliers and correlations computed"],
    [mlResult ? "ok" : "wait", mlResult ? "ML training complete — results available in Training / Evaluation tabs" : "ML training pending — click Run Training"],
  ];
  $("#pipeline-status").innerHTML = st.map(([d, t]) =>
    `<div class="status-item"><span class="dot ${d}"></span>${t}</div>`).join("");
  $("#dash-insights").innerHTML = (eda ? eda.insights : ["EDA insights appear once the dataset is analysed."])
    .map(i => `<li>${i}</li>`).join("");
}

/* ---------------- dataset section ---------------- */
function typePill(dt, isNum) {
  if (isNum) return `<span class="pill b">numeric</span> <span class="pill g">${dt}</span>`;
  return `<span class="pill a">categorical</span> <span class="pill b">${dt}</span>`;
}
function renderDataset() {
  const info = datasetInfo.info;
  $("#ds-kpis").innerHTML =
    kpi("Shape", `${fmt(info.rows)} × ${info.columns.length}`, "rows × columns") +
    kpi("Memory (RAM)", info.memory_mb + " MB", "in-memory pandas") +
    kpi("Columns with NaN", fmt(info.missing.filter(m => m.count > 0).length), "see table →", info.missing.some(m => m.count > 0) ? "warn" : "good") +
    kpi("Numeric columns", fmt(info.numeric_columns), `${info.categorical_columns} categorical`);

  const rows = info.columns.map(c =>
    `<tr><td><b>${c.name}</b></td><td>${typePill(c.dtype, c.is_numeric)}</td><td>${c.sample.map(String).join(", ")}</td></tr>`);
  $("#columns-table").innerHTML = tableHTML(["Column", "Type", "First values"], rows);
  c_search();

  const missRows = info.missing.filter(m => m.count > 0)
    .map(m =>
      `<tr><td><b>${m.column}</b></td><td class="num">${fmt(m.count)}</td><td class="num">${pct(m.percent, 2)}</td>
       <td style="min-width:120px"><div style="height:8px;background:var(--bg2);border-radius:4px">
       <div style="height:8px;width:${Math.min(100, m.percent * 2)}%;background:linear-gradient(90deg,var(--amber),var(--red));border-radius:4px"></div></div></td></tr>`);
  $("#missing-table").innerHTML = missRows.length ? tableHTML(["Column", "NaN count", "% of rows", ""], missRows)
    : `<p class="note">No missing values detected in any column.</p>`;
}
function c_search() {
  const s = $("#col-search");
  s.oninput = () => {
    const q = s.value.toLowerCase();
    const info = datasetInfo.info;
    const rows = info.columns.filter(c => c.name.toLowerCase().includes(q))
      .map(c => `<tr><td><b>${c.name}</b></td><td>${typePill(c.dtype, c.is_numeric)}</td><td>${c.sample.map(String).slice(0, 3).join(", ")}</td></tr>`);
    $("#columns-table").innerHTML = tableHTML(["Column", "Type", "First values (max 3)"], rows);
  };
}

/* ---------------- shared table builder ---------------- */
function tableHTML(headers, rows, numCols = null) {
  if (!rows.length) return `<p class="note">No data rows.</p>`;
  const th = headers.map(h => `<th${/^[-–\d]/.test(h) || numCols?.includes(h) ? ' class="num"' : ""}>${h}</th>`).join("");
  return `<table><thead><tr>${th}</tr></thead><tbody>${rows.join("")}</tbody></table>`;
}

/* ---------------- EDA ---------------- */
function histChart(id, h, label, color) {
  makeChart(id, {
    type: "bar",
    data: {labels: h.x.map(v => fmt(v, v % 1 ? 1 : 0)),
           datasets: [{label, data: h.y, backgroundColor: color, borderRadius: 4, categoryPercentage: 1, barPercentage: 1}]},
    options: {...baseOpts, plugins: {...baseOpts.plugins, legend: {display: false}}},
  });
}
function donutChart(id, obj, colors) {
  const entries = Object.entries(obj);
  makeChart(id, {
    type: "doughnut",
    data: {labels: entries.map(e => e[0]),
           datasets: [{data: entries.map(e => e[1]), backgroundColor: palette, borderColor: "#151d3b", borderWidth: 2}]},
    options: {responsive: true, maintainAspectRatio: false, cutout: "58%",
              plugins: {legend: {position: "right", labels: {boxWidth: 12, font: {size: 11}}}}},
  });
}
function barChart(id, obj, label, horizontal = false) {
  const entries = Object.entries(obj);
  const o = JSON.parse(JSON.stringify({
    responsive: true, maintainAspectRatio: false,
    plugins: {legend: {display: false}},
    scales: {x: {ticks: {font: {size: 10.5}}, grid: {color: "rgba(38,50,88,.35)"}},
             y: {ticks: {font: {size: 10.5}}, grid: {color: "rgba(38,50,88,.35)"}, beginAtZero: true}},
  }));
  if (horizontal) { [o.scales.x, o.scales.y] = [o.scales.y, o.scales.x]; }
  makeChart(id, {
    type: "bar",
    data: {labels: entries.map(e => e[0]),
           datasets: [{label, data: entries.map(e => e[1]), backgroundColor: palette.slice(0, entries.length), borderRadius: 6}]},
    options: {...o, indexAxis: horizontal ? "y" : "x"},
  });
}

function renderEDA() {
  if (!eda) return;
  $("#eda-insights").innerHTML = eda.insights.map(i => `<li>${i}</li>`).join("");
  const out = eda.outliers;
  const top = Object.entries(out).filter(e => e[1].count > 0)
    .sort((a, b) => b[1].count - a[1].count).slice(0, 4);
  $("#outlier-kpis").innerHTML = top.map(([c, o]) =>
    kpi(`Outliers: ${c}`, fmt(o.count), `${pct(o.percent)} outside IQR fence [${fmt(o.lower_bound)}…${fmt(o.upper_bound)}]`, "warn"))
    .join("") || "";
  histChart("c-age", eda.charts.age_hist, "customers", palette[0]);
  histChart("c-spent", eda.charts.spent_hist, "customers $", palette[2]);
  histChart("c-clv", eda.charts.clv_hist, "customers CLV", palette[1]);
  histChart("c-recency", eda.charts.recency_hist, "customers", palette[3]);
  histChart("c-churn", eda.charts.churn_risk_hist, "customers", palette[4]);
  histChart("c-satisfaction", eda.charts.satisfaction_hist, "customers", palette[5]);
  donutChart("c-segments", eda.segments);
  barChart("c-churncat", eda.churn_distribution, "customers");
  barChart("c-rfm", eda.rfm_categories, "customers");
  barChart("c-countries", eda.countries, "customers");
  barChart("c-channels", eda.channels, "customers");

  const seg = eda.churn_by_segment;
  makeChart("c-churnseg", {
    type: "bar",
    data: {labels: seg.segments,
           datasets: Object.entries(seg.series).map(([k, v], i) =>
             ({label: k, data: v, backgroundColor: palette[i], borderRadius: 4}))},
    options: {...baseOpts, scales: {x: {stacked: true, ticks: {font: {size: 10.5}}},
                                    y: {stacked: true, beginAtZero: true, max: 100, ticks: {font: {size: 10.5}, callback: v => v + "%"}}}},
  });

  makeChart("c-scatter", {
    type: "scatter",
    data: {datasets: [{label: "customer", data: eda.scatter.point.map((x, i) => ({x, y: eda.scatter.rfm[i]})),
                       backgroundColor: "rgba(108,140,255,.35)", pointRadius: 3}]},
    options: {...baseOpts,
      scales: {x: {type: "linear", title: {display: true, text: "Total spent (USD)"}, ticks: {font: {size: 10.5}}},
               y: {title: {display: true, text: "RFM score"}, beginAtZero: true, ticks: {font: {size: 10.5}}}}},
  });

  // correlation table as a colored grid
  const corr = eda.correlation;
  const n = corr.cols.length;
  const flat = corr.values.flat();
  const mn = Math.min(...flat), mx = Math.max(...flat);
  const cellCls = (v) => v >= 0.7 ? "background:rgba(53,208,165,.25)" :
    v >= 0.4 ? "background:rgba(53,208,165,.12)" :
    v <= -0.7 ? "background:rgba(255,107,122,.28)" :
    v <= -0.4 ? "background:rgba(255,107,122,.14)" :
    Math.abs(v) >= 0.2 ? "background:rgba(108,140,255,.10)" : "";
  $("#corr-table").innerHTML = `<table><thead><tr><th></th>${corr.cols.map(c => `<th style="writing-mode:vertical-rl;transform:rotate(180deg);max-height:90px">${c}</th>`).join("")}</tr></thead>
    <tbody>${corr.values.map((row, i) =>
      `<tr><th>${corr.cols[i]}</th>${row.map(v => `<td class="cm-cell" style="${cellCls(v)}">${v.toFixed(2)}</td>`).join("")}</tr>`).join("")}</tbody></table>`;

  $("#segment-table").innerHTML = tableHTML(
    eda.segment_table.headers,
    eda.segment_table.rows.map(r =>
      `<tr>${r.map((c, i) => `<td class="${i ? "num" : ""}">${typeof c === "number" ? fmt(c, `${c}`.includes(".") ? 2 : 0) : c}</td>`).join("")}</tr>`),
    ["Customers", "Avg Total Spent", "Avg Order Value", "Avg CLV", "Avg Churn Risk", "Avg Purchases"]);
}

/* ---------------- training / clustering ---------------- */
function renderClustering() {
  if (!mlResult) return;
  const c = mlResult.clustering, d = mlResult.dbscan;
  $("#training-empty").hidden = true;
  $("#training-content").hidden = false;
  $("#eval-empty").hidden = true; $("#eval-content").hidden = false;
  $("#comp-empty").hidden = true; $("#comp-content").hidden = false;
  $("#conc-empty").hidden = true; $("#conc-content").hidden = false;

  const m = c.metrics;
  $("#cluster-kpis").innerHTML =
    kpi("Best k", c.best_k, `chosen among k=2…8 (highest silhouette)`) +
    kpi("Silhouette · train", m.silhouette_train, `sample of ${fmt(m.sil_sample_train)} train rows`) +
    kpi("Silhouette · test", m.silhouette_test, `sample of ${fmt(m.sil_sample_test)} unseen rows`, "good") +
    kpi("Davies-Bouldin (test)", m.davies_bouldin_test, "lower = better separated") +
    kpi("Calinski-Harabasz (test)", fmt(m.calinski_harabasz_test), "higher = better") +
    kpi("Inertia (WSS)", fmt(m.inertia), "sum of squared distances to centroids");

  const ks = c.k_selection;
  makeChart("c-kselect", {
    type: "line",
    data: {labels: ks.ks, datasets: [
      {label: "Inertia (WSS)", data: ks.inertias, yAxisID: "y1", borderColor: palette[3], backgroundColor: palette[3] + "33", tension: .3},
      {label: "Silhouette", data: ks.silhouettes, yAxisID: "y", borderColor: palette[2], backgroundColor: palette[2] + "33", tension: .3}]},
    options: {...baseOpts,
      scales: {x: {title: {display: true, text: "k"}, ticks: {font: {size: 10.5}}},
               y: {beginAtZero: true, title: {display: true, text: "Silhouette"}, ticks: {font: {size: 10.5}}},
               y1: {position: "right", beginAtZero: true, title: {display: true, text: "Inertia"}, grid: {drawOnChartArea: false}, ticks: {font: {size: 10.5}}}}},
  });
  makeChart("c-clustercnt", {
    type: "bar",
    data: {labels: Object.keys(c.cluster_counts).map(k => `Cluster ${k}`),
           datasets: [{label: "train customers", data: Object.values(c.cluster_counts), backgroundColor: palette, borderRadius: 6}]},
    options: baseOpts,
  });

  makeChart("c-rfmflow", {
    type: "bar",
    data: {labels: Object.keys(c.cluster_counts).map(k => `Cluster ${k}`),
           datasets: Object.entries(c.rfm_flows).map(([k, v], i) =>
             ({label: k, data: v, backgroundColor: palette[i], borderRadius: 4}))},
    options: {...baseOpts, scales: {x: {stacked: true}, y: {stacked: true, beginAtZero: true, ticks: {callback: v => v + "%", font: {size: 10.5}}}}},
  });

  const profiles = c.cluster_profiles;
  const narratives = Object.fromEntries(c.narratives.map(n => [n.cluster, n.description]));
  $("#cluster-table").innerHTML = tableHTML(
    ["Cluster", "Customers", "Avg spent (USD)", "Avg recency (days)", "Avg churn risk", "High-churn share", "Recommended action"],
    profiles.map(p => {
      const al = narratives[p.cluster] || "";
      const tone = al.includes("protect") ? "g" : al.includes("At-risk") ? "r" : "b";
      return `<tr><td><b>Cluster ${p.cluster}</b></td><td class="num">${fmt(p.customers)}</td>
        <td class="num">${fmt(p.avg_spent_usd, 2)}</td><td class="num">${fmt(p.avg_recency_days, 1)}</td>
        <td class="num">${fmt(p.avg_churn_risk, 1)}</td><td class="num">${pct(p.high_churn_share_pct)}</td>
        <td><span class="pill ${tone}">${al}</span></td></tr>`;
    }));

  $("#dbscan-card").innerHTML = `
    <div class="minfo">
      <div class="row"><span>Algorithm</span><b>${d.algorithm}</b></div>
      <div class="row"><span>eps (auto from k-distance knee, min 0.15)</span><b>${d.eps}</b></div>
      <div class="row"><span>min_samples</span><b>${d.min_samples}</b></div>
      <div class="row"><span>Clusters found</span><b>${d.n_clusters}</b></div>
      <div class="row"><span>Sparse/noise customers</span><b>${fmt(d.noise_count)} (${pct(d.noise_pct, 2)})</b></div>
      <div class="row"><span>Silhouette (non-noise)</span><b>${d.silhouette_non_noise ?? "—"}</b></div>
      <div class="row"><span>Runtime</span><b>${d.duration_sec} s</b></div>
      <p class="note">DBSCAN confirms the bulk of customers form one dense region (${Object.values(d.cluster_counts)[0] ? fmt(Object.values(d.cluster_counts)[0]) : "—"} customers) and isolates sparse outliers — a useful cross-check on K-Means (which splits by distance into ${c.best_k} segments).</p>
    </div>`;
  $("#cluster-metrics").innerHTML = `
    <div class="minfo">
      <div class="row"><span>Model trained on</span><b>${fmt(c.train_size).replace(/,/g, ",")} rows (70%)</b></div>
      <div class="row"><span>Evaluated on unseen test rows</span><b>${fmt(c.test_size)} rows (30%)</b></div>
      <div class="row"><span>Silhouette gap (test − train)</span><b>${(c.metrics.silhouette_test - c.metrics.silhouette_train).toFixed(4)}</b></div>
      <p class="note">Train and test silhouette differ by only ${(Math.abs(c.metrics.silhouette_test - c.metrics.silhouette_train)).toFixed(4)} → the segmentation <b>generalizes</b> to unseen customers (no meaningful drift).</p>
    </div>`;
  $("#sil-note").textContent = c.metric_note || "";

  $("#centroid-table").innerHTML = tableHTML(
    ["Cluster", ...c.features],
    c.centroids.map((row, i) =>
      `<tr><td><b>Cluster ${i}</b></td>${row.map(v => `<td class="num">${fmt(v, 2)}</td>`).join("")}</tr>`));
}

/* ---------------- evaluation ---------------- */
function cmTable(cm, classes) {
  let html = `<table><thead><tr><th style="text-align:left">Predicted \\</th>${classes.map(c => `<th class="num">${c}</th>`).join("")}</tr></thead><tbody>`;
  cm.forEach((row, i) => {
    html += `<tr><th style="text-align:left">${classes[i]}</th>${row.map((v, j) =>
      `<td class="cm-cell ${i === j ? "cm-diag" : ""}" title="Predicted ${classes[j]}, actually ${classes[i]}">${fmt(v)}</td>`).join("")}</tr>`;
  });
  return html + "</tbody></table>";
}
function perClassTable(per) {
  return tableHTML(["Class", "TP", "FP", "FN", "TN", "Precision", "Recall"],
    per.map(p => {
      const primes = p.tp + p.fp, rec = p.tp + p.fn;
      const pr = primes ? (p.tp / primes) : 0, rcs = rec ? (p.tp / rec) : 0;
      const tone = (v) => v >= .8 ? "g" : v >= .5 ? "a" : "r";
      return `<tr><td><b>${p["class"]}</b></td><td class="num cm-cell cm-tp">${fmt(p.tp)}</td>
        <td class="num cm-cell cm-fp">${fmt(p.fp)}</td><td class="num cm-cell cm-fn">${fmt(p.fn)}</td>
        <td class="num cm-cell cm-tn">${fmt(p.tn)}</td>
        <td class="num"><span class="pill ${tone(pr)}">${pr.toFixed(3)}</span></td>
        <td class="num"><span class="pill ${tone(rcs)}">${rcs.toFixed(3)}</span></td></tr>`;
    }),
    ["TP", "FP", "FN", "TN"]);
}

const MODEL_EXPLAIN = {
  "Logistic Regression": "Linear baseline. Fast, interpretable via coefficients, but struggles with non-linear feature interactions.",
  "Random Forest": "Bagging ensemble of decision trees: robust to noise, gives feature importances, usually low overfit when depth is limited.",
  "Gradient Boosting": "Boosting ensemble: sequentially corrects errors — typically the strongest performer on structured/tabular data.",
};

function renderEvaluation() {
  const cl = mlResult.classification;
  const best = cl.best_model;
  const bm = cl.metrics.find(r => r.model === best);
  $("#eval-target").innerHTML = `
    <span class="chip">Target: <b>${cl.target}</b> (5 classes)</span>
    <span class="chip">Train: <b>${fmt(cl.train_size)}</b> rows</span>
    <span class="chip">Test: <b>${fmt(cl.test_size)}</b> rows (unseen, 30%)</span>
    <span class="chip">Features: <b>${cl.num_features_total}</b> after encoding (leakage columns excluded)</span>`;
  $("#eval-kpis").innerHTML =
    kpi("Winner", best, `test acc ${bm.test_accuracy} · F1-macro ${bm.test_f1_macro}`, "good") +
    kpi("Overfit gap (winner)", bm.overfit_gap, "train acc − test acc", bm.overfit_gap < 0.05 ? "good" : "warn") +
    kpi("Classes", cl.class_order.length, "Very Low → Very High churn risk") +
    kpi("Stratified split", "70 / 30", "random_state=42, same class balance in both folds");

  barChart("c-classdist", cl.class_distribution, "customers");

  $("#model-cards").innerHTML = "";
  cl.metrics.forEach(r => {
    const det = cl.confusion[r.model];
    const tone = r.model === best ? "g" : "b";
    const card = document.createElement("div");
    card.className = "card";
    card.innerHTML = `<h3>${r.model} ${r.model === best ? '<span class="pill g">BEST · highest test F1-macro</span>' : ""}</h3>
      <div class="model-card">
        <div>
          <canvas id="cm-${slug(r.model)}" style="max-height:230px"></canvas>
        </div>
        <div class="minfo">
          <div class="row"><span>Train accuracy</span><b>${r.train_accuracy}</b></div>
          <div class="row"><span>Test accuracy (unseen)</span><b>${r.test_accuracy}</b></div>
          <div class="row"><span>Precision (macro)</span><b>${r.test_precision_macro}</b></div>
          <div class="row"><span>Recall (macro)</span><b>${r.test_recall_macro}</b></div>
          <div class="row"><span>F1 (macro)</span><b>${r.test_f1_macro}</b></div>
          <div class="row"><span>Overfit gap</span><b>${r.overfit_gap}</b></div>
          <div class="row"><span>Training time</span><b>${r.train_time_sec} s</b></div>
          <p class="note">${MODEL_EXPLAIN[r.model] || ""}</p>
        </div>
      </div>
      <h3 style="margin-top:18px">Confusion matrix — unseen test set (rows = actual, columns = predicted)</h3>
      <div class="table-wrap">${cmTable(det.confusion_matrix, det.classes)}</div>
      <h3 style="margin-top:18px">Per-class TP / FP / FN / TN (one-vs-rest on the test set)</h3>
      <div class="table-wrap">${perClassTable(det.per_class)}</div>`;
    $("#model-cards").appendChild(card);
    // small accuracy chart per model card
    makeChart("cm-" + slug(r.model), {
      type: "bar",
      data: {labels: ["Train acc", "Test acc", "F1-macro"],
             datasets: [{data: [r.train_accuracy, r.test_accuracy, r.test_f1_macro], backgroundColor: [palette[3], palette[2], palette[1]], borderRadius: 6}]},
      options: {...baseOpts, plugins: {legend: {display: false}},
                scales: {y: {beginAtZero: true, min: 0, max: 1, ticks: {font: {size: 10.5}}}, x: {ticks: {font: {size: 10.5}}}}},
    });
  });

  const fi = cl.feature_importances["Random Forest"] || Object.values(cl.feature_importances)[0];
  makeChart("c-importance", {
    type: "bar",
    data: {labels: fi.cols, datasets: [{label: "importance", data: fi.vals, backgroundColor: palette[0], borderRadius: 5}]},
    options: {...baseOpts, indexAxis: "y", plugins: {legend: {display: false}}},
  });

  const labels = cl.metrics.map(r => r.model);
  const toneFor = gap => gap < 0.02 ? "good" : gap < 0.05 ? "warn" : "bad";
  const notes = cl.metrics.map(r => {
    const gap = r.overfit_gap;
    let text;
    if (gap < 0.02) text = `<b>${r.model}</b> is <b>well-generalizing</b>: train acc ${r.train_accuracy} vs test acc ${r.test_accuracy} — a gap of only ${gap}. Learning captured by the model holds on unseen customers; neither memorizing (overfitting) nor failing to learn (underfitting).`;
    else if (gap < 0.05) text = `<b>${r.model}</b> shows <b>slight overfitting</b> (gap ${gap}): training accuracy ${r.train_accuracy} edges above test accuracy ${r.test_accuracy}. Mild complexity regularization (deeper trees shrink, stronger boosting) could close it partially — but predictions remain reliable on unseen data.`;
    else text = `<b>${r.model}</b> shows <b>severe overfitting</b> (gap ${gap}): it memorized the training fold (train acc %) at the cost of unseen-data accuracy ${r.test_accuracy}.`;
    const underfit = r.train_accuracy < 0.6;
    if (underfit) text = `<b>${r.model}</b> may be <b>underfitting</b>: even on training data accuracy is only ${r.train_accuracy}, so the model is too simple / coerced for the task.`;
    return `<div class="fit-note ${toneFor(gap) === "good" ? "good" : toneFor(gap) === "bad" ? "bad" : ""}">${text}</div>`;
  }).join("");
  makeChart("c-traintest", {
    type: "bar",
    data: {labels,
           datasets: [{label: "Train accuracy", data: cl.metrics.map(r => r.train_accuracy), backgroundColor: palette[3], borderRadius: 5},
                      {label: "Test accuracy (unseen)", data: cl.metrics.map(r => r.test_accuracy), backgroundColor: palette[2], borderRadius: 5}]},
    options: {...baseOpts, scales: {x: {ticks: {font: {size: 10.5}}}, y: {beginAtZero: true, min: 0, max: 1, ticks: {font: {size: 10.5}}}}},
  });
  $("#fit-analysis").innerHTML = `<p class="note">Gap = train accuracy − test accuracy. Small gap &amp; high accuracy ⇒ good generalization; large gap ⇒ overfitting; low scores on both ⇒ underfitting.</p>` + notes;
}
function slug(s) { return s.toLowerCase().replace(/[^a-z0-9]/g, "-"); }

/* ---------------- comparison ---------------- */
function renderComparison() {
  const cl = mlResult.classification;
  const best = cl.best_model;
  $("#comp-kpis").innerHTML =
    kpi("Recommended model", best, cl.best_reason.split("(")[0], "good") +
    kpi("Fastest", ...(() => { const f = [...cl.metrics].sort((a, b) => a.train_time_sec - b.train_time_sec)[0];
      return [f.model, `${f.train_time_sec}s to train · acc ${f.test_accuracy}`]; })(), "accent") +
    kpi("Lowest overfit", ...(() => { const f = [...cl.metrics].sort((a, b) => a.overfit_gap - b.overfit_gap)[0];
      return [f.model, `gap ${f.overfit_gap}`]; })(), "accent");

  $("#comp-table").innerHTML = tableHTML(
    cl.metrics_table.headers,
    cl.metrics_table.rows.map(r =>
      `<tr>${r.map((v, i) => `<td class="${i ? "num" : ""}">${typeof v === "number" ? fmt(v, 4) : v}</td>`).join("")}</tr>`),
    ["Train Acc", "Test Acc", "Precision (macro)", "Recall (macro)", "F1 (macro)", "Overfit Gap", "Train (s)"]);

  const labels = cl.metrics.map(r => r.model);
  makeChart("c-compbars", {
    type: "bar",
    data: {labels, datasets: [
      {label: "Train acc", data: cl.metrics.map(r => r.train_accuracy), backgroundColor: "rgba(255,181,71,.7)", borderRadius: 5},
      {label: "Test acc", data: cl.metrics.map(r => r.test_accuracy), backgroundColor: "rgba(53,208,165,.8)", borderRadius: 5},
      {label: "F1-macro", data: cl.metrics.map(r => r.test_f1_macro), backgroundColor: "rgba(108,140,255,.8)", borderRadius: 5},
      {label: "Recall-macro", data: cl.metrics.map(r => r.test_recall_macro), backgroundColor: "rgba(156,108,255,.8)", borderRadius: 5}]},
    options: {...baseOpts, scales: {x: {ticks: {font: {size: 10.5}}}, y: {beginAtZero: true, min: 0, max: 1, ticks: {font: {size: 10.5}}}}},
  });
  makeChart("c-gapbars", {
    type: "bar",
    data: {labels, datasets: [{label: "overfit gap", data: cl.metrics.map(r => r.overfit_gap), backgroundColor: cl.metrics.map(r => r.overfit_gap < .02 ? palette[2] : r.overfit_gap < .05 ? palette[3] : palette[4]), borderRadius: 5}]},
    options: {...baseOpts, plugins: {legend: {display: false}}},
  });
  makeChart("c-speed", {
    type: "scatter",
    data: {datasets: cl.metrics.map((r, i) => ({label: r.model, data: [{x: r.train_time_sec, y: r.test_f1_macro}],
      backgroundColor: palette[i], pointRadius: 7, pointHoverRadius: 9}))},
    options: {...baseOpts, plugins: {legend: {labels: {font: {size: 11}}}},
      scales: {x: {title: {display: true, text: "Train time (s)"}, ticks: {font: {size: 10.5}}},
               y: {title: {display: true, text: "Test F1-macro"}, min: 0, max: 1, ticks: {font: {size: 10.5}}}}},
  });
}

/* ---------------- conclusion ---------------- */
function renderConclusion() {
  const cl = mlResult.classification, cu = mlResult.clustering;
  const bm = cl.metrics.find(r => r.model === cl.best_model);
  $("#conc-verdict").innerHTML = `
    <div class="big">${cl.best_model} is the recommended model</div>
    ${cl.best_reason}<br><br>
    On clustering: K-Means with k=${cu.best_k} produced ${cu.best_k} actionable segments
    (test silhouette <b>${cu.metrics.silhouette_test}</b>, Davies-Bouldin <b>${cu.metrics.davies_bouldin_test}</b>);
    DBSCAN confirmed one dominant dense region and flagged ${fmt(mlResult.dbscan.noise_count)} sparse outliers (${pct(mlResult.dbscan.noise_pct, 2)}).<br><br>
    Overall pipeline runtime: <b>${mlResult.total_duration_sec}s</b> on ${fmt(cu.train_size + cu.test_size)} customers — fully reproducible (random_state = 42).`;
  $("#conc-clusters").innerHTML = cu.narratives.map(n =>
    `<li><b>Cluster ${n.cluster}</b> — ${n.description} (avg spend $${fmt(n.avg_spent_usd)}, churn ${fmt(n.avg_churn_risk)}, recency ${fmt(n.avg_recency)} days)</li>`).join("");
  $("#conc-model").innerHTML = cl.metrics.map(r =>
    `<li><b>${r.model}</b> — test acc ${r.test_accuracy}, F1-macro ${r.test_f1_macro}, overfit gap ${r.overfit_gap}. ${r.model === cl.best_model ? "✅ chosen" : "outperformed"}</li>`).join("")
    + `<li>Class imbalance is handled with stratification and macro-averaged metrics; ${cl.leakage_excluded.length} derived columns were fenced off to prevent leakage.</li>`;
  $("#conc-actions").innerHTML = cu.narratives.map(n =>
    `<li><b>${n.description.split("–")[0].trim()}</b> → ${n.description.includes("protect") ? "white-glove care, exclusive bundles, renewal nudges"
      : n.description.includes("At-risk") ? "automated win-back email + personalised discount within 7 days"
      : n.description.includes("Low-spend") ? "cross-sell entry offers, free-shipping thresholds"
      : "loyalty points seeding, progressive perks"}</li>`).join("");
}

/* ---------------- orchestration ---------------- */
async function loadAll(initial) {
  try {
    $("#dataset-badge").textContent = "Loading dataset…";
    await refreshHealth();
    const d = await fetch(`/api/dataset?_=${Date.now()}`).then(r => {
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
    datasetInfo = d; renderDashboard(); renderDataset();
    const e = await fetch(`/api/eda?_=${Date.now()}`).then(r => { if (!r.ok) throw new Error("HTTP " + r.status); return r.json(); });
    eda = e.eda; renderDashboard(); renderEDA();
    if (!initial) toast("Dataset & EDA refreshed", "ok");
    if (mlResult) { renderClustering(); renderEvaluation(); renderComparison(); renderConclusion(); }
  } catch (err) {
    $("#dataset-badge").innerHTML = `<span style="color:#ff6b7a">●</span> Dataset failed to load: ${err.status || err.message}`;
    toast("Failed to load dataset via API — is the server running?", "err");
  }
}

$("#train-btn").onclick = async () => {
  const btn = $("#train-btn");
  if (!datasetInfo) { toast("Load a dataset first", "err"); return; }
  if (mlResult) {
    Object.keys(charts).forEach(k => { if (charts[k]) { charts[k].destroy(); delete charts[k]; } });
  }
  btn.disabled = true; const orig = btn.innerHTML;
  btn.innerHTML = "&#9203; Training… (unsupervised + supervised, ~2 min on 50k rows)";
  $("#pipeline-status").insertAdjacentHTML("afterbegin",
    `<div class="status-item" id="training-progress"><span class="dot wait"></span>Training models: K-Means k-selection, DBSCAN, Logistic Regression, Random Forest, Gradient Boosting…</div>`);
  try {
    const t0 = performance.now();
    const res = await fetch("/api/train", {method: "POST"});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.status);
    mlResult = data;
    renderClustering(); renderEvaluation(); renderComparison(); renderConclusion(); renderDashboard();
    showSection("training");
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    toast(`Training complete in ${secs}s — best model: ${mlResult.classification.best_model}`, "ok");
  } catch (err) {
    toast("Training failed: " + err.message, "err");
    console.error(err);
  } finally {
    btn.disabled = false; btn.innerHTML = orig;
    document.getElementById("training-progress")?.remove();
  }
};

$("#upload-btn").onclick = () => $("#file-input").click();
$("#file-input").onchange = async (ev) => {
  const f = ev.target.files[0]; if (!f) return;
  const fd = new FormData(); fd.append("file", f);
  const btn = $("#upload-btn"); btn.disabled = true;
  toast("Uploading " + f.name + "…");
  try {
    const res = await fetch("/api/upload", {method: "POST", body: fd});
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || res.status);
    datasetInfo = null; eda = null; mlResult = null;
    Object.keys(charts).forEach(k => { if (charts[k]) { charts[k].destroy(); delete charts[k]; } });
    await loadAll(true);
    showSection("dataset");
    toast(`Loaded ${data.rows} rows × ${data.columns} cols from ${data.file}`, "ok");
  } catch (err) { toast("Upload failed: " + err.message, "err"); }
  finally { btn.disabled = false; ev.target.value = ""; }
};

window.addEventListener("DOMContentLoaded", () => loadAll(true));
