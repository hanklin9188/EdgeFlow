const $ = (selector) => document.querySelector(selector);
const root = document.documentElement;
const state = {
  token: "",
  models: [],
  capabilities: [],
  services: [],
  experiments: {},
  runs: [],
  jobs: [],
  policies: [],
  evidence: [],
};

const escapeHtml = (value) => String(value ?? "").replace(/[&<>'"]/g, (character) => ({
  "&": "&amp;", "<": "&lt;", ">": "&gt;", "'": "&#39;", '"': "&quot;",
})[character]);

function badge(value) {
  const text = String(value || "PENDING");
  const key = text.toLowerCase();
  const className = key.includes("pass") || key.includes("succeed") || key === "available"
    ? "pass"
    : key.includes("fail") || key.includes("invalid") || key.includes("cancel") || key.includes("interrupt")
      ? "fail" : key.includes("run") || key.includes("queue") ? "warn" : "neutral";
  const icon = className === "pass" ? "✓" : className === "fail" ? "×" : className === "warn" ? "!" : "○";
  return `<span class="pill ${className}"><span aria-hidden="true">${icon}</span>${escapeHtml(text)}</span>`;
}

function formatBytes(bytes) {
  const value = Number(bytes || 0);
  if (value >= 1024 ** 3) return `${(value / 1024 ** 3).toFixed(2)} GiB`;
  if (value >= 1024 ** 2) return `${(value / 1024 ** 2).toFixed(1)} MiB`;
  if (value >= 1024) return `${(value / 1024).toFixed(1)} KiB`;
  return `${value} B`;
}

function median(values) {
  const sorted = values.filter((value) => Number.isFinite(value)).sort((a, b) => a - b);
  if (!sorted.length) return null;
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function formatMetric(value, unit) {
  return Number.isFinite(value) ? `${value.toFixed(value >= 100 ? 1 : 2)} ${unit}` : "Not measured";
}

function showToast(message) {
  const toast = $("#toast");
  toast.textContent = message;
  toast.hidden = false;
  window.setTimeout(() => { toast.hidden = true; }, 4200);
}

async function api(path, options = {}) {
  const headers = { Accept: "application/json", ...(options.headers || {}) };
  if (options.body) headers["Content-Type"] = "application/json";
  if (options.control) headers["X-EdgeFlow-Token"] = state.token;
  const response = await fetch(path, { ...options, headers, cache: "no-store" });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try { detail = (await response.json()).detail || detail; } catch (_) { /* response is not JSON */ }
    throw new Error(detail);
  }
  const contentType = response.headers.get("content-type") || "";
  return contentType.includes("json") ? response.json() : response.text();
}

function parsePromptDistribution(value) {
  const input = value.trim();
  if (!input.includes(":")) {
    const tokens = Number(input);
    if (!Number.isInteger(tokens) || tokens < 1 || tokens > 131072) throw new Error("Prompt tokens must be an integer from 1 to 131072.");
    return tokens;
  }
  const buckets = input.split(",").map((item) => {
    const [tokensText, probabilityText] = item.trim().split(":");
    const tokens = Number(tokensText);
    const probability = Number(probabilityText);
    if (!Number.isInteger(tokens) || tokens < 1 || tokens > 131072 || !(probability > 0 && probability <= 1)) {
      throw new Error("Use token:probability buckets such as 128:0.5,512:0.5.");
    }
    return { tokens, probability };
  });
  const total = buckets.reduce((sum, item) => sum + item.probability, 0);
  if (Math.abs(total - 1) > 0.000001) throw new Error(`Prompt probabilities sum to ${total}, not 1.0.`);
  return buckets;
}

function numberValue(selector) { return Number($(selector).value); }

function submissionPayload() {
  const backend = $("#backend").value;
  return {
    label: $("#label").value.trim(),
    model_id: $("#model").value,
    model_format: $("#modelFormat").value,
    backend,
    prompt_tokens: parsePromptDistribution($("#promptTokens").value),
    output_tokens: numberValue("#outputTokens"),
    batch_size: 1,
    concurrency: numberValue("#concurrency"),
    session_requests: numberValue("#sessionRequests"),
    quality_profile: $("#qualityProfile").value,
    seed: numberValue("#seed"),
    dtype: $("#dtype").value,
    compile_mode: $("#compileMode").value,
    dynamic_shapes: false,
    fullgraph: false,
    cuda_graph: false,
    quantization: backend === "llama_cpp" ? $("#quantization").value : null,
    external_base_url: $("#serverUrl").value,
    repetitions: numberValue("#repetitions"),
    warmup_requests: numberValue("#warmup"),
    experiment_id: $("#experiment").value,
    allow_download: $("#allowDownload").checked,
    allow_busy_gpu: $("#allowBusy").checked,
  };
}

function workloadFromSubmission(payload) {
  const promptLabel = Array.isArray(payload.prompt_tokens)
    ? `mix-${payload.prompt_tokens.map((item) => item.tokens).join("-")}` : payload.prompt_tokens;
  return {
    schema_version: "1.0",
    workload_id: `${payload.label}-p${promptLabel}-o${payload.output_tokens}-c${payload.concurrency}`,
    model_id: payload.model_id,
    prompt_source: { type: "synthetic", revision: "1.0", name: "edgeflow-corpus-v1", split: null, sample_ids_sha256: null },
    prompt_tokens: payload.prompt_tokens,
    output_tokens: payload.output_tokens,
    batch_size: 1,
    concurrency: payload.concurrency,
    arrival_pattern: "closed_loop",
    request_rate: null,
    sampling: { strategy: "greedy", temperature: 0, top_p: 1, top_k: null, ignore_eos: true },
    seed: payload.seed,
    streaming: true,
    session_requests: payload.session_requests,
    quality_profile: payload.quality_profile,
    notes: "Local-first UI screening workload.",
  };
}

function renderCapabilities() {
  const container = $("#runtimeList");
  if (!state.capabilities.length) {
    container.innerHTML = '<p class="placeholder">No runtime probe returned.</p>';
    return;
  }
  container.innerHTML = state.capabilities.map((item) => {
    const reason = item.available ? (item.version || "version not reported") : (item.reasons || []).join("; ");
    return `<div class="runtime-row"><b>${escapeHtml(item.backend)}</b><span class="runtime-status ${item.available ? "pass" : "fail"}">${item.available ? "✓ Available" : "! Unavailable"}</span><small>${escapeHtml(reason)}</small></div>`;
  }).join("");
}

function renderServices() {
  const container = $("#serviceList");
  if (!state.services.length) {
    container.innerHTML = '<p class="placeholder">No managed runtime definition returned.</p>';
    return;
  }
  const active = state.services.some((item) => ["STARTING", "RUNNING", "STOPPING"].includes(item.state));
  container.innerHTML = state.services.map((item) => {
    const selected = ["STARTING", "RUNNING", "STOPPING", "FAILED"].includes(item.state);
    const action = selected && item.state !== "FAILED"
      ? `<button class="row-button" data-service-stop="${escapeHtml(item.backend)}" type="button">Stop</button>`
      : `<button class="row-button" data-service-start="${escapeHtml(item.backend)}" type="button" ${!item.installed || active ? "disabled" : ""}>Start</button>`;
    const detail = item.installed ? item.message : "Install this pinned runtime first.";
    return `<div class="service-row"><div><b>${escapeHtml(item.backend)}</b><small>${escapeHtml(item.base_url)}</small></div>${badge(item.state)}<p>${escapeHtml(detail)}</p>${action}</div>`;
  }).join("");
}

function populateModels() {
  const select = $("#model");
  select.innerHTML = state.models.map((model) => `<option value="${escapeHtml(model.model_id)}">${escapeHtml(model.model_id)} · ${escapeHtml(model.role || "registered")}</option>`).join("");
  const preferred = state.models.find((item) => item.model_id === "smollm2-360m-instruct");
  if (preferred) select.value = preferred.model_id;
  syncRuntimeFields();
}

function syncRuntimeFields() {
  const backend = $("#backend").value;
  const model = state.models.find((item) => item.model_id === $("#model").value);
  const formats = model ? Object.keys(model.sources || {}).filter((key) => ["safetensors", "gguf"].includes(key)) : ["safetensors"];
  const requiredFormat = backend === "llama_cpp" ? "gguf" : "safetensors";
  $("#modelFormat").innerHTML = formats.map((format) => `<option value="${format}">${format === "gguf" ? "GGUF" : "safetensors"}</option>`).join("");
  if (formats.includes(requiredFormat)) $("#modelFormat").value = requiredFormat;
  $("#modelFormat").disabled = true;
  $("#compileField").hidden = backend !== "torch_compile";
  $("#serverField").hidden = !["llama_cpp", "vllm"].includes(backend);
  $("#quantizationField").hidden = backend !== "llama_cpp";
  $("#dtype").disabled = backend === "llama_cpp";
  if (backend === "llama_cpp") $("#serverUrl").value = "http://127.0.0.1:8001";
  if (backend === "vllm") $("#serverUrl").value = "http://127.0.0.1:8002";
  const experiment = { pytorch_eager: "E04", torch_compile: "E05", llama_cpp: "E07", vllm: "E08" }[backend];
  $("#experiment").value = experiment;
  const supported = model?.backends?.[backend] || "unsupported";
  const message = formats.includes(requiredFormat)
    ? `Registry support: ${supported}. Screening remains estimated until measured.`
    : `${model?.model_id || "Selected model"} has no registered ${requiredFormat} source for ${backend}.`;
  $("#formMessage").textContent = message;
  $("#formMessage").className = formats.includes(requiredFormat) ? "form-message" : "form-message fail";
  $("#queueButton").disabled = !formats.includes(requiredFormat);
}

function renderCandidates(result) {
  $("#candidateEmpty").hidden = true;
  $("#candidateSummary").hidden = false;
  $("#candidateCount").textContent = result.candidate_count;
  $("#prunedCount").textContent = result.pruned_count;
  $("#candidateList").innerHTML = result.candidates.slice(0, 40).map((plan) => `<div class="candidate"><b>${escapeHtml(plan.plan_id)}</b><small>${escapeHtml(plan.backend)} · ${escapeHtml(plan.dtype || plan.quantization || "native")} · unmeasured</small></div>`).join("") || '<p class="placeholder">Every candidate was pruned by capability or VRAM constraints.</p>';
  $("#prunedList").innerHTML = result.pruned.map((item) => `<div class="pruned-row"><b>${escapeHtml(item.plan_id)}</b><span>${escapeHtml(item.reason)}</span></div>`).join("") || '<p class="placeholder">No deterministic prunes.</p>';
}

function renderJobs() {
  $("#jobsEmpty").hidden = state.jobs.length > 0;
  $("#activeJobs").textContent = state.jobs.filter((item) => ["QUEUED", "RUNNING", "CANCELLING"].includes(item.status)).length;
  $("#jobList").innerHTML = state.jobs.map((job) => {
    const active = ["QUEUED", "RUNNING", "CANCELLING"].includes(job.status);
    const resultButton = job.result_available
      ? job.result?.run_id
        ? `<button class="row-button" data-run="${escapeHtml(job.result.run_id)}" type="button">Evidence</button>`
        : `<button class="row-button" data-job="${escapeHtml(job.job_id)}" type="button">Failure</button>`
      : "";
    const cancelButton = active ? `<button class="row-button" data-cancel="${escapeHtml(job.job_id)}" type="button">Cancel</button>` : resultButton;
    return `<article class="job-row"><div><code>${escapeHtml(job.job_id)}</code><p>${escapeHtml(job.message)}</p></div><div><b>${escapeHtml(job.model_id)}</b><p>${escapeHtml(job.backend)} · ${escapeHtml(job.experiment_id)}</p></div><div>${badge(job.status)}</div><time>${escapeHtml((job.started_at || job.created_at || "").replace("T", " ").slice(0, 19))}</time>${cancelButton}</article>`;
  }).join("");
}

function filteredRuns() {
  const query = $("#search").value.trim().toLowerCase();
  const eligibleOnly = $("#eligibleOnly").checked;
  return state.runs.filter((row) => {
    if (eligibleOnly && !row.validation?.policy_eligible) return false;
    return !query || JSON.stringify(row).toLowerCase().includes(query);
  });
}

function renderRuns() {
  const rows = filteredRuns();
  $("#runsEmpty").hidden = rows.length > 0;
  $("#runTable").hidden = rows.length === 0;
  $("#runRows").innerHTML = rows.map(({ manifest, validation }) => {
    const verdict = validation?.verdict || manifest.status;
    return `<tr><td><code>${escapeHtml(manifest.run_id)}</code><small>${escapeHtml((manifest.created_at || "").replace("T", " ").slice(0, 19))}</small></td><td><b>${escapeHtml(manifest.model_id)}</b><small>${escapeHtml(manifest.plan_id)}</small></td><td>${escapeHtml(manifest.experiment_id)}<small>${escapeHtml(manifest.protocol_version)}</small></td><td><span class="source-label">${escapeHtml(manifest.source_type)}</span></td><td>${badge(verdict)}</td><td>${validation?.policy_eligible ? "✓ Eligible" : "Excluded"}</td><td><button class="row-button" type="button" data-run="${escapeHtml(manifest.run_id)}">Open</button></td></tr>`;
  }).join("");
}

function renderPolicies() {
  $("#policyCount").textContent = state.policies.length;
  $("#policyBadge").textContent = state.policies.length;
  $("#policyEmpty").hidden = state.policies.length > 0;
  $("#policyList").innerHTML = state.policies.map((policy) => `<button class="policy row-button" type="button" data-policy="${escapeHtml(policy.policy_id)}"><span><b>${escapeHtml(policy.policy_id)}</b><small>${escapeHtml(policy.model_id)} · fallback ${escapeHtml(policy.fallback_plan_id)}</small></span>${badge(policy.status)}</button>`).join("");
}

function renderEvidence() {
  $("#evidenceCount").textContent = state.evidence.length;
  $("#evidenceEmpty").hidden = state.evidence.length > 0;
  $("#evidenceList").innerHTML = state.evidence.map((item) => `<button class="evidence-card row-button" type="button" data-evidence="${escapeHtml(item.evidence_id)}"><span><b>${escapeHtml(item.evidence_id)}</b><small>${escapeHtml(item.evidence_level)} · ${escapeHtml(item.status)}</small></span>${badge(item.status)}</button>`).join("");
}

function keyValue(label, value) {
  return `<div class="key-value"><span>${escapeHtml(label)}</span><b>${escapeHtml(value ?? "Not recorded")}</b></div>`;
}

async function openRun(runId) {
  if (!runId) return;
  const drawer = $("#drawer");
  $("#drawerEyebrow").textContent = "Measured run evidence";
  $("#drawerTitle").textContent = runId;
  $("#drawerBody").innerHTML = '<p class="placeholder">Loading manifest, raw metrics and validation…</p>';
  drawer.showModal();
  try {
    const [run, metrics, artifacts] = await Promise.all([
      api(`/api/v1/runs/${encodeURIComponent(runId)}`),
      api(`/api/v1/runs/${encodeURIComponent(runId)}/metrics`),
      api(`/api/v1/runs/${encodeURIComponent(runId)}/artifacts`),
    ]);
    const measured = metrics.filter((row) => row.phase === "end_to_end");
    const latency = median(measured.map((row) => row.metrics?.request_latency_ms));
    const ttft = median(measured.map((row) => row.metrics?.ttft_ms));
    const tpot = median(measured.map((row) => row.metrics?.tpot_ms));
    const validation = run.validation;
    const issues = validation?.issues || [];
    $("#drawerBody").innerHTML = `
      <section class="drawer-section"><div>${badge(validation?.verdict || run.manifest.status)} ${run.manifest.source_type === "measured" ? '<span class="source-label">MEASURED</span>' : `<span class="source-label estimated">${escapeHtml(run.manifest.source_type)}</span>`}</div><p>Metrics below are raw-run summaries and do not imply recommendation eligibility.</p></section>
      <section class="drawer-section"><h3>Identity & scope</h3><div class="key-grid">${keyValue("Model", run.manifest.model_id)}${keyValue("Plan", run.manifest.plan_id)}${keyValue("Experiment", run.manifest.experiment_id)}${keyValue("Protocol", run.manifest.protocol_version)}${keyValue("Hardware hash", run.manifest.hardware_fingerprint_sha256)}${keyValue("Model revision", run.manifest.model_revision)}</div></section>
      <section class="drawer-section"><h3>Measured distribution</h3><div class="metric-strip"><div><strong>${escapeHtml(formatMetric(latency, "ms"))}</strong><span>MEDIAN REQUEST LATENCY · ${measured.length} ROWS</span></div><div><strong>${escapeHtml(formatMetric(ttft, "ms"))}</strong><span>MEDIAN TTFT</span></div><div><strong>${escapeHtml(formatMetric(tpot, "ms/token"))}</strong><span>MEDIAN TPOT</span></div></div></section>
      <section class="drawer-section"><h3>Validation gates</h3><div class="key-grid">${keyValue("Policy eligible", validation?.policy_eligible ? "Yes" : "No")}${keyValue("Public claim eligible", validation?.public_claim_eligible ? "Yes" : "No")}${keyValue("Quality", validation?.quality_pass === true ? "PASS" : validation?.quality_pass === false ? "FAIL" : "Not established")}${keyValue("Verdict", validation?.verdict || "Not validated")}</div>${issues.map((issue) => `<div class="issue"><b>${escapeHtml(issue.code)}</b> · ${escapeHtml(issue.message)}</div>`).join("") || '<p>No validation issue recorded.</p>'}</section>
      <section class="drawer-section"><h3>Allowlisted artifacts</h3><div class="artifact-list">${artifacts.map((item) => `<a href="${escapeHtml(item.href)}" target="_blank" rel="noopener"><span>${escapeHtml(item.name)}</span><small>${escapeHtml(formatBytes(item.bytes))}</small></a>`).join("") || '<p>No readable artifact found.</p>'}</div></section>`;
  } catch (error) {
    $("#drawerBody").innerHTML = `<div class="issue"><b>Unable to load run</b> · ${escapeHtml(error.message)}</div>`;
  }
}

async function openEvidence(evidenceId) {
  $("#drawerEyebrow").textContent = "Evidence graph node";
  $("#drawerTitle").textContent = evidenceId;
  $("#drawerBody").innerHTML = '<p class="placeholder">Loading evidence edges…</p>';
  $("#drawer").showModal();
  try {
    const result = await api(`/api/v1/evidence/${encodeURIComponent(evidenceId)}`);
    $("#drawerBody").innerHTML = `<section class="drawer-section"><div>${badge(result.evidence.status)}</div><div class="key-grid">${keyValue("Level", result.evidence.evidence_level)}${keyValue("Run", result.evidence.run_id)}${keyValue("Claim", result.evidence.claim || result.evidence.hypothesis)}</div></section><section class="drawer-section"><h3>Connected edges</h3>${result.edges.map((edge) => `<div class="issue"><code>${escapeHtml(edge.source_id)}</code> ${escapeHtml(edge.relation)} <code>${escapeHtml(edge.target_id)}</code></div>`).join("") || '<p>No graph edge recorded.</p>'}</section>`;
  } catch (error) { $("#drawerBody").innerHTML = `<div class="issue">${escapeHtml(error.message)}</div>`; }
}

async function openJob(jobId) {
  $("#drawerEyebrow").textContent = "Local worker record";
  $("#drawerTitle").textContent = jobId;
  $("#drawerBody").innerHTML = '<p class="placeholder">Loading job result…</p>';
  $("#drawer").showModal();
  try {
    const job = await api(`/api/v1/jobs/${encodeURIComponent(jobId)}`);
    $("#drawerBody").innerHTML = `<section class="drawer-section"><div>${badge(job.status)}</div><p>${escapeHtml(job.message)}</p><div class="key-grid">${keyValue("Model", job.model_id)}${keyValue("Backend", job.backend)}${keyValue("Experiment", job.experiment_id)}${keyValue("Started", job.started_at)}</div></section>${job.result?.error ? `<section class="drawer-section"><h3>Bounded failure</h3><div class="issue"><b>${escapeHtml(job.result.error_type)}</b> · ${escapeHtml(job.result.error)}</div><p>Failure output is not eligible for policy or public claims.</p></section>` : ""}`;
  } catch (error) { $("#drawerBody").innerHTML = `<div class="issue">${escapeHtml(error.message)}</div>`; }
}

function openPolicy(policyId) {
  const policy = state.policies.find((item) => item.policy_id === policyId);
  if (!policy) return;
  $("#drawerEyebrow").textContent = "Deployment policy";
  $("#drawerTitle").textContent = policyId;
  $("#drawerBody").innerHTML = `<section class="drawer-section"><div>${badge(policy.status)}</div><div class="key-grid">${keyValue("Model", policy.model_id)}${keyValue("Fallback", policy.fallback_plan_id)}${keyValue("Hardware", policy.hardware_fingerprint_sha256)}${keyValue("Created", policy.created_at)}</div></section><section class="drawer-section"><h3>Decision list</h3>${(policy.rules || []).map((rule) => `<div class="issue"><b>${escapeHtml(rule.predicate)}</b><br>${escapeHtml(rule.plan_id)} · evidence ${escapeHtml((rule.evidence_ids || []).join(", "))}</div>`).join("") || '<p>No rule recorded.</p>'}</section>`;
  $("#drawer").showModal();
}

async function screenCandidates() {
  const message = $("#formMessage");
  try {
    const payload = submissionPayload();
    if (!payload.model_id) throw new Error("Select a registered model first.");
    $("#screenButton").disabled = true;
    message.textContent = "Running capability and conservative VRAM pruning…";
    message.className = "form-message";
    const result = await api("/api/v1/tune", { method: "POST", body: JSON.stringify(workloadFromSubmission(payload)) });
    renderCandidates(result);
    message.textContent = `Screened ${result.candidate_count + result.pruned_count} typed candidates; none are measured yet.`;
    message.className = "form-message pass";
  } catch (error) {
    message.textContent = error.message;
    message.className = "form-message fail";
  } finally { $("#screenButton").disabled = false; }
}

async function queueBenchmark(event) {
  event.preventDefault();
  const message = $("#formMessage");
  if (!event.currentTarget.reportValidity()) return;
  try {
    const payload = submissionPayload();
    $("#queueButton").disabled = true;
    message.textContent = "Submitting a typed specification to the isolated local worker…";
    message.className = "form-message";
    const job = await api("/api/v1/jobs/benchmark", { method: "POST", body: JSON.stringify(payload), control: true });
    message.textContent = `${job.job_id} queued. Measured does not mean validated.`;
    message.className = "form-message pass";
    showToast(`${job.job_id} queued on this machine.`);
    await refreshOperationalData();
    document.location.hash = "jobs";
  } catch (error) {
    message.textContent = error.message;
    message.className = "form-message fail";
  } finally { $("#queueButton").disabled = false; }
}

async function cancelJob(jobId) {
  if (!window.confirm(`Stop ${jobId}? Any committed failure artifact will remain auditable.`)) return;
  try {
    await api(`/api/v1/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST", control: true });
    showToast(`${jobId} cancellation requested.`);
    await refreshOperationalData();
  } catch (error) { showToast(`Cancel failed: ${error.message}`); }
}

async function startService(backend) {
  if (!window.confirm(`Start the pinned ${backend} service locally? First use may download registered model files.`)) return;
  try {
    await api(`/api/v1/runtime-services/${encodeURIComponent(backend)}/start`, { method: "POST", control: true });
    showToast(`${backend} is loading on the local GPU.`);
    await refreshOperationalData();
  } catch (error) { showToast(`Runtime start failed: ${error.message}`); }
}

async function stopService(backend) {
  if (!window.confirm(`Stop the managed ${backend} service?`)) return;
  try {
    await api(`/api/v1/runtime-services/${encodeURIComponent(backend)}/stop`, { method: "POST", control: true });
    showToast(`${backend} stopped.`);
    await refreshOperationalData();
  } catch (error) { showToast(`Runtime stop failed: ${error.message}`); }
}

async function refreshOperationalData() {
  const [runs, jobs, policies, evidence, services] = await Promise.all([
    api("/api/v1/runs?limit=500"), api("/api/v1/jobs?limit=200"), api("/api/v1/policies"), api("/api/v1/evidence"), api("/api/v1/runtime-services"),
  ]);
  state.runs = runs; state.jobs = jobs; state.policies = policies; state.evidence = evidence; state.services = services;
  $("#runCount").textContent = runs.length;
  $("#eligibleCount").textContent = runs.filter((item) => item.validation?.policy_eligible).length;
  renderJobs(); renderRuns(); renderPolicies(); renderEvidence(); renderServices();
}

async function refreshAll() {
  $("#refresh").disabled = true;
  try {
    const [health, inspection, capabilities] = await Promise.all([
      api("/health"), api("/api/v1/inspect", { method: "POST" }), api("/api/v1/runtime-capabilities"),
    ]);
    state.capabilities = capabilities;
    $("#health").className = "pill pass";
    $("#health").innerHTML = '<span aria-hidden="true">✓</span>Local API ready';
    $("#gpu").textContent = inspection.gpu?.name || "GPU not detected";
    $("#gpuDetail").textContent = `${formatBytes(inspection.gpu?.vram_bytes)} · SM ${inspection.gpu?.compute_capability || "unknown"} · CUDA ${inspection.software?.cuda_runtime || "unknown"}`;
    $("#fingerprint").textContent = inspection.fingerprint_id || "unavailable";
    $("#connectionBanner").className = "banner pass";
    $("#connectionBanner").textContent = `${health.mode} session ready. Control is loopback-only; artifacts are stored locally.`;
    $("#lastRefresh").textContent = `Last inspected ${new Date().toLocaleTimeString()}`;
    renderCapabilities();
    await refreshOperationalData();
  } catch (error) {
    $("#health").className = "pill fail";
    $("#health").innerHTML = '<span aria-hidden="true">×</span>API unavailable';
    $("#connectionBanner").className = "banner fail";
    $("#connectionBanner").textContent = `Local connection failed: ${error.message}`;
  } finally { $("#refresh").disabled = false; }
}

async function bootstrap() {
  root.dataset.theme = localStorage.getItem("edgeflow-theme") || "dark";
  try {
    const [session, models, experiments] = await Promise.all([
      api("/api/v1/session"), api("/api/v1/models"), api("/api/v1/experiments"),
    ]);
    state.token = session.control_token;
    state.models = models;
    state.experiments = experiments.experiments;
    populateModels();
    await refreshAll();
  } catch (error) {
    $("#connectionBanner").className = "banner fail";
    $("#connectionBanner").textContent = `Unable to initialize the localhost console: ${error.message}`;
  }
}

$("#theme").addEventListener("click", () => {
  root.dataset.theme = root.dataset.theme === "dark" ? "light" : "dark";
  localStorage.setItem("edgeflow-theme", root.dataset.theme);
});
$("#refresh").addEventListener("click", refreshAll);
$("#model").addEventListener("change", syncRuntimeFields);
$("#backend").addEventListener("change", syncRuntimeFields);
$("#screenButton").addEventListener("click", screenCandidates);
$("#workloadForm").addEventListener("submit", queueBenchmark);
$("#search").addEventListener("input", renderRuns);
$("#eligibleOnly").addEventListener("change", renderRuns);
$("#drawerClose").addEventListener("click", () => $("#drawer").close());
$("#drawer").addEventListener("click", (event) => { if (event.target === $("#drawer")) $("#drawer").close(); });
document.addEventListener("click", (event) => {
  const cancel = event.target.closest("[data-cancel]");
  const run = event.target.closest("[data-run]");
  const job = event.target.closest("[data-job]");
  const policy = event.target.closest("[data-policy]");
  const evidence = event.target.closest("[data-evidence]");
  const serviceStart = event.target.closest("[data-service-start]");
  const serviceStop = event.target.closest("[data-service-stop]");
  if (serviceStart) startService(serviceStart.dataset.serviceStart);
  else if (serviceStop) stopService(serviceStop.dataset.serviceStop);
  else if (cancel) cancelJob(cancel.dataset.cancel);
  else if (run?.dataset.run) openRun(run.dataset.run);
  else if (job) openJob(job.dataset.job);
  else if (policy) openPolicy(policy.dataset.policy);
  else if (evidence) openEvidence(evidence.dataset.evidence);
});
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshOperationalData().catch(() => {}); });
window.setInterval(() => { if (!document.hidden && state.token) refreshOperationalData().catch(() => {}); }, 3000);

bootstrap();
