const state = {
    apiKey: "dev-local-key",
    auth: {
        role: "unknown",
        scopes: [],
        capabilities: {},
        isAdmin: false,
        keyFingerprint: null,
    },
    authCatalog: [],
};

const TAB_CAPABILITY_MAP = {
    playground: "chat.invoke",
    rag: "rag.query",
    datasets: "datasets.read",
    batch: "batch.read",
    metrics: "metrics.read",
    agent: "agent.invoke",
    status: null,
    admin: "admin.access",
};

const BUTTON_CAPABILITY_MAP = {
    chat_btn: "chat.invoke",
    rag_btn: "rag.query",
    ds_upload_btn: "datasets.write",
    ds_list_btn: "datasets.read",
    batch_start_btn: "batch.write",
    batch_status_btn: "batch.read",
    batch_result_btn: "batch.read",
    metrics_dashboard_btn: "metrics.read",
    metrics_prom_btn: "metrics.read",
    agent_tools_btn: "agent.invoke",
    agent_run_btn: "agent.invoke",
    rag_backend_btn: "rag.read",
    tracing_status_btn: "metrics.read",
    admin_runtime_get_btn: "admin.access",
    admin_runtime_update_btn: "admin.access",
    admin_profile_list_btn: "admin.access",
    admin_profile_get_btn: "admin.access",
    admin_profile_upsert_btn: "admin.access",
    admin_profile_apply_btn: "admin.access",
    admin_profile_delete_btn: "admin.access",
    admin_maint_status_btn: "admin.access",
    admin_maint_enable_btn: "admin.access",
    admin_maint_disable_btn: "admin.access",
    admin_cb_status_btn: "admin.access",
    admin_cb_open_btn: "admin.access",
    admin_cb_reset_btn: "admin.access",
    admin_slo_status_btn: "admin.access",
    admin_slo_incidents_btn: "admin.access",
    admin_slo_current_btn: "admin.access",
    admin_slo_get_btn: "admin.access",
    admin_slo_ack_btn: "admin.access",
    admin_slo_note_btn: "admin.access",
    admin_tpl_list_btn: "admin.access",
    admin_tpl_get_btn: "admin.access",
    admin_tpl_upsert_btn: "admin.access",
    admin_tpl_delete_btn: "admin.access",
    admin_runbook_list_btn: "admin.access",
    admin_runbook_create_btn: "admin.access",
    admin_runbook_from_tpl_btn: "admin.access",
    admin_runbook_get_btn: "admin.access",
    admin_runbook_step_btn: "admin.access",
    admin_runbook_complete_btn: "admin.access",
    admin_runbook_abort_btn: "admin.access",
    admin_audit_btn: "admin.access",
    admin_state_export_btn: "admin.access",
    admin_state_export_to_import_btn: "admin.access",
    admin_state_import_btn: "admin.access",
};

function authHeaders(extra = {}) {
    const key = state.apiKey;
    return {
        ...extra,
        "x-api-key": key,
        "Authorization": `Bearer ${key}`,
    };
}

function byId(id) {
    return document.getElementById(id);
}

function boolFromSelect(id, fallback = false) {
    const value = (byId(id)?.value || "").trim().toLowerCase();
    if (!value) return fallback;
    return value === "true" || value === "1" || value === "yes" || value === "on";
}

function intFromInput(id, fallback = null) {
    const raw = (byId(id)?.value || "").trim();
    if (!raw) return fallback;
    const parsed = Number(raw);
    return Number.isFinite(parsed) ? Math.trunc(parsed) : fallback;
}

function parseJsonInput(id, fallback = {}) {
    const raw = (byId(id)?.value || "").trim();
    if (!raw) return fallback;
    try {
        return JSON.parse(raw);
    } catch {
        throw new Error(`Invalid JSON in ${id}`);
    }
}

function parseCsvSteps(id) {
    return (byId(id)?.value || "")
        .split(",")
        .map((x) => x.trim())
        .filter(Boolean);
}

function withQuery(path, params = {}) {
    const entries = Object.entries(params).filter(([, v]) => v !== null && v !== undefined && `${v}` !== "");
    if (!entries.length) return path;
    const qs = new URLSearchParams();
    entries.forEach(([k, v]) => qs.set(k, `${v}`));
    return `${path}?${qs.toString()}`;
}

function show(targetId, value) {
    byId(targetId).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setLoading(targetId, label) {
    show(targetId, `${label}...`);
}

function hasCapability(capability) {
    if (!capability) return true;
    return !!state.auth.capabilities?.[capability];
}

function escapeHtml(value) {
    return `${value ?? ""}`
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;")
        .replaceAll('"', "&quot;")
        .replaceAll("'", "&#39;");
}

function renderAuthCapabilityMatrix(rows) {
    const host = byId("auth_caps_matrix");
    if (!host) return;
    if (!Array.isArray(rows) || !rows.length) {
        host.innerHTML = "";
        return;
    }
    host.innerHTML = rows.map((row) => {
        const allowed = !!row?.allowed;
        const requirement = row?.required_role ? `role: ${row.required_role}` : (row?.required_scope ? `scope: ${row.required_scope}` : "n/a");
        return `
            <div class="auth-matrix-row">
                <span class="auth-chip ${allowed ? "allowed" : "denied"}">${allowed ? "allowed" : "denied"}</span>
                <span class="auth-matrix-cap">${escapeHtml(row?.capability || "unknown")}</span>
                <span class="auth-matrix-desc">${escapeHtml(row?.description || "")} (${escapeHtml(requirement)})</span>
            </div>
        `;
    }).join("");
}

function applyAuthUiState() {
    const authRoleLabel = byId("auth_role");
    const authStateLabel = byId("auth_state");
    const adminTab = document.querySelector('.tab[data-tab="admin"]');
    const adminPanel = byId("admin");
    const adminNotice = byId("admin_access_notice");
    if (!authRoleLabel || !authStateLabel || !adminTab || !adminPanel || !adminNotice) {
        return;
    }

    const role = state.auth.role || "unknown";
    const scopeCount = Array.isArray(state.auth.scopes) ? state.auth.scopes.length : 0;
    const capabilityCount = Object.values(state.auth.capabilities || {}).filter(Boolean).length;
    const fp = state.auth.keyFingerprint ? ` (${state.auth.keyFingerprint})` : "";
    authRoleLabel.textContent = `Role: ${role} | scopes: ${scopeCount} | caps: ${capabilityCount}${fp}`;

    const adminAllowed = hasCapability("admin.access");
    adminTab.disabled = !adminAllowed;
    adminTab.title = adminAllowed ? "" : "Admin key required";
    adminNotice.classList.toggle("visible", !adminAllowed);
    adminNotice.textContent = adminAllowed ? "" : "Admin operations are disabled for this API key (admin role required).";

    const controls = [...adminPanel.querySelectorAll("button, input, select, textarea")];
    controls.forEach((el) => {
        const id = el.id || "";
        if (id === "admin_output") return;
        el.disabled = !adminAllowed;
    });

    Object.entries(TAB_CAPABILITY_MAP).forEach(([tabKey, capability]) => {
        const tabEl = document.querySelector(`.tab[data-tab="${tabKey}"]`);
        if (!tabEl) return;
        const allowed = hasCapability(capability);
        tabEl.disabled = !allowed;
        tabEl.title = allowed ? "" : `Requires capability: ${capability}`;
        if (!allowed && tabEl.classList.contains("active")) {
            const fallback = document.querySelector('.tab[data-tab="status"]') || document.querySelector('.tab[data-tab="playground"]');
            if (fallback) fallback.click();
        }
    });

    Object.entries(BUTTON_CAPABILITY_MAP).forEach(([id, capability]) => {
        const el = byId(id);
        if (!el) return;
        const allowed = hasCapability(capability);
        el.disabled = !allowed;
        el.title = allowed ? "" : `Requires capability: ${capability}`;
    });

    if (!adminAllowed && adminTab.classList.contains("active")) {
        const fallback = document.querySelector('.tab[data-tab="playground"]');
        if (fallback) fallback.click();
    }

    if (!state.apiKey) {
        authStateLabel.textContent = "Missing key";
        authStateLabel.classList.add("warn");
    }
}

function updateAuthStateFromContext(payload) {
    const ctx = payload?.data || {};
    state.auth = {
        role: (ctx.role || "unknown").toString(),
        scopes: Array.isArray(ctx.scopes) ? ctx.scopes : [],
        capabilities: typeof ctx.capabilities === "object" && ctx.capabilities ? ctx.capabilities : {},
        isAdmin: !!ctx.is_admin,
        keyFingerprint: ctx.key_fingerprint || null,
    };
    applyAuthUiState();
}

async function refreshAuthCapabilities({ showErrors = true } = {}) {
    if (!state.apiKey) {
        state.authCatalog = [];
        renderAuthCapabilityMatrix([]);
        return null;
    }
    try {
        const payload = await requestJson("/v1/auth/capabilities", { headers: authHeaders(), skipAuthRefresh: true });
        const rows = Array.isArray(payload?.data) ? payload.data : [];
        state.authCatalog = rows;
        renderAuthCapabilityMatrix(rows);
        return payload;
    } catch (err) {
        state.authCatalog = [];
        renderAuthCapabilityMatrix([]);
        if (showErrors) {
            show("auth_ctx_output", err);
        }
        return null;
    }
}

async function refreshAuthContext({ showErrors = true } = {}) {
    const authStateLabel = byId("auth_state");
    if (!state.apiKey) {
        state.auth = { role: "unknown", scopes: [], capabilities: {}, isAdmin: false, keyFingerprint: null };
        applyAuthUiState();
        return null;
    }

    authStateLabel.textContent = "Checking key";
    authStateLabel.classList.remove("warn");
    try {
        const payload = await requestJson("/v1/auth/context", { headers: authHeaders(), skipAuthRefresh: true });
        updateAuthStateFromContext(payload);
        authStateLabel.textContent = "Key verified";
        authStateLabel.classList.remove("warn");
        show("auth_ctx_output", payload);
        await refreshAuthCapabilities({ showErrors });
        return payload;
    } catch (err) {
        state.auth = { role: "unknown", scopes: [], capabilities: {}, isAdmin: false, keyFingerprint: null };
        state.authCatalog = [];
        authStateLabel.textContent = "Key invalid";
        authStateLabel.classList.add("warn");
        applyAuthUiState();
        renderAuthCapabilityMatrix([]);
        if (showErrors) {
            show("admin_output", err);
            show("auth_ctx_output", err);
        }
        return null;
    }
}

async function requestJson(path, options = {}) {
    const { skipAuthRefresh = false, ...fetchOptions } = options;
    const response = await fetch(path, fetchOptions);
    const text = await response.text();
    const payload = text ? (() => {
        try { return JSON.parse(text); } catch { return { raw: text }; }
    })() : {};

    if (!response.ok) {
        if (!skipAuthRefresh && (response.status === 401 || response.status === 403)) {
            setTimeout(() => refreshAuthContext({ showErrors: false }), 0);
        }
        throw { status: response.status, payload };
    }
    return payload;
}

function initTabs() {
    const tabs = [...document.querySelectorAll(".tab")];
    tabs.forEach((tab) => {
        tab.addEventListener("click", () => {
            tabs.forEach((x) => x.classList.remove("active"));
            tab.classList.add("active");
            const key = tab.dataset.tab;
            document.querySelectorAll(".panel").forEach((panel) => panel.classList.remove("active"));
            byId(key).classList.add("active");
        });
    });
}

function initAuth() {
    applyAuthUiState();

    byId("save_key_btn").addEventListener("click", async () => {
        state.apiKey = byId("api_key").value.trim();
        const label = byId("auth_state");
        if (!state.apiKey) {
            label.textContent = "Missing key";
            label.classList.add("warn");
            state.auth = { role: "unknown", scopes: [], capabilities: {}, isAdmin: false, keyFingerprint: null };
            applyAuthUiState();
            return;
        }
        await refreshAuthContext({ showErrors: true });
    });

    refreshAuthContext({ showErrors: false });
}

function initPlayground() {
    byId("chat_btn").addEventListener("click", async () => {
        setLoading("chat_output", "Running chat");
        const payload = {
            prompt: byId("chat_prompt").value,
            mode: byId("chat_mode").value,
            refine_steps: Number(byId("chat_refine_steps").value || 1),
            max_new_tokens: Number(byId("chat_max_tokens").value || 160),
            temperature: Number(byId("chat_temperature").value || 0.2),
        };

        try {
            const data = await requestJson("/v1/chat", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show("chat_output", data);
        } catch (err) {
            show("chat_output", err);
        }
    });
}

function initRag() {
    byId("rag_btn").addEventListener("click", async () => {
        setLoading("rag_output", "Running retrieval+generation");
        const payload = {
            dataset_id: byId("rag_dataset").value.trim(),
            query: byId("rag_query").value,
            top_k: Number(byId("rag_top_k").value || 5),
        };
        try {
            const data = await requestJson("/v1/rag", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show("rag_output", data);
        } catch (err) {
            show("rag_output", err);
        }
    });
}

function initDatasets() {
    byId("ds_upload_btn").addEventListener("click", async () => {
        const file = byId("ds_file").files[0];
        if (!file) {
            show("ds_output", { error: "Pick a file first" });
            return;
        }
        setLoading("ds_output", "Uploading dataset");

        const fd = new FormData();
        fd.append("name", byId("ds_name").value.trim());
        fd.append("type", byId("ds_type").value);
        fd.append("metadata", byId("ds_metadata").value || "{}");
        fd.append("file", file);

        try {
            const response = await fetch("/v1/datasets/upload", {
                method: "POST",
                headers: authHeaders(),
                body: fd,
            });
            const body = await response.json();
            if (!response.ok) {
                throw { status: response.status, payload: body };
            }
            show("ds_output", body);
            if (body.dataset_id) {
                byId("batch_dataset_id").value = body.dataset_id;
                byId("rag_dataset").value = body.dataset_id;
            }
        } catch (err) {
            show("ds_output", err);
        }
    });

    byId("ds_list_btn").addEventListener("click", async () => {
        setLoading("ds_output", "Loading datasets");
        try {
            const data = await requestJson("/v1/datasets", {
                headers: authHeaders(),
            });
            show("ds_output", data);
        } catch (err) {
            show("ds_output", err);
        }
    });
}

function initBatch() {
    byId("batch_start_btn").addEventListener("click", async () => {
        setLoading("batch_output", "Starting batch eval");
        const criteria = byId("batch_criteria").value.split(",").map((x) => x.trim()).filter(Boolean);
        const payload = {
            dataset_id: byId("batch_dataset_id").value.trim(),
            criteria: criteria.length ? criteria : ["overall"],
            concurrency: Number(byId("batch_concurrency").value || 1),
        };
        try {
            const data = await requestJson("/v1/batch-evals", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show("batch_output", data);
            byId("batch_run_id").value = data.batch_eval_id || data.run_id || "";
        } catch (err) {
            show("batch_output", err);
        }
    });

    byId("batch_status_btn").addEventListener("click", async () => {
        const runId = byId("batch_run_id").value.trim();
        if (!runId) {
            show("batch_output", { error: "Set batch run id first" });
            return;
        }
        setLoading("batch_output", "Loading batch status");
        try {
            const data = await requestJson(`/v1/batch-evals/${runId}`, {
                headers: authHeaders(),
            });
            show("batch_output", data);
        } catch (err) {
            show("batch_output", err);
        }
    });

    byId("batch_result_btn").addEventListener("click", async () => {
        const runId = byId("batch_run_id").value.trim();
        if (!runId) {
            show("batch_output", { error: "Set batch run id first" });
            return;
        }
        setLoading("batch_output", "Loading batch result");
        try {
            const data = await requestJson(`/v1/batch-evals/${runId}/result`, {
                headers: authHeaders(),
            });
            show("batch_output", data);
        } catch (err) {
            show("batch_output", err);
        }
    });
}

function initMetrics() {
    byId("metrics_dashboard_btn").addEventListener("click", async () => {
        setLoading("metrics_output", "Loading dashboard");
        try {
            const data = await requestJson("/metrics/dashboard?window=15m", {
                headers: authHeaders(),
            });
            show("metrics_output", data);
        } catch (err) {
            show("metrics_output", err);
        }
    });

    byId("metrics_prom_btn").addEventListener("click", async () => {
        setLoading("metrics_output", "Loading Prometheus text");
        try {
            const response = await fetch("/v1/metrics?format=prometheus", {
                headers: authHeaders(),
            });
            const body = await response.text();
            if (!response.ok) {
                throw { status: response.status, payload: body };
            }
            show("metrics_output", body);
        } catch (err) {
            show("metrics_output", err);
        }
    });
}

function initAgent() {
    byId("agent_tools_btn").addEventListener("click", async () => {
        setLoading("agent_output", "Loading agent tools");
        try {
            const data = await requestJson("/v1/agent/tools", { headers: authHeaders() });
            show("agent_output", data);
        } catch (err) {
            show("agent_output", err);
        }
    });

    byId("agent_run_btn").addEventListener("click", async () => {
        setLoading("agent_output", "Running agent");
        try {
            const goal = (byId("agent_goal").value || "").trim();
            const requestedTools = parseCsvSteps("agent_requested_tools");
            const payload = { goal };
            if (requestedTools.length) {
                payload.requested_tools = requestedTools;
            }
            const data = await requestJson("/v1/agent/run", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show("agent_output", data);
        } catch (err) {
            show("agent_output", err);
        }
    });
}

function initStatus() {
    byId("status_btn").addEventListener("click", async () => {
        setLoading("status_output", "Loading status");
        try {
            const [health, ready, model] = await Promise.all([
                requestJson("/v1/health", { headers: authHeaders() }),
                requestJson("/readyz", { headers: authHeaders() }),
                requestJson("/model", { headers: authHeaders() }),
            ]);
            show("status_output", { health, ready, model });
        } catch (err) {
            show("status_output", err);
        }
    });

    byId("auth_ctx_btn").addEventListener("click", async () => {
        setLoading("auth_ctx_output", "Loading auth context");
        const payload = await refreshAuthContext({ showErrors: true });
        if (!payload) {
            show("auth_ctx_output", { error: "Unable to load auth context" });
        }
    });

    byId("auth_caps_btn").addEventListener("click", async () => {
        setLoading("auth_ctx_output", "Loading access matrix");
        const payload = await refreshAuthCapabilities({ showErrors: true });
        if (!payload) {
            show("auth_ctx_output", { error: "Unable to load access matrix" });
            renderAuthCapabilityMatrix([]);
        }
    });

    byId("rag_backend_btn").addEventListener("click", async () => {
        setLoading("status_output", "Loading RAG backend");
        try {
            const data = await requestJson("/v1/rag/vector-backend", { headers: authHeaders() });
            show("status_output", { rag_vector_backend: data });
        } catch (err) {
            show("status_output", err);
        }
    });

    byId("tracing_status_btn").addEventListener("click", async () => {
        setLoading("status_output", "Loading tracing status");
        try {
            const data = await requestJson("/v1/tracing/status", { headers: authHeaders() });
            show("status_output", { tracing: data });
        } catch (err) {
            show("status_output", err);
        }
    });
}

function initAdminOps() {
    const out = "admin_output";

    byId("admin_runtime_get_btn").addEventListener("click", async () => {
        setLoading(out, "Loading runtime config");
        try {
            const data = await requestJson("/v1/admin/runtime-config", { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runtime_update_btn").addEventListener("click", async () => {
        setLoading(out, "Updating runtime config");
        try {
            const payload = parseJsonInput("admin_runtime_patch", {});
            const dryRun = boolFromSelect("admin_runtime_dry_run", true);
            const data = await requestJson(withQuery("/v1/admin/runtime-config", { dry_run: dryRun }), {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show(out, data);
        } catch (err) {
            show(out, err.message ? { error: err.message } : err);
        }
    });

    byId("admin_profile_list_btn").addEventListener("click", async () => {
        setLoading(out, "Listing runtime profiles");
        try {
            const data = await requestJson("/v1/admin/runtime-config/profiles", { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_profile_get_btn").addEventListener("click", async () => {
        const name = (byId("admin_profile_name").value || "").trim();
        if (!name) return show(out, { error: "Set profile name first" });
        setLoading(out, "Loading profile");
        try {
            const data = await requestJson(`/v1/admin/runtime-config/profiles/${encodeURIComponent(name)}`, { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_profile_upsert_btn").addEventListener("click", async () => {
        const name = (byId("admin_profile_name").value || "").trim();
        if (!name) return show(out, { error: "Set profile name first" });
        setLoading(out, "Upserting profile");
        try {
            const patch = parseJsonInput("admin_profile_patch", {});
            const overwrite = boolFromSelect("admin_profile_overwrite", true);
            const data = await requestJson(withQuery(`/v1/admin/runtime-config/profiles/${encodeURIComponent(name)}`, { overwrite }), {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(patch),
            });
            show(out, data);
        } catch (err) {
            show(out, err.message ? { error: err.message } : err);
        }
    });

    byId("admin_profile_apply_btn").addEventListener("click", async () => {
        const name = (byId("admin_profile_name").value || "").trim();
        if (!name) return show(out, { error: "Set profile name first" });
        setLoading(out, "Applying profile");
        try {
            const dryRun = boolFromSelect("admin_profile_apply_dry_run", true);
            const data = await requestJson(withQuery(`/v1/admin/runtime-config/profiles/${encodeURIComponent(name)}/apply`, { dry_run: dryRun }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_profile_delete_btn").addEventListener("click", async () => {
        const name = (byId("admin_profile_name").value || "").trim();
        if (!name) return show(out, { error: "Set profile name first" });
        setLoading(out, "Deleting profile");
        try {
            const data = await requestJson(`/v1/admin/runtime-config/profiles/${encodeURIComponent(name)}`, {
                method: "DELETE",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_maint_status_btn").addEventListener("click", async () => {
        setLoading(out, "Loading maintenance status");
        try {
            const data = await requestJson("/v1/admin/maintenance", { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_maint_enable_btn").addEventListener("click", async () => {
        setLoading(out, "Enabling maintenance");
        try {
            const reason = (byId("admin_maint_reason").value || "").trim() || "maintenance";
            const durationSeconds = intFromInput("admin_maint_duration", 300);
            const readOnly = boolFromSelect("admin_maint_read_only", false);
            const data = await requestJson(withQuery("/v1/admin/maintenance/enable", {
                reason,
                duration_seconds: durationSeconds,
                read_only: readOnly,
            }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_maint_disable_btn").addEventListener("click", async () => {
        setLoading(out, "Disabling maintenance");
        try {
            const data = await requestJson("/v1/admin/maintenance/disable", {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_cb_status_btn").addEventListener("click", async () => {
        setLoading(out, "Loading breaker status");
        try {
            const data = await requestJson("/v1/admin/circuit-breaker", { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_cb_open_btn").addEventListener("click", async () => {
        setLoading(out, "Force-opening breaker");
        try {
            const reason = (byId("admin_cb_reason").value || "").trim() || "manual";
            const durationSeconds = intFromInput("admin_cb_duration", 60);
            const data = await requestJson(withQuery("/v1/admin/circuit-breaker/open", {
                reason,
                duration_seconds: durationSeconds,
            }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_cb_reset_btn").addEventListener("click", async () => {
        setLoading(out, "Resetting breaker");
        try {
            const data = await requestJson("/v1/admin/circuit-breaker/reset", {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_status_btn").addEventListener("click", async () => {
        setLoading(out, "Loading SLO status");
        try {
            const windowSeconds = intFromInput("admin_slo_window", 900);
            const data = await requestJson(withQuery("/v1/admin/slo/status", { window_seconds: windowSeconds }), {
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_incidents_btn").addEventListener("click", async () => {
        setLoading(out, "Loading incidents");
        try {
            const limit = intFromInput("admin_slo_incident_limit", 50);
            const status = (byId("admin_slo_incident_status").value || "").trim();
            const data = await requestJson(withQuery("/v1/admin/slo/incidents", { limit, status }), { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_current_btn").addEventListener("click", async () => {
        setLoading(out, "Loading current incident");
        try {
            const data = await requestJson("/v1/admin/slo/incidents/current", { headers: authHeaders() });
            show(out, data);
            const current = data?.data?.incident_id;
            if (current) byId("admin_slo_incident_id").value = current;
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_get_btn").addEventListener("click", async () => {
        const incidentId = (byId("admin_slo_incident_id").value || "").trim();
        if (!incidentId) return show(out, { error: "Set incident ID first" });
        setLoading(out, "Loading incident");
        try {
            const data = await requestJson(`/v1/admin/slo/incidents/${encodeURIComponent(incidentId)}`, { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_ack_btn").addEventListener("click", async () => {
        const incidentId = (byId("admin_slo_incident_id").value || "").trim();
        if (!incidentId) return show(out, { error: "Set incident ID first" });
        setLoading(out, "Acknowledging incident");
        try {
            const note = (byId("admin_slo_ack_note").value || "").trim();
            const data = await requestJson(withQuery(`/v1/admin/slo/incidents/${encodeURIComponent(incidentId)}/ack`, { note }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_slo_note_btn").addEventListener("click", async () => {
        const incidentId = (byId("admin_slo_incident_id").value || "").trim();
        if (!incidentId) return show(out, { error: "Set incident ID first" });
        setLoading(out, "Adding incident note");
        try {
            const note = (byId("admin_slo_note").value || "").trim();
            const data = await requestJson(`/v1/admin/slo/incidents/${encodeURIComponent(incidentId)}/notes`, {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify({ note }),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_tpl_list_btn").addEventListener("click", async () => {
        setLoading(out, "Listing templates");
        try {
            const limit = intFromInput("admin_tpl_limit", 50);
            const data = await requestJson(withQuery("/v1/admin/runbook-templates", { limit }), { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_tpl_get_btn").addEventListener("click", async () => {
        const templateId = (byId("admin_tpl_id").value || "").trim();
        if (!templateId) return show(out, { error: "Set template ID first" });
        setLoading(out, "Loading template");
        try {
            const data = await requestJson(`/v1/admin/runbook-templates/${encodeURIComponent(templateId)}`, { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_tpl_upsert_btn").addEventListener("click", async () => {
        const templateId = (byId("admin_tpl_id").value || "").trim();
        if (!templateId) return show(out, { error: "Set template ID first" });
        setLoading(out, "Upserting template");
        try {
            const payload = {
                name: (byId("admin_tpl_name").value || "").trim(),
                environment: (byId("admin_tpl_env").value || "").trim(),
                steps: parseCsvSteps("admin_tpl_steps"),
            };
            const overwrite = boolFromSelect("admin_tpl_overwrite", true);
            const data = await requestJson(withQuery(`/v1/admin/runbook-templates/${encodeURIComponent(templateId)}`, { overwrite }), {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_tpl_delete_btn").addEventListener("click", async () => {
        const templateId = (byId("admin_tpl_id").value || "").trim();
        if (!templateId) return show(out, { error: "Set template ID first" });
        setLoading(out, "Deleting template");
        try {
            const data = await requestJson(`/v1/admin/runbook-templates/${encodeURIComponent(templateId)}`, {
                method: "DELETE",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_list_btn").addEventListener("click", async () => {
        setLoading(out, "Listing runbooks");
        try {
            const limit = intFromInput("admin_runbook_limit", 50);
            const status = (byId("admin_runbook_status_filter").value || "").trim();
            const data = await requestJson(withQuery("/v1/admin/runbooks", { limit, status }), { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_create_btn").addEventListener("click", async () => {
        setLoading(out, "Creating runbook");
        try {
            const payload = {
                name: (byId("admin_runbook_name").value || "").trim(),
                environment: (byId("admin_runbook_env").value || "").trim(),
                steps: parseCsvSteps("admin_runbook_steps"),
            };
            const data = await requestJson("/v1/admin/runbooks", {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show(out, data);
            const id = data?.data?.runbook_id;
            if (id) byId("admin_runbook_id").value = id;
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_from_tpl_btn").addEventListener("click", async () => {
        const templateId = (byId("admin_tpl_id").value || "").trim();
        if (!templateId) return show(out, { error: "Set template ID first" });
        setLoading(out, "Creating runbook from template");
        try {
            const name = (byId("admin_runbook_name").value || "").trim();
            const environment = (byId("admin_runbook_env").value || "").trim();
            const data = await requestJson(withQuery(`/v1/admin/runbooks/from-template/${encodeURIComponent(templateId)}`, { name, environment }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
            const id = data?.data?.runbook_id;
            if (id) byId("admin_runbook_id").value = id;
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_get_btn").addEventListener("click", async () => {
        const runbookId = (byId("admin_runbook_id").value || "").trim();
        if (!runbookId) return show(out, { error: "Set runbook ID first" });
        setLoading(out, "Loading runbook");
        try {
            const data = await requestJson(`/v1/admin/runbooks/${encodeURIComponent(runbookId)}`, { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_step_btn").addEventListener("click", async () => {
        const runbookId = (byId("admin_runbook_id").value || "").trim();
        const stepId = (byId("admin_runbook_step_id").value || "").trim();
        if (!runbookId || !stepId) return show(out, { error: "Set runbook ID and step ID first" });
        setLoading(out, "Updating step");
        try {
            const payload = {
                status: (byId("admin_runbook_step_status").value || "pending").trim(),
                note: (byId("admin_runbook_step_note").value || "").trim(),
            };
            const data = await requestJson(`/v1/admin/runbooks/${encodeURIComponent(runbookId)}/steps/${encodeURIComponent(stepId)}`, {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_complete_btn").addEventListener("click", async () => {
        const runbookId = (byId("admin_runbook_id").value || "").trim();
        if (!runbookId) return show(out, { error: "Set runbook ID first" });
        setLoading(out, "Completing runbook");
        try {
            const data = await requestJson(`/v1/admin/runbooks/${encodeURIComponent(runbookId)}/complete`, {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_runbook_abort_btn").addEventListener("click", async () => {
        const runbookId = (byId("admin_runbook_id").value || "").trim();
        if (!runbookId) return show(out, { error: "Set runbook ID first" });
        setLoading(out, "Aborting runbook");
        try {
            const reason = (byId("admin_runbook_abort_reason").value || "").trim();
            const data = await requestJson(withQuery(`/v1/admin/runbooks/${encodeURIComponent(runbookId)}/abort`, { reason }), {
                method: "POST",
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_audit_btn").addEventListener("click", async () => {
        setLoading(out, "Loading audit logs");
        try {
            const action = (byId("admin_audit_action").value || "").trim();
            const resourceType = (byId("admin_audit_resource_type").value || "").trim();
            const sinceTs = intFromInput("admin_audit_since_ts", null);
            const limit = intFromInput("admin_audit_limit", 100);
            const data = await requestJson(withQuery("/v1/admin/audit-logs", {
                limit,
                action,
                resource_type: resourceType,
                since_ts: sinceTs,
            }), {
                headers: authHeaders(),
            });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_state_export_btn").addEventListener("click", async () => {
        setLoading(out, "Exporting state");
        try {
            const data = await requestJson("/v1/admin/state/export", { headers: authHeaders() });
            show(out, data);
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_state_export_to_import_btn").addEventListener("click", async () => {
        setLoading(out, "Exporting state to import buffer");
        try {
            const data = await requestJson("/v1/admin/state/export", { headers: authHeaders() });
            byId("admin_state_import_json").value = JSON.stringify(data, null, 2);
            show(out, { ok: true, message: "State export copied into import buffer", data });
        } catch (err) {
            show(out, err);
        }
    });

    byId("admin_state_import_btn").addEventListener("click", async () => {
        setLoading(out, "Importing state");
        try {
            const payload = parseJsonInput("admin_state_import_json", {});
            const mode = (byId("admin_state_import_mode").value || "replace").trim();
            const dryRun = boolFromSelect("admin_state_import_dry_run", true);
            const data = await requestJson(withQuery("/v1/admin/state/import", { mode, dry_run: dryRun }), {
                method: "POST",
                headers: authHeaders({ "Content-Type": "application/json" }),
                body: JSON.stringify(payload),
            });
            show(out, data);
        } catch (err) {
            show(out, err.message ? { error: err.message } : err);
        }
    });
}

initTabs();
initAuth();
initPlayground();
initRag();
initDatasets();
initBatch();
initMetrics();
initAgent();
initStatus();
initAdminOps();
