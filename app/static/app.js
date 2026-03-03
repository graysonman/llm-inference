const state = {
    apiKey: "dev-local-key",
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

function show(targetId, value) {
    byId(targetId).textContent = typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function setLoading(targetId, label) {
    show(targetId, `${label}...`);
}

async function requestJson(path, options = {}) {
    const response = await fetch(path, options);
    const text = await response.text();
    const payload = text ? (() => {
        try { return JSON.parse(text); } catch { return { raw: text }; }
    })() : {};

    if (!response.ok) {
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
    byId("save_key_btn").addEventListener("click", () => {
        state.apiKey = byId("api_key").value.trim();
        const label = byId("auth_state");
        if (!state.apiKey) {
            label.textContent = "Missing key";
            label.classList.add("warn");
            return;
        }
        label.textContent = "Key set";
        label.classList.remove("warn");
    });
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
}

initTabs();
initAuth();
initPlayground();
initRag();
initDatasets();
initBatch();
initMetrics();
initStatus();