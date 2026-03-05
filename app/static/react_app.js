import React, { useCallback, useMemo, useState } from "https://esm.sh/react@18.2.0";
import { createRoot } from "https://esm.sh/react-dom@18.2.0/client";

function requestJson(path, options = {}) {
    return fetch(path, options).then(async (response) => {
        const text = await response.text();
        const payload = text ? (() => {
            try {
                return JSON.parse(text);
            } catch {
                return { raw: text };
            }
        })() : {};
        if (!response.ok) {
            throw { status: response.status, payload };
        }
        return payload;
    });
}

function App() {
    const [apiKey, setApiKey] = useState("dev-local-key");
    const [tab, setTab] = useState("status");
    const [auth, setAuth] = useState(null);
    const [statusOut, setStatusOut] = useState("");
    const [agentOut, setAgentOut] = useState("");
    const [goal, setGoal] = useState("Inspect service status and dataset inventory.");
    const [tools, setTools] = useState("metrics.dashboard,datasets.list");

    const headers = useMemo(() => ({
        "x-api-key": apiKey,
        "Authorization": `Bearer ${apiKey}`,
    }), [apiKey]);

    const loadAuth = useCallback(async () => {
        const data = await requestJson("/v1/auth/context", { headers });
        setAuth(data.data || null);
        return data;
    }, [headers]);

    const loadStatus = useCallback(async () => {
        setStatusOut("Loading status...");
        try {
            const [health, ready, tracing, rag] = await Promise.all([
                requestJson("/v1/health", { headers }),
                requestJson("/readyz", { headers }),
                requestJson("/v1/tracing/status", { headers }),
                requestJson("/v1/rag/vector-backend", { headers }),
            ]);
            setStatusOut(JSON.stringify({ health, ready, tracing, rag }, null, 2));
            await loadAuth();
        } catch (err) {
            setStatusOut(JSON.stringify(err, null, 2));
        }
    }, [headers, loadAuth]);

    const listAgentTools = useCallback(async () => {
        setAgentOut("Loading agent tools...");
        try {
            const data = await requestJson("/v1/agent/tools", { headers });
            setAgentOut(JSON.stringify(data, null, 2));
            await loadAuth();
        } catch (err) {
            setAgentOut(JSON.stringify(err, null, 2));
        }
    }, [headers, loadAuth]);

    const runAgent = useCallback(async () => {
        setAgentOut("Running agent plan...");
        try {
            const requestedTools = tools.split(",").map((x) => x.trim()).filter(Boolean);
            const payload = { goal: goal.trim() };
            if (requestedTools.length) {
                payload.requested_tools = requestedTools;
            }
            const data = await requestJson("/v1/agent/run", {
                method: "POST",
                headers: { ...headers, "Content-Type": "application/json" },
                body: JSON.stringify(payload),
            });
            setAgentOut(JSON.stringify(data, null, 2));
            await loadAuth();
        } catch (err) {
            setAgentOut(JSON.stringify(err, null, 2));
        }
    }, [goal, headers, loadAuth, tools]);

    return React.createElement(
        "div",
        null,
        React.createElement(
            "div",
            { className: "panel active" },
            React.createElement(
                "div",
                { className: "grid" },
                React.createElement(
                    "label",
                    null,
                    "API Key",
                    React.createElement("input", {
                        value: apiKey,
                        onChange: (e) => setApiKey(e.target.value),
                        type: "password",
                    })
                ),
                React.createElement(
                    "label",
                    null,
                    "Active Tab",
                    React.createElement(
                        "select",
                        { value: tab, onChange: (e) => setTab(e.target.value) },
                        React.createElement("option", { value: "status" }, "Status"),
                        React.createElement("option", { value: "agent" }, "Agent")
                    )
                )
            ),
            React.createElement(
                "div",
                { className: "actions" },
                React.createElement("button", { type: "button", onClick: loadAuth }, "Refresh Auth"),
                React.createElement("button", { type: "button", onClick: loadStatus }, "Load Status"),
                React.createElement("button", { type: "button", onClick: listAgentTools }, "List Agent Tools")
            ),
            React.createElement(
                "pre",
                null,
                auth ? JSON.stringify({ auth }, null, 2) : "Auth context not loaded."
            )
        ),
        tab === "status" ? React.createElement(
            "section",
            { className: "panel active" },
            React.createElement("h2", null, "React Status Surface"),
            React.createElement("pre", null, statusOut || "Click Load Status.")
        ) : null,
        tab === "agent" ? React.createElement(
            "section",
            { className: "panel active" },
            React.createElement("h2", null, "React Agent Surface"),
            React.createElement(
                "div",
                { className: "grid" },
                React.createElement(
                    "label",
                    null,
                    "Goal",
                    React.createElement("textarea", {
                        value: goal,
                        onChange: (e) => setGoal(e.target.value),
                    })
                ),
                React.createElement(
                    "label",
                    null,
                    "Requested Tools (comma separated)",
                    React.createElement("input", {
                        value: tools,
                        onChange: (e) => setTools(e.target.value),
                    })
                )
            ),
            React.createElement(
                "div",
                { className: "actions" },
                React.createElement("button", { type: "button", onClick: runAgent }, "Run Agent")
            ),
            React.createElement("pre", null, agentOut || "Click Run Agent.")
        ) : null
    );
}

createRoot(document.getElementById("react-root")).render(React.createElement(App));
