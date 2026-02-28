async function runChat() {
    const prompt = document.getElementById("prompt").value;
    const mode = document.getElementById("mode").value;
    const refineSteps = parseInt(document.getElementById("refine_steps").value);
    const maxTokens = parseInt(document.getElementById("max_tokens").value);
    const temperature = parseFloat(document.getElementById("temperature").value);

    const payload = {
        prompt: prompt,
        mode: mode,
        refine_steps: refineSteps,
        max_new_tokens: maxTokens,
        temperature: temperature
    };

    const response = await fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    document.getElementById("chat_output").textContent =
        JSON.stringify(data, null, 2);
}

async function runEval() {
    const prompt = document.getElementById("eval_prompt").value;
    const responseText = document.getElementById("eval_response").value;

    const payload = {
        prompt: prompt,
        response: responseText,
        criteria: ["overall"]
    };

    const response = await fetch("/evaluate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
    });

    const data = await response.json();
    document.getElementById("eval_output").textContent =
        JSON.stringify(data, null, 2);
}

async function loadModel() {
    const response = await fetch("/model");
    const data = await response.json();
    document.getElementById("model_output").textContent =
        JSON.stringify(data, null, 2);
}