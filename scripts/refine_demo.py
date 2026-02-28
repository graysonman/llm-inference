import json
import urllib.request

API_URL = "http://localhost:8000/chat"


def post(payload):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.loads(resp.read().decode("utf-8"))


if __name__ == "__main__":
    prompt = "<|user|>\nExplain encoder-decoder vs decoder-only in 6 sentences.\n<|assistant|>"

    single = post(
        {
            "prompt": prompt,
            "mode": "single",
            "max_new_tokens": 160,
            "temperature": 0.2,
            "top_p": 0.95,
        }
    )

    refine = post(
        {
            "prompt": prompt,
            "mode": "refine",
            "refine_steps": 1,
            "max_new_tokens": 160,
            "temperature": 0.2,
            "top_p": 0.95,
            "critique_temperature": 0.2,
        }
    )

    print("\n=== SINGLE ===\n")
    print(single["response"])
    print("\nmeta:", {k: single[k] for k in ["latency_ms", "prompt_tokens", "completion_tokens", "output_to_input_ratio"]})

    print("\n=== REFINE: ORIGINAL ===\n")
    print(refine.get("original_response", ""))

    print("\n=== REFINE: CRITIQUE ===\n")
    print(refine.get("critique", ""))

    print("\n=== REFINE: IMPROVED ===\n")
    print(refine["response"])
    print("\nmeta:", {k: refine[k] for k in ["latency_ms", "prompt_tokens", "completion_tokens", "output_to_input_ratio", "refine_steps_used"]})