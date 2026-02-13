import json
import time
import urllib.request


API_URL = "http://localhost:8000/chat"

# Use chat formatting if your model expects it
BASE_PROMPT = "<|user|>\nExplain what a transformer is in simple terms.\n<|assistant|>\n"

# Inflate prompt length by repeating filler text
FILLER = " extra context"  


def post_chat(prompt: str, max_new_tokens: int = 64):
    payload = {
        "prompt": prompt,
        "max_new_tokens": max_new_tokens,
        "temperature": 0.2,
        "top_p": 0.95,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        API_URL,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = resp.read().decode("utf-8")
    elapsed_ms = int((time.perf_counter() - start) * 1000)
    return json.loads(body), elapsed_ms


def main():
    repeats = [0, 25, 75, 150, 250]

    print("Running stress test...")
    print("-" * 72)
    print(f"{'run':<4} {'prompt_tokens':<13} {'latency_ms':<10} {'context_used%':<13} {'est_ops(n^2)':<12}")
    print("-" * 72)

    for i, r in enumerate(repeats, start=1):
        prompt = BASE_PROMPT + (FILLER * r)
        resp, elapsed_ms = post_chat(prompt)

        pt = resp.get("prompt_tokens")
        lm = resp.get("latency_ms")
        cup = resp.get("context_used_pct")
        ops = resp.get("estimated_attention_ops")

        print(f"{i:<4} {str(pt):<13} {str(lm):<10} {str(cup):<13} {str(ops):<12}")

    print("-" * 72)
    print("Done.")


if __name__ == "__main__":
    main()