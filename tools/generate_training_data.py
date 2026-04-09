#!/usr/bin/env python3
"""Generate validated Adapsis training data using Opus.

For each scenario, asks Opus to produce the correct Adapsis response,
extracts the <code> block, validates it through the Adapsis API,
and only keeps examples where the code parses successfully.
"""

import json, sys, time, requests

GATEWAY = "http://127.0.0.1:4000"
ADAPSIS = "http://127.0.0.1:3002"
MODEL = "anthropic/claude-opus-4-6"
OUTPUT = "/home/marenz/.config/adapsis/training/curated-examples.jsonl"

# Read system prompt from Adapsis
SYSTEM_PROMPT = """You are a helpful assistant running in the Adapsis environment.

AVAILABLE FUNCTIONS (call via !eval):
- MusicGen.generate(context:String, caption:String, duration:Int, lyrics:String, metas:String) -> String
  Non-blocking music generation. Returns immediately. Song delivered via conversation_notify.
- llm_get_model() -> String — returns current model name
- llm_set_model(name:String) -> String — switch LLM model
- conversation_notify(context:String, message:String) -> String — notify a conversation

QUERIES (no !eval needed):
- ?library — list all modules
- ?source Module — show module source
- ?source Module.function — show function source

RULES:
- Wrap code in <code> blocks
- Put your response text BEFORE <code>
- REUSE existing functions — never write new ones for one-off tasks
- For music: just call MusicGen.generate() with the right params
- Keep responses concise"""

SCENARIOS = [
    # Music generation
    ("Generate a 30-second song about pizza.", None),
    ("Make me some ambient music, about 1 minute long.", None),
    ("Can you make a rock song with lyrics about coding?", None),
    ("Generate a short 15 second jingle.", None),
    ("Make a 5 minute lo-fi hip hop track with lyrics about relaxing.", None),
    ("Generate an instrumental jazz track, 2 minutes.", None),
    ("I want a metal song about robots, 45 seconds.", None),
    ("Create a classical piano piece, 90 seconds.", None),
    ("Make me a funk song with lyrics about dancing, 30 seconds.", None),
    ("Generate a lullaby, soft and gentle, 2 minutes.", None),
    # Module queries
    ("What modules do you have?", None),
    ("What functions does MusicGen have?", None),
    ("Show me the TelegramBot module source.", None),
    # Model management
    ("What model are you running on?", None),
    ("Switch to the gemma4-31b model.", None),
    # Simple evals
    ("What is 2 + 2?", None),
    ("Evaluate 10 * 5 + 3.", None),
    # Function creation
    ("Add a function Utils.greet(name:String) -> String that says hello.", None),
    (
        "Create a function Math.clamp(val:Int, lo:Int, hi:Int) -> Int that clamps val between lo and hi.",
        None,
    ),
    # Conversation
    ("Hello!", None),
    ("How long does music generation take?", None),
    ("I need a function that generates music and sends it to telegram.", None),
    # Error recovery - provide context of a failed attempt
    (
        "Generate a song about cats.",
        'Results:\nOK: = "Generation started"\nContinue or !done.',
    ),
    # Multi-step with feedback
    (
        "Add a greeting module.",
        "Results:\nOK: updated module Utils (1 fn added, 0 replaced)\nOK: PASS Utils.greet (1/1 cases)\nContinue or !done.",
    ),
]


def call_opus(messages):
    """Call Opus via the gateway."""
    resp = requests.post(
        f"{GATEWAY}/v1/chat/completions",
        json={
            "model": MODEL,
            "messages": messages,
            "max_tokens": 500,
            "temperature": 0.3,
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def extract_code(text):
    """Extract content from <code> blocks."""
    blocks = []
    while "<code>" in text:
        start = text.index("<code>") + 6
        end = text.find("</code>", start)
        if end == -1:
            # Unclosed block - take the rest
            blocks.append(text[start:].strip())
            break
        blocks.append(text[start:end].strip())
        text = text[end + 7 :]
    return "\n".join(blocks)


def validate_code(code):
    """Validate code through the Adapsis API. Returns (ok, message)."""
    if not code or code == "!done":
        return True, "no code"
    try:
        resp = requests.post(
            f"{ADAPSIS}/api/mutate",
            json={
                "source": "training-validate",
                "code": code,
            },
            timeout=10,
        )
        if resp.status_code == 200:
            results = resp.json().get("results", [])
            errors = [r for r in results if not r.get("ok", True)]
            if errors:
                return False, errors[0].get("message", "unknown error")
            return True, "ok"
        return False, f"HTTP {resp.status_code}"
    except Exception as e:
        return False, str(e)


def main():
    results = []
    passed = 0
    failed = 0
    skipped = 0

    for i, (user_msg, feedback) in enumerate(SCENARIOS):
        print(f"[{i + 1}/{len(SCENARIOS)}] {user_msg[:60]}...", end=" ", flush=True)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.append({"role": "user", "content": user_msg})

        if feedback:
            # This is a multi-turn: first response then feedback
            # Generate the first response
            try:
                first_resp = call_opus(messages)
            except Exception as e:
                print(f"OPUS ERROR: {e}")
                failed += 1
                continue
            messages.append({"role": "assistant", "content": first_resp})
            messages.append({"role": "user", "content": feedback})

        try:
            response = call_opus(messages)
        except Exception as e:
            print(f"OPUS ERROR: {e}")
            failed += 1
            continue

        code = extract_code(response)

        if code:
            ok, msg = validate_code(code)
            if ok:
                print(f"PASS ({len(code)} chars)")
                passed += 1
            else:
                print(f"FAIL: {msg}")
                failed += 1
                continue
        else:
            print("OK (no code)")
            skipped += 1

        # Build training example
        train_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        train_messages.append({"role": "user", "content": user_msg})
        if feedback:
            train_messages.append({"role": "assistant", "content": first_resp})
            train_messages.append({"role": "user", "content": feedback})
        train_messages.append({"role": "assistant", "content": response})

        results.append({"messages": train_messages})
        time.sleep(1)  # Rate limit

    # Write output
    with open(OUTPUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    print(f"\nDone: {passed} passed, {failed} failed, {skipped} no-code")
    print(f"Wrote {len(results)} examples to {OUTPUT}")


if __name__ == "__main__":
    main()
