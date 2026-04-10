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
    # === Music generation (various styles, durations, with/without lyrics) ===
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
    ("EDM banger, 3 minutes, heavy bass.", None),
    ("Country song about a truck, 1 minute, with lyrics.", None),
    ("Reggae vibes, sunny day, 45 seconds.", None),
    ("Epic orchestral soundtrack, 2 minutes.", None),
    ("Synthwave retro 80s track, 1 minute.", None),
    ("Blues guitar solo, 30 seconds.", None),
    ("Bossa nova, smooth, 90 seconds.", None),
    ("Drum and bass, high energy, 1 minute.", None),
    ("Acoustic folk song about nature, 2 minutes, with lyrics.", None),
    ("Make a birthday song for my friend Alex, 30 seconds.", None),
    ("Generate background music for a podcast, calm, 3 minutes.", None),
    ("Trap beat with 808s, 1 minute.", None),
    ("K-pop style, upbeat, 45 seconds.", None),
    ("Meditation music, very slow and peaceful, 5 minutes.", None),
    ("Video game boss battle music, intense, 1 minute.", None),
    ("Wedding march style, elegant, 30 seconds.", None),
    ("Punk rock, fast and angry, 30 seconds, with lyrics about rebellion.", None),
    ("R&B smooth groove, 2 minutes.", None),
    ("Cinematic suspense music, 1 minute.", None),
    ("Happy ukulele song, 30 seconds.", None),
    # === Music with specific lyrics ===
    (
        "Make a song with these lyrics: 'Hello world, this is my first song, coding all day long'",
        None,
    ),
    ("Generate a rap about artificial intelligence, 1 minute.", None),
    ("Sing about the ocean, slow ballad, 2 minutes.", None),
    # === Module queries ===
    ("What modules do you have?", None),
    ("What functions does MusicGen have?", None),
    ("Show me the TelegramBot module source.", None),
    ("List all available functions.", None),
    ("What does the Memory module do?", None),
    ("Show me the source of MusicGen.generate.", None),
    # === Model management ===
    ("What model are you running on?", None),
    ("Switch to the gemma4-31b model.", None),
    ("Switch to gemma4s.", None),
    ("What LLM models are available?", None),
    # === Simple evals ===
    ("What is 2 + 2?", None),
    ("Evaluate 10 * 5 + 3.", None),
    ("Calculate 100 / 4.", None),
    ("What is the length of the string 'hello world'?", None),
    # === Function creation ===
    ("Add a function Utils.greet(name:String) -> String that says hello.", None),
    (
        "Create a function Math.clamp(val:Int, lo:Int, hi:Int) -> Int that clamps val between lo and hi.",
        None,
    ),
    (
        "Write a function StringUtils.reverse(s:String) -> String that reverses a string.",
        None,
    ),
    (
        "Add a function Utils.repeat(s:String, n:Int) -> String that repeats s n times.",
        None,
    ),
    ("Create a function Math.factorial(n:Int) -> Int.", None),
    ("Write a function Utils.is_palindrome(s:String) -> Bool.", None),
    ("Add Utils.max_of_three(a:Int, b:Int, c:Int) -> Int.", None),
    ("Create a function that converts celsius to fahrenheit.", None),
    # === Function with tests ===
    (
        "Add Utils.capitalize(s:String) -> String that capitalizes the first letter. Include tests.",
        None,
    ),
    ("Write Math.is_even(n:Int) -> Bool with tests for 0, 1, 2, -1.", None),
    # === Error handling ===
    (
        "Write a function that reads a file and returns its content, handling errors.",
        None,
    ),
    ("Create a function that makes an HTTP GET request with error handling.", None),
    # === Conversation ===
    ("Hello!", None),
    ("How long does music generation take?", None),
    ("I need a function that generates music and sends it to telegram.", None),
    ("What can you do?", None),
    ("Help me understand how Adapsis works.", None),
    ("What's the difference between !eval and +fn?", None),
    ("How do I create a new module?", None),
    ("Can you explain the effect system?", None),
    ("What does [io,async] mean?", None),
    ("How do I write tests in Adapsis?", None),
    # === Multi-turn: success feedback ===
    (
        "Generate a song about cats.",
        'Results:\nOK: = "Generation started"\nContinue or !done.',
    ),
    (
        "Add a greeting module.",
        "Results:\nOK: updated module Utils (1 fn added, 0 replaced)\nOK: PASS Utils.greet (1/1 cases)\nContinue or !done.",
    ),
    ("What model am I on?", 'Results:\nOK: = "gemma4s"\nContinue or !done.'),
    # === Multi-turn: error recovery ===
    (
        "Add a reverse function.",
        "Errors:\nERROR: Parse error: line 2: expected `:` after binding name\nFix and continue.",
    ),
    (
        "Generate music about space.",
        "Errors:\nERROR: eval error: function 'MusicGen.generate' untested\nFix and continue.",
    ),
    # === Existing function reuse (anti-pattern correction) ===
    ("I need a function that generates music and sends it to telegram.", None),
    ("Write me a wrapper for sending telegram messages.", None),
    ("Create a music generation endpoint.", None),
    # === +doc usage ===
    ("Add documentation to the MusicGen module.", None),
    ("Document the Utils.greet function.", None),
    # === Roadmap/plan ===
    ("Add 'implement weather API' to the roadmap.", None),
    ("Show me the current roadmap.", None),
    ("What's the plan?", None),
    # === Shared vars ===
    ("What shared variables does TelegramBot have?", None),
    # === Complex multi-step ===
    (
        "Check what modules exist, then add a hello function to Utils if it doesn't exist.",
        None,
    ),
    ("Generate a song, then tell me what model you're running on.", None),
    # === Batch 2: More music variations ===
    ("Chiptune 8-bit game music, 30 seconds.", None),
    ("Gregorian chant style, 2 minutes.", None),
    ("Flamenco guitar, passionate, 1 minute.", None),
    ("Dubstep with heavy bass drops, 45 seconds.", None),
    ("Swing jazz, 1920s style, 90 seconds.", None),
    ("Afrobeat groove, 2 minutes.", None),
    ("Celtic folk with fiddle, 1 minute.", None),
    ("Psychedelic rock, spacey, 3 minutes.", None),
    ("Opera aria style, dramatic, 2 minutes.", None),
    ("Garage rock, raw and dirty, 30 seconds.", None),
    ("Tropical house, summer vibes, 1 minute.", None),
    ("Grunge, 90s style, distorted guitar, 45 seconds.", None),
    ("Bollywood dance music, 1 minute.", None),
    ("Acid techno, 303 bass, 2 minutes.", None),
    ("New age meditation, crystal bowls, 5 minutes.", None),
    # === Batch 2: More function creation ===
    ("Write a function that checks if a number is prime.", None),
    ("Create a function to find the longest word in a sentence.", None),
    ("Add a fibonacci function.", None),
    ("Write a function that counts vowels in a string.", None),
    ("Create a function that removes duplicates from a list.", None),
    ("Add a binary search function.", None),
    ("Write a function to check if a string is a valid email format.", None),
    ("Create a ROT13 cipher function.", None),
    ("Add a function that formats a number with commas (1000 -> 1,000).", None),
    ("Write a function that calculates the area of a circle given radius.", None),
    # === Batch 2: More system queries ===
    ("What time is it?", None),
    ("Show running services.", None),
    ("Check if the music daemon is running.", None),
    ("How much RAM is available?", None),
    ("What's my IP address?", None),
    ("Show recent log entries.", None),
    ("List files in my home directory.", None),
    ("Check if port 8080 is in use.", None),
    # === Batch 2: More Stratum/Memory ===
    ("Remember that I like dark roast coffee.", None),
    ("Store the fact that Earth is 4.5 billion years old.", None),
    ("What do you remember about my preferences?", None),
    ("Create a topic called 'Travel Plans'.", None),
    ("Find all memories related to music.", None),
    ("Delete the note about the meeting.", None),
    ("How many nodes are in the graph?", None),
    ("Show me all edge types.", None),
    ("Link my guitar practice memory to the music production topic.", None),
    ("Search for anything about Python.", None),
    # === Batch 2: More error recovery ===
    (
        "Write a greet function.",
        "Errors:\nERROR: function must be inside a module. Use: +module Name\nFix and continue.",
    ),
    (
        "Add a web scraper.",
        "Errors:\nERROR: module `WebScraper` is frozen — mutations are rejected.\nFix and continue.",
    ),
    (
        "Evaluate my_func().",
        "Errors:\nERROR: function 'my_func' not found\nFix and continue.",
    ),
    (
        "Run the tests.",
        "Results:\nOK: PASS Utils.double (3/3 cases)\nOK: PASS Utils.greet (1/1 cases)\nContinue or !done.",
    ),
    # === Batch 2: Telegram-specific ===
    ("Send a photo to my chat.", None),
    ("Who sent the last message?", None),
    ("How many messages are in my Telegram chat?", None),
    # === Batch 2: More !opencode ===
    ("Add a timer builtin that fires a callback after N seconds.", None),
    ("The type checker is too strict with generics. Can you relax it?", None),
    ("Add support for string interpolation like f-strings.", None),
    ("Make the web UI mobile-friendly.", None),
    # === Batch 2: GitHub ===
    ("What GitHub issues are open?", None),
    ("Show me issue #10.", None),
    ("Sync issues to the roadmap.", None),
    ("What are the comments on issue #25?", None),
    # === Batch 2: Multi-turn conversations ===
    (
        "Make a song about rain.",
        'Results:\nOK: = "Generation started"\nContinue or !done.',
    ),
    (
        "What functions does Utils have?",
        "Results:\nOK: Utils has 3 functions: double, greet, join_strings\nContinue or !done.",
    ),
    (
        "Switch to gemma4-31b.",
        'Results:\nOK: = "model set to: gemma4-31b"\nContinue or !done.',
    ),
    (
        "Add a capitalize function and test it.",
        "Results:\nOK: updated module Utils (1 fn added)\nOK: PASS Utils.capitalize (2/2 cases)\nContinue or !done.",
    ),
    # === Batch 2: Edge cases ===
    ("", None),  # Empty message
    ("Just say hello, no code needed.", None),
    ("What's 0 divided by 0?", None),
    ("Can you write Rust code?", None),
    ("Tell me a joke.", None),
    ("What programming languages do you know?", None),
    # === Batch 2: Complex tasks ===
    (
        "Read the MusicGen source, summarize what it does, then add a function to list all generated songs.",
        None,
    ),
    (
        "Check what model I'm on, switch to gemma4s, generate a test song, then switch back.",
        None,
    ),
    (
        "Create a Utils module with at least 5 math functions and test all of them.",
        None,
    ),
    (
        "Show me the Stratum module, explain the recall function, then use it to search for 'coding'.",
        None,
    ),
    # === Batch 2: Game mode ===
    ("Start game mode.", None),
    ("Stop game mode.", None),
    ("Is the VR client running?", None),
    ("Show Steam on the main display.", None),
    # === Batch 2: Module management ===
    ("List all modules.", None),
    ("How many functions does Stratum have?", None),
    ("Is MusicGen frozen?", None),
    ("Show me the TelegramBot webhook handler.", None),
    ("What routes are registered?", None),
    # === Batch 2: Model switching ===
    ("Use the big model for this.", None),
    ("Switch to a smaller model to save VRAM.", None),
    ("What models are available?", None),
    ("Which model is best for coding?", None),
    # === Batch 2: Reminders/habits ===
    ("I need to drink more water. Help me remember.", None),
    ("Remind me to exercise every morning.", None),
    ("What habits am I tracking?", None),
    ("I meditated today. Log it.", None),
    ("Show my progress on quitting smoking.", None),
    ("Add a creative writing goal: write 500 words per day.", None),
    ("What goals have I set?", None),
    ("I practiced guitar for 30 minutes today.", None),
    # === Batch 2: File operations ===
    ("Read the Adapsis config file.", None),
    ("What modules are in ~/.config/adapsis/modules/?", None),
    ("Show me the systemd service file for adapsis.", None),
    ("How big is the training data file?", None),
    # === Batch 2: Debugging/status ===
    ("Something is wrong with the music generation. Debug it.", None),
    ("Why is the bot not responding on Telegram?", None),
    ("Show me recent error logs.", None),
    ("Is the webhook endpoint working?", None),
    ("Check if Caddy is running.", None),
    ("What version of llama.cpp are we using?", None),
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
