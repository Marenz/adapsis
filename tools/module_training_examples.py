#!/usr/bin/env python3
"""Generate module-specific training examples using actual Adapsis module APIs.

These are hand-crafted examples that match the real function signatures.
No Opus needed - we know the correct syntax.
"""

import json

OUTPUT = "/home/marenz/.config/adapsis/training/module-examples.jsonl"

SYS = "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands. Put response text BEFORE code blocks."

examples = []


def add(user, assistant):
    examples.append(
        {
            "messages": [
                {"role": "system", "content": SYS},
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]
        }
    )


# ═══ MusicGen ═══
add(
    "Generate a 30-second song about pizza and send it to me.",
    'Generating your pizza song now!\n\n<code>\n!eval MusicGen.generate("telegram:1815217", "fun song about pizza, catchy pop melody", 30, "", "pop, fun, catchy")\n</code>',
)

add(
    "Make me a rock anthem with lyrics, 1 minute.",
    'Creating a 1-minute rock anthem with lyrics!\n\n<code>\n!eval MusicGen.generate("telegram:1815217", "powerful rock anthem, electric guitar, big drums", 60, "[verse]\\nWe rise up from the ashes\\nWe never back down\\nWith fire in our hearts\\nWe own this town\\n\\n[chorus]\\nWe are the champions tonight\\nWe burn so bright\\nNothing can stop us now", "rock, anthem, powerful, electric guitar, drums, 140bpm")\n</code>',
)

add(
    "Make ambient music, 3 minutes, no lyrics.",
    'Generating a 3-minute ambient piece!\n\n<code>\n!eval MusicGen.generate("telegram:1815217", "ambient atmospheric soundscape, gentle pads, reverb, space", 180, "", "ambient, atmospheric, ethereal, calm, pads")\n</code>',
)

add(
    "How long will the song take to generate?",
    "Music generation is very fast — about 2 seconds for a 30-second song, and about 16 seconds for a 5-minute song. You'll receive the audio automatically when it's done.",
)

add(
    "Generate a song but I don't want it on Telegram.",
    'Currently MusicGen.generate() delivers to a conversation context. If you tell me which context to use, I can send it there. For Telegram, use the context name like "telegram:CHAT_ID".',
)

# ═══ Stratum (graph DB) ═══
add(
    "Remember that my birthday is March 15.",
    'Storing that in memory.\n\n<code>\n!eval Stratum.remember("birthday", "User birthday is March 15")\n</code>',
)

add(
    "What do you remember about me?",
    'Let me search my memory.\n\n<code>\n!eval Stratum.recall("user preferences")\n</code>',
)

add(
    "Store a fact: Python was created by Guido van Rossum.",
    'Saving that fact.\n\n<code>\n!eval Stratum.remember_fact("Python creator", "Python was created by Guido van Rossum", "programming")\n</code>',
)

add(
    "Search your memory for anything about coding.",
    'Searching memory.\n\n<code>\n!eval Stratum.search("coding", 10)\n</code>',
)

add(
    "Create a topic node called 'Project Ideas'.",
    'Creating the topic.\n\n<code>\n!eval Stratum.create_topic("Project Ideas", "Collection of project ideas and brainstorming")\n</code>',
)

add(
    "Find all people you know about.",
    '<code>\n!eval Stratum.find_nodes("Person", "", 20)\n</code>',
)

add(
    "What types of nodes exist in the graph?",
    "<code>\n!eval Stratum.get_node_types()\n</code>",
)

add("Show me the graph stats.", '<code>\n!eval Stratum.get_stats("")\n</code>')

add(
    "Remember that Alex prefers dark mode.",
    '<code>\n!eval Stratum.create_person("Alex", "friend", "prefers dark mode")\n</code>',
)

add(
    "Link the node abc-123 to topic 'AI Research'.",
    '<code>\n!eval Stratum.link_to_topic("abc-123", "AI Research")\n</code>',
)

# ═══ Memory (high-level Stratum wrapper) ═══
add(
    "Store a memory about today's productive session.",
    '<code>\n!eval Memory.store_memory("Productive session", "Had a very productive coding session working on Adapsis features", "development")\n</code>',
)

add("What facts do you know?", "<code>\n!eval Memory.list_facts()\n</code>")

add("What topics are tracked?", "<code>\n!eval Memory.list_topics()\n</code>")

add(
    "Tell me everything about the 'music' topic.",
    '<code>\n!eval Memory.about_topic("music")\n</code>',
)

add(
    "Get context for: how does the telegram bot work?",
    '<code>\n!eval Memory.get_context("how does the telegram bot work")\n</code>',
)

# ═══ TelegramBot ═══
add(
    "Send a message to my Telegram chat saying 'hello'.",
    '<code>\n!eval TelegramBot.send_message(1815217, "hello")\n</code>',
)

add(
    "What's the bot's token?",
    "I can't share the bot token directly for security, but it's loaded from /home/marenz/.telegram-bot-token on startup via the +startup block.",
)

# ═══ GithubSync ═══
add(
    "Fetch the latest GitHub issues.",
    "<code>\n!eval GithubSync.fetch_issues()\n</code>",
)

add(
    "Sync GitHub issues to the roadmap.",
    "<code>\n!eval GithubSync.sync_to_roadmap()\n</code>",
)

# ═══ IssueReader ═══
add("Read GitHub issue #10.", "<code>\n!eval IssueReader.fetch_issue(10)\n</code>")

add(
    "Get the comments on issue #25.",
    "<code>\n!eval IssueReader.fetch_comments(25)\n</code>",
)

# ═══ System / Shell ═══
add(
    "Enter game mode.",
    'Starting game mode — launching Steam and VR!\n\n<code>\n!eval shell_exec("steam &")\n!eval shell_exec("vhclientx86_64 &")\n!eval shell_exec("DISPLAY=:0 steam steam://open/bigpicture &")\n</code>',
)

add(
    "Exit game mode.",
    'Shutting down game mode.\n\n<code>\n!eval shell_exec("killall steam vhclientx86_64 2>/dev/null; echo done")\n</code>',
)

add(
    "Is Steam running?",
    '<code>\n!eval shell_exec("pgrep -a steam | head -5 || echo Steam is not running")\n</code>',
)

add(
    "How much GPU memory is free?",
    '<code>\n!eval shell_exec("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader")\n</code>',
)

add(
    "What processes are on the GPU?",
    '<code>\n!eval shell_exec("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader")\n</code>',
)

add(
    "Restart the music daemon.",
    '<code>\n!eval shell_exec("systemctl --user restart ace-step-gen && echo Restarted")\n</code>',
)

add("Check disk space.", '<code>\n!eval shell_exec("df -h / /home | head -5")\n</code>')

add("What's the system uptime?", '<code>\n!eval shell_exec("uptime")\n</code>')

add("Who is logged in via SSH?", '<code>\n!eval shell_exec("who")\n</code>')

# ═══ Model management ═══
add(
    "What model are you using?",
    "Let me check.\n\n<code>\n!eval llm_get_model()\n</code>",
)

add("Switch to gemma4-31b.", '<code>\n!eval llm_set_model("gemma4-31b")\n</code>')

add("Switch to the small model.", '<code>\n!eval llm_set_model("gemma4s")\n</code>')

# ═══ Module/code queries ═══
add("What modules exist?", "<code>\n?library\n</code>")

add("Show me the MusicGen source.", "<code>\n?source MusicGen\n</code>")

add("What functions does Stratum have?", "<code>\n?source Stratum\n</code>")

add(
    "Show me the send_message function.",
    "<code>\n?source TelegramBot.send_message\n</code>",
)

# ═══ Function creation ═══
add(
    "Add a function Utils.double(n:Int) -> Int that doubles a number.",
    '<code>\n+module Utils\n+fn double(n:Int) -> Int\n  +return n * 2\n+end\n+doc "Double an integer"\n+test double\n  +with n=5 -> expect 10\n  +with n=0 -> expect 0\n  +with n=-3 -> expect -6\n</code>',
)

add(
    "Create Utils.join_strings(items:List<String>, sep:String) -> String.",
    '<code>\n+module Utils\n+fn join_strings(items:List<String>, sep:String) -> String\n  +let result:String = ""\n  +let i:Int = 0\n  +each item in items\n    +if i > 0\n      +let result:String = concat(result, sep)\n    +end\n    +let result:String = concat(result, item)\n    +let i:Int = i + 1\n  +end\n  +return result\n+end\n+doc "Join a list of strings with a separator"\n+test join_strings\n  +with items=["a","b","c"] sep="," -> expect "a,b,c"\n  +with items=["hello"] sep=" " -> expect "hello"\n</code>',
)

# ═══ Roadmap ═══
add(
    "Add 'implement weather module' to the roadmap.",
    "<code>\n!roadmap add implement weather module\n</code>",
)

add("Show the roadmap.", "<code>\n!roadmap show\n</code>")

add("Mark item 1 as done.", "<code>\n!roadmap done 1\n</code>")

# ═══ Anti-patterns ═══
add(
    "Write a function to generate and send music.",
    'MusicGen.generate() already handles everything — generation and delivery. Just call:\n\n<code>\n!eval MusicGen.generate("telegram:1815217", "your description", 30, "", "genre tags")\n</code>\n\nNo need to create a new function.',
)

add(
    "Create a TelegramSender module.",
    "TelegramBot already has send_message, send_reply, and send_audio_file functions. No need for a new module.",
)

add(
    "Make a new module called AudioGen.",
    "MusicGen already handles all audio generation. Use MusicGen.generate() instead of creating a new module.",
)

# ═══ Conversations ═══
add(
    "Hello!",
    "Hello! How can I help you? I can generate music, manage memories, check system status, and write code.",
)

add("Thanks!", "You're welcome! Let me know if you need anything else.")

add(
    "What can you do?",
    "I can:\n- **Generate music** — any genre, with or without lyrics\n- **Remember things** — store facts, notes, preferences\n- **Manage code** — create functions, modules, run tests\n- **System admin** — check GPU, disk, restart services\n- **Game mode** — start/stop Steam and VR\n- **GitHub** — fetch issues, sync roadmap",
)

# ═══ !opencode — Rust-level modifications ═══
add(
    "Add a new IO builtin called http_patch that works like http_post but uses PATCH method.",
    "This requires a Rust-level change. Let me dispatch it via !opencode.\n\n<code>\n!opencode Add a new IO builtin http_patch(url, body, content_type) that works like http_post but uses the PATCH HTTP method. Add the IoRequest variant, handler in coroutine.rs, builtin dispatch, and register in builtins.rs. Update prompt.rs with documentation.\n</code>",
)

add(
    "The parser doesn't handle multi-line strings. Can you fix that?",
    'That needs a parser change in Rust. Dispatching via !opencode.\n\n<code>\n!opencode Fix the parser to handle multi-line strings delimited by triple quotes ("""..."""). Update parser.rs to recognize triple-quote delimiters and collect lines until the closing triple-quote. Add tests in the parser test module.\n</code>',
)

add(
    "Add a sleep builtin so functions can pause execution.",
    'Adding a new IO builtin requires Rust changes.\n\n<code>\n!opencode Add a sleep(milliseconds:Int) IO builtin. Add IoRequest::Sleep variant, handle it with tokio::time::sleep in the coroutine runtime, add builtin dispatch for "sleep", and register in builtins.rs. Update prompt.rs.\n</code>',
)

add(
    "Can you make the web UI show conversation history?",
    "The web UI is in web/adapsis.html and needs Rust API changes. Dispatching via !opencode.\n\n<code>\n!opencode Add a GET /api/conversations endpoint that returns the list of conversation contexts with their message counts. Update web/adapsis.html to display conversation history in a collapsible panel.\n</code>",
)

add(
    "Add support for importing modules from URLs.",
    "<code>\n!opencode Add a !import command that downloads an .ax file from a URL and loads it as a module. Add the parser operation, implement HTTP fetch in the library loader, validate the downloaded code before applying. Add error handling for network failures and invalid modules.\n</code>",
)

add(
    "The error messages are terrible. Can you improve them?",
    "Error message improvement is a Rust-level task.\n\n<code>\n!opencode Improve error messages across the codebase: 1) Parser errors should include the line content and a caret pointing to the error position. 2) Eval errors should include the function name and module. 3) Type errors should show expected vs actual types. Focus on parser.rs and eval/mod.rs.\n</code>",
)

add(
    "Make !eval support async functions without needing +await.",
    "<code>\n!opencode Make !eval automatically detect if a function has [io,async] effects and use the async eval path (with coroutine handle) without requiring the user to write +await. This should only apply to !eval, not to function bodies where +await is still required for clarity.\n</code>",
)

add(
    "Add a websocket endpoint for real-time events.",
    "<code>\n!opencode Add a WebSocket endpoint at /ws that broadcasts the same events as /api/events SSE but over WebSocket. Use axum::extract::ws. Clients can subscribe to specific event types via a JSON message.\n</code>",
)

# ═══ Memory / Reminders ═══
add(
    "Remember that I should practice guitar every day.",
    'Storing that reminder.\n\n<code>\n!eval Memory.store_memory("daily guitar practice", "User wants to practice guitar every day as a habit", "habits")\n</code>',
)

add(
    "Remind me to stop smoking.",
    'I will remember that.\n\n<code>\n!eval Memory.store_memory("quit smoking", "User wants to quit smoking cannabis - important health goal", "habits")\n</code>',
)

add(
    "I want to write fiction more often. Help me remember.",
    '<code>\n!eval Memory.store_memory("practice fiction writing", "User wants to practice creative/fiction writing regularly", "habits")\n</code>',
)

add(
    "What habits should I be working on?",
    'Let me check your stored habits.\n\n<code>\n!eval Memory.recall_for("habits goals practice remind")\n</code>',
)

add(
    "Store a decision: I chose Rust over Go for the new project.",
    '<code>\n!eval Memory.store_decision("language choice", "Chose Rust over Go for the new project", "Rust has better performance, memory safety, and the team has experience with it", "projects")\n</code>',
)

add(
    "What do you remember about my projects?",
    '<code>\n!eval Memory.about_topic("projects")\n</code>',
)

add(
    "Remember that Alex's birthday is on March 15.",
    '<code>\n!eval Memory.store_fact("Alex birthday", "Alex birthday is March 15", "people", "friends")\n</code>',
)

add(
    "What facts do you know about my friends?",
    '<code>\n!eval Memory.recall_for("friends people birthday")\n</code>',
)

add(
    "Create a topic for tracking my music production learning.",
    '<code>\n!eval Stratum.create_topic("Music Production", "Learning music production - DAW, mixing, mastering, composition")\n</code>',
)

add(
    "I just learned about sidechain compression. Save that.",
    '<code>\n!eval Memory.store_memory("sidechain compression", "Learned about sidechain compression technique for mixing - uses a compressor triggered by another track (usually kick) to create pumping effect", "music production")\n</code>',
)

add(
    "What have I learned about music production?",
    '<code>\n!eval Memory.about_topic("music production")\n</code>',
)

add(
    "Remember to create my own song using the guitar.",
    '<code>\n!eval Memory.store_memory("create original guitar song", "User wants to create their own original song using guitar - a personal creative goal", "creative goals")\n</code>',
)

add("List all my topics.", "<code>\n!eval Memory.list_topics()\n</code>")

add("What are all my stored facts?", "<code>\n!eval Memory.list_facts()\n</code>")

# Write output
with open(OUTPUT, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"Wrote {len(examples)} module-specific examples to {OUTPUT}")
