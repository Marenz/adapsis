#!/usr/bin/env python3
"""Additional training scenarios - gaming, Stratum, system management."""

import json

# These get appended to the curated examples after the main generation finishes.
# They cover scenarios the bot should handle.

EXTRA_SCENARIOS = [
    # === Game mode ===
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands. shell_exec runs shell commands.",
            },
            {"role": "user", "content": "Enter game mode."},
            {
                "role": "assistant",
                "content": 'Starting game mode — launching Steam and VR client!\n\n<code>\n!eval shell_exec("steam &")\n!eval shell_exec("vhclientx86_64 &")\n!eval shell_exec("DISPLAY=:0 steam steam://open/bigpicture &")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Exit game mode."},
            {
                "role": "assistant",
                "content": 'Shutting down game mode.\n\n<code>\n!eval shell_exec("killall -9 steam || true")\n!eval shell_exec("killall -9 vhclientx86_64 || true")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Is Steam running?"},
            {
                "role": "assistant",
                "content": "Let me check.\n\n<code>\n!eval shell_exec(\"pgrep -a steam | head -5 || echo 'Steam is not running'\")\n</code>",
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Show the Steam window on the monitor."},
            {
                "role": "assistant",
                "content": 'Switching Steam to the monitor display.\n\n<code>\n!eval shell_exec("DISPLAY=:0 steam steam://open/bigpicture &")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "We're done gaming, shut everything down."},
            {
                "role": "assistant",
                "content": "Closing game mode — stopping Steam and VR client.\n\n<code>\n!eval shell_exec(\"killall steam vhclientx86_64 2>/dev/null; echo 'Game mode stopped'\")\n</code>",
            },
        ]
    },
    # === Stratum usage ===
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands. Stratum module provides graph database operations.",
            },
            {"role": "user", "content": "Remember that my favorite color is blue."},
            {
                "role": "assistant",
                "content": 'I\'ll save that to memory.\n\n<code>\n!eval Stratum.store("preference", "favorite_color", "blue")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "What's my favorite color?"},
            {
                "role": "assistant",
                "content": 'Let me check my memory.\n\n<code>\n!eval Stratum.recall("preference", "favorite_color")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {
                "role": "user",
                "content": "Store a note: meeting with Alex tomorrow at 3pm.",
            },
            {
                "role": "assistant",
                "content": 'Saving that note.\n\n<code>\n!eval Stratum.store("note", "meeting_alex", "Meeting with Alex tomorrow at 3pm")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "What notes do I have?"},
            {
                "role": "assistant",
                "content": 'Let me look up your notes.\n\n<code>\n!eval Stratum.list("note")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Delete the meeting note."},
            {
                "role": "assistant",
                "content": 'Removing that note.\n\n<code>\n!eval Stratum.delete("note", "meeting_alex")\n</code>',
            },
        ]
    },
    # === System management ===
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "How much GPU memory is being used?"},
            {
                "role": "assistant",
                "content": 'Let me check.\n\n<code>\n!eval shell_exec("nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "What processes are using the GPU?"},
            {
                "role": "assistant",
                "content": '<code>\n!eval shell_exec("nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Restart the music generation daemon."},
            {
                "role": "assistant",
                "content": "Restarting ace-step daemon.\n\n<code>\n!eval shell_exec(\"systemctl --user restart ace-step-gen && echo 'Restarted'\")\n</code>",
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "What's the system uptime?"},
            {
                "role": "assistant",
                "content": '<code>\n!eval shell_exec("uptime")\n</code>',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Check disk space."},
            {
                "role": "assistant",
                "content": '<code>\n!eval shell_exec("df -h / /home | head -5")\n</code>',
            },
        ]
    },
    # === Roadmap / Planning ===
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Add 'implement weather API' to the roadmap."},
            {
                "role": "assistant",
                "content": "Adding to the roadmap.\n\n<code>\n!roadmap add implement weather API\n</code>",
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Show me the roadmap."},
            {"role": "assistant", "content": "<code>\n!roadmap show\n</code>"},
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. Use <code> blocks for commands.",
            },
            {"role": "user", "content": "Mark roadmap item 1 as done."},
            {"role": "assistant", "content": "<code>\n!roadmap done 1\n</code>"},
        ]
    },
    # === Anti-patterns (teach NOT to do these) ===
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. REUSE existing functions. Never create new modules for one-off tasks.",
            },
            {
                "role": "user",
                "content": "Write me a function to send a telegram message.",
            },
            {
                "role": "assistant",
                "content": 'That already exists! TelegramBot.send_message(chat_id, text) handles sending messages. Just call it:\n\n<code>\n!eval TelegramBot.send_message(1815217, "Hello from Adapsis!")\n</code>\n\nNo need to write a new function.',
            },
        ]
    },
    {
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant running in the Adapsis environment. REUSE existing functions. Never create new modules for one-off tasks.",
            },
            {"role": "user", "content": "Create a SongGenerator module."},
            {
                "role": "assistant",
                "content": 'We already have MusicGen which handles all music generation. No need for a new module. To generate a song:\n\n<code>\n!eval MusicGen.generate("telegram:1815217", "your song description", 30, "", "genre tags")\n</code>',
            },
        ]
    },
]

# Append to existing training data
output = "/home/marenz/.config/adapsis/training/curated-examples.jsonl"
with open(output, "a") as f:
    for example in EXTRA_SCENARIOS:
        f.write(json.dumps(example) + "\n")

print(f"Appended {len(EXTRA_SCENARIOS)} examples to {output}")
