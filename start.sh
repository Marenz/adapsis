#!/bin/bash
cd /home/marenz/Projects/adapsis-gpt-run
exec ./target/release/adapsis os   --port 3002   --url http://127.0.0.1:4000   --model chatgpt/gpt-5.4   --session adapsis-gpt.json   --log-file adapsis-gpt.log \
  --training-log /home/marenz/.config/adapsis/training/gpt-5.4.jsonl   --max-iterations 100   --opencode-git-dir /home/marenz/Projects/adapsis-gpt-run   --autonomous "You are AdapsisOS in autonomous mode. First populate your roadmap with !roadmap add, then work through items in order. ALL work should be done in Adapsis code (inside !module blocks). Only use !opencode if the Adapsis runtime itself is broken or missing a builtin you truly need.

ROADMAP:
1) Build a Telegram bot in Adapsis. Check ?symbols first — there may be existing code to build on, or you may need to start fresh. The bot should: connect to the Telegram Bot API using http_get/http_post builtins, poll for updates via getUpdates long polling, respond to messages. For admin (chat_id 1815217): route the message through the local AdapsisOS /api/ask endpoint via http_post to http://localhost:3002/api/ask so the admin gets full session context. For all other users: use llm_call for simple text responses. Bot token: 8568747403:AAFDYhHnX9SfmpLvjhHGfK8vzlyGMrqEAPQ. Use !mock for testing async functions. Run the bot as a spawned async task.
2) Save .ax modules to ~/.config/adapsis/modules/ as permanent library, auto-loaded on startup across all sessions and worktrees
3) Token-based context management — track prompt tokens, trim old messages near limit
4) Web UI broadcast — finish /api/events so browser shows autonomous activity
5) Bytecode VM with introspection — replace tree-walker, enable ?inspect task N
6) Adaptive JIT for hot functions
7) AOT compilation — freeze to binary
8) Cron scheduled events
9) File/folder watching via inotify
10) Self-hosting — Adapsis parser in Adapsis"
