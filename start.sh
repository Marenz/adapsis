#!/bin/bash
cd /home/marenz/Projects/adapsis-gpt-run
# This is the self-improving dev loop, so it explicitly opts in to !opencode
# via --access-level full. Deployments that should NOT rewrite their own
# runtime (e.g. a family member's machine) must omit this flag — the default
# is now adapsis-only, which disables !opencode.
exec ./target/release/adapsis os   --port 3002   --url http://127.0.0.1:4000   --model mimo/mimo-v2-pro   --session adapsis-gpt.json   --log-file adapsis-mimo.log   --training-log /home/marenz/.config/adapsis/training/mimo.jsonl   --max-iterations 100   --access-level full   --opencode-git-dir /home/marenz/Projects/adapsis-gpt-run   --autonomous "Check !roadmap for your tasks. Pick the first undone item, create a !plan, and start working. Write ALL app logic in Adapsis. Use !opencode only for runtime bugs or elegant language enhancements. Do NOT create debugging/verification modules."   2>> adapsis-mimo-stderr.log
