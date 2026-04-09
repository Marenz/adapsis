#!/usr/bin/env python3
"""Merge all training data into a single file for fine-tuning."""

import json
import glob
import os

TRAINING_DIR = os.path.expanduser("~/.config/adapsis/training")
OUTPUT = os.path.join(TRAINING_DIR, "combined.jsonl")

examples = []
seen = set()

for path in sorted(glob.glob(os.path.join(TRAINING_DIR, "*.jsonl"))):
    if path == OUTPUT:
        continue
    count = 0
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            # Dedup by user message
            user_msgs = [m["content"] for m in d["messages"] if m["role"] == "user"]
            key = "|".join(user_msgs)
            if key not in seen:
                seen.add(key)
                examples.append(d)
                count += 1
    print(f"  {os.path.basename(path)}: {count} unique examples")

with open(OUTPUT, "w") as f:
    for ex in examples:
        f.write(json.dumps(ex) + "\n")

print(f"\nTotal: {len(examples)} unique training examples → {OUTPUT}")
