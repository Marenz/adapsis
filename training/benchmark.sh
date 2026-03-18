#!/bin/bash
# Forge Model Benchmark
# Tests each model on standardized tasks and collects results.

set -e

FORGE="cargo run -q --"
RESULTS_DIR="$(dirname "$0")/benchmark_results"
mkdir -p "$RESULTS_DIR"

MODELS=(
  "Qwen3.5-4B-Forge-LoRA-BF16.gguf|Qwen3.5-4B-LoRA|65536"
  "Qwen3.5-9B-Q4_K_M.gguf|Qwen3.5-9B|65536"
  "Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf|Qwen3.5-35B-A3B|65536"
  "gemma-3-4b-it-Q4_K_M.gguf|Gemma3-4B|8192"
  "Phi-4-mini-instruct-Q4_K_M.gguf|Phi4-mini-3.8B|16384"
  "Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf|Llama3.1-8B|8192"
)

TASKS=(
  "T1_add|Write a function called 'add' that takes two Int parameters 'a' and 'b' and returns their sum as Int. Include tests for positive numbers, negative numbers, and zero."
  "T2_validate|Define a User type with fields name:String and age:Int. Write validate_user that checks name not empty and age >= 0 and age <= 150. Returns Result<User> with [fail]. Include tests."
  "T3_is_even|Write a function 'is_even' that checks if an Int is even using the modulo operator (%). Include tests."
  "T4_multi_fn|Write two functions: 'square' that squares an Int, and 'sum_of_squares' that takes two Ints and returns the sum of their squares using the square function. Include tests for both."
)

MAX_ITER=5

start_model() {
  local gguf=$1
  local ctx=$2
  
  systemctl --user stop llama-server.service 2>/dev/null || true
  sleep 2
  
  # Temporarily update service
  cat > /tmp/forge-bench-override.conf << EOF
[Service]
ExecStart=
ExecStart=/usr/bin/podman run --rm --name llama-server \
  --device nvidia.com/gpu=all \
  -v /home/marenz/models:/models:ro \
  -p 0.0.0.0:8081:8081 \
  llama-server-cuda \
  -m /models/$gguf \
  -ngl 99 -c $ctx -fa on \
  --cache-type-k q8_0 --cache-type-v q8_0 \
  -np 1 --host 0.0.0.0 --port 8081
EOF

  mkdir -p ~/.config/systemd/user/llama-server.service.d/
  cp /tmp/forge-bench-override.conf ~/.config/systemd/user/llama-server.service.d/override.conf
  systemctl --user daemon-reload
  systemctl --user start llama-server.service
  
  # Wait for ready
  for i in $(seq 1 60); do
    if curl -s http://127.0.0.1:8081/health 2>/dev/null | grep -q "ok"; then
      return 0
    fi
    sleep 3
  done
  echo "TIMEOUT waiting for model $gguf"
  return 1
}

run_task() {
  local task_id=$1
  local task_desc=$2
  local model_name=$3
  local outfile="$RESULTS_DIR/${model_name}_${task_id}.txt"
  
  timeout 180 $FORGE run --task "$task_desc" --max-iterations $MAX_ITER 2>&1 | \
    grep -E "Iteration|OK:|ERROR:|PASS|FAIL|Model signals|Parse error|TYPE WARNING" \
    > "$outfile" 2>&1 || echo "TIMEOUT" >> "$outfile"
}

echo "=== Forge Model Benchmark ==="
echo "Models: ${#MODELS[@]}"
echo "Tasks: ${#TASKS[@]}"
echo "Max iterations per task: $MAX_ITER"
echo ""

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf name ctx <<< "$model_entry"
  echo "=== Testing: $name ($gguf) ==="
  
  if ! start_model "$gguf" "$ctx"; then
    echo "  SKIP: failed to start"
    continue
  fi
  sleep 5  # extra warmup
  
  for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r task_id task_desc <<< "$task_entry"
    echo -n "  $task_id... "
    run_task "$task_id" "$task_desc" "$name"
    
    # Count results
    passes=$(grep -c "PASS" "$RESULTS_DIR/${name}_${task_id}.txt" 2>/dev/null || echo 0)
    fails=$(grep -c "FAIL" "$RESULTS_DIR/${name}_${task_id}.txt" 2>/dev/null || echo 0)
    errors=$(grep -c "ERROR:" "$RESULTS_DIR/${name}_${task_id}.txt" 2>/dev/null || echo 0)
    iters=$(grep -c "Iteration" "$RESULTS_DIR/${name}_${task_id}.txt" 2>/dev/null || echo 0)
    done_sig=$(grep -c "Model signals" "$RESULTS_DIR/${name}_${task_id}.txt" 2>/dev/null || echo 0)
    
    echo "iters=$iters pass=$passes fail=$fails err=$errors done=$done_sig"
  done
  echo ""
done

# Cleanup override
rm -f ~/.config/systemd/user/llama-server.service.d/override.conf
systemctl --user daemon-reload

# Summary
echo "=== SUMMARY ==="
echo ""
printf "%-20s | %-8s | %-8s | %-8s | %-8s\n" "Model" "T1_add" "T2_valid" "T3_even" "T4_multi"
printf "%-20s-+-%-8s-+-%-8s-+-%-8s-+-%-8s\n" "--------------------" "--------" "--------" "--------" "--------"

for model_entry in "${MODELS[@]}"; do
  IFS='|' read -r gguf name ctx <<< "$model_entry"
  results=""
  for task_entry in "${TASKS[@]}"; do
    IFS='|' read -r task_id task_desc <<< "$task_entry"
    f="$RESULTS_DIR/${name}_${task_id}.txt"
    if [ -f "$f" ]; then
      p=$(grep -c "PASS" "$f" 2>/dev/null || echo 0)
      fl=$(grep -c "FAIL" "$f" 2>/dev/null || echo 0)
      it=$(grep -c "Iteration" "$f" 2>/dev/null || echo 0)
      results="$results | ${p}P/${fl}F/${it}i"
    else
      results="$results | skip    "
    fi
  done
  printf "%-20s%s\n" "$name" "$results"
done
