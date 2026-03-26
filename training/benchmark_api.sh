#!/bin/bash
# Benchmark ForgeOS API with various tasks and models
# Tests the full loop: ask → code → apply → eval → results

set -e

PORT=${1:-3002}
URL=${2:-http://localhost:4400}
MODEL=${3:-anthropic/claude-haiku-4-5-20251001}
BASE="http://127.0.0.1:$PORT"
RESULTS_DIR="$(dirname "$0")/benchmark_api_results"
mkdir -p "$RESULTS_DIR"

ask() {
  local task_id="$1"
  local message="$2"
  local max_retries="${3:-3}"
  local outfile="$RESULTS_DIR/${MODEL//\//_}_${task_id}.json"
  
  local all_ok=0
  for i in $(seq 1 $max_retries); do
    local result=$(curl -s --max-time 120 -X POST "$BASE/api/ask" \
      -H 'Content-Type: application/json' \
      -d "{\"message\": \"$message\"}")
    
    local has_errors=$(echo "$result" | python3 -c "import json,sys; print(json.load(sys.stdin).get('has_errors',True))" 2>/dev/null)
    local n_ok=$(echo "$result" | python3 -c "import json,sys; print(sum(1 for r in json.load(sys.stdin).get('results',[]) if r.get('success')))" 2>/dev/null)
    local n_err=$(echo "$result" | python3 -c "import json,sys; print(sum(1 for r in json.load(sys.stdin).get('results',[]) if not r.get('success')))" 2>/dev/null)
    local n_pass=$(echo "$result" | python3 -c "import json,sys; print(sum(1 for r in json.load(sys.stdin).get('test_results',[]) if r.get('pass')))" 2>/dev/null)
    local n_fail=$(echo "$result" | python3 -c "import json,sys; print(sum(1 for r in json.load(sys.stdin).get('test_results',[]) if not r.get('pass')))" 2>/dev/null)
    
    echo "$result" > "$outfile"
    
    if [ "$has_errors" = "False" ]; then
      echo "  [$task_id] iter=$i ok=$n_ok pass=$n_pass PASS"
      all_ok=1
      break
    else
      if [ $i -lt $max_retries ]; then
        # Feed error back
        local errors=$(echo "$result" | python3 -c "
import json,sys
d=json.load(sys.stdin)
errs = [r['message'] for r in d.get('results',[]) if not r.get('success')]
errs += [r['message'] for r in d.get('test_results',[]) if not r.get('pass')]
print(' '.join(errs[:3])[:200])
" 2>/dev/null)
        message="Fix these errors: $errors"
        echo "  [$task_id] iter=$i ok=$n_ok err=$n_err fail=$n_fail — retrying"
      else
        echo "  [$task_id] iter=$i ok=$n_ok err=$n_err fail=$n_fail FAIL"
      fi
    fi
  done
  return $((1 - all_ok))
}

# Start fresh session for each model
echo "=== ForgeOS API Benchmark ==="
echo "Model: $MODEL"
echo "Port: $PORT, LLM: $URL"
echo ""

# Reset session
curl -s -X POST "$BASE/api/rewind" -H 'Content-Type: application/json' -d '{"revision": 0}' > /dev/null

PASS=0
FAIL=0

# === Level 1: Basic functions ===
echo "--- Level 1: Basic functions ---"

ask "L1_add" "Write a function add(a:Int, b:Int)->Int that returns a+b. Test with !eval add a=3 b=4" 2
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L1_double" "Write a function double(x:Int)->Int that returns x*2. Include !test with 3 cases." 2
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L1_greet" "Write greet(name:String)->String that returns concat(\"Hello \", name). Test with !eval greet \"World\"" 2
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 2: Conditionals ===
echo "--- Level 2: Conditionals ---"

ask "L2_abs" "Write abs(x:Int)->Int using +if/+else. Test with positive, negative, and zero." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L2_fizzbuzz" "Write fizzbuzz(n:Int)->String using modulo and if/elif/else. Returns fizzbuzz/fizz/buzz/other. Test with !eval for 15, 3, 5, 7." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L2_grade" "Write grade(score:Int)->String: A for >=90, B >=80, C >=70, D >=60, F otherwise. Test all grades." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 3: Structs and validation ===
echo "--- Level 3: Structs and validation ---"

ask "L3_validate" "Define User type with name:String and age:Int. Write validate_user with [fail] that checks name not empty and age >= 0. Test with passing and failing cases." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L3_compose" "Write square(x:Int)->Int and sum_of_squares(a:Int, b:Int)->Int where sum_of_squares calls square. Test both." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 4: Loops and mutation ===
echo "--- Level 4: Loops and mutation ---"

ask "L4_count" "Write count_chars(s:String, target:String)->Int that counts occurrences of target in s using +while, char_at, and +set. Test with count_chars s=\"hello\" target=\"l\" expecting 2." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L4_reverse" "Write reverse_string(s:String)->String using while loop, char_at, and concat. Test with \"hello\" expecting \"olleh\"." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 5: Recursive types ===
echo "--- Level 5: Recursive types ---"

ask "L5_expr" "Define type Expr = Literal(Int) | Add(Expr, Expr) | Mul(Expr, Expr). Write eval_expr using +match. Test: eval_expr Add(Literal(1), Mul(Literal(2), Literal(3))) should be 7." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 6: IO ===
echo "--- Level 6: IO ---"

ask "L6_file" "Write a function that writes \"test123\" to /tmp/adapsis-bench-test.txt and reads it back, returning the content. Use file_write and file_read with [io,async]. Eval it." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L6_shell" "Write a function that runs shell_exec(\"echo hello\") and returns the result. Use [io,async]. Eval it." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

# === Level 7: Complex ===
echo "--- Level 7: Complex ---"

ask "L7_tokenizer" "Write is_digit(ch:String)->Bool using char_code, and parse_number(s:String, pos:Int)->Int that reads consecutive digits using while loop. Test parse_number s=\"123abc\" pos=0 expecting 123." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

ask "L7_hex" "Write hex_encode(n:Int)->String that converts an integer to hex string using while loop, modulo, and char lookup. Test with hex_encode 255 expecting \"ff\"." 3
[ $? -eq 0 ] && PASS=$((PASS+1)) || FAIL=$((FAIL+1))

echo ""
echo "=== RESULTS: $MODEL ==="
echo "PASS: $PASS  FAIL: $FAIL  TOTAL: $((PASS+FAIL))"
echo ""
