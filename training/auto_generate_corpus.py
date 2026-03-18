#!/usr/bin/env python3
"""Auto-generate training corpus by running Forge tasks through the LLM.

Uses the running llama.cpp server to generate Forge code for various tasks,
validates each result, and collects successful (task, code) pairs.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

FORGE = str(Path(__file__).parent.parent / "target" / "debug" / "forge")
OUTPUT = Path(__file__).parent / "forge_training_data_auto.jsonl"

# System prompt (same as in prompt.rs, abbreviated)
SYSTEM_PROMPT = (
    open(Path(__file__).parent / "system_prompt.txt").read()
    if (Path(__file__).parent / "system_prompt.txt").exists()
    else ""
)

# Tasks to generate — mix of difficulty levels
TASKS = [
    # === Basic arithmetic ===
    "Write a function 'double' that takes an Int x and returns x * 2. Include tests.",
    "Write a function 'triple' that takes an Int x and returns x * 3. Include tests.",
    "Write a function 'negate' that takes an Int x and returns -x (use 0 - x). Include tests.",
    "Write a function 'square' that takes an Int x and returns x * x. Include tests.",
    "Write a function 'cube' that takes an Int x and returns x * x * x. Include tests.",
    "Write a function 'add' that takes two Ints a and b and returns their sum. Include tests.",
    "Write a function 'subtract' that takes two Ints a and b and returns a - b. Include tests.",
    "Write a function 'multiply' that takes two Ints a and b and returns a * b. Include tests.",
    "Write a function 'sum_of_three' that takes three Ints a, b, c and returns a + b + c. Include tests.",
    # === Boolean logic ===
    "Write a function 'is_positive' that takes an Int and returns true if it's greater than 0. Include tests.",
    "Write a function 'is_negative' that takes an Int and returns true if it's less than 0. Include tests.",
    "Write a function 'is_zero' that takes an Int and returns true if it equals 0. Include tests.",
    "Write a function 'is_even' that checks if an Int is even using modulo (%). Include tests.",
    "Write a function 'is_odd' that checks if an Int is odd using modulo (%). Include tests.",
    "Write a function 'is_divisible_by' that takes a:Int and b:Int and returns true if a is divisible by b. Use modulo. Include tests.",
    "Write a function 'both_positive' that takes two Ints and returns true if both are positive. Include tests.",
    # === Conditional (if/elif/else) ===
    "Write a function 'abs' that returns the absolute value of an Int using if/else. Include tests.",
    "Write a function 'max' that returns the larger of two Ints using if/else. Include tests.",
    "Write a function 'min' that returns the smaller of two Ints using if/else. Include tests.",
    "Write a function 'sign' that returns 1 for positive, -1 for negative, and 0 for zero using if/elif/else. Include tests.",
    "Write a function 'clamp' that takes value, min_val, max_val Ints and returns value clamped to range using if/elif/else. Include tests.",
    "Write a function 'classify_age' that returns 'child' for age<13, 'teen' for 13-17, 'adult' for 18-64, 'senior' for 65+. Use if/elif/else. Include tests.",
    "Write a function 'grade' that takes a score Int (0-100) and returns 'A' for >=90, 'B' for >=80, 'C' for >=70, 'D' for >=60, 'F' otherwise. Include tests.",
    "Write a function 'fizzbuzz' that takes an Int n and returns 'fizzbuzz' if divisible by both 3 and 5, 'fizz' if by 3, 'buzz' if by 5, 'other' otherwise. Use modulo and if/elif/else. Include tests.",
    "Write a function 'day_type' that takes day:Int (1-7) and returns 'weekday' for 1-5, 'weekend' for 6-7. Use if/else. Include tests.",
    # === Validation (check + Result) ===
    "Define type User = {name:String, age:Int}. Write validate_user that checks name not empty and age >= 0 and age <= 150. Returns Result<User> [fail]. Include tests.",
    "Define type Email = {address:String}. Write validate_email that checks address not empty and address length >= 3. Returns Result<Email> [fail]. Include tests.",
    "Define type Password = {value:String}. Write validate_password that checks value not empty and length >= 8 and length <= 128. Returns Result<Password> [fail]. Include tests.",
    "Define type Port = {number:Int}. Write validate_port that checks number >= 1 and number <= 65535. Returns Result<Port> [fail]. Include tests.",
    "Define type Temperature = {celsius:Float}. Write validate_temp that checks celsius >= -273.15. Returns Result<Temperature> [fail]. Include tests.",
    "Define type Percentage = {value:Int}. Write validate_percentage that checks value >= 0 and value <= 100. Returns Result<Percentage> [fail]. Include tests.",
    "Define type Config = {host:String, port:Int, timeout:Int}. Write validate_config that checks host not empty, port 1-65535, timeout >= 100. Returns Result<Config> [fail]. Include tests.",
    # === Multi-function ===
    "Write 'square' and 'sum_of_squares' functions where sum_of_squares uses square. Include tests for both.",
    "Write 'celsius_to_fahrenheit' that converts C to F (c * 9 / 5 + 32) and 'is_freezing' that checks if temp <= 0. Include tests.",
    "Write 'is_even' and 'is_odd' where is_odd uses NOT is_even. Include tests for both.",
    # === String operations ===
    "Write a function 'greet' that takes name:String and returns concat('Hello, ', name). Include tests.",
    "Write a function 'is_blank' that takes s:String and returns true if s.len == 0. Include tests.",
    "Write a function 'format_name' that takes first:String and last:String and returns concat(first, concat(' ', last)). Include tests.",
    # === Modules ===
    "Create a MathUtils module with add, subtract, multiply functions. Include tests for all.",
    "Create a Validators module with validate_positive (checks Int > 0) and validate_non_empty (checks String not empty). Include tests.",
    # === Complex validation ===
    "Build a registration system: RegistrationRequest type with username, email, age. validate_registration checks username not empty (len>0), email not empty (len>0), age >= 18, age <= 120. Include 6+ tests.",
    "Build an order system: OrderItem type with product_name:String, quantity:Int, price:Float. validate_order_item checks product_name not empty, quantity > 0, quantity <= 100, price > 0. Include tests.",
    # === Error correction examples ===
    # These simulate the model needing to fix its own code
]


def run_forge_task(task: str, max_iterations: int = 5, timeout: int = 120) -> dict:
    """Run a Forge task and capture the output."""
    try:
        result = subprocess.run(
            [FORGE, "run", "--task", task, "--max-iterations", str(max_iterations)],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = result.stdout + result.stderr

        # Count passes and fails
        passes = output.count("PASS")
        fails = output.count("FAIL")
        parse_errors = output.count("Parse error")
        done = "Model signals completion" in output
        iterations = output.count("Iteration")

        return {
            "task": task,
            "output": output,
            "passes": passes,
            "fails": fails,
            "parse_errors": parse_errors,
            "done": done,
            "iterations": iterations,
            "success": passes > 0 and fails == 0 and done,
        }
    except subprocess.TimeoutExpired:
        return {
            "task": task,
            "output": "TIMEOUT",
            "passes": 0,
            "fails": 0,
            "parse_errors": 0,
            "done": False,
            "iterations": 0,
            "success": False,
        }
    except Exception as e:
        return {
            "task": task,
            "output": str(e),
            "passes": 0,
            "fails": 0,
            "parse_errors": 0,
            "done": False,
            "iterations": 0,
            "success": False,
        }


def extract_code_from_output(output: str) -> str:
    """Extract the last successful code block from the forge output."""
    # The output contains the LLM's responses with [code] markers
    # We want the last code block that was successfully validated
    lines = output.split("\n")
    code_blocks = []
    current_block = []
    in_code = False

    for line in lines:
        if "[code]" in line:
            in_code = True
            # Extract the code after [code] marker
            code_start = line.find("[code]") + len("[code]")
            rest = line[code_start:].strip()
            if rest:
                current_block = [rest]
            else:
                current_block = []
        elif in_code:
            if line.startswith("---") or line.startswith("===") or "Iteration" in line:
                if current_block:
                    code_blocks.append("\n".join(current_block))
                in_code = False
                current_block = []
            else:
                # Strip ANSI codes and log prefixes
                clean = line.strip()
                if (
                    clean
                    and not clean.startswith("OK:")
                    and not clean.startswith("ERROR:")
                    and not clean.startswith("PASS")
                    and not clean.startswith("FAIL")
                    and not clean.startswith("TYPE")
                ):
                    current_block.append(clean)

    if current_block:
        code_blocks.append("\n".join(current_block))

    return code_blocks[-1] if code_blocks else ""


def main():
    # Load existing manual corpus
    manual_path = Path(__file__).parent / "forge_training_data.jsonl"
    existing = []
    if manual_path.exists():
        with open(manual_path) as f:
            for line in f:
                existing.append(json.loads(line))

    print(f"Existing manual examples: {len(existing)}")
    print(f"Tasks to generate: {len(TASKS)}")
    print(f"Output: {OUTPUT}")
    print()

    # Check forge binary exists
    if not Path(FORGE).exists():
        print(f"ERROR: {FORGE} not found. Run 'cargo build' first.")
        sys.exit(1)

    # Check LLM server is running
    import urllib.request

    try:
        urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=5)
    except Exception:
        print("ERROR: LLM server not running at http://127.0.0.1:8081")
        sys.exit(1)

    results = []
    successes = 0
    failures = 0

    for i, task in enumerate(TASKS):
        print(f"[{i + 1}/{len(TASKS)}] {task[:70]}...")
        result = run_forge_task(task)
        results.append(result)

        if result["success"]:
            successes += 1
            print(
                f"  OK: {result['passes']}P/{result['fails']}F, {result['iterations']} iters"
            )
        else:
            failures += 1
            reason = (
                "timeout"
                if "TIMEOUT" in result["output"]
                else f"{result['passes']}P/{result['fails']}F, {result['parse_errors']} parse_err"
            )
            print(f"  FAIL: {reason}")

    # Save raw results
    raw_path = Path(__file__).parent / "auto_generate_raw.json"
    with open(raw_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n=== Results ===")
    print(f"Success: {successes}/{len(TASKS)}")
    print(f"Failed: {failures}/{len(TASKS)}")
    print(f"Raw results saved to: {raw_path}")

    # Note: To create actual training data from these results, we'd need to
    # extract the successful code blocks and pair them with the tasks.
    # For now, this gives us a success rate to evaluate model capability.


if __name__ == "__main__":
    main()
