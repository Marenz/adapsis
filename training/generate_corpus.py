#!/usr/bin/env python3
"""Generate synthetic training corpus for Forge LoRA fine-tuning.

Each example is a (system, user, assistant) triple where:
- system: the Forge spec (always the same)
- user: a natural language task description
- assistant: the correct Forge code response

The output is a JSONL file suitable for fine-tuning with unsloth/trl.
"""

import json
import sys
from pathlib import Path

# System prompt — abbreviated version of the full Forge spec
SYSTEM_PROMPT = """You are a Forge programmer. Forge is an AI-first programming language.

## Response Format
Wrap reasoning in <think> tags and code in <code> tags.

## Forge Syntax
Types: Int, Float, Bool, String, Byte, List<T>, Map<K,V>, Option<T>, Result<T>
Struct: +type Name = {field:Type, ...}
Union: +type Name = Variant1(Type) | Variant2
Effects: [io], [mut], [fail], [async], [rand], [yield], [parallel], [unsafe]

Statements (indented in function body):
  +let name:Type = expr
  +call name:Type = func(args)
  +check label condition ~err_label
  +if condition / +elif condition / +else
  +return expr
  +each collection item:Type (body indented below)

+fn name (params)->ReturnType [effects]
  body...

+module Name
  declarations...
end

Testing — single param: +with value -> expect result
Testing — multi param: +with key1=val1 key2=val2 -> expect result

!test function_name
  +with a=3 b=4 -> expect 7
  +with 5 -> expect 10

Rules:
- Pure functions have no effect brackets
- Every binding has explicit type
- No closures, no inheritance, no exceptions
- concat() for strings, not +
- Test blocks do NOT use end
- Use +if/+elif/+else for conditionals, +check for validation assertions
- Modulo: n % 2 == 0 for even check
- Error auto-propagation: +call val:T = func(x) with [fail] — errors bubble up
- Explicit error handling: +call res:Result<T> = func(x) — use res.is_ok, res.unwrap, res.error
"""

EXAMPLES = [
    # === Basic functions ===
    (
        "Write a function 'double' that takes an Int and returns it doubled.",
        """<code>
+fn double (x:Int)->Int
  +return x * 2

!test double
  +with 5 -> expect 10
  +with 0 -> expect 0
  +with -3 -> expect -6
</code>""",
    ),
    (
        "Write a function 'add' that takes two Ints and returns their sum.",
        """<code>
+fn add (a:Int, b:Int)->Int
  +return a + b

!test add
  +with a=3 b=4 -> expect 7
  +with a=0 b=0 -> expect 0
  +with a=-1 b=1 -> expect 0
</code>""",
    ),
    (
        "Write a function 'subtract' that takes two Ints and returns a - b.",
        """<code>
+fn subtract (a:Int, b:Int)->Int
  +return a - b

!test subtract
  +with a=5 b=3 -> expect 2
  +with a=0 b=0 -> expect 0
  +with a=3 b=5 -> expect -2
</code>""",
    ),
    (
        "Write a function 'multiply' that takes two Ints and returns a * b.",
        """<code>
+fn multiply (a:Int, b:Int)->Int
  +return a * b

!test multiply
  +with a=4 b=3 -> expect 12
  +with a=0 b=5 -> expect 0
  +with a=-2 b=3 -> expect -6
</code>""",
    ),
    (
        "Write a function 'square' that takes an Int and returns x * x.",
        """<code>
+fn square (x:Int)->Int
  +return x * x

!test square
  +with 3 -> expect 9
  +with 0 -> expect 0
  +with -2 -> expect 4
</code>""",
    ),
    (
        "Write 'square' and 'sum_of_squares' where sum_of_squares uses square.",
        """<code>
+fn square (x:Int)->Int
  +return x * x

+fn sum_of_squares (a:Int, b:Int)->Int
  +let a_sq:Int = square(a)
  +let b_sq:Int = square(b)
  +return a_sq + b_sq

!test square
  +with 3 -> expect 9
  +with 0 -> expect 0

!test sum_of_squares
  +with a=3 b=4 -> expect 25
  +with a=0 b=5 -> expect 25
</code>""",
    ),
    # === Boolean functions ===
    (
        "Write a function 'is_positive' that takes an Int and returns true if > 0.",
        """<code>
+fn is_positive (x:Int)->Bool
  +return x > 0

!test is_positive
  +with 5 -> expect true
  +with 0 -> expect false
  +with -1 -> expect false
</code>""",
    ),
    (
        "Write a function 'is_even' that checks if an Int is even using modulo.",
        """<code>
+fn is_even (x:Int)->Bool
  +let remainder:Int = x % 2
  +return remainder == 0

!test is_even
  +with 4 -> expect true
  +with 5 -> expect false
  +with 0 -> expect true
  +with -2 -> expect true
</code>""",
    ),
    (
        "Write 'is_divisible_by' that checks if a is divisible by b.",
        """<code>
+fn is_divisible_by (a:Int, b:Int)->Bool
  +let remainder:Int = a % b
  +return remainder == 0

!test is_divisible_by
  +with a=10 b=5 -> expect true
  +with a=10 b=3 -> expect false
  +with a=9 b=3 -> expect true
</code>""",
    ),
    # === If/elif/else ===
    (
        "Write 'abs' that returns absolute value using if/else.",
        """<code>
+fn abs (x:Int)->Int
  +if x >= 0
    +return x
  +else
    +return 0 - x

!test abs
  +with 5 -> expect 5
  +with -3 -> expect 3
  +with 0 -> expect 0
</code>""",
    ),
    (
        "Write 'max' that returns the larger of two Ints.",
        """<code>
+fn max (a:Int, b:Int)->Int
  +if a >= b
    +return a
  +else
    +return b

!test max
  +with a=5 b=3 -> expect 5
  +with a=3 b=5 -> expect 5
  +with a=5 b=5 -> expect 5
</code>""",
    ),
    (
        "Write 'min' that returns the smaller of two Ints.",
        """<code>
+fn min (a:Int, b:Int)->Int
  +if a <= b
    +return a
  +else
    +return b

!test min
  +with a=5 b=3 -> expect 3
  +with a=3 b=5 -> expect 3
  +with a=5 b=5 -> expect 5
</code>""",
    ),
    (
        "Write 'sign' that returns 1 for positive, -1 for negative, 0 for zero.",
        """<code>
+fn sign (x:Int)->Int
  +if x > 0
    +return 1
  +elif x < 0
    +return 0 - 1
  +else
    +return 0

!test sign
  +with 5 -> expect 1
  +with -3 -> expect -1
  +with 0 -> expect 0
</code>""",
    ),
    (
        "Write 'clamp' that clamps value between min_val and max_val.",
        """<code>
+fn clamp (value:Int, min_val:Int, max_val:Int)->Int
  +if value < min_val
    +return min_val
  +elif value > max_val
    +return max_val
  +else
    +return value

!test clamp
  +with value=5 min_val=0 max_val=10 -> expect 5
  +with value=-5 min_val=0 max_val=10 -> expect 0
  +with value=15 min_val=0 max_val=10 -> expect 10
</code>""",
    ),
    (
        "Write 'classify_age' that returns child/teen/adult/senior.",
        """<code>
+fn classify_age (age:Int)->String
  +if age < 13
    +return "child"
  +elif age < 18
    +return "teen"
  +elif age < 65
    +return "adult"
  +else
    +return "senior"

!test classify_age
  +with 5 -> expect "child"
  +with 15 -> expect "teen"
  +with 30 -> expect "adult"
  +with 70 -> expect "senior"
</code>""",
    ),
    (
        "Write 'grade' that converts a score to a letter grade.",
        """<code>
+fn grade (score:Int)->String
  +if score >= 90
    +return "A"
  +elif score >= 80
    +return "B"
  +elif score >= 70
    +return "C"
  +elif score >= 60
    +return "D"
  +else
    +return "F"

!test grade
  +with 95 -> expect "A"
  +with 85 -> expect "B"
  +with 75 -> expect "C"
  +with 65 -> expect "D"
  +with 50 -> expect "F"
</code>""",
    ),
    (
        "Write 'fizzbuzz' using modulo and if/elif/else.",
        """<code>
+fn fizzbuzz (n:Int)->String
  +let mod3:Int = n % 3
  +let mod5:Int = n % 5
  +if mod3 == 0 AND mod5 == 0
    +return "fizzbuzz"
  +elif mod3 == 0
    +return "fizz"
  +elif mod5 == 0
    +return "buzz"
  +else
    +return "other"

!test fizzbuzz
  +with 15 -> expect "fizzbuzz"
  +with 3 -> expect "fizz"
  +with 5 -> expect "buzz"
  +with 7 -> expect "other"
</code>""",
    ),
    # === Validation with check ===
    (
        "Define User and write validate_user with checks.",
        """<code>
+type User = {name:String, age:Int}

+fn validate_user (u:User)->Result<User> [fail]
  +check name u.name.len>0 ~err_empty_name
  +check age_min u.age>=0 ~err_negative_age
  +check age_max u.age<=150 ~err_age_too_high
  +return u

!test validate_user
  +with name="alice" age=25 -> expect Ok
  +with name="" age=25 -> expect Err(err_empty_name)
  +with name="alice" age=-1 -> expect Err(err_negative_age)
  +with name="alice" age=200 -> expect Err(err_age_too_high)
</code>""",
    ),
    (
        "Define Config and write validate_config.",
        """<code>
+type Config = {host:String, port:Int, timeout:Int}

+fn validate_config (cfg:Config)->Result<Config> [fail]
  +check host cfg.host.len>0 ~err_empty_host
  +check port_min cfg.port>=1 ~err_invalid_port
  +check port_max cfg.port<=65535 ~err_invalid_port
  +check timeout cfg.timeout>=100 ~err_timeout_too_low
  +return cfg

!test validate_config
  +with host="localhost" port=8080 timeout=5000 -> expect Ok
  +with host="" port=8080 timeout=5000 -> expect Err(err_empty_host)
  +with host="localhost" port=0 timeout=5000 -> expect Err(err_invalid_port)
  +with host="localhost" port=8080 timeout=50 -> expect Err(err_timeout_too_low)
</code>""",
    ),
    (
        "Define Password and write validate_password.",
        """<code>
+type Password = {value:String}

+fn validate_password (pw:Password)->Result<Password> [fail]
  +check not_empty pw.value.len>0 ~err_empty_password
  +check min_len pw.value.len>=8 ~err_too_short
  +check max_len pw.value.len<=128 ~err_too_long
  +return pw

!test validate_password
  +with value="secret123" -> expect Ok
  +with value="" -> expect Err(err_empty_password)
  +with value="short" -> expect Err(err_too_short)
</code>""",
    ),
    (
        "Define Percentage and validate it.",
        """<code>
+type Percentage = {value:Int}

+fn validate_percentage (p:Percentage)->Result<Percentage> [fail]
  +check min p.value>=0 ~err_below_zero
  +check max p.value<=100 ~err_above_hundred
  +return p

!test validate_percentage
  +with value=50 -> expect Ok
  +with value=0 -> expect Ok
  +with value=100 -> expect Ok
  +with value=-1 -> expect Err(err_below_zero)
  +with value=101 -> expect Err(err_above_hundred)
</code>""",
    ),
    # === Function composition with error propagation ===
    (
        "Write validate and process where process calls validate. Errors auto-propagate.",
        """<code>
+type Input = {name:String, age:Int}
+type Output = {greeting:String}

+fn validate (input:Input)->Result<Input> [fail]
  +check name input.name.len>0 ~err_empty_name
  +return input

+fn process (input:Input)->Result<Output> [fail]
  +call validated:Input = validate(input)
  +let greeting:String = concat("Hello, ", validated.name)
  +return {greeting: greeting}

!test process
  +with name="alice" age=25 -> expect Ok
  +with name="" age=25 -> expect Err(err_empty_name)
</code>""",
    ),
    (
        "Write validate and process_safe where process_safe handles errors explicitly.",
        """<code>
+type Input = {name:String, age:Int}
+type Output = {greeting:String}

+fn validate (input:Input)->Result<Input> [fail]
  +check name input.name.len>0 ~err_empty_name
  +return input

+fn process_safe (input:Input)->Output
  +call result:Result<Input> = validate(input)
  +if result.is_ok
    +let greeting:String = concat("Hello, ", result.unwrap.name)
    +return {greeting: greeting}
  +else
    +return {greeting: "anonymous"}

!test process_safe
  +with name="alice" age=25 -> expect {greeting: "Hello, alice"}
  +with name="" age=25 -> expect {greeting: "anonymous"}
</code>""",
    ),
    # === String functions ===
    (
        "Write 'greet' that takes name and returns a greeting.",
        """<code>
+fn greet (name:String)->String
  +return concat("Hello, ", name)

!test greet
  +with "Alice" -> expect "Hello, Alice"
  +with "Bob" -> expect "Hello, Bob"
</code>""",
    ),
    (
        "Write 'is_blank' that checks if a string is empty.",
        """<code>
+fn is_blank (s:String)->Bool
  +return s.len == 0

!test is_blank
  +with "" -> expect true
  +with "hello" -> expect false
</code>""",
    ),
    (
        "Write 'format_name' from first and last name.",
        """<code>
+fn format_name (first:String, last:String)->String
  +let full:String = concat(first, " ")
  +let display:String = concat(full, last)
  +return display

!test format_name
  +with first="John" last="Doe" -> expect "John Doe"
  +with first="Alice" last="Smith" -> expect "Alice Smith"
</code>""",
    ),
    # === Modules ===
    (
        "Create a MathUtils module with add, subtract, multiply.",
        """<code>
+module MathUtils
  +fn add (a:Int, b:Int)->Int
    +return a + b

  +fn subtract (a:Int, b:Int)->Int
    +return a - b

  +fn multiply (a:Int, b:Int)->Int
    +return a * b
end

!test add
  +with a=3 b=2 -> expect 5

!test subtract
  +with a=5 b=3 -> expect 2

!test multiply
  +with a=4 b=3 -> expect 12
</code>""",
    ),
    # === Complex validation ===
    (
        "Build a registration system with username, email, age validation.",
        """<code>
+type Registration = {username:String, email:String, age:Int}

+fn validate_registration (reg:Registration)->Result<Registration> [fail]
  +check username reg.username.len>0 ~err_empty_username
  +check username_len reg.username.len>=3 ~err_username_too_short
  +check email reg.email.len>0 ~err_empty_email
  +check age_min reg.age>=18 ~err_too_young
  +check age_max reg.age<=120 ~err_too_old
  +return reg

!test validate_registration
  +with username="alice" email="a@b.com" age=25 -> expect Ok
  +with username="" email="a@b.com" age=25 -> expect Err(err_empty_username)
  +with username="ab" email="a@b.com" age=25 -> expect Err(err_username_too_short)
  +with username="alice" email="" age=25 -> expect Err(err_empty_email)
  +with username="alice" email="a@b.com" age=15 -> expect Err(err_too_young)
  +with username="alice" email="a@b.com" age=130 -> expect Err(err_too_old)
</code>""",
    ),
    # === DONE signal ===
    (
        "All validations and tests passed. The implementation is complete.",
        """<code>
DONE
</code>""",
    ),
    # === Error correction ===
    (
        """Test failed. Fix the function.
FAIL: let binding missing type.
+fn add (a:Int, b:Int)->Int
  +let result = a + b
  +return result""",
        """<think>
The let binding is missing a type annotation. In Forge every binding needs explicit type.
</think>
<code>
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result

!test add
  +with a=3 b=4 -> expect 7
</code>""",
    ),
    # === Type definitions ===
    (
        "Define a Point type and a distance function.",
        """<code>
+type Point = {x:Float, y:Float}

+fn distance_sq (p:Point)->Float
  +let x_sq:Float = p.x * p.x
  +let y_sq:Float = p.y * p.y
  +return x_sq + y_sq
</code>""",
    ),
    (
        "Define a Color tagged union.",
        """<code>
+type Color = Red | Green | Blue
</code>""",
    ),
    # === Multi-function with if/else ===
    (
        "Write celsius_to_fahrenheit and is_freezing.",
        """<code>
+fn celsius_to_fahrenheit (c:Float)->Float
  +return c * 1.8 + 32.0

+fn is_freezing (c:Float)->Bool
  +if c <= 0.0
    +return true
  +else
    +return false

!test is_freezing
  +with 0.0 -> expect true
  +with -5.0 -> expect true
  +with 10.0 -> expect false
</code>""",
    ),
    # === Day type with if/else ===
    (
        "Write day_type: weekday for 1-5, weekend for 6-7.",
        """<code>
+fn day_type (day:Int)->String
  +if day >= 1 AND day <= 5
    +return "weekday"
  +else
    +return "weekend"

!test day_type
  +with 1 -> expect "weekday"
  +with 5 -> expect "weekday"
  +with 6 -> expect "weekend"
  +with 7 -> expect "weekend"
</code>""",
    ),
    # === Using Ok()/Err() explicitly ===
    (
        "Write safe_divide that returns Ok on success, Err on division by zero.",
        """<code>
+fn safe_divide (a:Int, b:Int)->Result<Int> [fail]
  +if b == 0
    +return Err("division_by_zero")
  +else
    +return Ok(a / b)

!test safe_divide
  +with a=10 b=2 -> expect Ok
  +with a=10 b=0 -> expect Err(division_by_zero)
</code>""",
    ),
]


def main():
    output_path = Path(__file__).parent / "forge_training_data.jsonl"

    with open(output_path, "w") as f:
        for task, response in EXAMPLES:
            example = {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": task},
                    {"role": "assistant", "content": response},
                ]
            }
            f.write(json.dumps(example) + "\n")

    print(f"Generated {len(EXAMPLES)} training examples -> {output_path}")

    # Validate that code blocks parse
    forge_binary = Path(__file__).parent.parent / "target" / "debug" / "forge"
    if not forge_binary.exists():
        print(f"  (skip validation — {forge_binary} not found)")
        return

    import subprocess
    import tempfile

    passed = 0
    failed = 0
    for i, (task, response) in enumerate(EXAMPLES):
        code = ""
        if "<code>" in response and "</code>" in response:
            start = response.index("<code>") + len("<code>")
            end = response.index("</code>")
            code = response[start:end].strip()
        else:
            code = response.strip()

        if code == "DONE" or not code:
            passed += 1
            continue

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".ax", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        result = subprocess.run(
            [str(forge_binary), "check", tmp_path],
            capture_output=True,
            text=True,
        )
        Path(tmp_path).unlink()

        if result.returncode == 0 and "ERROR" not in result.stderr:
            passed += 1
        else:
            failed += 1
            print(f"  FAIL [{i}]: {task[:60]}...")
            for line in (result.stdout + result.stderr).split("\n"):
                if "ERROR" in line:
                    print(f"    {line.strip()[:200]}")

    print(f"Validation: {passed} passed, {failed} failed out of {len(EXAMPLES)}")


if __name__ == "__main__":
    main()
