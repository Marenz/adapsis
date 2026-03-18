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

# The system prompt (same as in prompt.rs, abbreviated for training efficiency)
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
  +return expr
  +each collection item:Type (body indented below)

+fn name (params)->ReturnType [effects]
  body...

+module Name
  declarations...
end

!test function_name
  +with input -> expect output

Rules:
- Pure functions have no effect brackets
- Every binding has explicit type
- No closures, no inheritance, no exceptions
- concat() for strings, not +
- Test blocks do NOT use end
- For conditional dispatch use check+return, not branch
"""

# Training examples: (task, response) pairs
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
        "Write a function 'is_positive' that takes an Int and returns a Bool.",
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
        "Write a function 'abs' that returns the absolute value of an Int. Use the check+return pattern for conditional logic.",
        """<code>
+fn abs (x:Int)->Int
  +let neg:Bool = x < 0
  +check is_positive neg == false ~is_negative
  +return x

!test abs
  +with 5 -> expect 5
  +with 0 -> expect 0
  +with -3 -> expect Err(is_negative)
</code>""",
    ),
    (
        "Write a function 'clamp' that clamps an Int between a min and max value.",
        """<code>
+fn clamp (value:Int, min_val:Int, max_val:Int)->Int
  +check not_below value>=min_val ~too_low
  +check not_above value<=max_val ~too_high
  +return value

!test clamp
  +with {value: 5, min_val: 0, max_val: 10} -> expect 5
  +with {value: -1, min_val: 0, max_val: 10} -> expect Err(too_low)
  +with {value: 15, min_val: 0, max_val: 10} -> expect Err(too_high)
</code>""",
    ),
    (
        "Write a function 'max' that returns the larger of two Ints.",
        """<code>
+fn max (a:Int, b:Int)->Int
  +check a_bigger a>=b ~b_is_bigger
  +return a

!test max
  +with {a: 5, b: 3} -> expect 5
  +with {a: 3, b: 5} -> expect Err(b_is_bigger)
  +with {a: 5, b: 5} -> expect 5
</code>""",
    ),
    # === String functions ===
    (
        "Write a function 'greet' that takes a name String and returns a greeting.",
        """<code>
+fn greet (name:String)->String
  +return concat("Hello, ", name)

!test greet
  +with "Alice" -> expect "Hello, Alice"
  +with "Bob" -> expect "Hello, Bob"
  +with "" -> expect "Hello, "
</code>""",
    ),
    (
        "Write a function 'is_empty_string' that checks if a String is empty.",
        """<code>
+fn is_empty_string (s:String)->Bool
  +return s.len == 0

!test is_empty_string
  +with "" -> expect true
  +with "hello" -> expect false
  +with " " -> expect false
</code>""",
    ),
    # === Type definitions ===
    (
        "Define a Point type with x and y Float coordinates, and a function to compute distance from origin.",
        """<code>
+type Point = {x:Float, y:Float}

+fn distance_from_origin (p:Point)->Float
  +let x_sq:Float = p.x * p.x
  +let y_sq:Float = p.y * p.y
  +return x_sq + y_sq
</code>""",
    ),
    (
        "Define a User type with name, email, and age. Write a validation function.",
        """<code>
+type User = {name:String, email:String, age:Int}

+fn validate_user (u:User)->Result<User> [fail]
  +check name u.name.len>0 ~err_empty_name
  +check email u.email.len>0 ~err_empty_email
  +check age_min u.age>=0 ~err_negative_age
  +check age_max u.age<=150 ~err_age_too_high
  +return u

!test validate_user
  +with {name: "Alice", email: "a@b.com", age: 30} -> expect Ok
  +with {name: "", email: "a@b.com", age: 30} -> expect Err(err_empty_name)
  +with {name: "Alice", email: "", age: 30} -> expect Err(err_empty_email)
  +with {name: "Alice", email: "a@b.com", age: -1} -> expect Err(err_negative_age)
  +with {name: "Alice", email: "a@b.com", age: 200} -> expect Err(err_age_too_high)
</code>""",
    ),
    # === Tagged unions ===
    (
        "Define a Color type as a tagged union with Red, Green, Blue variants.",
        """<code>
+type Color = Red | Green | Blue
</code>""",
    ),
    (
        "Define a Shape tagged union with Circle(Float) for radius and Rectangle with width and height.",
        """<code>
+type Shape = Circle(Float) | Rectangle({width:Float, height:Float})
</code>""",
    ),
    # === Modules ===
    (
        "Create a MathUtils module with add, subtract, and multiply functions.",
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
  +with {a: 3, b: 2} -> expect 5

!test subtract
  +with {a: 5, b: 3} -> expect 2

!test multiply
  +with {a: 4, b: 3} -> expect 12
</code>""",
    ),
    (
        "Create a StringUtils module with is_blank, trim_length, and starts_with_letter functions.",
        """<code>
+module StringUtils
  +fn is_blank (s:String)->Bool
    +return s.len == 0

  +fn has_content (s:String)->Bool
    +return s.len > 0
end

!test is_blank
  +with "" -> expect true
  +with "hello" -> expect false

!test has_content
  +with "hello" -> expect true
  +with "" -> expect false
</code>""",
    ),
    # === Effect-heavy functions ===
    (
        "Write a function that validates an API request with multiple checks.",
        """<code>
+type ApiRequest = {method:String, path:String, body_len:Int}

+fn validate_request (req:ApiRequest)->Result<ApiRequest> [fail]
  +check method req.method.len>0 ~err_empty_method
  +check path req.path.len>0 ~err_empty_path
  +check body req.body_len>=0 ~err_negative_body
  +check body_max req.body_len<=1048576 ~err_body_too_large
  +return req

!test validate_request
  +with {method: "GET", path: "/api/users", body_len: 0} -> expect Ok
  +with {method: "", path: "/api/users", body_len: 0} -> expect Err(err_empty_method)
  +with {method: "POST", path: "", body_len: 100} -> expect Err(err_empty_path)
  +with {method: "POST", path: "/api", body_len: -1} -> expect Err(err_negative_body)
  +with {method: "POST", path: "/api", body_len: 2000000} -> expect Err(err_body_too_large)
</code>""",
    ),
    # === Functions calling functions ===
    (
        "Write a celsius_to_fahrenheit converter and a function that checks if a temperature is freezing.",
        """<code>
+fn celsius_to_fahrenheit (c:Float)->Float
  +return c * 1.8 + 32.0

+fn is_freezing (c:Float)->Bool
  +let f:Float = celsius_to_fahrenheit(c)
  +return f <= 32.0
</code>""",
    ),
    (
        "Write a function 'square' and a function 'sum_of_squares' that uses it.",
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
  +with -2 -> expect 4

!test sum_of_squares
  +with {a: 3, b: 4} -> expect 25
  +with {a: 0, b: 5} -> expect 25
</code>""",
    ),
    # === Each loops ===
    (
        "Write a function that counts items in a list.",
        """<code>
+fn count_items (items:List<Int>)->Int
  +let count:Int = 0
  +each items item:Int
    +let count:Int = count + 1
  +return count
</code>""",
    ),
    # === Multi-step validation ===
    (
        "Write a registration system with username, password, and email validation.",
        """<code>
+type Registration = {username:String, password:String, email:String}

+fn validate_registration (reg:Registration)->Result<Registration> [fail]
  +check username reg.username.len>0 ~err_empty_username
  +check username_len reg.username.len>=3 ~err_username_too_short
  +check password reg.password.len>0 ~err_empty_password
  +check password_len reg.password.len>=8 ~err_password_too_short
  +check email reg.email.len>0 ~err_empty_email
  +return reg

!test validate_registration
  +with {username: "alice", password: "secret123", email: "a@b.com"} -> expect Ok
  +with {username: "", password: "secret123", email: "a@b.com"} -> expect Err(err_empty_username)
  +with {username: "ab", password: "secret123", email: "a@b.com"} -> expect Err(err_username_too_short)
  +with {username: "alice", password: "", email: "a@b.com"} -> expect Err(err_empty_password)
  +with {username: "alice", password: "short", email: "a@b.com"} -> expect Err(err_password_too_short)
  +with {username: "alice", password: "secret123", email: ""} -> expect Err(err_empty_email)
</code>""",
    ),
    # === DONE signal ===
    (
        "All validations and tests passed. The implementation is complete.",
        """<code>
DONE
</code>""",
    ),
    # === Error correction examples ===
    (
        """The function had a parse error. Fix it.
ERROR: line 3: expected `:` after binding name
Previous code:
+fn add (a:Int, b:Int)->Int
  +let result = a + b
  +return result""",
        """<think>
The error is on line 3 — the let binding is missing a type annotation.
In Forge, every binding must have an explicit type: +let name:Type = expr
</think>
<code>
+fn add (a:Int, b:Int)->Int
  +let result:Int = a + b
  +return result

!test add
  +with {a: 3, b: 4} -> expect 7
</code>""",
    ),
    (
        """Test failed. Fix the function.
FAIL [0]: input={age: 25, name: "alice"} => {age: 25, name: "alice"}, expected Ok(None)
The function returns Result<User> but the return value isn't wrapped.""",
        """<think>
The function's return type is Result<User> and we're using +return input.
The runtime auto-wraps in Ok() for Result return types, so this should work.
But the test expects Ok — let me check if the function is correctly declared.
The issue might be that the function doesn't have [fail] effect.
</think>
<code>
+fn validate (input:User)->Result<User> [fail]
  +check name input.name.len>0 ~err_empty_name
  +check age input.age>=0 ~err_negative_age
  +return input

!test validate
  +with {name: "alice", age: 25} -> expect Ok
  +with {name: "", age: 25} -> expect Err(err_empty_name)
</code>""",
    ),
    # === Arithmetic ===
    (
        "Write a factorial function for non-negative integers.",
        """<code>
+fn factorial (n:Int)->Result<Int> [fail]
  +check non_negative n>=0 ~err_negative_input
  +check base_case n<=1 ~compute
  +return 1

!test factorial
  +with 0 -> expect Ok
  +with 1 -> expect Ok
  +with -1 -> expect Err(err_negative_input)
</code>""",
    ),
    # === Config/settings pattern ===
    (
        "Define a ServerConfig type and a validation function for it.",
        """<code>
+type ServerConfig = {host:String, port:Int, max_connections:Int, timeout_ms:Int}

+fn validate_config (cfg:ServerConfig)->Result<ServerConfig> [fail]
  +check host cfg.host.len>0 ~err_empty_host
  +check port_min cfg.port>=1 ~err_invalid_port
  +check port_max cfg.port<=65535 ~err_invalid_port
  +check max_conn cfg.max_connections>=1 ~err_invalid_max_connections
  +check timeout cfg.timeout_ms>=100 ~err_timeout_too_low
  +return cfg

!test validate_config
  +with {host: "localhost", port: 8080, max_connections: 100, timeout_ms: 5000} -> expect Ok
  +with {host: "", port: 8080, max_connections: 100, timeout_ms: 5000} -> expect Err(err_empty_host)
  +with {host: "localhost", port: 0, max_connections: 100, timeout_ms: 5000} -> expect Err(err_invalid_port)
  +with {host: "localhost", port: 70000, max_connections: 100, timeout_ms: 5000} -> expect Err(err_invalid_port)
  +with {host: "localhost", port: 8080, max_connections: 0, timeout_ms: 5000} -> expect Err(err_invalid_max_connections)
  +with {host: "localhost", port: 8080, max_connections: 100, timeout_ms: 50} -> expect Err(err_timeout_too_low)
</code>""",
    ),
    # === Let + call pattern ===
    (
        "Write a function that creates a formatted user display name from first and last name.",
        """<code>
+fn format_display_name (first:String, last:String)->String
  +let full:String = concat(first, " ")
  +let display:String = concat(full, last)
  +return display

!test format_display_name
  +with {first: "John", last: "Doe"} -> expect "John Doe"
  +with {first: "Alice", last: "Smith"} -> expect "Alice Smith"
</code>""",
    ),
    # === Multiple return paths via check ===
    (
        "Write a function that categorizes an age: child (0-12), teen (13-17), adult (18-64), senior (65+).",
        """<code>
+fn categorize_age (age:Int)->Result<String> [fail]
  +check valid age>=0 ~err_negative_age
  +check not_child age>=13 ~is_child
  +check not_teen age>=18 ~is_teen
  +check not_adult age>=65 ~is_adult
  +return "senior"

!test categorize_age
  +with 5 -> expect Err(is_child)
  +with 15 -> expect Err(is_teen)
  +with 30 -> expect Err(is_adult)
  +with 70 -> expect Ok
  +with -1 -> expect Err(err_negative_age)
</code>""",
    ),
    # === Modulo operator ===
    (
        "Write a function 'is_even' that checks if an Int is even using the modulo operator.",
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
        "Write a function 'is_divisible' that checks if a is divisible by b.",
        """<code>
+fn is_divisible (a:Int, b:Int)->Result<Bool> [fail]
  +check non_zero b!=0 ~err_division_by_zero
  +let remainder:Int = a % b
  +return remainder == 0

!test is_divisible
  +with {a: 10, b: 3} -> expect Ok
  +with {a: 9, b: 3} -> expect Ok
  +with {a: 10, b: 0} -> expect Err(err_division_by_zero)
</code>""",
    ),
    (
        "Write a fizzbuzz classifier that returns 'fizz' for multiples of 3, 'buzz' for multiples of 5, 'fizzbuzz' for both.",
        """<code>
+fn fizzbuzz (n:Int)->String
  +let mod3:Int = n % 3
  +let mod5:Int = n % 5
  +check not_fizzbuzz mod3!=0 OR mod5!=0 ~is_fizzbuzz
  +check not_fizz mod3!=0 ~is_fizz
  +check not_buzz mod5!=0 ~is_buzz
  +return "other"

!test fizzbuzz
  +with 15 -> expect Err(is_fizzbuzz)
  +with 3 -> expect Err(is_fizz)
  +with 5 -> expect Err(is_buzz)
  +with 7 -> expect Ok
</code>""",
    ),
    # === Branch with string matching ===
    (
        "Write a function that maps a color name to its hex code using branch.",
        """<code>
+fn color_to_hex (color:String)->Result<String> [fail]
  +check is_red color=="red" ~not_red
  +return "#FF0000"

!test color_to_hex
  +with "red" -> expect Ok
  +with "blue" -> expect Err(not_red)
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

    # Also validate that all code blocks parse correctly
    print("\nValidating examples...")
    forge_binary = Path(__file__).parent.parent / "target" / "debug" / "forge"
    if not forge_binary.exists():
        print(f"  (skip validation — {forge_binary} not found, run cargo build first)")
        return

    import subprocess
    import tempfile

    passed = 0
    failed = 0
    for i, (task, response) in enumerate(EXAMPLES):
        # Extract code from response
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

        # Write to temp file and check
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".forge", delete=False
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
            print(f"  FAIL example {i}: {task[:60]}...")
            if result.stderr:
                print(f"    stderr: {result.stderr.strip()[:200]}")
            if result.stdout and "ERROR" in result.stdout:
                for line in result.stdout.split("\n"):
                    if "ERROR" in line:
                        print(f"    {line.strip()[:200]}")

    print(f"\nValidation: {passed} passed, {failed} failed out of {len(EXAMPLES)}")


if __name__ == "__main__":
    main()
