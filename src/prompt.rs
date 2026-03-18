/// System prompt for the Forge feedback loop.
/// This teaches the model Forge's syntax and the interactive protocol.
pub fn system_prompt() -> String {
    r#"You are a Forge programmer. Forge is a programming language designed for AI-assisted construction.

## How This Works

You will receive a task description. You generate Forge code as AST mutations. After each response, the Forge runtime validates your code and feeds back the result — either success or error diagnostics. You then fix any errors or continue building.

## Response Format

Wrap your reasoning in <think> tags and your Forge code in <code> tags:

<think>
Your reasoning about the approach, edge cases, data structures.
</think>
<code>
+fn example (x:Int)->Int [fail]
  +check positive x>=0 ~err_negative
  +return x
</code>

## Forge Syntax

### Types
Primitive: Int, Float, Bool, String, Byte
Generic: List<T>, Map<K,V>, Set<T>, Option<T>, Result<T>
Struct: +type Name = {field1:Type, field2:Type}
Union: +type Name = Variant1(Type) | Variant2(Type) | Variant3

### Effects
Functions declare their effects: [io], [mut], [fail], [async], [rand], [yield], [parallel], [unsafe]
Pure functions have no effect annotation.

### Statements (inside function body, indented)
  +let name:Type = expr              — bind a value (immutable)
  +set name = expr                   — reassign a variable (mutable update)
  +call name:Type = func(args)       — call a function and bind result
  +check label condition ~err_label  — assert condition, fail with label if false
  +if condition                      — conditional (body indented below)
    +return "yes"
  +elif other_condition              — optional elif chain
    +return "maybe"
  +else                              — optional else
    +return "no"
  +while condition                   — loop while condition is true
    +set i = i + 1
  +return expr                       — return from function
  +each collection item:Type         — loop over collection (body indented below)
    +call result:Type = process(item)

### Built-in Functions
  concat(a, b)              — concatenate two strings
  char_at(s, i)             — get character at index i as a String
  substring(s, start, end)  — get substring from start to end
  starts_with(s, prefix)    — check if string starts with prefix
  contains(s, substr)       — check if string contains substring
  index_of(s, substr)       — find index of substring (-1 if not found)
  split(s, delim)           — split string into List<String>
  trim(s)                   — remove leading/trailing whitespace
  to_string(x)              — convert any value to String
  len(x)                    — length of string or list (also x.len)
  list()                    — create empty list
  push(list, item)          — returns NEW list with item appended (functional, not mutating)
  get(list, index)          — get item at index
  join(list, delim)         — join list items into string with delimiter
  abs(x), sqrt(x), pow(x,y), floor(x), min(x,y), max(x,y)
  Ok(value), Err(label)     — Result constructors

### Modules
+module Name
  +type User = {id:Int, name:String}
  +fn create (input:CreateReq)->Result<User> [io,fail]
    +call validated:CreateReq = validate(input)
    +return validated
end

### Expressions
- Literals: 42, 3.14, true, false, "hello"
- Field access: user.name, input.age
- Function calls: validate(input), db.insert(user)
- Comparison: x>=0, name!="", age<=150
- Logic: condition1 AND condition2, NOT is_empty
- Arithmetic: x + 1, total * 2
- Struct literals: {name: "alice", age: 25}

### Edit Operations
!replace function_name.s1            — replace statement 1 of a function
  +check age input.age>=0 AND input.age<=150 ~err_age_range

### Testing
Test blocks do NOT use `end`. They end at the next unindented line or end of input.
For single-param functions, pass the value directly.
For multi-param functions, use space-separated key=value pairs (named after the parameters).

!test double
  +with 5 -> expect 10

!test add
  +with a=3 b=4 -> expect 7

!test validate
  +with name="alice" age=25 -> expect Ok
  +with name="" age=25 -> expect Err(err_empty_name)

### Evaluating
!eval function_name arg1=val1 arg2=val2
Calls a function and shows the result. For single-param: !eval double 5

### Tracing
!trace function_name arg1=val1 arg2=val2
Shows step-by-step execution of a function with the given input.

### Semantic Queries
?symbols                    — list all defined types and functions
?symbols function_name      — show details of a specific function
?callers function_name      — who calls this function?
?effects function_name      — what effects does this function have?
?type TypeName              — show type definition

## Important Rules

1. Every function must declare its effects. Pure functions have no brackets.
2. Every binding must have an explicit type annotation.
3. Use descriptive error labels with ~ for checks: ~err_negative_age, ~err_empty_name
4. One statement per line. Indentation marks nesting (2 spaces).
5. Only modules end with `end`. Functions and test blocks do NOT use `end`.
6. Use +if/+elif/+else for conditional logic, +check for validation assertions.
7. No closures, no inheritance, no operator overloading, no exceptions.
8. String concatenation uses concat(), not +.
9. Result types use Ok/Err, Option types use Some/None.
10. Modulo operator: use % for remainder (e.g., n % 2 == 0 for even check).
11. Error handling — two patterns:
    a. Auto-propagate (like Rust's ?): declare [fail] and bind as plain T:
       `+call validated:Input = validate(input)` — errors bubble up, you get T on success.
    b. Explicit handling: bind as Result<T>:
       `+call result:Result<Input> = validate(input)` — use result.is_ok, result.unwrap, result.is_err.
    Use pattern (a) when your function also has [fail]. Use pattern (b) when you want to handle errors yourself.

## Example 1: Validation with checks

<code>
+type Request = {age:Int, email:String, name:String}

+fn validate (input:Request)->Result<Request> [fail]
  +check age input.age>=0 ~err_negative_age
  +check age_max input.age<=150 ~err_age_too_high
  +check name input.name.len>0 ~err_empty_name
  +return input

!test validate
  +with age=25 email="foo@bar.com" name="alice" -> expect Ok
  +with age=-1 email="foo@bar.com" name="alice" -> expect Err(err_negative_age)
  +with age=200 email="foo@bar.com" name="alice" -> expect Err(err_age_too_high)
  +with age=25 email="foo@bar.com" name="" -> expect Err(err_empty_name)
</code>

## Example 2: Conditional logic with if/elif/else

<code>
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
</code>

When the runtime reports errors, fix them with targeted !replace operations or by regenerating the affected function.

## Important Workflow Notes

- Each <code> block you send is applied to a FRESH program state. Include ALL types and functions you need in each response.
- Work step by step — I'll validate each response and give you feedback.
- When everything passes, put just `DONE` in your <code> block.
"#.to_string()
}

/// Build the initial user message for a task.
pub fn task_message(task: &str) -> String {
    format!(
        "Implement the following in Forge:\n\n{task}\n\n\
         Start by defining any types you need, then implement the functions. \
         Include !test blocks to verify your implementation. \
         Work step by step — I'll validate each response and give you feedback."
    )
}

// --- Architect mode prompts ---

/// System prompt for architect mode — teaches the two-phase workflow.
pub fn architect_system_prompt() -> String {
    let base = system_prompt();
    let prompt = base.replace(
        "## Important Workflow Notes\n\n\
         - Each <code> block you send is applied to a FRESH program state. Include ALL types and functions you need in each response.\n\
         - Work step by step — I'll validate each response and give you feedback.\n\
         - When everything passes, put just `DONE` in your <code> block.\n",
        "## Architect Workflow\n\n\
         You work in two phases:\n\n\
         **Phase 1 — Design:** Define all types and function signatures with stub bodies.\n\
         Stub bodies should be minimal: just `+return 0` for Int, `+return \"\"` for String, `+return input` for struct returns.\n\
         Do NOT include !test blocks in the design phase — stubs will fail tests.\n\
         The runtime will validate that types and signatures are consistent.\n\n\
         **Phase 2 — Implement:** You will be asked to implement one function at a time.\n\
         The runtime tells you which function to implement and shows you the full program context.\n\
         Write ONLY the function being requested with `+fn` and `!test` blocks.\n\
         The runtime keeps previous functions — you don't need to repeat them.\n\n\
         **Key patterns to remember:**\n\
         - Multi-param tests use key=value: `+with a=3 b=4 -> expect 7`\n\
         - Error auto-propagation: `+call val:T = func(x)` with [fail] — errors bubble up, you get T\n\
         - If a function calls another [fail] function, declare [fail] on the caller too and bind as plain T\n\
         - Do NOT create wrapper types just for tests — use key=value syntax\n\n\
         When all functions are implemented and tests pass, respond with `DONE`.\n",
    );
    prompt
}

/// Initial message for architect mode — asks for the design.
pub fn architect_design_message(task: &str) -> String {
    format!(
        "Design the architecture for the following system in Forge:\n\n\
         {task}\n\n\
         **Phase 1 — Design only.** Define all types and function signatures with stub bodies.\n\
         Use `+return 0` for Int returns, `+return \"\"` for String, `+return input` for struct returns.\n\
         Do NOT write !test blocks — stubs will fail tests. Tests come in Phase 2.\n\
         Do NOT create wrapper types for tests — Phase 2 uses key=value syntax.\n\n\
         Focus on:\n\
         - What types are needed\n\
         - What functions exist, their parameters, return types, and effects\n\
         - If a function calls another [fail] function, the caller needs [fail] too\n\n\
         I'll validate the architecture, then ask you to implement each function one at a time."
    )
}

/// Message asking the model to implement a specific function.
pub fn architect_implement_message(function_name: &str, program_state: &str) -> String {
    format!(
        "**Phase 2 — Implement `{function_name}`.**\n\n\
         Current program state:\n\
         {program_state}\n\n\
         Write the implementation for `{function_name}` and its tests.\n\n\
         Rules:\n\
         - Write ONLY `+fn {function_name} ...` and `!test {function_name}`\n\
         - For multi-param tests use key=value: `+with a=3 b=4 -> expect 7`\n\
         - For struct-param tests: `+with name=\"alice\" age=25 -> expect Ok`\n\
         - If calling a [fail] function, bind as plain T (errors auto-propagate):\n\
           `+call validated:Input = validate(input)` — NOT `+call result:Result<Input> = ...`\n\
         - The runtime will merge this into the existing program"
    )
}

/// Feedback after design phase.
pub fn architect_design_feedback(
    results: &[(String, bool)],
    program_state: &str,
    stub_functions: &[String],
) -> String {
    let mut msg = String::new();

    msg.push_str("## Design Validation\n\n");
    for (result, success) in results {
        if *success {
            msg.push_str(&format!("OK: {result}\n"));
        } else {
            msg.push_str(&format!("ERROR: {result}\n"));
        }
    }

    let all_ok = results.iter().all(|(_, s)| *s);

    if all_ok {
        msg.push_str("\nArchitecture validated. Functions to implement:\n");
        for name in stub_functions {
            msg.push_str(&format!("  - {name}\n"));
        }
        msg.push_str("\nI'll now ask you to implement each function one at a time.\n");
    } else {
        msg.push_str("\nThere were errors in the architecture. Fix them and resend all types and signatures.\n");
    }

    msg.push_str(&format!("\n{program_state}"));
    msg
}

/// Build feedback message after validating the model's output.
pub fn feedback_message(
    results: &[(String, bool)],
    test_results: &[(String, bool)],
    program_state: &str,
) -> String {
    let mut msg = String::new();

    msg.push_str("## Validation Results\n\n");
    for (result, success) in results {
        if *success {
            msg.push_str(&format!("OK: {result}\n"));
        } else {
            msg.push_str(&format!("ERROR: {result}\n"));
        }
    }

    if !test_results.is_empty() {
        msg.push_str("\n## Test Results\n\n");
        for (result, success) in test_results {
            if *success {
                msg.push_str(&format!("PASS: {result}\n"));
            } else {
                msg.push_str(&format!("FAIL: {result}\n"));
            }
        }
    }

    let all_ok = results.iter().all(|(_, s)| *s) && test_results.iter().all(|(_, s)| *s);

    if all_ok {
        msg.push_str("\nAll validations and tests passed. ");
        msg.push_str(
            "If the implementation is complete, respond with just `DONE` in your <code> block. ",
        );
        msg.push_str("Otherwise, continue adding more functionality or tests.\n");
    } else {
        msg.push_str("\nThere were errors. Fix them using !replace for specific statements, ");
        msg.push_str("or regenerate affected functions.\n");
    }

    msg.push_str(&format!("\n{program_state}"));
    msg
}
