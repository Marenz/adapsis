# Typed Python frontend

These files use Python syntax and execute in Adapsis's typed AST runtime. They
are not run by CPython and do not import or execute Python packages.

## Run locally, without a model or running service

```sh
cargo build --release
./target/release/adapsis test examples/python/basic.py --language python --module PyBasic --tests examples/python/test_basic.py
./target/release/adapsis test examples/python/flow.py --language python --module PyFlow --tests examples/python/test_flow.py
```

The command requires both the module namespace and paired assertion file. It
uses the existing isolated test runtime, reports `TEST SUMMARY`, and exits
nonzero for a failed assertion or unsupported syntax. Existing `.ax` test
commands retain their behavior.

## Native tool workflow

Use `mutate` with `{"language":"python","module":"PyBasic","source":"..."}`,
then `test` with the same language/module and the assertion file's source.
Evaluate using `{"function":"PyBasic.add_two","language":"python","input":"3"}`.
Inspect with `{"kind":"source","target":"PyBasic.add_two","format":"python"}`.
Call `finish` explicitly after testing. Functions changed in the turn still owe
their ordinary Adapsis tests; Python syntax does not bypass the completion gate.

Omitting `language` or source `format` preserves Adapsis syntax. Python `module`
metadata selects a namespace, never permission. Live gateway compatibility is
a separate deployment/certification gate; these local examples need no gateway.

## Implemented surface (parity work in progress)

Full `.ax` parity is **not yet verified**. The strict `python_parity` inventory
reports invalid original fixtures separately from unsupported constructs, exact
AST differences and non-idempotent printing. No fixture exclusions count as passes.
Boolean evaluation also differs between the existing execution tiers (#90), so
the frontend retains its call-free, statically Boolean `and`/`or` restriction.
The entries below describe implemented frontend forms, not a whole-corpus or
live-provider certification; native tools retain their additional admission rules.

- Annotated functions, parameters, returns, and new local bindings; reassignment
  uses `name = expression`. Module/function docstrings become docs.
- Types: `int`, `float`, `bool`, `str`, `Byte`, `list[T]`, `Map[K, V]`, `Set[T]`,
  `Option[T]`, `Result[T]`, and named struct/union types. Structs are `@dataclass`
  classes (keyword-arg construction + field access); tagged unions are `@union`
  classes with `Variant: tuple[...]` payload attributes (`tuple[()]` = none;
  the `None` variant is written `None_`).
- Statements: `if`/`elif`/`else`, `while`, `for`, and `match` with `case` patterns
  (declared variants, `Ok`/`Err`/`Some`/`None`, nested class patterns, literal
  sub-patterns, `_`). A match must be exhaustive. Match subjects may call IO
  builtins directly. Module-level `name: Type = literal` declares a `+shared`
  variable (bare-name reads, plain-assignment writes).
- Numeric `+`, `-`, `*`, comparisons, typed string `+`, and Boolean `and`/`or`/`not`.
  Integer division and remainder are the reserved helpers `div(a, b)` / `rem(a, b)`
  (Python `/` and `%` have different semantics and are rejected).
- Reserved helpers: `propagate(f(x))` (`+call`), `check("label", cond, "err")`
  (`+check`), `spawn(f(x))` (`+spawn`), `source_add/remove/replace/list(...)` and
  `event_register/emit(...)`. Module-level `@route("METHOD", "/path")` and
  `@source(name, type=..., k=v)` decorators; reserved `async def startup()/shutdown() -> None`.
- Effects: `@effects("io", "fail", "mut", "rand", "yield", "parallel", "unsafe")`
  plus `async` from `async def`. Await requires an annotated direct-call binding.
  Awaiting a Result-returning **user function** is rejected ([#82]); direct IO
  builtin awaits keep their Result wrapper.
- Call-site exceptions apply only to the root call. A direct unshadowed IO builtin
  match subject needs an `io`+`async` caller; a user async function cannot use that
  exception. Nested call arguments keep ordinary call rules. `spawn` requires an
  async caller and a qualified user-function target after lowering, plus `io` if
  the target has it. It schedules a runtime task, not a native LLM agent.
  The shared runtime-argument validator visits match arms, checks, spawns and
  lifecycle bodies; native mutation still refuses lifecycle/source/route forms.
  Encoded source operations are explicitly refused by that validator rather than
  having timer/handler execution silently bypass call checks.
- Test matchers: `assert contains(f(x), "s")`, `assert starts_with(...)`,
  `assert is_ok(f(x))` / `assert is_err(f(x))`, `assert f(x) == Err("msg")`, and
  side-effect checks `assert_after("target", "matcher", "value")`.

Rejected with source locations: imports, arbitrary decorators, defaults/variadic
parameters, exceptions, lambdas, comprehensions, f-strings, subscripts,
`break`/`continue`, `**`, truthiness-based selection, and short-circuiting
`and`/`or` with calls. Integer literals range from −9223372036854775807 to
9223372036854775807.

The AST remains the sole persisted source of truth. Python inspection prints
canonical supported source after session/library reload. Unsupported legacy AST
constructs produce a precise error; `format:"adapsis"` remains available. Source
comments and original formatting are not preserved. Submitting a function
replaces it as Adapsis declarations do, so resubmitted functions may need tests
again. No full-file deletion or edit-diff behavior is inferred.

Native Python mutation checks syntax and direct-reference permissions before
consulting permitted runtime signatures. This supports inspecting and resubmitting
one function that calls existing functions, without resending its entire module.
Python source inspection also excludes unauthorized signatures and can truthfully
refuse a wrapper whose referenced signatures are unavailable to the caller.

Python execution/inspection conservatively rejects current builtin-name collisions
(both pure and IO registries, including aliases), even in otherwise authorized
modules. This protects installed Python after later `.ax` changes. Narrower
helper-specific usability is deferred to [#83](https://github.com/Marenz/adapsis/issues/83);
the broad safe check is intentional in this first release.

## Direct IO match runtime limitation (#94)

The no-await direct builtin `match http_get(url):` form is admitted by the
frontend, but it is **not VM-portable after an earlier IO operation in the same
function**. The VM's compiled direct IO expression has no coroutine handle, and
whole-function fallback is deliberately forbidden after IO because it would
replay completed side effects. Prefer binding the result first:

```python
@effects("io")
async def fetch_ok(url: str) -> str:
    response: Result[str] = await http_get(url)
    match response:
        case Ok(value):
            return value
        case Err(error):
            return error
```

This form is covered by native mutation/evaluation with an in-memory HTTP
fixture. It uses a **direct runtime IO builtin** await; awaiting a
Result-returning user function remains unsupported (#82). Never retry the whole
function or fall back after IO. #93 fixes dedicated Result value matching, not
IO expression portability.

## Shared declarations and constructor-name safety

Module-level `name: Type = default` declares a `+shared` variable. Native Python
mutation requires an explicit annotation and a data-only default: scalars, checked
list literals, record constructors, and variant constructors (`Leaf(7)`, `Empty`)
whose payloads match the declared annotation. A zero-payload variant uses the bare
constructor spelling (`kind: Kind = Empty`); `Leaf(7)` uses the call form. Empty
`[]` inherits its element type from the annotation; any other unresolved default is
refused. Nested calls or
references inside a list, record or variant default are rejected, not silently
accepted by an outer literal check. A variant default cannot be combined with a
same-named function declaration: the function/variant collision guards still apply.

A Python-authored function may not share a spelling with a union variant declared
in another module. The tree walker prefers variants and the VM prefers functions,
so mutation refuses the collision rather than installing a program whose
behaviour depends on the execution tier. A legacy program that already contains
one keeps evaluating unrelated functions; only entries whose reachable call graph
hits the ambiguous spelling are refused.
