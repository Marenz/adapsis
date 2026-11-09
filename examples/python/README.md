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

Each test definition contains one assertion. Within one submission, definitions
for the same exact qualified target form one ordered test set: all cases must pass
before that set is stored. Targets execute in first-seen order, so interleaved
targets are regrouped; cases within each target retain source order. A later
submission replaces that target's stored set rather than appending stale cases.

An unrelated legacy builtin-name collision no longer blocks a narrow scalar-only
test/evaluate/source-inspection entry: scalar parameters/return, fresh scalar local
assignments then return, with literals, bound names and unary/binary operators only.
Execution also requires scalar literal test data/arguments (no matcher/after-check),
and all implicitly initialized shared defaults must be proven scalar and call-free.
Inspection uses authorized metadata and does not initialize defaults. This proof is
rechecked after edits. Calls, fields/pseudo-fields, constructors, effects, control
flow, list data and unproved defaults still use the conservative collision gate;
this is a bounded exception, not complete support for legacy helper collisions.

The lowerer validates the structural operation IR before execution:
unsupported casts, misplaced operations, malformed call/pattern/type shapes and
source/event fragment injection are refused independently of the `.ax` roundtrip.
This is an additional guard, not a replacement for type/effect checking or native
authorization. Structural-IR errors currently report a submission-level location
and operation path. Native Python and CLI tests now use these typed operations
directly. `legacy_export()` is an explicit, fallible compatibility export/oracle;
it is not an execution or replay fallback. Library `.py` activation remains a
separate #121 gate; retained legacy library hooks can still write `.ax`.

Python mutation history keeps exact submitted source and module/version metadata
alongside the actual affected-module outcome, including rejected/partial updates.
New stored tests carry typed data for save/load and automatic re-testing. Old
session strings remain readable as legacy input. Rewind is staged and fails
without replacing live state if historical outcomes cannot be reconstructed.
New Python declaration edits after rewinding to an earlier revision are currently
refused until an explicit history-branch policy exists; inspection/evaluation/tests
remain available. Future history is not silently deleted. This is completed-state
recovery, not in-flight exactly-once execution or a replacement for library conflict
checks, and it does not reconstruct historical standalone test events.

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
- Runtime properties, written as ATTRIBUTES (the only spelling that lowers to the same
  AST as `.ax`): `s.len` / `s.is_empty` on `str` or `list[T]` (`str` length is BYTES),
  and `r.is_ok` / `r.is_err` / `r.unwrap` on `Result[T]` (`unwrap` has the inner type;
  `r.unwrap.name` chains into a struct). They are type-gated: a struct field of the same
  name always wins; `int`, `Map`, `Set`, `Option`, `.len` on a `Result`, and the names
  `error`, `unwrap_err`, `is_some`, `is_none` are refused. This text is **not runnable
  CPython** (`str` has no `.len`); like `Ok(..)` and `@union` it exists only for this
  runtime. Functions that use them always execute on the tree walker (#114).
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
  Awaiting a Result-returning **user function** preserves its Result wrapper ([#82]);
  bind it as `Result[T]`, then match it. A plain `T` await binding is refused. Direct
  IO builtin awaits retain their existing Result behavior.
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

Names that read as Python execution escapes (`eval`, `exec`, `open`, `getattr`, `setattr`,
`delattr`, `globals`, `locals`, `compile`, `vars`, `import`, and anything containing `__`)
are rejected everywhere. `input` is the one exception, and only as a *binding*: it may be a
parameter, local, loop target, match capture or the base of a field access. It is still
rejected as a callee (`input(x)`), function, type, member, shared-variable, annotation or
attribute name, so it can shadow but never call Python's stdin builtin.

Python execution/inspection conservatively rejects current builtin-name collisions
(both pure and IO registries, including aliases), even in otherwise authorized
modules. This protects installed Python after later `.ax` changes. Narrower
helper-specific usability is deferred to [#83](https://github.com/Marenz/adapsis/issues/83);
the broad safe check is intentional in this first release.

## Pure JSON presence/type queries

`json_kind(payload: str, path: str) -> str` is a pure builtin; do not await it.
Its closed labels are `missing`, `null`, `bool`, `int`, `float`, `string`, `array`,
`object`. It can distinguish missing from present-empty and null from a String
containing `null`, without changing `json_get`'s existing polymorphic results.

```python
@effects("fail")
def value_kind(payload: str, path: str) -> str:
    return json_kind(payload, path)
```

Both helpers share the same existing path resolver: dot-separated components,
empty components ignored (empty/all-dot path is root), `usize`-like components
are array indices including `+0`/`00`, otherwise literal object keys. Numeric
object keys therefore do not become addressable via array-like components.
No bracket/escape grammar is added. Wrong containers/out-of-bounds are `missing`;
wrong argument type/arity and malformed JSON are errors.

`int` requires serde's represented signed-i64 integer. Represented u64 above
i64 errors; finite floating representation is `float`, including 1.0/-0.0 and
oversized integer lexemes serde rounded into floats. Parser overflow errors.
This deliberately does not claim lexical numeric precision. The pending text
port's heuristics are not replaced merely by adding this primitive.

Native evaluate currently renders an evaluated failure as `Err(json_kind...)`
even when its aggregate `has_errors` remains false. Such output is an error, not
one of the eight valid kind Strings; the primitive does not repair that existing
native status-reporting limitation.

`json_valid(payload: str) -> bool` is a pure total builtin; do not await it. It is true
iff the shared JSON parse step used by `json_get`/`json_kind`/`json_array_len`
succeeds; it is **not strict RFC 8259**. Nesting deeper than 127 containers and
non-finite numbers such as `1e400` are false. It never errors for any `str`:
empty, truncated, BOM-prefixed, HTML or trailing-garbage text is simply `False`.
Validity is not representability: `18446744073709551615` is valid while
`json_kind` still errors on it. Wrong arity or a non-`str` argument is a lowering error.

```python
def usable(body: str) -> bool:
    return json_valid(body)
```

## Result returns and IO boundaries (#82, #86 G7)

A function declaring `Result[T]` may return a plain, statically known value of **exactly T**.
The runtime wraps it once at invocation; lowering keeps the original Return AST and does
not insert `Ok(...)`. This is an **Adapsis-specific return rule, not CPython inference**.
It does not widen assignments, arguments, numeric conversions or general `compatible()`
checking. Named types retain declaration owners. `Result[Result[T]]` and
`Result[Option[T]]` require explicit outer wrapping, e.g. `return Ok(inner)`:
Some and Result success share the runtime's `Value::Ok` representation. A
`Result[list[Option[T]]]` can still implicitly wrap a known list (its outer value is distinct).
Existing explicit-constructor/catalog payload-validation limitations remain unchanged;
runtime nested-wrapper checks still apply.

```python
async def checked_value(n: int) -> Result[int]:
    return n

async def consume(n: int) -> int:
    result: Result[int] = await checked_value(n)
    match result:
        case Ok(value):
            return value
        case Err(error):
            return 0
```

The same-source, existing-signature and inspection/resubmission paths use this rule.
Async and IO-effect requirements still apply; user Result awaits do not confer authority.
Native tests remain literal-data-only: use scalar consumers when checking Result behavior.
Native Python evaluate/test preflights the current entry's reachable user awaits for
outer Result/non-Result binding agreement, in both directions after legacy edits.
This is neither general payload-type validation nor a new transitive permission policy;
diagnostics do not disclose hidden callee names or schemas.

The no-await direct builtin `match http_get(url):` form is admitted by the frontend.
Non-Await builtin IO, user awaits and declared Result graphs select the interpreter
before execution; the VM's direct IO expression has no coroutine handle. Whole-function
fallback after execution is forbidden because it would replay completed effects.
An annotated binding makes the wrapper explicit:

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
fixture. User Result-returning awaits now preserve the same wrapper, but production VM/JIT
Result quarantine is not lifted by frontend acceptance. Never retry a whole function or
fall back after IO. This does not certify every #94 runtime/VM combination.

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
