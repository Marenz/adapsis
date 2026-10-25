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

## Supported subset

- Annotated functions, parameters, returns, and new local bindings. Reassignments
  use ordinary `name = expression`. Module/function docstrings become docs.
- `int`, `float`, `bool`, `str`, `list[T]`, and effectful `Result[T]` declarations.
- `if`/`elif`/`else`, `while`, and `for` over statically typed read-only lists.
  A loop target must be fresh and used only in its loop body. Newly declared
  block locals cannot escape; previously declared variables may be reassigned.
- Numeric `+`, `-`, `*`, single comparisons, and typed string concatenation.
  Boolean operators require total, call-free Boolean operands.
- List literals, iteration, and `len(list_value)` (not string length); local calls resolve within the explicit
  module, and cross-module calls use `Module.function(...)` without imports.
- Test files contain zero-argument `test_*` functions, each with one equality
  assertion of a function call against a scalar or read-only list literal.
  Calls require exactly the declared argument count or keyword names. Parameter
  and keyword names `_0`, `_1`, … are reserved for runtime positional carriers.
- An inert `@effects("io", "fail", "mut")` declaration and `async def` describe
  runtime effects. Await requires an annotated direct-call binding. IO tests
  need trusted runtime mocks; Python cannot install mocks or enable live IO.
  Pure async calls need await but do not need an IO effect; synchronous calls
  cannot be awaited. Awaiting a Result-returning **user function** is currently
  rejected because the inherited runtime unwraps that result ([#82](https://github.com/Marenz/adapsis/issues/82)).
  Direct IO builtin awaits retain their Result wrapper and remain supported.

This is a deliberately bounded language. Imports, classes, arbitrary decorators,
defaults/variadic parameters, exceptions, lambdas, comprehensions, f-strings,
mutable list methods, subscripts, `break`/`continue`, loop `else`, `/`, `//`, `%`,
`**`, and truthiness-based operand selection are rejected with source locations.
Ordinary functions require a value return; `-> None` is reserved for test wrappers.
Integer literals currently range from −9223372036854775807 to
9223372036854775807: the existing canonical parser cannot yet represent
−9223372036854775808. Numeric storage/overflow follows the typed Adapsis runtime,
not arbitrary-precision CPython integers.

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
