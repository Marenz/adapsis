# FORGE

**An AI-First Programming Language**

*Language Design Document v0.1 — March 2026*

A programming language designed for LLMs to write, read, debug, and reason about.
Flat. Concrete. Provenance-tracked. Grammar-constrained.

---

## 1. Executive Summary

**Forge** is an AI-native program representation and mutation protocol with a constrained surface language. It is not primarily "the next programming language" — it is a system for AI-assisted software construction, where the AST is the source of truth, changes happen through incremental mutations with immediate compiler feedback, and the language is the surface that makes the protocol work.

The core insight: the real failure mode of LLM code generation is not syntax errors — it's maintaining global consistency over time under limited context. Forge attacks this directly through an interactive compilation loop where the model emits small, validated mutations rather than generating complete programs blind.

The innovation is the loop, not the syntax. The AST runtime, the mutation protocol, the semantic query system, and the browser-based multi-representation interface are the real contribution. The language design serves these by being flat, explicit, and easy to validate — but the language alone is not the point.

**Key proposition:** Incremental, feedback-driven program construction produces more correct software than one-shot code generation, and a purpose-built language and protocol make that loop maximally effective.

**End goal:** Forge is expressive enough to implement itself — its own AST runtime, compiler, and provenance tracker — closing the loop where an LLM improves its own toolchain in the language designed for it.

---

## 2. Motivation: Why Existing Languages Fail LLMs

Current programming languages were designed for humans with keyboards and screens. LLMs writing these languages face systematic problems that no amount of prompting can solve.

**Syntactic noise.** Brackets, semicolons, indentation-as-semantics, matching delimiters across long spans. Every closing brace that matches an opening brace 200 tokens ago is a long-range dependency the model must track. Off-by-one delimiter errors are a classic LLM failure mode.

**Overloaded semantics.** `+` means addition, string concatenation, and list merge depending on context. `this` in JavaScript changes meaning based on call site. Operator overloading and implicit coercions force the model to maintain type-state it is unreliable at tracking.

**Implicit control flow.** Exceptions, decorators, metaclasses, macro expansion, implicit constructors and destructors. Vast amounts of behavior invisible in the local token window.

**Name-based reasoning.** Understanding code requires chasing reference chains through potentially thousands of tokens: knowing that `usr_mgr` is a UserManager instance with method `get_active()` returning `List<User>`. This is exactly the kind of long-range dependency LLMs handle poorly.

**Multiple equivalent forms.** List comprehensions, for-loops, map/filter, and generator expressions can all express the same logic. LLMs waste context window and decision budget navigating stylistic variety that carries no semantic difference.

---

## 3. Core Architecture

Forge is not just a language. It is an integrated system with four tightly coupled components.

### 3.1 The Language

A flat, concrete, keyword-heavy programming language with explicit types, explicit effects, no nesting beyond one level, and exactly one syntactic form per semantic concept. Designed so that any valid token sequence in the grammar is a structurally sound program fragment.

### 3.2 The Grammar Constraint

A GBNF (or equivalent) grammar that runs in the inference loop, masking invalid tokens at every generation step. Syntax errors become impossible by construction. The grammar and the language are co-designed: the language is defined as whatever the grammar accepts.

### 3.3 The AST Runtime

A persistent service that maintains the program's abstract syntax tree. The model does not generate source code text. Instead, it emits AST mutation operations at variable granularity. The runtime applies each mutation, validates it, and returns the updated state or error diagnostics for the next generation step.

### 3.4 The Provenance Tracker

A system that tracks which code was generated from the same intent or pattern. When the model stamps out concrete implementations of the same logic for different types, the tracker maintains links between them. Bug fixes propagate through these links, giving the safety benefits of DRY without encoding abstraction in the language.

---

## 4. Language Design

### 4.1 Design Principles

**Shallow bounded nesting, not zero nesting.** Deep nesting is bad for LLMs, but banning it entirely makes code verbose and full of pseudo-gotos. The rule is: one level of nesting for match arms, loop bodies, and branch blocks. No arbitrary depth. The canonical internal IR stays flat; the human-facing representation can show structured nesting. The model works with the flat form; humans see the structured form.

**Statements over expressions, but not dogmatically.** Deeply nested expressions like `f(g(x) + h(y))` should be broken into sequential statements with named intermediates. But simple one-level expressions within a statement are fine — `input.age >= 0` doesn't need to be three separate operations.

**Types are always inline and visible.** No type inference to chase. Every binding carries its type in the token stream. Redundant for humans, essential for a model that cannot reliably look up a definition thousands of tokens away.

**No overloading, no implicit coercion, one way to do things.** Every operation has exactly one meaning. String concatenation is `concat`, not `+`. There is no polymorphic dispatch. If interfaces exist, the resolved concrete type is annotated at every call site.

**Explicit effect markers.** Every function signature declares what it does: IO, mutation, failure, async, randomness. The model never has to guess whether a function is pure.

**Statements, not expressions.** Expressions nest; statements don't. Instead of `f(g(x) + h(y))`, three sequential statements with named intermediates. This keeps the grammar simple and the generation tree shallow.

**Keyword-heavy, punctuation-light.** Keywords are single tokens the grammar can match exactly. Punctuation-based syntax with matching delimiters across spans is harder for both the grammar and the model.

### 4.2 Type System

Go-level expressiveness with Rust-level type explicitness:

- **Primitive types:** Int, Float, Bool, String, Byte
- **Tagged unions / sum types:** Models handle these well; they map directly to branch trees
- **Product types / structs:** Named fields, no inheritance
- **Option and Result:** Built-in, not library types. No null.
- **Collections:** List, Map, Set as built-in types with known semantics
- **Simple generics ("boring generics"):** `List<User>`, `Map<String, Config>` — fully monomorphized at compile time. No trait bounds, no associated types, no type-level computation. The model always sees and works with the concrete monomorphized version in the feedback loop, never reasons about generic code directly. This is the compromise between fully concrete stamping (which explodes code volume) and full parametric polymorphism (which requires complex type-level reasoning).

### 4.3 Expressiveness Level

The language operates at two layers:

**Core layer** (roughly Go/early Pascal level): concrete types, first-order functions, explicit everything, simple control flow via branch/loop/sequence. This is what the model generates directly.

**Composition layer:** interfaces (no inheritance, no nesting), parameterized modules that stamp out concrete versions, and pattern matching over tagged unions. Complexity lives in the module composition graph, not in any individual function.

### 4.4 Deliberately Excluded Features

| Feature | Status | Rationale |
|---------|--------|-----------|
| Simple generics (`List<T>`) | **Allowed** | Fully monomorphized, no trait bounds, no type-level computation. Model sees concrete types only. |
| Named function references | **Allowed** | `sort(users, compare_by_age)` is fine — the reference is named, typed, and inspectable. |
| Closures / capturing lambdas | **Excluded** | Anonymous logic with captured state creates unnamed dependencies the model can't track. |
| Type-level programming | **Excluded** | Trait bounds, associated types, GATs — anything that requires reasoning about types abstractly. |
| Operator overloading | **Excluded** | Same syntax, different semantics depending on type context. Use distinct named operations. |
| Inheritance | **Excluded** | Implicit behavior through prototype chains. Use composition via interfaces. |
| Macros | **Excluded** | The model IS the macro system. It generates repetitive code directly. |
| Implicit conversions | **Excluded** | Hidden type coercion is a hallucination vector. Use explicit conversion functions. |
| Exceptions | **Excluded** | Invisible control flow. Use Result types with explicit propagation. |

*The model itself replaces most abstraction mechanisms. The allowed features (boring generics, named function references) are the minimum needed to avoid clumsy APIs, while everything that requires non-local reasoning or creates hidden dependencies is excluded.*

### 4.5 Syntax

The mutation format and the code format converge. A function body is a list of add-statement operations. There is no separate "code" versus "edit operations" layer.

```
+fn validate (input:Request)->Result<Valid> [fail]
  +check age input.age>=0 ~err_negative
  +check email is_valid_email(input.email) ~err_email
  +return input

+fn process (req:Request)->Response [io,fail]
  +call valid:Valid = validate(req)
  +call record:DbRecord = db.insert(valid)
  +call resp:Response = build_response(record)
  +return resp
```

Where `+` means add, `~` means on-fail-goto, and structure is positional rather than labeled. After `+fn` comes name, params, return type, effects — always in that order. The model learns the positional grammar; no labels or delimiters needed.

### 4.6 Multi-Scale Mutations

The model chooses its granularity based on confidence and complexity:

| Granularity | Operation | Use case |
|-------------|-----------|----------|
| Scaffold | `+module` with type stubs and fn signatures | Bootstrapping a new service |
| Function | `+fn` with complete body | Confident, straightforward logic |
| Statement | `+stmt` / `!replace target.id` | Precise edits, bug fixes |
| Expression | `!patch target.id.expr` | Sub-expression tweaks |

Scaffolding a CRUD service: one or two big operations. Debugging a subtle race condition: fine-grained single-statement mutations with compiler feedback after each one.

```
// Coarse: scaffold an entire module
+module UserService
  +type User = {id:Id, name:String, email:String}
  +fn create (input:CreateReq)->Result<User> [io,fail]
    +call validated:CreateReq = validate_create(input)
    +call user:User = db.insert_user(validated)
    +return user
  +fn get (id:Id)->Option<User> [io]
    +call result:Option<User> = db.find_user(id)
    +return result
end

// Fine: fix a single statement
!replace UserService.create.s1
  +check name input.name.len>0 ~err_empty_name
```

---

## 5. Generation Pipeline

### 5.1 Two-Phase Generation

The model operates in two distinct phases per step:

**Phase 1 — Unconstrained thinking.** The model reasons in natural language about the problem: approach, edge cases, data structures, trade-offs. No grammar constraint. This is where algorithmic creativity lives.

**Phase 2 — Constrained code emission.** The model generates a Forge AST mutation under the GBNF grammar. Decisions are mostly made; the model is translating its plan into the target format, and the grammar ensures well-formedness.

```
<think>
The validate function needs to check age is non-negative
and email matches a basic pattern. On failure, return
specific error types so the caller can distinguish them.
</think>
<code>
+fn validate (input:Request)->Result<Valid> [fail]
  +check age input.age>=0 ~err_negative_age
  +check email is_valid_email(input.email) ~err_bad_email
  +return input as Valid
</code>
```

The grammar activates only after the `<code>` tag. The thinking phase remains fully unconstrained. Forcing thinking into the grammar would be destructive — the model needs free-form space to explore and backtrack.

### 5.2 Interactive Compilation Loop

Instead of generating a complete program in one shot, the model builds incrementally with continuous feedback:

1. Model emits an AST mutation (constrained by grammar)
2. Runtime applies the mutation and validates (type-check, effect-check)
3. Result is injected back into context: success confirmation, new symbol table, or error diagnostics
4. Model emits the next mutation, informed by the result
5. Repeat until program is complete

This sidesteps the biggest weakness of one-shot generation: the model no longer needs to hold the entire program's consistency in its head. The AST runtime acts as external memory and external type-checker. The model only needs to make one locally-correct decision at a time.

It also solves the context-sensitive limitation of GBNF grammars. The runtime provides context-sensitive information (current symbols, types, available functions) as injected text between generation steps. The grammar stays simple and context-free.

### 5.3 Semantic Queries

The model can emit query operations between mutations:

```
?symbols UserService          // what's defined here?
?callers db.insert_user        // who calls this?
?effects UserService.create    // what effects does this have?
?similar UserService.create.s1 // structurally similar code?
```

The runtime answers with structured data that becomes part of the model's context for the next generation step. This gives the model the benefits of a full semantic index without needing the entire codebase in context.

---

## 6. Provenance and Propagation

Provenance tracking is one of Forge's most novel features — and its most dangerous. It manages relationships between related code fragments at the tooling level, enabling bug fixes to flow between structurally similar code. But without a careful trust model, it's a confident bug copier. This section defines both the mechanism and its safety constraints.

### 6.1 When Provenance Matters

With boring generics (Section 4.2), the pure type-stamping problem is reduced — `List<User>` and `List<Config>` share a generic definition. But provenance still matters for structurally similar but non-generic code: multiple API endpoint handlers that follow the same validate-process-respond pattern, multiple data parsers with similar structure, or parallel implementations for different backends.

### 6.2 The Provenance Graph

The AST runtime maintains a first-class provenance graph. Every generated code fragment carries metadata:

- **Source intent:** the natural language description or prompt that produced it
- **Generation context:** what AST state existed when it was generated
- **Structural fingerprint:** a hash of the code's shape (ignoring names and literals)
- **Lineage:** what other fragments were generated from the same intent

This is not hidden metadata — it's a queryable graph the model and human can both inspect:

```
?provenance UserService.validate
> source_intent: "validate user input: check age non-negative, email valid"
> generated_at: step 42
> similar: [AdminService.validate (0.91), GuestService.validate (0.87)]
> shared_lineage: [AdminService.validate]
```

### 6.3 Propagation with Safety Gates

When a bug is fixed in one fragment, the system can propose the same fix to related fragments. But propagation is never automatic. It passes through safety gates:

**Gate 1 — Similarity confidence.** The runtime computes structural similarity between the fix source and each target. Only targets above a confidence threshold are proposed. "Same source intent" (shared lineage) is treated differently from "structurally similar" (coincidental resemblance) — the former is high confidence, the latter requires human or model review.

**Gate 2 — Proof obligation.** Before a propagated fix is applied, it must pass the target's existing tests. If the target has no tests, the runtime flags this and either the model generates tests first or the fix is held pending review.

**Gate 3 — Per-target validation.** Each target is validated independently. A fix that works for `UserService.validate` might not apply cleanly to `AdminService.validate` if the admin version has diverged. The runtime detects structural divergence and flags it.

```
!fix UserService.validate.s1
  +check age input.age>=0 AND input.age<=150 ~err_age_range

!propagate UserService.validate.s1
> target: AdminService.validate.s1
  similarity: 0.91, lineage: shared, tests: 3/3 pass → APPLY
> target: GuestService.validate.s1
  similarity: 0.87, lineage: none, tests: 0 → HOLD (no tests)
> target: ReportService.parse_age
  similarity: 0.62, lineage: none → SKIP (below threshold)
```

### 6.4 Advantages over Pure Generics

- **Granular control:** Each propagation target is reviewed individually. A generic fix to `List<T>.get` silently changes every instantiation. Propagation lets you accept, modify, or reject per target.
- **Model-assisted review:** "Here's a bug in UserService.validate, here are 3 structurally similar functions, which ones have the same bug?" is a task LLMs excel at.
- **Works across non-generic similarity:** Generics only help when code is literally parameterized by type. Provenance works for any structural similarity — different endpoints with similar logic, parallel implementations for different backends, copy-paste-evolved code.

---

## 7. Grammar Constraint Design

### 7.1 How It Works

The GBNF grammar runs in the inference loop (via llama.cpp, vLLM, or SGLang). At every token, the grammar masks invalid continuations from the model's logit distribution. The model literally cannot produce a syntactically invalid token sequence.

This eliminates an entire error class by construction. Half of compiler engineering is about error recovery from partial parses. With always-valid generation, that machinery disappears. The compiler becomes a thin translation layer.

### 7.2 What GBNF Can and Cannot Enforce

| Enforceable (context-free) | Not enforceable (context-sensitive) |
|---------------------------|-------------------------------------|
| Balanced delimiters | Variable declared before use |
| Correct keyword sequencing | Function call matches signature |
| Valid literal formats | Type compatibility |
| Required clause ordering | Effect consistency |
| Arity structure (fixed patterns) | Scope rules |

The interactive compilation loop (Section 5.2) solves the context-sensitive gap. The grammar ensures syntactic validity; the runtime validates semantic correctness between steps.

### 7.3 Sketch Grammar (GBNF)

```
root        ::= module+
module      ::= "+module" IDENT "\n" decl* "end\n"
decl        ::= type_decl | state_decl | function
type_decl   ::= "  +type" IDENT "=" typeexpr "\n"
state_decl  ::= "  +state" IDENT ":" typeexpr "\n"
function    ::= "  +fn" IDENT params "->" typeexpr effects "\n" body
effects     ::= ("[" effect ("," effect)* "]")?
effect      ::= "io" | "mut" | "fail" | "async" | "rand"
body        ::= statement+
statement   ::= let_s | call_s | check_s | branch_s | return_s | each_s
let_s       ::= "    +let" IDENT ":" typeexpr "=" expr "\n"
call_s      ::= "    +call" IDENT ":" typeexpr "=" callexpr "\n"
check_s     ::= "    +check" IDENT expr "~" IDENT "\n"
branch_s    ::= "    +branch" IDENT IDENT "->" IDENT "|" IDENT "->" IDENT "\n"
return_s    ::= "    +return" IDENT "\n"
each_s      ::= "    +each" IDENT IDENT ":" typeexpr "\n" body
```

At every point, the model has a narrow, well-defined set of valid next tokens. The grammar channels generation into structurally valid code. Because the language was designed for this grammar rather than the reverse, there is no tension between expressiveness and constraint.

### 7.4 Impact on Model Creativity

The grammar constrains syntax, not semantics. The interesting decisions — which data structure to use, how to decompose a problem, what edge cases to handle — happen at the semantic level: choosing which statements to emit, in which order, with which values. The grammar doesn't touch that.

A very tight grammar reduces the token-level branching factor, which *can* push the model into repetitive output if reduced too aggressively. The sweet spot: rigid structure at the statement level (keyword-verb-target-type patterns), but open expression space within values and logic. The grammar ensures every statement is well-formed; the model decides what the program does.

---

## 8. Compilation and Execution

### 8.1 Compilation Story

Forge compiles to native binaries. The AST is already a clean intermediate representation. Since the language is flat, fully typed, has no generics, and no dynamic dispatch, compilation is simpler than most languages. Every type is known at compile time, every call is direct, every effect is annotated.

Target options, in order of practicality:

- **Transpile to Rust or C:** Simplest path. Forge's flat, concrete code maps almost directly to Rust without borrowing complexity (all ownership is explicit). Leverage Rust/C's mature optimization backends.
- **LLVM IR:** Direct LLVM codegen. More control, more work. The flat structure maps cleanly to LLVM's SSA-based IR.
- **Cranelift:** Faster compilation, slightly less optimization. Good for development iteration where the model is generating and testing rapidly.

### 8.2 Why Compilation is Straightforward

The deliberate design constraints keep compilation simple:

- Boring generics are fully monomorphized — the compiler stamps concrete types, no complex type resolution
- No inheritance means no vtable dispatch
- No exceptions means no unwinding tables
- No closures means no closure conversion (named function references are just function pointers)
- Explicit effects mean the compiler knows exactly what each function does
- Shallow control flow maps directly to basic blocks

A Forge compiler would be dramatically simpler than a Rust or C++ compiler while producing code of comparable runtime performance.

### 8.3 Memory Management: Profile-Guided Automatic Strategy Selection

Forge takes a novel approach to memory management: instead of one global strategy, the compiler automatically selects the optimal strategy per function based on static analysis, effect annotations, and real execution profiles from the evaluation phase.

The model never thinks about memory. The human never configures anything. The compiled code has near-manual-allocation performance.

#### 8.3.1 Why Forge Can Do This

Traditional languages can't easily infer memory strategies because of aliasing, opaque function calls, and hidden state. Forge has structural advantages that make aggressive inference tractable:

- **Types are fully known** — boring generics monomorphize, every concrete type is visible at compile time
- **Effects tell you what functions do with references** — a pure function can't stash a reference in global state; an `[io]` function might
- **No closures** — no hidden captures that extend lifetimes unpredictably
- **Named function references don't capture state** — they're just function pointers
- **Shallow control flow** — the lifetime of every value is obvious from the structure
- **The evaluation phase already ran the code** — real allocation profiles are available before compilation

#### 8.3.2 Strategy Selection

The compiler chooses from a hierarchy of strategies, preferring the cheapest one that's safe:

**Stack allocation (cheapest).** For values that don't escape their declaring function. Pure functions with local intermediates are the primary target. The compiler can prove the value dies when the function returns, so it goes on the stack. Zero overhead.

**Arena / region allocation.** For values that live for a bounded scope — a loop iteration, a request handler, a pipeline stage. Everything allocated during that scope lives in one arena. When the scope ends, the entire arena is freed in one operation. Near-zero overhead per allocation.

**Reference counting.** For values with genuinely shared ownership that outlive their creating scope. This is the fallback, not the default. RC only kicks in for values the compiler can't prove have bounded lifetimes.

**Reference counting with cycle detection.** For the rare cases where reference cycles are possible (mutually referencing data structures). The cycle detector runs as a background sweep triggered by allocation pressure, not on every deallocation.

**Manual / unsafe.** Behind `[unsafe]` for runtime internals and performance-critical code that needs precise control.

| Strategy | Cost | When used |
|----------|------|-----------|
| Stack | Zero | Local values that don't escape their function |
| Arena | Near-zero (bulk free) | Loop iterations, request handlers, pipeline stages |
| Reference counting | Low (per-copy increment/decrement) | Shared ownership, dynamic lifetime |
| RC + cycle detection | Low + periodic sweep | Mutually referencing structures |
| Manual / unsafe | Zero (but dangerous) | Runtime internals, FFI, hot paths |

#### 8.3.3 The Evaluation Phase as Profiler

During evaluation and testing (Section 12), the runtime is already executing the code. These runs are instrumented to collect allocation profiles:

- How many allocations per function
- What sizes
- How long values live
- Whether cycles exist
- Which values escape their creating scope

This data feeds directly into the compiler's strategy selection. The flow is:

1. Model generates function
2. Runtime evaluates with test inputs (instrumented)
3. Allocation profile collected: "validate() allocates 3 intermediates, none escape, avg lifetime 12μs"
4. Compiler selects strategy: stack allocation for all intermediates
5. Compiled code uses optimal strategy — zero memory management overhead

This is **profile-guided optimization for memory management**, happening automatically as part of the normal generation loop.

#### 8.3.4 Arena-Per-Request Pattern

Especially powerful for server workloads — the kind of systems Forge would build. The compiler detects the pattern: a function marked `[io]` that processes a request and returns a response, where all intermediates are scoped to the function body.

```
+fn handle_request (req:Request)->Response [io,fail]
  // Compiler detects: all intermediates scoped to this function.
  // Strategy: arena allocation. Bulk free on return.
  +call parsed:ParsedBody = parse_body(req)         // arena
  +call validated:Valid = validate(parsed)            // arena
  +call enriched:Enriched = enrich(validated)         // arena
  +call result:DbResult = db.save(enriched)           // arena
  +call resp:Response = build_response(result)        // escapes → RC or caller's arena
  +return resp
```

Each incoming request gets its own arena. Everything allocated during processing lives in that arena. When the request completes, one bulk free. No per-object reference counting, no cycle detection, no individual frees. Near-zero memory management overhead for the hot path.

#### 8.3.5 Escape Analysis on Steroids

Forge's constraints enable much more aggressive escape analysis than Rust or Go:

- A pure function (no `[io]`, no `[mut]`) cannot stash references in external state — all allocations are provably local
- Named function references are just pointers — passing `compare_by_age` to `sort` doesn't extend any lifetime
- Flat control flow with explicit branches means every value's lifetime is structurally visible
- No inheritance or dynamic dispatch means no hidden vtable indirection that could extend lifetimes

The result: most allocations in typical Forge code end up on the stack or in arenas. Reference counting is the fallback for genuinely shared, long-lived, dynamically scoped values — not the common case.

#### 8.3.6 The Model's Perspective

The model writes:

```
+let x:Foo = create_foo()
+call y:Bar = transform(x)
+return y
```

It never writes allocation annotations, lifetime markers, or ownership hints. The compiler handles everything. If the compiler's static analysis is uncertain, it falls back to reference counting — safe and correct, just not maximally fast. The profiling data from evaluation narrows the uncertainty over time.

### 8.4 Dual-Backend Strategy

The recommended approach is to use both Cranelift and Rust transpilation. Cranelift for development — the tight generation loop where the model emits mutations, compiles, tests, and iterates. A millisecond compile means the model can do hundreds of generate-test cycles in the time it takes `rustc` to do one. Rust transpilation for release builds — full LLVM optimization, easy ecosystem access, mature tooling.

| | Transpile to Rust | Cranelift |
|---|---|---|
| Compilation speed | Slow (full rustc) | Fast (milliseconds) |
| Rust crate FFI | Trivial (native) | Needs auto-generated C shim |
| C FFI | Through Rust's libc | Direct (native ABI) |
| Optimization | Excellent (LLVM) | Good, not great |
| Iteration loop | Seconds per cycle | Milliseconds per cycle |

This is a proven pattern — the Rust compiler itself uses Cranelift as an optional fast backend for debug builds while keeping LLVM for release. The Forge compiler would have two backends, and the model doesn't need to know which one is active — the extern declarations and code are the same either way.

---

## 9. Building Complex Systems

Can Forge build a web server, a game engine, a database? Yes, but the workflow differs from traditional development.

### 9.1 Complexity Lives in Composition

Individual Forge modules are deliberately simple: flat functions, concrete types, explicit effects. Complexity emerges from the composition graph between modules. A web server is not one complex module but dozens of simple modules with explicit interfaces between them.

The model scaffolds large systems as module graphs, each module internally simple and concrete. The provenance system handles cross-cutting concerns. The semantic query system lets the model maintain coherence across a large codebase without needing it all in context.

### 9.2 Scale Strategy

| System size | Approach |
|-------------|----------|
| Small (single module) | One-shot scaffold, fine-tune statements |
| Medium (5–20 modules) | Scaffold module graph, implement per-module, query cross-module dependencies |
| Large (50+ modules) | Hierarchical: scaffold subsystems, scaffold modules within each, implement bottom-up with semantic queries for integration |

### 9.3 What About Performance-Critical Code?

Forge's flat, concrete, fully-typed nature is actually ideal for performance. No dynamic dispatch, no boxing, no runtime type checks, no garbage collector overhead. The compiled output should be comparable to C for computational kernels.

For cases requiring hand-optimized SIMD or unsafe memory access, Forge would support a `[unsafe]` effect marker and a raw-operation mode, similar to Rust's `unsafe` blocks but with the same explicit, flat syntax.

---

## 10. Foreign Function Interface

Forge can't exist in a vacuum. Real systems need access to existing libraries, operating system APIs, and mature ecosystems. The FFI design follows the same philosophy as everything else: explicit, flat, effects declared. The model should never be surprised by what a foreign function does.

### 10.1 Three Tiers

**Tier 1: Forge modules.** Native Forge code. Fully validated, fully tracked, all effects known. This is the default.

**Tier 2: Rust crate bindings.** The sweet spot for FFI because Rust's type system is rich enough that safe bindings can be generated semi-automatically. Since Forge's simplest compilation path is transpiling to Rust, Rust FFI is almost free — the Forge compiler generates `extern crate` references and wrapper functions. Cargo handles dependency management.

**Tier 3: C FFI.** For raw C libraries (OpenSSL, SQLite, system calls). Inherently unsafe because C provides no type safety guarantees.

### 10.2 Rust Bindings

The model declares an external module with Forge-typed signatures and effect annotations. The compiler generates the glue and verifies compatibility:

```
+extern rust serde_json from "serde_json"
  +fn parse (input:String)->Result<JsonValue> [fail]
  +fn serialize (value:JsonValue)->Result<String> [fail]
end
```

The compiler checks that the Rust crate's actual signatures are compatible with the declared Forge signatures. If they don't match, it's a compile-time error. The effect annotations are declared by the model (or human) and trusted, but can be verified against Rust's type information.

### 10.3 C Bindings

C FFI requires the `[unsafe]` effect because the Forge compiler can't verify anything about C code:

```
+extern c sqlite3 from "libsqlite3"
  +fn open (path:String)->Result<DbHandle> [io,fail,unsafe]
  +fn exec (db:DbHandle, sql:String)->Result<Rows> [io,fail,unsafe]
  +fn close (db:DbHandle)->Result<Ok> [io,unsafe]
end
```

Everything gets `[unsafe]`. The model knows these functions are dangerous. Callers know they're dangerous. The provenance tracker can flag all code paths that depend on unsafe C calls.

### 10.4 The Wrapping Pattern

The practical pattern is containment: write a thin safe Forge module around unsafe bindings, validate inputs, handle errors, and expose a clean interface. Everything above the wrapper never touches unsafe:

```
+module SafeDb
  +extern c sqlite3 from "libsqlite3"
    +fn raw_open (path:String)->Result<DbHandle> [io,fail,unsafe]
    +fn raw_exec (db:DbHandle, sql:String)->Result<Rows> [io,fail,unsafe]
    +fn raw_close (db:DbHandle)->Result<Ok> [io,unsafe]
  end

  +fn open (path:String)->Result<Db> [io,fail]
    +check path path.len>0 ~err_empty_path
    +call handle:Result<DbHandle> = raw_open(path) [unsafe]
    +check handle handle.is_ok ~err_open_failed
    +return Db{handle:handle.unwrap}

  +fn query (db:Db, sql:String)->Result<Rows> [io,fail]
    +check sql sql.len>0 ~err_empty_sql
    +call rows:Result<Rows> = raw_exec(db.handle, sql) [unsafe]
    +return rows
end
```

The `[unsafe]` is contained within `SafeDb`. Everything outside sees only `[io,fail]` — clean, safe Forge functions. This mirrors the pattern used in the Rust ecosystem, where unsafe FFI is wrapped in safe abstractions.

### 10.5 Querying Available Libraries

The model can discover and inspect external libraries through semantic queries in the interactive compilation loop:

```
?extern list                    // what external modules are available?
?extern serde_json.functions    // what can I call from serde_json?
?extern sqlite3.effects         // what effects does sqlite3 have?
```

The AST runtime answers with declared signatures, and the model generates calls against them. The grammar constrains the call syntax; the runtime validates that the referenced extern actually exists and the types match.

### 10.6 Effect Boundaries

A key property: effects from external code never leak silently. If a Rust crate function can panic, the Forge binding must declare `[fail]`. If a C function does IO, the binding must declare `[io]`. The compiler enforces that no unannounced effects cross the FFI boundary.

This means the model can always reason about effects locally, even when calling into foreign code. The FFI declaration is the contract, and the contract is visible at every call site.

---

## 11. Concurrency and Parallelism

Forge separates concurrency (doing multiple things by interleaving on one thread) from parallelism (doing multiple things simultaneously on multiple cores). Each has its own model, its own effect markers, and its own level of danger. The model defaults to the safest option and escalates explicitly.

### 11.1 Coroutines: Cooperative Concurrency

Coroutines are Forge's primary concurrency primitive. A coroutine is a function that can pause at marked suspension points, hand a value to its caller, and later resume from where it paused. The function's local state stays alive between yields.

```
+fn stream_users (query:Query)->Yield<User> [io,yield]
  +call batch:List<User> = db.query_batch(query, offset=0, limit=100)
  +each batch u:User
    +yield u
  +call more:List<User> = db.query_batch(query, offset=100, limit=100)
  +each more u:User
    +yield u
```

The `[yield]` effect marker makes suspension explicit. Callers know they're consuming a yielding function. There is no hidden control flow.

The caller treats a coroutine like a lazy collection:

```
+fn process_all (query:Query)->Result<Count> [io,yield]
  +call stream:Yield<User> = stream_users(query)
  +each stream u:User
    +call result:Result<Ok> = validate_and_save(u)
    +check result result.is_ok ~err_processing
  +return stream.count
```

Items are produced one at a time, on demand. A million users never all sit in memory at once.

### 11.2 Stackless Design

Forge uses stackless coroutines. A function can only yield if it directly contains a `+yield` statement. It cannot call some other function that secretly yields on its behalf.

If you see a function without `[yield]` in its effects, you know with certainty it runs straight through. No surprises. The model can reason about each function independently.

The compiler transforms coroutines into state machines. Each yield point becomes a state transition. The function's local variables become fields in a struct. Resuming the coroutine is just calling a function that switches on the current state. This is a well-understood transformation (it's what Rust's async/await and Python's generators do under the hood), and Forge's flat structure makes it particularly easy — there's no deep nesting to unravel.

The "coloring problem" (async functions infecting every caller) is a non-issue in Forge because effect annotations are mandatory from the start. The `[yield]` effect *is* the color, and it's required anyway. There's no separate async/sync split — just functions with or without the yield effect.

### 11.3 Structured Parallelism: Threads Done Safely

For CPU-bound work that needs multiple cores, Forge provides structured parallelism via the `[parallel]` effect:

```
+fn parallel_process (items:List<Work>)->List<Result<Done>> [parallel]
  +call chunks:List<List<Work>> = split(items, chunk_size=4)
  +parallel chunks chunk:List<Work> -> results:List<Result<Done>>
    +each chunk item:Work
      +call r:Result<Done> = process(item)
      +yield r
  +return results
```

The `+parallel` statement is a structured parallel-for: it spawns threads, each processes a chunk, results are collected. Critically, there is no shared mutable state between parallel branches. Each chunk is independent. The runtime enforces this — if two parallel branches try to access the same mutable state, that's a compile-time error.

This is the fork-join model. Split work, run in parallel, join results. No locks, no mutexes, no shared memory.

### 11.4 Channels: Thread Communication

When parallel branches need to communicate, they use typed, bounded channels:

```
+fn producer_consumer ()->Result<Done> [parallel,io]
  +let ch:Channel<Message> = channel(buffer=10)
  +spawn producer(ch.sender) [io]
  +spawn consumer(ch.receiver) [io]
  +call result:Result<Done> = ch.join
  +return result
```

Channels are the only way threads communicate. Data flow is explicit: messages go in one end, come out the other. The model can reason about them because there's no spooky action at a distance.

### 11.5 The Concurrency Hierarchy

Each level is more powerful and more dangerous, and the effect system makes the danger level visible in every function signature:

| Level | Mechanism | Effect | Use case | Danger |
|-------|-----------|--------|----------|--------|
| 1 | Coroutines | `[yield]` | IO-bound work: web servers, DB queries, file processing | None — single-threaded, cooperative |
| 2 | Structured parallelism | `[parallel]` | CPU-bound work: map-reduce, batch processing, parallel compilation | Low — no shared state, fork-join only |
| 3 | Channels | `[parallel]` | Communication between parallel branches | Low — typed, bounded, explicit |
| 4 | Unstructured threads | `[unsafe,parallel]` | Runtime internals, FFI with thread-based C libraries | High — shared mutable state possible |

The model defaults to Level 1, escalates to Level 2 for CPU-bound work, uses Level 3 when parallel branches need to communicate, and almost never touches Level 4.

### 11.6 What Forge Deliberately Does Not Support

- **Shared mutable state between threads.** No `Arc<Mutex<T>>`, no atomic operations as language primitives. Shared state goes through channels or a dedicated state-holder thread.
- **Unstructured thread spawning.** No fire-and-forget threads. Every `+spawn` must be joined or its scope must be bounded. The compiler verifies this.
- **Locks and mutexes as language primitives.** These are expert-level tools that produce bugs even experts struggle with. If needed for runtime internals, they live behind `[unsafe]`.

### 11.7 Self-Hosting Implications

A cooperative coroutine scheduler is a relatively simple piece of code — a good early target for Forge implementing its own infrastructure. The scheduler itself would use Level 4 (unsafe thread primitives) internally, but everything built on top uses the safe structured primitives. This mirrors how Rust's tokio runtime uses unsafe internally to provide a safe async interface.

---

## 12. Evaluation, Simulation, and Testing

Because Forge functions are flat, explicitly typed, and have declared effects, the AST runtime can evaluate, trace, and simulate code at any granularity — from individual expressions up to full modules — without compiling a complete program.

### 12.1 Function-Level Evaluation

The model generates a function and immediately emits test cases. The runtime evaluates them on the spot — no full compilation, no linking, no binary:

```
+fn validate (input:Request)->Result<Valid> [fail]
  +check age input.age>=0 ~err_negative
  +check email is_valid_email(input.email) ~err_email
  +return input

!test validate
  +with {age: 25, email: "foo@bar.com"} -> expect Ok
  +with {age: -1, email: "foo@bar.com"} -> expect Err(err_negative)
  +with {age: 25, email: ""} -> expect Err(err_email)
```

Results go back into context. The model either moves on or fixes the function with a targeted `!replace`.

### 12.2 Statement-Level Tracing

The runtime can evaluate a function statement by statement, showing the state after each one:

```
!trace validate {age: 25, email: "foo@bar.com"}

> s1: check age   | age=25, 25>=0 = true          | pass
> s2: check email | email="foo@bar.com", valid=true | pass
> s3: return      | Result<Valid>(Ok)
```

The model gets a step-by-step execution trace. If something goes wrong, it can see exactly which statement produced the wrong result and emit a targeted fix for that one statement. This is dramatically more efficient than regenerating an entire function.

### 12.3 Expression-Level Evaluation

The model can ask what any expression evaluates to in a given context — essentially a REPL built into the AST runtime:

```
!eval input.age>=0 AND input.age<=150
  +with {age: 200} -> false
  +with {age: 25}  -> true
  +with {age: -1}  -> false
```

### 12.4 Symbolic Simulation

Instead of concrete values, the runtime can reason about constraints symbolically, enumerating all execution paths through a function:

```
!simulate validate
  +input age:Int, email:String
  > path 1: age<0 -> Err(err_negative)
  > path 2: age>=0 AND NOT is_valid_email(email) -> Err(err_email)
  > path 3: age>=0 AND is_valid_email(email) -> Ok(input)
  > coverage: 3 paths, all reachable
```

This is lightweight formal verification — not full theorem proving, but path enumeration through flat, branch-based code is tractable precisely because Forge has no deep nesting or recursion to blow up the path space.

### 12.5 Testing Effectful Functions

Pure functions can be evaluated directly. For functions with effects, Forge's effect system provides the mock boundary. Since every function declares its effects, the runtime knows exactly what external dependencies exist and can intercept at the effect boundary.

Instead of mock objects, dependency injection, or test frameworks, the model scripts what the outside world should look like for each test:

```
!test fetch_user [io:scripted]
  +mock http.get("/users/42") -> Ok({status:200, body:"{name:\"alice\"}"})
  +mock http.get("/users/99") -> Err(timeout)
  +with 42 -> expect Ok(User{name:"alice"})
  +with 99 -> expect Err(err_network)
```

The `[io:scripted]` tells the runtime to replace all IO operations with the provided responses. No actual network calls, no external dependencies. Just a table of "when you see this call, return this value."

This works because effects are explicit and the runtime controls them. In a traditional language, an `http.get` call could be buried three layers deep and you'd need complex mock infrastructure to intercept it. In Forge, the runtime knows exactly which functions do IO and intercepts at that boundary trivially.

### 12.6 Effect-Specific Test Modes

Each effect type has a corresponding test mode:

| Effect | Test mode | Behavior |
|--------|-----------|----------|
| `[io]` | `io:scripted` | Scripted request/response pairs for network, database, file system |
| `[io]` | `io:virtual_fs` | In-memory virtual file system with pre-populated files |
| `[rand]` | `rand:seed(N)` | Deterministic PRNG with fixed seed |
| `[time]` | `time:fixed("...")` | Frozen clock at specified timestamp |
| `[yield]` | `yield:collect` | Collect all yielded values into a list |
| `[parallel]` | `parallel:sequential` | Run parallel branches sequentially for deterministic testing |

```
// Deterministic randomness
!test roll_dice [rand:seed(12345)]
  +with 6 -> expect 3

// Frozen clock
!test check_expiry [time:fixed("2026-03-18T12:00:00")]
  +with token -> expect Err(expired)

// Virtual file system
!test read_config [io:virtual_fs]
  +mock fs.read("/etc/config.toml") -> Ok("port=8080")
  +with "/etc/config.toml" -> expect Config{port:8080}

// Collect coroutine output
!test stream_items [yield:collect]
  +with query -> expect [item1, item2, item3]
```

### 12.7 The Generation Loop with Testing

The complete feedback loop during code generation becomes:

1. Model emits function (grammar-constrained)
2. Runtime type-checks it
3. Runtime auto-generates edge case inputs from the types
4. Model emits test cases (or the runtime proposes them)
5. Runtime evaluates or simulates the function
6. Results go back into context: "passes 8/10 cases, fails on empty string and negative overflow"
7. Model fixes the specific failing cases
8. Repeat until all paths pass

The model never has to guess whether its code is correct. It gets immediate, concrete feedback at whatever granularity it needs.

### 12.8 Integration Tests

For testing that modules actually work together with real IO, Forge runs the compiled binary against a real or containerized environment. This is separate from the generation loop — during generation, the model tests against scripted effects for speed. Integration tests run after the program is complete and use the full compilation backend.

---

## 13. Browser Interface and Multi-Representation

Forge has no source files. The AST is the single source of truth, and it can be projected into whatever shape is most useful for whoever is looking at it. The browser serves as the primary human interface — a live, interactive window into the program being built.

### 13.1 Three Representations

**Model representation:** the flat `+fn +check +call` mutation syntax, optimized for token efficiency and grammar-constrained generation. This is what the LLM reads and writes.

**Internal representation:** the AST in the runtime. A typed tree with metadata, provenance links, test results, execution traces. This is the source of truth.

**Human representation:** whatever the browser renders. Pseudocode, flowcharts, block diagrams, tables, forms. Multiple views of the same AST, each optimized for a different task. This is what humans see and interact with.

Nobody ever edits "source files." The model emits mutations, the human interacts through the browser, both operate on the same AST.

### 13.2 Human-Readable Code View

The browser translates the AST into a clean, readable format that looks closer to pseudocode than to the model's mutation syntax:

```
┌─ validate(input: Request) → Result<Valid>  [fail]
│
│  ✓ check: input.age ≥ 0               → err_negative
│  ✓ check: is_valid_email(input.email)  → err_email
│  ↩ return input
│
└─ tests: 3/3 passed
```

Or a visual flowchart where each statement is a block, branches show as forking paths, and clicking any block reveals its type, value during the last trace, and provenance links. The human never needs to read the model's `+fn +check +call` format.

### 13.3 Live Mutation Feed

As the model emits AST mutations, the browser shows them in real time. The AST runtime streams events over WebSocket:

```
{event: "mutation", op: "+fn", target: "UserService.validate", status: "ok"}
{event: "test", target: "UserService.validate", cases: 3, passed: 3}
{event: "mutation", op: "!replace", target: "UserService.validate.s2", status: "ok"}
{event: "trace", target: "UserService.validate", input: {...}, steps: [...]}
```

A new function appears, nodes light up green. A replacement happens, the old node fades out and the new one fades in. A `!propagate` shows the fix flowing along provenance links to related nodes. The human watches the program being constructed in real time.

### 13.4 Interactive Evaluation

The browser is not just a viewer — it's an evaluation surface. The human can:

- **Run a function:** click it, type inputs into a form, hit run. The AST runtime evaluates it instantly. The trace visualization highlights the execution path through the function, showing values at each step.
- **Explore edge cases:** the runtime auto-generates edge cases from the types. The human flips through them — "what happens with an empty string? age=MAX_INT? missing email?" — each running instantly.
- **Script mock effects:** instead of writing `+mock http.get(...) -> Ok(...)` in the model's syntax, the human fills in a table of request/response pairs. Same semantics, human-friendly interface.
- **Modify test cases:** add, edit, or remove test inputs and expected outputs through a form, then re-run.

### 13.5 Human Steering

The browser also serves as an input channel back to the model. A human can:

- **Mark a function as wrong:** click it, describe what's broken in natural language. The instruction gets fed to the model as context for the next generation step.
- **Request regeneration:** select a subtree and say "redo this." The model receives the current AST state minus the selected subtree plus the human's instructions.
- **Add constraints:** "this function must handle the case where the database is unreachable" — the constraint attaches to the AST node and the model sees it.
- **Approve or reject propagations:** when the model proposes a `!propagate` fix, the human reviews each target and accepts, modifies, or rejects individually.

### 13.6 Multiple Simultaneous Views

Because the browser renders from the AST, multiple views can coexist — each optimized for a different role:

| View | Audience | Shows |
|------|----------|-------|
| Module graph | Architect | Module dependencies, effect flow, unsafe boundaries |
| Pseudocode | Reviewer | Human-readable code with inline type annotations |
| Flowchart | Debugger | Execution paths, branch logic, trace highlighting |
| Test dashboard | Tester | Test cases, pass/fail status, coverage, edge case explorer |
| Mutation log | Auditor | Chronological history of all changes, provenance links |
| Effect map | Security reviewer | Which modules do IO, which touch unsafe, which extern libraries are called |

All views show the same program. All update live as the model works. A reviewer can watch pseudocode while a tester explores edge cases while an architect checks the module graph — all seeing real-time updates from the same AST.

### 13.7 The Thinking Panel

While the model is in its `<think>` phase, the browser can display the raw reasoning in a side panel. The human sees the model's thought process — which approach it's considering, what edge cases it's worried about, why it chose a particular data structure. When the model switches to `<code>`, the AST panel updates. Intent followed by execution, visible in real time.

This provides a natural audit trail: the human can check not just what the model generated, but *why* it made each decision.

### 13.8 Technology

The implementation is straightforward. The AST runtime (already a Rust service) adds a WebSocket endpoint that streams mutation events. The frontend is a React or Svelte app with a graph rendering library (D3, Cytoscape, or elkjs for automatic layout). Each mutation event carries enough information to animate the change. The evaluation UI is forms and tables backed by the same AST runtime evaluation API used by the model's `!test` and `!trace` commands.

---

## 14. Self-Hosting: The Ultimate Validation

If Forge can implement its own AST runtime, compiler, and provenance tracker, that proves the language is expressive enough for real systems work.

### 14.1 Bootstrap Path

**Stage 1:** AST runtime and compiler written in Rust by hand. Model generates Forge code, Rust toolchain compiles it.

**Stage 2:** Rewrite the AST runtime in Forge, compiled by the Rust-based compiler. Forge now maintains its own core data structure.

**Stage 3:** Rewrite the compiler in Forge. Compile it with the Rust-based compiler one last time. The Forge compiler now compiles itself.

**Stage 4:** The model generates improvements to the Forge compiler *in Forge*, validated by the Forge compiler. The loop closes.

This is the classic self-hosting bootstrap (same path C, Rust, and Go all took), but with the twist that Stage 4 involves an LLM improving its own toolchain. The model generates a compiler improvement as AST mutations, the current compiler validates and compiles it, tests run, and if they pass, the new compiler replaces the old one.

---

## 15. Layered Architecture: From Feedback Loop to Full Constraint

The Forge system is designed as independent, stackable layers. Each layer reduces the error rate, but they're independent — you can ship a useful system with just the first layer and add the others incrementally.

### 15.1 The Four Layers

**Layer 0 — Feedback loop (the foundation).** A base model, a system prompt describing Forge's syntax, and the AST runtime validating each step. The model generates something, gets told what's wrong, fixes it. Even without any fine-tuning or grammar constraints, this is already massively better than one-shot generation. A type error on line 3 gets caught and fixed immediately rather than cascading into 50 lines of broken code.

**Layer 1 — Few-shot examples.** Add examples of natural language → Forge mutations to the prompt. The model learns the idioms. Most mutations land correctly on the first try. The feedback loop catches the rest. This is hours of work.

**Layer 2 — LoRA fine-tuning.** Generate a synthetic training corpus using a capable model (Claude, GPT-4), fine-tune a LoRA adapter. The model rarely produces invalid syntax at all. The feedback loop becomes a safety net rather than the primary correction mechanism. This is days of work.

**Layer 3 — Grammar constraints.** GBNF or equivalent grammar running in the inference loop, masking invalid tokens at every step. Syntax errors become literally impossible by construction. The feedback loop only handles semantic issues. This is the final optimization, not a prerequisite.

| Layer | What it adds | Error rate | Effort |
|-------|-------------|------------|--------|
| 0: Feedback loop | Immediate validation, targeted fixes | Moderate — model corrects on retry | Weekend |
| 1: Few-shot | Idiomatic generation | Low — most mutations valid first try | Hours |
| 2: LoRA | Learned Forge patterns | Very low — rare syntax errors | Days |
| 3: GBNF grammar | Impossible to produce invalid syntax | Zero syntactic errors by construction | Weeks |

The key insight: **the feedback loop alone buys most of the value.** Each subsequent layer provides diminishing but real improvements. The system is useful from Layer 0 onward.

### 15.2 Target Model: Qwen 3.5

The Qwen 3.5 family is the recommended starting point. Specifically:

**Qwen3.5-9B** for the primary development model. Strong coding and reasoning benchmarks at a size that fits on an RTX 3090 at 4-bit quantization with room for long context (262K native context window). The long context is critical — the interactive compilation loop means conversations with lots of injected AST state.

**Qwen3.5-35B-A3B** as a stretch option. A 35B total parameter MoE model with only 3B active parameters per forward pass. May fit on a 3090 with aggressive quantization while significantly outperforming the 9B on complex reasoning.

**Qwen3.5-4B** for rapid iteration and testing. Fast enough to run hundreds of generate-test cycles per minute. Good for validating the feedback loop at Layer 0 before investing in LoRA training.

Why Qwen 3.5 specifically suits Forge:

- **Native thinking mode.** The models generate `<think>` content before structured output by default, which aligns perfectly with Forge's two-phase generation. The LoRA fine-tuning only needs to teach that the structured output phase is Forge mutations.
- **Agent and tool-use training.** The models are explicitly trained for agent-style workflows with tool calls and structured feedback — exactly the interactive compilation loop pattern.
- **vLLM and SGLang support.** Both inference backends are supported out of the box, which are the same backends needed for grammar-constrained generation at Layer 3.
- **MoE efficiency.** The sparse architecture means more intelligence per VRAM dollar, which matters when running on a single 3090.

### 15.3 Synthetic Training Data

For Layer 2 (LoRA fine-tuning), use a capable model to generate the training corpus: describe the Forge spec, provide examples, have it translate a corpus of small programs into Forge mutations. The feedback loop means the fine-tuned model's output doesn't need to be perfect — the AST runtime catches whatever the LoRA misses.

If grammar constraints are added at Layer 3, they act as an additional safety net — the LoRA teaches semantics, the grammar enforces syntax, and the feedback loop handles everything else. Each component compensates for the others' weaknesses.

### 15.4 Tokenizer Co-Design (Optional)

For maximum efficiency at Layer 3 and beyond, co-design the tokenizer with the language. Common patterns like `+fn`, `+check`, `->`, `~` become single tokens. This is a significant optimization but not required for the prototype — the standard Qwen tokenizer works fine through Layers 0–2.

### 15.5 Thinking Steps Stay Unconstrained

Critical design decision across all layers: the grammar constraint (if used) only applies to code emission. The thinking/reasoning phase remains unconstrained natural language. Forcing thinking into the grammar would be destructive — the model needs free-form space to explore, backtrack, and consider alternatives.

---

## 16. Implementation Stack

### 16.1 Prototype Architecture

| Component | Technology | Role |
|-----------|-----------|------|
| Inference | vLLM / SGLang | Serves model, supports grammar constraints when ready |
| Model | Qwen3.5-9B (primary) / Qwen3.5-4B (fast iteration) | Generates AST mutations |
| AST Runtime | Rust service | Maintains AST, validates mutations, answers queries |
| Provenance | SQLite / embedded DB | Tracks generation origins and propagation links |
| Compiler | Rust → Cranelift (dev) / Rust transpilation (release) | Produces native binaries from AST |
| Browser UI | React/Svelte + D3/elkjs + WebSocket | Human visualization, evaluation, and steering |

### 16.2 Hardware Requirements

The entire prototype stack runs on a single machine with an RTX 3090 (24GB VRAM). Qwen3.5-9B at 4-bit quantization fits comfortably with room for its 262K context window. The Qwen3.5-35B-A3B (3B active parameters) may also fit with aggressive quantization and is worth testing. The AST runtime, compiler, and browser UI are lightweight CPU-bound processes.

### 16.3 Development Phases

**Phase 1 — Feedback loop proof of concept (1 weekend).** Qwen3.5-4B or 9B via vLLM/ollama, a system prompt with Forge syntax examples, a small Python or Rust service that parses the output and feeds errors back. No grammar constraints, no fine-tuning. Test the core hypothesis: does incremental generation with compiler feedback produce better code than one-shot generation?

**Phase 2 — AST runtime (2–4 weeks).** Build the mutation-based AST service in Rust. Implement proper validation, type-checking, the semantic query system, and function-level evaluation. This is the foundation everything else builds on.

**Phase 3 — Browser interface (1–2 weeks).** WebSocket event stream from the AST runtime, React/Svelte frontend rendering the AST as a tree, live mutation feed, basic evaluation UI. Humans can now watch and interact with the generation process.

**Phase 4 — LoRA training (1–2 weeks).** Generate synthetic training corpus using Claude or GPT-4, fine-tune a LoRA adapter on Qwen3.5-9B, evaluate correctness improvement over the few-shot baseline.

**Phase 5 — Compilation backend (2–4 weeks).** Cranelift for fast dev compilation, Rust transpilation for optimized release builds. Produce running binaries. Benchmark against equivalent hand-written programs.

**Phase 6 — Grammar constraints (1–2 weeks).** Design and test GBNF grammar for Forge syntax. Integrate with vLLM/SGLang constrained generation. Measure improvement over Layer 2 (LoRA only).

**Phase 7 — Provenance and propagation (2–3 weeks).** Build the tracking system. Test on a medium-scale project (10–20 module service).

**Phase 8 — Self-hosting (4–8 weeks).** Rewrite AST runtime in Forge, then compiler. Validate that Forge can build itself.

---

## 17. Key Experiment

The central hypothesis is testable at each layer. Take a benchmark set of programming tasks (e.g., a subset of HumanEval or MBPP), generate solutions under different configurations, and compare:

- **Syntactic validity rate:** What percentage of generated programs parse without errors?
- **Semantic correctness rate:** What percentage pass unit tests?
- **Token efficiency:** How many total tokens (including retries and feedback) to reach a correct solution?
- **Repair efficiency:** When a generated program has a bug, how many additional steps to fix it?

Test configurations, in order:

| Configuration | What it tests |
|--------------|---------------|
| Qwen3.5-9B → one-shot Python | Baseline |
| Qwen3.5-9B → one-shot Forge (few-shot prompted) | Does the language design help even without feedback? |
| Qwen3.5-9B → Forge + feedback loop | Does incremental generation with compiler feedback improve correctness? |
| Qwen3.5-9B → Forge + feedback + LoRA | Does fine-tuning reduce the number of feedback cycles needed? |
| Qwen3.5-9B → Forge + feedback + LoRA + GBNF | Does grammar constraint eliminate remaining syntax errors? |

The prediction: the feedback loop alone (Layer 0) will show the largest single improvement over the baseline. Each subsequent layer will provide diminishing but measurable gains. The total system will show near-100% syntactic validity, higher semantic correctness, and dramatically better repair efficiency than one-shot Python generation.

---

## 18. Open Questions

- **Standard library scope:** How much built-in functionality? Minimal (like C) or batteries-included (like Python)? The answer affects how much the model needs to generate from scratch vs. compose existing pieces.
- **Dynamic grammar vs. static:** Should the grammar update during generation to track declared symbols? More correct but computationally expensive. The interactive loop may make this unnecessary.
- **Coroutine scheduler design:** What scheduling strategy for the cooperative runtime? Round-robin, priority-based, work-stealing? This becomes a self-hosting question once Forge implements its own scheduler.
- **Cycle detection strategy:** How aggressive should the reference counting cycle detector be? Background sweep on allocation pressure is the default, but real-time systems may need different guarantees.
- **Arena scope heuristics:** How does the compiler decide arena boundaries? Per-request is obvious for servers, but what about batch processing, game loops, or long-running computations? May need configurable scope hints.
- **Semantic query protocol:** The query system (`?symbols`, `?callers`, `?effects`, `?similar`) needs a precise, formal specification. What queries are supported? What's the response format? What's the cost model? This protocol may be as important as the language itself.
- **Provenance confidence tuning:** What similarity threshold for propagation proposals? How to weight structural similarity vs. shared lineage? This likely needs empirical tuning against real codebases.
- **Governance:** If AI systems communicate in an opaque language, how do humans audit them? The inspectability of Forge's AST, provenance graph, and browser interface partially addresses this, but the broader question remains open.

---

## 19. Summary

Forge inverts the normal relationship between language and tooling. The interactive compilation feedback loop is the foundation — the model emits code, the runtime validates it, errors feed back, the model fixes them. Everything else is optimization layered on top: few-shot examples, LoRA fine-tuning, grammar constraints, tokenizer co-design. Each layer is independently valuable and incrementally deployable.

The result is a system where:

- The compiler is a conversational partner providing feedback between generation steps
- Code is a sequence of AST mutations at variable granularity
- The browser renders the AST into human-readable views with live evaluation
- Abstraction is handled by tooling (provenance tracking) rather than language features (generics)
- Grammar constraints can eliminate syntax errors by construction, but aren't required to start
- The system is useful from day one (feedback loop only) and improves as layers are added
- The ultimate validation is self-hosting: Forge compiling Forge, improved by an LLM writing Forge

*A good AI-first language is not easy to write. It is easy to understand, verify, and rewrite without hallucinating.*
