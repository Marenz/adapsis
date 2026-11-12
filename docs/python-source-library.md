# Opt-in authoritative Python module library

This checkpoint implements issue #121's source-storage boundary. The language is
the Adapsis Python frontend, not CPython: source is parsed by RustPython and
lowered into the existing Adapsis AST/runtime. No Python interpreter or imports
execute. Canonical printing does not preserve comments or original formatting.

## Explicit activation

Use an **isolated** `ADAPSIS_MODULES_DIR` containing `library-sources.json`:

```json
{
  "version": 1,
  "modules": {
    "Helpers": {"format": "python", "dependencies": []},
    "Example": {"format": "python", "dependencies": ["Helpers"]}
  }
}
```

Each managed module has one maintained file, `<Module>.py`. Its stored cases are
part of that file, identified by the trusted persistence-only decorator:

```python
def increment(x: int) -> int:
    return x + 1

@stored_test("Example.increment")
def test_case_0() -> None:
    assert increment(1) == 2
```

**Trusted stored-test data.** Persisted `@stored_test` cases restore the `.ax`
test semantics: inputs and expectations may call pure functions (`concat(..)`,
effect-free user helpers) and use the `Ok`/`Err`/`Some`/`None` constructors.
Calls take positional arguments and keep their exact callee spelling: a bare
helper stays bare, so it resolves at run time exactly as the original case did.
This differs on purpose from function bodies: a bare call in a body is only
qualified when it provably resolves to the caller's own declaration, otherwise
the save refuses. A bare call in stored test data is never rewritten, so it
binds like the runtime binds it, by module order. If an earlier module that is
not a dependency declares the same name, the test data calls THAT module's
function (pinned by `bare_call_in_stored_test_data_resolves_by_module_order_not_caller_module`).
Qualify the call in the test if the caller's own function is meant.
Pure means a non-IO builtin or constructor, or a user function (resolved as the
runtime resolves the spelling) declaring no effect other than `fail`. Anything
else is refused, both when saving and when loading, with a diagnostic naming
the case and callee. `@stored_test` does not expand native mutation/test input policy. The native test/evaluate tools and the CLI
keep the literal-only policy. Function names
starting with `test_` alone do not identify persisted tests. Every supported
member and every stored case must be represented; an unsupported member/test
refuses the entire save. Cases retain order within their target function.
Cases loaded from `.py` attach a versioned typed payload (`store_test_payload(...,
true)`); the printer prefers a present payload and refuses the save when it is
invalid, using display strings only for legacy records without one. Transitional
`legacy_ax` imports keep the legacy string adapter.
Parsing source is not evidence of passing tests, so a loaded case starts
**untested**. After the whole managed set has validated, startup re-runs every
stored case once through the existing test evaluator with its strictly mock-only
coroutine handle: no mocks are supplied, so any IO request fails closed and no
real IO is possible. A case that passes is marked passed; a failing or erroring
case stays untested and never refuses startup. This keeps the eval gate (functions
with more than two statements need passing tests) working after a restart. The
startup log has one summary line (cases, passed, untested, milliseconds) and one
line per affected module naming functions and counts; the error journal gets one
entry per module. Neither contains case content. Pass flags are runtime state and
are never written to source.

**Proven local-call qualification.** Legacy modules often call their own
functions bare (`g(x)` inside module `M`). The persistence printer prints such a
call as `M.g(x)` only when the runtime's bare resolution, `Program::get_function`,
returns exactly `M`'s own `g` declaration: no top-level `g` exists, and no
earlier module claims the name. Runtime resolution is unchanged. The proof runs
against the printed catalog and again against the whole live Program before a
save; otherwise the save refuses. The round-trip gate is "equal modulo proven
qualification": the original AST is rewritten only at the ledgered bare call
sites, then compared strictly, so any other difference still fails. A name
declared by several modules is still qualifiable when the runtime resolves it to
the caller's own module today. Such sites are classified **order-dependent**: the
save freezes today's observed binding, and the save log names each
(module, callee). The opposite case refuses: a bare call that runs another module's
function today. The save receipt also counts the rewritten sites per module.

**Runtime types.** `Attachment` and any other runtime-defined named types
(`builtins::RUNTIME_TYPES`, the single list) need no declaring module in the
serializer's type catalog. A user declaration of the same name still takes
precedence; other undeclared types are still refused.

Dependencies form a closed, acyclic set inside the manifest. Include callable
and type owners, including dependencies needed by supported stored tests. Each
module lowers against only its declared dependency closure, never stale session
definitions or implicit filename ordering. Top-level function/type catalogs and
cycles are refused in this first slice. A same-name `.py` and `.ax` pair is an
error, even if the manifest selects one. Put rollback copies outside the active
library.

`legacy_ax` is an explicit transitional manifest format for supported imports.
It is checked against the Python serializer too; it is not an escape hatch for
unsupported modules. Refused imports leave original bytes and the session cache
intact, and cannot proceed to startup. This is temporary migration eligibility,
not a permanent dual-source format. Changing formats requires an explicit
manifest/file change and server restart. This slice does not automatically
convert files, create a migration backup, or flip the default. With **no manifest**,
legacy library behavior remains unchanged.

An active manifest is a source whitelist, including when its module set is empty.
Unlisted `.ax` and `.py` files are not read or applied: diagnostics enumerate the
unlisted names. Detached snapshot writes of unlisted modules refuse instead of
implicitly creating legacy files. Cached top-level functions/types **refuse
startup**, preserving the recovery checkpoint: every function needs a module to
be saved in. A cached module absent from the manifest does not refuse startup; it
is adopted or dropped (see "Modules created at run time"), so ignoring an unlisted
file cannot leave its stale session version executable without source.
An unreadable manifest (including a dangling symlink) never selects legacy fallback.
Once loaded, manifest presence remains policy even for an **empty** module set.
Deleting the file or replacing it with a dangling link cannot re-enable legacy
reload or snapshot writes in that process. Entry guards combine retained policy
with fallible `symlink_metadata` inspection; inspection errors never mean absent.

## Modules created at run time (#128)

The manifest follows the live Program; the writer is the only thing that changes
it. Authorization happens where a module enters the live Program, never here.

**Creation.** Under a manifest a module is created by a Python mutation only.
Adapsis-source creation (a `+module` declaration of a new name, `!move` into a
new module, the `module_create`, `mutate` and `move_symbols` builtins, an agent
branch) is refused before any state change with "this library stores Python
source; create the module with a Python mutation". Editing an existing managed
module with Adapsis source is unchanged. A Python creation is acknowledged as
"accepted in memory; it will be added to the source manifest with its first
source save (pending, not yet durable)".

**Enrolment.** At each save `flush_live_sources` enrols every live module the
manifest does not list, as `python`, and keeps every listed module's dependencies
equal to what its code references: owners of qualified callees, of foreign named
types, of event sources, and of calls in stored-test data
(`library/source_enrol.rs`). Dependency sets are compared as sets; a manifest
whose declared dependencies already equal the code is never rewritten (not even
reformatted). `RunToolScratch` and names starting with `__` are never enrolled
and never reported. Not enrolled, and reported on every save until fixed:

- a module whose checked Python print fails (the module is live but has no
  source and will be dropped at the next restart);
- a module on a cross-module cycle. Cycles stay refused. Both modules are named
  and the text says how to break the cycle (move what both sides need into a
  third module). The mutation that closes the cycle gets the same text in its
  acknowledgement, as an error: the change is live, its source is not saved. A
  listed module on a cycle keeps its recorded dependencies and its old source.

**Every manifest the writer commits loads with the files on disk at that
moment.** The manifest and each source file are separate renames, so two rules
hold:

1. A dependency is only **added** before a module's source is saved (`grow`) and
   only **removed** after it was saved (`shrink`). A recorded dependency the
   source no longer needs is harmless; a missing one refuses the next start.
2. Before each commit `source_io::would_load` decodes the listed sources as they
   are on disk (plus the first text of modules being enrolled) against the new
   manifest, exactly as a start does, executing nothing. A change that does not
   pass is not committed; the module stays pending and is reported. When the
   whole new manifest does not pass, each changed module is tried on its own,
   dependencies first.

One save is: `grow` (enrol + add dependencies), save every managed module,
`shrink`. When that leaves something behind, the same save runs the three steps
once more, because its own source saves may have unblocked it: a new module
using a function another module gained in the same turn is listed only after
that module's source is on disk, and a dependency whose direction was reversed
is recorded only after the old direction was removed. Only what the last pass
leaves behind is reported. A manifest commit costs one decode of all listed
sources; a save that changes no dependency and enrols nothing commits nothing.

**Write order** per enrolment: (1) the module's `.py` (owner-only, temporary
file, rename, directory sync), (2) the manifest (the commit point; the loaded
manifest must still be the one on disk), (3) the session, later, by the autosave.
Nothing is written for a module that is not going to be listed.

| Crash after | On disk | Next start |
|---|---|---|
| nothing written | old manifest, session with or without the module | cached module adopted (below), or absent |
| (1) source | unlisted `.py`, old manifest | file ignored; if the session has the module it is adopted and the identical file reused |
| (2) manifest | listed `.py` | module loads from source, whatever the session holds |
| (3) session | all three agree | ordinary start |
| a dependency was added, source not yet saved | old source, one recorded dependency more | loads (checked before the commit) |
| source saved, dependency not yet removed | new source, one recorded dependency too many | loads; removed by the next save |

**Several source files in one save.** The source files are separate renames as
well, and a start decodes each module against the files of its dependencies.
The writer therefore prints and checks every module first and writes nothing
yet. Then a changed file is written only when the set on disk, with that one
file replaced, still loads (the same decode a start does). Dependencies are
tried first and the remaining files again after every write, so both usual
orders are found:

- a module gains a function, then its new caller is written;
- a caller stops using a function, then the module that drops it is written.

What cannot be written that way:

- The complete new set does not load, because a module it needs could not be
  printed or written. The dependent modules are not written either: they stay
  pending, and the message says which module they wait for. The set on disk
  still loads. (Before this rule the dependent was written alone and the next
  start was refused until someone repaired the files by hand.)
- **Remaining gap:** the complete new set loads, but no single file can go first
  (functions moved between two modules in both directions). The files are
  written one after another and the save log says so. A crash between those
  renames leaves a set the next start refuses (exit 78, below); the session
  cache still holds the complete Program.

A save that changes no file decodes nothing. A save that changes k files costs
at least k decodes of all listed sources (about 270 ms each on the five live
modules), under the Program read guard, off the reactor.

An unlisted leftover file with the name of a module being enrolled is reused
when its bytes equal the new source; otherwise it is moved aside as
`<Module>.py.orphan-<unix ms>` (also a leftover `.ax`), never deleted and never
read. Remove orphans by hand once inspected.

**Startup adoption.** A module that exists only in the session cache (the process
stopped before its first save) is handled before anything can run
(`adopt_cached`), in three steps: plan without writing, write the recovery file
of every module that will not be enrolled, then commit the enrolment. If it
prints it is enrolled exactly as above; otherwise it is
removed from the Program and its AST is kept in
`<library>/recovery/<Module>-<unix ms>.json` (owner-only; subdirectories are never
scanned for sources). Each adoption and each drop is one startup log line and one
journal entry (`library` category, module as source); a dropped module's cached
routes are retired. Scratch modules leave the cache silently. Lifecycle blocks of
an adopted module run with the ordinary startup, not earlier.

**Removal.** Removing a function or type inside a managed module is an ordinary
edit: the next save writes the module without it. Removing a whole managed module
is refused before any state change, naming #126 (the source would stay and the
module would return at the next start). A module that is not listed yet can still
be removed.

**Nesting.** The printer puts every binary operation and `not` operand in
parentheses and qualifies local calls, so the printed form of a function scans
up to twice as deep as its submission (measured: 199 for 100 chained operators
or nested calls). Stored source, meaning the text the printer produces and the
managed files read back, is therefore scanned with its own limit of 250
(`MAX_STORED_NESTING`); submissions keep 100. Everything a submission can
contain can be saved and loaded again. A save can still be refused for nesting
after a deeper Adapsis-source edit (more than about 124 chained operators): the
message says the printed form exceeds the limit and to split the expression,
the module stays pending and its old source stays in place. A start prints
every module it loads, so a hand-edited file is accepted only if its printed
form fits the stored limit. The limit comes down again when the printer emits
minimal parentheses (#140).

**Residual.** `!sandbox discard` restores an older Program. A module created and
enrolled inside the sandbox is then listed but not live: each save reports
"managed module absent from current program" and the module returns at the next
start, the same state as whole-module removal (#126).

## Startup, reload, and recovery

A start refused by the source policy exits with **code 78** (`EX_CONFIG`,
`library::EXIT_SOURCE_REFUSED`): the same files would refuse again, so the
service unit should carry `RestartPreventExitStatus=78`. A session file that
cannot be read (truncated, corrupt, nested deeper than the reader's cap) is the
same kind of refusal, with or without a manifest: exit 78, the file named and
left untouched, nothing initialized. The runtime keeps no backup of the session
file; restore the operator's last backup or move the file aside. "No session
yet" must be proven: only a path for which the file system answers "not found"
starts an empty session. A dangling link, a link loop, a directory, or a path
that cannot be inspected is a refused start.

**What a refused start may have written.** Never a listed source file, never the
manifest, never the session. The one exception is startup adoption: recovery
files are written BEFORE any enrolment (plan, recovery files, then source and
manifest), so a recovery file that cannot be written refuses the start with
source and manifest untouched, possibly leaving recovery files of other dropped
modules behind. Only if an enrolment write itself fails halfway and the recovery
file of a module it should have enrolled cannot be written either, an unlisted
`.py` can remain (a start ignores it; the next enrolment reuses or moves it).


The whole managed dependency set validates before replacing its in-memory
modules. Validation executes no lifecycle block; the only code it runs is the
mock-only stored-case re-run described above. Valid source
wins over session AST cache. A failed managed load aborts OS startup **before**
route reconciliation, shared initialization, memory database opening, or lifecycle
IO. It leaves both persisted source and session recovery checkpoint untouched.
Failure to create, inspect, or enumerate the source directory is likewise a
startup-blocking operational error, including an explicit non-directory path or
an inaccessible directory. Policy that could not be inspected is not legacy mode.
The diagnostic identifies the module/file and failure category without echoing
source text. Repair the isolated source/manifest and restart; the ordinary CLI
test commands remain available while OS startup is refused.

**All hot reload in a manifest-managed library requires restart**, including
unchanged routes. The refusal leaves the live program, source hashes, pending
edits and registrations untouched. This avoids invalidating in-flight mutation
snapshots without introducing a new conflict protocol across every executor.
Successful startup removes cached managed routes, then registers the newly
loaded routes once.

`!undo` and `/api/rewind` are **refused before any state change** under an active
manifest (retained, including an empty one), with the categorical error
"rewind/undo not supported for manifest-managed libraries; make an explicit edit
instead". Rewind rebuilds the Program from mutation history only; that history
has no startup baseline for disk-loaded modules, so a rewind would silently drop
or misstate them and could never be persisted truthfully. Make an explicit edit
instead. A per-module startup baseline is tracked as future work. Without a
manifest, rewind is unchanged, and the #122 earlier-rewind edit refusal is
independent of this rule.

A replacement rejected because existing stored tests fail restores the complete
prior function declaration (parameters, return type, effects, body, docs and test
flags), not only its body, so the live writer cannot persist a new signature over
an old body. Unmanifested legacy reload remains unchanged.

## Persistence and failure semantics

The manifest is the durable format authority. `LibraryState` owns the runtime
source-byte hashes and errors; no duplicate session format/hash metadata exists.
The session contains an AST recovery cache, not another editable source authority.
Managed mutation hooks enqueue names, not AST snapshots. Their acknowledgement
means **accepted in memory / source save pending**, not a disk-durability receipt.
Detached `persist_module` / `persist_program_module` writes refuse managed names.
The single writer, `flush_live_sources`, obtains the **current live Program** on
a blocking worker, then keeps its Program read guard through serialization,
source/manifest checks and rename. A delayed queued request therefore cannot
write an old snapshot. CoW mutation commits wait briefly for that guard; two
disjoint committed edits are both visible to the next flush. No new AST epoch
or silent merge rejection exists.

Lock order is Program read guard → source-transaction mutex → source metadata
locks. No model generation, evaluation, coroutine IO or async await occurs under
these guards. Source reading/writing/printing happens off the Tokio reactor.
Budgets are 64 KiB manifest, 128 modules, 2 MiB source/output per module, and
16 MiB serialized AST catalog per print. These bound inputs, **not OS filesystem
latency**; there is no hard IO deadline. Failed modules remain visibly pending
and retain their live edits. Autosave retries on later notifications through the
existing two-second debounce, never a tight failure loop.

The writer returns an immutable receipt containing saved names, failures and the
Program snapshot captured under its guard. That snapshot is used for session
checkpointing. A refusal retains the live change; reconcile it instead of
replaying the mutation. If shutdown/crash occurs before source saving, only an
actually successful session checkpoint can preserve pending edits. A merely
queued acknowledgement guarantees neither source durability nor crash recovery;
the checkpoint remains recovery material and never silently overrides valid
authoritative source on restart.

Managed source saves require a successful load baseline. Unknown, missing,
malformed, or externally changed source is never overwritten by the legacy
corrupt-file recovery exception. The manifest hash is checked too: changed or
missing policy blocks writes until reconciliation and restart. Source output goes through
a unique owner-only (0600) sibling temporary file, file sync, manifest/source
rechecks, rename, and directory sync. Interrupted temporary files are ignored.

There is a narrow **cross-process race** between the final checks and rename:
this is not a compare-and-swap filesystem API or an interprocess lock. A directory
sync failure after rename reports a committed-but-not-confirmed-durable state;
reconcile and restart before retrying. Source and session writes are not one transaction. If
source succeeds and session checkpointing fails, committed source stays
authoritative and the log explicitly reports the failed recovery checkpoint.

No live module migration is performed here. A deployment migration still needs
an explicit complete inventory, external backup/rollback bundle, review, and
isolated restart gates before replacing real files.

## Focused checks

```sh
cargo test --release managed_
cargo test --release --test python_source_startup
```

The unit fixtures use fresh temporary directories and the real serializer API.
They cover source load/validated mutation/save/reload, ordered cases without
forged pass flags, dependency closure, external edits, manifest drift,
unsupported-case refusal, all managed hot-reload refusal, delayed queued flushes,
detached old-snapshot refusal, disjoint mutations and recovery cache preservation.
`library/source_enrol_tests.rs` covers enrolment, dependency updates, cycles,
leftover files, the crash windows, startup adoption/drop, the creation and
removal refusals and the nesting notice; `api/python_turn_tests.rs` creates a
module through a native Python `mutate` turn.
The process test uses a private HOME/library/session and verifies that a broken
Python source prevents a cached startup file-write effect and database creation,
that a refused start exits with code 78, and that a cached unlisted module is
adopted or dropped without refusing the start.
