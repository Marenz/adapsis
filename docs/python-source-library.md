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

`@stored_test` does not expand native mutation/test input policy. Function names
starting with `test_` alone do not identify persisted tests. Every supported
member and every stored case must be represented; an unsupported member/test
refuses the entire save. Cases retain order within their target function.
Loading restores expectations but marks cases **untested**. Parsing source is
not evidence of passing tests; rerun them through the existing test executor.

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
unlisted names and require explicit entries/dependencies before restart. Unlisted
writes refuse instead of implicitly creating legacy files. A cached module absent
from the manifest also **refuses startup**, naming the excluded modules while
preserving the recovery checkpoint. Cached top-level functions/types likewise
refuse: this first slice requires explicit module ownership. Thus ignoring an
unlisted file cannot accidentally leave its stale session version executable.
An unreadable manifest (including a dangling symlink) never selects legacy fallback.
Once loaded, manifest presence remains policy even for an **empty** module set.
Deleting the file or replacing it with a dangling link cannot re-enable legacy
reload or snapshot writes in that process. Entry guards combine retained policy
with fallible `symlink_metadata` inspection; inspection errors never mean absent.

## Startup, reload, and recovery

The whole managed dependency set validates before replacing its in-memory
modules. Validation executes neither lifecycle blocks nor tests. Valid source
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
loaded routes once. Unmanifested legacy reload remains unchanged.

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
The process test uses a private HOME/library/session and verifies that a broken
Python source prevents a cached startup file-write effect and database creation.
