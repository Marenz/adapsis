#!/usr/bin/env python3
"""Manager-only capture preparation. Default is dry-run. Never installs/restarts.

Stages: --prepare; --apply-helpers (requires --backup-dir and --session);
--drain-capture; --stop-service. Each requires a separate explicit invocation.
No shared value is returned through eval/SSE: helpers write precreated private files.
On any failure leave state for operator reconciliation; never retry effects or roll back
automatically. Capture files are raw UTF-8 String bytes, canonical decimal Int, and
literal lowercase Bool. No credentials, API errors, or private values are printed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time
import urllib.request
import uuid

FIELDS = [(n, "String") for n in (
    "telegram_token", "token_path", "allowed_path", "admin_user_ids",
    "allowed_user_ids", "sandbox_perm_model", "intake_mode",
)] + [("poll_offset", "Int"), ("poll_running", "Bool")]
ROOT = Path("/tmp/opencode")


def require(condition):
    if not condition:
        raise RuntimeError("migration precondition failed")


def private_directory(path, create=False):
    path = Path(path)
    require(path.parent == ROOT and path.name not in ("", ".", ".."))
    require(ROOT.resolve() == ROOT)
    if create:
        path.mkdir(mode=0o700)
    info = path.lstat()
    require(stat.S_ISDIR(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o700)
    require(info.st_uid == os.getuid())
    return path


def private_read(path):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, "rb") as stream:
        info = os.fstat(stream.fileno())
        require(stat.S_ISREG(info.st_mode) and stat.S_IMODE(info.st_mode) == 0o600)
        require(info.st_uid == os.getuid() and info.st_nlink == 1)
        return stream.read()


def private_create(path, data):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    private_read(path)


def name(nonce, field):
    require(re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,31}", nonce) is not None)
    return f"migration_{nonce}_{field}"


def source(directory, nonce):
    lines = ["+module TelegramBot", f"+fn {name(nonce, 'stop')} ()->Bool [mut]",
             "  +set poll_running = false", "  +return false", "+end"]
    for suffix in ("", "_recheck"):
        for field, _ in FIELDS:
            target = json.dumps(str(directory / (field + suffix + ".value")))
            lines += [f"+fn {name(nonce, field + suffix)} ()->String [io,async]",
                      f"  +await _:String = write_file({target}, {field})",
                      '  +return "captured"', "+end"]
    return "\n".join(lines) + "\n"


def api(path, payload=None):
    request = urllib.request.Request("http://127.0.0.1:3002" + path,
        data=None if payload is None else json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=15) as response:
        require(response.status == 200)
        return json.load(response)


def evaluate(function, expected):
    result = api("/api/eval", {"function": "TelegramBot." + function, "input": ""})
    require(result.get("success") is True and result.get("result") == expected)


def prepare(args):
    validate_candidate(args.candidate, require_runtime=False)
    candidate = private_directory(args.candidate)
    directory = private_directory(args.capture_dir, create=True)
    job_id = uuid.uuid4().hex
    nonce = "job" + job_id[:24]
    for suffix in ("", "_recheck"):
        for field, _ in FIELDS:
            private_create(directory / (field + suffix + ".value"), b"")
    private_create(directory / "helpers.ax", source(directory, nonce).encode())
    record(directory, "request.json", {
        "schema_version": 3, "job_id": job_id, "nonce": nonce, "fields": FIELDS,
        "initial_candidate": str(candidate), "initial_manifest_sha256": file_hash(candidate / "manifest.json"),
        "initial_library_sha256": file_hash(candidate / "library-verified.json"),
    })


def record(directory, filename, value):
    private_create(directory / filename, json.dumps(value, sort_keys=True).encode())


def file_hash(path):
    return hashlib.sha256(private_read(path)).hexdigest()


def document(directory, filename):
    return json.loads(private_read(directory / filename))


def service_pid():
    result = subprocess.run(["systemctl", "--user", "show", "adapsis.service",
                             "-p", "MainPID", "--value"], capture_output=True, check=True)
    pid = int(result.stdout.strip())
    require(pid > 0)
    return pid


def logging_private(pid):
    env = dict(x.split(b"=", 1) for x in Path(f"/proc/{pid}/environ").read_bytes().split(b"\0") if b"=" in x)
    require(env.get(b"ADAPSIS_LOG_CONTENT", b"0") != b"1")


def runtime_identity():
    pid = service_pid()
    proc = Path(f"/proc/{pid}")
    start = (proc / "stat").read_text().rsplit(")", 1)[1].split()[19]
    executable = hashlib.sha256((proc / "exe").read_bytes()).hexdigest()
    require(service_pid() == pid)
    require((proc / "stat").read_text().rsplit(")", 1)[1].split()[19] == start)
    return {"pid": pid, "start_ticks": start, "exe_sha256": executable}


def unchanged(identity):
    require(runtime_identity() == identity)


def job_chain(directory, request):
    require(request["schema_version"] == 3)
    initial = private_directory(request["initial_candidate"])
    validate_candidate(initial, require_runtime=False)
    require(file_hash(initial / "manifest.json") == request["initial_manifest_sha256"])
    require(file_hash(initial / "library-verified.json") == request["initial_library_sha256"])
    helper = document(directory, "helpers-job.json")
    require(helper["job_id"] == request["job_id"] and helper["request_sha256"] == file_hash(directory / "request.json"))
    require(helper["helpers_sha256"] == file_hash(directory / "helpers.ax"))
    backup = private_directory(helper["backup"])
    require(file_hash(backup / "backup.json") == helper["backup_manifest_sha256"])
    saved = document(backup, "backup.json")
    require(saved["job_id"] == request["job_id"] and saved["request_sha256"] == helper["request_sha256"])
    require(set(saved["hashes"]) == {"TelegramBot.ax", "TgFormat.ax", "session-before-helpers.json"})
    for filename, expected in saved["hashes"].items():
        require(file_hash(backup / filename) == expected)
    initial_manifest = document(initial, "manifest.json")
    require([saved["hashes"][m + ".ax"] for m in ("TelegramBot", "TgFormat")] == initial_manifest["live_sha256"])
    applied = document(directory, "helpers-applied.json")
    require(applied["job_id"] == request["job_id"] and applied["helpers_job_sha256"] == file_hash(directory / "helpers-job.json"))
    return {"job_id": request["job_id"], "request_sha256": helper["request_sha256"],
            "helpers_job_sha256": file_hash(directory / "helpers-job.json"),
            "helpers_applied_sha256": file_hash(directory / "helpers-applied.json"),
            "backup_manifest_sha256": helper["backup_manifest_sha256"],
            "initial_manifest_sha256": request["initial_manifest_sha256"],
            "initial_library_sha256": request["initial_library_sha256"]}


def apply_helpers(args, directory, request):
    require(args.backup_dir and args.session)
    candidate = validate_candidate(args.candidate, require_runtime=False)
    require(file_hash(Path(args.candidate) / "manifest.json") == request["initial_manifest_sha256"])
    require(file_hash(Path(args.candidate) / "library-verified.json") == request["initial_library_sha256"])
    logging_private(service_pid())
    for field, _ in FIELDS:
        require(private_read(directory / (field + ".value")) == b"")
    # Check live symbols and original disk sources, without returning either to stdout.
    symbols = json.dumps(api("/api/query", {"query": "?symbols"}))
    helpers = [name(request["nonce"], field) for field in ["stop"] + [f + s for s in ("", "_recheck") for f, _ in FIELDS]]
    require(not any(helper in symbols for helper in helpers))
    backup = private_directory(args.backup_dir, create=True)
    files = [(Path(args.library) / (m + ".ax"), m + ".ax") for m in ("TelegramBot", "TgFormat")]
    files.append((Path(args.session), "session-before-helpers.json"))
    hashes = {}
    for src, dst in files:
        data = src.read_bytes()
        require(not any(helper.encode() in data for helper in helpers))
        private_create(backup / dst, data)
        hashes[dst] = hashlib.sha256(data).hexdigest()
    require([hashes[m + ".ax"] for m in ("TelegramBot", "TgFormat")] == candidate["live_sha256"])
    record(backup, "backup.json", {"job_id": request["job_id"], "request_sha256": file_hash(directory / "request.json"), "hashes": hashes})
    body = private_read(directory / "helpers.ax")
    require(body == source(directory, request["nonce"]).encode())
    record(directory, "helpers-job.json", {"job_id": request["job_id"], "request_sha256": file_hash(directory / "request.json"),
           "backup": str(backup), "backup_manifest_sha256": file_hash(backup / "backup.json"), "helpers_sha256": file_hash(directory / "helpers.ax")})
    # Exactly one mutation; failure is not retried. Backups survive any partial result.
    response = api("/api/mutate", {"source": body.decode()})
    require(bool(response.get("results")) and all(r.get("success") is True for r in response["results"]))
    record(directory, "helpers-applied.json", {"job_id": request["job_id"], "helpers_job_sha256": file_hash(directory / "helpers-job.json")})


def active(tasks):
    return [t for t in tasks if not t.get("status", "").startswith(("done", "failed"))]


def parse_value(field, kind, raw):
    value = raw.decode("utf-8")
    if kind == "Bool":
        require(value in ("true", "false"))
        return value == "true"
    if kind == "Int":
        require(re.fullmatch(r"0|[1-9][0-9]*", value) is not None)
        result = int(value)
        require(0 <= result < 2**63)
        return result
    require(kind == "String")
    return value


class TaskLedger:
    """Historical terminal failures may be excluded only at the pre-stop baseline.
    Every initially active/new task must reach DONE; a new poll generation is forbidden.
    Terminal registry entries are retained by this runtime; never infer done from absence.
    """
    def __init__(self, tasks=None, saved=None):
        if saved is not None:
            self.seen = set(saved["seen"])
            self.polls = set(saved["polls"])
            self.tracked = {int(k): v for k, v in saved["tracked"].items()}
            self.done = set(saved["done"])
            return
        self.seen = {t["id"] for t in tasks}
        require(len(self.seen) == len(tasks))
        self.polls = {t["id"] for t in tasks if t.get("function") == "TelegramBot.poll_loop"}
        self.tracked = {t["id"]: t.get("function") for t in active(tasks)}
        self.done = set()
        require(len([t for t in active(tasks) if t["id"] in self.polls]) == 1)

    def observe(self, tasks):
        by_id = {t["id"]: t for t in tasks}
        require(len(by_id) == len(tasks))
        for task in tasks:
            ident = task["id"]
            require(task.get("function") != "TelegramBot.poll_loop" or ident in self.polls)
            if ident not in self.seen:
                self.tracked[ident] = task.get("function")
            self.seen.add(ident)
        for ident, function in self.tracked.items():
            task = by_id.get(ident)
            if task is None:
                require(ident in self.done)
                continue
            require(task.get("function") == function and task["status"] != "failed")
            if ident in self.done:
                require(task["status"] == "done")
            if task["status"] == "done":
                self.done.add(ident)
        return set(self.tracked) == self.done and not active(tasks)

    def snapshot(self):
        return {"seen": sorted(self.seen), "polls": sorted(self.polls),
                "tracked": self.tracked, "done": sorted(self.done)}


def observe(identity, ledger, idle=False):
    unchanged(identity)
    result = ledger.observe(api("/api/tasks")["tasks"])
    unchanged(identity)
    if idle:
        require(result)
    return result


def capture_values(directory, request, identity, ledger, suffix=""):
    hashes = {}
    for field, kind in FIELDS:
        observe(identity, ledger, idle=True)
        path = directory / (field + suffix + ".value")
        require(private_read(path) == b"")
        evaluate(name(request["nonce"], field + suffix), '"captured"')
        observe(identity, ledger, idle=True)
        raw = private_read(path)
        value = parse_value(field, kind, raw)
        if field == "poll_running":
            require(value is False)
        hashes[field] = hashlib.sha256(raw).hexdigest()
    return hashes


def drain_capture(directory, request):
    chain = job_chain(directory, request)
    identity = runtime_identity()
    logging_private(identity["pid"])
    tasks = api("/api/tasks")["tasks"]
    unchanged(identity)
    ledger = TaskLedger(tasks)
    record(directory, "drain-started.json", {"chain": chain, "identity": identity, "tasks": ledger.snapshot()})
    unchanged(identity)
    evaluate(name(request["nonce"], "stop"), "false")
    deadline = time.monotonic() + 90
    while not observe(identity, ledger):
        require(time.monotonic() < deadline)
        time.sleep(0.5)
    hashes = capture_values(directory, request, identity, ledger)
    observe(identity, ledger, idle=True)
    record(directory, "capture-complete.json", {
        "schema_version": 3, "chain": chain, "identity": identity,
        "drain_started_sha256": file_hash(directory / "drain-started.json"),
        "fields": FIELDS, "sha256": hashes, "poll_terminal": True,
        "runtime_state": True, "tasks": ledger.snapshot(),
    })


def stop_service(directory, request):
    chain = job_chain(directory, request)
    complete = document(directory, "capture-complete.json")
    require(complete["schema_version"] == 3 and complete["chain"] == chain)
    identity = complete["identity"]
    require(complete["drain_started_sha256"] == file_hash(directory / "drain-started.json"))
    require(document(directory, "drain-started.json")["identity"] == identity)
    unchanged(identity)
    for field, _ in FIELDS:
        require(file_hash(directory / (field + ".value")) == complete["sha256"][field])
    ledger = TaskLedger(saved=complete["tasks"])
    observe(identity, ledger, idle=True)
    # Never overwrite the immutable baseline: separate writer functions/files.
    record(directory, "recapture-started.json", {"chain": chain, "identity": identity})
    fresh = capture_values(directory, request, identity, ledger, "_recheck")
    require(fresh == complete["sha256"])
    record(directory, "recapture-complete.json", {
        "chain": chain, "identity": identity, "sha256": fresh,
        "capture_sha256": file_hash(directory / "capture-complete.json"), "tasks": ledger.snapshot(),
    })
    observe(identity, ledger, idle=True)
    subprocess.run(["systemctl", "--user", "stop", "adapsis.service"], capture_output=True, check=True)
    state = subprocess.run(["systemctl", "--user", "show", "adapsis.service", "-p", "MainPID", "--value"], capture_output=True, check=True)
    require(state.stdout.strip() == b"0")
    record(directory, "service-stopped.json", {"schema_version": 3, "stopped": True,
        "chain": chain, "identity": identity, "capture_sha256": file_hash(directory / "capture-complete.json"),
        "recapture_sha256": file_hash(directory / "recapture-complete.json")})


def stopped_provenance(directory):
    directory = private_directory(directory)
    request = document(directory, "request.json")
    chain = job_chain(directory, request)
    complete = document(directory, "capture-complete.json")
    fresh = document(directory, "recapture-complete.json")
    stopped = document(directory, "service-stopped.json")
    require(complete["schema_version"] == 3 and stopped["schema_version"] == 3 and stopped["stopped"] is True)
    require(complete["chain"] == fresh["chain"] == stopped["chain"] == chain)
    require(complete["identity"] == fresh["identity"] == stopped["identity"])
    require(complete["sha256"] == fresh["sha256"])
    require(complete["drain_started_sha256"] == file_hash(directory / "drain-started.json"))
    require(document(directory, "drain-started.json")["identity"] == complete["identity"])
    require(document(directory, "drain-started.json")["chain"] == chain)
    capture_hash = file_hash(directory / "capture-complete.json")
    require(fresh["capture_sha256"] == stopped["capture_sha256"] == capture_hash)
    require(stopped["recapture_sha256"] == file_hash(directory / "recapture-complete.json"))
    require(set(complete["sha256"]) == {f for f, _ in FIELDS})
    for field, _ in FIELDS:
        require(file_hash(directory / (field + ".value")) == complete["sha256"][field])
        require(file_hash(directory / (field + "_recheck.value")) == complete["sha256"][field])
    return dict(chain, identity=complete["identity"], capture_sha256=capture_hash,
                recapture_sha256=file_hash(directory / "recapture-complete.json"),
                stopped_sha256=file_hash(directory / "service-stopped.json"))


def validate_candidate(path, require_runtime, capture_dir=None):
    require(path is not None)
    directory = private_directory(path)
    raw = private_read(directory / "manifest.json")
    manifest = json.loads(raw)
    require(manifest.get("schema_version") == 3 and manifest.get("superseded") is False)
    if require_runtime:
        require(manifest.get("runtime_shared_state_captured") is True)
        require(capture_dir is not None and manifest.get("capture_provenance") == stopped_provenance(capture_dir))
        request = document(private_directory(capture_dir), "request.json")
        require(manifest.get("live_sha256") == document(private_directory(request["initial_candidate"]), "manifest.json")["live_sha256"])
    attestation = json.loads(private_read(directory / "library-verified.json"))
    require(attestation.get("schema_version") == 3 and attestation.get("load_reload_passed") is True)
    require(attestation.get("manifest_sha256") == hashlib.sha256(raw).hexdigest())
    require(attestation.get("artifacts") == manifest.get("artifacts"))
    require(attestation.get("stored_cases") == manifest.get("pure_cases_passed"))
    require(set(manifest["artifacts"]) == {m + ext for m in ("TelegramBot", "TgFormat") for ext in (".ax", ".py")})
    require({p.name for p in directory.glob("*.ax")} == {"TelegramBot.ax", "TgFormat.ax"})
    for name, expected in manifest["artifacts"].items():
        require(hashlib.sha256(private_read(directory / name)).hexdigest() == expected)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", required=True)
    parser.add_argument("--backup-dir")
    parser.add_argument("--session")
    parser.add_argument("--candidate", help="reviewed v3 private bundle with hash-bound load/reload attestation")
    parser.add_argument("--library", default=str(Path.home() / ".config/adapsis/modules"))
    actions = parser.add_mutually_exclusive_group()
    for option in ("prepare", "apply-helpers", "drain-capture", "stop-service", "verify-candidate"):
        actions.add_argument("--" + option, action="store_true")
    args = parser.parse_args()
    if not any((args.prepare, args.apply_helpers, args.drain_capture, args.stop_service, args.verify_candidate)):
        print("DRY RUN: no writes, API calls, or service actions; choose one explicit stage")
        return
    if args.verify_candidate:
        validate_candidate(args.candidate, require_runtime=True, capture_dir=args.capture_dir)
    elif args.prepare:
        prepare(args)
    else:
        directory = private_directory(args.capture_dir)
        request = json.loads(private_read(directory / "request.json"))
        require(request["fields"] == [list(x) for x in FIELDS])
        validate_candidate(args.candidate, require_runtime=False)
        if args.apply_helpers:
            apply_helpers(args, directory, request)
        elif args.drain_capture:
            drain_capture(directory, request)
        else:
            stop_service(directory, request)
    print("STAGE COMPLETE (private values omitted)")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        raise SystemExit("STAGE BLOCKED: inspect private state; no automatic retry or rollback")
