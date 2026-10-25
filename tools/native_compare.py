#!/usr/bin/env python3
"""Native tools + Python frontend vs legacy Adapsis protocol comparison harness.

Drives the released ./target/release/adapsis binary as fully isolated instances
(loopback ports 3110+, per-run temp session/modules/contexts/memory), sends one
task per fresh instance via POST /api/ask, then scores the turn with HIDDEN
eval cases the model never saw.

Isolation contract (mandatory): never touches the production instance on :3002
or its session/modules/contexts/memory. Every instance is started with
ADAPSIS_MODULES_DIR / ADAPSIS_CONTEXTS_DIR / ADAPSIS_MEMORY_DB pointed at a
per-run temp dir, an absolute temp session path, and --host 127.0.0.1 only.

No Rust edits. Read-only over the source tree. Sole writer of the two tools/
files listed by the manager plus the docs/ summary.
"""

import argparse
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, ".."))
DEFAULT_TASKS = os.path.join(HERE, "native_compare_tasks.json")
WORK_ROOT = "/tmp/opencode/compare"

CONDITIONS = {
    "legacy": "legacy",
    "native-adapsis": "native-adapsis",
    "native-python": "native-python",
}

USAGE_RE = re.compile(r"\[llm:main\] latency_ms=(\d+) usage: input=(\d+) .*? output=(\d+) provider_total=(\d+)")

# Markers used to classify outcomes.
ITER_LIMIT_MARKER = "Iteration limit reached"
LLM_ERROR_MARKER = "LLM error"
PROTOCOL_ERROR_MARKERS = ("Native protocol error", "Native provider error")
UNTESTED_MARKER = "no passing tests"

# Syntax/validation error fragments seen in executor results.
VALIDATION_FRAGMENTS = (
    "parse error",
    "must be a qualified function name",
    "mutate forbids",
    "language must be",
    "permission denied",
    "not a valid expression",
    "lowering",
    "unexpected field",
    "missing required field",
    "arguments exceed",
    "must not contain newlines",
    "python module metadata",
    "module metadata is accepted only",
    "test tool admits",
    "cannot guarantee literal",
    "authored",
)


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# HTTP helpers (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def post_json(url: str, payload: dict, timeout: float) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    return json.loads(body)


def get_json(url: str, timeout: float) -> dict:
    req = urllib.request.Request(url, headers={"Content-Type": "application/json"}, method="GET")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8", "replace")
    return json.loads(body)


def pick_free_port(base: int, attempt: int) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]
    finally:
        s.close()


_PORT_LOCK = threading.Lock()
_PORT_COUNTER = {"n": 0}
_IN_USE = set()


def alloc_port(base: int) -> int:
    """Allocate a distinct, verified-free loopback port across concurrent workers."""
    with _PORT_LOCK:
        while True:
            _PORT_COUNTER["n"] += 1
            port = base + _PORT_COUNTER["n"]
            if port in _IN_USE:
                continue
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                s.bind(("127.0.0.1", port))
                _IN_USE.add(port)
                return port
            except OSError:
                continue
            finally:
                s.close()


def release_port(port: int) -> None:
    with _PORT_LOCK:
        _IN_USE.discard(port)


# ─────────────────────────────────────────────────────────────────────────────
# Instance management
# ─────────────────────────────────────────────────────────────────────────────

class Instance:
    def __init__(self, binary, url, model, run_dir, port, native):
        self.binary = binary
        self.url = url
        self.model = model
        self.run_dir = run_dir
        self.port = port
        self.native = native
        self.proc = None
        self.stderr_path = os.path.join(run_dir, "instance.log")

    def start(self) -> None:
        modules = os.path.join(self.run_dir, "modules")
        contexts = os.path.join(self.run_dir, "contexts")
        memory = os.path.join(self.run_dir, "memory")
        for d in (modules, contexts, memory):
            os.makedirs(d, exist_ok=True)

        env = os.environ.copy()
        env["ADAPSIS_MODULES_DIR"] = modules
        env["ADAPSIS_CONTEXTS_DIR"] = contexts
        env["ADAPSIS_MEMORY_DB"] = os.path.join(memory, "memory.lbug")

        session_path = os.path.join(self.run_dir, "session.json")
        log_file = os.path.join(self.run_dir, "adapsisos.log")
        training_log = os.path.join(self.run_dir, "training.jsonl")

        cmd = [
            self.binary, "os",
            "--port", str(self.port),
            "-s", session_path,
            "--url", self.url,
            "--model", self.model,
            "--access-level", "adapsis-only",
            "--host", "127.0.0.1",
            "--log-file", log_file,
            "--training-log", training_log,
        ]
        if self.native:
            cmd.append("--native-tools")

        with open(self.stderr_path, "wb") as out:
            self.proc = subprocess.Popen(
                cmd,
                stdout=out,
                stderr=subprocess.STDOUT,
                env=env,
                cwd=WORK_ROOT,
                start_new_session=True,
            )

    def wait_ready(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        base = f"http://127.0.0.1:{self.port}"
        while time.time() < deadline:
            if self.proc.poll() is not None:
                return False
            try:
                get_json(base + "/api/status", timeout=2.0)
                return True
            except Exception:
                time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
            except Exception:
                pass
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
        if self.proc.poll() is None:
            try:
                self.proc.kill()
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────────────────────
# Task text construction
# ─────────────────────────────────────────────────────────────────────────────

def build_task_text(task: dict, condition: str, glossary: str) -> str:
    base = task["base_task"]
    text = f"{base}\n\n{glossary}"
    if condition == "legacy":
        return text
    lang = "python" if condition == "native-python" else "adapsis"
    directive = (
        f"\n\nUse the native tool protocol with language=\"{lang}\" for mutate and test, "
        f"and language=\"{lang}\" for evaluate. Call finish once the task is done."
    )
    return text + directive


# ─────────────────────────────────────────────────────────────────────────────
# Outcome classification
# ─────────────────────────────────────────────────────────────────────────────

def classify(resp: dict, hidden: list, condition: str, completed: bool) -> str:
    """Return the failure class for a run that did not fully succeed."""
    msgs = [r.get("message", "") for r in resp.get("results", [])] + [
        r.get("message", "") for r in resp.get("test_results", [])
    ]
    joined = "\n".join(msgs) + "\n" + resp.get("reply", "")
    if not completed:
        if any(m in joined for m in PROTOCOL_ERROR_MARKERS) or LLM_ERROR_MARKER in joined:
            return "protocol-error"
        if any(f in joined.lower() for f in VALIDATION_FRAGMENTS):
            return "no-completion-validation"
        return "no-completion"
    # completed but wrong
    if any(h["actual_startswith_untested"] for h in hidden):
        return "untested-eval-refusal"
    return "wrong-result"


def summarize_hidden(hidden: list) -> dict:
    passed = sum(1 for h in hidden if h["ok"])
    return {
        "passed": passed,
        "total": len(hidden),
        "all_pass": passed == len(hidden) and len(hidden) > 0,
        "untested_refusals": sum(1 for h in hidden if h.get("actual_startswith_untested")),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Single run
# ─────────────────────────────────────────────────────────────────────────────

def run_one(cfg: dict, task: dict, condition: str, model: str, repeat: int, port: int) -> dict:
    run_id = f"{task['id']}.{condition}.{model}.{repeat}"
    run_dir = os.path.join(WORK_ROOT, "runs", run_id)
    os.makedirs(run_dir, exist_ok=True)

    record = {
        "task": task["id"],
        "condition": condition,
        "model": model,
        "repeat": repeat,
        "run_id": run_id,
        "port": port,
        "completed": False,
        "success": False,
        "hidden_passed": 0,
        "hidden_total": len(task["hidden_cases"]),
        "provider_calls": 0,
        "wall_ms": None,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_latency_ms": 0,
        "failure_class": None,
        "observed_errors": False,
        "stderr_path": os.path.join(run_dir, "instance.log"),
        "hidden": [],
    }

    instance = Instance(cfg["binary"], cfg["url"], model, run_dir, port, condition != "legacy")
    try:
        instance.start()
        if not instance.wait_ready(timeout=90):
            record["failure_class"] = "startup-failure"
            return record

        base = f"http://127.0.0.1:{port}"
        text = build_task_text(task, condition, cfg["glossary"])

        started = time.time()
        try:
            resp = post_json(base + "/api/ask", {"message": text}, timeout=cfg["timeout"])
        except urllib.error.HTTPError as e:
            record["wall_ms"] = int((time.time() - started) * 1000)
            record["failure_class"] = "http-error"
            record["observed_errors"] = True
            return record
        except Exception as e:
            record["wall_ms"] = int((time.time() - started) * 1000)
            record["failure_class"] = "timeout" if "timed out" in str(e) else "http-error"
            return record
        record["wall_ms"] = int((time.time() - started) * 1000)

        # Completion determination.
        if condition == "legacy":
            msgs = [r.get("message", "") for r in resp.get("results", [])]
            completed = not any(ITER_LIMIT_MARKER in m for m in msgs)
        else:
            completed = resp.get("completed") is True
        record["completed"] = completed

        # Observed errors (may coexist with completion for native).
        record["observed_errors"] = bool(resp.get("has_errors")) or any(
            not r.get("success", False) for r in resp.get("results", [])
        ) or any(not r.get("pass", False) for r in resp.get("test_results", []))

        # Hidden checks via isolated /api/eval.
        hidden = []
        for case in task["hidden_cases"]:
            h = {"input": case["input"], "expected": case["expected"], "actual": None, "ok": False,
                 "actual_startswith_untested": False}
            try:
                ev = post_json(
                    base + "/api/eval",
                    {"function": task["function"], "input": case["input"]},
                    timeout=30,
                )
                actual = (ev.get("result") or "").strip()
                h["actual"] = actual
                if not ev.get("success") and UNTESTED_MARKER in actual:
                    h["actual_startswith_untested"] = True
                h["ok"] = bool(ev.get("success")) and actual == case["expected"]
            except Exception as e:
                h["actual"] = f"<eval request failed: {e}>"
            hidden.append(h)
        record["hidden"] = hidden
        sh = summarize_hidden(hidden)
        record["hidden_passed"] = sh["passed"]
        record["success"] = completed and sh["all_pass"]

        if record["success"]:
            record["failure_class"] = "none"
        else:
            record["failure_class"] = classify(resp, hidden, condition, completed)

        record["reply_preview"] = (resp.get("reply") or "")[:400]
        record["result_tail"] = [r.get("message", "") for r in resp.get("results", [])][-5:]

        return record
    finally:
        instance.stop()
        # Always parse stderr accounting, even on timeout/error paths, so partial
        # provider calls and token usage are not lost when /api/ask did not return.
        record["provider_calls"], record["llm_latency_ms"], tok = parse_stderr(record["stderr_path"])
        record["prompt_tokens"] = tok[0]
        record["completion_tokens"] = tok[1]
        record["total_tokens"] = tok[2]


def parse_stderr(path: str):
    calls = 0
    latency = 0
    pt = ct = tt = 0
    try:
        with open(path, "r", errors="replace") as f:
            for line in f:
                m = USAGE_RE.search(line)
                if m:
                    calls += 1
                    latency += int(m.group(1))
                    pt += int(m.group(2))
                    ct += int(m.group(3))
                    tt += int(m.group(4))
    except OSError:
        pass
    return calls, latency, (pt, ct, tt)


# ─────────────────────────────────────────────────────────────────────────────
# Orchestration
# ─────────────────────────────────────────────────────────────────────────────

def build_matrix(cfg: dict, args) -> list:
    tasks = [t for t in cfg["tasks"] if not args.task_ids or t["id"] in args.task_ids]
    if args.smoke:
        tasks = tasks[:1]
    models = args.models
    if args.smoke:
        models = models[:1]
    conditions = args.conditions
    repeats = 1 if args.smoke else args.repeats
    matrix = []
    for task in tasks:
        for cond in conditions:
            for model in models:
                for r in range(1, repeats + 1):
                    matrix.append((task, cond, model, r))
    return matrix


def write_summary(cfg: dict, records: list, summary_path: str) -> str:
    from collections import defaultdict
    groups = defaultdict(list)
    for rec in records:
        groups[(rec["condition"], rec["model"])].append(rec)

    lines = []
    lines.append("")
    lines.append("SUMMARY (condition x model)")
    lines.append("=" * 90)
    header = (
        f"{'condition/model':<28} {'n':>3} {'succ':>5} {'done':>5} "
        f"{'iter':>6} {'inTok':>7} {'outTok':>7} {'latMs':>7}"
    )
    lines.append(header)
    for key in sorted(groups.keys()):
        recs = groups[key]
        n = len(recs)
        succ = sum(1 for r in recs if r["success"]) / n if n else 0
        done = sum(1 for r in recs if r["completed"]) / n if n else 0
        iters = sum(r["provider_calls"] for r in recs) / n if n else 0
        pt = sum(r["prompt_tokens"] for r in recs) / n if n else 0
        ct = sum(r["completion_tokens"] for r in recs) / n if n else 0
        lat = sum(r["llm_latency_ms"] for r in recs) / n if n else 0
        fc = defaultdict(int)
        for r in recs:
            fc[r["failure_class"]] += 1
        label = f"{key[0]} / {key[1]}"
        lines.append(
            f"{label:<28} {n:>3} {succ*100:>4.0f}% {done*100:>4.0f}% "
            f"{iters:>6.1f} {pt:>7.0f} {ct:>7.0f} {lat:>7.0f}"
        )
        lines.append(f"{'':<28}    failures: {dict(fc)}")
    lines.append("")
    text = "\n".join(lines)
    log(text)
    with open(summary_path, "w") as f:
        f.write(text + "\n")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tasks", default=DEFAULT_TASKS)
    ap.add_argument("--models", default="deepseek-v4-flash,deepseek-v4-pro")
    ap.add_argument("--conditions", default="legacy,native-adapsis,native-python")
    ap.add_argument("--task-ids", default="")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--concurrency", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--binary", default=os.path.join(REPO, "target/release/adapsis"))
    ap.add_argument("--url", default="http://127.0.0.1:4000")
    ap.add_argument("--results-dir", default=os.path.join(WORK_ROOT, "results"))
    ap.add_argument("--port-base", type=int, default=3110)
    args = ap.parse_args()

    with open(args.tasks) as f:
        task_file = json.load(f)
    cfg = {
        "binary": args.binary,
        "url": args.url,
        "timeout": args.timeout,
        "glossary": task_file.get("type_glossary", ""),
        "tasks": task_file["tasks"],
    }
    args.models = [m.strip() for m in args.models.split(",") if m.strip()]
    args.conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    args.task_ids = {t.strip() for t in args.task_ids.split(",") if t.strip()}

    if not os.path.exists(args.binary):
        log(f"binary not found: {args.binary}")
        return 2

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs(os.path.join(WORK_ROOT, "runs"), exist_ok=True)

    matrix = build_matrix(cfg, args)
    log(f"matrix: {len(matrix)} runs "
        f"(tasks={len({t[0]['id'] for t in matrix})} "
        f"conditions={len(set(t[1] for t in matrix))} "
        f"models={len(set(t[2] for t in matrix))} repeats={args.repeats if not args.smoke else 1})")
    total = len(matrix)
    if total == 0:
        log("empty matrix")
        return 2

    jsonl_path = os.path.join(args.results_dir, "runs.jsonl")
    lock = threading.Lock()
    done_count = 0

    def work(item) -> dict:
        nonlocal done_count
        task, cond, model, repeat = item
        port = alloc_port(args.port_base)
        rec = run_one(cfg, task, cond, model, repeat, port)
        release_port(port)
        with lock:
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            done_count += 1
            log(f"[{done_count}/{total}] {rec['run_id']} success={rec['success']} "
                f"class={rec['failure_class']} calls={rec['provider_calls']} "
                f"wall={rec['wall_ms']}ms tokens={rec['total_tokens']}")
        return rec

    records = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futures = [ex.submit(work, item) for item in matrix]
        for fut in as_completed(futures):
            records.append(fut.result())

    summary_path = os.path.join(args.results_dir, "summary.txt")
    write_summary(cfg, records, summary_path)
    log(f"results dir: {args.results_dir}")
    log(f"jsonl: {jsonl_path}")
    log(f"summary: {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
