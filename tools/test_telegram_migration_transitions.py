"""Discriminating lifecycle tests. All service/API/eval effects are mocked."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import telegram_migration_capture as c

IDENTITY = {"pid": 123, "start_ticks": "456", "exe_sha256": "e" * 64}


def task(ident, status, function="Worker.run"):
    return {"id": ident, "status": status, "function": function}


class TransitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.patch = patch.object(c, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def candidate(self, name, provenance=None):
        directory = c.private_directory(self.root / name, True)
        hashes = {}
        for module in ("TelegramBot", "TgFormat"):
            for ext in (".ax", ".py"):
                filename = module + ext
                c.private_create(directory / filename, module.encode())
                hashes[filename] = hashlib.sha256(module.encode()).hexdigest()
        manifest = {"schema_version": 3, "superseded": False,
                    "runtime_shared_state_captured": provenance is not None,
                    "capture_provenance": provenance, "artifacts": hashes, "pure_cases_passed": 2,
                    "live_sha256": [hashes[m + ".ax"] for m in ("TelegramBot", "TgFormat")]}
        c.record(directory, "manifest.json", manifest)
        c.record(directory, "library-verified.json", {"schema_version": 3, "load_reload_passed": True,
                 "stored_cases": 2, "manifest_sha256": c.file_hash(directory / "manifest.json"), "artifacts": hashes})
        return directory

    def job(self, label):
        initial = self.candidate("initial" + label)
        directory = c.private_directory(self.root / ("job" + label), True)
        for suffix in ("", "_recheck"):
            for field, _ in c.FIELDS:
                c.private_create(directory / (field + suffix + ".value"), b"")
        c.private_create(directory / "helpers.ax", c.source(directory, "fixture" + label).encode())
        request = {"schema_version": 3, "job_id": label, "nonce": "fixture" + label, "fields": c.FIELDS,
                   "initial_candidate": str(initial), "initial_manifest_sha256": c.file_hash(initial / "manifest.json"),
                   "initial_library_sha256": c.file_hash(initial / "library-verified.json")}
        c.record(directory, "request.json", request)
        backup = c.private_directory(self.root / ("backup" + label), True)
        hashes = {}
        for filename, raw in [("TelegramBot.ax", b"TelegramBot"), ("TgFormat.ax", b"TgFormat"), ("session-before-helpers.json", b"{}")]:
            c.private_create(backup / filename, raw)
            hashes[filename] = hashlib.sha256(raw).hexdigest()
        c.record(backup, "backup.json", {"job_id": label, "request_sha256": c.file_hash(directory / "request.json"), "hashes": hashes})
        c.record(directory, "helpers-job.json", {"job_id": label, "request_sha256": c.file_hash(directory / "request.json"),
                 "backup": str(backup), "backup_manifest_sha256": c.file_hash(backup / "backup.json"),
                 "helpers_sha256": c.file_hash(directory / "helpers.ax")})
        c.record(directory, "helpers-applied.json", {"job_id": label, "helpers_job_sha256": c.file_hash(directory / "helpers-job.json")})
        return directory, request

    def effects(self, directory, request, drift=None):
        state = {"stopped_flag": False, "captures": 0}

        def evaluate(function, expected):
            if function == c.name(request["nonce"], "stop"):
                state["stopped_flag"] = True
                return
            for suffix in ("", "_recheck"):
                for field, kind in c.FIELDS:
                    if function == c.name(request["nonce"], field + suffix):
                        raw = b"false" if kind == "Bool" else b"7" if kind == "Int" else b"synthetic"
                        if suffix and field == drift:
                            raw = b"8" if kind == "Int" else b"changed authorization"
                        (directory / (field + suffix + ".value")).write_bytes(raw)
                        state["captures"] += 1
                        return
            self.fail("unexpected helper")

        def api(*args, **kwargs):
            status = "done" if state["stopped_flag"] else "running"
            return {"tasks": [task(1, status, "TelegramBot.poll_loop"), task(2, status), task(99, "failed")]}
        return state, evaluate, api

    def captured(self, label):
        directory, request = self.job(label)
        state, evaluate, api = self.effects(directory, request)
        with patch.object(c, "runtime_identity", return_value=IDENTITY), patch.object(c, "logging_private"), \
             patch.object(c, "evaluate", side_effect=evaluate), patch.object(c, "api", side_effect=api):
            c.drain_capture(directory, request)
        self.assertEqual(state["captures"], 9)
        return directory, request

    def test_pid_or_executable_changes_during_capture_blocks_completion(self):
        for label, key, value in [("pid", "pid", 124), ("exe", "exe_sha256", "f" * 64), ("start", "start_ticks", "457")]:
            directory, request = self.job(label)
            state, evaluate, api = self.effects(directory, request)
            def identity():
                result = dict(IDENTITY)
                if state["captures"] >= 3:
                    result[key] = value
                return result
            with patch.object(c, "runtime_identity", side_effect=identity), patch.object(c, "logging_private"), \
                 patch.object(c, "evaluate", side_effect=evaluate), patch.object(c, "api", side_effect=api):
                with self.assertRaises(RuntimeError):
                    c.drain_capture(directory, request)
            self.assertFalse((directory / "capture-complete.json").exists())

    def test_failed_or_disappeared_worker_blocks_capture(self):
        for label, final in [("failed", [task(1, "done", "TelegramBot.poll_loop"), task(2, "failed")]),
                             ("gone", [task(1, "done", "TelegramBot.poll_loop")])]:
            directory, request = self.job(label)
            baseline = [task(1, "running", "TelegramBot.poll_loop"), task(2, "running")]
            with patch.object(c, "runtime_identity", return_value=IDENTITY), patch.object(c, "logging_private"), \
                 patch.object(c, "evaluate") as evaluate, patch.object(c, "api", side_effect=[{"tasks": baseline}, {"tasks": final}]):
                with self.assertRaises(RuntimeError):
                    c.drain_capture(directory, request)
                self.assertEqual(evaluate.call_count, 1)  # stop flag only; no capture.
            self.assertFalse((directory / "capture-complete.json").exists())

    def test_new_work_is_tracked_until_done_and_new_poll_generation_rejected(self):
        ledger = c.TaskLedger([task(1, "running", "TelegramBot.poll_loop"), task(99, "failed")])
        self.assertFalse(ledger.observe([task(1, "done", "TelegramBot.poll_loop"), task(3, "running")]))
        with self.assertRaises(RuntimeError):
            ledger.observe([task(1, "done", "TelegramBot.poll_loop"), task(3, "failed")])
        with self.assertRaises(RuntimeError):
            ledger.observe([task(1, "done", "TelegramBot.poll_loop"), task(4, "done", "TelegramBot.poll_loop")])

    def test_same_pid_idle_cursor_or_auth_drift_prevents_service_stop(self):
        for label, field in [("cursor", "poll_offset"), ("auth", "admin_user_ids")]:
            directory, request = self.captured(label)
            baseline = {f: c.file_hash(directory / (f + ".value")) for f, _ in c.FIELDS}
            state, evaluate, api = self.effects(directory, request, drift=field)
            state["stopped_flag"] = True
            with patch.object(c, "runtime_identity", return_value=IDENTITY), patch.object(c, "evaluate", side_effect=evaluate), \
                 patch.object(c, "api", side_effect=api), patch.object(c.subprocess, "run") as stop:
                with self.assertRaises(RuntimeError):
                    c.stop_service(directory, request)
                stop.assert_not_called()
            self.assertEqual(baseline, {f: c.file_hash(directory / (f + ".value")) for f, _ in c.FIELDS})
            self.assertFalse((directory / "service-stopped.json").exists())

    def stopped(self, label):
        directory, request = self.captured(label)
        state, evaluate, api = self.effects(directory, request)
        state["stopped_flag"] = True
        with patch.object(c, "runtime_identity", return_value=IDENTITY), patch.object(c, "evaluate", side_effect=evaluate), \
             patch.object(c, "api", side_effect=api), patch.object(c.subprocess, "run", return_value=SimpleNamespace(stdout=b"0")):
            c.stop_service(directory, request)
        return directory

    def test_two_valid_jobs_cannot_be_swapped_and_stop_attestation_is_mandatory(self):
        job_a, job_b = self.stopped("A"), self.stopped("B")
        final = self.candidate("final", c.stopped_provenance(job_a))
        c.validate_candidate(final, require_runtime=True, capture_dir=job_a)
        with self.assertRaises(RuntimeError):
            c.validate_candidate(final, require_runtime=True, capture_dir=job_b)
        (job_a / "service-stopped.json").unlink()
        with self.assertRaises(FileNotFoundError):
            c.stopped_provenance(job_a)


if __name__ == "__main__":
    unittest.main()
