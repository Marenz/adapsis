import contextlib
import io
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import telegram_migration_capture as capture


class CaptureSafetyTests(unittest.TestCase):
    def test_private_files_are_exclusive_and_refuse_symlinks_and_public_mode(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "value"
            capture.private_create(path, b"false")
            self.assertEqual(capture.private_read(path), b"false")
            with self.assertRaises(FileExistsError):
                capture.private_create(path, b"replacement")
            link = Path(root) / "link"
            link.symlink_to(path)
            with self.assertRaises(OSError):
                capture.private_read(link)
            os.chmod(path, 0o644)
            with self.assertRaises(RuntimeError):
                capture.private_read(path)

    def test_directory_refuses_symlink_and_requires_0700(self):
        with tempfile.TemporaryDirectory() as root, patch.object(capture, "ROOT", Path(root)):
            private = capture.private_directory(Path(root) / "capture", create=True)
            link = Path(root) / "link"
            link.symlink_to(private, target_is_directory=True)
            with self.assertRaises(RuntimeError):
                capture.private_directory(link)
            os.chmod(private, 0o755)
            with self.assertRaises(RuntimeError):
                capture.private_directory(private)

    def test_typed_values_preserve_strings_and_refuse_lossy_numbers(self):
        self.assertEqual(capture.parse_value("telegram_token", "String", b"  text\n"), "  text\n")
        self.assertIs(capture.parse_value("poll_running", "Bool", b"false"), False)
        self.assertEqual(capture.parse_value("poll_offset", "Int", b"0"), 0)
        self.assertEqual(capture.parse_value("poll_offset", "Int", b"9223372036854775807"), 2**63-1)
        for value in [b"-1", b"+1", b"01", b"1\n", b"9223372036854775808"]:
            with self.assertRaises(RuntimeError):
                capture.parse_value("poll_offset", "Int", value)
        for value in [b"False", b"0", b"false\n"]:
            with self.assertRaises(RuntimeError):
                capture.parse_value("poll_running", "Bool", value)

    def test_default_dry_run_has_no_actions(self):
        args = ["capture", "--capture-dir", "/tmp/opencode/example"]
        with patch("sys.argv", args), patch.object(capture, "api") as api, patch.object(capture, "prepare") as prepare:
            with contextlib.redirect_stdout(io.StringIO()):
                capture.main()
            api.assert_not_called()
            prepare.assert_not_called()

    def test_helpers_never_return_shared_values(self):
        source = capture.source(Path("/tmp/opencode/example"), "testnonce")
        self.assertEqual(source.count("+fn "), 19)
        self.assertEqual(source.count("write_file("), 18)
        self.assertEqual(source.count('+return "captured"'), 18)
        self.assertEqual(source.count("+return false"), 1)
        for field, _ in capture.FIELDS:
            self.assertNotIn("+return " + field, source)
        with self.assertRaises(RuntimeError):
            capture.source(Path("/tmp/opencode/example"), 'unsafe"nonce')

    def test_candidate_gate_requires_v3_attestation_and_runtime_capture(self):
        with tempfile.TemporaryDirectory() as root, patch.object(capture, "ROOT", Path(root)):
            directory = capture.private_directory(Path(root) / "candidate", create=True)
            hashes = {}
            for module in ("TelegramBot", "TgFormat"):
                for ext in (".py", ".ax"):
                    key = module + ext
                    capture.private_create(directory / key, b"inert fixture")
                    hashes[key] = hashlib.sha256(b"inert fixture").hexdigest()
            manifest = {"schema_version": 3, "superseded": False,
                        "runtime_shared_state_captured": False, "pure_cases_passed": 2, "artifacts": hashes}
            raw = json.dumps(manifest).encode()
            capture.private_create(directory / "manifest.json", raw)
            capture.private_create(directory / "library-verified.json", json.dumps({
                "schema_version": 3, "load_reload_passed": True, "stored_cases": 2,
                "manifest_sha256": hashlib.sha256(raw).hexdigest(), "artifacts": hashes,
            }).encode())
            capture.validate_candidate(directory, require_runtime=False)
            with self.assertRaises(RuntimeError):
                capture.validate_candidate(directory, require_runtime=True)
            manifest["schema_version"] = 1
            (directory / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(RuntimeError):
                capture.validate_candidate(directory, require_runtime=False)


if __name__ == "__main__":
    unittest.main()
