import unittest

import check_guest_creative_config as checker


FRAGMENT = {
    "principals": {"telegram:user:100000101": "creative"},
    "guest_profiles": {"creative": ["music_generate", "cad_prepare"]},
    "model": {
        "creative": {
            "core": "none",
            "user": "none",
            "opencode": False,
        }
    },
}


class CreativeConfigCheckTests(unittest.TestCase):
    def test_unrelated_live_entries_are_preserved_and_ignored(self):
        live = {
            "principals": {
                "telegram:user:100000101": "creative",
                "telegram:user:1": "admin",
            },
            "guest_profiles": {
                "creative": ["music_generate", "cad_prepare"],
                "other": ["web_read"],
            },
            "model": {
                "creative": {
                    "core": "none",
                    "user": "none",
                    "opencode": False,
                    "future_setting": "preserved",
                },
                "admin": {"core": "write"},
            },
            "groups": {"core": ["TelegramBot"]},
        }
        self.assertEqual(checker.mismatches(FRAGMENT, live), [])

    def test_missing_and_changed_reviewed_entries_fail(self):
        live = {
            "principals": {"telegram:user:100000101": "admin"},
            "guest_profiles": {"creative": ["music_generate"]},
            "model": {"creative": {"core": "none", "user": "write"}},
        }
        errors = checker.mismatches(FRAGMENT, live)
        self.assertTrue(any("principals.telegram:user:100000101" in error for error in errors))
        self.assertTrue(any("guest_profiles.creative" in error for error in errors))
        self.assertTrue(any("model.creative.user" in error for error in errors))
        self.assertTrue(any("model.creative.opencode" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
