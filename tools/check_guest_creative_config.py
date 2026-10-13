#!/usr/bin/env python3
"""Verify that reviewed creative-profile TOML entries exist unchanged in a live config."""

from pathlib import Path
import sys
import tomllib


REVIEWED_TABLES = (("principals",), ("guest_profiles",), ("model", "creative"))


def _table(document, path):
    value = document
    for key in path:
        if not isinstance(value, dict) or key not in value:
            return None
        value = value[key]
    return value if isinstance(value, dict) else None


def mismatches(fragment, live):
    errors = []
    for path in REVIEWED_TABLES:
        expected = _table(fragment, path)
        actual = _table(live, path)
        label = ".".join(path)
        if expected is None:
            errors.append(f"reviewed fragment is missing table [{label}]")
            continue
        if actual is None:
            errors.append(f"live config is missing table [{label}]")
            continue
        for key, value in expected.items():
            if key not in actual:
                errors.append(f"live config is missing {label}.{key}")
            elif actual[key] != value:
                errors.append(
                    f"live config value differs for {label}.{key}: "
                    f"expected {value!r}, got {actual[key]!r}"
                )
    return errors


def load(path):
    with Path(path).open("rb") as source:
        return tomllib.load(source)


def main(argv):
    if len(argv) != 3:
        print(
            "usage: check_guest_creative_config.py REVIEWED_FRAGMENT LIVE_CONFIG",
            file=sys.stderr,
        )
        return 2
    try:
        errors = mismatches(load(argv[1]), load(argv[2]))
    except (OSError, tomllib.TOMLDecodeError) as error:
        print(f"creative config check failed: {error}", file=sys.stderr)
        return 2
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1
    print("reviewed creative configuration matches live config")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
