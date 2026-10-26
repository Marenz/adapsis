//! Task 5: Python test-matcher lowering roundtrips to the identical
//! TestCase/TestMatcher/AfterCheck AST.
use super::*;

#[test]
fn test_matchers_roundtrip_to_adapsis() {
    let source = "def test_f():\n    assert contains(f('x'), 'sub')\n    assert_after('routes', 'contains', '/chat')\n\ndef test_g():\n    assert starts_with(g('x'), 'pre')\n\ndef test_h():\n    assert is_ok(h(1))\n\ndef test_i():\n    assert is_err(i(1))\n\ndef test_j():\n    assert j(1) == Err('oops')\n\ndef test_k():\n    assert k(1) == Err(err_label)\n";
    let lowered = lower_tests(source, "Example").unwrap();
    let canonical = lowered.canonical_adapsis;
    assert!(canonical.contains("expect contains(\"sub\")"), "{canonical}");
    assert!(canonical.contains("+after routes contains \"/chat\""), "{canonical}");
    assert!(canonical.contains("expect starts_with(\"pre\")"), "{canonical}");
    assert!(canonical.contains("expect Ok"), "{canonical}");
    assert!(canonical.contains("expect Err\n"), "{canonical}");
    assert!(canonical.contains("expect Err(\"oops\")"), "{canonical}");
    assert!(canonical.contains("expect Err(err_label)"), "{canonical}");
}

#[test]
fn test_matchers_reject_bad_forms() {
    for (source, fragment) in [
        ("def test_f():\n    assert f(1) > 1\n", "one == comparison"),
        (
            "def test_f():\n    assert f(1) == 1\n    assert_after('routes')\n",
            "3 string arguments",
        ),
        (
            "def test_f():\n    assert is_ok(1, 2)\n",
            "1 argument",
        ),
    ] {
        let error = lower_tests(source, "Example").unwrap_err();
        assert!(error.message.contains(fragment), "{fragment:?}: {error}");
    }
}
