//! String index helpers for the builtins. Adapsis string indices are BYTE offsets
//! (`len`, `index_of`), so every builtin that slices must stay on character
//! boundaries: slicing inside a multi-byte character panics in Rust (#134).

/// The largest character boundary of `text` that is not above `index`
/// (`index` itself when it is one; `text.len()` when it is past the end).
pub(super) fn floor_char_boundary(text: &str, index: usize) -> usize {
    let mut index = index.min(text.len());
    while !text.is_char_boundary(index) {
        index -= 1;
    }
    index
}

#[cfg(test)]
mod tests {
    use super::super::{eval_builtin_string, Value};
    use super::*;

    fn call(name: &str, args: Vec<Value>) -> anyhow::Result<String> {
        match eval_builtin_string(name, args)? {
            Value::String(text) => Ok(text.to_string()),
            other => panic!("expected a String, got {other:?}"),
        }
    }

    fn substring(text: &str, start: i64, end: i64) -> anyhow::Result<String> {
        call("substring", vec![Value::string(text), Value::Int(start), Value::Int(end)])
    }

    #[test]
    fn floor_char_boundary_never_lands_inside_a_character() {
        let text = "aä€😀";
        assert_eq!((0..=12).map(|i| floor_char_boundary(text, i)).collect::<Vec<_>>(), vec![0, 1, 1, 3, 3, 3, 6, 6, 6, 6, 10, 10, 10]);
        assert_eq!(floor_char_boundary("", 5), 0);
    }

    /// #134: every byte range of a multi-byte string is answered without a panic,
    /// and cutting at any offset never loses or repeats a character.
    #[test]
    fn substring_cuts_on_character_boundaries_instead_of_panicking() {
        let text = "Grüße 😀 – naïve café €5";
        let len = text.len() as i64;
        for cut in 0..=len {
            let (head, tail) = (substring(text, 0, cut).unwrap(), substring(text, cut, len).unwrap());
            assert_eq!(format!("{head}{tail}"), text, "cut at byte {cut}");
            for end in cut..=len + 3 {
                let piece = substring(text, cut, end).unwrap();
                assert!(text.contains(&piece), "{cut}..{end}");
            }
        }
        // Offsets on boundaries behave exactly as before.
        assert_eq!(substring("hello", 1, 3).unwrap(), "el");
        assert_eq!(substring("hi", 0, 99).unwrap(), "hi");
        assert_eq!(substring("Grüße", 2, 4).unwrap(), "ü");
        // Inside "ü" (bytes 2..4): both ends move down to byte 2.
        assert_eq!(substring("Grüße", 3, 5).unwrap(), "üß"[..2].to_string());
        assert_eq!(substring("Grüße", 0, 3).unwrap(), "Gr");
        assert_eq!(substring("Grüße", 3, 3).unwrap(), "");
        assert_eq!(call("substr", vec![Value::string("😀"), Value::Int(1), Value::Int(3)]).unwrap(), "");
        // Invalid ranges are still errors, never panics.
        for (start, end) in [(4, 2), (99, 100), (-1, 2)] {
            let error = substring("Grüße", start, end).unwrap_err().to_string();
            assert!(error.contains("substring: invalid range"), "{error}");
        }
        assert!(substring("x", 0, 0).is_ok());
    }

    #[test]
    fn char_at_returns_the_whole_character_containing_the_byte() {
        let at = |text: &str, index: i64| call("char_at", vec![Value::string(text), Value::Int(index)]);
        assert_eq!(at("hello", 1).unwrap(), "e");
        // "aä€😀": a=0, ä=1..3, €=3..6, 😀=6..10. Formerly every one of these
        // panicked except index 0.
        let text = "aä€😀";
        let expected = ["a", "ä", "ä", "€", "€", "€", "😀", "😀", "😀", "😀"];
        for (index, character) in expected.iter().enumerate() {
            assert_eq!(at(text, index as i64).unwrap(), *character, "byte {index}");
        }
        for index in [10, 11, -1] {
            assert!(at(text, index).unwrap_err().to_string().contains("out of bounds"), "{index}");
        }
    }
}
