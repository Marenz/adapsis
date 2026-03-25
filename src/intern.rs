use std::collections::HashMap;

/// Interned string identifier. A `u32` index into the `StringInterner` table.
/// Using `u32` instead of `usize` saves 4 bytes per key on 64-bit platforms
/// and the 4-billion limit is more than sufficient for variable names.
pub type InternedId = u32;

/// A fast string interner that maps strings to compact `u32` identifiers.
///
/// Variable lookup in the evaluator's scope stack is one of the hottest paths
/// in the interpreter. By interning variable names we replace `HashMap<String, Value>`
/// with `HashMap<u32, Value>`, turning every lookup from a string hash + comparison
/// into a single integer hash + comparison.
///
/// The interner is append-only: strings are never removed, which is fine for
/// a session-scoped interpreter where variable names are a bounded set.
#[derive(Debug, Clone)]
pub struct StringInterner {
    /// Maps string → id for O(1) dedup during interning.
    map: HashMap<String, InternedId>,
    /// Maps id → string for reverse lookups (error messages, debug display).
    strings: Vec<String>,
}

impl Default for StringInterner {
    fn default() -> Self {
        Self::new()
    }
}

impl StringInterner {
    /// Create a new empty interner.
    pub fn new() -> Self {
        Self {
            map: HashMap::new(),
            strings: Vec::new(),
        }
    }

    /// Create a new interner with pre-allocated capacity.
    pub fn with_capacity(cap: usize) -> Self {
        Self {
            map: HashMap::with_capacity(cap),
            strings: Vec::with_capacity(cap),
        }
    }

    /// Intern a string, returning its unique id.
    /// If the string was already interned, returns the existing id.
    /// If new, assigns the next sequential id.
    pub fn intern(&mut self, s: &str) -> InternedId {
        if let Some(&id) = self.map.get(s) {
            return id;
        }
        let id = self.strings.len() as InternedId;
        self.strings.push(s.to_string());
        self.map.insert(s.to_string(), id);
        id
    }

    /// Look up the string for a given interned id.
    /// Returns `None` if the id is out of range.
    pub fn resolve(&self, id: InternedId) -> Option<&str> {
        self.strings.get(id as usize).map(|s| s.as_str())
    }

    /// Look up the id for a string without interning it.
    /// Returns `None` if the string has not been interned.
    pub fn get(&self, s: &str) -> Option<InternedId> {
        self.map.get(s).copied()
    }

    /// The number of interned strings.
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.strings.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_intern_basic() {
        let mut interner = StringInterner::new();
        let id_a = interner.intern("hello");
        let id_b = interner.intern("world");
        let id_a2 = interner.intern("hello");

        // Same string returns same id
        assert_eq!(id_a, id_a2);
        // Different strings get different ids
        assert_ne!(id_a, id_b);
        // Resolve works
        assert_eq!(interner.resolve(id_a), Some("hello"));
        assert_eq!(interner.resolve(id_b), Some("world"));
    }

    #[test]
    fn test_intern_sequential_ids() {
        let mut interner = StringInterner::new();
        assert_eq!(interner.intern("a"), 0);
        assert_eq!(interner.intern("b"), 1);
        assert_eq!(interner.intern("c"), 2);
        // Re-interning returns existing id
        assert_eq!(interner.intern("a"), 0);
        assert_eq!(interner.len(), 3);
    }

    #[test]
    fn test_resolve_out_of_range() {
        let interner = StringInterner::new();
        assert_eq!(interner.resolve(0), None);
        assert_eq!(interner.resolve(999), None);
    }

    #[test]
    fn test_get_without_interning() {
        let mut interner = StringInterner::new();
        assert_eq!(interner.get("x"), None);
        let id = interner.intern("x");
        assert_eq!(interner.get("x"), Some(id));
        // Unknown string still returns None
        assert_eq!(interner.get("y"), None);
    }

    #[test]
    fn test_empty_string() {
        let mut interner = StringInterner::new();
        let id = interner.intern("");
        assert_eq!(interner.resolve(id), Some(""));
        // Re-intern returns same id
        assert_eq!(interner.intern(""), id);
    }

    #[test]
    fn test_with_capacity() {
        let interner = StringInterner::with_capacity(100);
        assert!(interner.is_empty());
        assert_eq!(interner.len(), 0);
    }

    #[test]
    fn test_many_strings() {
        let mut interner = StringInterner::new();
        let mut ids = Vec::new();
        for i in 0..1000 {
            let s = format!("var_{i}");
            ids.push(interner.intern(&s));
        }
        assert_eq!(interner.len(), 1000);
        // Verify all resolve correctly
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(interner.resolve(*id), Some(format!("var_{i}").as_str()));
        }
        // Re-interning all returns same ids
        for (i, expected_id) in ids.iter().enumerate() {
            let s = format!("var_{i}");
            assert_eq!(interner.intern(&s), *expected_id);
        }
        // Length unchanged after re-interning
        assert_eq!(interner.len(), 1000);
    }
}
