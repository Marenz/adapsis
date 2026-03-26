use std::collections::HashMap;
use std::sync::Arc;

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

    /// Freeze this interner into a shared, cheaply-cloneable `SharedInterner`.
    /// The returned handle wraps the data in `Arc` so cloning is O(1).
    pub fn shared(&self) -> SharedInterner {
        SharedInterner {
            inner: Arc::new(self.clone()),
        }
    }
}

/// A cheaply-cloneable interner handle backed by `Arc<StringInterner>`.
///
/// Cloning a `SharedInterner` is an O(1) reference-count bump — no HashMap or
/// Vec allocation — making it suitable for passing into every `Env` created
/// during function calls without copying the entire interner.
///
/// Read-only lookups (`get`, `resolve`) go straight through the shared data.
/// If a truly new name needs to be interned at runtime (rare when the Program
/// interner pre-seeds all AST names), call `intern()` which performs
/// copy-on-write: the first mutation clones the inner `StringInterner` so that
/// other holders of the same `Arc` are unaffected.
#[derive(Debug, Clone)]
pub struct SharedInterner {
    inner: Arc<StringInterner>,
}

impl SharedInterner {
    /// Create a new empty shared interner.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(StringInterner::new()),
        }
    }

    /// Look up the id for a string without interning it.
    /// Returns `None` if the string has not been interned.
    #[inline]
    pub fn get(&self, s: &str) -> Option<InternedId> {
        self.inner.get(s)
    }

    /// Look up the string for a given interned id.
    #[inline]
    pub fn resolve(&self, id: InternedId) -> Option<&str> {
        self.inner.resolve(id)
    }

    /// Intern a new string, using copy-on-write semantics.
    /// If the string already exists, returns its id with no allocation.
    /// If it's truly new, clones the inner interner (once) and inserts.
    #[inline]
    pub fn intern(&mut self, s: &str) -> InternedId {
        // Fast path: already interned — no write needed
        if let Some(id) = self.inner.get(s) {
            return id;
        }
        // Slow path: copy-on-write — clone inner if shared, then insert
        Arc::make_mut(&mut self.inner).intern(s)
    }

    /// The number of interned strings.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the interner is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

impl Default for SharedInterner {
    fn default() -> Self {
        Self::new()
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

    #[test]
    fn test_clone_preserves_ids() {
        // Cloning an interner should preserve all interned ids so that
        // a cloned interner can be used as a seed for a new context.
        let mut original = StringInterner::new();
        let id_x = original.intern("x");
        let id_y = original.intern("y");
        let id_z = original.intern("z");

        let mut cloned = original.clone();
        // Cloned interner should resolve the same ids
        assert_eq!(cloned.resolve(id_x), Some("x"));
        assert_eq!(cloned.resolve(id_y), Some("y"));
        assert_eq!(cloned.resolve(id_z), Some("z"));

        // Interning the same strings in the clone should return the same ids
        assert_eq!(cloned.intern("x"), id_x);
        assert_eq!(cloned.intern("y"), id_y);

        // Interning a new string in the clone should not affect the original
        let id_w = cloned.intern("w");
        assert!(
            original.get("w").is_none(),
            "original should not see clone's new entries"
        );
        assert_eq!(cloned.resolve(id_w), Some("w"));
    }

    #[test]
    fn test_clone_then_extend() {
        // After cloning, new strings get sequential ids that don't collide
        let mut base = StringInterner::new();
        base.intern("a"); // id 0
        base.intern("b"); // id 1

        let mut extended = base.clone();
        let id_c = extended.intern("c"); // should be id 2
        assert_eq!(id_c, 2);
        assert_eq!(extended.len(), 3);
        assert_eq!(base.len(), 2); // base unchanged
    }

    // --- SharedInterner tests ---

    #[test]
    fn test_shared_interner_basic() {
        let mut base = StringInterner::new();
        base.intern("hello");
        base.intern("world");

        let shared = base.shared();
        assert_eq!(shared.get("hello"), Some(0));
        assert_eq!(shared.get("world"), Some(1));
        assert_eq!(shared.get("missing"), None);
        assert_eq!(shared.resolve(0), Some("hello"));
        assert_eq!(shared.resolve(1), Some("world"));
        assert_eq!(shared.len(), 2);
    }

    #[test]
    fn test_shared_interner_clone_is_cheap() {
        let mut base = StringInterner::new();
        for i in 0..100 {
            base.intern(&format!("var_{i}"));
        }
        let shared1 = base.shared();
        let shared2 = shared1.clone(); // Should be O(1) Arc clone

        // Both see the same data
        assert_eq!(shared1.get("var_0"), shared2.get("var_0"));
        assert_eq!(shared1.get("var_99"), shared2.get("var_99"));
        assert_eq!(shared1.len(), shared2.len());
    }

    #[test]
    fn test_shared_interner_copy_on_write() {
        let mut base = StringInterner::new();
        let id_a = base.intern("a");
        let id_b = base.intern("b");

        let shared_original = base.shared();
        let mut shared_clone = shared_original.clone();

        // Intern existing name — no copy-on-write needed
        assert_eq!(shared_clone.intern("a"), id_a);

        // Intern new name — triggers copy-on-write
        let id_c = shared_clone.intern("c");
        assert_eq!(id_c, 2);
        assert_eq!(shared_clone.resolve(id_c), Some("c"));
        assert_eq!(shared_clone.len(), 3);

        // Original is unaffected
        assert_eq!(shared_original.get("c"), None);
        assert_eq!(shared_original.len(), 2);
        // But existing ids still work in both
        assert_eq!(shared_original.resolve(id_a), Some("a"));
        assert_eq!(shared_original.resolve(id_b), Some("b"));
    }

    #[test]
    fn test_shared_interner_default() {
        let shared = SharedInterner::default();
        assert!(shared.is_empty());
        assert_eq!(shared.len(), 0);
    }
}
