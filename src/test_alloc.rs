//! Test-only counting allocator: peak heap use of ONE thread during a closure.
//!
//! Used by the `run` hardening tests to assert that a builtin called with hostile
//! arguments never allocates more than a stated multiple of the active value cap
//! (#129, review finding F1). Counting is off unless a test turns it on for its thread,
//! so every other test pays two thread-local reads per allocation.
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;

pub struct Counting;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static CURRENT: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

fn add(delta: isize) {
    // `try_with`: the allocator also runs while a thread's locals are being torn down.
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = CURRENT.try_with(|current| {
                let now = current.get() + delta;
                current.set(now);
                let _ = PEAK.try_with(|peak| peak.set(peak.get().max(now)));
            });
        }
    });
}

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as isize);
        unsafe { System.alloc(layout) }
    }
    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        add(-(layout.size() as isize));
        unsafe { System.dealloc(pointer, layout) }
    }
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        add(layout.size() as isize);
        unsafe { System.alloc_zeroed(layout) }
    }
    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        add(new_size as isize - layout.size() as isize);
        unsafe { System.realloc(pointer, layout, new_size) }
    }
}

/// Run `work` on this thread and return its result with the peak number of bytes that
/// were allocated on this thread, above the level at entry, while it ran.
pub fn peak<T>(work: impl FnOnce() -> T) -> (T, usize) {
    CURRENT.with(|current| current.set(0));
    PEAK.with(|peak| peak.set(0));
    ON.with(|on| on.set(true));
    let result = work();
    ON.with(|on| on.set(false));
    (result, PEAK.with(Cell::get).max(0) as usize)
}

#[test]
fn peak_counts_this_threads_allocations_and_frees() {
    let (kept, bytes) = peak(|| {
        let transient = vec![0u8; 3_000_000];
        drop(transient);
        vec![0u8; 1_000_000]
    });
    assert_eq!(kept.len(), 1_000_000);
    assert!((3_000_000..3_100_000).contains(&bytes), "{bytes}");
    let (_, bytes) = peak(|| 1 + 1);
    assert_eq!(bytes, 0);
}
