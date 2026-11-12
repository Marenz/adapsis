//! #96, the symptom itself: code and data of one JIT module more than 2 GiB
//! apart, arranged for real in a child process.
//!
//! No global allocator is involved. The child uses two facts about glibc
//! malloc and the kernel:
//!
//! * with a fixed mmap threshold, a multi-megabyte allocation (the read-only
//!   block of a large string literal) is its own `mmap`, while a one-page
//!   allocation (the code block) comes from the thread's arena;
//! * `mmap` takes the first free range in its search order that fits, so after
//!   every nearer range that fits has been reserved, the literal's mapping
//!   lands far from the arena.
//!
//! The parent runs the child in two modes and requires both outcomes:
//!
//! * `absolute` (what the compiler emits now): the module compiles, the two
//!   blocks are verified to be more than 2 GiB apart, and the function returns
//!   the right string.
//! * `colocated` (the relocations emitted before the fix, through the same
//!   dispatch path as production): Cranelift panics in `compiled_blob.rs`,
//!   the panic is contained, and the call returns the VM result.
//!
//! The second mode is what shows the arrangement reproduces the defect; the
//! first is the regression test. Linux/glibc/x86-64 only: elsewhere the
//! layout assumptions do not hold and the tests do nothing.
#![cfg(all(target_os = "linux", target_env = "gnu", target_arch = "x86_64"))]

use super::admission_measurement_tests::program;
use crate::{ast, eval, parser};

const MODE_VARIABLE: &str = "ADAPSIS_TEST_JIT_FAR_APART";
const CHILD_TEST: &str = "compiler::far_apart_tests::jit_far_apart_child";
const GIB: usize = 1 << 30;
const LITERAL_BYTES: usize = 4 * 1024 * 1024;
/// A displacement of this size no longer fits a signed 32-bit relocation.
const REL32_REACH: usize = 2 * GIB;
/// Mappings closer than this to the arena are reserved away.
const NEAR: usize = 6 * GIB;

impl super::CompiledProgram {
    fn function_address(&self, name: &str) -> usize {
        self.module.get_finalized_function(self.functions[name]) as usize
    }

    /// Address of the string a zero-argument function returns.
    fn returned_string_address(&self, name: &str) -> usize {
        #[repr(C)]
        struct StringReturn {
            ptr: i64,
            len: i64,
        }
        // SAFETY: the function was compiled from `()->String`, which uses this
        // signature (see `call_string`).
        let call: fn() -> StringReturn = unsafe { std::mem::transmute(self.module.get_finalized_function(self.functions[name])) };
        call().ptr as usize
    }
}

fn literal() -> String {
    let mut text = "x".repeat(LITERAL_BYTES - 4);
    text.push_str("-END");
    text
}

fn big_literal_program() -> ast::Program {
    program(&format!("+fn big ()->String\n  +return \"{}\"\n+end\n", literal()))
}

fn reserve(bytes: usize) -> usize {
    // SAFETY: an anonymous PROT_NONE mapping at an address the kernel
    // chooses; it is never read or written.
    let address = unsafe {
        libc::mmap(std::ptr::null_mut(), bytes, libc::PROT_NONE, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_NORESERVE, -1, 0)
    };
    assert_ne!(address, libc::MAP_FAILED, "address-space reservation failed");
    address as usize
}

fn release(address: usize, bytes: usize) {
    // SAFETY: exactly a mapping made by `reserve`.
    unsafe { libc::munmap(address as *mut libc::c_void, bytes) };
}

/// Make the next `mmap` of about `bytes` land more than `NEAR` away from
/// `anchor`, by reserving (and leaking) every nearer range the kernel would
/// choose first. Returns the number of reservations.
fn push_mappings_away_from(anchor: usize, bytes: usize) -> usize {
    let mut reservations = 0;
    // Take the large free space next to the existing mappings in one piece.
    for _ in 0..8 {
        let block = reserve(2 * NEAR);
        if block.abs_diff(anchor) > 3 * NEAR {
            release(block, 2 * NEAR);
            break;
        }
        reservations += 1;
    }
    // Then every remaining nearer gap that could hold the allocation, also a
    // little smaller than it, so no gap of an in-between size is left.
    for probe in [bytes / 2, bytes + 16 * 4096, 2 * bytes] {
        loop {
            let block = reserve(probe);
            if block.abs_diff(anchor) > NEAR {
                release(block, probe);
                break;
            }
            reservations += 1;
            assert!(reservations < 100_000, "the address space near the arena does not fill up");
        }
    }
    reservations
}

/// Where a one-page, page-aligned allocation of this thread lands now: the
/// same request Cranelift makes for a code block.
fn arena_address() -> usize {
    let layout = std::alloc::Layout::from_size_align(4096, 4096).unwrap();
    // SAFETY: a non-zero layout; the block is freed again with the same layout.
    unsafe {
        let block = std::alloc::alloc(layout);
        assert!(!block.is_null());
        std::alloc::dealloc(block, layout);
        block as usize
    }
}

/// Runs only as the child of the two tests below.
#[test]
fn jit_far_apart_child() {
    let Ok(mode) = std::env::var(MODE_VARIABLE) else { return };
    let program = big_literal_program();
    // SAFETY: plain libc call. A fixed threshold also switches off glibc's
    // dynamic threshold, so the literal's block is certainly its own mapping.
    assert_eq!(unsafe { libc::mallopt(libc::M_MMAP_THRESHOLD, 128 * 1024) }, 1);
    let anchor = arena_address();
    let reservations = push_mappings_away_from(anchor, LITERAL_BYTES);
    eprintln!("FAR_APART arena={anchor:#x} reservations={reservations}");
    match mode.as_str() {
        "absolute" => {
            let mut compiled = super::compile(&program).expect("the module compiles whatever the distance");
            let (code, data) = (compiled.function_address("big"), compiled.returned_string_address("big"));
            let distance = code.abs_diff(data);
            assert!(distance > REL32_REACH, "blocks are only {distance} bytes apart (code {code:#x}, data {data:#x}); the arrangement failed");
            assert_eq!(compiled.call_string("big", &[]).unwrap(), literal());
            println!("FAR_APART_OK absolute distance_gib={}", distance / GIB);
        }
        "colocated" => {
            // The relocations emitted before the fix, unchecked, through the
            // production dispatch path.
            super::relocation::COLOCATED_FOR_TESTS.with(|flag| flag.set(true));
            let cache = eval::new_jit_cache();
            let _seam = eval::jit_seam::hook(|| {});
            let input = parser::parse_single_expr("{}").unwrap();
            for _ in 0..2 {
                let (value, compiled) = eval::eval_compiled_or_interpreted_cached(&program, "big", &input, Some(&cache), 1).unwrap();
                assert!(!compiled, "the colocated module must not have been usable at this distance");
                assert_eq!(value, format!("\"{}\"", literal()), "the VM result");
            }
            assert_eq!((eval::jit_seam::attempts(), eval::jit_seam::journaled()), (1, 1), "one contained panic, remembered");
            assert!(matches!(&*cache.lock().unwrap(), Some((1, slot)) if slot.is_refused()));
            println!("FAR_APART_OK colocated contained");
        }
        other => panic!("unknown mode {other}"),
    }
}

/// Longest a child may take. It normally finishes in about 0.1 s; the bound
/// only has to tell "slow machine" from "hung".
const CHILD_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(120);

/// Run the child with a deadline. A child that does not finish is killed and
/// the test fails with a message that says so; it cannot hang the suite.
fn run_child(mode: &str) -> (bool, String, String) {
    run_child_with(mode, CHILD_TEST, CHILD_TIMEOUT)
}

fn run_child_with(mode: &str, test: &str, timeout: std::time::Duration) -> (bool, String, String) {
    use std::io::Read;
    use std::process::Stdio;
    let mut child = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture", "--test-threads=1"])
        .env(MODE_VARIABLE, mode)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    // Drain both pipes on their own threads so a full pipe cannot block the child.
    let drain = |mut pipe: Box<dyn Read + Send>| {
        std::thread::spawn(move || {
            let mut bytes = Vec::new();
            let _ = pipe.read_to_end(&mut bytes);
            String::from_utf8_lossy(&bytes).into_owned()
        })
    };
    let stdout = drain(Box::new(child.stdout.take().unwrap()));
    let stderr = drain(Box::new(child.stderr.take().unwrap()));
    let deadline = std::time::Instant::now() + timeout;
    let status = loop {
        match child.try_wait().unwrap() {
            Some(status) => break Some(status),
            None if std::time::Instant::now() >= deadline => break None,
            None => std::thread::sleep(std::time::Duration::from_millis(5)),
        }
    };
    let Some(status) = status else {
        let _ = child.kill();
        let _ = child.wait();
        let (stdout, stderr) = (stdout.join().unwrap(), stderr.join().unwrap());
        panic!(
            "the far-apart child `{mode}` did not finish within {timeout:?} and was killed\nstdout:\n{}\nstderr:\n{}",
            tail(&stdout),
            tail(&stderr)
        );
    };
    (status.success(), stdout.join().unwrap(), stderr.join().unwrap())
}

/// Runs only as a child of `jit_far_apart_child_runner_kills_a_hung_child`.
#[test]
fn jit_far_apart_hanging_child() {
    if std::env::var(MODE_VARIABLE).as_deref() == Ok("hang") {
        println!("HANGING_CHILD_STARTED");
        std::thread::sleep(std::time::Duration::from_secs(600));
    }
}

#[test]
fn jit_far_apart_child_runner_kills_a_hung_child() {
    let started = std::time::Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        run_child_with("hang", "compiler::far_apart_tests::jit_far_apart_hanging_child", std::time::Duration::from_millis(1500))
    });
    let message = match outcome {
        Ok(_) => panic!("a child that sleeps for ten minutes was reported as finished"),
        Err(payload) => payload.downcast_ref::<String>().cloned().unwrap_or_default(),
    };
    assert!(started.elapsed() < std::time::Duration::from_secs(60), "took {:?}", started.elapsed());
    assert!(message.contains("did not finish within 1.5s and was killed"), "{message}");
    assert!(message.contains("HANGING_CHILD_STARTED"), "the child's output is part of the failure: {message}");
}

fn tail(text: &str) -> String {
    let lines: Vec<&str> = text.lines().filter(|line| line.len() < 600).collect();
    lines[lines.len().saturating_sub(12)..].join("\n")
}

#[test]
fn jit_blocks_more_than_2gib_apart_compile_and_run() {
    let (success, stdout, stderr) = run_child("absolute");
    assert!(success && stdout.contains("FAR_APART_OK absolute"), "stdout:\n{}\nstderr:\n{}", tail(&stdout), tail(&stderr));
    assert!(!stderr.contains("compiled_blob.rs"), "Cranelift's relocation panic fired:\n{}", tail(&stderr));
}

#[test]
fn jit_relocation_panic_of_the_old_codegen_is_reproduced_and_contained() {
    let (success, stdout, stderr) = run_child("colocated");
    assert!(success && stdout.contains("FAR_APART_OK colocated contained"), "stdout:\n{}\nstderr:\n{}", tail(&stdout), tail(&stderr));
    // The genuine Cranelift panic happened (and was contained): this is what
    // proves that the child's memory arrangement reproduces #96.
    assert!(stderr.contains("compiled_blob.rs") && stderr.contains("TryFromIntError"), "the relocation panic did not fire:\n{}", tail(&stderr));
}
