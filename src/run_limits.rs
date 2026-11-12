//! Resource limits for native `run` blocks (#129).
//!
//! **These numbers are policy, not measurements.** They were decided by the owner on
//! 2026-11-10 and live here, in one place. Enforcement code (`eval/meter.rs`, the native
//! turn loop) reads them only through [`RunLimits`], so a per-profile override from
//! `permissions.toml` can later construct a different `RunLimits` without touching it.
use std::time::Duration;

const MIB: usize = 1024 * 1024;

/// Restricted speakers (guests, execute-only): statements, loop iterations and calls.
const RESTRICTED_STEPS: u64 = 200_000;
/// Restricted speakers: deep size of any single value.
const RESTRICTED_VALUE_BYTES: usize = MIB;
/// Restricted speakers: bytes produced over the whole block (not live memory).
const RESTRICTED_PRODUCED_BYTES: usize = 32 * MIB;
/// Restricted speakers: IO/tool calls per block, calls made by called functions included.
const RESTRICTED_IO_CALLS: usize = 8;
/// Restricted speakers: calls per block of EACH tool in [`COSTLY_TOOLS`].
const RESTRICTED_COSTLY_CALLS: usize = 1;
/// Restricted speakers: wall clock per block. Sized so eight web fetches of a few seconds
/// each do not trip it.
const RESTRICTED_WALL_CLOCK: Duration = Duration::from_secs(60);
/// Restricted speakers: turn-bound tool calls per turn, direct calls and calls inside
/// blocks together. Legacy `<code>` turns have no such bound; this one is new.
const RESTRICTED_TURN_TOOL_CALLS: usize = 24;
/// Restricted speakers: wall clock of all blocks of one TURN together. Pure blocks draw
/// nothing from the tool-call budget, so without this a turn could hold a blocking thread
/// for (rounds x 60 s).
const RESTRICTED_TURN_BLOCK_TIME: Duration = Duration::from_secs(180);
/// Owner: deep size of any single value. The only owner limit; it stops the doubling
/// loop and the list of large values before they take the process down.
const OWNER_VALUE_BYTES: usize = 64 * MIB;

/// Tools that start expensive work (GPU jobs, an off-host CAD author, video encoding).
pub const COSTLY_TOOLS: &[&str] = &["music_generate", "cad_prepare", "video_stabilize"];

/// One step is charged per this many bytes of builtin arguments, on top of the step for
/// the call itself, so the step budget bounds work and not only the number of calls: a
/// linear builtin over a 1 MiB value costs 257 steps, so a restricted block gets about
/// 780 of them.
pub const ARGUMENT_BYTES_PER_STEP: usize = 4096;

/// Regular expressions in restricted blocks (review finding F3).
///
/// MEASURED on this machine (RTX-3090 desktop, release build, 2026-11-10): when the lazy
/// DFA gives up, a search costs 1.1e-10 s per (compiled-program byte x text byte); the
/// worst family found is `(?:a?){n}a{n}[^a]` on a text of `a`s (n=1000: 262,144-byte
/// program class, 131,072-byte text, 3.05 s; n=10000 on 512 KiB: 123 s). Patterns the lazy
/// DFA can run are 100 to 10,000 times cheaper. The compiled size is known only as a
/// class: the smallest power-of-two `size_limit` under which the pattern compiles.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RegexLimits {
    /// Replacement searches can copy capture slots for active program states, even
    /// through find_iter. Bound both scratch and work before the FIRST search:
    /// <=8 explicit groups, class * (2 * captures_len) <=512 KiB, and the same
    /// slot-weighted class * text work limit. This is conservative admission policy,
    /// not an exact prediction of engine allocations. Measured counted-optionals at
    /// the admission boundary: <=2.1 MiB peak (including output), <=0.33 s; the
    /// 200/400-group attack is rejected before searching. See bounded_delta_tests.
    pub capture_groups: usize,
    pub capture_state_bytes: usize,
    /// Maximum of (program class bytes) x (text bytes) for one call. 8 GiB^2 is about
    /// 0.95 s at the measured worst-case rate. Program classes for reference: ASCII
    /// classes 1 to 8 KiB, `\d+` 8 KiB, `\w+` 64 KiB, `\w+@\w+\.\w+` 256 KiB; so a
    /// Unicode `\w` pattern may run on about 128 KiB of text and an ASCII one on 1 MiB.
    pub work: u64,
    /// Largest program class tried at all (compiling costs time and memory too).
    pub max_program: usize,
    /// Longest pattern text. Parsing a pattern allocates about 300 bytes per pattern
    /// byte before any size limit applies (MEASURED: a 1 MiB pattern, 300 MiB, 2.5 s).
    pub max_pattern: usize,
    /// Lazy-DFA cache per regex (the regex crate's default is 2 MiB).
    pub dfa_cache: usize,
    /// A `regex_replace` call is stopped between matches after this long: iterating all
    /// matches is worst-case quadratic in the text even when every search is bounded.
    pub call_time: Duration,
    /// Steps charged per second of regex work (worst-case model, or measured time if
    /// that is longer), so repeated calls drain the step budget: 200,000 steps allow
    /// about four seconds of regex work per block.
    pub steps_per_second: u64,
}

const RESTRICTED_REGEX: RegexLimits = RegexLimits {
    capture_groups: 8,
    capture_state_bytes: 512 * 1024,
    work: 8 << 30,
    max_program: 512 * 1024,
    max_pattern: 2048,
    dfa_cache: 512 * 1024,
    call_time: Duration::from_secs(1),
    steps_per_second: 50_000,
};

/// The interpreter's evaluation stack for `run` blocks and async `evaluate` (review
/// finding F4). MEASURED (release build, `eval/stack_tests.rs`): one level of
/// user-function recursion costs 10.2 KiB of stack through a plain call (9.5 KiB through
/// an await) and 19.7 KiB through nested `each`/`match`/`if`/`while` bodies; the
/// call-depth limit is 256, so 256 levels need 2.5 to 4.9 MiB. A 2 MiB worker stack
/// overflowed at about depth 100 to 200. The stack is virtual memory; only pages
/// that are touched are committed.
pub const EVAL_STACK_BYTES: usize = 64 * MIB;
/// Any thread: a call is refused cleanly when less than this much stack is left.
pub const STACK_RESERVE_BYTES: usize = 256 * 1024;

/// Structural, not policy: values nested deeper than this are refused for every class,
/// because rendering and dropping a value recurse once per level.
pub const MAX_VALUE_DEPTH: usize = 64;

/// What one `run` block may consume. `None` means unlimited.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RunLimits {
    pub steps: Option<u64>,
    pub value_bytes: usize,
    pub produced_bytes: Option<usize>,
    pub io_calls: Option<usize>,
    pub costly_calls: Option<usize>,
    pub wall_clock: Option<Duration>,
    pub turn_tool_calls: Option<usize>,
    /// Wall clock of all blocks of one turn together.
    pub turn_block_time: Option<Duration>,
    /// `None`: no regex work limit (the output cap still applies).
    pub regex: Option<RegexLimits>,
}

impl RunLimits {
    /// Guests and execute-only speakers: the author of the block is untrusted.
    pub const fn restricted() -> Self {
        Self {
            steps: Some(RESTRICTED_STEPS),
            value_bytes: RESTRICTED_VALUE_BYTES,
            produced_bytes: Some(RESTRICTED_PRODUCED_BYTES),
            io_calls: Some(RESTRICTED_IO_CALLS),
            costly_calls: Some(RESTRICTED_COSTLY_CALLS),
            wall_clock: Some(RESTRICTED_WALL_CLOCK),
            turn_tool_calls: Some(RESTRICTED_TURN_TOOL_CALLS),
            turn_block_time: Some(RESTRICTED_TURN_BLOCK_TIME),
            regex: Some(RESTRICTED_REGEX),
        }
    }

    /// Write-capable speakers. Steps, IO calls and wall clock belong to #131.
    pub const fn owner() -> Self {
        Self {
            steps: None,
            value_bytes: OWNER_VALUE_BYTES,
            produced_bytes: None,
            io_calls: None,
            costly_calls: None,
            wall_clock: None,
            turn_tool_calls: None,
            turn_block_time: None,
            regex: None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The decided numbers, pinned so a change is a visible diff of policy.
    #[test]
    fn the_decided_limits() {
        let restricted = RunLimits::restricted();
        assert_eq!(restricted.steps, Some(200_000));
        assert_eq!(restricted.value_bytes, 1024 * 1024);
        assert_eq!(restricted.produced_bytes, Some(32 * 1024 * 1024));
        assert_eq!(restricted.io_calls, Some(8));
        assert_eq!(restricted.costly_calls, Some(1));
        assert_eq!(restricted.wall_clock, Some(Duration::from_secs(60)));
        assert_eq!(restricted.turn_tool_calls, Some(24));
        assert_eq!(restricted.turn_block_time, Some(Duration::from_secs(180)));
        let regex = restricted.regex.unwrap();
        assert_eq!((regex.work, regex.max_program, regex.dfa_cache), (8 << 30, 512 * 1024, 512 * 1024));
        assert_eq!(regex.max_pattern, 2048);
        assert_eq!((regex.call_time, regex.steps_per_second), (Duration::from_secs(1), 50_000));
        assert_eq!((ARGUMENT_BYTES_PER_STEP, EVAL_STACK_BYTES, STACK_RESERVE_BYTES), (4096, 64 * 1024 * 1024, 256 * 1024));
        let owner = RunLimits::owner();
        assert_eq!(owner.value_bytes, 64 * 1024 * 1024);
        assert_eq!(
            (owner.steps, owner.produced_bytes, owner.io_calls, owner.costly_calls, owner.wall_clock, owner.turn_tool_calls),
            (None, None, None, None, None, None)
        );
        assert_eq!((owner.turn_block_time, owner.regex), (None, None));
        for tool in COSTLY_TOOLS {
            assert!(crate::coroutine::scoped::GUEST_TOOLS.contains(tool), "{tool}");
        }
    }
}
