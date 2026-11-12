//! The run-block meter (#129): resource limits and the restricted-speaker call policy,
//! enforced INSIDE the tree-walker for the duration of one native `run` block.
//!
//! **Cost outside a block.** The meter is thread-local and exists only between
//! [`install`] and the drop of its guard (`eval_detached_block`). Everything else the
//! interpreter runs (module code of the live bot, `evaluate`, routes, spawned tasks,
//! tests) pays one [`active`] check, a thread-local `Cell<bool>` read, at each site:
//!
//! | site | file |
//! |---|---|
//! | every statement (`meter::tick`) | `eval/mod.rs::execute_statement` |
//! | every loop iteration (`each` ×3, `while`) | `eval/mod.rs::execute_statement` |
//! | every call (`eval_builtin_or_user`) | `eval/mod.rs` |
//! | builtin `await`, `spawn`, source/event statements | `eval/mod.rs::execute_statement` |
//! | struct construction | `eval/mod.rs::eval_ast_expr` |
//! | declared-function entry (frame marker) | `eval/invocation.rs::invoke` |
//! | every IO builtin dispatch | `coroutine/mod.rs::execute_await` |
//!
//! **Bounded before allocating.** The meter admits arguments before a builtin runs and
//! measures its result afterwards, which bounds values but not what one call allocates on
//! the way. Every builtin whose output is not a small constant times its input therefore
//! computes a bound first and calls [`reserve`]/[`reserve_shape`], renders through
//! [`render`] (a writer that stops at the cap), or uses the bounded regex and JSON entry
//! points (`eval/bounded.rs`). The classification of all pure builtins is
//! `builtins/pure.rs::COSTS`; `eval/builtin_alloc_tests.rs` measures real allocations.
//!
//! **Value size without re-walking.** Every value a block produces has a bounded deep
//! size: the bytes it occupies in memory, shared parts counted each time they appear
//! (one [`SLOT`] per value, one [`HEAP`] header per string or container, plus payload). Sizes of containers are remembered per allocation ([`Sizes`]), so
//! `xs = push(xs, x)` costs one lookup for `xs` plus the size of `x`, not a walk of `xs`.
//! An entry holds a `Weak` to its allocation: the address cannot be reused while the
//! entry exists, and `Arc::make_mut` never mutates an allocation that has a `Weak`, so a
//! live entry always describes the value at that address.
use super::Value;
use crate::intern::InternedId;
use crate::run_limits::{RunLimits, COSTLY_TOOLS, MAX_VALUE_DEPTH};
use anyhow::{anyhow, Result};
use std::cell::{Cell, RefCell};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, Weak};
use std::time::{Duration, Instant};

/// Bytes of one value where it sits: a list element, a map key, a boxed payload, a
/// binding. This is the real size of the enum (176 bytes at the time of writing, because
/// a coroutine handle is stored inline), so the accounting is real memory.
pub(crate) const SLOT: usize = std::mem::size_of::<Value>();
/// Heap header of a string or container (reference counts plus the vector or string
/// header) beyond its slot.
pub(crate) const HEAP: usize = 40;

/// Builtins that return their first argument with one more element. Their result is
/// recorded as "sum of the arguments", an upper bound that needs no walk.
const GROWERS: &[&str] = &["push", "insert", "map_set", "set_add"];

thread_local! {
    static ACTIVE: Cell<bool> = const { Cell::new(false) };
    static METER: RefCell<Option<Meter>> = const { RefCell::new(None) };
}

/// The one check code outside a block pays.
#[inline(always)]
pub(crate) fn active() -> bool {
    ACTIVE.with(Cell::get)
}

/// What a restricted speaker's block may call from its OWN frame. Functions it calls are
/// owner-written code and run as they do under `evaluate`.
#[derive(Clone, Debug, Default)]
pub struct BlockPolicy {
    /// Turn-bound tools of the speaker's profile, callable as awaited builtins.
    pub tools: HashSet<String>,
    /// Qualified user functions the block was admitted to call.
    pub functions: HashSet<String>,
}

/// What one conversational turn may spend across all its tool calls: turn-bound tool
/// calls (direct calls and IO inside blocks draw from the same count) and the wall clock
/// of all its `run` blocks together.
#[derive(Debug)]
pub struct TurnBudget {
    limit: Option<usize>,
    used: AtomicUsize,
    block_time: Option<Duration>,
    block_time_used: Mutex<Duration>,
}

impl TurnBudget {
    /// A budget with a tool-call limit only.
    pub fn new(limit: Option<usize>) -> Arc<Self> {
        Arc::new(Self { limit, used: AtomicUsize::new(0), block_time: None, block_time_used: Mutex::new(Duration::ZERO) })
    }

    /// The per-turn budget of a speaker with these limits.
    pub fn for_limits(limits: &RunLimits) -> Arc<Self> {
        Arc::new(Self {
            limit: limits.turn_tool_calls,
            used: AtomicUsize::new(0),
            block_time: limits.turn_block_time,
            block_time_used: Mutex::new(Duration::ZERO),
        })
    }

    pub fn take(&self, what: &str) -> Result<(), String> {
        let Some(limit) = self.limit else { return Ok(()) };
        if self.used.fetch_add(1, Ordering::SeqCst) >= limit {
            return Err(format!(
                "`{what}` refused: this turn has used its {limit} tool calls. Answer with what you have, or ask the person to continue in a new message."
            ));
        }
        Ok(())
    }

    /// Block run time still available in this turn, with the turn's limit.
    fn block_time_left(&self) -> Option<(Duration, Duration)> {
        let limit = self.block_time?;
        let used = *self.block_time_used.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        Some((limit.saturating_sub(used), limit))
    }

    fn spend_block_time(&self, elapsed: Duration) {
        let mut used = self.block_time_used.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        *used = used.saturating_add(elapsed);
    }

    #[cfg(test)]
    pub fn used(&self) -> usize {
        self.used.load(Ordering::SeqCst)
    }

    #[cfg(test)]
    pub fn block_time_used(&self) -> Duration {
        *self.block_time_used.lock().unwrap()
    }
}

/// Everything one block runs under.
pub struct RunContext {
    pub limits: RunLimits,
    /// `None` for write-capable speakers: no call policy.
    pub policy: Option<BlockPolicy>,
    pub turn: Arc<TurnBudget>,
}

impl RunContext {
    pub fn owner() -> Self {
        Self { limits: RunLimits::owner(), policy: None, turn: TurnBudget::new(None) }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Frame {
    /// The block's own statements: the call policy applies.
    Block,
    /// A declared function the block (or another callee) invoked.
    Callee,
}

enum Held {
    List(Weak<Vec<Value>>),
    Map(Weak<Vec<(Value, Value)>>),
    Struct(Weak<HashMap<InternedId, Value>>),
}

impl Held {
    fn alive(&self) -> bool {
        match self {
            Held::List(weak) => weak.strong_count() > 0,
            Held::Map(weak) => weak.strong_count() > 0,
            Held::Struct(weak) => weak.strong_count() > 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Measure {
    bytes: usize,
    depth: usize,
}

/// Remembered container sizes, keyed by allocation address (see the module comment).
#[derive(Default)]
struct Sizes {
    entries: HashMap<usize, (Held, Measure)>,
    since_prune: usize,
}

impl Sizes {
    fn key(value: &Value) -> Option<(usize, Held)> {
        match value {
            Value::List(items) | Value::Set(items) => {
                Some((Arc::as_ptr(items) as *const () as usize, Held::List(Arc::downgrade(items))))
            }
            Value::Map(entries) => {
                Some((Arc::as_ptr(entries) as *const () as usize, Held::Map(Arc::downgrade(entries))))
            }
            Value::Struct(_, fields) => {
                Some((Arc::as_ptr(fields) as *const () as usize, Held::Struct(Arc::downgrade(fields))))
            }
            _ => None,
        }
    }

    fn address(value: &Value) -> Option<usize> {
        match value {
            Value::List(items) | Value::Set(items) => Some(Arc::as_ptr(items) as *const () as usize),
            Value::Map(entries) => Some(Arc::as_ptr(entries) as *const () as usize),
            Value::Struct(_, fields) => Some(Arc::as_ptr(fields) as *const () as usize),
            _ => None,
        }
    }

    fn get(&mut self, value: &Value) -> Option<Measure> {
        let address = Self::address(value)?;
        match self.entries.get(&address) {
            Some((held, measure)) if held.alive() => Some(*measure),
            Some(_) => {
                self.entries.remove(&address);
                None
            }
            None => None,
        }
    }

    fn record(&mut self, value: &Value, measure: Measure) {
        let Some((address, held)) = Self::key(value) else { return };
        self.entries.insert(address, (held, measure));
        self.since_prune += 1;
        if self.since_prune >= 1024 {
            self.since_prune = 0;
            self.entries.retain(|_, (held, _)| held.alive());
        }
    }
}

struct Meter {
    limits: RunLimits,
    policy: Option<BlockPolicy>,
    turn: Arc<TurnBudget>,
    started: Instant,
    /// The clock this block runs against: its own limit or what is left of the turn's
    /// block time, whichever is shorter, with the refusal text of the one that binds.
    deadline: Option<(Duration, String)>,
    steps: u64,
    produced: usize,
    io_calls: usize,
    costly: HashMap<&'static str, usize>,
    frames: Vec<Frame>,
    sizes: Sizes,
    /// Nodes visited by size walks; the non-quadratic property is asserted on this.
    walked: u64,
    /// Walk budget of the measurement in progress.
    walk_left: usize,
    /// Once a limit trips, every later check fails with the same text, so code that
    /// swallows an error (a `Result` boundary, a startup loop) cannot continue the block.
    tripped: Option<String>,
}

impl Meter {
    fn trip(&mut self, message: String) -> anyhow::Error {
        let message = format!("run stopped: {message}");
        self.tripped.get_or_insert(message.clone());
        anyhow!(message)
    }

    fn alive(&mut self) -> Result<()> {
        if let Some(message) = &self.tripped {
            return Err(anyhow!(message.clone()));
        }
        if let Some((limit, message)) = &self.deadline {
            if self.started.elapsed() > *limit {
                let message = message.clone();
                return Err(self.trip(message));
            }
        }
        Ok(())
    }

    fn step(&mut self) -> Result<()> {
        self.charge_steps(1)
    }

    /// Charge work to the step budget (one statement, loop iteration or call is one
    /// step; builtins add steps for the bytes they are handed and regex calls for the
    /// work they may do).
    fn charge_steps(&mut self, steps: u64) -> Result<()> {
        self.alive()?;
        self.steps = self.steps.saturating_add(steps);
        if self.limits.steps.is_some_and(|limit| self.steps > limit) {
            let limit = self.limits.steps.unwrap_or_default();
            return Err(self.trip(format!("it executed more than {limit} steps (limit for this conversation)")));
        }
        Ok(())
    }

    fn restricted_frame(&self) -> Option<&BlockPolicy> {
        match self.frames.last() {
            Some(Frame::Block) => self.policy.as_ref(),
            _ => None,
        }
    }

    /// Deep size with remembered containers short-circuiting the walk.
    fn walk(&mut self, value: &Value, depth: usize) -> Result<Measure, String> {
        if depth > MAX_VALUE_DEPTH {
            return Err(format!("a value is nested deeper than {MAX_VALUE_DEPTH} levels"));
        }
        if self.walk_left == 0 {
            return Err(self.too_large());
        }
        self.walk_left -= 1;
        self.walked += 1;
        if let Some(measure) = self.sizes.get(value) {
            if depth + measure.depth > MAX_VALUE_DEPTH {
                return Err(format!("a value is nested deeper than {MAX_VALUE_DEPTH} levels"));
            }
            return Ok(measure);
        }
        let mut total = Measure { bytes: SLOT, depth: 0 };
        if matches!(value, Value::String(_) | Value::List(_) | Value::Set(_) | Value::Map(_) | Value::Struct(..) | Value::Union { .. }) {
            total.bytes += HEAP;
        }
        let add = |meter: &mut Self, child: &Value, total: &mut Measure| -> Result<(), String> {
            let measure = meter.walk(child, depth + 1)?;
            total.bytes = total.bytes.saturating_add(measure.bytes);
            total.depth = total.depth.max(measure.depth + 1);
            Ok(())
        };
        match value {
            Value::Int(_) | Value::Bool(_) | Value::None | Value::CoroutineHandle(_) | Value::TaskHandle(_) => {}
            Value::Float(_) => {}
            Value::String(text) => total.bytes += text.len(),
            Value::Err(text) => total.bytes += text.len(),
            Value::Attachment(attachment) => {
                total.bytes += attachment.name.len() + attachment.mime_type.len();
                if let crate::attachment::AttachmentData::Memory(data) = &attachment.data {
                    total.bytes += data.len();
                }
            }
            Value::Ok(inner) => add(self, inner, &mut total)?,
            Value::Union { payload, .. } => {
                for item in payload {
                    add(self, item, &mut total)?;
                }
            }
            Value::List(items) | Value::Set(items) => {
                for item in items.iter() {
                    add(self, item, &mut total)?;
                }
            }
            Value::Map(entries) => {
                for (key, item) in entries.iter() {
                    add(self, key, &mut total)?;
                    add(self, item, &mut total)?;
                }
            }
            Value::Struct(_, fields) => {
                for item in fields.values() {
                    total.bytes += SLOT;
                    add(self, item, &mut total)?;
                }
            }
        }
        self.sizes.record(value, total);
        Ok(total)
    }

    fn too_large(&self) -> String {
        format!("a value exceeds {} (limit for a single value in this conversation)", bytes(self.limits.value_bytes))
    }

    /// Measure one value against the single-value cap. The walk visits at most
    /// `cap / SLOT` nodes: every node accounts for at least `SLOT` bytes.
    fn admit(&mut self, value: &Value) -> Result<Measure> {
        self.walk_left = self.limits.value_bytes / SLOT + 2;
        let measure = match self.walk(value, 0) {
            Ok(measure) => measure,
            Err(message) => return Err(self.trip(message)),
        };
        if measure.bytes > self.limits.value_bytes {
            let message = self.too_large();
            return Err(self.trip(message));
        }
        Ok(measure)
    }

    /// Bytes a result newly occupies at its top level (elements are shared, not copied).
    fn charge(&mut self, value: &Value) -> Result<()> {
        if self.limits.produced_bytes.is_none() {
            return Ok(());
        }
        let shallow = match value {
            Value::String(text) => SLOT + HEAP + text.len(),
            Value::Err(text) => SLOT + text.len(),
            Value::List(items) | Value::Set(items) => HEAP + SLOT * (items.len() + 1),
            Value::Map(entries) => HEAP + SLOT * (2 * entries.len() + 1),
            Value::Struct(_, fields) => HEAP + SLOT * (2 * fields.len() + 1),
            Value::Union { payload, .. } => HEAP + SLOT * (payload.len() + 1),
            Value::Attachment(attachment) => match &attachment.data {
                crate::attachment::AttachmentData::Memory(data) => SLOT + data.len(),
                crate::attachment::AttachmentData::File(_) => SLOT,
            },
            _ => SLOT,
        };
        self.charge_bytes(shallow)
    }

    fn charge_bytes(&mut self, amount: usize) -> Result<()> {
        let Some(limit) = self.limits.produced_bytes else { return Ok(()) };
        self.produced = self.produced.saturating_add(amount);
        if self.produced > limit {
            return Err(self.trip(format!("it produced more than {} of values in total (limit for this conversation)", bytes(limit))));
        }
        Ok(())
    }
}

fn bytes(count: usize) -> String {
    const MIB: usize = 1024 * 1024;
    if count % MIB == 0 { format!("{} MiB", count / MIB) } else { format!("{count} bytes") }
}

fn with<T>(run: impl FnOnce(&mut Meter) -> T) -> Option<T> {
    METER.with(|meter| meter.borrow_mut().as_mut().map(run))
}

/// Clears the meter when the block ends, on every path including a panic: blocking-pool
/// threads are reused, and a leftover meter would restrict unrelated code.
pub struct Installed(());

impl Drop for Installed {
    fn drop(&mut self) {
        ACTIVE.with(|active| active.set(false));
        if let Some(meter) = METER.with(|meter| meter.borrow_mut().take()) {
            meter.turn.spend_block_time(meter.started.elapsed());
        }
    }
}

/// Start metering the block that is about to run on this thread, and mark its frame.
pub fn install(context: RunContext) -> Installed {
    let block = context.limits.wall_clock.map(|limit| {
        (limit, format!("it ran longer than {} seconds (limit for this conversation)", limit.as_secs()))
    });
    let turn = context.turn.block_time_left().map(|(left, limit)| {
        (left, format!("the blocks of this turn ran longer than {} seconds in total (limit for one turn in this conversation)", limit.as_secs()))
    });
    let deadline = match (block, turn) {
        (Some(block), Some(turn)) => Some(if turn.0 < block.0 { turn } else { block }),
        (block, turn) => block.or(turn),
    };
    METER.with(|meter| {
        *meter.borrow_mut() = Some(Meter {
            limits: context.limits,
            policy: context.policy,
            turn: context.turn,
            started: Instant::now(),
            deadline,
            steps: 0,
            produced: 0,
            io_calls: 0,
            costly: HashMap::new(),
            frames: vec![Frame::Block],
            sizes: Sizes::default(),
            walked: 0,
            walk_left: 0,
            tripped: None,
        });
    });
    ACTIVE.with(|active| active.set(true));
    Installed(())
}

/// One statement, loop iteration or call.
pub(crate) fn step() -> Result<()> {
    with(Meter::step).unwrap_or(Ok(()))
}

/// [`step`] behind the [`active`] check: the form the interpreter's hot paths use.
#[inline(always)]
pub(crate) fn tick() -> Result<()> {
    if active() { step() } else { Ok(()) }
}

/// A call made while a block is running: count it, apply the block's call policy to its
/// own frame, and keep every value a builtin consumes or produces within the size cap.
pub(super) fn call(
    program: &crate::ast::Program,
    callee: &str,
    args: Vec<Value>,
    env: &mut super::Env,
) -> Result<Value> {
    step()?;
    if program.get_function(callee).is_some() {
        check_user_call(&program.qualify_function_name(callee))?;
        return super::eval_builtin_or_user_unmetered(program, callee, args, env);
    }
    // A union constructor may share a builtin's spelling; it is data, not that builtin.
    let builtin = (!super::is_union_variant(program, callee)).then_some(callee);
    if let Some(name) = builtin {
        check_builtin(name)?;
    }
    let arguments = admit_arguments(&args)?;
    // The step budget bounds work, not only the number of calls.
    charge((arguments.bytes / crate::run_limits::ARGUMENT_BYTES_PER_STEP) as u64)?;
    if builtin == Some("set") && regex_limits().is_some() {
        // Worst case: every item is unique, n(n-1)/2 equality comparisons.
        // One potential comparison is one step, before any scan starts.
        let n = args.len() as u64;
        charge(n.saturating_mul(n.saturating_sub(1)) / 2)?;
    }
    let result = super::eval_builtin_or_user_unmetered(program, callee, args, env)?;
    admit_result(builtin, &arguments, &result)?;
    Ok(result)
}

/// Extra steps for work a builtin does beyond being called.
pub(crate) fn charge(steps: u64) -> Result<()> {
    if steps == 0 {
        return Ok(());
    }
    with(|meter| meter.charge_steps(steps)).unwrap_or(Ok(()))
}

/// The block's clock, for loops inside one builtin call.
pub(crate) fn alive() -> Result<()> {
    with(Meter::alive).unwrap_or(Ok(()))
}

/// The single-value cap of the running block (`None` outside a block).
pub(crate) fn value_cap() -> Option<usize> {
    with(|meter| meter.limits.value_bytes)
}

/// The regex limits of the running block (`None` outside a block and for the owner).
pub(crate) fn regex_limits() -> Option<crate::run_limits::RegexLimits> {
    with(|meter| meter.limits.regex).flatten()
}

/// Refuse BEFORE allocating: a builtin is about to build a string of `bytes` bytes.
/// No-op outside a block.
pub(crate) fn reserve(bytes: usize) -> Result<()> {
    reserve_shape(1, 1, bytes)
}

/// Refuse BEFORE allocating a value of `nodes` values, `heaps` strings or containers and
/// `payload` bytes of text. No-op outside a block.
pub(crate) fn reserve_shape(nodes: usize, heaps: usize, payload: usize) -> Result<()> {
    with(|meter| {
        let bytes = nodes.saturating_mul(SLOT).saturating_add(heaps.saturating_mul(HEAP)).saturating_add(payload);
        if bytes > meter.limits.value_bytes {
            let message = meter.too_large();
            return Err(meter.trip(message));
        }
        Ok(())
    })
    .unwrap_or(Ok(()))
}

/// A writer that stops at a byte limit instead of growing.
struct Capped {
    text: String,
    limit: usize,
}

impl std::fmt::Write for Capped {
    fn write_str(&mut self, piece: &str) -> std::fmt::Result {
        if self.text.len() + piece.len() > self.limit {
            return Err(std::fmt::Error);
        }
        self.text.push_str(piece);
        Ok(())
    }
}

/// `format!("{value}")`, except that inside a block rendering stops at the value cap
/// (minus `already` bytes the caller has collected) instead of building the whole text.
/// A value's text can be larger than the value: a float renders in up to 330 characters.
pub(crate) fn render(value: &Value, already: usize) -> Result<String> {
    use std::fmt::Write;
    let Some(cap) = value_cap() else { return Ok(format!("{value}")) };
    let mut out = Capped { text: String::new(), limit: cap.saturating_sub(already.saturating_add(SLOT + HEAP)) };
    if write!(out, "{value}").is_err() {
        return Err(with(|meter| {
            let message = meter.too_large();
            meter.trip(message)
        })
        .unwrap_or_else(|| anyhow!("rendering failed")));
    }
    Ok(out.text)
}

/// Pops the callee marker when the invocation ends, also on error and unwind.
pub(crate) struct CalleeFrame(());

impl Drop for CalleeFrame {
    fn drop(&mut self) {
        with(|meter| {
            if meter.frames.len() > 1 {
                meter.frames.pop();
            }
        });
    }
}

/// Mark the frame of a declared function. Its statements are not the block's own.
pub(crate) fn enter_callee() -> Option<CalleeFrame> {
    with(|meter| meter.frames.push(Frame::Callee)).map(|()| CalleeFrame(()))
}

pub(crate) fn refusal(what: &str, policy: &BlockPolicy) -> anyhow::Error {
    let mut tools: Vec<&str> = policy.tools.iter().map(String::as_str).collect();
    tools.sort();
    let tools = if tools.is_empty() { String::new() } else { format!(", your tools ({})", tools.join(", ")) };
    anyhow!("run: {what} is not available in this conversation. A block here may call pure builtins{tools} and the functions it was admitted to call.")
}

/// A user function called from the current frame.
pub(crate) fn check_user_call(qualified: &str) -> Result<()> {
    with(|meter| match meter.restricted_frame() {
        Some(policy) if !policy.functions.contains(qualified) => Err(refusal(&format!("function `{qualified}`"), policy)),
        _ => Ok(()),
    })
    .unwrap_or(Ok(()))
}

/// A builtin (pure or IO) called from the current frame.
pub(crate) fn check_builtin(name: &str) -> Result<()> {
    with(|meter| match meter.restricted_frame() {
        Some(policy)
            if !crate::builtins::pure::allowed_in_restricted_block(name) && !policy.tools.contains(name) =>
        {
            Err(refusal(&format!("`{name}`"), policy))
        }
        _ => Ok(()),
    })
    .unwrap_or(Ok(()))
}

/// Statement forms a restricted block may not execute itself (spawn, sources, events).
pub(crate) fn check_statement(what: &str) -> Result<()> {
    with(|meter| match meter.restricted_frame() {
        Some(policy) => Err(refusal(what, policy)),
        None => Ok(()),
    })
    .unwrap_or(Ok(()))
}

/// One IO builtin dispatch (or spawn) anywhere under the block.
pub(crate) fn io_call(op: &str) -> Result<()> {
    with(|meter| {
        meter.alive()?;
        meter.io_calls += 1;
        if meter.limits.io_calls.is_some_and(|limit| meter.io_calls > limit) {
            let limit = meter.limits.io_calls.unwrap_or_default();
            return Err(meter.trip(format!("`{op}` would be IO call {} of this block; the limit for this conversation is {limit} per block", meter.io_calls)));
        }
        if let (Some(limit), Some(tool)) = (meter.limits.costly_calls, COSTLY_TOOLS.iter().find(|tool| **tool == op)) {
            let count = meter.costly.entry(tool).or_default();
            *count += 1;
            if *count > limit {
                return Err(meter.trip(format!("`{op}` may be called {limit} time(s) per block in this conversation")));
            }
        }
        let turn = meter.turn.clone();
        turn.take(op).map_err(|message| meter.trip(message))
    })
    .unwrap_or(Ok(()))
}

/// Sizes of a builtin's arguments, taken before the builtin consumes them.
pub(crate) struct Arguments {
    bytes: usize,
    depth: usize,
    first_depth: usize,
    first_bytes: usize,
    /// Allocation of the first argument, to see whether a grower extended it in place.
    first_address: Option<usize>,
}

/// Every argument must itself be within the single-value cap, so no builtin ever
/// renders or copies an unbounded value.
pub(crate) fn admit_arguments(args: &[Value]) -> Result<Arguments> {
    with(|meter| {
        let mut total = Arguments { bytes: 0, depth: 0, first_depth: 0, first_bytes: 0, first_address: args.first().and_then(Sizes::address) };
        for (index, arg) in args.iter().enumerate() {
            let measure = meter.admit(arg)?;
            total.bytes = total.bytes.saturating_add(measure.bytes);
            if index == 0 {
                total.first_depth = measure.depth;
                total.first_bytes = measure.bytes;
            } else {
                total.depth = total.depth.max(measure.depth + 1);
            }
        }
        // A builtin may combine its arguments (`concat(s, s, s, ...)`). The result is
        // checked after it exists, so bound what can be allocated before that check.
        if total.bytes > meter.limits.value_bytes.saturating_mul(2) {
            let message = meter.too_large();
            return Err(meter.trip(message));
        }
        Ok(total)
    })
    .unwrap_or(Ok(Arguments { bytes: 0, depth: 0, first_depth: 0, first_bytes: 0, first_address: None }))
}

/// Account for a builtin's result. `builtin` is `Some` only for a real registry builtin
/// (not a user function or a union constructor that happens to share the spelling).
pub(crate) fn admit_result(builtin: Option<&str>, arguments: &Arguments, result: &Value) -> Result<()> {
    with(|meter| {
        let grower = builtin.is_some_and(|name| GROWERS.contains(&name));
        let bound = arguments.bytes.saturating_add(SLOT);
        if grower && bound <= meter.limits.value_bytes {
            let depth = arguments.first_depth.max(arguments.depth);
            if depth > MAX_VALUE_DEPTH {
                return Err(meter.trip(format!("a value is nested deeper than {MAX_VALUE_DEPTH} levels")));
            }
            meter.sizes.record(result, Measure { bytes: bound, depth });
            // Extended in place (the list was not shared): only the new element is new
            // memory. Today `xs = push(xs, x)` always copies, and the copy is charged.
            if arguments.first_address.is_some() && arguments.first_address == Sizes::address(result) {
                return meter.charge_bytes(bound.saturating_sub(arguments.first_bytes));
            }
        } else {
            // Exact measurement: the bound was too coarse, or there is no rule.
            meter.admit(result)?;
        }
        meter.charge(result)
    })
    .unwrap_or(Ok(()))
}

/// A value that entered the block from outside a builtin: an IO result, a constructed
/// struct, the block's own return value.
pub(crate) fn admit_value(value: &Value) -> Result<()> {
    with(|meter| {
        meter.admit(value)?;
        meter.charge(value)
    })
    .unwrap_or(Ok(()))
}

#[cfg(test)]
pub(crate) fn walked_nodes() -> u64 {
    with(|meter| meter.walked).unwrap_or(0)
}

#[cfg(test)]
pub(crate) fn frame_is_block() -> Option<bool> {
    with(|meter| meter.frames.last() == Some(&Frame::Block))
}

#[cfg(test)]
pub(crate) fn recorded_bytes(value: &Value) -> Option<usize> {
    with(|meter| meter.sizes.get(value).map(|measure| measure.bytes)).flatten()
}

#[cfg(test)]
#[path = "meter_tests.rs"]
mod tests;
