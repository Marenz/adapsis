//! Execution of the workspace tools for one conversation turn.
//!
//! A [`Turn`] is created at the first workspace tool call of a turn and lives
//! until the turn ends. It holds what the turn pinned (site, result caps) and,
//! once loaded, the conversation's binding with its read ledger. Between turns
//! nothing is kept in memory: the binding is a file in the workspace state
//! directory (`crate::workspace::binding`) and is loaded again.
//!
//! Exclusion (DESIGN T4, one writer per workspace), outermost first:
//!
//! 1. the conversation's TURN lock, held by the turn's future;
//! 2. the conversation's WORK lock (`api/turn_lock.rs`), taken before every
//!    blocking call and moved INTO the blocking task: it is held until the
//!    library call has really finished, also when the turn's future is dropped
//!    meanwhile (the task then finishes and publishes; nothing is rolled back);
//! 3. the library's lifecycle lock, inside `open` and `close` only, released
//!    before the next step;
//! 4. the workspace's reservation, inside `BindingStore::claim`;
//! 5. try-locks on the turn and work locks of other conversations (never
//!    waiting), held by `claim` until the claimant's record is published;
//! 6. the state files.
//!
//! A conversation's binding record is written or removed only under that
//! conversation's work lock: by its own blocking task, or by a claimant that
//! got it idle. All of these locks are process-local (one active daemon per
//! state root).
use super::{Call, ResultCaps, Site};
use crate::api::{turn_lock, AppConfig};
use crate::workspace::binding::IdleGuard;
use crate::workspace::error::shown;
use crate::workspace::limits::{WORK_WAIT_MAX, WORK_WAIT_NOTED};
use crate::workspace::{self, Binding, Error, SearchMode, Workspace};
use std::sync::Arc;

/// What one workspace tool call produced.
#[derive(Debug)]
pub(in crate::api) struct Outcome {
    /// The tool result the model sees: plain text. A failure starts with `error:`.
    pub text: String,
    /// Structural success flag for the loop, the response and the training log.
    pub success: bool,
    /// What the event stream and the HTTP response say about the call: the
    /// tool and whether it worked, nothing else. Every subscriber of
    /// `/api/events` sees it, so it carries no path, pattern or name.
    pub label: String,
}

impl Outcome {
    fn new(name: &str, success: bool, text: String, cap: usize) -> Self {
        Self { text: fit(text, cap), success, label: format!("`{}`: {}", shown(name), if success { "ok" } else { "failed" }) }
    }

    pub(in crate::api) fn failed(name: &str, message: impl std::fmt::Display) -> Self {
        Self::new(name, false, format!("error: {message}"), usize::MAX)
    }
}

/// Last line of defence for the per-message cap: every tool bounds its own
/// output below it, so this cuts only what a tool failed to bound.
fn fit(text: String, cap: usize) -> String {
    const NOTE: &str = "\n[result cut by the runtime at the per-message limit]";
    if text.len() <= cap {
        return text;
    }
    let mut end = cap.saturating_sub(NOTE.len());
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    format!("{}{NOTE}", &text[..end])
}

/// The conversation's workspace as far as this turn knows it.
#[derive(Default)]
enum Bound {
    /// Not looked up yet in this turn.
    #[default]
    Unknown,
    None,
    Some(Box<(Workspace, Binding)>),
}

/// Everything a blocking tool call needs; moved into the blocking task and back.
struct Work {
    site: Arc<Site>,
    context: String,
    caps: ResultCaps,
    bound: Bound,
}

pub(in crate::api) struct Turn {
    /// `None` only while a call is running, or after its task died.
    work: Option<Work>,
    site: Arc<Site>,
    context: String,
    caps: ResultCaps,
    /// How long a call waits for earlier work of this conversation (a policy
    /// value; tests shorten it).
    pub(in crate::api) wait: std::time::Duration,
}

impl Turn {
    /// `config` is the turn's pinned snapshot, `model` the generating model.
    /// `None` when the turn has no workspace tools, or `context` is not the
    /// conversation the snapshot was made for.
    pub(in crate::api) fn start(config: &AppConfig, context: &str, model: &str) -> Option<Self> {
        if !super::available(config) || super::conversation(config) != context {
            return None;
        }
        Some(Self { work: None, site: config.workspaces.clone()?, context: context.to_string(), caps: ResultCaps::for_model(model), wait: WORK_WAIT_MAX })
    }

    /// Run one workspace tool. Never panics on any arguments; a panic inside
    /// the library is contained and reported as a failed call.
    pub(in crate::api) async fn call(&mut self, name: &str, arguments: &str) -> Outcome {
        let call = match super::parse(name, arguments) {
            Ok(call) => call,
            Err(error) => return Outcome::failed(name, error),
        };
        let started = std::time::Instant::now();
        // The work lock: free unless a call of an earlier, dropped turn of this
        // conversation is still running, or a claimant is unbinding us right now.
        let (held, waited) = match turn_lock::work_now(self.site.bindings.root(), &self.context) {
            Some(held) => (held, None),
            None => match turn_lock::work(self.site.bindings.root(), &self.context, self.wait).await {
                Some(held) => (held, Some(started.elapsed())),
                None => {
                    return Outcome::failed(
                        name,
                        format!(
                            "an earlier workspace operation of this conversation is still running after {} s, so `{name}` was not started and nothing was changed. Try again later; workspace_status shows the state once it has finished",
                            self.wait.as_secs()
                        ),
                    );
                }
            },
        };
        // Whatever ran meanwhile may have changed the binding: load it again.
        let mut work = self.work.take().filter(|_| waited.is_none()).unwrap_or_else(|| Work {
            site: self.site.clone(),
            context: self.context.clone(),
            caps: self.caps,
            bound: Bound::Unknown,
        });
        let cap = self.caps.message;
        let joined = tokio::task::spawn_blocking(move || {
            // Owned by this task: held until the library call has finished,
            // whatever happens to the future that started it.
            let _held = held;
            let result = work.run(call);
            (work, result)
        })
        .await;
        let (mut outcome, workspace) = match joined {
            Ok((work, result)) => {
                let workspace = work.label();
                self.work = Some(work);
                match result {
                    Ok(text) => (Outcome::new(name, true, text, cap), workspace),
                    // Binding store failures are journaled where they arise, once each.
                    Err(error) => (Outcome::new(name, false, format!("error: {error}"), cap), workspace),
                }
            }
            // The in-memory state is gone with the task; the next call loads the binding again.
            Err(_) => (Outcome::failed(name, format!("`{name}` stopped unexpectedly; nothing can be assumed about what it did")), String::new()),
        };
        if let Some(waited) = waited.filter(|waited| *waited >= WORK_WAIT_NOTED) {
            // Not a hang: say why the call took long.
            outcome.text = fit(
                format!("{}\n[note: this call first waited {} s for an earlier workspace operation of this conversation to finish]", outcome.text, waited.as_secs()),
                cap,
            );
        }
        // Metadata only (#88): paths, patterns and content never reach the log.
        eprintln!(
            "[workspace:{}] tool={name} ok={} workspace={} result_bytes={} ms={}",
            self.context,
            outcome.success,
            if workspace.is_empty() { "-" } else { &workspace },
            outcome.text.len(),
            started.elapsed().as_millis()
        );
        if crate::logging::content() {
            eprintln!("[workspace:{}] {name} {arguments} -> {}", self.context, shown(outcome.text.lines().next().unwrap_or_default()));
        }
        outcome
    }
}

/// The runtime-tail line that tells the model which workspace this
/// conversation has. `None` when the turn has no workspace tools. Reads only
/// the binding record (no git), and not while earlier work is still running.
pub(in crate::api) async fn tail_line(config: Option<&AppConfig>) -> Option<String> {
    let config = config.filter(|config| super::available(config))?;
    Some(tail_line_of(config.workspaces.as_ref()?, super::conversation(config)).await)
}

async fn tail_line_of(site: &Arc<Site>, context: &str) -> String {
    let (site, context) = (site.clone(), context.to_string());
    let Some(held) = turn_lock::work_now(site.bindings.root(), &context) else {
        return "Workspace of this conversation: an earlier workspace operation of this conversation is still finishing; workspace_status says more.".into();
    };
    let line = tokio::task::spawn_blocking(move || {
        let _held = held;
        match site.bindings.load(&context) {
            Ok(Some(binding)) => match site.registry.get(&binding.repo) {
                Ok(repo) => format!(
                    "Workspace of this conversation: {}/{} at {} (branch adapsis/{}). read, search and glob act on it.",
                    binding.repo,
                    binding.workspace,
                    repo.worktree_parent.join(&binding.workspace).display(),
                    binding.workspace
                ),
                Err(_) => format!(
                    "Workspace of this conversation: {}/{}, but that repository is not in the workspace registry; workspace_status says more.",
                    binding.repo, binding.workspace
                ),
            },
            Ok(None) => format!(
                "Workspace of this conversation: none. Registered repositories: {}. workspace_status lists open workspaces; workspace_open or workspace_use binds one.",
                site.registry.names().join(", ")
            ),
            Err(_) => "Workspace of this conversation: unknown, the stored binding could not be read; workspace_status says more.".to_string(),
        }
    })
    .await;
    line.unwrap_or_else(|_| "Workspace of this conversation: unknown; workspace_status says more.".into())
}


impl Work {
    fn label(&self) -> String {
        match &self.bound {
            Bound::Some(bound) => format!("{}/{}", bound.1.repo, bound.1.workspace),
            _ => String::new(),
        }
    }

    fn run(&mut self, call: Call) -> Result<String, Error> {
        match call {
            Call::Open { repo, name, base } => self.open(&repo, &name, base.as_deref()),
            Call::Use { repo, name } => self.use_existing(&repo, &name),
            Call::Status => self.status(),
            Call::Close => self.close(),
            Call::Read { path, offset, limit } => self.read(&path, offset, limit),
            Call::Search(mut request) => {
                request.max_output_bytes = self.caps.search;
                let (workspace, _) = self.bound()?;
                let result = workspace::search(workspace, &request)?;
                let unit = if result.mode == SearchMode::Content { "matches" } else { "files" };
                let more = if result.omitted > 0 { format!(", {} more not shown", result.omitted) } else { String::new() };
                Ok(format!("search `{}` in `{}`: {} {unit} shown{more}\n{}", shown(&request.pattern), shown(&request.path), result.shown, result.render()))
            }
            Call::Glob { pattern, path } => {
                let cap = self.caps.glob;
                let (workspace, _) = self.bound()?;
                let result = workspace::glob_within(workspace, &pattern, &path, cap)?;
                Ok(format!("glob `{}` in `{}`: {} of {} paths\n{}", shown(&pattern), shown(&path), result.entries.len(), result.total, result.render()))
            }
        }
    }

    /// The conversation's workspace, loaded from its binding on first use.
    ///
    /// The binding (and with it the ledger) is removed in ONE case: git
    /// answered and lists no such worktree (`Error::NotOpen`). Every other
    /// failure (git did not run or timed out, the checkout or the worktree
    /// directory is missing, which an unmounted disk also looks like, the
    /// repository is not in the registry) keeps it and reports the cause.
    fn bound(&mut self) -> Result<&mut (Workspace, Binding), Error> {
        if matches!(self.bound, Bound::Unknown) {
            self.bound = match self.site.bindings.load(&self.context)? {
                None => Bound::None,
                Some(binding) => {
                    let label = format!("{}/{}", binding.repo, binding.workspace);
                    let kept = |cause: String| {
                        Error::Lifecycle(format!(
                            "this conversation's workspace {label} cannot be used right now: {cause}. The binding is kept; try again, or bind another workspace with workspace_open or workspace_use"
                        ))
                    };
                    let repository = self.site.registry.get(&binding.repo).map_err(|_| kept("its repository is not in the workspace registry".into()))?;
                    match workspace::find(repository, &binding.workspace) {
                        Ok(workspace) => Bound::Some(Box::new((workspace, binding))),
                        Err(Error::NotOpen(reason)) => {
                            self.site.bindings.release(&self.context, None)?;
                            self.bound = Bound::None;
                            return Err(Error::Lifecycle(format!(
                                "this conversation's workspace {label} no longer exists, so it was unbound ({reason}). Use workspace_status, then workspace_open or workspace_use"
                            )));
                        }
                        Err(other) => return Err(kept(other.to_string())),
                    }
                }
            };
        }
        match &mut self.bound {
            Bound::Some(bound) => Ok(bound),
            _ => Err(Error::Lifecycle(
                "this conversation has no workspace. Use workspace_status to see repositories and open workspaces, then workspace_open or workspace_use".into(),
            )),
        }
    }

    /// Make `workspace` this conversation's workspace: the atomic claim.
    fn claim(&mut self, workspace: Workspace) -> Result<workspace::binding::Claimed, Error> {
        // "Is that conversation idle": both of its locks, without waiting, kept by the
        // claim until it has published.
        let scope = self.site.bindings.root().to_path_buf();
        let idle = |other: &str| turn_lock::idle(&scope, other).map(|guards| Box::new(guards) as IdleGuard);
        let claimed = self.site.bindings.claim(&self.context, &workspace, &idle)?;
        let binding = Binding::new(&workspace.repo, &workspace.name);
        self.bound = Bound::Some(Box::new((workspace, binding)));
        Ok(claimed)
    }

    /// What this conversation was bound to before, for the result text.
    fn previous(&mut self, repo: &str, name: &str) -> String {
        match self.bound() {
            Ok((_, binding)) if binding.repo != repo || binding.workspace != name => {
                format!("\nThis conversation was bound to {}/{} before; that workspace stays open.", binding.repo, binding.workspace)
            }
            _ => String::new(),
        }
    }

    fn open(&mut self, repo: &str, name: &str, base: Option<&str>) -> Result<String, Error> {
        let site = self.site.clone();
        let repository = site.registry.get(repo)?;
        workspace::lifecycle::validate_name("workspace name", name)?;
        let previous = self.previous(repo, name);
        // Advisory, before anything is created: a record left over from an
        // earlier workspace of this name would hold the new one, and its
        // conversation must not be working. The claim below decides.
        for holder in site.bindings.holders(repo, name)? {
            if holder != self.context && turn_lock::idle(site.bindings.root(), &holder).is_none() {
                return Err(Error::Lifecycle(format!(
                    "workspace {repo}/{name} was not opened: a conversation that is working right now holds a binding of that name; nothing was changed. Try again when it has finished, or choose another name"
                )));
            }
        }
        let workspace = workspace::open(repository, name, base)?;
        let text = format!(
            "opened workspace {repo}/{name}\npath: {}\nbranch: {}\nbase: {}{previous}",
            workspace.path.display(),
            workspace.branch,
            workspace.base
        );
        match self.claim(workspace) {
            Ok(_) => Ok(text),
            // Another conversation got it between the creation and the claim.
            // Which one is not said: a conversation key can identify a person.
            Err(Error::Lifecycle(_)) => Err(Error::Lifecycle(format!(
                "workspace {repo}/{name} was created, but it is bound elsewhere and not to this conversation. workspace_use can take it when the conversation that has it is idle"
            ))),
            Err(error) => Err(Error::Lifecycle(format!(
                "workspace {repo}/{name} was created, but it could not be bound to this conversation ({error}); use workspace_use once that is resolved"
            ))),
        }
    }

    fn use_existing(&mut self, repo: &str, name: &str) -> Result<String, Error> {
        let site = self.site.clone();
        let workspace = workspace::find(site.registry.get(repo)?, name)?;
        if matches!(self.bound(), Ok((_, binding)) if binding.repo == repo && binding.workspace == name) {
            return Ok(format!("workspace {repo}/{name} is already this conversation's workspace\npath: {}", workspace.path.display()));
        }
        let previous = self.previous(repo, name);
        let header = format!("using workspace {repo}/{name}\npath: {}\nbranch: {}\nbase: {}", workspace.path.display(), workspace.branch, workspace.base);
        let claimed = self.claim(workspace)?;
        Ok(format!(
            "{header}\n{}Nothing counts as read here yet: read a file before changing it.{previous}",
            if claimed.taken { "It was bound to another conversation, which no longer has it.\n" } else { "" }
        ))
    }

    fn status(&mut self) -> Result<String, Error> {
        let unbound = match self.bound() {
            Ok((workspace, binding)) => {
                let mut text = format!("{}\nrepository: {}\npath: {}", workspace::status(workspace)?.render(), binding.repo, workspace.path.display());
                for root in &workspace.read_roots {
                    text.push_str(&format!("\nread-only root: {}", root.display()));
                }
                return Ok(text);
            }
            Err(error) => error,
        };
        // Without a workspace the useful answer is what exists.
        match (unbound, &self.bound) {
            (Error::Lifecycle(reason), Bound::None) => Ok(format!("{reason}\n{}", self.inventory())),
            (error, _) => Err(error),
        }
    }

    /// Registered repositories and their open workspaces.
    fn inventory(&self) -> String {
        let mut lines = vec!["registered repositories:".to_string()];
        for name in self.site.registry.names() {
            let Ok(repo) = self.site.registry.get(name) else { continue };
            lines.push(format!("  {name} (default base {})", repo.base));
            match workspace::list(repo) {
                Ok(open) if open.is_empty() => lines.push("    no open workspaces".into()),
                Ok(open) => {
                    for workspace in open {
                        let holder = match self.site.bindings.holders(name, &workspace) {
                            Ok(holders) if holders.is_empty() => "not bound to a conversation",
                            Ok(_) => "bound to another conversation",
                            Err(_) => "binding unknown",
                        };
                        lines.push(format!("    {workspace} ({holder})"));
                    }
                }
                Err(error) => lines.push(format!("    workspaces could not be listed: {error}")),
            }
        }
        lines.join("\n")
    }

    fn close(&mut self) -> Result<String, Error> {
        let (workspace, binding) = self.bound()?;
        let label = format!("{}/{}", binding.repo, binding.workspace);
        let workspace = workspace.clone();
        workspace::close(&workspace)?;
        self.bound = Bound::None;
        self.site.bindings.release(&self.context, Some(&workspace))?;
        Ok(format!("closed workspace {label}; its directory is removed and branch {} is kept", workspace.branch))
    }

    fn read(&mut self, path: &str, offset: Option<u64>, limit: Option<usize>) -> Result<String, Error> {
        let cap = self.caps.read;
        let (context, site) = (self.context.clone(), self.site.clone());
        let (workspace, binding) = self.bound()?;
        let result = workspace::read_within(workspace, path, offset, limit, cap)?;
        let mut text = result.render();
        // The ledger is what lets a later edit prove the file is what was read.
        let mut lost = false;
        if result.writable && binding.ledger.recorded(&result.path) != Some(&result.facts) {
            binding.ledger.note(&result);
            // Conditional: written only while this conversation still holds the workspace.
            let saved = site.bindings.save_ledger(&context, workspace, &binding.ledger);
            if !matches!(saved, Ok(true)) {
                binding.ledger.forget(&result.path);
                text.push_str("\n[note: this read could not be recorded; read the file again before changing it]");
                // The record no longer binds this workspace: do not go on from memory.
                lost = matches!(saved, Ok(false));
            }
        }
        if lost {
            self.bound = Bound::Unknown;
        }
        Ok(text)
    }
}

#[cfg(test)]
#[path = "exec_tests.rs"]
mod tests;
