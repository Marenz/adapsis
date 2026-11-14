//! Workspace tools through the real takeover loop (#129, slice W1): mock provider, a
//! temporary git repository, registry and state root per test.
//!
//! Workspace tools exist only in the administrator's own private conversation, and there
//! are exactly two spellings of that key. Tests therefore share the conversation KEY
//! (its turn lock is process-wide) but nothing else: each has its own session, its own
//! repository name and its own binding store.
use super::*;
use crate::api::native_tools::workspace::Site;
use crate::workspace::test_support::{commit_all, write, RepoFixture};
use std::path::PathBuf;

/// The administrator's private conversation (`memory_graph::is_private_admin`), in the
/// spelling no other test uses (another test writes an identity file for the other one,
/// which would change the stable prefix between two turns here).
const DM: &str = "telegram:user:100000001";
#[path = "takeover_workspace_change_tests.rs"]
mod changes;

struct Repo {
    fixture: RepoFixture,
    site: Arc<Site>,
    name: String,
    worktrees: PathBuf,
}

const LIB: &str = "pub fn first() {}\n// needle in the haystack\npub fn third() {}\n";

/// A registered repository `ws-<tag>` with one commit: README.md and src/lib.rs.
fn repo(tag: &str) -> Repo {
    let fixture = RepoFixture::new();
    write(&fixture.repository.checkout, "src/lib.rs", LIB);
    commit_all(&fixture.repository.checkout, "source");
    let name = format!("ws-{tag}");
    Repo { site: fixture.site(&name), worktrees: fixture.repository.worktree_parent.clone(), name, fixture }
}

fn owner(repo: &Repo) -> Harness {
    let mut harness = Harness::new(true);
    harness.workspaces = Some(repo.site.clone());
    harness
}

/// The tool result the model was sent for call `id`: plain text.
fn text_of(request: &Value, id: &str) -> String {
    request["messages"]
        .as_array()
        .unwrap()
        .iter()
        .find(|message| message["tool_call_id"] == id)
        .unwrap_or_else(|| panic!("no result for call {id}"))["content"]
        .as_str()
        .unwrap()
        .to_string()
}

fn tool_names(request: &Value) -> Vec<String> {
    request["tools"].as_array().map(|tools| tools.iter().map(|tool| tool["function"]["name"].as_str().unwrap().to_string()).collect()).unwrap_or_default()
}

fn binding(repo: &Repo, context: &str) -> Option<crate::workspace::Binding> {
    repo.site.bindings.load(context).unwrap()
}

fn done(text: &str, id: &str) -> Value {
    response(text, vec![call(id, "finish", json!({}))])
}

/// Nothing a request carries says anything about workspaces.
fn assert_no_workspace_surface(label: &str, request: &Value, repo: &Repo) {
    let listed = tool_names(request);
    assert!(!listed.is_empty() && listed.iter().all(|tool| !crate::api::native_tools::workspace::is_workspace_name(tool)), "{label}: {listed:?}");
    let sent = request.to_string();
    assert!(!sent.contains("orkspace") && !sent.contains("worktree") && !sent.contains(&repo.name), "{label}: {sent}");
}

/// The whole first slice in three turns of the owner's private conversation: look, open,
/// read, search, list; a later turn knows the workspace from the runtime tail; after a
/// restart of the process (nothing in memory) the binding and what was read are still there.
#[tokio::test]
async fn an_owner_opens_reads_and_closes_a_workspace_across_turns_and_a_restart() {
    let repo = repo("flow");
    let path = repo.worktrees.join("one");
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("look", "workspace_status", json!({})),
        tool("open", "workspace_open", json!({"repo": repo.name, "name": "one"})),
        response("", vec![
            call("rd", "read", json!({"path": "src/lib.rs"})),
            call("gl", "glob", json!({"pattern": "**/*.rs"})),
            call("se", "search", json!({"pattern": "needle"})),
            call("miss", "read", json!({"path": "src/absent.rs"})),
        ]),
        done("Opened and read.", "f1"),
        tool("st", "workspace_status", json!({})),
        done("Still there.", "f2"),
    ])
    .await;
    assert_eq!(harness.turn(&provider, DM, ADMIN, "look at the repository").await, "Opened and read.");
    let requests = provider.requests.lock().unwrap().clone();
    let first = &requests[0];
    for name in ["workspace_open", "workspace_use", "workspace_status", "workspace_close", "read", "search", "glob"] {
        assert!(tool_names(first).iter().any(|tool| tool == name), "{name} missing: {:?}", tool_names(first));
    }
    assert!(system_text(first).contains("## Workspace tools"), "{}", system_text(first));
    assert!(all_text(first).contains(&format!("Workspace of this conversation: none. Registered repositories: {}.", repo.name)), "{}", all_text(first));
    let look = text_of(&requests[1], "look");
    assert!(look.starts_with("this conversation has no workspace."), "{look}");
    assert!(look.contains(&format!("registered repositories:\n  {} (default base main)\n    no open workspaces", repo.name)), "{look}");
    assert_eq!(
        text_of(&requests[2], "open"),
        format!("opened workspace {}/one\npath: {}\nbranch: adapsis/one\nbase: refs/heads/main", repo.name, path.display())
    );
    // Plain text under a header line, not a JSON envelope.
    let read = text_of(&requests[3], "rd");
    assert_eq!(read, format!("src/lib.rs: lines 1-3 of 3 ({} bytes)\n1: pub fn first() {{}}\n2: // needle in the haystack\n3: pub fn third() {{}}", LIB.len()));
    assert_eq!(text_of(&requests[3], "gl"), "glob `**/*.rs` in `.`: 1 of 1 paths\nsrc/lib.rs");
    assert_eq!(text_of(&requests[3], "se"), "search `needle` in `.`: 1 matches shown\nsrc/lib.rs:2: // needle in the haystack");
    assert_eq!(text_of(&requests[3], "miss"), "error: `src/absent.rs` does not exist in the workspace");
    // The binding and the read are on disk.
    let stored = binding(&repo, DM).expect("bound");
    assert_eq!((stored.repo.as_str(), stored.workspace.as_str()), (repo.name.as_str(), "one"));
    assert!(stored.ledger.recorded("src/lib.rs").is_some_and(|facts| facts.size == LIB.len() as u64 && facts.sha256.is_some()));
    assert_eq!(stored.ledger.len(), 1, "a failed read records nothing");
    assert!(path.join("src/lib.rs").is_file());

    // A later turn: the runtime tail names the workspace; the stable prefix is unchanged.
    assert_eq!(harness.turn(&provider, DM, ADMIN, "status?").await, "Still there.");
    let requests = provider.requests.lock().unwrap().clone();
    let tail = format!("Workspace of this conversation: {}/one at {} (branch adapsis/one).", repo.name, path.display());
    assert!(all_text(&requests[4]).contains(&tail), "{}", all_text(&requests[4]));
    assert!(!all_text(&requests[4]).contains("Workspace of this conversation: none"), "the tail is not part of the history");
    assert_eq!(requests[4]["messages"][0]["content"], requests[0]["messages"][0]["content"]);
    assert_eq!(tool_names(&requests[4]), tool_names(&requests[0]));
    let status = text_of(&requests[5], "st");
    assert!(status.starts_with("workspace one\nbranch: adapsis/one\nbase: refs/heads/main (ahead 0, behind 0)\nclean: no uncommitted changes"), "{status}");
    assert!(status.ends_with(&format!("repository: {}\npath: {}", repo.name, path.display())), "{status}");
    // History keeps the first turn's results as they were sent, and stays acceptable to the gateway.
    assert_eq!(text_of(&requests[4], "rd"), read);
    assert!(!all_text(&requests[4]).contains("are not shown in this turn"), "the owner's own turn sees its workspace batches");
    for request in &requests {
        assert_eq!(crate::context_log::gateway_boundary_errors(request["messages"].as_array().unwrap()), Vec::<String>::new());
    }

    // A restart: a new process state, the same registry and state directory.
    let restarted = owner(&repo);
    let provider = super::provider(vec![
        tool("again", "read", json!({"path": "src/lib.rs", "offset": 2, "limit": 1})),
        tool("close", "workspace_close", json!({})),
        tool("gone", "read", json!({"path": "src/lib.rs"})),
        done("Closed.", "f3"),
    ])
    .await;
    assert_eq!(restarted.turn(&provider, DM, ADMIN, "finish up").await, "Closed.");
    let requests = provider.requests.lock().unwrap().clone();
    assert!(all_text(&requests[0]).contains(&tail), "bound after the restart");
    assert_eq!(text_of(&requests[1], "again"), format!("src/lib.rs: lines 2-2 of 3 ({} bytes)\n2: // needle in the haystack\n[1 more lines; continue with offset=3]", LIB.len()));
    assert_eq!(text_of(&requests[2], "close"), format!("closed workspace {}/one; its directory is removed and branch adapsis/one is kept", repo.name));
    assert!(text_of(&requests[3], "gone").starts_with("error: this conversation has no workspace."), "{}", text_of(&requests[3], "gone"));
    assert!(!path.exists() && binding(&repo, DM).is_none());
}

/// Lifecycle and path refusals reach the model as text and change nothing. A workspace
/// git no longer lists is unbound with a reason; a failure to ASK (the checkout is not
/// reachable) keeps the binding and its ledger.
#[tokio::test]
async fn lifecycle_refusals_are_told_and_only_a_workspace_git_no_longer_lists_is_unbound() {
    let repo = repo("edge");
    let harness = owner(&repo);
    let provider = provider(vec![
        response("<iteration_budget>12</iteration_budget>", vec![call("norepo", "workspace_open", json!({"repo": "elsewhere", "name": "a"}))]),
        tool("badname", "workspace_open", json!({"repo": repo.name, "name": "Bad_Name"})),
        tool("nows", "workspace_use", json!({"repo": repo.name, "name": "absent"})),
        tool("noclose", "workspace_close", json!({})),
        tool("open", "workspace_open", json!({"repo": repo.name, "name": "edge"})),
        tool("twice", "workspace_open", json!({"repo": repo.name, "name": "edge"})),
        tool("escape", "read", json!({"path": "../../etc/passwd"})),
        tool("outside", "read", json!({"path": "/etc/hostname"})),
        tool("rd", "read", json!({"path": "README.md"})),
        done("Checked.", "f1"),
        // Turn 2: the checkout is unreachable (as with an unmounted disk).
        tool("transient", "read", json!({"path": "README.md"})),
        tool("status", "workspace_status", json!({})),
        done("Unreachable.", "f2"),
        // Turn 3: reachable again, nothing was lost.
        tool("back", "read", json!({"path": "README.md"})),
        done("Back.", "f3"),
        // Turn 4: the owner removed the worktree with git.
        tool("vanished", "read", json!({"path": "README.md"})),
        tool("after", "workspace_status", json!({})),
        done("Gone.", "f4"),
    ])
    .await;
    assert_eq!(harness.turn(&provider, DM, ADMIN, "try things").await, "Checked.");
    let requests = provider.requests.lock().unwrap().clone();
    let refusal = |index: usize, id: &str| text_of(&requests[index], id);
    assert_eq!(refusal(1, "norepo"), format!("error: no repository `elsewhere` in the workspace registry (registered: {})", repo.name));
    assert!(refusal(2, "badname").starts_with("error: invalid workspace name: `Bad_Name` may only contain lowercase letters, digits and dashes"));
    assert!(refusal(3, "nows").starts_with(&format!("error: no workspace `absent` in repository `{}` (none are open)", repo.name)));
    assert!(refusal(4, "noclose").starts_with("error: this conversation has no workspace."));
    assert!(refusal(6, "twice").contains("branch `adapsis/edge` already exists") && refusal(6, "twice").contains("use that workspace"), "{}", refusal(6, "twice"));
    assert!(refusal(7, "escape").starts_with("error: path `../../etc/passwd` refused: its `..` leaves the workspace"), "{}", refusal(7, "escape"));
    assert!(refusal(8, "outside").starts_with("error: path `/etc/hostname` refused: it is outside the workspace"), "{}", refusal(8, "outside"));
    let ledger = binding(&repo, DM).expect("refusals did not unbind").ledger;
    assert_eq!(ledger.len(), 1);

    // Transient: the checkout directory is not there. The binding and ledger stay.
    let checkout = repo.fixture.repository.checkout.clone();
    let parked = repo.fixture.top.join("checkout-parked");
    std::fs::rename(&checkout, &parked).unwrap();
    assert_eq!(harness.turn(&provider, DM, ADMIN, "read").await, "Unreachable.");
    let requests = provider.requests.lock().unwrap().clone();
    for (index, id) in [(11, "transient"), (12, "status")] {
        let text = text_of(&requests[index], id);
        assert!(text.starts_with(&format!("error: this conversation's workspace {}/edge cannot be used right now: repository `{}`: its checkout", repo.name, repo.name)), "{text}");
        assert!(text.ends_with("The binding is kept; try again, or bind another workspace with workspace_open or workspace_use"), "{text}");
    }
    assert_eq!(binding(&repo, DM).expect("kept").ledger, ledger, "the ledger survived the transient failure");
    std::fs::rename(&parked, &checkout).unwrap();
    assert_eq!(harness.turn(&provider, DM, ADMIN, "read").await, "Back.");
    let requests = provider.requests.lock().unwrap().clone();
    assert!(text_of(&requests[14], "back").starts_with("README.md: lines 1-1 of 1"));

    // Positive: git lists no such worktree any more.
    crate::workspace::test_support::git(&repo.worktrees.join("edge"), &["worktree", "remove", "--force", "."]);
    assert_eq!(harness.turn(&provider, DM, ADMIN, "read").await, "Gone.");
    let requests = provider.requests.lock().unwrap().clone();
    let vanished = text_of(&requests[16], "vanished");
    assert!(vanished.starts_with(&format!("error: this conversation's workspace {}/edge no longer exists, so it was unbound", repo.name)), "{vanished}");
    assert!(text_of(&requests[17], "after").starts_with("this conversation has no workspace."));
    assert!(binding(&repo, DM).is_none());
}

/// Results are bounded by the generating model's per-message cap, so the conversation
/// view never cuts one again or drops its batch. The fixture model is unknown to the
/// limits table: 128k window, 40,000 bytes per message, 20,000 per result.
#[tokio::test]
async fn results_stay_under_the_per_message_cap_and_survive_in_the_next_turns_history() {
    let repo = repo("caps");
    let checkout = repo.fixture.repository.checkout.clone();
    let big: String = (1..=4000).map(|n| format!("line {n} {}\n", "x".repeat(60))).collect();
    write(&checkout, "big.txt", &big);
    for index in 0..600 {
        write(&checkout, &format!("many/{}-{index:03}.txt", "p".repeat(50)), "needle needle needle needle needle needle needle needle\n");
    }
    commit_all(&checkout, "large content");
    let harness = owner(&repo);
    let provider = provider(vec![
        tool("open", "workspace_open", json!({"repo": repo.name, "name": "caps"})),
        response("", vec![
            call("rd", "read", json!({"path": "big.txt"})),
            call("se", "search", json!({"pattern": "needle", "limit": 2000})),
            call("gl", "glob", json!({"pattern": "**/*.txt"})),
        ]),
        tool("rest", "read", json!({"path": "big.txt", "offset": 270})),
        done("Read.", "f1"),
        done("Second turn.", "f2"),
    ])
    .await;
    assert_eq!(harness.turn(&provider, DM, ADMIN, "read the big file").await, "Read.");
    let requests = provider.requests.lock().unwrap().clone();
    let caps = crate::api::native_tools::workspace::ResultCaps::for_model("takeover-fixture");
    assert_eq!((caps.message, caps.read), (40_000, 20_000));
    let read = text_of(&requests[2], "rd");
    assert!(read.len() <= caps.read + 200 && read.len() > caps.read - 200, "{}", read.len());
    assert!(read.starts_with(&format!("big.txt: lines 1-269 of 4000 ({} bytes)\n1: line 1 ", big.len())), "{}", &read[..80]);
    assert!(read.ends_with("[3731 more lines; continue with offset=270]"), "{}", &read[read.len() - 80..]);
    let search = text_of(&requests[2], "se");
    assert!(search.len() <= caps.search + 100 && search.contains("more matches not shown (output size cap)"), "{}", search.len());
    assert!(search.starts_with("search `needle` in `.`: "), "{}", &search[..60]);
    let glob = text_of(&requests[2], "gl");
    assert!(glob.len() <= caps.glob + 100 && glob.starts_with("glob `**/*.txt` in `.`: ") && glob.contains("of 601 paths\n"), "{} {}", glob.len(), &glob[..60]);
    assert!(text_of(&requests[3], "rest").starts_with("big.txt: lines 270-"), "the stated offset continues the read");
    // The next turn's view carries every result exactly as it was sent.
    assert_eq!(harness.turn(&provider, DM, ADMIN, "and now?").await, "Second turn.");
    let requests = provider.requests.lock().unwrap().clone();
    for (id, sent) in [("rd", &read), ("se", &search), ("gl", &glob)] {
        assert_eq!(&text_of(&requests[4], id), sent, "{id} changed in the next turn's history");
    }
    assert!(!all_text(&requests[4]).contains("omitted here by the runtime"));
}

/// Workspace tools exist only where the owner is the only speaker. On a node WITH a
/// registry, real turns of: an execute-only speaker in a group, a guest in a DM, and the
/// OWNER in a group. Nothing they are sent mentions a workspace, and forged calls get
/// the refusal of a name that was never a tool, creating nothing.
#[tokio::test]
async fn outside_the_owners_private_conversation_there_are_no_workspace_tools() {
    let repo = repo("others");
    for (label, context, speaker, profile) in [
        ("execute-only", "telegram:group:-88241", "telegram:user:100000778", Some("viewer")),
        ("guest", restricted::GUEST_DM, restricted::GUEST, None),
        ("owner in a group", "telegram:group:-88242", ADMIN, None),
        ("another person in the owner's DM key", "telegram:100000001", "telegram:user:100000780", None),
    ] {
        let mut harness = Harness::new(true);
        harness.permissions = restricted::permissions();
        harness.workspaces = Some(repo.site.clone());
        if let Some(profile) = profile {
            harness.set_profile(context, Some(profile));
        }
        let provider = provider(vec![
            tool("forged-open", "workspace_open", json!({"repo": repo.name, "name": "theirs"})),
            tool("forged-read", "read", json!({"path": "README.md"})),
            tool("forged-none", "zz_no_such_tool", json!({})),
            done("Nothing.", "f1"),
        ])
        .await;
        assert_eq!(harness.turn(&provider, context, speaker, "open the repository").await, "Nothing.", "{label}");
        let requests = provider.requests.lock().unwrap().clone();
        assert_eq!(requests.len(), 4, "{label}");
        assert_no_workspace_surface(label, &requests[0], &repo);
        let refusal = |index: usize, id: &str| text_of(&requests[index], id);
        let unknown = refusal(3, "forged-none");
        assert!(unknown.contains("unknown native tool `zz_no_such_tool`"), "{label}: {unknown}");
        assert_eq!(refusal(1, "forged-open"), unknown.replace("zz_no_such_tool", "workspace_open"), "{label}");
        assert_eq!(refusal(2, "forged-read"), unknown.replace("zz_no_such_tool", "read"), "{label}");
        assert!(binding(&repo, context).is_none() && !repo.worktrees.exists(), "{label}: nothing was created");
    }
}

/// The reviewer's leak, as a regression. A shared conversation whose stored history
/// holds a workspace batch (here: stored directly, as an older version or a later
/// widening could leave it): no turn in that conversation is shown it, neither the
/// guest's nor the owner's, because neither has the workspace tools there. The stored
/// history is untouched. In the owner's private conversation the same batch is shown.
#[tokio::test]
async fn workspace_batches_in_a_history_are_hidden_from_every_turn_without_the_tools() {
    let repo = repo("history");
    let group = "telegram:group:-988294821";
    let canary = "SYNTHETIC_REPO_CANARY_94821";
    let mut harness = owner(&repo);
    harness.permissions = restricted::permissions();
    let store = |context: &str| {
        let mut meta = harness.meta.lock().unwrap();
        let conv = meta.conversations.get_or_create(context);
        conv.push_user("owner: open the repository");
        conv.push_native_batch(
            "Reading.".into(),
            vec![
                crate::llm::ToolCall { id: "o1".into(), name: "workspace_open".into(), arguments: json!({"repo": repo.name, "name": "history"}).to_string() },
                crate::llm::ToolCall { id: "r1".into(), name: "read".into(), arguments: json!({"path": "private-source.txt"}).to_string() },
            ],
            vec![format!("opened workspace {}/history", repo.name), format!("private-source.txt: lines 1-1 of 1 (28 bytes)\n1: {canary}")],
        );
        conv.push_native_batch("".into(), vec![crate::llm::ToolCall { id: "f0".into(), name: "finish".into(), arguments: "{}".into() }], vec!["{\"done\":true}".into()]);
        conv.push_assistant("Owner finished.");
        conv.messages.len()
    };
    let stored = store(group);
    let provider = provider(vec![done("Guest finished.", "g1"), done("Owner in group finished.", "g2"), done("Owner alone finished.", "g3")]).await;
    harness.turn(&provider, group, restricted::GUEST, "what did the file contain?").await;
    harness.turn(&provider, group, ADMIN, "and for me?").await;
    let requests = provider.requests.lock().unwrap().clone();
    for (label, request) in [("guest", &requests[0]), ("owner in the group", &requests[1])] {
        let sent = request.to_string();
        assert!(!sent.contains(canary) && !sent.contains("private-source.txt"), "{label}: file content or path reached a shared turn");
        for name in ["workspace_open", "\"read\""] {
            assert!(!sent.contains(name), "{label}: `{name}` is in the request");
        }
        // The rest of the history is there, and the turn is told something is not shown.
        assert!(sent.contains("owner: open the repository") && sent.contains("Owner finished.") && sent.contains("\"f0\""), "{label}");
        assert!(all_text(request).contains("[System: 1 earlier tool exchange(s) of this conversation are not shown in this turn.]"), "{label}: {}", all_text(request));
        assert_eq!(crate::context_log::gateway_boundary_errors(request["messages"].as_array().unwrap()), Vec::<String>::new(), "{label}");
    }
    // The notice is runtime text of the current turn, not history.
    let history = requests[0]["messages"].as_array().unwrap();
    assert!(history[1..history.len() - 1].iter().all(|message| !message.to_string().contains("are not shown in this turn")));
    {
        let meta = harness.meta.lock().unwrap();
        let conv = meta.conversations.get(group).unwrap();
        assert!(conv.messages.len() > stored && conv.messages.iter().any(|message| message.content.contains(canary)), "the stored history is complete");
    }
    // The owner's private conversation: the same kind of batch is shown.
    store(DM);
    harness.turn(&provider, DM, ADMIN, "what did the file contain?").await;
    let requests = provider.requests.lock().unwrap().clone();
    let own = requests[2].to_string();
    assert!(own.contains(canary) && own.contains("workspace_open") && !own.contains("are not shown in this turn"), "{own}");
}

/// The reviewer's scenario itself: the owner tries to open and read in a group, a guest
/// asks next. With workspace tools confined to the owner's private conversation the
/// owner's calls do nothing there, so nothing can leak.
#[tokio::test]
async fn an_owner_cannot_read_a_repository_into_a_group_history() {
    let repo = repo("review-history");
    write(&repo.fixture.repository.checkout, "private-source.txt", "SYNTHETIC_REPO_CANARY_94821\n");
    commit_all(&repo.fixture.repository.checkout, "synthetic source");
    let mut harness = owner(&repo);
    harness.permissions = restricted::permissions();
    let context = "telegram:group:-988294822";
    let provider = provider(vec![
        tool("open", "workspace_open", json!({"repo": repo.name, "name": "history"})),
        tool("read-file", "read", json!({"path": "private-source.txt"})),
        done("Owner finished.", "finish-owner"),
        done("Guest finished.", "finish-guest"),
    ])
    .await;
    harness.turn(&provider, context, ADMIN, "open repository").await;
    harness.turn(&provider, context, restricted::GUEST, "what did the file contain?").await;
    let requests = provider.requests.lock().unwrap().clone();
    assert_no_workspace_surface("owner in the group", &requests[0], &repo);
    let guest = &requests[3];
    assert!(!tool_names(guest).iter().any(|name| name == "read" || name.starts_with("workspace_")));
    assert!(!guest.to_string().contains("SYNTHETIC_REPO_CANARY_94821") && !guest.to_string().contains("workspace_open"), "{guest}");
    assert!(!repo.worktrees.exists() && binding(&repo, context).is_none());
}

/// A node without a registry: an owner turn is byte for byte what it was before
/// workspace tools existed, including the refusal of a forged call.
#[tokio::test]
async fn without_a_registry_an_owner_turn_is_unchanged() {
    let harness = Harness::new(true);
    let provider = provider(vec![
        tool("forged", "read", json!({"path": "README.md"})),
        tool("forged-none", "zz_no_such_tool", json!({})),
        done("No.", "f1"),
    ])
    .await;
    assert_eq!(harness.turn(&provider, DM, ADMIN, "read a file").await, "No.");
    let requests = provider.requests.lock().unwrap().clone();
    // The tool list of the writer class before workspace tools.
    assert_eq!(
        tool_names(&requests[0]),
        ["inspect", "mutate", "test", "evaluate", "run", "finish", "memory_inspect", "memory_remember", "memory_correct", "memory_forget"]
    );
    assert!(!requests[0].to_string().contains("orkspace"), "{}", requests[0]);
    let unknown = text_of(&requests[2], "forged-none");
    assert!(unknown.contains("unknown native tool `zz_no_such_tool`"), "{unknown}");
    assert_eq!(text_of(&requests[1], "forged"), unknown.replace("zz_no_such_tool", "read"));
}

/// A fragment of a cut-off response that ends up as the delivered reply (the model then
/// finishes with a tool call only) is marked; three cut-off responses in a row end the
/// turn with its own message instead of running to the round ceiling.
#[tokio::test]
async fn a_cut_off_reply_is_marked_and_repeated_cut_offs_end_the_turn() {
    let cut = |text: &str| {
        let mut output = response(text, vec![]);
        output["choices"][0]["finish_reason"] = json!("length");
        output
    };
    let harness = Harness::new(true);
    let provider = provider(vec![cut("The answer starts here and then"), tool("f1", "finish", json!({}))]).await;
    assert_eq!(
        harness.turn(&provider, "telegram:group:-88271", ADMIN, "explain").await,
        "The answer starts here and then [cut off at the model's output limit]"
    );
    // Later complete prose replaces the fragment: no marker.
    let provider = super::provider(vec![cut("A fragment"), done("The whole answer.", "f2")]).await;
    assert_eq!(harness.turn(&provider, "telegram:group:-88272", ADMIN, "explain").await, "The whole answer.");

    let provider = super::provider(vec![cut("one"), cut("two"), cut("three"), done("never asked", "f3")]).await;
    let reply = harness.turn(&provider, "telegram:group:-88273", ADMIN, "explain").await;
    assert_eq!(
        reply,
        "three [cut off at the model's output limit]\n\n(The model's responses were cut off at its output limit three times in a row, so this turn was stopped. Nothing from the cut-off responses was carried out.)"
    );
    assert_eq!(provider.requests.lock().unwrap().len(), 3, "no fourth generation");
    // A complete response in between resets the count.
    let provider = super::provider(vec![cut("one"), cut("two"), tool("look", "inspect", json!({"kind": "symbols"})), cut("three"), cut("four"), done("Done.", "f4")]).await;
    assert_eq!(harness.turn(&provider, "telegram:group:-88274", ADMIN, "explain").await, "Done.");
    assert_eq!(provider.requests.lock().unwrap().len(), 6);
}

/// The prose-only final round is already out of budget: a length stop marks
/// its partial reply but must never buy another generation.
#[tokio::test]
async fn a_cut_off_final_status_reply_is_marked_without_another_generation() {
    let harness = Harness::new(true);
    let mut final_fragment = response("The final status was partially", vec![]);
    final_fragment["choices"][0]["finish_reason"] = json!("length");
    let provider = provider(vec![
        response("<iteration_budget>1</iteration_budget>Looking.", vec![call("look", "inspect", json!({"kind":"symbols"}))]),
        final_fragment,
        done("Never requested.", "never"),
    ]).await;
    let reply = harness.turn(&provider, "telegram:group:-88275", ADMIN, "status").await;
    assert!(reply.starts_with("The final status was partially [cut off at the model's output limit]"), "{reply}");
    assert!(reply.contains("before the task was confirmed complete"), "{reply}");
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 2, "no correction after the final status round");
    assert_eq!(requests[1]["tool_choice"], "none");
}

/// #146's accepted-finish notice and the cutoff marker are independent reply
/// annotations: a cutoff fragment followed by a second, accepted finish must
/// carry both. The first finish still enforces this turn's unsaved edit once.
#[tokio::test]
async fn unsaved_finish_notice_and_cut_off_reply_marker_both_survive_integration() {
    let harness = Harness::new(true);
    let library = tempfile::tempdir().unwrap();
    std::fs::write(library.path().join("library-sources.json"),
        r#"{"version":1,"modules":{"IntegratedSave":{"format":"python","dependencies":[]}}}"#).unwrap();
    std::fs::write(library.path().join("IntegratedSave.py"), "def value() -> int:\n    return 1\n").unwrap();
    let mut program = crate::ast::Program::default();
    let state = crate::library::load_library_from_dir_for_tests(&mut program, library.path());
    state.require_authoritative_sources().unwrap();
    program.require_modules = true;
    *harness.program.write().await = program;
    harness.meta.lock().unwrap().library_state = Some(state);
    // Accepted by the legacy source surface, but its printed form exceeds the
    // stored-source limit, so this edit remains live and unsaved.
    let deep = format!("+module IntegratedSave\n+fn long (x:Int)->Int\n  +return x{}\n+end\n", " + 1".repeat(130));
    let mut fragment = response("Partial status", vec![]);
    fragment["choices"][0]["finish_reason"] = json!("length");
    let provider = provider(vec![
        tool("edit", "mutate", json!({"source":deep})),
        tool("first", "finish", json!({})),
        fragment,
        tool("second", "finish", json!({})),
    ]).await;
    let reply = harness.turn(&provider, "telegram:group:-88276", ADMIN, "edit and report").await;
    assert!(reply.starts_with("Partial status [cut off at the model's output limit]"), "{reply}");
    assert!(reply.contains("[Runtime: not saved to source: IntegratedSave ("), "{reply}");
    assert!(reply.ends_with("The change is live only until the next restart.]"), "{reply}");
    let requests = provider.requests.lock().unwrap();
    assert_eq!(requests.len(), 4);
    assert!(text_of(&requests[2], "first").contains("Cannot accept finish yet"));
}

/// A result stored under a model with a large window stays usable after a switch to a
/// model with a small one: the view cuts the over-long RESULT with the omission marker
/// (C4) and keeps the batch, instead of dropping the whole batch.
#[tokio::test]
async fn a_large_result_survives_a_switch_to_a_model_with_a_smaller_window() {
    let repo = repo("switch");
    let checkout = repo.fixture.repository.checkout.clone();
    let big: String = (1..=2000).map(|n| format!("line {n} {}\n", "x".repeat(60))).collect();
    write(&checkout, "big.txt", &big);
    commit_all(&checkout, "large content");
    let mut harness = owner(&repo);
    harness.model = "anthropic/claude-opus-5";
    let provider = provider(vec![
        tool("open", "workspace_open", json!({"repo": repo.name, "name": "switch"})),
        tool("rd", "read", json!({"path": "big.txt"})),
        done("Read.", "f1"),
        done("Smaller model.", "f2"),
    ])
    .await;
    harness.turn(&provider, DM, ADMIN, "read the big file").await;
    let large = text_of(&provider.requests.lock().unwrap()[2], "rd");
    assert!(large.len() > 100_000 && large.len() <= 102_400 + 200, "{}", large.len());
    // The same conversation, now generated by a model unknown to the limits table.
    harness.model = "takeover-fixture";
    assert_eq!(harness.turn(&provider, DM, ADMIN, "and now?").await, "Smaller model.");
    let requests = provider.requests.lock().unwrap().clone();
    let small = &requests[3];
    let cap = crate::model_limits::for_model("takeover-fixture").max_message_bytes();
    let shown = text_of(small, "rd");
    assert!(shown.len() <= cap && shown.len() > cap - 200, "{} of {cap}", shown.len());
    assert!(shown.starts_with("big.txt: lines 1-") && shown.ends_with(&large[large.len() - 200..]), "head and tail are kept");
    assert!(shown.contains("characters omitted here by the runtime to fit the model context; the stored message is complete"), "{}", &shown[cap / 2 - 200..cap / 2 + 200]);
    // The batch is whole: its call, the other batches, and a valid protocol.
    assert!(small.to_string().contains("\"id\":\"rd\"") && small.to_string().contains("\"id\":\"open\""));
    assert_eq!(crate::context_log::gateway_boundary_errors(small["messages"].as_array().unwrap()), Vec::<String>::new());
    // Stored complete.
    let meta = harness.meta.lock().unwrap();
    assert!(meta.conversations.get(DM).unwrap().messages.iter().any(|message| message.content == large));
}
