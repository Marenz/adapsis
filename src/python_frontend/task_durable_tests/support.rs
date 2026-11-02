use crate::memory_graph::records::{RecordBinding, RecordStore};
use crate::{
    ast,
    coroutine::{CoroutineHandle, IoRequest, Runtime, TurnIdentity},
    eval, parser,
};
use std::sync::{
    Arc,
    atomic::{AtomicBool, AtomicUsize, Ordering},
};
use tokio::sync::{Barrier, mpsc, oneshot};

pub(super) struct Gate {
    remaining: AtomicUsize,
    barrier: Barrier,
    lose_commit: AtomicBool,
    commits: std::sync::Mutex<Vec<(String, String)>>,
}

pub(super) struct Harness {
    pub store: RecordStore,
    pub grant: RecordBinding,
    pub turn: TurnIdentity,
    pub sender: mpsc::Sender<IoRequest>,
    gate: Arc<Gate>,
    service: tokio::task::JoinHandle<()>,
}

impl Harness {
    pub fn open(path: &std::path::Path) -> Self {
        let store = RecordStore::disk_for_test(path).unwrap();
        let turn = TurnIdentity {
            principal: "fixture-owner".into(),
            context: "fixture-context".into(),
            may_write: true,
            guest_tools: None,
        };
        let grant = store.grant_for_test("fixture-namespace", &turn, "fixture-grant", true);
        let (runtime, mut receiver) = Runtime::new();
        let sender = runtime.io_sender();
        let runtime = Arc::new(runtime);
        let gate = Arc::new(Gate {
            remaining: AtomicUsize::new(0),
            barrier: Barrier::new(2),
            lose_commit: AtomicBool::new(false),
            commits: std::sync::Mutex::new(Vec::new()),
        });
        let service_gate = gate.clone();
        let service = tokio::spawn(async move {
            let mut workers = tokio::task::JoinSet::new();
            while let Some(request) = receiver.recv().await {
                let runtime = runtime.clone();
                let gate = service_gate.clone();
                workers.spawn(async move { handle(runtime, gate, request).await });
                while workers.try_join_next().is_some() {}
            }
            while workers.join_next().await.is_some() {}
        });
        Self {
            store,
            grant,
            turn,
            sender,
            gate,
            service,
        }
    }

    pub fn handle(&self) -> CoroutineHandle {
        CoroutineHandle::new(self.sender.clone())
            .with_turn(Some(self.turn.clone()))
            .with_records_for_test(self.store.clone(), self.grant.clone())
    }

    pub fn gate_next_two_reads(&self) {
        self.gate.remaining.store(2, Ordering::SeqCst);
    }

    pub fn lose_next_commit_reply(&self) {
        self.gate.lose_commit.store(true, Ordering::SeqCst);
    }

    pub fn committed_batch(&self, id: &str) -> String {
        self.gate
            .commits
            .lock()
            .unwrap()
            .iter()
            .find(|(request, _)| request == id)
            .unwrap()
            .1
            .clone()
    }

    pub async fn close(self) {
        self.service.abort();
        let _ = self.service.await;
    }
}

async fn handle(runtime: Arc<Runtime>, gate: Arc<Gate>, request: IoRequest) {
    let IoRequest::Records { request, reply } = request else {
        panic!("record IO only")
    };
    if let Some(commit) = request.commit_for_test() {
        gate.commits.lock().unwrap().push(commit);
    }
    let gated = request.is_read_for_test()
        && gate
            .remaining
            .fetch_update(Ordering::SeqCst, Ordering::SeqCst, |n| n.checked_sub(1))
            .is_ok();
    let lose_commit =
        request.is_commit_for_test() && gate.lose_commit.swap(false, Ordering::SeqCst);
    if gated || lose_commit {
        let (tx, rx) = oneshot::channel();
        runtime
            .handle_io(IoRequest::Records { request, reply: tx })
            .await;
        let result = rx.await.unwrap(); // Real DB has finished and released its lock.
        if lose_commit {
            drop(reply);
            return;
        } // Committed backend result, deliberately lost ACK.
        tokio::time::timeout(std::time::Duration::from_secs(5), gate.barrier.wait())
            .await
            .unwrap();
        let _ = reply.send(result); // Delay only; never fabricate/modify DB data.
    } else {
        runtime
            .handle_io(IoRequest::Records { request, reply })
            .await;
    }
}

pub(super) fn arguments(values: &[&str]) -> parser::Expr {
    if let [value] = values {
        return parser::parse_single_expr(&serde_json::to_string(value).unwrap()).unwrap();
    }
    let fields = values
        .iter()
        .enumerate()
        .map(|(index, value)| format!("_{}:{}", index, serde_json::to_string(value).unwrap()))
        .collect::<Vec<_>>()
        .join(",");
    parser::parse_single_expr(&format!("{{{fields}}}")).unwrap()
}

pub(super) async fn invoke(
    program: Arc<ast::Program>,
    handle: CoroutineHandle,
    function: &str,
    values: &[&str],
) -> (String, Vec<String>) {
    let function = function.to_string();
    let input = arguments(values);
    tokio::task::spawn_blocking(move || {
        crate::intern::set_display_interner(&program.shared_interner);
        let result = eval::eval_async_function(&program, &function, &input, handle).unwrap();
        let eval::Value::Union { variant, payload } = result else {
            panic!("expected application union: {result:?}")
        };
        let name = crate::intern::resolve_display(variant);
        let payload = payload
            .into_iter()
            .map(|value| match value {
                eval::Value::String(text) => text.as_ref().clone(),
                _ => panic!("unexpected application outcome payload"),
            })
            .collect();
        (name, payload)
    })
    .await
    .unwrap()
}

pub(super) async fn initialize(
    program: Arc<ast::Program>,
    harness: &Harness,
    task: &str,
    run: &str,
    title: &str,
) {
    let outcome = invoke(
        program,
        harness.handle(),
        "TaskDurable.initialize",
        &[
            "workspace",
            "project",
            task,
            run,
            title,
            &format!("init-{task}"),
        ],
    )
    .await;
    assert_eq!(outcome.0, "DurableCommitted", "{outcome:?}");
}

pub(super) async fn raw(harness: &Harness, op: &str, args: &[&str]) -> eval::Value {
    let handle = harness.handle();
    let op = op.to_string();
    let args = args
        .iter()
        .map(|s| eval::Value::string(*s))
        .collect::<Vec<_>>();
    tokio::task::spawn_blocking(move || handle.execute_await(&op, &args).unwrap())
        .await
        .unwrap()
}

pub(super) fn row(harness: &Harness, key: &str) -> crate::memory_graph::records::Record {
    harness
        .store
        .read(
            Some(&harness.grant),
            Some(&harness.turn),
            &serde_json::json!({"version":1,"keys":[key]}).to_string(),
        )
        .unwrap()
        .remove(0)
        .unwrap()
}

pub(super) fn field(row: &crate::memory_graph::records::Record, name: &str) -> String {
    row.fields
        .iter()
        .find(|f| f.name == name)
        .unwrap()
        .value
        .clone()
}
