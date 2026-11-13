//! Inert callback transport with the real declared-function evaluator and registry.
use super::*;
use crate::coroutine::{IoRequest, TaskInfo, TaskRegistry, WaitReason};
use crate::eval;
use tokio::sync::{Semaphore, mpsc};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct Launch {
    pub function: String,
    pub destination: String,
    pub message: String,
}

pub(super) struct CallbackIo {
    pub queue_delay: std::time::Duration,
    pub sender: mpsc::Sender<IoRequest>,
    pub registry: TaskRegistry,
    pub launches: Arc<Mutex<Vec<Launch>>>,
    pub delivered: Arc<Mutex<Vec<String>>>,
    pub gate: Arc<Semaphore>,
    pub training: Arc<tokio::sync::Mutex<tokio::fs::File>>,
    pub training_file: tempfile::NamedTempFile,
    driver: tokio::task::JoinHandle<()>,
}

const CALLBACKS: &str = r#"+module ProgressFixture
+fn reply (destination:String, message:String)->String [io,async]
  +await response:Result<String> = http_post("https://progress.invalid/deliver", concat(destination, concat("|", message)), "text/plain")
  +return "attempted"
+end
+fn fail (destination:String, message:String)->String
  +return regex_replace("[CALLBACK_PRIVATE_ERROR", "", "")
+end
"#;

impl CallbackIo {
    pub async fn new(harness: &Harness, blocked: bool) -> Self {
        // Install trusted parsed declarations without a persistence operation:
        // each fixture owns its Program, not a shared library-source filename.
        {
            let mut program = harness.program.write().await;
            for operation in crate::parser::parse(CALLBACKS).unwrap() {
                crate::validator::apply_and_validate(&mut program, &operation).unwrap();
            }
        }
        let (sender, receiver) = mpsc::channel(8);
        let registry = TaskRegistry::default();
        let launches = Arc::new(Mutex::new(Vec::new()));
        let delivered = Arc::new(Mutex::new(Vec::new()));
        let gate = Arc::new(Semaphore::new(if blocked { 0 } else { 100 }));
        let training_file = tempfile::NamedTempFile::new_in("/tmp/opencode").unwrap();
        let training = Arc::new(tokio::sync::Mutex::new(tokio::fs::File::from_std(
            training_file.reopen().unwrap(),
        )));
        let driver = tokio::spawn(drive(
            receiver,
            sender.clone(),
            registry.clone(),
            harness.program.clone(),
            harness.meta.clone(),
            launches.clone(),
            delivered.clone(),
            gate.clone(),
        ));
        Self {
            queue_delay: std::time::Duration::ZERO,
            sender,
            registry,
            launches,
            delivered,
            gate,
            training,
            training_file,
            driver,
        }
    }

    pub async fn turn(
        &self,
        harness: &Harness,
        provider: &Provider,
        context: &str,
        principal: &str,
        text: &str,
        pair: (Option<&str>, Option<&str>),
    ) -> String {
        let future = crate::api::handle_llm_takeover(
            context.into(),
            text.into(),
            None,
            Some(json!({"speaker_id":principal}).to_string()),
            harness.graph.clone(),
            Arc::new(crate::memory_graph::MemoryEmbedder::deterministic()),
            pair.0.map(str::to_string),
            pair.1.map(str::to_string),
            harness.meta.clone(),
            harness.program.clone(),
            Default::default(),
            &provider.url,
            harness.model,
            None,
            self.sender.clone(),
            self.registry.clone(),
            Default::default(),
            Arc::new(tokio::sync::Mutex::new(())),
            ".".into(),
            Some(self.training.clone()),
            crate::permissions::AccessLevel::Full,
            harness.permissions.clone(),
            None,
            None,
            harness.native_tools,
        );
        let future =
            crate::api::takeover_progress::TEST_QUEUE_DELAY.scope(self.queue_delay, future);
        tokio::time::timeout(std::time::Duration::from_secs(15), future)
            .await
            .unwrap()
            .unwrap()
    }

    pub async fn queued_turn(
        &self,
        harness: &Harness,
        provider: &Provider,
        context: &str,
        principal: &str,
        text: &str,
        pair: (Option<&str>, Option<&str>),
    ) -> String {
        let (blocker, release) = held_provider(harness.native_tools, false).await;
        let first = self.turn(
            harness,
            &blocker,
            context,
            ADMIN,
            "queue blocker",
            (None, None),
        );
        tokio::pin!(first);
        tokio::select! {
            _ = &mut first => panic!("blocker provider is held"),
            _ = until(|| blocker.requests.lock().unwrap().len() == 1) => {},
        }
        let before = self.launches.lock().unwrap().len();
        let second = self.turn(harness, provider, context, principal, text, pair);
        tokio::pin!(second);
        tokio::select! {
            _ = &mut second => panic!("must queue behind first turn"),
            _ = until(|| self.launches.lock().unwrap().len() > before) => {},
        }
        release.add_permits(1);
        let (_, result) = tokio::join!(first, second);
        result
    }
}

impl Drop for CallbackIo {
    fn drop(&mut self) {
        self.gate.add_permits(100);
        self.driver.abort();
    }
}

async fn drive(
    mut receiver: mpsc::Receiver<IoRequest>,
    sender: mpsc::Sender<IoRequest>,
    registry: TaskRegistry,
    program: Arc<tokio::sync::RwLock<crate::ast::Program>>,
    meta: SharedMeta,
    launches: Arc<Mutex<Vec<Launch>>>,
    delivered: Arc<Mutex<Vec<String>>>,
    gate: Arc<Semaphore>,
) {
    let mut next_id = 1;
    while let Some(request) = receiver.recv().await {
        match request {
            IoRequest::Spawn {
                function_name,
                args,
                reply,
            } => {
                launches.lock().unwrap().push(Launch {
                    function: function_name.clone(),
                    destination: string(&args[0]),
                    message: string(&args[1]),
                });
                let id = next_id;
                next_id += 1;
                registry.lock().unwrap().insert(
                    id,
                    TaskInfo {
                        id,
                        function_name: function_name.clone(),
                        status: WaitReason::Running,
                        started_at: "0".into(),
                    },
                );
                let _ = reply.send(Ok(id));
                let (program, meta, sender, registry) = (
                    program.clone(),
                    meta.clone(),
                    sender.clone(),
                    registry.clone(),
                );
                tokio::task::spawn_blocking(move || {
                    let program = program.blocking_read().clone();
                    let ctx = eval::EvalContext::new_minimal(
                        Default::default(),
                        meta,
                        &program,
                        eval::make_shared_program_mut(&program),
                    );
                    ctx.install();
                    let handle = crate::coroutine::CoroutineHandle::new_with_task(
                        sender,
                        id,
                        registry.clone(),
                        Default::default(),
                    );
                    let mut env = eval::Env::new_with_shared_interner(&program.shared_interner);
                    env.set("__coroutine_handle", eval::Value::CoroutineHandle(handle));
                    let declaration = program.get_function(&function_name).unwrap();
                    for (parameter, value) in declaration.params.iter().zip(args) {
                        env.set(&parameter.name, value);
                    }
                    let status = match eval::eval_declared_function(
                        &program,
                        &function_name,
                        declaration,
                        &mut env,
                    ) {
                        Ok(value) => WaitReason::Completed(format!("{value}")),
                        Err(error) => WaitReason::Failed(format!("{error:#}")),
                    };
                    registry.lock().unwrap().get_mut(&id).unwrap().status = status;
                });
            }
            IoRequest::HttpPost {
                url, body, reply, ..
            } => {
                assert_eq!(url, "https://progress.invalid/deliver");
                let (gate, delivered) = (gate.clone(), delivered.clone());
                tokio::spawn(async move {
                    gate.acquire().await.unwrap().forget();
                    delivered.lock().unwrap().push(body);
                    let _ = reply.send(Ok("{}".into()));
                });
            }
            _ => panic!("unexpected callback operation"),
        }
    }
}

fn string(value: &eval::Value) -> String {
    let eval::Value::String(value) = value else {
        panic!("expected callback String")
    };
    value.to_string()
}

pub(super) async fn until(mut predicate: impl FnMut() -> bool) {
    tokio::time::timeout(std::time::Duration::from_secs(6), async {
        while !predicate() {
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    })
    .await
    .expect("fixture condition timed out");
}

pub(super) fn output(native: bool, index: usize) -> Value {
    if native {
        response(
            "final answer",
            vec![call(
                &format!("progress-finish-{index}"),
                "finish",
                json!({}),
            )],
        )
    } else {
        response("final answer<code>!done</code>", vec![])
    }
}

pub(super) async fn held_provider(native: bool, error: bool) -> (Provider, Arc<Semaphore>) {
    use axum::response::IntoResponse;
    let requests = Arc::new(Mutex::new(Vec::new()));
    let recorded = requests.clone();
    let gate = Arc::new(Semaphore::new(0));
    let held = gate.clone();
    let app = axum::Router::new().route(
        "/v1/chat/completions",
        axum::routing::post(move |axum::Json(body): axum::Json<Value>| {
            let (recorded, held) = (recorded.clone(), held.clone());
            async move {
                let index = {
                    let mut requests = recorded.lock().unwrap();
                    let index = requests.len();
                    requests.push(body.clone());
                    index
                };
                held.acquire().await.unwrap().forget();
                if error {
                    return (axum::http::StatusCode::BAD_REQUEST, "scripted error").into_response();
                }
                sse_or_json(&output(native, index), body["stream"] == true)
            }
        }),
    );
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let url = format!("http://{}", listener.local_addr().unwrap());
    let server = tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    (
        Provider {
            url,
            requests,
            server,
        },
        gate,
    )
}
