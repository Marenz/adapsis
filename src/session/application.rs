//! Legacy and typed submissions share these application steps and messages.
use super::*;
type Replies = Vec<(String, bool)>;

impl Session {
    pub(super) fn apply_control(
        &mut self,
        operation: &parser::Operation,
        replies: &mut Replies,
    ) -> bool {
        if let Some(message) = self.apply_mock_control(operation) {
            replies.push((message, true));
            return true;
        }
        match operation {
            parser::Operation::Plan(action) => self.apply_plan_control(action, replies),
            parser::Operation::Roadmap(action) => replies.push(self.handle_roadmap(action)),
            parser::Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                let route = ast::HttpRoute {
                    method: method.clone(),
                    path: path.clone(),
                    handler_fn: self.qualify_route_handler(handler_fn),
                };
                replies.push((self.add_route(route), true));
            }
            parser::Operation::RemoveRoute { method, path } => {
                match self.remove_route(method, path) {
                    Ok(message) => replies.push((message, true)),
                    Err(error) => replies.push((error.to_string(), false)),
                }
            }
            parser::Operation::Sandbox(action) => replies.push(self.handle_sandbox(action)),
            _ => return false,
        }
        true
    }

    fn apply_mock_control(&mut self, operation: &parser::Operation) -> Option<String> {
        match operation {
            parser::Operation::Mock {
                operation,
                patterns,
                response,
            } => {
                let display = patterns
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.meta.io_mocks.push(IoMock {
                    operation: operation.clone(),
                    patterns: patterns.clone(),
                    response: response.clone(),
                });
                Some(format!(
                    "mock: {operation} {display} -> \"{}\"",
                    response.chars().take(50).collect::<String>()
                ))
            }
            parser::Operation::Unmock => {
                let count = self.meta.io_mocks.len();
                self.meta.io_mocks.clear();
                Some(format!("cleared {count} mocks"))
            }
            parser::Operation::Stub {
                function_name,
                patterns,
                response_expr,
            } => {
                let display = patterns
                    .iter()
                    .map(|p| format!("\"{p}\""))
                    .collect::<Vec<_>>()
                    .join(" ");
                self.meta.function_stubs.push(FunctionStub {
                    function_name: function_name.clone(),
                    patterns: patterns.clone(),
                    response_expr: response_expr.clone(),
                });
                Some(format!(
                    "stub: {function_name} {display} -> {}",
                    response_expr.chars().take(60).collect::<String>()
                ))
            }
            parser::Operation::Unstub => {
                let count = self.meta.function_stubs.len();
                self.meta.function_stubs.clear();
                Some(format!("cleared {count} stubs"))
            }
            _ => None,
        }
    }

    fn apply_plan_control(&mut self, action: &parser::PlanAction, replies: &mut Replies) {
        match action {
            parser::PlanAction::Set(steps) => {
                self.meta.plan = steps
                    .iter()
                    .map(|description| PlanStep {
                        description: description.clone(),
                        status: PlanStatus::Pending,
                    })
                    .collect();
                replies.push((format!("Plan: {} steps", steps.len()), true));
            }
            parser::PlanAction::Progress(index) => {
                if let Some(step) = self.meta.plan.get_mut(index.saturating_sub(1)) {
                    step.status = PlanStatus::Done;
                    replies.push((format!("Step {index} done"), true));
                }
            }
            parser::PlanAction::Fail(index) => {
                if let Some(step) = self.meta.plan.get_mut(index.saturating_sub(1)) {
                    step.status = PlanStatus::Failed;
                    replies.push((format!("Step {index} failed"), true));
                }
            }
            parser::PlanAction::Show => {
                let plan = self
                    .meta
                    .plan
                    .iter()
                    .enumerate()
                    .map(|(i, step)| {
                        let icon = match step.status {
                            PlanStatus::Pending => "[ ]",
                            PlanStatus::InProgress => "[~]",
                            PlanStatus::Done => "[x]",
                            PlanStatus::Failed => "[!]",
                        };
                        format!("{} {}: {}", icon, i + 1, step.description)
                    })
                    .collect::<Vec<_>>()
                    .join("\n");
                replies.push((
                    if plan.is_empty() {
                        "No plan set".into()
                    } else {
                        format!("Plan:\n{plan}")
                    },
                    true,
                ));
            }
        }
    }

    pub(super) async fn apply_declaration(
        &mut self,
        operation: &parser::Operation,
        sender: Option<&tokio::sync::mpsc::Sender<crate::coroutine::IoRequest>>,
        replies: &mut Replies,
        typed: bool,
    ) {
        let backups = self.backup_affected_bodies(operation);
        match validator::apply_and_validate(&mut self.program, operation) {
            Ok(message) => {
                let routes: Vec<_> = self.program.pending_routes.drain(..).collect();
                for route in routes {
                    replies.push((self.add_route(route), true));
                }
                replies.push((message, true));
                for (_, passed, detail) in self.invalidate_and_retest(operation, backups) {
                    replies.push((detail, passed));
                }
                if let parser::Operation::Module(module) = operation {
                    for child in &module.body {
                        if let parser::Operation::Test(test) = child {
                            if sender.is_some() {
                                self.run_test_async(test, sender, replies, typed).await;
                            } else {
                                self.run_test_sync(test, replies, typed);
                            }
                        }
                    }
                }
            }
            Err(error) => replies.push((error.to_string(), false)),
        }
    }
}
