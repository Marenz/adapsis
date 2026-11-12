//! Trusted local replay data, not a remotely callable AST interface.
use crate::{ast, parser, validator};
use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};
pub(crate) mod limits;
#[cfg(test)]
mod tests;

#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
#[serde(tag = "format", rename_all = "lowercase", deny_unknown_fields)]
pub enum SourceOrigin {
    #[default]
    Adapsis,
    Python {
        module: String,
    },
}

impl SourceOrigin {
    pub fn is_adapsis(&self) -> bool {
        matches!(self, Self::Adapsis)
    }
}

#[derive(Debug, Clone)]
pub struct Submission {
    pub source: String,
    pub origin: SourceOrigin,
}

impl Submission {
    pub fn adapsis(source: &str) -> Self {
        Self {
            source: source.into(),
            origin: SourceOrigin::Adapsis,
        }
    }

    pub fn python(module: &str, source: &str) -> Self {
        Self {
            source: source.into(),
            origin: SourceOrigin::Python {
                module: module.into(),
            },
        }
    }

    pub fn check_recording(
        &self,
        operations: &[parser::Operation],
        meta: &super::SessionMeta,
    ) -> Result<()> {
        let SourceOrigin::Python { module } = &self.origin else {
            return Ok(());
        };
        if operations.iter().all(|op| {
            matches!(
                op,
                parser::Operation::Test(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Query(_)
            )
        }) {
            return Ok(());
        }
        if meta.revision != meta.sources.len() {
            bail!(
                "cannot record a Python mutation on rewound or inconsistent history; an explicit history branch policy is required"
            );
        }
        let [parser::Operation::Module(declaration)] = operations else {
            bail!("unsupported Python mutation replay shape")
        };
        if declaration.name != *module {
            bail!("Python mutation module mismatch")
        }
        Ok(())
    }

    pub fn preflight_record(
        &self,
        operations: &[parser::Operation],
        before: &ast::Program,
        meta: &super::SessionMeta,
    ) -> Result<()> {
        self.check_recording(operations, meta)?;
        let SourceOrigin::Python { module } = &self.origin else {
            return Ok(());
        };
        if self.source.len() > 1024 * 1024 {
            bail!("Python replay source exceeds 1 MiB")
        }
        if !operations
            .iter()
            .any(|op| matches!(op, parser::Operation::Module(_)))
        {
            return Ok(());
        }
        let [parser::Operation::Module(declaration)] = operations else {
            bail!("unsupported Python mutation replay shape")
        };
        if declaration.name != *module {
            bail!("Python mutation module mismatch")
        }
        crate::python_frontend::validate_operation_ir(operations)
            .map_err(|_| anyhow::anyhow!("invalid Python mutation IR"))?;
        let previous = before.modules.iter().find(|m| m.name == *module);
        if previous.is_some_and(|m| !m.modules.is_empty()) {
            bail!("nested Python checkpoints are unsupported")
        }
        // Pure validation preview only. A failure may leave a partial candidate,
        // just as actual application does; do not turn that into a transaction.
        let mut candidate = before.clone();
        let _ = validator::apply_and_validate(&mut candidate, &operations[0]);
        let prospective = candidate.modules.iter().find(|m| m.name == *module);
        // A retest may restore old bodies inside the candidate. Budget BOTH
        // complete states, which bounds any such combination before execution.
        limits::check(
            &serde_json::json!({"preview":[previous, prospective], "operations":operations,
            "source":self.source, "version":1, "format":"python", "module":module,
            "outcome":"partial", "checkpoint":null}),
        )
    }

    pub fn record(
        &self,
        operations: &[parser::Operation],
        before: &ast::Program,
        after: &ast::Program,
        success: bool,
    ) -> ReplayEntry {
        match &self.origin {
            SourceOrigin::Adapsis => ReplayEntry::Legacy(self.source.clone()),
            SourceOrigin::Python { module } => {
                let previous = before.modules.iter().find(|m| &m.name == module);
                let checkpoint = after.modules.iter().find(|m| &m.name == module).cloned();
                let outcome = if success {
                    Outcome::Accepted
                } else {
                    match (
                        serde_json::to_value(previous),
                        serde_json::to_value(&checkpoint),
                    ) {
                        (Ok(before), Ok(after)) if before == after => Outcome::Rejected,
                        _ => Outcome::Partial,
                    }
                };
                ReplayEntry::Python(PythonReplay {
                    version: 1,
                    format: "python".into(),
                    module: module.clone(),
                    source: self.source.clone(),
                    operations: operations.to_vec(),
                    checkpoint,
                    outcome,
                })
            }
        }
    }

    pub fn record_into(
        &self,
        meta: &mut super::SessionMeta,
        operations: &[parser::Operation],
        before: &ast::Program,
        after: &ast::Program,
        summary: String,
        success: bool,
    ) {
        let record = self.record(operations, before, after, success);
        meta.revision += 1;
        meta.mutations.push(super::MutationEntry {
            revision: meta.revision,
            timestamp: super::now(),
            source: self.source.clone(),
            origin: self.origin.clone(),
            summary,
            success,
        });
        meta.sources.push(record);
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
enum Outcome {
    Accepted,
    Rejected,
    Partial,
}

/// Existing Vec<String> sessions decode and reserialize as exactly those strings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum ReplayEntry {
    Legacy(String),
    Python(PythonReplay),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PythonReplay {
    version: u32,
    format: String,
    module: String,
    source: String,
    operations: Vec<parser::Operation>,
    checkpoint: Option<ast::Module>,
    outcome: Outcome,
}

impl ReplayEntry {
    pub fn operations(&self) -> Result<Vec<parser::Operation>> {
        match self {
            Self::Legacy(source) => {
                parser::parse(source).map_err(|_| anyhow::anyhow!("legacy replay parse failed"))
            }
            Self::Python(record) => {
                if record.source.len() > 1024 * 1024 {
                    bail!("Python replay source exceeds 1 MiB")
                }
                if record.version != 1 || record.format != "python" {
                    bail!("unsupported Python replay version/format");
                }
                limits::check(
                    &serde_json::to_value(record)
                        .map_err(|_| anyhow::anyhow!("invalid replay payload"))?,
                )?;
                let [parser::Operation::Module(module)] = record.operations.as_slice() else {
                    bail!("Python replay requires one declaration module");
                };
                if module.name != record.module {
                    bail!("Python replay module mismatch");
                }
                crate::python_frontend::validate_operation_ir(&record.operations)
                    .map_err(|_| anyhow::anyhow!("invalid Python replay operations"))?;
                if let Some(checkpoint) = &record.checkpoint {
                    if checkpoint.name != record.module
                        || checkpoint.id != record.module
                        || !checkpoint.modules.is_empty()
                    {
                        bail!("invalid Python checkpoint scope");
                    }
                    let mut names = std::collections::HashSet::new();
                    if checkpoint
                        .functions
                        .iter()
                        .any(|function| !names.insert(&function.name))
                    {
                        bail!("duplicate checkpoint function");
                    }
                    for function in &checkpoint.functions {
                        for test in &function.tests {
                            if let Some(typed) = &test.typed {
                                typed.case()?;
                            }
                        }
                    }
                } else if !matches!(record.outcome, Outcome::Rejected) {
                    bail!("missing Python checkpoint");
                }
                Ok(record.operations.clone())
            }
        }
    }

    pub fn kind(&self) -> &'static str {
        match self {
            Self::Legacy(_) => "adapsis",
            Self::Python(_) => "python",
        }
    }
}

/// Rebuild on fresh state. No Session apply, evaluation, library IO, startup or
/// historical test execution; validators construct declarations only.
pub fn reconstruct(
    sources: &[ReplayEntry],
    mutations: &[super::MutationEntry],
    revision: usize,
) -> Result<ast::Program> {
    if revision > sources.len() {
        bail!("revision does not exist");
    }
    let mut program = ast::Program::default();
    for (index, source) in sources[..revision].iter().enumerate() {
        let operations = source.operations().map_err(|_| {
            anyhow::anyhow!(
                "replay revision {} ({}): invalid source record",
                index + 1,
                source.kind()
            )
        })?;
        if let ReplayEntry::Python(record) = source {
            if let Some(checkpoint) = &record.checkpoint {
                if let Some(slot) = program.modules.iter_mut().find(|m| m.name == record.module) {
                    *slot = checkpoint.clone();
                } else {
                    program.modules.push(checkpoint.clone());
                }
            } else if program.modules.iter().any(|m| m.name == record.module) {
                bail!(
                    "replay revision {} (python): unsupported module deletion",
                    index + 1
                );
            }
            program.rebuild_function_index();
            continue; // #122: restore the ACTUAL outcome, never rerun rejected IR.
        }
        let ReplayEntry::Legacy(text) = source else {
            unreachable!()
        };
        let certain = mutations.get(index).is_some_and(|entry| {
            entry.success
                && entry.revision == index + 1
                && entry.source == *text
                && matches!(entry.origin, SourceOrigin::Adapsis)
        });
        if !certain {
            bail!(
                "replay revision {} (adapsis): historical outcome is not reconstructible",
                index + 1
            );
        }
        for operation in &operations {
            if matches!(
                operation,
                parser::Operation::Test(_)
                    | parser::Operation::Trace(_)
                    | parser::Operation::Eval(_)
                    | parser::Operation::Query(_)
            ) {
                continue;
            }
            validator::apply_and_validate(&mut program, operation).map_err(|_| {
                anyhow::anyhow!(
                    "replay revision {} ({}): declaration validation failed",
                    index + 1,
                    source.kind()
                )
            })?;
        }
    }
    program.rebuild_function_index();
    Ok(program)
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TypedTestCase {
    version: u32,
    case: parser::TestCase,
}

impl TypedTestCase {
    pub fn new(case: &parser::TestCase) -> Self {
        Self {
            version: 1,
            case: case.clone(),
        }
    }

    pub fn case(&self) -> Result<parser::TestCase> {
        if self.version != 1 {
            bail!("unsupported stored test version");
        }
        limits::check(
            &serde_json::to_value(&self.case)
                .map_err(|_| anyhow::anyhow!("invalid stored test payload"))?,
        )?;
        Ok(self.case.clone())
    }
}

pub fn validate_stored_tests(program: &ast::Program) -> Result<()> {
    let mut functions: Vec<_> = program.functions.iter().collect();
    let mut modules: Vec<_> = program.modules.iter().collect();
    while let Some(module) = modules.pop() {
        functions.extend(&module.functions);
        modules.extend(&module.modules);
    }
    for function in functions {
        for test in &function.tests {
            if let Some(typed) = &test.typed {
                typed.case()?;
            }
        }
    }
    Ok(())
}
