use anyhow::{anyhow, bail, Context, Result};

#[derive(Debug, Clone)]
pub enum PlanAction {
    Set(Vec<String>),
    Progress(usize),
    Fail(usize),
    Show,
}

#[derive(Debug, Clone)]
pub enum RoadmapAction {
    Show,
    Add(String),
    Done(usize),
    Remove(usize),
}

#[derive(Debug, Clone)]
pub enum Operation {
    Module(ModuleDecl),
    Type(TypeDecl),
    Function(FunctionDecl),
    Let(LetDecl),
    Set(SetDecl),
    Call(CallDecl),
    Check(CheckDecl),
    Branch(BranchDecl),
    If(IfDecl),
    Return(ReturnDecl),
    Each(EachDecl),
    While(WhileDecl),
    Match(MatchDecl),
    Await(AwaitDecl),
    Spawn(SpawnDecl),
    Replace(ReplaceMutation),
    Test(TestMutation),
    Trace(TraceMutation),
    Eval(EvalMutation),
    Move {
        function_names: Vec<String>,
        target_module: String,
    },
    Plan(PlanAction),
    Roadmap(RoadmapAction),
    /// Remove a function, type, or module: !remove Module.function or !remove Module
    Remove(String),
    /// Remove an HTTP route by method+path: !remove route POST /api/ask
    RemoveRoute {
        method: String,
        path: String,
    },
    Watch {
        function_name: String,
        args: String,
        interval_ms: u64,
    },
    Undo,
    Agent {
        name: String,
        scope: String,
        task: String,
    },
    OpenCode(String),
    /// Register an HTTP route: +route POST "/path" -> handler_fn
    Route {
        method: String,
        path: String,
        handler_fn: String,
    },
    /// Send a message to an agent: !msg <agent> <text>
    Message {
        to: String,
        content: String,
    },
    /// Register IO mock: !mock <operation> "<pattern>" ... -> "<response>"
    Mock {
        operation: String,
        /// One pattern per argument position; single-element for backward compat.
        patterns: Vec<String>,
        response: String,
    },
    /// Clear all mocks: !unmock
    Unmock,
    /// Signal task completion
    Done,
    /// Check inbox: ?inbox [agent_name]
    Query(String),
    /// Shared variable declaration: +shared name:Type = default_expr
    SharedVar(SharedVarDecl),
    /// Sandbox mode: !sandbox [enter|merge|discard|status]
    Sandbox(SandboxAction),
}

#[derive(Debug, Clone)]
pub enum SandboxAction {
    Enter,
    Merge,
    Discard,
    Status,
}

/// A shared variable declaration at the parser level.
#[derive(Debug, Clone)]
pub struct SharedVarDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub default: Expr,
}

#[derive(Debug, Clone)]
pub struct MatchDecl {
    pub expr: Expr,
    pub arms: Vec<MatchArmDecl>,
}

#[derive(Debug, Clone)]
pub struct MatchArmDecl {
    pub variant: String,
    pub bindings: Vec<String>,
    pub patterns: Option<Vec<MatchPatternDecl>>,
    pub body: Vec<Operation>,
}

/// Parsed pattern for nested matching inside +case arms.
#[derive(Debug, Clone)]
pub enum MatchPatternDecl {
    /// Simple binding: `x` or `_`
    Binding(String),
    /// Nested variant match: `Literal(x)` or `Add(Literal(0), y)`
    Variant {
        variant: String,
        sub_patterns: Vec<MatchPatternDecl>,
    },
    /// Literal value match: `0`, `true`, `"hello"`
    LiteralInt(i64),
    LiteralBool(bool),
    LiteralString(String),
}

#[derive(Debug, Clone)]
pub struct AwaitDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub call: Expr,
}

#[derive(Debug, Clone)]
pub struct SpawnDecl {
    pub call: Expr,
    /// Optional binding for the task handle: +spawn t:Int = func(args)
    pub binding: Option<(String, TypeExpr)>,
}

#[derive(Debug, Clone)]
pub struct SetDecl {
    pub name: String,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct WhileDecl {
    pub condition: Expr,
    pub body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct EvalMutation {
    pub function_name: String,
    pub input: Expr,
    /// When set, evaluate this expression directly instead of calling function_name.
    pub inline_expr: Option<Expr>,
}

#[derive(Debug, Clone)]
pub struct TraceMutation {
    pub function_name: String,
    pub input: Expr,
}

#[derive(Debug, Clone)]
pub struct ModuleDecl {
    pub name: String,
    pub body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct TypeDecl {
    pub name: String,
    pub body: TypeBody,
}

#[derive(Debug, Clone)]
pub enum TypeBody {
    Struct(Vec<FieldType>),
    Union(Vec<VariantType>),
    Alias(TypeExpr),
}

#[derive(Debug, Clone)]
pub struct FieldType {
    pub name: String,
    pub ty: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct VariantType {
    pub name: String,
    pub payload: Vec<TypeExpr>,
}

#[derive(Debug, Clone)]
pub struct FunctionDecl {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: TypeExpr,
    pub effects: Vec<String>,
    pub body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: TypeExpr,
}

#[derive(Debug, Clone)]
pub struct LetDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct CallDecl {
    pub name: String,
    pub ty: TypeExpr,
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct CheckDecl {
    pub name: String,
    pub expr: Expr,
    pub err_label: String,
}

#[derive(Debug, Clone)]
#[allow(dead_code)]
pub struct BranchDecl {
    pub name: String,
    pub expr: Expr,
    pub arms: Vec<BranchArm>,
}

#[derive(Debug, Clone)]
pub struct BranchArm {
    pub pattern: Expr,
    pub target: String,
}

#[derive(Debug, Clone)]
pub struct ReturnDecl {
    pub expr: Expr,
}

#[derive(Debug, Clone)]
pub struct EachDecl {
    pub collection: Expr,
    pub item: String,
    pub item_type: TypeExpr,
    pub body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct IfDecl {
    pub condition: Expr,
    pub then_body: Vec<Operation>,
    pub elif_branches: Vec<(Expr, Vec<Operation>)>,
    pub else_body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct ReplaceMutation {
    pub target: String,
    pub body: Vec<Operation>,
}

#[derive(Debug, Clone)]
pub struct TestMutation {
    pub function_name: String,
    pub cases: Vec<TestCase>,
}

#[derive(Debug, Clone)]
pub struct TestCase {
    pub input: Expr,
    pub expected: Expr,
    /// Optional matcher for flexible assertions (e.g. contains, starts_with, AnyOk, AnyErr).
    pub matcher: Option<TestMatcher>,
    /// Post-execution side-effect checks (e.g. routes/tasks/modules contain a value).
    pub after_checks: Vec<AfterCheck>,
}

/// Flexible test matchers — used instead of exact value comparison.
#[derive(Debug, Clone)]
pub enum TestMatcher {
    /// Result string contains substring
    Contains(String),
    /// Result string starts with prefix
    StartsWith(String),
    /// Result is any Ok value (without checking inner)
    AnyOk,
    /// Result is any Err value
    AnyErr,
    /// Result is Err with a specific message
    ErrContaining(String),
}

/// Post-execution side-effect check: `+after <target> <matcher> "<value>"`
#[derive(Debug, Clone)]
pub struct AfterCheck {
    /// What to inspect: "routes", "tasks", "modules", "mocks"
    pub target: String,
    /// How to check: "contains"
    pub matcher: String,
    /// The value to look for
    pub value: String,
}

#[derive(Debug, Clone)]
pub enum TypeExpr {
    Named(String),
    Generic { name: String, args: Vec<TypeExpr> },
    Struct(Vec<FieldType>),
}

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),
    Ident(String),
    FieldAccess {
        base: Box<Expr>,
        field: String,
    },
    Call {
        callee: Box<Expr>,
        args: Vec<Expr>,
    },
    Binary {
        op: BinaryOp,
        left: Box<Expr>,
        right: Box<Expr>,
    },
    Unary {
        op: UnaryOp,
        expr: Box<Expr>,
    },
    StructLiteral(Vec<FieldValue>),
    Cast {
        expr: Box<Expr>,
        #[allow(dead_code)]
        ty: TypeExpr,
    },
}

#[derive(Debug, Clone)]
pub struct FieldValue {
    pub name: String,
    pub value: Expr,
}

#[derive(Debug, Clone)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Gte,
    Lte,
    Eq,
    Neq,
    Gt,
    Lt,
    And,
    Or,
}

#[derive(Debug, Clone)]
pub enum UnaryOp {
    Not,
    Neg,
}

pub fn parse(source: &str) -> Result<Vec<Operation>> {
    Parser::new(source).parse()
}

struct Parser<'a> {
    lines: Vec<SourceLine<'a>>,
    index: usize,
}

#[derive(Clone, Copy)]
struct SourceLine<'a> {
    number: usize,
    indent: usize,
    text: &'a str,
}

impl<'a> Parser<'a> {
    fn new(source: &'a str) -> Self {
        let mut lines = Vec::new();

        for (idx, raw) in source.lines().enumerate() {
            let number = idx + 1;
            let trimmed = raw.trim();
            if trimmed.is_empty()
                || trimmed.starts_with("//")
                || trimmed.starts_with('#')
                || trimmed == "```"
                || trimmed.starts_with("```forge")
                || trimmed == "<code>"
                || trimmed == "</code>"
            {
                continue;
            }

            let indent = count_indent(raw);
            let text = raw.trim_start_matches([' ', '\t']);
            lines.push(SourceLine {
                number,
                indent,
                text,
            });
        }

        Self { lines, index: 0 }
    }

    fn parse(mut self) -> Result<Vec<Operation>> {
        let ops = self.parse_block(0, BlockMode::TopLevel)?;
        if let Some(line) = self.current() {
            bail!(
                "line {}: unexpected trailing input `{}`",
                line.number,
                line.text
            );
        }
        Ok(ops)
    }

    fn parse_block(&mut self, indent: usize, mode: BlockMode) -> Result<Vec<Operation>> {
        let mut ops = Vec::new();

        while let Some(line) = self.current() {
            // +end closes the current block — don't consume, let caller handle it
            if line.text == "end" || line.text == "+end" {
                if mode == BlockMode::TopLevel {
                    // Top-level: just stop, don't consume
                }
                break;
            }

            // In nested blocks, stop when indentation decreases below the block level
            if mode == BlockMode::Normal && line.indent < indent {
                break;
            }

            // Sibling keywords close the current sub-block without consuming
            if mode == BlockMode::Normal {
                if line.text.starts_with("+elif")
                    || line.text == "+else"
                    || line.text.starts_with("+case")
                {
                    break; // Don't consume — caller handles these
                }
            }

            ops.push(self.parse_operation(line.indent)?);
        }

        Ok(ops)
    }

    /// Consume a +end / end token if present. Returns true if consumed.
    fn consume_end(&mut self) -> bool {
        if let Some(line) = self.current() {
            if line.text == "+end" || line.text == "end" {
                self.index += 1;
                return true;
            }
        }
        false
    }

    fn parse_operation(&mut self, indent: usize) -> Result<Operation> {
        let line = self.current().context("internal parser state error")?;
        let text = line.text;

        // !module Name — state change, everything after goes into this module
        if let Some(rest) = text
            .strip_prefix("!module")
            .or_else(|| text.strip_prefix("+module"))
        {
            let name = rest.trim();
            if name.is_empty() {
                bail!("line {}: expected module name", line.number);
            }
            self.index += 1;
            // Collect +fn and +type operations until next !module, ! command, or EOF
            let mut body = Vec::new();
            while let Some(next) = self.current() {
                if next.text.starts_with("!module") || next.text.starts_with("+module") {
                    break;
                }
                if next.text == "+end" || next.text == "end" {
                    self.index += 1; // Skip optional +end (backward compat)
                    break;
                }
                // Only +fn, +type, +shared belong inside a module — everything else is top-level
                if !next.text.starts_with("+fn")
                    && !next.text.starts_with("+type")
                    && !next.text.starts_with("+shared")
                {
                    break;
                }
                body.push(self.parse_operation(next.indent)?);
            }
            return Ok(Operation::Module(ModuleDecl {
                name: name.to_string(),
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("+type") {
            let mut type_text = rest.trim().to_string();
            self.index += 1;
            // Multiline types: join continuation lines until blank line or new +operation
            while let Some(next) = self.current() {
                let t = next.text.trim();
                if t.is_empty()
                    || t.starts_with('+')
                    || t.starts_with('!')
                    || t.starts_with('?')
                    || t == "end"
                {
                    break;
                }
                type_text.push_str(", ");
                type_text.push_str(t.trim_start_matches(',').trim());
                self.index += 1;
            }
            let decl = parse_type_decl(line.number, &type_text)?;
            return Ok(Operation::Type(decl));
        }

        if let Some(rest) = text.strip_prefix("+fn") {
            let header = parse_function_header(line.number, rest.trim())?;
            self.index += 1;
            let body = self.parse_nested_block(indent)?;
            self.consume_end();
            return Ok(Operation::Function(FunctionDecl {
                name: header.name,
                params: header.params,
                return_type: header.return_type,
                effects: header.effects,
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("+route") {
            // Format: +route METHOD "path" -> handler_fn
            // e.g.    +route POST "/webhook/telegram" -> handle_telegram
            let rest = rest.trim();
            let (method, rest) = rest.split_once(char::is_whitespace).ok_or_else(|| {
                anyhow!(
                    "line {}: expected `+route METHOD \"path\" -> handler_fn`",
                    line.number
                )
            })?;
            let rest = rest.trim();
            // Parse quoted path
            let path = if rest.starts_with('"') {
                let end = rest[1..].find('"').ok_or_else(|| {
                    anyhow!("line {}: unterminated string in +route path", line.number)
                })?;
                let p = &rest[1..1 + end];
                (p, &rest[2 + end..])
            } else {
                let end = rest.find(char::is_whitespace).ok_or_else(|| {
                    anyhow!(
                        "line {}: expected path and -> handler in +route",
                        line.number
                    )
                })?;
                (&rest[..end], &rest[end..])
            };
            let rest = path.1.trim();
            let handler_fn = rest
                .strip_prefix("->")
                .ok_or_else(|| anyhow!("line {}: expected `->` after path in +route", line.number))?
                .trim();
            if handler_fn.is_empty() {
                bail!(
                    "line {}: expected handler function name after `->` in +route",
                    line.number
                );
            }
            self.index += 1;
            return Ok(Operation::Route {
                method: method.to_uppercase(),
                path: path.0.to_string(),
                handler_fn: handler_fn.to_string(),
            });
        }

        if let Some(rest) = text.strip_prefix("+let") {
            let decl = parse_binding_decl(line.number, rest.trim(), false)?;
            self.index += 1;
            return Ok(Operation::Let(decl));
        }

        if let Some(rest) = text.strip_prefix("+shared") {
            // Format: +shared name:Type = default_expr
            let rest = rest.trim();
            let (name_type, expr_text) = rest.split_once('=').ok_or_else(|| {
                anyhow!(
                    "line {}: expected `name:Type = expr` in +shared",
                    line.number
                )
            })?;
            let name_type = name_type.trim();
            let (name, type_text) = name_type
                .split_once(':')
                .ok_or_else(|| anyhow!("line {}: expected `name:Type` in +shared", line.number))?;
            let ty = parse_type(line.number, type_text.trim())?;
            let default = parse_expr(line.number, expr_text.trim())?;
            self.index += 1;
            return Ok(Operation::SharedVar(SharedVarDecl {
                name: name.trim().to_string(),
                ty,
                default,
            }));
        }

        if let Some(rest) = text.strip_prefix("+set") {
            let rest = rest.trim();
            // Format: +set name = expr
            let (name, expr_text) = rest
                .split_once('=')
                .ok_or_else(|| anyhow!("line {}: expected `name = expr` in +set", line.number))?;
            let expr = parse_expr(line.number, expr_text.trim())?;
            self.index += 1;
            return Ok(Operation::Set(SetDecl {
                name: name.trim().to_string(),
                expr,
            }));
        }

        if let Some(rest) = text.strip_prefix("+while") {
            let condition = parse_expr(line.number, rest.trim())?;
            self.index += 1;
            let body = self.parse_nested_block(indent)?;
            self.consume_end();
            return Ok(Operation::While(WhileDecl { condition, body }));
        }

        if let Some(rest) = text.strip_prefix("+match") {
            let expr = parse_expr(line.number, rest.trim())?;
            self.index += 1;

            // Parse +case arms until +end
            let mut arms = Vec::new();
            loop {
                let next = match self.current() {
                    Some(l) => l,
                    _ => break,
                };
                if next.text == "+end" || next.text == "end" {
                    self.index += 1;
                    break;
                }
                if let Some(case_rest) = next.text.strip_prefix("+case") {
                    let case_rest = case_rest.trim();
                    let arm = parse_case_pattern(next.number, case_rest)?;
                    self.index += 1;
                    let body = self.parse_nested_block(indent)?;
                    arms.push(MatchArmDecl {
                        variant: arm.0,
                        bindings: arm.1,
                        patterns: arm.2,
                        body,
                    });
                } else {
                    break;
                }
            }

            return Ok(Operation::Match(MatchDecl { expr, arms }));
        }

        if let Some(rest) = text.strip_prefix("+await") {
            // Format: +await name:Type = func(args)
            let decl = parse_binding_decl(line.number, rest.trim(), false)?;
            self.index += 1;
            return Ok(Operation::Await(AwaitDecl {
                name: decl.name,
                ty: decl.ty,
                call: decl.expr,
            }));
        }

        if let Some(rest) = text.strip_prefix("+spawn") {
            let rest = rest.trim();
            // Check for binding form: +spawn name:Type = func(args)
            if rest.contains('=') && !rest.starts_with('(') {
                let decl = parse_binding_decl(line.number, rest, false)?;
                self.index += 1;
                return Ok(Operation::Spawn(SpawnDecl {
                    call: decl.expr,
                    binding: Some((decl.name, decl.ty)),
                }));
            }
            let expr = parse_expr(line.number, rest)?;
            self.index += 1;
            return Ok(Operation::Spawn(SpawnDecl {
                call: expr,
                binding: None,
            }));
        }

        if let Some(rest) = text.strip_prefix("+call") {
            let decl = parse_call_decl(line.number, rest.trim())?;
            self.index += 1;
            return Ok(Operation::Call(decl));
        }

        if let Some(rest) = text.strip_prefix("+check") {
            let decl = parse_check_decl(line.number, rest.trim())?;
            self.index += 1;
            return Ok(Operation::Check(decl));
        }

        if let Some(rest) = text.strip_prefix("+branch") {
            let decl = parse_branch_decl(line.number, rest.trim())?;
            self.index += 1;
            return Ok(Operation::Branch(decl));
        }

        if let Some(rest) = text.strip_prefix("+if") {
            let condition = parse_expr(line.number, rest.trim())?;
            self.index += 1;
            let then_body = self.parse_nested_block(indent)?;

            let mut elif_branches = Vec::new();
            let mut else_body = Vec::new();

            // Parse +elif and +else — they close the previous branch body
            loop {
                let next = match self.current() {
                    Some(l) => l,
                    _ => break,
                };
                if let Some(rest) = next.text.strip_prefix("+elif") {
                    let elif_cond = parse_expr(next.number, rest.trim())?;
                    self.index += 1;
                    let elif_body = self.parse_nested_block(indent)?;
                    elif_branches.push((elif_cond, elif_body));
                } else if next.text == "+else" {
                    self.index += 1;
                    else_body = self.parse_nested_block(indent)?;
                    break;
                } else {
                    break;
                }
            }
            self.consume_end();

            return Ok(Operation::If(IfDecl {
                condition,
                then_body,
                elif_branches,
                else_body,
            }));
        }

        if let Some(rest) = text.strip_prefix("+return") {
            let expr_text = rest.trim();
            if expr_text.is_empty() {
                bail!("line {}: expected return expression", line.number);
            }
            let expr = parse_expr(line.number, expr_text)?;
            self.index += 1;
            return Ok(Operation::Return(ReturnDecl { expr }));
        }

        if let Some(rest) = text.strip_prefix("+each") {
            let (collection, item, item_type) = parse_each_header(line.number, rest.trim())?;
            self.index += 1;
            let body = self.parse_nested_block(indent)?;
            self.consume_end();
            return Ok(Operation::Each(EachDecl {
                collection,
                item,
                item_type,
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("!replace") {
            let target = rest.trim();
            if target.is_empty() {
                bail!("line {}: expected replace target", line.number);
            }
            self.index += 1;
            let body = self.parse_nested_block(indent)?;
            self.consume_end();
            return Ok(Operation::Replace(ReplaceMutation {
                target: target.to_string(),
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("!test") {
            let function_name = rest.trim();
            if function_name.is_empty() {
                bail!("line {}: expected function name after !test", line.number);
            }
            self.index += 1;
            let cases = self.parse_test_cases(indent)?;
            return Ok(Operation::Test(TestMutation {
                function_name: function_name.to_string(),
                cases,
            }));
        }

        if let Some(rest) = text.strip_prefix("!move") {
            // !move fn1 fn2 fn3 ModuleName (last arg is module)
            let parts: Vec<&str> = rest.trim().split_whitespace().collect();
            if parts.len() < 2 {
                bail!(
                    "line {}: expected `!move function_name(s) ModuleName`",
                    line.number
                );
            }
            let target_module = parts.last().unwrap().to_string();
            let function_names = parts[..parts.len() - 1]
                .iter()
                .map(|s| s.to_string())
                .collect();
            self.index += 1;
            return Ok(Operation::Move {
                function_names,
                target_module,
            });
        }

        if let Some(rest) = text.strip_prefix("!plan") {
            let rest = rest.trim();
            self.index += 1;
            if rest.is_empty() {
                return Ok(Operation::Plan(PlanAction::Show));
            } else if let Some(steps) = rest.strip_prefix("set") {
                // Consume all following lines as plan steps
                let mut plan_steps: Vec<String> = steps
                    .trim()
                    .split('\n')
                    .map(|s| {
                        let s = s.trim().trim_matches('"').trim_start_matches("- ");
                        // Strip leading "N. " or "N) " numbering
                        if s.len() > 1 && s.as_bytes()[0].is_ascii_digit() {
                            let rest = s.trim_start_matches(|c: char| c.is_ascii_digit());
                            if let Some(r) =
                                rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))
                            {
                                return r.to_string();
                            }
                        }
                        s.to_string()
                    })
                    .filter(|s| !s.is_empty())
                    .collect();
                while let Some(next) = self.current() {
                    let t = next.text.trim();
                    if t.starts_with('+')
                        || t.starts_with('!')
                        || t.starts_with('?')
                        || t.is_empty()
                    {
                        break;
                    }
                    let step = t.trim_matches('"').trim_start_matches("- ");
                    // Strip leading "N. " or "N) " numbering — steps are auto-numbered
                    let step = if step.len() > 1 && step.as_bytes()[0].is_ascii_digit() {
                        let rest = step.trim_start_matches(|c: char| c.is_ascii_digit());
                        if let Some(r) = rest.strip_prefix(". ").or_else(|| rest.strip_prefix(") "))
                        {
                            r
                        } else {
                            step
                        }
                    } else {
                        step
                    };
                    plan_steps.push(step.to_string());
                    self.index += 1;
                }
                return Ok(Operation::Plan(PlanAction::Set(plan_steps)));
            } else if let Some(n) = rest.strip_prefix("done") {
                let n: usize = n.trim().parse().unwrap_or(1);
                return Ok(Operation::Plan(PlanAction::Progress(n)));
            } else if let Some(n) = rest.strip_prefix("fail") {
                let n: usize = n.trim().parse().unwrap_or(1);
                return Ok(Operation::Plan(PlanAction::Fail(n)));
            } else {
                // Treat as set with single step
                return Ok(Operation::Plan(PlanAction::Set(vec![rest.to_string()])));
            }
        }

        if text == "!done" {
            self.index += 1;
            return Ok(Operation::Done);
        }

        if let Some(rest) = text.strip_prefix("!remove") {
            let rest_trimmed = rest.trim();
            // Check for route removal: !remove route METHOD /path
            if let Some(route_rest) = rest_trimmed.strip_prefix("route") {
                let route_rest = route_rest.trim();
                let parts: Vec<&str> = route_rest.splitn(2, char::is_whitespace).collect();
                if parts.len() != 2 || parts[0].is_empty() || parts[1].trim().is_empty() {
                    bail!(
                        "line {}: !remove route requires METHOD and path, e.g. !remove route POST /api/ask",
                        line.number
                    );
                }
                let method = parts[0].to_uppercase();
                let path = parts[1].trim().to_string();
                self.index += 1;
                return Ok(Operation::RemoveRoute { method, path });
            }
            let target = rest_trimmed.to_string();
            if target.is_empty() {
                bail!(
                    "line {}: !remove requires a target (Module.function, Module, TypeName, or route METHOD /path)",
                    line.number
                );
            }
            self.index += 1;
            return Ok(Operation::Remove(target));
        }

        if let Some(rest) = text.strip_prefix("!unroute") {
            let rest = rest.trim();
            let parts: Vec<&str> = rest.splitn(2, char::is_whitespace).collect();
            if parts.len() != 2 || parts[0].is_empty() || parts[1].trim().is_empty() {
                bail!(
                    "line {}: !unroute requires METHOD and path, e.g. !unroute GET /path",
                    line.number
                );
            }
            let method = parts[0].to_uppercase();
            let path = parts[1].trim().to_string();
            self.index += 1;
            return Ok(Operation::RemoveRoute { method, path });
        }

        if let Some(rest) = text.strip_prefix("!roadmap") {
            let rest = rest.trim();
            self.index += 1;
            if rest.is_empty() || rest == "show" {
                return Ok(Operation::Roadmap(RoadmapAction::Show));
            } else if let Some(desc) = rest.strip_prefix("add") {
                return Ok(Operation::Roadmap(RoadmapAction::Add(
                    desc.trim().to_string(),
                )));
            } else if let Some(n) = rest.strip_prefix("done") {
                let n: usize = n.trim().parse().unwrap_or(1);
                return Ok(Operation::Roadmap(RoadmapAction::Done(n)));
            } else if let Some(n) = rest.strip_prefix("remove") {
                let n: usize = n.trim().parse().unwrap_or(1);
                return Ok(Operation::Roadmap(RoadmapAction::Remove(n)));
            } else {
                // Treat as add
                return Ok(Operation::Roadmap(RoadmapAction::Add(rest.to_string())));
            }
        }

        if let Some(rest) = text.strip_prefix("!watch") {
            // !watch function_name args interval_ms
            let parts: Vec<&str> = rest.trim().rsplitn(2, ' ').collect();
            if parts.len() >= 2 {
                let interval_ms: u64 = parts[0].trim().parse().unwrap_or(60000);
                let func_and_args = parts[1].trim();
                let (func, args) = func_and_args.split_once(' ').unwrap_or((func_and_args, ""));
                self.index += 1;
                return Ok(Operation::Watch {
                    function_name: func.to_string(),
                    args: args.to_string(),
                    interval_ms,
                });
            } else {
                bail!(
                    "line {}: expected !watch <function> [args] <interval_ms>",
                    line.number
                );
            }
        }

        if text == "!undo" {
            self.index += 1;
            return Ok(Operation::Undo);
        }

        if let Some(rest) = text.strip_prefix("!agent") {
            // !agent name --scope "scope" task description
            // !agent test --scope read-only write tests for all functions
            // !agent refactor --scope "module Crypto" reorganize the crypto code
            let rest = rest.trim();
            let (name, rest) = rest.split_once(' ').ok_or_else(|| {
                anyhow!(
                    "line {}: expected !agent <name> [--scope <scope>] <task>",
                    line.number
                )
            })?;
            let rest = rest.trim();
            let (scope, task) = if rest.starts_with("--scope") {
                let after_flag = rest.strip_prefix("--scope").unwrap().trim();
                // Scope might be quoted: --scope "module Crypto" or bare: --scope read-only
                if after_flag.starts_with('"') {
                    let close = after_flag[1..]
                        .find('"')
                        .ok_or_else(|| anyhow!("line {}: unterminated scope quote", line.number))?;
                    let scope = &after_flag[1..close + 1];
                    let task = after_flag[close + 2..].trim();
                    (scope.to_string(), task.to_string())
                } else {
                    let (scope, task) = after_flag.split_once(' ').unwrap_or((after_flag, ""));
                    (scope.to_string(), task.trim().to_string())
                }
            } else {
                ("full".to_string(), rest.to_string())
            };

            // Consume continuation lines (like !opencode)
            self.index += 1;
            let mut full_task = task;
            while let Some(next) = self.current() {
                let t = next.text.trim();
                if t.starts_with("+fn ")
                    || t.starts_with("+type ")
                    || t.starts_with("!test ")
                    || t.starts_with("!eval ")
                    || t.starts_with("?")
                    || t.starts_with("+module ")
                {
                    break;
                }
                full_task.push('\n');
                full_task.push_str(next.text);
                self.index += 1;
            }

            return Ok(Operation::Agent {
                name: name.to_string(),
                scope,
                task: full_task.trim().to_string(),
            });
        }

        if let Some(rest) = text.strip_prefix("!msg") {
            // !msg <agent_name> <message content>
            let rest = rest.trim();
            let (to, content) = rest
                .split_once(' ')
                .ok_or_else(|| anyhow!("line {}: expected !msg <agent> <message>", line.number))?;
            self.index += 1;
            return Ok(Operation::Message {
                to: to.to_string(),
                content: content.trim().to_string(),
            });
        }

        if text.trim() == "!unmock" {
            self.index += 1;
            return Ok(Operation::Unmock);
        }

        if let Some(rest) = text.strip_prefix("!sandbox") {
            let sub = rest.trim();
            let action = match sub {
                "merge" => SandboxAction::Merge,
                "discard" => SandboxAction::Discard,
                "status" => SandboxAction::Status,
                "" | "enter" => SandboxAction::Enter,
                other => bail!("line {}: unknown sandbox action `{other}`, expected enter/merge/discard/status", line.number),
            };
            self.index += 1;
            return Ok(Operation::Sandbox(action));
        }

        if let Some(rest) = text.strip_prefix("!mock") {
            // !mock http_get "https://..." -> "response"
            let rest = rest.trim();
            let (operation, rest) = rest.split_once(' ').ok_or_else(|| {
                anyhow!(
                    "line {}: expected !mock <operation> \"<pattern>\" -> \"<response>\"",
                    line.number
                )
            })?;
            let mut rest = rest.trim();
            // Parse: "pattern1" "pattern2" ... -> "response"
            // Collect all quoted strings before the '->' as patterns.
            let mut patterns = Vec::new();
            loop {
                let trimmed = rest.trim();
                if trimmed.starts_with("->") {
                    rest = trimmed;
                    break;
                }
                if !trimmed.starts_with('"') {
                    bail!(
                        "line {}: expected quoted string or '->' in !mock, got `{}`",
                        line.number,
                        &trimmed[..trimmed.len().min(20)]
                    );
                }
                let (pat, remaining) = parse_mock_string(line.number, trimmed)?;
                patterns.push(pat);
                rest = remaining;
            }
            if patterns.is_empty() {
                bail!(
                    "line {}: expected at least one pattern string in !mock",
                    line.number
                );
            }
            let rest = rest.strip_prefix("->").ok_or_else(|| {
                anyhow!(
                    "line {}: expected '->' between pattern and response in !mock",
                    line.number
                )
            })?;
            let rest = rest.trim();
            let (response_part, _) = parse_mock_string(line.number, rest)?;
            self.index += 1;
            return Ok(Operation::Mock {
                operation: operation.to_string(),
                patterns,
                response: response_part,
            });
        }

        if let Some(rest) = text.strip_prefix("!opencode") {
            // Consume ALL following lines until end of input or a Forge operation
            let mut description = rest.trim().to_string();
            self.index += 1;
            while let Some(next) = self.current() {
                let t = next.text.trim();
                // Stop at Forge operations (lines starting with + ! ? that aren't bullets)
                if t.starts_with("+fn ")
                    || t.starts_with("+type ")
                    || t.starts_with("+module ")
                    || t.starts_with("+let ")
                    || t.starts_with("+call ")
                    || t.starts_with("+return")
                    || t.starts_with("!test ")
                    || t.starts_with("!eval ")
                    || t.starts_with("!move ")
                    || t.starts_with("?")
                {
                    break;
                }
                description.push('\n');
                description.push_str(next.text);
                self.index += 1;
            }
            return Ok(Operation::OpenCode(description.trim().to_string()));
        }

        if let Some(rest) = text.strip_prefix("!trace") {
            let rest = rest.trim();
            // Parse: !trace function_name {input_expr}
            let (fn_name, input_text) = rest.split_once(' ').unwrap_or((rest, "{}"));
            let input = parse_expr(line.number, input_text.trim())?;
            self.index += 1;
            return Ok(Operation::Trace(TraceMutation {
                function_name: fn_name.to_string(),
                input,
            }));
        }

        if let Some(rest) = text.strip_prefix("!eval") {
            let rest = rest.trim();

            // First, try to parse the entire argument as an inline expression.
            // This handles cases like: !eval 1 + 2, !eval concat("a", "b"), !eval len("hello")
            // We accept it as inline if it parses as a complete expression AND is not
            // a bare identifier (which would be the existing "!eval func_name" syntax).
            if !rest.is_empty() {
                if let Ok(expr) = parse_expr(line.number, rest) {
                    let is_bare_ident = matches!(&expr, Expr::Ident(_));
                    if !is_bare_ident {
                        self.index += 1;
                        return Ok(Operation::Eval(EvalMutation {
                            function_name: String::new(),
                            input: Expr::StructLiteral(vec![]),
                            inline_expr: Some(expr),
                        }));
                    }
                }
            }

            // Fall back to current behavior: function_name + arguments
            let (fn_name, input_text) = rest.split_once(' ').unwrap_or((rest, ""));
            let input = if input_text.trim().is_empty() {
                Expr::StructLiteral(vec![])
            } else {
                parse_test_input(line.number, input_text.trim())?
            };
            self.index += 1;
            return Ok(Operation::Eval(EvalMutation {
                function_name: fn_name.to_string(),
                input,
                inline_expr: None,
            }));
        }

        if text.starts_with('?') {
            // Semantic query — pass through as-is
            let query = text.to_string();
            self.index += 1;
            return Ok(Operation::Query(query));
        }

        // Treat unknown ! operations as queries (AI often uses !symbols instead of ?symbols)
        if text.starts_with('!') {
            let as_query = format!("?{}", &text[1..]);
            eprintln!(
                "[parser] line {}: treating `{}` as `{}`",
                line.number, text, as_query
            );
            self.index += 1;
            return Ok(Operation::Query(as_query));
        }
        // Skip truly unknown operations
        eprintln!(
            "[parser] line {}: skipping unknown `{}`",
            line.number, line.text
        );
        self.index += 1;
        Ok(Operation::Query(format!(
            "WARNING: unknown operation `{}` (skipped)",
            line.text
        )))
    }

    fn parse_test_cases(&mut self, _parent_indent: usize) -> Result<Vec<TestCase>> {
        let mut cases = Vec::new();

        while let Some(line) = self.current() {
            let trimmed = line.text.trim();
            if !trimmed.starts_with("+with") {
                break;
            }

            let rest = line
                .text
                .strip_prefix("+with")
                .ok_or_else(|| anyhow!("line {}: expected `+with` test case", line.number))?
                .trim();

            let (input_text, expected_text) = split_test_case(line.number, rest)?;
            let input = parse_test_input(line.number, input_text.trim())?;
            let expected_trimmed = expected_text.trim();

            // Check for matcher syntax in expected position
            let (expected, matcher) = parse_expected_with_matcher(line.number, expected_trimmed)?;

            self.index += 1;

            // Collect any +after lines that follow this +with
            let mut after_checks = Vec::new();
            while let Some(next) = self.current() {
                let next_trimmed = next.text.trim();
                if let Some(after_rest) = next_trimmed.strip_prefix("+after") {
                    let after_rest = after_rest.trim();
                    let check = parse_after_check(next.number, after_rest)?;
                    after_checks.push(check);
                    self.index += 1;
                } else {
                    break;
                }
            }

            cases.push(TestCase {
                input,
                expected,
                matcher,
                after_checks,
            });
        }

        if cases.is_empty() {
            let line = self
                .lines
                .get(self.index.saturating_sub(1))
                .map(|line| line.number)
                .unwrap_or(1);
            bail!("line {}: !test requires at least one `+with` case", line);
        }

        Ok(cases)
    }

    fn parse_nested_block(&mut self, parent_indent: usize) -> Result<Vec<Operation>> {
        let Some(indent) = self.child_indent(parent_indent) else {
            return Ok(Vec::new());
        };
        self.parse_block(indent, BlockMode::Normal)
    }

    fn child_indent(&self, parent_indent: usize) -> Option<usize> {
        let line = self.current()?;
        if line.indent > parent_indent {
            Some(line.indent)
        } else {
            None
        }
    }

    fn current(&self) -> Option<SourceLine<'a>> {
        self.lines.get(self.index).copied()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BlockMode {
    TopLevel,
    Normal,
}

struct FunctionHeader {
    name: String,
    params: Vec<Param>,
    return_type: TypeExpr,
    effects: Vec<String>,
}

fn parse_type_decl(line: usize, input: &str) -> Result<TypeDecl> {
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected type name", line))?;
    let rest = rest.trim();
    let value = rest
        .strip_prefix('=')
        .ok_or_else(|| anyhow!("line {}: expected `=` in type declaration", line))?
        .trim();

    // Strip optional braces — accept both {a:Int, b:String} and a:Int, b:String
    let value = if value.starts_with('{') && value.ends_with('}') {
        &value[1..value.len() - 1]
    } else {
        value
    };

    if value.contains('|') {
        let mut variants = Vec::new();
        for part in split_top_level(value, '|') {
            let part = part.trim();
            if part.is_empty() {
                bail!("line {}: empty union variant", line);
            }

            if let Some(open) = find_matching_open_paren_variant(part) {
                let close = part
                    .rfind(')')
                    .ok_or_else(|| anyhow!("line {}: malformed union variant `{}`", line, part))?;
                if close != part.len() - 1 {
                    bail!("line {}: malformed union variant `{}`", line, part);
                }
                let variant_name = part[..open].trim();
                let payload_text = part[open + 1..close].trim();
                if variant_name.is_empty() {
                    bail!("line {}: missing variant name", line);
                }
                let payload = if payload_text.is_empty() {
                    vec![]
                } else {
                    // Parse comma-separated types
                    payload_text
                        .split(',')
                        .map(|s| parse_type(line, s.trim()))
                        .collect::<Result<Vec<_>>>()?
                };
                variants.push(VariantType {
                    name: variant_name.to_string(),
                    payload,
                });
            } else {
                variants.push(VariantType {
                    name: part.to_string(),
                    payload: vec![],
                });
            }
        }

        return Ok(TypeDecl {
            name: name.to_string(),
            body: TypeBody::Union(variants),
        });
    }

    // Check if it looks like struct fields: name:Type, name:Type
    if value.contains(':') {
        let fields = value
            .split(',')
            .map(|f| {
                let f = f.trim();
                let (fname, ftype) = f.split_once(':').ok_or_else(|| {
                    anyhow!("line {}: expected field:Type in struct, got `{f}`", line)
                })?;
                Ok(FieldType {
                    name: fname.trim().to_string(),
                    ty: parse_type(line, ftype.trim())?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(TypeDecl {
            name: name.to_string(),
            body: TypeBody::Struct(fields),
        });
    }

    Ok(TypeDecl {
        name: name.to_string(),
        body: TypeBody::Alias(parse_type(line, value)?),
    })
}

fn parse_function_header(line: usize, input: &str) -> Result<FunctionHeader> {
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected function name", line))?;
    // Support Module.function_name — strip the module prefix, keep just the function name
    let (name, rest) = if rest.starts_with('.') {
        let after_dot = &rest[1..];
        let (fn_name, rest2) = take_ident(after_dot)
            .ok_or_else(|| anyhow!("line {}: expected function name after '.'", line))?;
        (fn_name, rest2)
    } else {
        (name, rest)
    };
    let rest = rest.trim();

    let params_start = rest
        .find('(')
        .ok_or_else(|| anyhow!("line {}: expected parameter list for function declaration. If this is inside a test expectation, your expected string may contain newlines that split across lines — use a shorter expectation or test specific parts", line))?;
    if params_start != 0 {
        bail!("line {}: unexpected text before parameter list", line);
    }

    let params_end = find_matching_delim(rest, 0, '(', ')')
        .ok_or_else(|| anyhow!("line {}: unclosed parameter list", line))?;
    let params_text = &rest[1..params_end];
    let mut tail = rest[params_end + 1..].trim();
    tail = tail
        .strip_prefix("->")
        .ok_or_else(|| anyhow!("line {}: expected `->` after parameters", line))?
        .trim();

    let (return_text, effects_text) = split_effects(tail);
    if return_text.trim().is_empty() {
        bail!("line {}: expected return type", line);
    }

    Ok(FunctionHeader {
        name: name.to_string(),
        params: parse_params(line, params_text)?,
        return_type: parse_type(line, return_text.trim())?,
        effects: parse_effects(line, effects_text)?,
    })
}

fn parse_params(line: usize, input: &str) -> Result<Vec<Param>> {
    if input.trim().is_empty() {
        return Ok(Vec::new());
    }

    split_top_level(input, ',')
        .into_iter()
        .map(|part| {
            let part = part.trim();
            let (name, ty_text) = split_once_required(line, part, ":")?;
            Ok(Param {
                name: name.trim().to_string(),
                ty: parse_type(line, ty_text.trim())?,
            })
        })
        .collect()
}

fn parse_effects(line: usize, input: Option<&str>) -> Result<Vec<String>> {
    let Some(input) = input else {
        return Ok(Vec::new());
    };

    let inner = input.trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }

    split_top_level(inner, ',')
        .into_iter()
        .map(|effect| {
            let effect = effect.trim();
            if effect.is_empty() {
                bail!("line {}: empty effect name", line);
            }
            Ok(effect.to_string())
        })
        .collect()
}

fn parse_binding_decl(line: usize, input: &str, _allow_call_expr: bool) -> Result<LetDecl> {
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected binding name", line))?;
    let rest = rest.trim();
    let rest = rest
        .strip_prefix(':')
        .ok_or_else(|| anyhow!("line {}: expected `:` after binding name", line))?
        .trim();
    let (type_text, expr_text) = split_once_required(line, rest, "=")?;
    Ok(LetDecl {
        name: name.to_string(),
        ty: parse_type(line, type_text.trim())?,
        expr: parse_expr(line, expr_text.trim())?,
    })
}

fn parse_call_decl(line: usize, input: &str) -> Result<CallDecl> {
    let let_decl = parse_binding_decl(line, input, true)?;
    Ok(CallDecl {
        name: let_decl.name,
        ty: let_decl.ty,
        expr: let_decl.expr,
    })
}

fn parse_check_decl(line: usize, input: &str) -> Result<CheckDecl> {
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected check name", line))?;
    let rest = rest.trim();

    // Find the ~err_label at the end. Search from the right to handle complex expressions.
    if let Some(tilde_pos) = rest.rfind('~') {
        let expr_text = rest[..tilde_pos].trim();
        let err_label = rest[tilde_pos + 1..].trim();
        if err_label.is_empty() {
            bail!("line {}: expected error label after `~`", line);
        }
        if expr_text.is_empty() {
            bail!(
                "line {}: expected condition expression between check name and `~`",
                line
            );
        }
        Ok(CheckDecl {
            name: name.to_string(),
            expr: parse_expr(line, expr_text)?,
            err_label: err_label.to_string(),
        })
    } else {
        bail!(
            "line {}: check statement needs `~err_label` at the end. Format: +check label condition ~err_label",
            line
        );
    }
}

fn parse_branch_decl(line: usize, input: &str) -> Result<BranchDecl> {
    // Format: +branch name expr -> target | expr -> target | ...
    // The name is the branch label, expr is what we're matching on,
    // and each arm is pattern -> target separated by |
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected branch name", line))?;
    let rest = rest.trim();

    // Split on | to get arms, but we need the first part to contain the expression
    // Format: <expr> <pattern1> -> <target1> | <pattern2> -> <target2>
    // Actually, simpler: +branch name <match_expr> | <pattern> -> <target> | <pattern> -> <target>
    // But the design doc says: +branch name ident -> label | ident -> label
    // Let's generalize: everything before the first -> is split into expr + first pattern
    // Better approach: split by | first, then each arm has pattern -> target

    // Find the expression being matched — it's everything before the first pattern->target
    // We'll treat it as: +branch name <expr> <arm1> | <arm2> | ...
    // where each arm is: <pattern> -> <target>

    // Split all arms by |
    let arm_strs: Vec<&str> = rest.split('|').collect();
    if arm_strs.is_empty() {
        bail!("line {}: branch needs at least one arm", line);
    }

    // First segment: may contain the match expression before the first ->
    // Or it could just be pattern -> target if the expr is implicit
    let mut arms = Vec::new();
    let mut match_expr = None;

    for (i, arm_str) in arm_strs.iter().enumerate() {
        let arm_str = arm_str.trim();
        if arm_str.is_empty() {
            continue;
        }

        if let Some((pattern_str, target)) = arm_str.split_once("->") {
            let pattern_str = pattern_str.trim();
            let target = target.trim();
            if target.is_empty() {
                bail!("line {}: empty target in branch arm", line);
            }

            // For the first arm, check if there's an expression before the pattern
            // If the pattern contains spaces and we haven't set match_expr yet,
            // the first word(s) might be the match expression
            if i == 0 && match_expr.is_none() {
                // Try to split: the match expression is everything except the last token before ->
                // Heuristic: if pattern_str has multiple words, first word is the match expr
                let words: Vec<&str> = pattern_str.split_whitespace().collect();
                if words.len() >= 2 {
                    // First tokens are the match expr, last token is the pattern
                    let last = words.last().unwrap();
                    let expr_str = &pattern_str[..pattern_str.rfind(last).unwrap()].trim();
                    if !expr_str.is_empty() {
                        match_expr = Some(parse_expr(line, expr_str)?);
                    }
                    arms.push(BranchArm {
                        pattern: parse_expr(line, last)?,
                        target: target.to_string(),
                    });
                    continue;
                }
            }

            arms.push(BranchArm {
                pattern: parse_expr(line, pattern_str)?,
                target: target.to_string(),
            });
        } else {
            bail!(
                "line {}: expected `pattern -> target` in branch arm, got `{}`",
                line,
                arm_str
            );
        }
    }

    if arms.is_empty() {
        bail!("line {}: branch needs at least one arm", line);
    }

    // If no explicit match expression, use the branch name as an identifier
    let expr = match_expr.unwrap_or_else(|| Expr::Ident(name.to_string()));

    Ok(BranchDecl {
        name: name.to_string(),
        expr,
        arms,
    })
}

fn parse_each_header(line: usize, input: &str) -> Result<(Expr, String, TypeExpr)> {
    let mut split_at = None;
    let mut depth_paren = 0usize;
    let mut depth_brace = 0usize;
    let mut depth_angle = 0usize;
    let chars: Vec<(usize, char)> = input.char_indices().collect();

    for (idx, ch) in chars.iter().copied() {
        match ch {
            '(' => depth_paren += 1,
            ')' => depth_paren = depth_paren.saturating_sub(1),
            '{' => depth_brace += 1,
            '}' => depth_brace = depth_brace.saturating_sub(1),
            '<' => depth_angle += 1,
            '>' => depth_angle = depth_angle.saturating_sub(1),
            ' ' if depth_paren == 0 && depth_brace == 0 && depth_angle == 0 => {
                let next = input[idx..].trim_start();
                if let Some((item, tail)) = take_ident(next) {
                    let tail = tail.trim();
                    if tail.starts_with(':') {
                        split_at = Some((idx, item.to_string(), tail[1..].trim().to_string()));
                        break;
                    }
                }
            }
            _ => {}
        }
    }

    let (idx, item, ty_text) =
        split_at.ok_or_else(|| anyhow!("line {}: expected `collection item:Type`", line))?;
    let collection_text = input[..idx].trim();
    if collection_text.is_empty() {
        bail!("line {}: expected collection expression", line);
    }

    Ok((
        parse_expr(line, collection_text)?,
        item,
        parse_type(line, &ty_text)?,
    ))
}

fn parse_type(line: usize, input: &str) -> Result<TypeExpr> {
    let input = input.trim();
    if input.is_empty() {
        bail!("line {}: expected type", line);
    }

    if input.starts_with('{') {
        if !input.ends_with('}') {
            bail!("line {}: unclosed struct type", line);
        }
        let inner = &input[1..input.len() - 1];
        let mut fields = Vec::new();
        if !inner.trim().is_empty() {
            for field in split_top_level(inner, ',') {
                let (name, ty_text) = split_once_required(line, field.trim(), ":")?;
                fields.push(FieldType {
                    name: name.trim().to_string(),
                    ty: parse_type(line, ty_text.trim())?,
                });
            }
        }
        return Ok(TypeExpr::Struct(fields));
    }

    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: invalid type `{}`", line, input))?;
    let rest = rest.trim();
    if rest.is_empty() {
        return Ok(TypeExpr::Named(name.to_string()));
    }

    if rest.starts_with('<') {
        let close = find_matching_delim(rest, 0, '<', '>')
            .ok_or_else(|| anyhow!("line {}: unclosed generic type `{}`", line, input))?;
        if close != rest.len() - 1 {
            bail!(
                "line {}: unexpected trailing characters in type `{}`",
                line,
                input
            );
        }
        let inner = &rest[1..close];
        let args = if inner.trim().is_empty() {
            Vec::new()
        } else {
            split_top_level(inner, ',')
                .into_iter()
                .map(|part| parse_type(line, part.trim()))
                .collect::<Result<Vec<_>>>()?
        };
        return Ok(TypeExpr::Generic {
            name: name.to_string(),
            args,
        });
    }

    bail!("line {}: could not parse type `{}`", line, input)
}

fn parse_expr(line: usize, input: &str) -> Result<Expr> {
    let tokens = tokenize_expr(line, input)?;
    let mut parser = ExprParser {
        tokens: &tokens,
        index: 0,
        line,
    };
    let expr = parser.parse_bp(0)?;
    if !parser.is_done() {
        bail!(
            "line {}: unexpected token in expression near `{}`",
            line,
            parser.remaining_text()
        );
    }
    Ok(expr)
}

/// Public wrapper for `parse_expr` — used by the API to parse inline `!eval` expressions.
pub fn parse_expr_pub(line: usize, input: &str) -> Result<Expr> {
    parse_expr(line, input)
}

#[derive(Clone, Debug)]
enum Token {
    Ident(String),
    Int(i64),
    Float(f64),
    String(String),
    Symbol(char),
    Arrow,
    EqEq,
    NotEq,
    Gte,
    Lte,
    Gt,
    Lt,
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
}

struct ExprParser<'a> {
    tokens: &'a [Token],
    index: usize,
    line: usize,
}

impl<'a> ExprParser<'a> {
    fn parse_bp(&mut self, min_bp: u8) -> Result<Expr> {
        let mut lhs = self.parse_prefix()?;

        loop {
            if let Some(Token::Symbol('.')) = self.peek() {
                if 20 < min_bp {
                    break;
                }
                self.index += 1;
                let field = self.expect_ident()?;
                lhs = Expr::FieldAccess {
                    base: Box::new(lhs),
                    field,
                };
                continue;
            }

            if let Some(Token::Symbol('(')) = self.peek() {
                if 20 < min_bp {
                    break;
                }
                self.index += 1;
                let mut args = Vec::new();
                if !matches!(self.peek(), Some(Token::Symbol(')'))) {
                    loop {
                        args.push(self.parse_bp(0)?);
                        if matches!(self.peek(), Some(Token::Symbol(','))) {
                            self.index += 1;
                            continue;
                        }
                        break;
                    }
                }
                self.expect_symbol(')')?;
                lhs = Expr::Call {
                    callee: Box::new(lhs),
                    args,
                };
                continue;
            }

            if matches!(self.peek_ident(), Some("as")) {
                if 8 < min_bp {
                    break;
                }
                self.index += 1;
                let ty_text = self.collect_type_tokens();
                if ty_text.is_empty() {
                    bail!("line {}: expected type after `as`", self.line);
                }
                lhs = Expr::Cast {
                    expr: Box::new(lhs),
                    ty: parse_type(self.line, &ty_text)?,
                };
                continue;
            }

            let (left_bp, right_bp, op) = match self.peek() {
                Some(Token::Ident(op)) if op == "OR" => (1, 2, BinaryOp::Or),
                Some(Token::Ident(op)) if op == "AND" => (3, 4, BinaryOp::And),
                Some(Token::EqEq) => (5, 6, BinaryOp::Eq),
                Some(Token::NotEq) => (5, 6, BinaryOp::Neq),
                Some(Token::Gte) => (5, 6, BinaryOp::Gte),
                Some(Token::Lte) => (5, 6, BinaryOp::Lte),
                Some(Token::Gt) => (5, 6, BinaryOp::Gt),
                Some(Token::Lt) => (5, 6, BinaryOp::Lt),
                Some(Token::Plus) => (7, 8, BinaryOp::Add),
                Some(Token::Minus) => (7, 8, BinaryOp::Sub),
                Some(Token::Star) => (9, 10, BinaryOp::Mul),
                Some(Token::Slash) => (9, 10, BinaryOp::Div),
                Some(Token::Percent) => (9, 10, BinaryOp::Mod),
                _ => break,
            };

            if left_bp < min_bp {
                break;
            }

            self.index += 1;
            let rhs = self.parse_bp(right_bp)?;
            lhs = Expr::Binary {
                op,
                left: Box::new(lhs),
                right: Box::new(rhs),
            };
        }

        Ok(lhs)
    }

    fn parse_prefix(&mut self) -> Result<Expr> {
        match self.next().cloned() {
            Some(Token::Int(value)) => Ok(Expr::Int(value)),
            Some(Token::Float(value)) => Ok(Expr::Float(value)),
            Some(Token::String(value)) => Ok(Expr::String(value)),
            Some(Token::Ident(value)) if value == "true" => Ok(Expr::Bool(true)),
            Some(Token::Ident(value)) if value == "false" => Ok(Expr::Bool(false)),
            Some(Token::Ident(value)) if value == "NOT" => Ok(Expr::Unary {
                op: UnaryOp::Not,
                expr: Box::new(self.parse_bp(11)?),
            }),
            Some(Token::Minus) => Ok(Expr::Unary {
                op: UnaryOp::Neg,
                expr: Box::new(self.parse_bp(11)?),
            }),
            Some(Token::Ident(value)) => Ok(Expr::Ident(value)),
            Some(Token::Symbol('(')) => {
                let expr = self.parse_bp(0)?;
                self.expect_symbol(')')?;
                Ok(expr)
            }
            Some(Token::Symbol('{')) => self.parse_struct_literal(),
            Some(token) => bail!(
                "line {}: unexpected token {:?} in expression",
                self.line,
                token
            ),
            None => bail!("line {}: expected expression", self.line),
        }
    }

    fn parse_struct_literal(&mut self) -> Result<Expr> {
        let mut fields = Vec::new();
        if matches!(self.peek(), Some(Token::Symbol('}'))) {
            self.index += 1;
            return Ok(Expr::StructLiteral(fields));
        }

        loop {
            let name = self.expect_ident()?;
            self.expect_symbol(':')?;
            let value = self.parse_bp(0)?;
            fields.push(FieldValue { name, value });

            match self.peek() {
                Some(Token::Symbol(',')) => {
                    self.index += 1;
                }
                Some(Token::Symbol('}')) => {
                    self.index += 1;
                    break;
                }
                _ => bail!("line {}: expected `,` or `}}` in struct literal", self.line),
            }
        }

        Ok(Expr::StructLiteral(fields))
    }

    fn collect_type_tokens(&mut self) -> String {
        let start = self.index;
        let mut depth_angle = 0usize;
        let mut depth_brace = 0usize;
        let mut depth_paren = 0usize;

        while let Some(token) = self.peek() {
            match token {
                Token::Symbol('(') => depth_paren += 1,
                Token::Symbol(')') => {
                    if depth_paren == 0 {
                        break;
                    }
                    depth_paren -= 1;
                }
                Token::Symbol('{') => depth_brace += 1,
                Token::Symbol('}') => {
                    if depth_brace == 0 {
                        break;
                    }
                    depth_brace -= 1;
                }
                Token::Lt => depth_angle += 1,
                Token::Gt => {
                    if depth_angle > 0 {
                        depth_angle -= 1;
                    } else {
                        break;
                    }
                }
                Token::Ident(op)
                    if depth_angle == 0
                        && depth_brace == 0
                        && depth_paren == 0
                        && (op == "AND" || op == "OR" || op == "as") =>
                {
                    break
                }
                Token::EqEq | Token::NotEq | Token::Gte | Token::Lte
                    if depth_angle == 0 && depth_brace == 0 && depth_paren == 0 =>
                {
                    break
                }
                Token::Plus | Token::Minus | Token::Star | Token::Slash | Token::Percent
                    if depth_angle == 0 && depth_brace == 0 && depth_paren == 0 =>
                {
                    break
                }
                _ => {}
            }
            self.index += 1;
        }

        self.tokens[start..self.index]
            .iter()
            .map(token_text)
            .collect::<Vec<_>>()
            .join("")
    }

    fn expect_ident(&mut self) -> Result<String> {
        match self.next().cloned() {
            Some(Token::Ident(value)) => Ok(value),
            _ => bail!("line {}: expected identifier", self.line),
        }
    }

    fn expect_symbol(&mut self, ch: char) -> Result<()> {
        match self.next() {
            Some(Token::Symbol(found)) if *found == ch => Ok(()),
            _ => bail!("line {}: expected `{}`", self.line, ch),
        }
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn peek_ident(&self) -> Option<&str> {
        match self.peek() {
            Some(Token::Ident(value)) => Some(value.as_str()),
            _ => None,
        }
    }

    fn next(&mut self) -> Option<&Token> {
        let token = self.tokens.get(self.index);
        if token.is_some() {
            self.index += 1;
        }
        token
    }

    fn is_done(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn remaining_text(&self) -> String {
        self.tokens[self.index..]
            .iter()
            .map(token_text)
            .collect::<Vec<_>>()
            .join("")
    }
}

fn tokenize_expr(line: usize, input: &str) -> Result<Vec<Token>> {
    let mut chars = input.char_indices().peekable();
    let mut tokens = Vec::new();

    while let Some((idx, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }

        match ch {
            '(' | ')' | '{' | '}' | ',' | ':' | '.' => tokens.push(Token::Symbol(ch)),
            '+' => tokens.push(Token::Plus),
            '*' => tokens.push(Token::Star),
            '/' => tokens.push(Token::Slash),
            '%' => tokens.push(Token::Percent),
            '-' => {
                if matches!(chars.peek(), Some((_, '>'))) {
                    chars.next();
                    tokens.push(Token::Arrow);
                } else {
                    tokens.push(Token::Minus);
                }
            }
            '>' => {
                if matches!(chars.peek(), Some((_, '='))) {
                    chars.next();
                    tokens.push(Token::Gte);
                } else {
                    tokens.push(Token::Gt);
                }
            }
            '<' => {
                if matches!(chars.peek(), Some((_, '='))) {
                    chars.next();
                    tokens.push(Token::Lte);
                } else {
                    tokens.push(Token::Lt);
                }
            }
            '=' => {
                if matches!(chars.peek(), Some((_, '='))) {
                    chars.next();
                    tokens.push(Token::EqEq);
                } else {
                    tokens.push(Token::Symbol('='));
                }
            }
            '!' => {
                if matches!(chars.peek(), Some((_, '='))) {
                    chars.next();
                    tokens.push(Token::NotEq);
                } else {
                    bail!("line {}: unexpected `!` in expression", line);
                }
            }
            '"' => {
                let mut value = String::new();
                let mut terminated = false;
                while let Some((_, next)) = chars.next() {
                    match next {
                        '"' => {
                            terminated = true;
                            break;
                        }
                        '\\' => {
                            let (_, escaped) = chars.next().ok_or_else(|| {
                                anyhow!("line {}: unfinished escape sequence", line)
                            })?;
                            value.push(match escaped {
                                'n' => '\n',
                                'r' => '\r',
                                't' => '\t',
                                '\\' => '\\',
                                '"' => '"',
                                other => other,
                            });
                        }
                        other => value.push(other),
                    }
                }
                if !terminated {
                    bail!("line {}: unterminated string literal", line);
                }
                tokens.push(Token::String(value));
            }
            c if c.is_ascii_digit() => {
                let mut end = idx + c.len_utf8();
                while let Some((next_idx, next)) = chars.peek().copied() {
                    if next.is_ascii_digit() || next == '.' {
                        chars.next();
                        end = next_idx + next.len_utf8();
                    } else {
                        break;
                    }
                }
                let slice = &input[idx..end];
                tokens.push(parse_number_token(line, slice)?);
            }
            c if is_ident_start(c) => {
                let mut end = idx + c.len_utf8();
                while let Some((next_idx, next)) = chars.peek().copied() {
                    if is_ident_continue(next) {
                        chars.next();
                        end = next_idx + next.len_utf8();
                    } else {
                        break;
                    }
                }
                tokens.push(Token::Ident(input[idx..end].to_string()));
            }
            _ => bail!("line {}: unexpected character `{}` in expression", line, ch),
        }
    }

    Ok(tokens)
}

fn parse_number_token(line: usize, value: &str) -> Result<Token> {
    if value.contains('.') {
        Ok(Token::Float(value.parse().with_context(|| {
            format!("line {}: invalid float literal `{}`", line, value)
        })?))
    } else {
        Ok(Token::Int(value.parse().with_context(|| {
            format!("line {}: invalid integer literal `{}`", line, value)
        })?))
    }
}

fn token_text(token: &Token) -> String {
    match token {
        Token::Ident(value) => value.clone(),
        Token::Int(value) => value.to_string(),
        Token::Float(value) => value.to_string(),
        Token::String(value) => format!("\"{}\"", value),
        Token::Symbol(ch) => ch.to_string(),
        Token::Arrow => "->".to_string(),
        Token::EqEq => "==".to_string(),
        Token::NotEq => "!=".to_string(),
        Token::Gte => ">=".to_string(),
        Token::Lte => "<=".to_string(),
        Token::Gt => ">".to_string(),
        Token::Lt => "<".to_string(),
        Token::Plus => "+".to_string(),
        Token::Minus => "-".to_string(),
        Token::Star => "*".to_string(),
        Token::Slash => "/".to_string(),
        Token::Percent => "%".to_string(),
    }
}

fn split_effects(input: &str) -> (&str, Option<&str>) {
    let input = input.trim();
    if let Some(open) = input.rfind('[') {
        if input.ends_with(']') {
            return (&input[..open], Some(&input[open + 1..input.len() - 1]));
        }
    }
    (input, None)
}

fn split_once_required<'a>(
    line: usize,
    input: &'a str,
    needle: &str,
) -> Result<(&'a str, &'a str)> {
    input
        .split_once(needle)
        .ok_or_else(|| anyhow!("line {}: expected `{}` in `{}`", line, needle, input))
}

/// Parse test input — supports both `key=value` pairs and regular expressions.
///
/// `key=value` format: `name="alice" age=25 active=true`
/// Regular format: `{name: "alice", age: 25}` or `5` or `"hello"`
pub fn parse_test_input(line: usize, input: &str) -> Result<Expr> {
    let input = input.trim();
    if input.is_empty() {
        // Empty input is valid for zero-parameter functions:
        // +with -> expect "value"
        return Ok(Expr::StructLiteral(vec![]));
    }

    // Detect key=value format: first non-whitespace token contains '=' and doesn't start with {
    // but is not a comparison like `x==5` or `x>=3`
    if !input.starts_with('{')
        && !input.starts_with('"')
        && !input.starts_with('-')
        && !input.chars().next().unwrap_or(' ').is_ascii_digit()
        && input.contains('=')
        && !input.contains("==")
        && !input.contains(">=")
        && !input.contains("<=")
        && !input.contains("!=")
    {
        // Parse as space-separated key=value pairs
        let mut fields = Vec::new();
        let mut rest = input;

        while !rest.is_empty() {
            rest = rest.trim_start();
            if rest.is_empty() {
                break;
            }

            // Find key
            let eq_pos = rest.find('=').ok_or_else(|| {
                anyhow!("line {}: expected `key=value` pair, got `{}`", line, rest)
            })?;
            let key = rest[..eq_pos].trim();
            rest = &rest[eq_pos + 1..];

            // Parse value — could be quoted string, number, bool, or nested struct
            let (value, remaining) = parse_test_value(line, rest)?;
            fields.push(FieldValue {
                name: key.to_string(),
                value,
            });
            rest = remaining;
        }

        if fields.is_empty() {
            bail!("line {}: no key=value pairs found in test input", line);
        }

        Ok(Expr::StructLiteral(fields))
    } else {
        // Regular expression format — try as single expression first
        match parse_expr(line, input) {
            Ok(expr) => Ok(expr),
            Err(_) => {
                // Single expression failed — try parsing as multiple space-separated
                // values (e.g. "hello" "world" 42). This handles positional args for
                // !eval and +with when there's no key=value syntax.
                let mut values = Vec::new();
                let mut rest = input;
                while !rest.is_empty() {
                    rest = rest.trim_start();
                    if rest.is_empty() {
                        break;
                    }
                    let (val, remaining) = parse_test_value(line, rest)?;
                    values.push(val);
                    rest = remaining;
                }
                if values.len() <= 1 {
                    // Re-run parse_expr for its original error message
                    parse_expr(line, input)
                } else {
                    // Multiple positional values: wrap as StructLiteral with _0, _1, ...
                    let fields = values
                        .into_iter()
                        .enumerate()
                        .map(|(i, v)| FieldValue {
                            name: format!("_{i}"),
                            value: v,
                        })
                        .collect();
                    Ok(Expr::StructLiteral(fields))
                }
            }
        }
    }
}

/// Parse a single value from key=value format, returning (value, remaining_input).
fn parse_test_value(line: usize, input: &str) -> Result<(Expr, &str)> {
    let input = input.trim_start();

    if input.starts_with('"') {
        // Quoted string — scan for unescaped closing quote, handling backslash escapes
        let mut value = String::new();
        let mut chars = input[1..].char_indices();
        let mut end_offset = None; // offset into input (past the closing quote)
        while let Some((i, ch)) = chars.next() {
            match ch {
                '"' => {
                    end_offset = Some(1 + i + 1); // 1 for opening quote + i + 1 past closing quote
                    break;
                }
                '\\' => {
                    if let Some((_, escaped)) = chars.next() {
                        value.push(match escaped {
                            'n' => '\n',
                            'r' => '\r',
                            't' => '\t',
                            '\\' => '\\',
                            '"' => '"',
                            other => other,
                        });
                    } else {
                        bail!("line {}: unfinished escape in test value string", line);
                    }
                }
                other => value.push(other),
            }
        }
        let end_offset = end_offset
            .ok_or_else(|| anyhow!("line {}: unterminated string in test value", line))?;
        let rest = &input[end_offset..];
        Ok((Expr::String(value), rest))
    } else if input.starts_with('{') {
        // Nested struct literal — find matching brace
        let mut depth = 0;
        let mut end = 0;
        for (i, ch) in input.char_indices() {
            match ch {
                '{' => depth += 1,
                '}' => {
                    depth -= 1;
                    if depth == 0 {
                        end = i + 1;
                        break;
                    }
                }
                _ => {}
            }
        }
        if depth != 0 {
            bail!("line {}: unmatched brace in test value", line);
        }
        let expr = parse_expr(line, &input[..end])?;
        Ok((expr, &input[end..]))
    } else {
        // Check for constructor call: Name(args) or Name(args, args)
        // Find the identifier part first
        let ident_end = input
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(input.len());
        let ident = &input[..ident_end];
        let after_ident = &input[ident_end..];

        if !ident.is_empty() && after_ident.starts_with('(') {
            // Constructor call: Name(arg1, arg2, ...)
            // Find matching closing paren
            let mut depth = 0;
            let mut paren_end = 0;
            for (i, ch) in after_ident.char_indices() {
                match ch {
                    '(' => depth += 1,
                    ')' => {
                        depth -= 1;
                        if depth == 0 {
                            paren_end = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            if depth != 0 {
                bail!("line {}: unmatched paren in test value", line);
            }
            let full = &input[..ident_end + paren_end];
            let rest = &input[ident_end + paren_end..];
            // Parse the full expression using the main expression parser
            let expr = parse_expr(line, full)?;
            Ok((expr, rest))
        } else {
            // Simple token: number, bool, or identifier
            let end = input
                .find(|c: char| c.is_whitespace())
                .unwrap_or(input.len());
            let token = &input[..end];
            let rest = &input[end..];

            let expr = if token == "true" {
                Expr::Bool(true)
            } else if token == "false" {
                Expr::Bool(false)
            } else if let Ok(n) = token.parse::<i64>() {
                Expr::Int(n)
            } else if let Ok(f) = token.parse::<f64>() {
                Expr::Float(f)
            } else {
                Expr::Ident(token.to_string())
            };

            Ok((expr, rest))
        }
    }
}

fn split_test_case<'a>(line: usize, input: &'a str) -> Result<(&'a str, &'a str)> {
    let (left, right) = split_once_required(line, input, "->")?;
    let right = right.trim();
    let expected = right
        .strip_prefix("expect")
        .ok_or_else(|| anyhow!("line {}: expected `expect` after `->`", line))?;
    Ok((left, expected))
}

/// Parse the expected portion of a test case, detecting matcher syntax.
///
/// Matcher forms:
///   `contains("substring")`     → TestMatcher::Contains
///   `starts_with("prefix")`     → TestMatcher::StartsWith
///   `Ok`                        → TestMatcher::AnyOk  (bare, no parens)
///   `Err`                       → TestMatcher::AnyErr (bare, no parens)
///   `Err("specific msg")`       → TestMatcher::ErrContaining
///
/// Anything else is parsed as a normal expression (no matcher).
fn parse_expected_with_matcher(line: usize, input: &str) -> Result<(Expr, Option<TestMatcher>)> {
    let trimmed = input.trim();

    // contains("...")
    if let Some(inner) = trimmed.strip_prefix("contains(") {
        let inner = inner.trim();
        if let Some(inner) = inner.strip_suffix(')') {
            let inner = inner.trim();
            if inner.starts_with('"') && inner.ends_with('"') && inner.len() >= 2 {
                let s = unescape_string_content(&inner[1..inner.len() - 1]);
                // The expected expr is a dummy — matcher takes over
                return Ok((Expr::String(s.clone()), Some(TestMatcher::Contains(s))));
            }
        }
        bail!("line {}: contains() expects a quoted string argument", line);
    }

    // starts_with("...")
    if let Some(inner) = trimmed.strip_prefix("starts_with(") {
        let inner = inner.trim();
        if let Some(inner) = inner.strip_suffix(')') {
            let inner = inner.trim();
            if inner.starts_with('"') && inner.ends_with('"') && inner.len() >= 2 {
                let s = unescape_string_content(&inner[1..inner.len() - 1]);
                return Ok((Expr::String(s.clone()), Some(TestMatcher::StartsWith(s))));
            }
        }
        bail!(
            "line {}: starts_with() expects a quoted string argument",
            line
        );
    }

    // Bare Ok (no parens) — AnyOk matcher
    if trimmed == "Ok" {
        return Ok((Expr::Ident("Ok".to_string()), Some(TestMatcher::AnyOk)));
    }

    // Bare Err (no parens) — AnyErr matcher
    if trimmed == "Err" {
        return Ok((Expr::Ident("Err".to_string()), Some(TestMatcher::AnyErr)));
    }

    // Err("specific message") — ErrContaining matcher
    if let Some(inner) = trimmed.strip_prefix("Err(") {
        let inner = inner.trim();
        if let Some(inner) = inner.strip_suffix(')') {
            let inner = inner.trim();
            if inner.starts_with('"') && inner.ends_with('"') && inner.len() >= 2 {
                let s = unescape_string_content(&inner[1..inner.len() - 1]);
                return Ok((
                    Expr::Call {
                        callee: Box::new(Expr::Ident("Err".to_string())),
                        args: vec![Expr::String(s.clone())],
                    },
                    Some(TestMatcher::ErrContaining(s)),
                ));
            }
            // Non-string arg in Err(...) — fall through to normal parsing.
            // This handles Err(err_label) as before (exact match, no matcher).
        }
    }

    // No matcher — normal expression
    let expr = parse_expr(line, trimmed)?;
    Ok((expr, None))
}

/// Simple unescape for string content between quotes (handles \n, \t, \\, \").
fn unescape_string_content(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(ch) = chars.next() {
        if ch == '\\' {
            match chars.next() {
                Some('n') => out.push('\n'),
                Some('r') => out.push('\r'),
                Some('t') => out.push('\t'),
                Some('\\') => out.push('\\'),
                Some('"') => out.push('"'),
                Some(other) => {
                    out.push('\\');
                    out.push(other);
                }
                None => out.push('\\'),
            }
        } else {
            out.push(ch);
        }
    }
    out
}

/// Parse a `+after` check line. Format: `<target> <matcher> "<value>"`
fn parse_after_check(line: usize, input: &str) -> Result<AfterCheck> {
    let input = input.trim();
    // Split: target matcher "value"
    let (target, rest) = input.split_once(char::is_whitespace).ok_or_else(|| {
        anyhow!(
            "line {}: +after expects `<target> <matcher> \"<value>\"`",
            line
        )
    })?;
    let rest = rest.trim();
    let (matcher, rest) = rest.split_once(char::is_whitespace).ok_or_else(|| {
        anyhow!(
            "line {}: +after expects `<target> <matcher> \"<value>\"`",
            line
        )
    })?;
    let rest = rest.trim();

    // Value can be quoted or bare
    let value = if rest.starts_with('"') && rest.ends_with('"') && rest.len() >= 2 {
        unescape_string_content(&rest[1..rest.len() - 1])
    } else {
        rest.to_string()
    };

    Ok(AfterCheck {
        target: target.to_string(),
        matcher: matcher.to_string(),
        value,
    })
}

fn take_ident(input: &str) -> Option<(&str, &str)> {
    let input = input.trim_start();
    let mut chars = input.char_indices();
    let (_, first) = chars.next()?;
    if !is_ident_start(first) {
        return None;
    }
    let mut end = first.len_utf8();
    for (idx, ch) in chars {
        if is_ident_continue(ch) {
            end = idx + ch.len_utf8();
        } else {
            break;
        }
    }
    Some((&input[..end], &input[end..]))
}

fn is_ident_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_ident_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

fn count_indent(input: &str) -> usize {
    input
        .chars()
        .take_while(|ch| *ch == ' ' || *ch == '\t')
        .map(|ch| if ch == '\t' { 2 } else { 1 })
        .sum()
}

fn find_matching_delim(input: &str, start: usize, open: char, close: char) -> Option<usize> {
    let mut depth = 0usize;
    for (idx, ch) in input.char_indices().skip(start) {
        if ch == open {
            depth += 1;
        } else if ch == close {
            depth = depth.saturating_sub(1);
            if depth == 0 {
                return Some(idx);
            }
        }
    }
    None
}

fn split_top_level(input: &str, needle: char) -> Vec<&str> {
    let mut parts = Vec::new();
    let mut start = 0usize;
    let mut paren = 0usize;
    let mut brace = 0usize;
    let mut angle = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (idx, ch) in input.char_indices() {
        if in_string {
            if escaped {
                escaped = false;
                continue;
            }
            match ch {
                '\\' => escaped = true,
                '"' => in_string = false,
                _ => {}
            }
            continue;
        }

        match ch {
            '"' => in_string = true,
            '(' => paren += 1,
            ')' => paren = paren.saturating_sub(1),
            '{' => brace += 1,
            '}' => brace = brace.saturating_sub(1),
            '<' => angle += 1,
            '>' => angle = angle.saturating_sub(1),
            _ if ch == needle && paren == 0 && brace == 0 && angle == 0 => {
                parts.push(&input[start..idx]);
                start = idx + ch.len_utf8();
            }
            _ => {}
        }
    }

    parts.push(&input[start..]);
    parts
}

fn find_matching_open_paren_variant(input: &str) -> Option<usize> {
    let open = input.find('(')?;
    if input.ends_with(')') {
        Some(open)
    } else {
        None
    }
}

/// Parse a +case pattern. Returns (variant_name, flat_bindings, optional nested patterns).
/// Simple cases like `Literal(x)` produce flat bindings with no patterns.
/// Nested cases like `Add(Literal(x), y)` produce patterns.
/// Wildcard `_` produces variant="_" with no bindings.
fn parse_case_pattern(
    line_num: usize,
    input: &str,
) -> Result<(String, Vec<String>, Option<Vec<MatchPatternDecl>>)> {
    let input = input.trim();

    // Wildcard: +case _
    if input == "_" {
        return Ok(("_".to_string(), vec![], None));
    }

    // No parentheses: +case VariantName
    let Some(paren) = input.find('(') else {
        return Ok((input.to_string(), vec![], None));
    };

    let variant = input[..paren].trim().to_string();
    let close = input
        .rfind(')')
        .ok_or_else(|| anyhow!("line {line_num}: expected ')' in +case"))?;
    let inner = &input[paren + 1..close];

    // Parse the inner patterns
    let parts = split_top_level(inner, ',');
    let mut patterns = Vec::new();
    let mut flat_bindings = Vec::new();
    let mut has_nested = false;

    for part in &parts {
        let p = parse_match_pattern(line_num, part.trim())?;
        match &p {
            MatchPatternDecl::Binding(name) => {
                flat_bindings.push(name.clone());
            }
            _ => {
                has_nested = true;
                // For nested patterns, we need a placeholder binding
                flat_bindings.push(format!("__pat_{}", flat_bindings.len()));
            }
        }
        patterns.push(p);
    }

    if has_nested {
        Ok((variant, flat_bindings, Some(patterns)))
    } else {
        Ok((variant, flat_bindings, None))
    }
}

/// Parse a single match pattern element.
fn parse_match_pattern(line_num: usize, input: &str) -> Result<MatchPatternDecl> {
    let input = input.trim();

    // Empty
    if input.is_empty() {
        return Ok(MatchPatternDecl::Binding("_".to_string()));
    }

    // Literal integer (including negative)
    if let Ok(n) = input.parse::<i64>() {
        return Ok(MatchPatternDecl::LiteralInt(n));
    }

    // Literal bool
    if input == "true" {
        return Ok(MatchPatternDecl::LiteralBool(true));
    }
    if input == "false" {
        return Ok(MatchPatternDecl::LiteralBool(false));
    }

    // Literal string
    if input.starts_with('"') && input.ends_with('"') && input.len() >= 2 {
        return Ok(MatchPatternDecl::LiteralString(
            input[1..input.len() - 1].to_string(),
        ));
    }

    // Variant with sub-patterns: e.g. Literal(x) or Add(Literal(0), y)
    if let Some(paren) = input.find('(') {
        let close = input
            .rfind(')')
            .ok_or_else(|| anyhow!("line {line_num}: expected ')' in nested pattern"))?;
        let variant = input[..paren].trim().to_string();
        let inner = &input[paren + 1..close];
        let parts = split_top_level(inner, ',');
        let mut sub_patterns = Vec::new();
        for part in &parts {
            sub_patterns.push(parse_match_pattern(line_num, part.trim())?);
        }
        return Ok(MatchPatternDecl::Variant {
            variant,
            sub_patterns,
        });
    }

    // Simple binding (identifier or _)
    Ok(MatchPatternDecl::Binding(input.to_string()))
}

/// Parse a quoted string from `!mock`, decoding standard escape sequences
/// (`\"`, `\\`, `\n`, `\r`, `\t`). Returns the decoded string content and
/// the remaining unparsed input.  If the input doesn't start with `"`, the
/// content up to the next whitespace (or end) is returned verbatim.
fn parse_mock_string(line: usize, input: &str) -> Result<(String, &str)> {
    let input = input.trim_start();
    if !input.starts_with('"') {
        // Unquoted token — take until whitespace or end
        let end = input.find(char::is_whitespace).unwrap_or(input.len());
        return Ok((input[..end].to_string(), &input[end..]));
    }
    // Quoted string with escape processing
    let mut chars = input[1..].char_indices();
    let mut value = String::new();
    while let Some((i, ch)) = chars.next() {
        match ch {
            '"' => {
                let consumed = 1 + i + 1; // opening quote + content up to i + closing quote
                return Ok((value, &input[consumed..]));
            }
            '\\' => {
                let (_, escaped) = chars
                    .next()
                    .ok_or_else(|| anyhow!("line {line}: unfinished escape in !mock string"))?;
                value.push(match escaped {
                    'n' => '\n',
                    'r' => '\r',
                    't' => '\t',
                    '\\' => '\\',
                    '"' => '"',
                    other => other,
                });
            }
            other => value.push(other),
        }
    }
    bail!("line {line}: unterminated string in !mock")
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Helper: parse single operation ───────────────────────────────────

    fn parse_one(source: &str) -> Operation {
        let ops = parse(source).expect("parse failed");
        assert_eq!(ops.len(), 1, "expected 1 operation, got {}", ops.len());
        ops.into_iter().next().unwrap()
    }

    fn parse_ops(source: &str) -> Vec<Operation> {
        parse(source).expect("parse failed")
    }

    // ═════════════════════════════════════════════════════════════════════
    // 1. Expression parsing (via parse_expr_pub)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn expr_int_literal() {
        let e = parse_expr_pub(0, "42").unwrap();
        assert!(matches!(e, Expr::Int(42)));
    }

    #[test]
    fn expr_negative_int() {
        let e = parse_expr_pub(0, "-7").unwrap();
        match e {
            Expr::Unary {
                op: UnaryOp::Neg,
                expr,
            } => {
                assert!(matches!(*expr, Expr::Int(7)));
            }
            _ => panic!("expected Unary(Neg, Int(7)), got {e:?}"),
        }
    }

    #[test]
    fn expr_float_literal() {
        let e = parse_expr_pub(0, "3.14").unwrap();
        match e {
            Expr::Float(f) => assert!((f - 3.14).abs() < 1e-10),
            _ => panic!("expected Float, got {e:?}"),
        }
    }

    #[test]
    fn expr_bool_true() {
        let e = parse_expr_pub(0, "true").unwrap();
        assert!(matches!(e, Expr::Bool(true)));
    }

    #[test]
    fn expr_bool_false() {
        let e = parse_expr_pub(0, "false").unwrap();
        assert!(matches!(e, Expr::Bool(false)));
    }

    #[test]
    fn expr_string_literal() {
        let e = parse_expr_pub(0, r#""hello world""#).unwrap();
        match e {
            Expr::String(s) => assert_eq!(s, "hello world"),
            _ => panic!("expected String, got {e:?}"),
        }
    }

    #[test]
    fn expr_string_with_escapes() {
        let e = parse_expr_pub(0, r#""line1\nline2\ttab\\slash\"quote""#).unwrap();
        match e {
            Expr::String(s) => assert_eq!(s, "line1\nline2\ttab\\slash\"quote"),
            _ => panic!("expected String with escapes, got {e:?}"),
        }
    }

    #[test]
    fn expr_ident() {
        let e = parse_expr_pub(0, "my_var").unwrap();
        match e {
            Expr::Ident(name) => assert_eq!(name, "my_var"),
            _ => panic!("expected Ident, got {e:?}"),
        }
    }

    #[test]
    fn expr_add() {
        let e = parse_expr_pub(0, "2 + 3").unwrap();
        match e {
            Expr::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*left, Expr::Int(2)));
                assert!(matches!(*right, Expr::Int(3)));
            }
            _ => panic!("expected Binary Add, got {e:?}"),
        }
    }

    #[test]
    fn expr_sub() {
        let e = parse_expr_pub(0, "10 - 4").unwrap();
        match e {
            Expr::Binary {
                op: BinaryOp::Sub,
                left,
                right,
            } => {
                assert!(matches!(*left, Expr::Int(10)));
                assert!(matches!(*right, Expr::Int(4)));
            }
            _ => panic!("expected Binary Sub, got {e:?}"),
        }
    }

    #[test]
    fn expr_mul_add_precedence() {
        // 1 * 2 + 3 should be (1*2) + 3
        let e = parse_expr_pub(0, "1 * 2 + 3").unwrap();
        match e {
            Expr::Binary {
                op: BinaryOp::Add,
                left,
                right,
            } => {
                assert!(matches!(*right, Expr::Int(3)));
                match *left {
                    Expr::Binary {
                        op: BinaryOp::Mul,
                        left: l2,
                        right: r2,
                    } => {
                        assert!(matches!(*l2, Expr::Int(1)));
                        assert!(matches!(*r2, Expr::Int(2)));
                    }
                    _ => panic!("expected Mul on left of Add"),
                }
            }
            _ => panic!("expected Add at top, got {e:?}"),
        }
    }

    #[test]
    fn expr_div_and_mod() {
        let e = parse_expr_pub(0, "10 / 3").unwrap();
        assert!(matches!(
            e,
            Expr::Binary {
                op: BinaryOp::Div,
                ..
            }
        ));

        let e = parse_expr_pub(0, "10 % 3").unwrap();
        assert!(matches!(
            e,
            Expr::Binary {
                op: BinaryOp::Mod,
                ..
            }
        ));
    }

    #[test]
    fn expr_comparison_operators() {
        assert!(matches!(
            parse_expr_pub(0, "a >= b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Gte,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_pub(0, "a <= b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Lte,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_pub(0, "a == b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Eq,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_pub(0, "a != b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Neq,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_pub(0, "a > b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Gt,
                ..
            }
        ));
        assert!(matches!(
            parse_expr_pub(0, "a < b").unwrap(),
            Expr::Binary {
                op: BinaryOp::Lt,
                ..
            }
        ));
    }

    #[test]
    fn expr_and_or() {
        let e = parse_expr_pub(0, "x AND y").unwrap();
        assert!(matches!(
            e,
            Expr::Binary {
                op: BinaryOp::And,
                ..
            }
        ));

        let e = parse_expr_pub(0, "x OR y").unwrap();
        assert!(matches!(
            e,
            Expr::Binary {
                op: BinaryOp::Or,
                ..
            }
        ));
    }

    #[test]
    fn expr_not() {
        let e = parse_expr_pub(0, "NOT x").unwrap();
        match e {
            Expr::Unary {
                op: UnaryOp::Not,
                expr,
            } => {
                assert!(matches!(*expr, Expr::Ident(ref n) if n == "x"));
            }
            _ => panic!("expected NOT unary, got {e:?}"),
        }
    }

    #[test]
    fn expr_and_or_precedence() {
        // x OR y AND z  →  x OR (y AND z) because AND binds tighter
        let e = parse_expr_pub(0, "x OR y AND z").unwrap();
        match e {
            Expr::Binary {
                op: BinaryOp::Or,
                left,
                right,
            } => {
                assert!(matches!(*left, Expr::Ident(ref n) if n == "x"));
                assert!(matches!(
                    *right,
                    Expr::Binary {
                        op: BinaryOp::And,
                        ..
                    }
                ));
            }
            _ => panic!("expected OR at top, got {e:?}"),
        }
    }

    #[test]
    fn expr_function_call_no_args() {
        let e = parse_expr_pub(0, "foo()").unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert!(matches!(*callee, Expr::Ident(ref n) if n == "foo"));
                assert!(args.is_empty());
            }
            _ => panic!("expected Call, got {e:?}"),
        }
    }

    #[test]
    fn expr_function_call_one_arg() {
        let e = parse_expr_pub(0, "len(x)").unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert!(matches!(*callee, Expr::Ident(ref n) if n == "len"));
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expr::Ident(ref n) if n == "x"));
            }
            _ => panic!("expected Call, got {e:?}"),
        }
    }

    #[test]
    fn expr_function_call_multi_args() {
        let e = parse_expr_pub(0, r#"concat("a", "b", "c")"#).unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert!(matches!(*callee, Expr::Ident(ref n) if n == "concat"));
                assert_eq!(args.len(), 3);
                assert!(matches!(&args[0], Expr::String(s) if s == "a"));
                assert!(matches!(&args[1], Expr::String(s) if s == "b"));
                assert!(matches!(&args[2], Expr::String(s) if s == "c"));
            }
            _ => panic!("expected Call, got {e:?}"),
        }
    }

    #[test]
    fn expr_nested_function_calls() {
        let e = parse_expr_pub(0, r#"len(concat("a", "b"))"#).unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert!(matches!(*callee, Expr::Ident(ref n) if n == "len"));
                assert_eq!(args.len(), 1);
                assert!(matches!(args[0], Expr::Call { .. }));
            }
            _ => panic!("expected nested Call, got {e:?}"),
        }
    }

    #[test]
    fn expr_field_access() {
        let e = parse_expr_pub(0, "user.name").unwrap();
        match e {
            Expr::FieldAccess { base, field } => {
                assert!(matches!(*base, Expr::Ident(ref n) if n == "user"));
                assert_eq!(field, "name");
            }
            _ => panic!("expected FieldAccess, got {e:?}"),
        }
    }

    #[test]
    fn expr_chained_field_access() {
        let e = parse_expr_pub(0, "a.b.c").unwrap();
        match e {
            Expr::FieldAccess { base, field } => {
                assert_eq!(field, "c");
                match *base {
                    Expr::FieldAccess {
                        base: inner,
                        field: f2,
                    } => {
                        assert_eq!(f2, "b");
                        assert!(matches!(*inner, Expr::Ident(ref n) if n == "a"));
                    }
                    _ => panic!("expected nested FieldAccess"),
                }
            }
            _ => panic!("expected FieldAccess, got {e:?}"),
        }
    }

    #[test]
    fn expr_method_call_on_field() {
        // user.name.len() — field access + call
        let e = parse_expr_pub(0, "user.name.len()").unwrap();
        // Should be Call { callee: FieldAccess { FieldAccess { user, name }, len }, args: [] }
        // Actually: the Pratt parser sees user.name first as FieldAccess, then .len as another FieldAccess,
        // then () as a Call on that.
        match e {
            Expr::Call { callee, args } => {
                assert!(args.is_empty());
                match *callee {
                    Expr::FieldAccess { field, .. } => assert_eq!(field, "len"),
                    _ => panic!("expected FieldAccess inside Call"),
                }
            }
            _ => panic!("expected Call, got {e:?}"),
        }
    }

    #[test]
    fn expr_struct_literal() {
        let e = parse_expr_pub(0, r#"{name: "alice", age: 25}"#).unwrap();
        match e {
            Expr::StructLiteral(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "name");
                assert!(matches!(&fields[0].value, Expr::String(s) if s == "alice"));
                assert_eq!(fields[1].name, "age");
                assert!(matches!(fields[1].value, Expr::Int(25)));
            }
            _ => panic!("expected StructLiteral, got {e:?}"),
        }
    }

    #[test]
    fn expr_list_call() {
        let e = parse_expr_pub(0, "list(1, 2, 3)").unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert!(matches!(*callee, Expr::Ident(ref n) if n == "list"));
                assert_eq!(args.len(), 3);
                assert!(matches!(args[0], Expr::Int(1)));
                assert!(matches!(args[1], Expr::Int(2)));
                assert!(matches!(args[2], Expr::Int(3)));
            }
            _ => panic!("expected Call(list, ...), got {e:?}"),
        }
    }

    #[test]
    fn expr_parenthesized() {
        // (1 + 2) * 3
        let e = parse_expr_pub(0, "(1 + 2) * 3").unwrap();
        match e {
            Expr::Binary {
                op: BinaryOp::Mul,
                left,
                right,
            } => {
                assert!(matches!(*right, Expr::Int(3)));
                assert!(matches!(
                    *left,
                    Expr::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected Mul at top, got {e:?}"),
        }
    }

    #[test]
    fn expr_cast() {
        let e = parse_expr_pub(0, "x as Float").unwrap();
        match e {
            Expr::Cast { expr, ty } => {
                assert!(matches!(*expr, Expr::Ident(ref n) if n == "x"));
                assert!(matches!(ty, TypeExpr::Named(ref n) if n == "Float"));
            }
            _ => panic!("expected Cast, got {e:?}"),
        }
    }

    #[test]
    fn expr_complex_compound() {
        // a + b * c >= d AND NOT e
        let e = parse_expr_pub(0, "a + b * c >= d AND NOT e").unwrap();
        // Should be AND at top
        assert!(matches!(
            e,
            Expr::Binary {
                op: BinaryOp::And,
                ..
            }
        ));
    }

    #[test]
    fn expr_parse_error_incomplete() {
        assert!(parse_expr_pub(0, "1 +").is_err());
    }

    #[test]
    fn expr_parse_error_trailing_tokens() {
        assert!(parse_expr_pub(0, "1 2").is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 2. Statement parsing (via parse(), within a function body)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn stmt_let() {
        let op = parse_one("+let x:Int = 42");
        match op {
            Operation::Let(decl) => {
                assert_eq!(decl.name, "x");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "Int"));
                assert!(matches!(decl.expr, Expr::Int(42)));
            }
            _ => panic!("expected Let, got {op:?}"),
        }
    }

    #[test]
    fn stmt_let_with_expr() {
        let op = parse_one("+let msg:String = concat(a, b)");
        match op {
            Operation::Let(decl) => {
                assert_eq!(decl.name, "msg");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "String"));
                assert!(matches!(decl.expr, Expr::Call { .. }));
            }
            _ => panic!("expected Let with call, got {op:?}"),
        }
    }

    #[test]
    fn stmt_set() {
        let op = parse_one("+set count = count + 1");
        match op {
            Operation::Set(decl) => {
                assert_eq!(decl.name, "count");
                assert!(matches!(
                    decl.expr,
                    Expr::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected Set, got {op:?}"),
        }
    }

    #[test]
    fn stmt_call() {
        let op = parse_one("+call result:String = validate(input)");
        match op {
            Operation::Call(decl) => {
                assert_eq!(decl.name, "result");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "String"));
                assert!(matches!(decl.expr, Expr::Call { .. }));
            }
            _ => panic!("expected Call, got {op:?}"),
        }
    }

    #[test]
    fn stmt_check() {
        let op = parse_one("+check positive x > 0 ~err_negative");
        match op {
            Operation::Check(decl) => {
                assert_eq!(decl.name, "positive");
                assert!(matches!(
                    decl.expr,
                    Expr::Binary {
                        op: BinaryOp::Gt,
                        ..
                    }
                ));
                assert_eq!(decl.err_label, "err_negative");
            }
            _ => panic!("expected Check, got {op:?}"),
        }
    }

    #[test]
    fn stmt_return() {
        let op = parse_one("+return 42");
        match op {
            Operation::Return(decl) => {
                assert!(matches!(decl.expr, Expr::Int(42)));
            }
            _ => panic!("expected Return, got {op:?}"),
        }
    }

    #[test]
    fn stmt_return_expr() {
        let op = parse_one(r#"+return concat("hello", " ", "world")"#);
        match op {
            Operation::Return(decl) => {
                assert!(matches!(decl.expr, Expr::Call { .. }));
            }
            _ => panic!("expected Return, got {op:?}"),
        }
    }

    #[test]
    fn stmt_await() {
        let op = parse_one("+await data:String = http_get(url)");
        match op {
            Operation::Await(decl) => {
                assert_eq!(decl.name, "data");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "String"));
                assert!(matches!(decl.call, Expr::Call { .. }));
            }
            _ => panic!("expected Await, got {op:?}"),
        }
    }

    #[test]
    fn stmt_spawn() {
        let op = parse_one("+spawn poll_loop()");
        match op {
            Operation::Spawn(decl) => {
                assert!(matches!(decl.call, Expr::Call { .. }));
            }
            _ => panic!("expected Spawn, got {op:?}"),
        }
    }

    #[test]
    fn stmt_if_then_end() {
        let source = "\
+if x > 0
  +return x
+end";
        let op = parse_one(source);
        match op {
            Operation::If(decl) => {
                assert!(matches!(
                    decl.condition,
                    Expr::Binary {
                        op: BinaryOp::Gt,
                        ..
                    }
                ));
                assert_eq!(decl.then_body.len(), 1);
                assert!(matches!(decl.then_body[0], Operation::Return(_)));
                assert!(decl.elif_branches.is_empty());
                assert!(decl.else_body.is_empty());
            }
            _ => panic!("expected If, got {op:?}"),
        }
    }

    #[test]
    fn stmt_if_elif_else_end() {
        let source = "\
+if x > 0
  +return \"positive\"
+elif x == 0
  +return \"zero\"
+else
  +return \"negative\"
+end";
        let op = parse_one(source);
        match op {
            Operation::If(decl) => {
                assert_eq!(decl.then_body.len(), 1);
                assert_eq!(decl.elif_branches.len(), 1);
                assert_eq!(decl.elif_branches[0].1.len(), 1);
                assert_eq!(decl.else_body.len(), 1);
            }
            _ => panic!("expected If with elif/else, got {op:?}"),
        }
    }

    #[test]
    fn stmt_while_end() {
        let source = "\
+while i < 10
  +set i = i + 1
+end";
        let op = parse_one(source);
        match op {
            Operation::While(decl) => {
                assert!(matches!(
                    decl.condition,
                    Expr::Binary {
                        op: BinaryOp::Lt,
                        ..
                    }
                ));
                assert_eq!(decl.body.len(), 1);
                assert!(matches!(decl.body[0], Operation::Set(_)));
            }
            _ => panic!("expected While, got {op:?}"),
        }
    }

    #[test]
    fn stmt_match_case_end() {
        let source = "\
+match result
+case Ok(value)
  +return value
+case Err(msg)
  +return msg
+end";
        let op = parse_one(source);
        match op {
            Operation::Match(decl) => {
                assert!(matches!(decl.expr, Expr::Ident(ref n) if n == "result"));
                assert_eq!(decl.arms.len(), 2);
                assert_eq!(decl.arms[0].variant, "Ok");
                assert_eq!(decl.arms[0].bindings, vec!["value"]);
                assert_eq!(decl.arms[1].variant, "Err");
                assert_eq!(decl.arms[1].bindings, vec!["msg"]);
            }
            _ => panic!("expected Match, got {op:?}"),
        }
    }

    #[test]
    fn stmt_match_wildcard() {
        let source = "\
+match x
+case Some(v)
  +return v
+case _
  +return 0
+end";
        let op = parse_one(source);
        match op {
            Operation::Match(decl) => {
                assert_eq!(decl.arms.len(), 2);
                assert_eq!(decl.arms[1].variant, "_");
            }
            _ => panic!("expected Match with wildcard, got {op:?}"),
        }
    }

    #[test]
    fn stmt_each_end() {
        let source = "\
+each items item:String
  +call r:String = process(item)
+end";
        let op = parse_one(source);
        match op {
            Operation::Each(decl) => {
                assert!(matches!(decl.collection, Expr::Ident(ref n) if n == "items"));
                assert_eq!(decl.item, "item");
                assert!(matches!(decl.item_type, TypeExpr::Named(ref n) if n == "String"));
                assert_eq!(decl.body.len(), 1);
            }
            _ => panic!("expected Each, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 3. Function parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fn_pure_no_params() {
        let source = "\
+fn greet ()->String
  +return \"hello\"
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "greet");
                assert!(decl.params.is_empty());
                assert!(matches!(decl.return_type, TypeExpr::Named(ref n) if n == "String"));
                assert!(decl.effects.is_empty());
                assert_eq!(decl.body.len(), 1);
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_with_params() {
        let source = "\
+fn add (a:Int, b:Int)->Int
  +return a + b
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "add");
                assert_eq!(decl.params.len(), 2);
                assert_eq!(decl.params[0].name, "a");
                assert!(matches!(decl.params[0].ty, TypeExpr::Named(ref n) if n == "Int"));
                assert_eq!(decl.params[1].name, "b");
                assert!(matches!(decl.params[1].ty, TypeExpr::Named(ref n) if n == "Int"));
                assert!(matches!(decl.return_type, TypeExpr::Named(ref n) if n == "Int"));
                assert!(decl.effects.is_empty());
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_with_effects() {
        let source = "\
+fn fetch (url:String)->String [io,async]
  +await data:String = http_get(url)
  +return data
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "fetch");
                assert_eq!(decl.effects.len(), 2);
                assert!(decl.effects.contains(&"io".to_string()));
                assert!(decl.effects.contains(&"async".to_string()));
            }
            _ => panic!("expected Function with effects, got {op:?}"),
        }
    }

    #[test]
    fn fn_result_return_type() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check positive x > 0 ~err_negative
  +return x
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "validate");
                match &decl.return_type {
                    TypeExpr::Generic { name, args } => {
                        assert_eq!(name, "Result");
                        assert_eq!(args.len(), 1);
                        assert!(matches!(&args[0], TypeExpr::Named(n) if n == "Int"));
                    }
                    _ => panic!("expected Result<Int>, got {:?}", decl.return_type),
                }
                assert_eq!(decl.effects, vec!["fail"]);
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_list_return_type() {
        let source = "\
+fn get_items ()->List<String>
  +return list()
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => match &decl.return_type {
                TypeExpr::Generic { name, args } => {
                    assert_eq!(name, "List");
                    assert_eq!(args.len(), 1);
                    assert!(matches!(&args[0], TypeExpr::Named(n) if n == "String"));
                }
                _ => panic!("expected List<String>, got {:?}", decl.return_type),
            },
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_empty_body() {
        // Functions with no body statements (just the +end)
        let source = "\
+fn noop ()->Int
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "noop");
                assert!(decl.body.is_empty());
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_body_without_explicit_end() {
        // When +end is omitted, the body ends at dedent / next top-level item.
        // In standalone parse, the function body includes all indented lines.
        let source = "\
+fn double (x:Int)->Int
  +let result:Int = x * 2
  +return result
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.name, "double");
                assert_eq!(decl.body.len(), 2);
                assert!(matches!(decl.body[0], Operation::Let(_)));
                assert!(matches!(decl.body[1], Operation::Return(_)));
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn fn_multiple_effects() {
        let source = "\
+fn do_stuff ()->String [io,async,fail,mut]
  +return \"done\"
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects.len(), 4);
                assert!(decl.effects.contains(&"io".to_string()));
                assert!(decl.effects.contains(&"async".to_string()));
                assert!(decl.effects.contains(&"fail".to_string()));
                assert!(decl.effects.contains(&"mut".to_string()));
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 4. Type parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn type_struct() {
        let op = parse_one("+type User = id:Int, name:String");
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "User");
                match decl.body {
                    TypeBody::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name, "id");
                        assert!(matches!(fields[0].ty, TypeExpr::Named(ref n) if n == "Int"));
                        assert_eq!(fields[1].name, "name");
                        assert!(matches!(fields[1].ty, TypeExpr::Named(ref n) if n == "String"));
                    }
                    _ => panic!("expected struct TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn type_union_variants() {
        let op = parse_one("+type Color = Red | Green | Blue");
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Color");
                match decl.body {
                    TypeBody::Union(variants) => {
                        assert_eq!(variants.len(), 3);
                        assert_eq!(variants[0].name, "Red");
                        assert!(variants[0].payload.is_empty());
                        assert_eq!(variants[1].name, "Green");
                        assert_eq!(variants[2].name, "Blue");
                    }
                    _ => panic!("expected union TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn type_union_with_payload() {
        let op = parse_one("+type Shape = Circle(Float) | Rect(Float, Float) | Point");
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Shape");
                match decl.body {
                    TypeBody::Union(variants) => {
                        assert_eq!(variants.len(), 3);
                        assert_eq!(variants[0].name, "Circle");
                        assert_eq!(variants[0].payload.len(), 1);
                        assert_eq!(variants[1].name, "Rect");
                        assert_eq!(variants[1].payload.len(), 2);
                        assert_eq!(variants[2].name, "Point");
                        assert!(variants[2].payload.is_empty());
                    }
                    _ => panic!("expected union TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn type_recursive() {
        let op = parse_one("+type Expr = Literal(Int) | Add(Expr, Expr) | Mul(Expr, Expr)");
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Expr");
                match decl.body {
                    TypeBody::Union(variants) => {
                        assert_eq!(variants.len(), 3);
                        assert_eq!(variants[0].name, "Literal");
                        assert_eq!(variants[1].name, "Add");
                        assert_eq!(variants[1].payload.len(), 2);
                        // Both payload types should be Named("Expr") — recursive
                        assert!(
                            matches!(&variants[1].payload[0], TypeExpr::Named(n) if n == "Expr")
                        );
                        assert!(
                            matches!(&variants[1].payload[1], TypeExpr::Named(n) if n == "Expr")
                        );
                    }
                    _ => panic!("expected union TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn type_struct_with_generic_field() {
        let op = parse_one("+type State = items:List<String>, count:Int");
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "State");
                match decl.body {
                    TypeBody::Struct(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name, "items");
                        match &fields[0].ty {
                            TypeExpr::Generic { name, args } => {
                                assert_eq!(name, "List");
                                assert_eq!(args.len(), 1);
                            }
                            _ => panic!("expected List<String> generic"),
                        }
                    }
                    _ => panic!("expected struct TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 5. Module parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn module_with_function() {
        let source = "\
!module MyModule
+fn greet ()->String
  +return \"hello\"
";
        let op = parse_one(source);
        match op {
            Operation::Module(decl) => {
                assert_eq!(decl.name, "MyModule");
                assert_eq!(decl.body.len(), 1);
                assert!(matches!(decl.body[0], Operation::Function(_)));
            }
            _ => panic!("expected Module, got {op:?}"),
        }
    }

    #[test]
    fn module_with_type_and_function() {
        let source = "\
!module Users
+type User = id:Int, name:String
+fn create (name:String)->User
  +let u:User = {id: 1, name: name}
  +return u
";
        let op = parse_one(source);
        match op {
            Operation::Module(decl) => {
                assert_eq!(decl.name, "Users");
                assert_eq!(decl.body.len(), 2);
                assert!(matches!(decl.body[0], Operation::Type(_)));
                assert!(matches!(decl.body[1], Operation::Function(_)));
            }
            _ => panic!("expected Module, got {op:?}"),
        }
    }

    #[test]
    fn module_stops_at_next_module() {
        let source = "\
!module A
+fn a_fn ()->Int
  +return 1

!module B
+fn b_fn ()->Int
  +return 2
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operation::Module(m) => {
                assert_eq!(m.name, "A");
                assert_eq!(m.body.len(), 1);
            }
            _ => panic!("expected Module A"),
        }
        match &ops[1] {
            Operation::Module(m) => {
                assert_eq!(m.name, "B");
                assert_eq!(m.body.len(), 1);
            }
            _ => panic!("expected Module B"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 6. Route parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn route_post() {
        let op = parse_one(r#"+route POST "/webhook/telegram" -> handle_telegram"#);
        match op {
            Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                assert_eq!(method, "POST");
                assert_eq!(path, "/webhook/telegram");
                assert_eq!(handler_fn, "handle_telegram");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    #[test]
    fn route_get() {
        let op = parse_one(r#"+route GET "/health" -> health_check"#);
        match op {
            Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                assert_eq!(method, "GET");
                assert_eq!(path, "/health");
                assert_eq!(handler_fn, "health_check");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 7. Test / Eval / Mock / Command parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn test_block_simple() {
        let source = "\
!test double
  +with 5 -> expect 10
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.function_name, "double");
                assert_eq!(test.cases.len(), 1);
                assert!(matches!(test.cases[0].input, Expr::Int(5)));
                assert!(matches!(test.cases[0].expected, Expr::Int(10)));
            }
            _ => panic!("expected Test, got {op:?}"),
        }
    }

    #[test]
    fn test_block_key_value() {
        let source = "\
!test add
  +with a=3 b=4 -> expect 7
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.function_name, "add");
                assert_eq!(test.cases.len(), 1);
                match &test.cases[0].input {
                    Expr::StructLiteral(fields) => {
                        assert_eq!(fields.len(), 2);
                        assert_eq!(fields[0].name, "a");
                        assert_eq!(fields[1].name, "b");
                    }
                    _ => panic!("expected struct literal input"),
                }
            }
            _ => panic!("expected Test, got {op:?}"),
        }
    }

    #[test]
    fn test_block_multiple_cases() {
        let source = "\
!test validate
  +with name=\"alice\" age=25 -> expect Ok
  +with name=\"\" age=25 -> expect Err
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.function_name, "validate");
                assert_eq!(test.cases.len(), 2);
            }
            _ => panic!("expected Test, got {op:?}"),
        }
    }

    #[test]
    fn test_with_matcher() {
        let source = "\
!test fetch
  +with url=\"http://example.com\" -> expect contains(\"hello\")
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.cases.len(), 1);
                assert!(test.cases[0].matcher.is_some());
                match &test.cases[0].matcher {
                    Some(TestMatcher::Contains(s)) => assert_eq!(s, "hello"),
                    _ => panic!("expected Contains matcher"),
                }
            }
            _ => panic!("expected Test, got {op:?}"),
        }
    }

    #[test]
    fn eval_function_name() {
        let op = parse_one("!eval my_func");
        match op {
            Operation::Eval(ev) => {
                assert_eq!(ev.function_name, "my_func");
                assert!(ev.inline_expr.is_none());
            }
            _ => panic!("expected Eval, got {op:?}"),
        }
    }

    #[test]
    fn eval_inline_expression() {
        let op = parse_one("!eval 1 + 2");
        match op {
            Operation::Eval(ev) => {
                assert!(ev.inline_expr.is_some());
                assert!(matches!(
                    ev.inline_expr.as_ref().unwrap(),
                    Expr::Binary {
                        op: BinaryOp::Add,
                        ..
                    }
                ));
            }
            _ => panic!("expected Eval, got {op:?}"),
        }
    }

    #[test]
    fn eval_function_with_args() {
        let op = parse_one("!eval add a=3 b=4");
        match op {
            Operation::Eval(ev) => {
                assert_eq!(ev.function_name, "add");
                assert!(ev.inline_expr.is_none());
            }
            _ => panic!("expected Eval, got {op:?}"),
        }
    }

    #[test]
    fn mock_operation() {
        let op = parse_one(r#"!mock http_get "http://example.com" -> "hello""#);
        match op {
            Operation::Mock {
                operation,
                patterns,
                response,
            } => {
                assert_eq!(operation, "http_get");
                assert_eq!(patterns, vec!["http://example.com"]);
                assert_eq!(response, "hello");
            }
            _ => panic!("expected Mock, got {op:?}"),
        }
    }

    #[test]
    fn query_passthrough() {
        let op = parse_one("?symbols");
        match op {
            Operation::Query(q) => assert_eq!(q, "?symbols"),
            _ => panic!("expected Query, got {op:?}"),
        }
    }

    #[test]
    fn remove_operation() {
        let op = parse_one("!remove MyModule.my_func");
        match op {
            Operation::Remove(target) => assert_eq!(target, "MyModule.my_func"),
            _ => panic!("expected Remove, got {op:?}"),
        }
    }

    #[test]
    fn done_operation() {
        let op = parse_one("!done");
        assert!(matches!(op, Operation::Done));
    }

    // ═════════════════════════════════════════════════════════════════════
    // 8. Edge cases
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn comments_are_skipped() {
        let source = "\
// This is a comment
+let x:Int = 42
# Another comment style
+let y:Int = 7
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 2);
        assert!(matches!(&ops[0], Operation::Let(d) if d.name == "x"));
        assert!(matches!(&ops[1], Operation::Let(d) if d.name == "y"));
    }

    #[test]
    fn empty_source() {
        let ops = parse("").unwrap();
        assert!(ops.is_empty());
    }

    #[test]
    fn blank_lines_skipped() {
        let source = "\n\n+let x:Int = 1\n\n\n";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 1);
    }

    #[test]
    fn nested_if_in_function() {
        let source = "\
+fn classify (x:Int)->String
  +if x > 0
    +if x > 100
      +return \"big\"
    +else
      +return \"small\"
    +end
  +else
    +return \"negative\"
  +end
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.body.len(), 1);
                match &decl.body[0] {
                    Operation::If(outer) => {
                        assert_eq!(outer.then_body.len(), 1);
                        assert!(matches!(outer.then_body[0], Operation::If(_)));
                        assert_eq!(outer.else_body.len(), 1);
                    }
                    _ => panic!("expected nested If"),
                }
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn while_in_function() {
        let source = "\
+fn count_to (n:Int)->Int [mut]
  +let i:Int = 0
  +while i < n
    +set i = i + 1
  +end
  +return i
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.body.len(), 3); // let, while, return
                assert!(matches!(decl.body[1], Operation::While(_)));
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn each_in_function() {
        let source = "\
+fn process_all (items:List<String>)->Int
  +let count:Int = 0
  +each items item:String
    +set count = count + 1
  +end
  +return count
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.body.len(), 3); // let, each, return
                assert!(matches!(decl.body[1], Operation::Each(_)));
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn match_in_function() {
        let source = "\
+fn eval_expr (e:Expr)->Int
  +match e
  +case Literal(val)
    +return val
  +case Add(left, right)
    +let l:Int = eval_expr(left)
    +let r:Int = eval_expr(right)
    +return l + r
  +end
+end";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.body.len(), 1); // just the match
                match &decl.body[0] {
                    Operation::Match(m) => {
                        assert_eq!(m.arms.len(), 2);
                        assert_eq!(m.arms[0].variant, "Literal");
                        assert_eq!(m.arms[0].bindings, vec!["val"]);
                        assert_eq!(m.arms[1].variant, "Add");
                        assert_eq!(m.arms[1].bindings, vec!["left", "right"]);
                        assert_eq!(m.arms[1].body.len(), 3); // let, let, return
                    }
                    _ => panic!("expected Match"),
                }
            }
            _ => panic!("expected Function, got {op:?}"),
        }
    }

    #[test]
    fn multiline_type_continuation() {
        // Type declarations can span multiple lines (continuation line joined with comma)
        let source = "\
+type Config = host:String, port:Int
  timeout:Int, debug:Bool
";
        let op = parse_one(source);
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Config");
                match decl.body {
                    TypeBody::Struct(fields) => {
                        assert_eq!(fields.len(), 4);
                        assert_eq!(fields[0].name, "host");
                        assert_eq!(fields[1].name, "port");
                        assert_eq!(fields[2].name, "timeout");
                        assert_eq!(fields[3].name, "debug");
                    }
                    _ => panic!("expected struct TypeBody"),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn shared_var_declaration() {
        let op = parse_one("+shared count:Int = 0");
        match op {
            Operation::SharedVar(decl) => {
                assert_eq!(decl.name, "count");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "Int"));
                assert!(matches!(decl.default, Expr::Int(0)));
            }
            _ => panic!("expected SharedVar, got {op:?}"),
        }
    }

    #[test]
    fn move_operation() {
        let op = parse_one("!move add subtract Math");
        match op {
            Operation::Move {
                function_names,
                target_module,
            } => {
                assert_eq!(function_names, vec!["add", "subtract"]);
                assert_eq!(target_module, "Math");
            }
            _ => panic!("expected Move, got {op:?}"),
        }
    }

    #[test]
    fn replace_mutation() {
        let source = "\
!replace my_func.s1
  +check positive x > 0 ~err_neg
";
        let op = parse_one(source);
        match op {
            Operation::Replace(r) => {
                assert_eq!(r.target, "my_func.s1");
                assert_eq!(r.body.len(), 1);
                assert!(matches!(r.body[0], Operation::Check(_)));
            }
            _ => panic!("expected Replace, got {op:?}"),
        }
    }

    #[test]
    fn code_fences_stripped() {
        // Code fences (```) should be ignored by the parser
        let source = "```\n+let x:Int = 1\n```";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 1);
        assert!(matches!(&ops[0], Operation::Let(d) if d.name == "x"));
    }

    #[test]
    fn full_program_multiple_ops() {
        let source = "\
+type User = id:Int, name:String

+fn create_user (name:String)->User
  +let u:User = {id: 1, name: name}
  +return u

!test create_user
  +with name=\"alice\" -> expect {id: 1, name: \"alice\"}
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 3);
        assert!(matches!(&ops[0], Operation::Type(_)));
        assert!(matches!(&ops[1], Operation::Function(_)));
        assert!(matches!(&ops[2], Operation::Test(_)));
    }

    // ── parse_test_input tests ───────────────────────────────────────────

    #[test]
    fn test_input_empty() {
        let e = parse_test_input(0, "").unwrap();
        assert!(matches!(e, Expr::StructLiteral(ref f) if f.is_empty()));
    }

    #[test]
    fn test_input_key_value() {
        let e = parse_test_input(0, "a=3 b=4").unwrap();
        match e {
            Expr::StructLiteral(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "a");
                assert_eq!(fields[1].name, "b");
            }
            _ => panic!("expected StructLiteral, got {e:?}"),
        }
    }

    #[test]
    fn test_input_single_expr() {
        let e = parse_test_input(0, "42").unwrap();
        assert!(matches!(e, Expr::Int(42)));
    }

    #[test]
    fn test_input_string() {
        let e = parse_test_input(0, r#""hello""#).unwrap();
        assert!(matches!(e, Expr::String(ref s) if s == "hello"));
    }

    #[test]
    fn test_input_function_call() {
        let e = parse_test_input(0, r#"concat("a", "b")"#).unwrap();
        assert!(matches!(e, Expr::Call { .. }));
    }

    #[test]
    fn test_input_positional_multiple() {
        // Multiple space-separated values become positional _0, _1, ...
        let e = parse_test_input(0, r#""hello" "world""#).unwrap();
        match e {
            Expr::StructLiteral(fields) => {
                assert_eq!(fields.len(), 2);
                assert_eq!(fields[0].name, "_0");
                assert_eq!(fields[1].name, "_1");
            }
            _ => panic!("expected positional StructLiteral, got {e:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 9. Module parsing (extended)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn module_empty() {
        // Module with no body (next line is a non-module operation)
        let source = "\
!module Empty
!eval greet
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operation::Module(m) => {
                assert_eq!(m.name, "Empty");
                assert!(m.body.is_empty());
            }
            _ => panic!("expected Module"),
        }
        assert!(matches!(&ops[1], Operation::Eval(_)));
    }

    #[test]
    fn module_with_shared_var() {
        let source = "\
!module Counter
+shared count:Int = 0
+fn increment ()->Int
  +set count = count + 1
  +return count
";
        let op = parse_one(source);
        match op {
            Operation::Module(m) => {
                assert_eq!(m.name, "Counter");
                // +shared and +fn should NOT both be in body unless +shared is recognized
                // Actually +shared is recognized inside modules (see parser code line 516-520)
                // But the check is: !next.text.starts_with("+fn") && !starts_with("+type") && !starts_with("+shared")
                // So +shared IS collected. Let's verify:
                assert_eq!(m.body.len(), 2);
                assert!(matches!(&m.body[0], Operation::SharedVar(_)));
                assert!(matches!(&m.body[1], Operation::Function(_)));
            }
            _ => panic!("expected Module, got {op:?}"),
        }
    }

    #[test]
    fn module_with_multiple_types_and_functions() {
        let source = "\
!module Data
+type Point = x:Int, y:Int
+type Color = Red | Green | Blue
+fn origin ()->Point
  +return {x: 0, y: 0}
+fn default_color ()->Color
  +return Red
";
        let op = parse_one(source);
        match op {
            Operation::Module(m) => {
                assert_eq!(m.name, "Data");
                assert_eq!(m.body.len(), 4);
                assert!(matches!(&m.body[0], Operation::Type(_)));
                assert!(matches!(&m.body[1], Operation::Type(_)));
                assert!(matches!(&m.body[2], Operation::Function(_)));
                assert!(matches!(&m.body[3], Operation::Function(_)));
            }
            _ => panic!("expected Module, got {op:?}"),
        }
    }

    #[test]
    fn module_qualified_function_call_expr() {
        // Module.func(x) parses as Call { callee: FieldAccess { Ident("Module"), "func" }, args }
        let e = parse_expr_pub(0, "Math.add(1, 2)").unwrap();
        match e {
            Expr::Call { callee, args } => {
                assert_eq!(args.len(), 2);
                match *callee {
                    Expr::FieldAccess { base, field } => {
                        assert!(matches!(*base, Expr::Ident(ref n) if n == "Math"));
                        assert_eq!(field, "add");
                    }
                    _ => panic!("expected FieldAccess in callee"),
                }
            }
            _ => panic!("expected Call, got {e:?}"),
        }
    }

    #[test]
    fn module_qualified_call_in_statement() {
        let op = parse_one("+call result:Int = Math.add(a, b)");
        match op {
            Operation::Call(decl) => {
                assert_eq!(decl.name, "result");
                match decl.expr {
                    Expr::Call { callee, args } => {
                        assert_eq!(args.len(), 2);
                        assert!(matches!(*callee, Expr::FieldAccess { .. }));
                    }
                    _ => panic!("expected Call expr"),
                }
            }
            _ => panic!("expected Call op, got {op:?}"),
        }
    }

    #[test]
    fn module_qualified_in_let() {
        let op = parse_one("+let p:Point = Geometry.origin()");
        match op {
            Operation::Let(decl) => {
                assert_eq!(decl.name, "p");
                match decl.expr {
                    Expr::Call { callee, .. } => match *callee {
                        Expr::FieldAccess { base, field } => {
                            assert!(matches!(*base, Expr::Ident(ref n) if n == "Geometry"));
                            assert_eq!(field, "origin");
                        }
                        _ => panic!("expected FieldAccess in callee"),
                    },
                    _ => panic!("expected Call expr"),
                }
            }
            _ => panic!("expected Let, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 10. Route parsing (extended)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn route_put() {
        let op = parse_one(r#"+route PUT "/api/users" -> update_user"#);
        match op {
            Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                assert_eq!(method, "PUT");
                assert_eq!(path, "/api/users");
                assert_eq!(handler_fn, "update_user");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    #[test]
    fn route_delete() {
        let op = parse_one(r#"+route DELETE "/api/users" -> delete_user"#);
        match op {
            Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                assert_eq!(method, "DELETE");
                assert_eq!(path, "/api/users");
                assert_eq!(handler_fn, "delete_user");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    #[test]
    fn route_method_lowercased_gets_uppercased() {
        let op = parse_one(r#"+route get "/health" -> health"#);
        match op {
            Operation::Route { method, .. } => {
                assert_eq!(method, "GET");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    #[test]
    fn route_unquoted_path() {
        // Paths can be unquoted too
        let op = parse_one("+route GET /api/status -> get_status");
        match op {
            Operation::Route {
                method,
                path,
                handler_fn,
            } => {
                assert_eq!(method, "GET");
                assert_eq!(path, "/api/status");
                assert_eq!(handler_fn, "get_status");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    #[test]
    fn route_with_complex_path() {
        let op = parse_one(r#"+route POST "/webhook/telegram/v2" -> handle_v2"#);
        match op {
            Operation::Route {
                path, handler_fn, ..
            } => {
                assert_eq!(path, "/webhook/telegram/v2");
                assert_eq!(handler_fn, "handle_v2");
            }
            _ => panic!("expected Route, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 11. Shared variable parsing (extended)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn shared_string_default() {
        let op = parse_one(r#"+shared label:String = "default""#);
        match op {
            Operation::SharedVar(decl) => {
                assert_eq!(decl.name, "label");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "String"));
                assert!(matches!(decl.default, Expr::String(ref s) if s == "default"));
            }
            _ => panic!("expected SharedVar, got {op:?}"),
        }
    }

    #[test]
    fn shared_bool_default() {
        let op = parse_one("+shared running:Bool = false");
        match op {
            Operation::SharedVar(decl) => {
                assert_eq!(decl.name, "running");
                assert!(matches!(decl.ty, TypeExpr::Named(ref n) if n == "Bool"));
                assert!(matches!(decl.default, Expr::Bool(false)));
            }
            _ => panic!("expected SharedVar, got {op:?}"),
        }
    }

    #[test]
    fn shared_with_generic_type() {
        let op = parse_one("+shared items:List<String> = list()");
        match op {
            Operation::SharedVar(decl) => {
                assert_eq!(decl.name, "items");
                match &decl.ty {
                    TypeExpr::Generic { name, args } => {
                        assert_eq!(name, "List");
                        assert_eq!(args.len(), 1);
                    }
                    _ => panic!("expected Generic type"),
                }
                assert!(matches!(decl.default, Expr::Call { .. }));
            }
            _ => panic!("expected SharedVar, got {op:?}"),
        }
    }

    #[test]
    fn shared_with_struct_literal_default() {
        let op = parse_one(r#"+shared config:Config = {host: "localhost", port: 8080}"#);
        match op {
            Operation::SharedVar(decl) => {
                assert_eq!(decl.name, "config");
                assert!(matches!(decl.default, Expr::StructLiteral(_)));
            }
            _ => panic!("expected SharedVar, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 12. Edit operations parsing (extended)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn replace_with_multiple_statements() {
        let source = "\
!replace validate.s1
  +check age input.age >= 0 AND input.age <= 150 ~err_age_range
  +check name len(input.name) > 0 ~err_empty_name
";
        let op = parse_one(source);
        match op {
            Operation::Replace(r) => {
                assert_eq!(r.target, "validate.s1");
                assert_eq!(r.body.len(), 2);
                assert!(matches!(r.body[0], Operation::Check(_)));
                assert!(matches!(r.body[1], Operation::Check(_)));
            }
            _ => panic!("expected Replace, got {op:?}"),
        }
    }

    #[test]
    fn replace_with_let_and_return() {
        let source = "\
!replace my_func.s2
  +let result:Int = x * 2 + 1
  +return result
";
        let op = parse_one(source);
        match op {
            Operation::Replace(r) => {
                assert_eq!(r.target, "my_func.s2");
                assert_eq!(r.body.len(), 2);
                assert!(matches!(r.body[0], Operation::Let(_)));
                assert!(matches!(r.body[1], Operation::Return(_)));
            }
            _ => panic!("expected Replace, got {op:?}"),
        }
    }

    #[test]
    fn remove_module() {
        let op = parse_one("!remove MyModule");
        match op {
            Operation::Remove(target) => assert_eq!(target, "MyModule"),
            _ => panic!("expected Remove, got {op:?}"),
        }
    }

    #[test]
    fn remove_type() {
        let op = parse_one("!remove UserType");
        match op {
            Operation::Remove(target) => assert_eq!(target, "UserType"),
            _ => panic!("expected Remove, got {op:?}"),
        }
    }

    #[test]
    fn remove_route() {
        let op = parse_one("!remove route POST /api/ask");
        match op {
            Operation::RemoveRoute { method, path } => {
                assert_eq!(method, "POST");
                assert_eq!(path, "/api/ask");
            }
            _ => panic!("expected RemoveRoute, got {op:?}"),
        }
    }

    #[test]
    fn remove_route_lowercased_method() {
        let op = parse_one("!remove route get /health");
        match op {
            Operation::RemoveRoute { method, path } => {
                assert_eq!(method, "GET");
                assert_eq!(path, "/health");
            }
            _ => panic!("expected RemoveRoute, got {op:?}"),
        }
    }

    #[test]
    fn unroute_shorthand() {
        let op = parse_one("!unroute GET /api/status");
        match op {
            Operation::RemoveRoute { method, path } => {
                assert_eq!(method, "GET");
                assert_eq!(path, "/api/status");
            }
            _ => panic!("expected RemoveRoute, got {op:?}"),
        }
    }

    #[test]
    fn move_single_function() {
        let op = parse_one("!move validate Utils");
        match op {
            Operation::Move {
                function_names,
                target_module,
            } => {
                assert_eq!(function_names, vec!["validate"]);
                assert_eq!(target_module, "Utils");
            }
            _ => panic!("expected Move, got {op:?}"),
        }
    }

    #[test]
    fn move_multiple_functions() {
        let op = parse_one("!move add subtract multiply Math");
        match op {
            Operation::Move {
                function_names,
                target_module,
            } => {
                assert_eq!(function_names, vec!["add", "subtract", "multiply"]);
                assert_eq!(target_module, "Math");
            }
            _ => panic!("expected Move, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 13. Multi-line type definitions (extended)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn multiline_type_three_lines() {
        let source = "\
+type Config = host:String
  port:Int
  debug:Bool
";
        let op = parse_one(source);
        match op {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Config");
                match decl.body {
                    TypeBody::Struct(fields) => {
                        assert_eq!(fields.len(), 3);
                        assert_eq!(fields[0].name, "host");
                        assert_eq!(fields[1].name, "port");
                        assert_eq!(fields[2].name, "debug");
                    }
                    _ => panic!("expected Struct, got {:?}", decl.body),
                }
            }
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    #[test]
    fn multiline_type_stops_at_blank_line() {
        let source = "\
+type Point = x:Int
  y:Int

+fn origin ()->Point
  +return {x: 0, y: 0}
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operation::Type(decl) => {
                assert_eq!(decl.name, "Point");
                match &decl.body {
                    TypeBody::Struct(fields) => assert_eq!(fields.len(), 2),
                    _ => panic!("expected Struct"),
                }
            }
            _ => panic!("expected Type"),
        }
        assert!(matches!(&ops[1], Operation::Function(_)));
    }

    #[test]
    fn multiline_type_stops_at_plus_operation() {
        let source = "\
+type State = count:Int
  name:String
+fn reset ()->State
  +return {count: 0, name: \"none\"}
";
        let ops = parse_ops(source);
        assert_eq!(ops.len(), 2);
        match &ops[0] {
            Operation::Type(decl) => match &decl.body {
                TypeBody::Struct(fields) => assert_eq!(fields.len(), 2),
                _ => panic!("expected Struct"),
            },
            _ => panic!("expected Type"),
        }
    }

    #[test]
    fn multiline_type_with_leading_comma_stripped() {
        // Continuation lines that start with comma get the comma stripped
        let source = "\
+type Config = host:String
  , port:Int
  , debug:Bool
";
        let op = parse_one(source);
        match op {
            Operation::Type(decl) => match decl.body {
                TypeBody::Struct(fields) => {
                    assert_eq!(fields.len(), 3);
                    assert_eq!(fields[0].name, "host");
                    assert_eq!(fields[1].name, "port");
                    assert_eq!(fields[2].name, "debug");
                }
                _ => panic!("expected Struct"),
            },
            _ => panic!("expected Type, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 14. Union types with various payload counts
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn union_zero_payload_only() {
        let op = parse_one("+type Direction = North | South | East | West");
        match op {
            Operation::Type(decl) => match decl.body {
                TypeBody::Union(variants) => {
                    assert_eq!(variants.len(), 4);
                    for v in &variants {
                        assert!(v.payload.is_empty(), "{} should have no payload", v.name);
                    }
                }
                _ => panic!("expected Union"),
            },
            _ => panic!("expected Type"),
        }
    }

    #[test]
    fn union_mixed_payload_counts() {
        let op = parse_one(
            "+type Value = Null | Bool(Bool) | Pair(String, Int) | Triple(Int, Int, Int)",
        );
        match op {
            Operation::Type(decl) => match decl.body {
                TypeBody::Union(variants) => {
                    assert_eq!(variants.len(), 4);
                    assert_eq!(variants[0].name, "Null");
                    assert_eq!(variants[0].payload.len(), 0);
                    assert_eq!(variants[1].name, "Bool");
                    assert_eq!(variants[1].payload.len(), 1);
                    assert_eq!(variants[2].name, "Pair");
                    assert_eq!(variants[2].payload.len(), 2);
                    assert_eq!(variants[3].name, "Triple");
                    assert_eq!(variants[3].payload.len(), 3);
                }
                _ => panic!("expected Union"),
            },
            _ => panic!("expected Type"),
        }
    }

    #[test]
    fn union_with_generic_payload() {
        let op = parse_one("+type Container = Empty | Single(String) | Many(List<String>)");
        match op {
            Operation::Type(decl) => match decl.body {
                TypeBody::Union(variants) => {
                    assert_eq!(variants.len(), 3);
                    assert_eq!(variants[2].name, "Many");
                    assert_eq!(variants[2].payload.len(), 1);
                    match &variants[2].payload[0] {
                        TypeExpr::Generic { name, args } => {
                            assert_eq!(name, "List");
                            assert_eq!(args.len(), 1);
                        }
                        _ => panic!("expected Generic payload"),
                    }
                }
                _ => panic!("expected Union"),
            },
            _ => panic!("expected Type"),
        }
    }

    #[test]
    fn union_single_variant_needs_pipe() {
        // A single variant requires | to distinguish from struct syntax
        // Wrap(Int) without | is not valid union syntax
        let op = parse_one("+type Wrapper = Wrap(Int) | None");
        match op {
            Operation::Type(decl) => match decl.body {
                TypeBody::Union(variants) => {
                    assert_eq!(variants.len(), 2);
                    assert_eq!(variants[0].name, "Wrap");
                    assert_eq!(variants[0].payload.len(), 1);
                    assert_eq!(variants[1].name, "None");
                    assert!(variants[1].payload.is_empty());
                }
                _ => panic!("expected Union"),
            },
            _ => panic!("expected Type"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 15. Effect combinations
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn fn_single_effect_io() {
        let source = "\
+fn read_file (path:String)->String [io]
  +return path
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects, vec!["io"]);
            }
            _ => panic!("expected Function"),
        }
    }

    #[test]
    fn fn_single_effect_fail() {
        let source = "\
+fn validate (x:Int)->Result<Int> [fail]
  +check pos x > 0 ~err
  +return x
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects, vec!["fail"]);
            }
            _ => panic!("expected Function"),
        }
    }

    #[test]
    fn fn_io_async_combination() {
        let source = "\
+fn fetch (url:String)->String [io,async]
  +await data:String = http_get(url)
  +return data
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects.len(), 2);
                assert!(decl.effects.contains(&"io".to_string()));
                assert!(decl.effects.contains(&"async".to_string()));
            }
            _ => panic!("expected Function"),
        }
    }

    #[test]
    fn fn_io_async_fail_combination() {
        let source = "\
+fn safe_fetch (url:String)->Result<String> [io,async,fail]
  +check valid len(url) > 0 ~err_empty
  +await data:String = http_get(url)
  +return data
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects.len(), 3);
                assert!(decl.effects.contains(&"io".to_string()));
                assert!(decl.effects.contains(&"async".to_string()));
                assert!(decl.effects.contains(&"fail".to_string()));
            }
            _ => panic!("expected Function"),
        }
    }

    #[test]
    fn fn_all_common_effects() {
        let source = "\
+fn do_everything ()->String [io,async,fail,mut,rand]
  +return \"done\"
";
        let op = parse_one(source);
        match op {
            Operation::Function(decl) => {
                assert_eq!(decl.effects.len(), 5);
                for eff in &["io", "async", "fail", "mut", "rand"] {
                    assert!(
                        decl.effects.contains(&eff.to_string()),
                        "missing effect: {eff}"
                    );
                }
            }
            _ => panic!("expected Function"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 16. Plan and Roadmap parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn plan_show() {
        let op = parse_one("!plan");
        assert!(matches!(op, Operation::Plan(PlanAction::Show)));
    }

    #[test]
    fn plan_set_with_continuation_lines() {
        let source = "\
!plan set
Define types
Write functions
Add tests
";
        let op = parse_one(source);
        match op {
            Operation::Plan(PlanAction::Set(steps)) => {
                assert_eq!(steps.len(), 3);
                assert_eq!(steps[0], "Define types");
                assert_eq!(steps[1], "Write functions");
                assert_eq!(steps[2], "Add tests");
            }
            _ => panic!("expected Plan Set, got {op:?}"),
        }
    }

    #[test]
    fn plan_set_strips_numbering() {
        let source = "\
!plan set
1. First step
2. Second step
3. Third step
";
        let op = parse_one(source);
        match op {
            Operation::Plan(PlanAction::Set(steps)) => {
                assert_eq!(steps.len(), 3);
                assert_eq!(steps[0], "First step");
                assert_eq!(steps[1], "Second step");
                assert_eq!(steps[2], "Third step");
            }
            _ => panic!("expected Plan Set, got {op:?}"),
        }
    }

    #[test]
    fn plan_done() {
        let op = parse_one("!plan done 2");
        match op {
            Operation::Plan(PlanAction::Progress(n)) => assert_eq!(n, 2),
            _ => panic!("expected Plan Progress, got {op:?}"),
        }
    }

    #[test]
    fn plan_fail() {
        let op = parse_one("!plan fail 3");
        match op {
            Operation::Plan(PlanAction::Fail(n)) => assert_eq!(n, 3),
            _ => panic!("expected Plan Fail, got {op:?}"),
        }
    }

    #[test]
    fn roadmap_show() {
        let op = parse_one("!roadmap");
        assert!(matches!(op, Operation::Roadmap(RoadmapAction::Show)));

        let op = parse_one("!roadmap show");
        assert!(matches!(op, Operation::Roadmap(RoadmapAction::Show)));
    }

    #[test]
    fn roadmap_add() {
        let op = parse_one("!roadmap add Implement user authentication");
        match op {
            Operation::Roadmap(RoadmapAction::Add(desc)) => {
                assert_eq!(desc, "Implement user authentication");
            }
            _ => panic!("expected Roadmap Add, got {op:?}"),
        }
    }

    #[test]
    fn roadmap_done() {
        let op = parse_one("!roadmap done 1");
        match op {
            Operation::Roadmap(RoadmapAction::Done(n)) => assert_eq!(n, 1),
            _ => panic!("expected Roadmap Done, got {op:?}"),
        }
    }

    #[test]
    fn roadmap_remove() {
        let op = parse_one("!roadmap remove 2");
        match op {
            Operation::Roadmap(RoadmapAction::Remove(n)) => assert_eq!(n, 2),
            _ => panic!("expected Roadmap Remove, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 17. Agent parsing
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn agent_basic() {
        let op = parse_one("!agent test write tests for all functions");
        match op {
            Operation::Agent { name, scope, task } => {
                assert_eq!(name, "test");
                assert_eq!(scope, "full"); // default scope
                assert_eq!(task, "write tests for all functions");
            }
            _ => panic!("expected Agent, got {op:?}"),
        }
    }

    #[test]
    fn agent_with_bare_scope() {
        let op = parse_one("!agent refactor --scope read-only reorganize code");
        match op {
            Operation::Agent { name, scope, task } => {
                assert_eq!(name, "refactor");
                assert_eq!(scope, "read-only");
                assert_eq!(task, "reorganize code");
            }
            _ => panic!("expected Agent, got {op:?}"),
        }
    }

    #[test]
    fn agent_with_quoted_scope() {
        let op = parse_one(r#"!agent crypto --scope "module Crypto" rewrite encryption"#);
        match op {
            Operation::Agent { name, scope, task } => {
                assert_eq!(name, "crypto");
                assert_eq!(scope, "module Crypto");
                assert_eq!(task, "rewrite encryption");
            }
            _ => panic!("expected Agent, got {op:?}"),
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // 18. Error recovery — parser produces clear error messages
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn error_module_missing_name() {
        let err = parse("!module").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected module name"), "got: {msg}");
    }

    #[test]
    fn error_replace_missing_target() {
        let err = parse("!replace\n  +return 1\n").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("expected replace target"), "got: {msg}");
    }

    #[test]
    fn error_remove_missing_target() {
        let err = parse("!remove").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("!remove requires a target"), "got: {msg}");
    }

    #[test]
    fn error_remove_route_missing_path() {
        let err = parse("!remove route POST").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("!remove route requires METHOD and path"),
            "got: {msg}"
        );
    }

    #[test]
    fn error_move_needs_at_least_two_args() {
        let err = parse("!move validate").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("!move function_name(s) ModuleName"),
            "got: {msg}"
        );
    }

    #[test]
    fn error_route_missing_arrow() {
        let err = parse(r#"+route POST "/api/test""#).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("->"), "got: {msg}");
    }

    #[test]
    fn error_route_missing_handler() {
        let err = parse(r#"+route POST "/api/test" ->"#).unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("handler function name"), "got: {msg}");
    }

    #[test]
    fn error_shared_missing_equals() {
        let err = parse("+shared count:Int").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("name:Type = expr"), "got: {msg}");
    }

    #[test]
    fn error_shared_missing_type() {
        let err = parse("+shared count = 0").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("name:Type"), "got: {msg}");
    }

    #[test]
    fn error_unroute_missing_path() {
        let err = parse("!unroute GET").unwrap_err();
        let msg = format!("{err}");
        assert!(
            msg.contains("!unroute requires METHOD and path"),
            "got: {msg}"
        );
    }

    #[test]
    fn error_test_missing_function_name() {
        let err = parse("!test\n  +with 1 -> expect 2\n").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("function name"), "got: {msg}");
    }

    #[test]
    fn error_expr_incomplete_binary() {
        let err = parse_expr_pub(0, "1 +");
        assert!(err.is_err());
    }

    #[test]
    fn error_expr_unmatched_paren() {
        let err = parse_expr_pub(0, "(1 + 2");
        assert!(err.is_err());
    }

    #[test]
    fn error_expr_empty_input() {
        let err = parse_expr_pub(0, "");
        assert!(err.is_err());
    }

    // ═════════════════════════════════════════════════════════════════════
    // 19. Miscellaneous edge cases
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn opencode_command() {
        let op = parse_one("!opencode Add a new builtin for file reading");
        match op {
            Operation::OpenCode(desc) => {
                assert_eq!(desc.trim(), "Add a new builtin for file reading");
            }
            _ => panic!("expected OpenCode, got {op:?}"),
        }
    }

    #[test]
    fn unmock_command() {
        let op = parse_one("!unmock");
        assert!(matches!(op, Operation::Unmock));
    }

    #[test]
    fn undo_command() {
        let op = parse_one("!undo");
        assert!(matches!(op, Operation::Undo));
    }

    #[test]
    fn message_command() {
        let op = parse_one("!msg worker Start processing the queue");
        match op {
            Operation::Message { to, content } => {
                assert_eq!(to, "worker");
                assert_eq!(content, "Start processing the queue");
            }
            _ => panic!("expected Message, got {op:?}"),
        }
    }

    #[test]
    fn mock_with_multiple_patterns() {
        let op = parse_one(r#"!mock http_get "http://a.com" "http://b.com" -> "response""#);
        match op {
            Operation::Mock {
                operation,
                patterns,
                response,
            } => {
                assert_eq!(operation, "http_get");
                assert_eq!(patterns.len(), 2);
                assert_eq!(patterns[0], "http://a.com");
                assert_eq!(patterns[1], "http://b.com");
                assert_eq!(response, "response");
            }
            _ => panic!("expected Mock, got {op:?}"),
        }
    }

    #[test]
    fn query_source() {
        let op = parse_one("?source MyModule.my_func");
        match op {
            Operation::Query(q) => assert_eq!(q, "?source MyModule.my_func"),
            _ => panic!("expected Query, got {op:?}"),
        }
    }

    #[test]
    fn query_deps() {
        let op = parse_one("?deps");
        match op {
            Operation::Query(q) => assert_eq!(q, "?deps"),
            _ => panic!("expected Query, got {op:?}"),
        }
    }

    #[test]
    fn test_with_after_assertion() {
        let source = "\
!test setup
  +with -> expect contains(\"ready\")
  +after routes contains \"/chat\"
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.cases.len(), 1);
                assert_eq!(test.cases[0].after_checks.len(), 1);
                assert_eq!(test.cases[0].after_checks[0].target, "routes");
                assert_eq!(test.cases[0].after_checks[0].matcher, "contains");
                assert_eq!(test.cases[0].after_checks[0].value, "/chat");
            }
            _ => panic!("expected Test, got {op:?}"),
        }
    }

    #[test]
    fn test_starts_with_matcher_parse() {
        let source = "\
!test greet
  +with name=\"alice\" -> expect starts_with(\"Hello\")
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.cases.len(), 1);
                match &test.cases[0].matcher {
                    Some(TestMatcher::StartsWith(s)) => assert_eq!(s, "Hello"),
                    _ => panic!("expected StartsWith matcher"),
                }
            }
            _ => panic!("expected Test"),
        }
    }

    #[test]
    fn test_err_matcher_parse() {
        let source = "\
!test validate
  +with x=-1 -> expect Err(\"negative\")
";
        let op = parse_one(source);
        match op {
            Operation::Test(test) => {
                assert_eq!(test.cases.len(), 1);
                match &test.cases[0].matcher {
                    Some(TestMatcher::ErrContaining(s)) => assert_eq!(s, "negative"),
                    _ => panic!(
                        "expected ErrContaining matcher, got {:?}",
                        test.cases[0].matcher
                    ),
                }
            }
            _ => panic!("expected Test"),
        }
    }
}
