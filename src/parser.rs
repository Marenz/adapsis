use anyhow::{anyhow, bail, Context, Result};

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
    Undo,
    Agent {
        name: String,
        scope: String,
        task: String,
    },
    OpenCode(String),
    Query(String),
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

    fn parse_block(&mut self, indent: usize, _mode: BlockMode) -> Result<Vec<Operation>> {
        let mut ops = Vec::new();

        while let Some(line) = self.current() {
            if line.indent < indent {
                break;
            }

            if line.indent > indent {
                bail!("line {}: unexpected indentation", line.number);
            }

            // Skip stray `end` tokens (LLMs often add these after test blocks)
            if line.text == "end" {
                self.index += 1;
                continue;
            }

            ops.push(self.parse_operation(indent)?);
        }

        Ok(ops)
    }

    fn parse_operation(&mut self, indent: usize) -> Result<Operation> {
        let line = self.current().context("internal parser state error")?;
        let text = line.text;

        if let Some(rest) = text.strip_prefix("+module") {
            let name = rest.trim();
            if name.is_empty() {
                bail!("line {}: expected module name", line.number);
            }
            self.index += 1;
            let body = self.parse_module_body(indent)?;
            return Ok(Operation::Module(ModuleDecl {
                name: name.to_string(),
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("+type") {
            let decl = parse_type_decl(line.number, rest.trim())?;
            self.index += 1;
            return Ok(Operation::Type(decl));
        }

        if let Some(rest) = text.strip_prefix("+fn") {
            let header = parse_function_header(line.number, rest.trim())?;
            self.index += 1;
            let body = self.parse_nested_block(indent)?;
            return Ok(Operation::Function(FunctionDecl {
                name: header.name,
                params: header.params,
                return_type: header.return_type,
                effects: header.effects,
                body,
            }));
        }

        if let Some(rest) = text.strip_prefix("+let") {
            let decl = parse_binding_decl(line.number, rest.trim(), false)?;
            self.index += 1;
            return Ok(Operation::Let(decl));
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
            return Ok(Operation::While(WhileDecl { condition, body }));
        }

        if let Some(rest) = text.strip_prefix("+match") {
            let expr = parse_expr(line.number, rest.trim())?;
            self.index += 1;

            // Parse +case arms at the same indent level
            let mut arms = Vec::new();
            loop {
                let next = match self.current() {
                    Some(l) if l.indent == indent => l,
                    _ => break,
                };
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
            let expr = parse_expr(line.number, rest.trim())?;
            self.index += 1;
            return Ok(Operation::Spawn(SpawnDecl { call: expr }));
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

            // Parse +elif and +else at the same indentation
            loop {
                let next = match self.current() {
                    Some(l) if l.indent == indent => l,
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
            // Parse: !eval function_name arg1 arg2  OR  !eval function_name key=val key=val
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
            }));
        }

        if text.starts_with('?') {
            // Semantic query — pass through as-is
            let query = text.to_string();
            self.index += 1;
            return Ok(Operation::Query(query));
        }

        bail!("line {}: unknown operation `{}`", line.number, line.text)
    }

    fn parse_test_cases(&mut self, parent_indent: usize) -> Result<Vec<TestCase>> {
        let mut cases = Vec::new();
        let Some(indent) = self.child_indent(parent_indent) else {
            let line = self
                .lines
                .get(self.index.saturating_sub(1))
                .map(|line| line.number)
                .unwrap_or(1);
            bail!(
                "line {}: !test requires at least one indented `+with` case",
                line
            );
        };

        while let Some(line) = self.current() {
            if line.indent < indent {
                break;
            }

            if line.indent > indent {
                bail!("line {}: unexpected indentation in test block", line.number);
            }

            let rest = line
                .text
                .strip_prefix("+with")
                .ok_or_else(|| anyhow!("line {}: expected `+with` test case", line.number))?
                .trim();

            let (input_text, expected_text) = split_test_case(line.number, rest)?;
            let input = parse_test_input(line.number, input_text.trim())?;
            let expected = parse_expr(line.number, expected_text.trim())?;
            cases.push(TestCase { input, expected });
            self.index += 1;
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

    fn parse_module_body(&mut self, module_indent: usize) -> Result<Vec<Operation>> {
        let mut ops = Vec::new();
        let child_indent = self.child_indent(module_indent);

        while let Some(line) = self.current() {
            if line.indent == module_indent && line.text == "end" {
                self.index += 1;
                return Ok(ops);
            }

            let Some(indent) = child_indent else {
                bail!(
                    "line {}: expected `end` after module declaration",
                    line.number
                );
            };

            if line.indent < indent {
                bail!("line {}: module block is missing `end`", line.number);
            }

            if line.indent != indent {
                bail!("line {}: unexpected indentation inside module", line.number);
            }

            ops.push(self.parse_operation(indent)?);
        }

        let line = self.lines.last().map(|line| line.number).unwrap_or(1);
        bail!("line {}: module block is missing `end`", line)
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

    if value.starts_with('{') {
        let ty = parse_type(line, value)?;
        let TypeExpr::Struct(fields) = ty else {
            bail!("line {}: invalid struct type declaration", line);
        };
        return Ok(TypeDecl {
            name: name.to_string(),
            body: TypeBody::Struct(fields),
        });
    }

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

    Ok(TypeDecl {
        name: name.to_string(),
        body: TypeBody::Alias(parse_type(line, value)?),
    })
}

fn parse_function_header(line: usize, input: &str) -> Result<FunctionHeader> {
    let (name, rest) =
        take_ident(input).ok_or_else(|| anyhow!("line {}: expected function name", line))?;
    let rest = rest.trim();

    let params_start = rest
        .find('(')
        .ok_or_else(|| anyhow!("line {}: expected parameter list", line))?;
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
fn parse_test_input(line: usize, input: &str) -> Result<Expr> {
    let input = input.trim();
    if input.is_empty() {
        bail!("line {}: empty test input", line);
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
        // Regular expression format
        parse_expr(line, input)
    }
}

/// Parse a single value from key=value format, returning (value, remaining_input).
fn parse_test_value(line: usize, input: &str) -> Result<(Expr, &str)> {
    let input = input.trim_start();

    if input.starts_with('"') {
        // Quoted string — find closing quote
        let end = input[1..]
            .find('"')
            .ok_or_else(|| anyhow!("line {}: unterminated string in test value", line))?;
        let s = &input[1..end + 1];
        let rest = &input[end + 2..];
        Ok((Expr::String(s.to_string()), rest))
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
        // Number, bool, or identifier — read until whitespace
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
            // Treat as identifier
            Expr::Ident(token.to_string())
        };

        Ok((expr, rest))
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
