//! Nesting guard for untrusted Python source (#133).
//!
//! The frontend and everything behind it recurse on the syntax tree: lowering
//! (expressions, statements, patterns, types), canonical emission and its reparse,
//! validation, type checking and evaluation. The rustpython parser itself is
//! table-driven, but dropping the tree it built is recursive too. MEASURED
//! 2026-11-10 on a 2 MiB stack (tokio worker and blocking threads): lowering
//! overflows at about 730 nested calls or blocks and about 1,270 chained
//! operators; parsing (the drop) at about 15,000-31,000 levels. A stack overflow
//! Evaluating an `elif` chain overflows at about 660 branches (each `elif` nests
//! in the previous branch, in the Python tree and in the runtime). A stack
//! overflow aborts the whole process, so depth is bounded BEFORE the parser runs.
//!
//! The scan is lexical and conservative: it never parses, it only maintains an
//! upper bound of the nesting at each point of the source:
//!
//! - one level per open bracket `(`, `[`, `{`;
//! - one level per indentation level (block nesting), and one per `elif` since
//!   the `if` that opened the chain (an `elif` nests in the branch before it);
//! - inside one bracket level, one level per depth-forming token since the last
//!   separator: operators, `.` and the keywords `not and or if else in is for
//!   await yield from`. `,` `;` `:` `=` and `->` separate siblings (elements,
//!   key and value, target and value, parameters and return type), so they
//!   reset that count; the keyword that opens a statement is not counted;
//! - a closed bracket group counts one level plus the deepest point reached
//!   inside it, at the enclosing level: what follows the group (`(a + b) * c`,
//!   `f(x)(y)[z]`) is built on top of everything inside it;
//! - `lambda` counts for the rest of its bracket level (its own parameter commas
//!   must not reset it).
//!
//! f-string replacement fields are bounded separately: their expression text is
//! opaque to this scan (each tree level needs at least one source byte, so the
//! bytes inside the braces bound the depth). f-strings are not lowered at all;
//! that bound only protects the parser and is far below its measured depth.
//!
//! Strings and comments are skipped with the tokenizer's own rules. Where this
//! scan and the real tokenizer could disagree, the source is not valid Python
//! and the parser returns an error without building a deep tree.

/// Maximum nesting accepted by the frontend. More than six times below the
/// smallest measured overflow depth, and half of CPython's own bracket limit (200).
pub(crate) const MAX_NESTING: usize = 100;
/// Most expression bytes inside the `{}` fields of one f-string literal.
pub(super) const MAX_FSTRING_FIELD_BYTES: usize = 2_000;

/// A refusal: byte offset of the token that exceeded a limit, and the message.
#[derive(Debug, PartialEq, Eq)]
pub(super) struct TooDeep {
    pub offset: usize,
    pub message: String,
}

/// One open bracket level (index 0: the current logical line).
#[derive(Default)]
struct Frame {
    /// Depth-forming tokens that a comma does not reset (`lambda`).
    sticky: usize,
    /// Depth-forming tokens since the last comma.
    chain: usize,
    /// Largest `sticky + chain` this level has reached.
    peak: usize,
}

const OPERATORS: &[&str] = &[
    "**=", "//=", ">>=", "<<=", "...", "**", "//", ">>", "<<", "<=", ">=", "==", "!=", "->", ":=", "+=", "-=",
    "*=", "/=", "%=", "&=", "|=", "^=", "@=", "+", "-", "*", "/", "%", "@", "&", "|", "^", "~", "<", ">", ".",
    "!", "=",
];

/// Operators that separate siblings instead of nesting them.
const SEPARATORS: &[&str] = &["=", "->"];
/// Statement keywords that are also depth keywords inside expressions.
const STATEMENT_KEYWORDS: &[&[u8]] = &[b"if", b"elif", b"else", b"for", b"from"];

const DEPTH_KEYWORDS: &[&str] = &["not", "and", "or", "if", "else", "in", "is", "for", "await", "yield", "from"];

struct Scan<'a> {
    bytes: &'a [u8],
    at: usize,
    frames: Vec<Frame>,
    /// Indentation widths of the enclosing blocks.
    indents: Vec<usize>,
    /// `elif` lines since the last `if`, per indentation level.
    elifs: Vec<usize>,
    /// Deepest point seen so far.
    deepest: usize,
}

fn identifier_byte(byte: u8) -> bool {
    byte.is_ascii_alphanumeric() || byte == b'_' || byte >= 0x80
}

impl Scan<'_> {
    fn peek(&self, ahead: usize) -> u8 {
        self.bytes.get(self.at + ahead).copied().unwrap_or(0)
    }

    fn depth(&self) -> usize {
        self.indents.len()
            + self.elifs.iter().sum::<usize>()
            + self.frames.len()
            - 1
            + self.frames.iter().map(|frame| frame.sticky + frame.chain).sum::<usize>()
    }

    fn check(&mut self, offset: usize) -> Result<(), TooDeep> {
        let depth = self.depth();
        self.deepest = self.deepest.max(depth);
        if depth <= MAX_NESTING {
            return Ok(());
        }
        Err(TooDeep {
            offset,
            message: format!(
                "nesting too deep: more than {MAX_NESTING} levels of nested brackets, chained operators, \
                 indented blocks and elif branches at this point (limit {MAX_NESTING}); split the expression \
                 into typed local bindings, or move inner blocks and later branches into helper functions"
            ),
        })
    }

    /// Count `levels` of depth for the token that starts at `offset`.
    fn count(&mut self, offset: usize, levels: usize, sticky: bool) -> Result<(), TooDeep> {
        let frame = self.frames.last_mut().expect("the line frame");
        if sticky {
            frame.sticky += levels;
        } else {
            frame.chain += levels;
        }
        frame.peak = frame.peak.max(frame.sticky + frame.chain);
        self.check(offset)
    }

    /// At the first byte of a line, outside brackets: indentation, then the
    /// statement keyword. Returns false when the line is blank or a comment.
    fn line_start(&mut self) -> Result<bool, TooDeep> {
        let start = self.at;
        while matches!(self.peek(0), b' ' | b'\t' | 0x0c) && self.at < self.bytes.len() {
            self.at += 1;
        }
        if self.at >= self.bytes.len() || matches!(self.peek(0), b'\n' | b'\r' | b'#') {
            return Ok(false);
        }
        let indent = self.at - start;
        while self.indents.last().is_some_and(|&width| width > indent) {
            self.indents.pop();
        }
        if indent > self.indents.last().copied().unwrap_or(0) {
            self.indents.push(indent);
        }
        // `elifs[level]`: elif branches of the chain open at this indentation.
        // Deeper chains ended with the dedent; a new `if` starts a new chain.
        let level = self.indents.len();
        self.elifs.resize(level + 1, 0);
        let word_end = (self.at..self.bytes.len()).find(|&i| !identifier_byte(self.bytes[i])).unwrap_or(self.bytes.len());
        match &self.bytes[self.at..word_end] {
            b"if" => self.elifs[level] = 0,
            b"elif" => self.elifs[level] += 1,
            _ => {}
        }
        self.frames[0] = Frame::default();
        self.check(self.at)?;
        // The keyword that opens the statement is the block, not an operator.
        if STATEMENT_KEYWORDS.contains(&&self.bytes[self.at..word_end]) {
            self.at = word_end;
        }
        Ok(true)
    }

    /// A string literal whose opening quote is at `self.at`. `formatted`: it has
    /// an `f` prefix, so the bytes inside its `{}` fields are expression text.
    fn string(&mut self, formatted: bool) -> Result<(), TooDeep> {
        let start = self.at;
        let quote = self.peek(0);
        let triple = self.peek(1) == quote && self.peek(2) == quote;
        self.at += if triple { 3 } else { 1 };
        let (mut braces, mut field_bytes) = (0usize, 0usize);
        while self.at < self.bytes.len() {
            let byte = self.peek(0);
            if byte == b'\\' {
                self.at += 2;
                continue;
            }
            if byte == quote && (!triple || (self.peek(1) == quote && self.peek(2) == quote)) {
                self.at += if triple { 3 } else { 1 };
                break;
            }
            if byte == b'\n' && !triple {
                break; // unterminated: the tokenizer reports it
            }
            if formatted {
                match byte {
                    b'{' if braces == 0 && self.peek(1) == b'{' => self.at += 1,
                    b'{' => braces += 1,
                    b'}' if braces > 0 => braces -= 1,
                    _ if braces > 0 => field_bytes += 1,
                    _ => {}
                }
            }
            self.at += 1;
        }
        if field_bytes > MAX_FSTRING_FIELD_BYTES {
            return Err(TooDeep {
                offset: start,
                message: format!(
                    "f-string replacement fields too large: more than {MAX_FSTRING_FIELD_BYTES} bytes of expressions \
                     in one f-string (limit {MAX_FSTRING_FIELD_BYTES}); f-strings are not supported, build the text \
                     with typed bindings and concat"
                ),
            });
        }
        Ok(())
    }

    /// A numeric literal: digits, letters, `_`, `.` and an exponent sign.
    fn number(&mut self) {
        let start = self.at;
        while identifier_byte(self.peek(0)) || self.peek(0) == b'.' {
            self.at += 1;
        }
        let hex = self.bytes[start..self.at].len() > 1 && matches!(self.bytes[start + 1], b'x' | b'X');
        if !hex
            && matches!(self.bytes[self.at - 1], b'e' | b'E')
            && matches!(self.peek(0), b'+' | b'-')
            && self.peek(1).is_ascii_digit()
        {
            self.at += 1;
            while identifier_byte(self.peek(0)) {
                self.at += 1;
            }
        }
    }

    /// An identifier, keyword or string prefix starting at `self.at`.
    fn word(&mut self) -> Result<(), TooDeep> {
        let start = self.at;
        while identifier_byte(self.peek(0)) {
            self.at += 1;
        }
        let word = &self.bytes[start..self.at];
        if matches!(self.peek(0), b'"' | b'\'') {
            return self.string(word.iter().any(|byte| byte.eq_ignore_ascii_case(&b'f')));
        }
        if word == b"lambda" {
            return self.count(start, 1, true);
        }
        if DEPTH_KEYWORDS.iter().any(|keyword| keyword.as_bytes() == word) {
            return self.count(start, 1, false);
        }
        Ok(())
    }

    fn operator(&mut self) -> Result<(), TooDeep> {
        let start = self.at;
        let rest = &self.bytes[start..];
        match OPERATORS.iter().find(|operator| rest.starts_with(operator.as_bytes())) {
            Some(operator) => {
                self.at += operator.len();
                if SEPARATORS.contains(operator) {
                    self.frames.last_mut().expect("the line frame").chain = 0;
                    Ok(())
                } else {
                    self.count(start, 1, false)
                }
            }
            // Not Python (`$`, `?`, a stray backslash): the tokenizer reports it.
            None => {
                self.at += 1;
                Ok(())
            }
        }
    }

    fn run(&mut self) -> Result<(), TooDeep> {
        let mut line_start = true;
        while self.at < self.bytes.len() {
            if line_start && self.frames.len() == 1 {
                line_start = !self.line_start()?;
                if line_start {
                    // Blank or comment-only line: skip it whole.
                    while self.at < self.bytes.len() && self.peek(0) != b'\n' {
                        self.at += 1;
                    }
                    self.at += 1;
                }
                continue;
            }
            let (start, byte) = (self.at, self.peek(0));
            match byte {
                b'\n' => {
                    self.at += 1;
                    line_start = self.frames.len() == 1;
                }
                b' ' | b'\t' | b'\r' | 0x0c => self.at += 1,
                // Explicit line joining: the next line continues this one.
                b'\\' if self.peek(1) == b'\n' => self.at += 2,
                b'\\' if self.peek(1) == b'\r' && self.peek(2) == b'\n' => self.at += 3,
                b'#' => {
                    while self.at < self.bytes.len() && self.peek(0) != b'\n' {
                        self.at += 1;
                    }
                }
                b'"' | b'\'' => self.string(false)?,
                b'(' | b'[' | b'{' => {
                    self.frames.push(Frame::default());
                    self.at += 1;
                    self.check(start)?;
                }
                b')' | b']' | b'}' => {
                    self.at += 1;
                    if self.frames.len() > 1 {
                        // The closed group is an operand or a call/subscript link,
                        // and whatever follows it nests everything inside it.
                        let inside = self.frames.pop().map_or(0, |frame| frame.peak);
                        self.count(start, inside + 1, false)?;
                    }
                }
                b',' | b';' => {
                    self.frames.last_mut().expect("the line frame").chain = 0;
                    self.at += 1;
                }
                b':' if self.peek(1) != b'=' => {
                    self.frames.last_mut().expect("the line frame").chain = 0;
                    self.at += 1;
                }
                b'0'..=b'9' => self.number(),
                b'.' if self.peek(1).is_ascii_digit() => self.number(),
                _ if identifier_byte(byte) => self.word()?,
                _ => self.operator()?,
            }
        }
        Ok(())
    }
}

/// Refuse `source` if any point of it nests deeper than the frontend can lower
/// safely. Linear in the source size, no recursion, no allocation per token.
pub(super) fn check(source: &str) -> Result<(), TooDeep> {
    scan(source).run()
}

fn scan(source: &str) -> Scan<'_> {
    Scan { bytes: source.as_bytes(), at: 0, frames: vec![Frame::default()], indents: Vec::new(), elifs: vec![0], deepest: 0 }
}

/// The deepest nesting estimate of an accepted source (test-only: headroom).
#[cfg(test)]
pub(super) fn deepest(source: &str) -> usize {
    let mut scan = scan(source);
    scan.run().expect("accepted source");
    scan.deepest
}

#[cfg(test)]
#[path = "nesting_tests.rs"]
mod tests;
