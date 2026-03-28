use crate::eval::Value;

#[derive(Debug, Clone)]
pub enum Op {
    LoadConst(Value),
    LoadLocal(usize),
    StoreLocal(usize),
    Pop,
    Return,
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Not,
    Negate,
    Jump(usize),
    JumpIfFalse(usize),
    Call(String, usize),
    FieldAccess(String),
    BuildStruct(Vec<String>),
    BuildList(usize),
    CheckFail(String),
    Concat(usize),
}

#[derive(Debug, Clone)]
pub struct Chunk {
    pub name: String,
    pub ops: Vec<Op>,
    pub local_count: usize,
}

#[derive(Debug, Default)]
pub struct VM {
    stack: Vec<Value>,
    locals: Vec<Value>,
}

impl VM {
    pub fn new() -> Self {
        VM {
            stack: Vec::new(),
            locals: Vec::new(),
        }
    }

    pub fn execute(&mut self, chunk: &Chunk) -> Result<Value, String> {
        self.locals = vec![Value::Int(0); chunk.local_count];
        self.stack.clear();

        let mut ip = 0;
        while ip < chunk.ops.len() {
            match &chunk.ops[ip] {
                Op::LoadConst(v) => self.stack.push(v.clone()),
                Op::Return => return self.stack.pop().ok_or("empty stack on return".into()),
                Op::Pop => {
                    self.stack.pop();
                }
                _ => return Err(format!("unimplemented opcode: {:?}", chunk.ops[ip])),
            }
            ip += 1;
        }

        Err("no return".into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn execute_returns_loaded_constant() {
        let chunk = Chunk {
            name: "const".to_string(),
            ops: vec![Op::LoadConst(Value::Int(42)), Op::Return],
            local_count: 0,
        };
        let mut vm = VM::new();

        let result = vm.execute(&chunk).unwrap();
        match result {
            Value::Int(n) => assert_eq!(n, 42),
            other => panic!("expected int result, got {other:?}"),
        }
    }

    #[test]
    fn execute_return_on_empty_stack_fails() {
        let chunk = Chunk {
            name: "empty".to_string(),
            ops: vec![Op::Return],
            local_count: 0,
        };
        let mut vm = VM::new();

        let err = vm.execute(&chunk).unwrap_err();
        assert_eq!(err, "empty stack on return");
    }

    #[test]
    fn execute_without_return_fails() {
        let chunk = Chunk {
            name: "no_return".to_string(),
            ops: vec![Op::LoadConst(Value::Int(1))],
            local_count: 0,
        };
        let mut vm = VM::new();

        let err = vm.execute(&chunk).unwrap_err();
        assert_eq!(err, "no return");
    }

    #[test]
    fn execute_unimplemented_opcode_fails() {
        let chunk = Chunk {
            name: "add".to_string(),
            ops: vec![Op::Add],
            local_count: 0,
        };
        let mut vm = VM::new();

        let err = vm.execute(&chunk).unwrap_err();
        assert!(err.contains("unimplemented opcode"), "got: {err}");
    }
}
