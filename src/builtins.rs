//! Builtin function registry.
//! Single source of truth for all builtin names and their descriptions.

/// A builtin function entry.
pub struct Builtin {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub description: &'static str,
    pub category: BuiltinCategory,
}

#[derive(Clone, Copy, PartialEq)]
pub enum BuiltinCategory {
    String,
    Math,
    List,
    Bitwise,
    Conversion,
    State,
    Result,
    Regex,
    Io,
}

/// All registered builtins.
pub static BUILTINS: &[Builtin] = &[
    // String
    Builtin {
        name: "concat",
        aliases: &[],
        description: "concatenate two strings",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "char_at",
        aliases: &[],
        description: "get character at index i as String",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "substring",
        aliases: &["substr"],
        description: "get substring from start to end",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "starts_with",
        aliases: &[],
        description: "check if string starts with prefix",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "ends_with",
        aliases: &[],
        description: "check if string ends with suffix",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "contains",
        aliases: &[],
        description: "check if string contains substring",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "index_of",
        aliases: &[],
        description: "find index of substring (-1 if not found)",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "split",
        aliases: &[],
        description: "split string into List<String>",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "trim",
        aliases: &[],
        description: "remove leading/trailing whitespace",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "len",
        aliases: &["length"],
        description: "length of string or list",
        category: BuiltinCategory::String,
    },
    // Conversion
    Builtin {
        name: "to_string",
        aliases: &["str"],
        description: "convert any value to String",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_int",
        aliases: &["parse_int", "int"],
        description: "convert String/Float/Bool to Int",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "digit_value",
        aliases: &[],
        description: "single char to digit Int ('5'->5, 'a'->-1)",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "is_digit_char",
        aliases: &[],
        description: "true if single char is 0-9",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "char_code",
        aliases: &["ord"],
        description: "character to ASCII code Int",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "from_char_code",
        aliases: &["chr"],
        description: "ASCII code to single-char String",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_hex",
        aliases: &[],
        description: "Int to 8-char hex string (32-bit)",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "u32_wrap",
        aliases: &[],
        description: "wrap Int to unsigned 32-bit range",
        category: BuiltinCategory::Conversion,
    },
    // Math
    Builtin {
        name: "abs",
        aliases: &[],
        description: "absolute value",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "sqrt",
        aliases: &[],
        description: "square root",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "pow",
        aliases: &[],
        description: "power (base, exponent)",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "floor",
        aliases: &[],
        description: "floor to integer",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "min",
        aliases: &[],
        description: "minimum of two values",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "max",
        aliases: &[],
        description: "maximum of two values",
        category: BuiltinCategory::Math,
    },
    // Bitwise
    Builtin {
        name: "bit_and",
        aliases: &[],
        description: "bitwise AND",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_or",
        aliases: &[],
        description: "bitwise OR",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_xor",
        aliases: &[],
        description: "bitwise XOR",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_not",
        aliases: &[],
        description: "bitwise NOT",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shl",
        aliases: &["bit_shl"],
        description: "shift left",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shr",
        aliases: &["bit_shr"],
        description: "shift right",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "left_rotate",
        aliases: &["rotl"],
        description: "32-bit left rotation",
        category: BuiltinCategory::Bitwise,
    },
    // List
    Builtin {
        name: "list",
        aliases: &[],
        description: "create empty list or list with items",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "push",
        aliases: &[],
        description: "returns NEW list with item appended",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "get",
        aliases: &[],
        description: "get item at index",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "join",
        aliases: &[],
        description: "join list items into string with delimiter",
        category: BuiltinCategory::List,
    },
    // State
    Builtin {
        name: "state",
        aliases: &[],
        description: "create shared state handle",
        category: BuiltinCategory::State,
    },
    Builtin {
        name: "get_state",
        aliases: &[],
        description: "read shared state",
        category: BuiltinCategory::State,
    },
    Builtin {
        name: "set_state",
        aliases: &[],
        description: "write shared state",
        category: BuiltinCategory::State,
    },
    // Result
    Builtin {
        name: "Ok",
        aliases: &[],
        description: "Result success constructor",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Err",
        aliases: &[],
        description: "Result error constructor",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Some",
        aliases: &[],
        description: "Option value constructor",
        category: BuiltinCategory::Result,
    },
    // Regex
    Builtin {
        name: "regex_match",
        aliases: &[],
        description: "test if text matches regex pattern (Bool)",
        category: BuiltinCategory::Regex,
    },
    Builtin {
        name: "regex_replace",
        aliases: &[],
        description: "regex replace all matches (String)",
        category: BuiltinCategory::Regex,
    },
    // Base64
    Builtin {
        name: "base64_encode",
        aliases: &[],
        description: "encode string to base64",
        category: BuiltinCategory::Conversion,
    },
    // JSON
    Builtin {
        name: "json_get",
        aliases: &[],
        description: "extract value from JSON string by key path: json_get(json, \"key.0.nested\")",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_array_len",
        aliases: &[],
        description: "get length of JSON array: json_array_len(json) -> Int",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_escape",
        aliases: &[],
        description: "escape string for use in JSON value: json_escape(s) -> String",
        category: BuiltinCategory::String,
    },
];

/// IO builtins (used with +await).
pub static IO_BUILTINS: &[Builtin] = &[
    Builtin {
        name: "tcp_listen",
        aliases: &[],
        description: "listen on TCP port, returns handle",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_accept",
        aliases: &[],
        description: "accept connection, returns handle",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_connect",
        aliases: &[],
        description: "connect to TCP server, returns handle",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_read",
        aliases: &[],
        description: "read from connection, returns String",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_write",
        aliases: &[],
        description: "write String to connection",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_close",
        aliases: &[],
        description: "close connection",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_read",
        aliases: &["read_file"],
        description: "read entire file as String",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_write",
        aliases: &["write_file"],
        description: "write String to file",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_exists",
        aliases: &[],
        description: "check if file exists (Bool)",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "list_dir",
        aliases: &[],
        description: "list directory entries as List<String>",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "read_line",
        aliases: &["stdin_read_line"],
        description: "read line from stdin with prompt",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "print",
        aliases: &[],
        description: "print text (no newline)",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "println",
        aliases: &[],
        description: "print text with newline",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "shell_exec",
        aliases: &["exec"],
        description: "run shell command, returns stdout",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "self_restart",
        aliases: &["restart"],
        description: "restart AdapsisOS process",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sleep",
        aliases: &[],
        description: "sleep for milliseconds",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_call",
        aliases: &[],
        description: "single LLM text generation: llm_call(system, prompt[, model])",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_agent",
        aliases: &[],
        description: "full agentic LLM loop until DONE: llm_agent(system, task[, model])",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_get",
        aliases: &[],
        description: "HTTP GET request, returns response body as String",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_post",
        aliases: &[],
        description: "HTTP POST request: http_post(url, body, content_type) -> String",
        category: BuiltinCategory::Io,
    },
];

/// Registered query commands (?-prefixed).
pub struct QueryCommand {
    pub name: &'static str,
    pub args: &'static str,
    pub description: &'static str,
}

pub static QUERIES: &[QueryCommand] = &[
    QueryCommand {
        name: "?symbols",
        args: "[name]",
        description: "list all types and functions, or details of one",
    },
    QueryCommand {
        name: "?source",
        args: "<fn>",
        description: "show reconstructed source code of a function or type",
    },
    QueryCommand {
        name: "?callers",
        args: "<fn>",
        description: "who calls this function",
    },
    QueryCommand {
        name: "?callees",
        args: "<fn>",
        description: "what does this function call (1 level)",
    },
    QueryCommand {
        name: "?deps",
        args: "<fn>",
        description: "direct dependencies (1 level)",
    },
    QueryCommand {
        name: "?deps-all",
        args: "<fn>",
        description: "full transitive dependency tree",
    },
    QueryCommand {
        name: "?deps-modules",
        args: "<fn>",
        description: "dependencies grouped by module",
    },
    QueryCommand {
        name: "?effects",
        args: "<fn>",
        description: "what effects does this function have",
    },
    QueryCommand {
        name: "?type",
        args: "<type>",
        description: "show type definition",
    },
    QueryCommand {
        name: "?agents",
        args: "",
        description: "show status of all agents",
    },
    QueryCommand {
        name: "?plan",
        args: "",
        description: "show current plan (same as !plan)",
    },
    QueryCommand {
        name: "?inbox",
        args: "",
        description: "check messages from other agents or main session",
    },
    QueryCommand {
        name: "?tasks",
        args: "",
        description: "list all spawned async tasks and their current wait state",
    },
];

/// Registered mutation/action commands (!-prefixed).
pub struct ActionCommand {
    pub name: &'static str,
    pub args: &'static str,
    pub description: &'static str,
}

pub static ACTIONS: &[ActionCommand] = &[
    ActionCommand {
        name: "!test",
        args: "<fn>\\n  +with ...",
        description: "run test cases for a function",
    },
    ActionCommand {
        name: "!eval",
        args: "<fn> [args]",
        description: "evaluate a function or builtin",
    },
    ActionCommand {
        name: "!trace",
        args: "<fn> [args]",
        description: "step-by-step execution trace",
    },
    ActionCommand {
        name: "!replace",
        args: "<fn.sN>\\n  ...",
        description: "replace a statement in a function",
    },
    ActionCommand {
        name: "!move",
        args: "<symbols...> <Module>",
        description: "move functions/types into a module (auto-updates call sites)",
    },
    ActionCommand {
        name: "!undo",
        args: "",
        description: "revert last mutation",
    },
    ActionCommand {
        name: "!plan",
        args: "set/done N/fail N/show",
        description:
            "manage task plan. Steps are auto-numbered, do NOT include numbers in step text",
    },
    ActionCommand {
        name: "!watch",
        args: "<fn> [args] <ms>",
        description: "poll a function periodically, alert on change",
    },
    ActionCommand {
        name: "!agent",
        args: "<name> --scope <scope> <task>",
        description: "spawn background agent (scopes: read-only, new-only, module X, full)",
    },
    ActionCommand {
        name: "!mock",
        args: "<operation> \"<pattern>\" -> \"<response>\"",
        description: "register mock IO response for testing. During !test, IO calls matching the pattern return the mock instead of real IO.",
    },
    ActionCommand {
        name: "!unmock",
        args: "",
        description: "clear all IO mocks",
    },
    ActionCommand {
        name: "!msg",
        args: "<agent> <message>",
        description: "send a message to an agent (or 'main' for the main session). Agents see messages in their feedback.",
    },
    ActionCommand {
        name: "!opencode",
        args: "<description>",
        description: "request Rust-level change via OpenCode (rebuild + restart)",
    },
];

/// Get all builtin names (for the validator's shadow check).
pub fn all_builtin_names() -> Vec<&'static str> {
    let mut names = Vec::new();
    for b in BUILTINS {
        names.push(b.name);
        for alias in b.aliases {
            names.push(alias);
        }
    }
    for b in IO_BUILTINS {
        names.push(b.name);
        for alias in b.aliases {
            names.push(alias);
        }
    }
    names
}

/// Format builtins for the system prompt.
pub fn format_for_prompt() -> String {
    let mut out = String::new();
    out.push_str("### Built-in Functions\n");
    for b in BUILTINS {
        let aliases = if b.aliases.is_empty() {
            String::new()
        } else {
            format!(" (aliases: {})", b.aliases.join(", "))
        };
        out.push_str(&format!("  {}{} — {}\n", b.name, aliases, b.description));
    }
    out.push_str("\n### IO Functions (require [io,async] effect, use +await)\n");
    for b in IO_BUILTINS {
        out.push_str(&format!("  {} — {}\n", b.name, b.description));
    }
    out.push_str("\n### Queries (inspect program state)\n");
    for q in QUERIES {
        out.push_str(&format!("  {} {} — {}\n", q.name, q.args, q.description));
    }
    out.push_str("\n### Commands\n");
    for a in ACTIONS {
        out.push_str(&format!("  {} {} — {}\n", a.name, a.args, a.description));
    }
    out
}

/// Check if a name is a builtin.
pub fn is_builtin(name: &str) -> bool {
    BUILTINS
        .iter()
        .any(|b| b.name == name || b.aliases.contains(&name))
        || IO_BUILTINS
            .iter()
            .any(|b| b.name == name || b.aliases.contains(&name))
}
