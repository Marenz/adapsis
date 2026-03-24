//! Builtin function registry.
//! Single source of truth for all builtin names and their descriptions.

/// A builtin function entry.
pub struct Builtin {
    pub name: &'static str,
    pub aliases: &'static [&'static str],
    pub short: &'static str,
    pub long: &'static str,
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
        short: "concatenate two or more strings",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "char_at",
        aliases: &[],
        short: "get character at index i as String",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "substring",
        aliases: &["substr"],
        short: "get substring from start to end",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "starts_with",
        aliases: &[],
        short: "check if string starts with prefix",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "ends_with",
        aliases: &[],
        short: "check if string ends with suffix",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "contains",
        aliases: &[],
        short: "check if string contains substring",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "index_of",
        aliases: &[],
        short: "find index of substring (-1 if not found)",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "split",
        aliases: &[],
        short: "split string into List<String>",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "trim",
        aliases: &[],
        short: "remove leading/trailing whitespace",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "len",
        aliases: &["length"],
        short: "length of string or list",
        long: "",
        category: BuiltinCategory::String,
    },
    // Conversion
    Builtin {
        name: "to_string",
        aliases: &["str"],
        short: "convert any value to String",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_int",
        aliases: &["parse_int", "int"],
        short: "convert String/Float/Bool to Int",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "digit_value",
        aliases: &[],
        short: "single char to digit Int ('5'->5, 'a'->-1)",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "is_digit_char",
        aliases: &[],
        short: "true if single char is 0-9",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "char_code",
        aliases: &["ord"],
        short: "character to ASCII code Int",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "from_char_code",
        aliases: &["chr"],
        short: "ASCII code to single-char String",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_hex",
        aliases: &[],
        short: "Int to 8-char hex string (32-bit)",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "u32_wrap",
        aliases: &[],
        short: "wrap Int to unsigned 32-bit range",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    // Math
    Builtin {
        name: "abs",
        aliases: &[],
        short: "absolute value",
        long: "",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "sqrt",
        aliases: &[],
        short: "square root",
        long: "",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "pow",
        aliases: &[],
        short: "power (base, exponent)",
        long: "",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "floor",
        aliases: &[],
        short: "floor to integer",
        long: "",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "min",
        aliases: &[],
        short: "minimum of two values",
        long: "",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "max",
        aliases: &[],
        short: "maximum of two values",
        long: "",
        category: BuiltinCategory::Math,
    },
    // Bitwise
    Builtin {
        name: "bit_and",
        aliases: &[],
        short: "bitwise AND",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_or",
        aliases: &[],
        short: "bitwise OR",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_xor",
        aliases: &[],
        short: "bitwise XOR",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_not",
        aliases: &[],
        short: "bitwise NOT",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shl",
        aliases: &["bit_shl"],
        short: "shift left",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shr",
        aliases: &["bit_shr"],
        short: "shift right",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "left_rotate",
        aliases: &["rotl"],
        short: "32-bit left rotation",
        long: "",
        category: BuiltinCategory::Bitwise,
    },
    // List
    Builtin {
        name: "list",
        aliases: &[],
        short: "create empty list or list with items",
        long: "",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "push",
        aliases: &[],
        short: "returns NEW list with item appended",
        long: "",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "get",
        aliases: &[],
        short: "get item at index",
        long: "",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "join",
        aliases: &[],
        short: "join list items into string with delimiter",
        long: "",
        category: BuiltinCategory::List,
    },
    // State
    Builtin {
        name: "state",
        aliases: &[],
        short: "create shared state handle",
        long: "",
        category: BuiltinCategory::State,
    },
    Builtin {
        name: "get_state",
        aliases: &[],
        short: "read shared state",
        long: "",
        category: BuiltinCategory::State,
    },
    Builtin {
        name: "set_state",
        aliases: &[],
        short: "write shared state",
        long: "",
        category: BuiltinCategory::State,
    },
    // Result
    Builtin {
        name: "Ok",
        aliases: &[],
        short: "Result success constructor",
        long: "",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Err",
        aliases: &[],
        short: "Result error constructor",
        long: "",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Some",
        aliases: &[],
        short: "Option value constructor",
        long: "",
        category: BuiltinCategory::Result,
    },
    // Regex
    Builtin {
        name: "regex_match",
        aliases: &[],
        short: "test if text matches regex pattern (Bool)",
        long: "",
        category: BuiltinCategory::Regex,
    },
    Builtin {
        name: "regex_replace",
        aliases: &[],
        short: "regex replace all matches (String)",
        long: "",
        category: BuiltinCategory::Regex,
    },
    // Base64
    Builtin {
        name: "base64_encode",
        aliases: &[],
        short: "encode string to base64",
        long: "",
        category: BuiltinCategory::Conversion,
    },
    // JSON
    Builtin {
        name: "json_get",
        aliases: &[],
        short: "extract value from JSON string by key path",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_array_len",
        aliases: &[],
        short: "get length of JSON array",
        long: "",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_escape",
        aliases: &[],
        short: "escape string for use in JSON value",
        long: "",
        category: BuiltinCategory::String,
    },
];

/// IO builtins (used with +await).
pub static IO_BUILTINS: &[Builtin] = &[
    Builtin {
        name: "tcp_listen",
        aliases: &[],
        short: "listen on TCP port, returns handle",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_accept",
        aliases: &[],
        short: "accept connection, returns handle",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_connect",
        aliases: &[],
        short: "connect to TCP server, returns handle",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_read",
        aliases: &[],
        short: "read from connection, returns String",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_write",
        aliases: &[],
        short: "write String to connection",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_close",
        aliases: &[],
        short: "close connection",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_read",
        aliases: &["read_file"],
        short: "read entire file as String",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_write",
        aliases: &["write_file"],
        short: "write String to file",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_exists",
        aliases: &[],
        short: "check if file exists (Bool)",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "list_dir",
        aliases: &[],
        short: "list directory entries as List<String>",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "read_line",
        aliases: &["stdin_read_line"],
        short: "read line from stdin with prompt",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "print",
        aliases: &[],
        short: "print text (no newline)",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "println",
        aliases: &[],
        short: "print text with newline",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "shell_exec",
        aliases: &["exec"],
        short: "run shell command, returns stdout",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "self_restart",
        aliases: &["restart"],
        short: "restart AdapsisOS process",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sleep",
        aliases: &[],
        short: "sleep for milliseconds",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_call",
        aliases: &[],
        short: "single LLM text generation: llm_call(system, prompt[, model])",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_agent",
        aliases: &[],
        short: "full agentic LLM loop until DONE: llm_agent(system, task[, model])",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_get",
        aliases: &[],
        short: "HTTP GET request, returns response body as String",
        long: "",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_post",
        aliases: &[],
        short: "HTTP POST request: http_post(url, body, content_type) -> String",
        long: "",
        category: BuiltinCategory::Io,
    },
];

/// Registered query commands (?-prefixed).
pub struct QueryCommand {
    pub name: &'static str,
    pub args: &'static str,
    pub short: &'static str,
    pub long: &'static str,
}

pub static QUERIES: &[QueryCommand] = &[
    QueryCommand {
        name: "?symbols",
        args: "[name]",
        short: "list all types and functions, or details of one",
        long: "",
    },
    QueryCommand {
        name: "?source",
        args: "<fn>",
        short: "show reconstructed source code of a function or type",
        long: "",
    },
    QueryCommand {
        name: "?callers",
        args: "<fn>",
        short: "who calls this function",
        long: "",
    },
    QueryCommand {
        name: "?callees",
        args: "<fn>",
        short: "what does this function call (1 level)",
        long: "",
    },
    QueryCommand {
        name: "?deps",
        args: "<fn>",
        short: "direct dependencies (1 level)",
        long: "",
    },
    QueryCommand {
        name: "?deps-all",
        args: "<fn>",
        short: "full transitive dependency tree",
        long: "",
    },
    QueryCommand {
        name: "?deps-modules",
        args: "<fn>",
        short: "dependencies grouped by module",
        long: "",
    },
    QueryCommand {
        name: "?effects",
        args: "<fn>",
        short: "what effects does this function have",
        long: "",
    },
    QueryCommand {
        name: "?type",
        args: "<type>",
        short: "show type definition",
        long: "",
    },
    QueryCommand {
        name: "?agents",
        args: "",
        short: "show status of all agents",
        long: "",
    },
    QueryCommand {
        name: "?plan",
        args: "",
        short: "show current plan (same as !plan)",
        long: "",
    },
    QueryCommand {
        name: "?inbox",
        args: "",
        short: "check messages from other agents or main session",
        long: "",
    },
    QueryCommand {
        name: "?tasks",
        args: "",
        short: "list all spawned async tasks and their current wait state",
        long: "",
    },
    QueryCommand {
        name: "?library",
        args: "",
        short: "show persistent module library: dir path, loaded modules, files on disk, load/save errors",
        long: "",
    },
];

/// Registered mutation/action commands (!-prefixed).
pub struct ActionCommand {
    pub name: &'static str,
    pub args: &'static str,
    pub short: &'static str,
    pub long: &'static str,
}

pub static ACTIONS: &[ActionCommand] = &[
    ActionCommand {
        name: "!test",
        args: "<fn>\\n  +with ...",
        short: "run test cases for a function",
        long: "",
    },
    ActionCommand {
        name: "!eval",
        args: "<fn> [args]",
        short: "evaluate a function or builtin",
        long: "",
    },
    ActionCommand {
        name: "!trace",
        args: "<fn> [args]",
        short: "step-by-step execution trace",
        long: "",
    },
    ActionCommand {
        name: "!replace",
        args: "<fn.sN>\\n  ...",
        short: "replace a statement in a function",
        long: "",
    },
    ActionCommand {
        name: "!remove",
        args: "<Module.function | Module | TypeName>",
        short: "remove a function, type, or entire module",
        long: "",
    },
    ActionCommand {
        name: "!move",
        args: "<symbols...> <Module>",
        short: "move functions/types into a module (auto-updates call sites)",
        long: "",
    },
    ActionCommand {
        name: "!undo",
        args: "",
        short: "revert last mutation",
        long: "",
    },
    ActionCommand {
        name: "!roadmap",
        args: "show/add <item>/done N/remove N",
        short: "manage long-term roadmap. Persists across sessions.",
        long: "",
    },
    ActionCommand {
        name: "!plan",
        args: "set/done N/fail N/show",
        short:
            "manage task plan. Steps are auto-numbered, do NOT include numbers in step text",
        long: "",
    },
    ActionCommand {
        name: "!watch",
        args: "<fn> [args] <ms>",
        short: "poll a function periodically, alert on change",
        long: "",
    },
    ActionCommand {
        name: "!agent",
        args: "<name> --scope <scope> <task>",
        short: "spawn background agent (scopes: read-only, new-only, module X, full)",
        long: "",
    },
    ActionCommand {
        name: "!mock",
        args: "<operation> \"<pattern>\" -> \"<response>\"",
        short: "register mock IO response for testing. During !test, IO calls matching the pattern return the mock instead of real IO.",
        long: "",
    },
    ActionCommand {
        name: "!unmock",
        args: "",
        short: "clear all IO mocks",
        long: "",
    },
    ActionCommand {
        name: "!msg",
        args: "<agent> <message>",
        short: "send a message to an agent (or 'main' for the main session). Agents see messages in their feedback.",
        long: "",
    },
    ActionCommand {
        name: "!opencode",
        args: "<description>",
        short: "request Rust-level change via OpenCode (rebuild + restart)",
        long: "",
    },
    ActionCommand {
        name: "!module",
        args: "<Name>",
        short: "switch module context — subsequent +fn/+type definitions go into this module",
        long: "",
    },
    ActionCommand {
        name: "!done",
        args: "",
        short: "signal that the current task is complete",
        long: "",
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
        out.push_str(&format!("  {}{} — {}\n", b.name, aliases, b.short));
        if !b.long.is_empty() {
            for line in b.long.lines() {
                out.push_str(&format!("    {}\n", line));
            }
        }
    }
    out.push_str("\n### IO Functions (require [io,async] effect, use +await)\n");
    for b in IO_BUILTINS {
        out.push_str(&format!("  {} — {}\n", b.name, b.short));
        if !b.long.is_empty() {
            for line in b.long.lines() {
                out.push_str(&format!("    {}\n", line));
            }
        }
    }
    out.push_str("\n### Queries (inspect program state)\n");
    for q in QUERIES {
        out.push_str(&format!("  {} {} — {}\n", q.name, q.args, q.short));
        if !q.long.is_empty() {
            for line in q.long.lines() {
                out.push_str(&format!("    {}\n", line));
            }
        }
    }
    out.push_str("\n### Commands\n");
    for a in ACTIONS {
        out.push_str(&format!("  {} {} — {}\n", a.name, a.args, a.short));
        if !a.long.is_empty() {
            for line in a.long.lines() {
                out.push_str(&format!("    {}\n", line));
            }
        }
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
