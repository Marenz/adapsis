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
        long: "Concatenates two or more values into one String. String arguments are appended directly; other values are converted to text first.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "char_at",
        aliases: &[],
        short: "get character at index i as String",
        long: "Returns the character at a 0-based index as a single-character String. Takes (String, Int) and errors if the index is out of bounds.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "substring",
        aliases: &["substr"],
        short: "get substring from start to end",
        long: "Returns the slice of a String from start to end. Takes (String, Int, Int); end is capped to the string length, but invalid ranges still error.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "starts_with",
        aliases: &[],
        short: "check if string starts with prefix",
        long: "Checks whether a String begins with a prefix and returns Bool. Use it for exact prefix tests without manual slicing.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "ends_with",
        aliases: &[],
        short: "check if string ends with suffix",
        long: "Checks whether a String ends with a suffix and returns Bool. Useful for file extensions, markers, and simple pattern checks.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "contains",
        aliases: &[],
        short: "check if string contains substring",
        long: "Checks whether a String contains a substring and returns Bool. This is a plain substring search, not a regex match.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "index_of",
        aliases: &[],
        short: "find index of substring (-1 if not found)",
        long: "Finds the first occurrence of a substring and returns its 0-based index as Int. Returns -1 if the substring is not found.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "split",
        aliases: &[],
        short: "split string into List<String>",
        long: "Splits a String by a delimiter and returns a List of substrings. Empty segments are included when delimiters are adjacent or at the ends.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "trim",
        aliases: &[],
        short: "remove leading/trailing whitespace",
        long: "Removes leading and trailing whitespace from a String and returns the trimmed String. Internal whitespace is left unchanged.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "len",
        aliases: &["length"],
        short: "length of string or list",
        long: "Returns the length of a String or List as an Int. Use it to count bytes in a string or items in a list.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "error_suggest",
        aliases: &["failure_suggest"],
        short: "suggest a fix for an error message",
        long: "Returns a suggestion String for a common mutation or validation error. Recognizes patterns like undefined variables, struct syntax errors, missing effects, out-of-range statement indexes, and `len()` misuse. Returns an empty String when no pattern matches. Alias: `failure_suggest`.",
        category: BuiltinCategory::String,
    },
    // Conversion
    Builtin {
        name: "to_string",
        aliases: &["str"],
        short: "convert any value to String",
        long: "Converts one value to its String representation. Takes any single value and returns a String.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_int",
        aliases: &["parse_int", "int"],
        short: "convert String/Float/Bool to Int",
        long: "Converts a String, Float, Bool, or Int to Int. String parsing errors fail, and Float values are truncated toward zero.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "digit_value",
        aliases: &[],
        short: "single char to digit Int ('5'->5, 'a'->-1)",
        long: "Converts a single-character digit String like \"5\" to its Int value. Returns -1 for non-digits or strings longer than one character.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "is_digit_char",
        aliases: &[],
        short: "true if single char is 0-9",
        long: "Returns true if the input is exactly one ASCII digit character. Non-String values return false instead of failing.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "char_code",
        aliases: &["ord"],
        short: "character to ASCII code Int",
        long: "Returns the byte value of a single-character String as an Int. This is for single-byte/ASCII-style character codes, not full Unicode code points.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "from_char_code",
        aliases: &["chr"],
        short: "ASCII code to single-char String",
        long: "Converts an Int to a single-character String using its low 8 bits. Use it for byte-to-character conversion.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "to_hex",
        aliases: &[],
        short: "Int to 8-char hex string (32-bit)",
        long: "Formats an Int as an 8-character lowercase hexadecimal String in 32-bit form. Negative values are shown using 32-bit wrapping.",
        category: BuiltinCategory::Conversion,
    },
    Builtin {
        name: "u32_wrap",
        aliases: &[],
        short: "wrap Int to unsigned 32-bit range",
        long: "Wraps an Int into the unsigned 32-bit range and returns the wrapped Int value. Useful before bitwise or hashing-style operations.",
        category: BuiltinCategory::Conversion,
    },
    // Math
    Builtin {
        name: "abs",
        aliases: &[],
        short: "absolute value",
        long: "Returns the absolute value of an Int or Float. The return type matches the numeric input type.",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "sqrt",
        aliases: &[],
        short: "square root",
        long: "Computes the square root of an Int or Float and returns a Float. Use it when you always want a floating-point result.",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "pow",
        aliases: &[],
        short: "power (base, exponent)",
        long: "Raises a base to an exponent. Supports Int and Float combinations; Int^Int returns Int, while any Float input returns Float.",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "floor",
        aliases: &[],
        short: "floor to integer",
        long: "Rounds a Float down to the nearest whole number and returns Int. If the input is already Int, it is returned unchanged.",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "min",
        aliases: &[],
        short: "minimum of two values",
        long: "Returns the smaller of two numbers. Both arguments must be the same numeric type: Int or Float.",
        category: BuiltinCategory::Math,
    },
    Builtin {
        name: "max",
        aliases: &[],
        short: "maximum of two values",
        long: "Returns the larger of two numbers. Both arguments must be the same numeric type: Int or Float.",
        category: BuiltinCategory::Math,
    },
    // Bitwise
    Builtin {
        name: "bit_and",
        aliases: &[],
        short: "bitwise AND",
        long: "Performs bitwise AND on two Int values and returns an Int. Use it to mask bits.",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_or",
        aliases: &[],
        short: "bitwise OR",
        long: "Performs bitwise OR on two Int values and returns an Int. Use it to combine bit flags.",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_xor",
        aliases: &[],
        short: "bitwise XOR",
        long: "Performs bitwise XOR on two Int values and returns an Int. Useful for toggling or comparing bit patterns.",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "bit_not",
        aliases: &[],
        short: "bitwise NOT",
        long: "Performs bitwise NOT on an Int and returns the inverted Int value. This flips all bits in the integer representation.",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shl",
        aliases: &["bit_shl"],
        short: "shift left",
        long: "Shifts an Int left by a given number of bits and returns an Int. Takes (Int, Int).",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "shr",
        aliases: &["bit_shr"],
        short: "shift right",
        long: "Shifts an Int right by a given number of bits and returns an Int. Takes (Int, Int).",
        category: BuiltinCategory::Bitwise,
    },
    Builtin {
        name: "left_rotate",
        aliases: &["rotl"],
        short: "32-bit left rotation",
        long: "Rotates an Int left within 32 bits and returns the rotated Int. The shift count is taken modulo 32.",
        category: BuiltinCategory::Bitwise,
    },
    // List
    Builtin {
        name: "list",
        aliases: &[],
        short: "create empty list or list with items",
        long: "Creates a List from zero or more values. `list()` returns an empty list, and `list(a, b, c)` returns those items in order.",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "push",
        aliases: &[],
        short: "returns NEW list with item appended",
        long: "Returns a new List with one item appended to the end. It does not mutate the original list value.",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "get",
        aliases: &[],
        short: "get item at index",
        long: "Returns the item at a 0-based index from a List. Takes (List, Int) and errors if the index is out of bounds.",
        category: BuiltinCategory::List,
    },
    Builtin {
        name: "join",
        aliases: &[],
        short: "join list items into string with delimiter",
        long: "Joins List items into a String using a delimiter. Each item is converted to text before joining.",
        category: BuiltinCategory::List,
    },
    // State
    // Result
    Builtin {
        name: "Ok",
        aliases: &[],
        short: "Result success constructor",
        long: "Constructs a successful Result value. Usually used when a function returns `Result<T, String>` and you want the success branch.",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Err",
        aliases: &[],
        short: "Result error constructor",
        long: "Constructs an error Result value. If the payload is not already a String, it is converted to text.",
        category: BuiltinCategory::Result,
    },
    Builtin {
        name: "Some",
        aliases: &[],
        short: "Option value constructor",
        long: "Constructs a `Some(value)` variant for optional data. Use it when returning or matching Option-like values.",
        category: BuiltinCategory::Result,
    },
    // Regex
    Builtin {
        name: "regex_match",
        aliases: &[],
        short: "test if text matches regex pattern (Bool)",
        long: "Tests whether text matches a regex pattern and returns Bool. Invalid regex patterns return false instead of failing.",
        category: BuiltinCategory::Regex,
    },
    Builtin {
        name: "regex_replace",
        aliases: &[],
        short: "regex replace all matches (String)",
        long: "Replaces all regex matches in a String and returns the updated String. Invalid patterns fail with an error.",
        category: BuiltinCategory::Regex,
    },
    // Base64
    Builtin {
        name: "base64_encode",
        aliases: &[],
        short: "encode string to base64",
        long: "Encodes a String as standard Base64 and returns the encoded String. Use it for text or byte-like data already represented as a String.",
        category: BuiltinCategory::Conversion,
    },
    // JSON
    Builtin {
        name: "json_get",
        aliases: &[],
        short: "extract value from JSON string by key path",
        long: "Extracts a value from a JSON String using a dot-separated key path such as `user.name` or `items.0`. Missing keys or invalid JSON fail; objects and arrays are returned as JSON Strings.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_array_len",
        aliases: &[],
        short: "get length of JSON array",
        long: "Parses a JSON String that must be a top-level array and returns its length as Int. Invalid JSON or non-array input fails.",
        category: BuiltinCategory::String,
    },
    Builtin {
        name: "json_escape",
        aliases: &[],
        short: "escape string for use in JSON value",
        long: "Escapes backslashes, quotes, and common control characters for safe insertion into JSON string values. It returns only the escaped contents, not surrounding quotes.",
        category: BuiltinCategory::String,
    },
];

/// IO builtins (used with +await).
pub static IO_BUILTINS: &[Builtin] = &[
    Builtin {
        name: "tcp_listen",
        aliases: &[],
        short: "listen on TCP port, returns handle",
        long: "Starts listening on a TCP port and returns an Int handle for the listener. Requires `+await` and blocks until the listener is created.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_accept",
        aliases: &[],
        short: "accept connection, returns handle",
        long: "Accepts the next incoming connection from a TCP listener handle and returns a new connection handle as Int. Requires `+await` and blocks until a client connects.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_connect",
        aliases: &[],
        short: "connect to TCP server, returns handle",
        long: "Connects to a TCP server given a host String and port Int, returning a connection handle as Int. Requires `+await` and blocks until the connection attempt completes.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_read",
        aliases: &[],
        short: "read from connection, returns String",
        long: "Reads data from a TCP connection handle and returns it as a String. Requires `+await` and blocks until data is available or the read completes.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_write",
        aliases: &[],
        short: "write String to connection",
        long: "Writes data to a TCP connection handle. Non-String values are converted to text, and the call returns Int 0 on success.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "tcp_close",
        aliases: &[],
        short: "close connection",
        long: "Closes a TCP connection handle and returns Int 0. Use it to release sockets when you are done with them.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_read",
        aliases: &["read_file"],
        short: "read entire file as String",
        long: "Reads an entire file from a path String and returns its contents as a String. Requires `+await` and fails if the file cannot be read.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_write",
        aliases: &["write_file"],
        short: "write String to file",
        long: "Writes text to a file path and returns the String `OK` on success. Non-String data is converted to text before writing.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "file_exists",
        aliases: &[],
        short: "check if file exists (Bool)",
        long: "Checks whether a file or path exists and returns Bool. Use it before reading or writing when existence matters.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "list_dir",
        aliases: &[],
        short: "list directory entries as List<String>",
        long: "Lists directory entry names for a path and returns `List<String>`. It returns names only, not full metadata.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "read_line",
        aliases: &["stdin_read_line"],
        short: "read line from stdin with prompt",
        long: "Prompts on stdin and reads one line of user input, returning it as a String. Requires `+await` and blocks until the user responds.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "print",
        aliases: &[],
        short: "print text (no newline)",
        long: "Prints text immediately without a trailing newline. Non-String values are converted to text, and the call returns Int 0.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "println",
        aliases: &[],
        short: "print text with newline",
        long: "Prints text followed by a newline. Non-String values are converted to text, and the call returns Int 0.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "shell_exec",
        aliases: &["exec"],
        short: "run shell command, returns stdout",
        long: "Runs a shell command and returns stdout as a String on success. On nonzero exit it still returns a String, formatted as `EXIT <code>: <stderr>`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "self_restart",
        aliases: &["restart"],
        short: "restart AdapsisOS process",
        long: "Requests that the AdapsisOS process restart and returns the String `restarting...`. Use it for controlled self-restarts from within the runtime.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sleep",
        aliases: &[],
        short: "sleep for milliseconds",
        long: "Sleeps for the given number of milliseconds and returns Int 0. Requires `+await` and blocks until the delay finishes.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_call",
        aliases: &[],
        short: "single LLM text generation: llm_call(system, prompt[, model])",
        long: "Makes a single LLM text-generation request and returns the model response as a String. Takes `(system:String, prompt:String[, model:String])`, requires `+await`, and blocks until the response arrives.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_agent",
        aliases: &[],
        short: "full agentic LLM loop until DONE: llm_agent(system, task[, model])",
        long: "Runs the full agentic LLM loop for a task and returns the final output as a String. Takes `(system:String, task:String[, model:String])`, requires `+await`, and blocks until the agent finishes.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "llm_takeover",
        aliases: &[],
        short: "conversational LLM with per-context history: llm_takeover(context, message[, reply_fn, reply_arg])",
        long: "Calls the LLM with per-context conversation history and returns the text reply. \
               The context name (e.g. \"telegram:123\", \"agent:builder\") identifies the conversation — \
               each context has independent history. If the LLM response contains code, it is executed \
               in the background. If the code spawns agents (!agent), the agent runs asynchronously and \
               the reply callback is invoked when it completes. \
               Optional reply_fn/reply_arg configure a callback function for async notifications: \
               reply_fn(reply_arg, text) is called to deliver agent completion summaries.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_get",
        aliases: &[],
        short: "HTTP GET request, returns response body as String",
        long: "Makes an HTTP GET request and returns the response body as a String. Requires `+await` and blocks until the response arrives.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_post",
        aliases: &[],
        short: "HTTP POST request: http_post(url, body, content_type) -> String",
        long: "Makes an HTTP POST request and returns the response body as a String. Takes `(url:String, body:String, content_type:String)`; if content type is omitted, it defaults to `application/json`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "http_upload",
        aliases: &[],
        short: "multipart file upload: http_upload(url, file_path, file_field, extra_fields) -> String",
        long: "Uploads a local file via HTTP multipart/form-data POST. Returns the response body as a String. \
               Takes `(url:String, file_path:String, file_field:String, extra_fields:String)`. \
               file_field is the form field name for the file (e.g. \"audio\", \"document\", \"photo\"). \
               extra_fields is optional, format: \"key1=val1&key2=val2\" for additional form fields. \
               Example: http_upload(concat(\"https://api.telegram.org/bot\", token, \"/sendAudio\"), \"/tmp/song.wav\", \"audio\", concat(\"chat_id=\", to_string(chat_id)))",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "roadmap_list",
        aliases: &[],
        short: "get current roadmap as String",
        long: "Returns the current roadmap as a formatted String. Each line is `[ ] N: description` or `[x] N: description` for done items. Returns `Roadmap is empty.` when empty. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "roadmap_add",
        aliases: &[],
        short: "add item to roadmap, returns the item text",
        long: "Adds a new item to the persistent roadmap and returns the item description String. Takes `(item:String)`. Fails if the description is empty. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "roadmap_done",
        aliases: &[],
        short: "mark roadmap item N as done",
        long: "Marks a roadmap item as done by its 1-based index and returns a confirmation String like `Roadmap: #N done.`. Takes `(n:Int)`. Fails if the index is out of bounds. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "plan_show",
        aliases: &[],
        short: "get current plan as String",
        long: "Returns the current plan as a formatted String. Each line is `[ ] N: description`, `[x] N: description` (done), or `[!] N: description` (failed). Returns `No plan set.` when empty. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "plan_set",
        aliases: &[],
        short: "replace current plan from newline-separated steps",
        long: "Replaces the current plan. Takes `(steps:String)` where each line is a step description. Steps are auto-numbered starting from 1. Returns `Plan set with N steps.`. Fails if steps string is empty. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "plan_done",
        aliases: &[],
        short: "mark plan step N as done",
        long: "Marks plan step N (1-based) as done and returns `Plan: step N done.`. Takes `(n:Int)`. Fails if the index is out of bounds. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "plan_fail",
        aliases: &[],
        short: "mark plan step N as failed",
        long: "Marks plan step N (1-based) as failed and returns `Plan: step N failed.`. Takes `(n:Int)`. Fails if the index is out of bounds. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    // Program introspection queries — callable versions of ? query commands
    Builtin {
        name: "query_symbols",
        aliases: &["symbols_list"],
        short: "list all types and functions (same as ?symbols)",
        long: "Returns a formatted String listing all types and functions in the current program. Same output as `?symbols`. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_symbols_detail",
        aliases: &[],
        short: "get details for one symbol (same as ?symbols <name>)",
        long: "Returns detailed information about a specific symbol (function, type, or module) as a String. Same output as `?symbols <name>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_source",
        aliases: &["source_get"],
        short: "get reconstructed source code (same as ?source <fn>)",
        long: "Returns the reconstructed Adapsis source code for a function or type as a String. Same output as `?source <fn>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_callers",
        aliases: &["callers_get"],
        short: "who calls this function (same as ?callers <fn>)",
        long: "Returns which functions directly call the given function as a String. Same output as `?callers <fn>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_callees",
        aliases: &["callees_get"],
        short: "what this function calls (same as ?callees <fn>)",
        long: "Returns the direct functions called by the given function as a String. Same output as `?callees <fn>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_deps",
        aliases: &[],
        short: "direct dependencies (same as ?deps <fn>)",
        long: "Returns the direct dependencies of a function as a String. Same output as `?deps <fn>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_deps_all",
        aliases: &["deps_get"],
        short: "full transitive dependency tree (same as ?deps-all <fn>)",
        long: "Returns the full transitive dependency tree for a function as a String. Same output as `?deps-all <fn>`. Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_routes",
        aliases: &["routes_list"],
        short: "list registered HTTP routes (same as ?routes)",
        long: "Returns registered HTTP routes as a formatted String. Same output as `?routes`. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_tasks",
        aliases: &[],
        short: "list spawned async tasks (same as ?tasks)",
        long: "Returns spawned async tasks and their current wait state as a String. Same output as `?tasks`. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_inbox",
        aliases: &[],
        short: "show inbox contents (same as ?inbox)",
        long: "Returns the current session inbox as a formatted String. Same output as `?inbox`: each line is `[timestamp] from sender: message`, or `No messages.` when empty. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_library",
        aliases: &[],
        short: "show library status (same as ?library)",
        long: "Returns persistent module library status as a String, including directory path, loaded modules, and files on disk. Same output as `?library`. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    // Program mutation builtins — modify the live program AST
    Builtin {
        name: "mutate",
        aliases: &[],
        short: "apply Adapsis code mutations to the program: mutate(code) -> String",
        long: "Parses a String of Adapsis code (mutations like +fn, +type, +module, etc.) and applies them to the live program. \
               Returns a summary like \"Applied 3 mutations\" on success, or fails with the parse/validation error. \
               This is the general-purpose mutation builtin — it does the same thing as sending code to the main loop. \
               Takes `(code:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "fn_remove",
        aliases: &[],
        short: "remove a function by name: fn_remove(name) -> String",
        long: "Removes a function by fully-qualified name (e.g. \"MyModule.my_func\" or bare \"my_func\" for top-level). \
               Returns \"Removed <name>\" on success, or fails if the function is not found. \
               Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "type_remove",
        aliases: &[],
        short: "remove a type by name: type_remove(name) -> String",
        long: "Removes a type by name (e.g. \"MyModule.MyType\" or bare \"MyType\" for top-level). \
               Returns \"Removed <name>\" on success, or fails if the type is not found. \
               Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "module_remove",
        aliases: &[],
        short: "remove an entire module: module_remove(name) -> String",
        long: "Removes an entire module and all its contents (functions, types, shared vars). \
               Returns \"Removed module <name>\" on success, or fails if the module is not found. \
               Takes `(name:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    // Programmatic equivalents of !move, !watch, !agent, !msg, !trace commands
    Builtin {
        name: "move_symbols",
        aliases: &[],
        short: "move symbols into a module: move_symbols(symbols, target_module) -> String",
        long: "Moves comma-separated symbol names (functions, types, or modules) into the target module \
               and updates all call sites automatically. Same as `!move`. \
               Takes `(symbols:String, target_module:String)`. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "watch_start",
        aliases: &[],
        short: "start watching a function periodically: watch_start(fn_name, interval_ms) -> String",
        long: "Starts polling a function periodically and alerts when its result changes. Same as `!watch`. \
               Takes `(fn_name:String, interval_ms:Int)`. Returns confirmation. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "agent_spawn",
        aliases: &[],
        short: "spawn a background agent: agent_spawn(name, scope, task) -> String",
        long: "Spawns a background agent with the given name, scope, and task. Same as `!agent`. \
               Scope can be \"read-only\", \"new-only\", \"module X\", or \"full\". \
               Takes `(name:String, scope:String, task:String)`. Returns confirmation. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "msg_send",
        aliases: &[],
        short: "send a message to an agent or main: msg_send(target, message) -> String",
        long: "Sends a message to a named agent or to \"main\". Same as `!msg`. \
               The recipient sees it in `?inbox` or agent feedback. \
               Takes `(target:String, message:String)`. Returns confirmation. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "inbox_read",
        aliases: &[],
        short: "drain pending inbox messages as JSON array string",
        long: "Reads all pending inbox messages for the current main session and returns a JSON array String of message contents such as `[\"msg1\",\"msg2\"]`. Clears the inbox after reading and returns `[]` when no messages are pending. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "inbox_clear",
        aliases: &[],
        short: "clear all pending inbox messages",
        long: "Clears all pending inbox messages for the current main session and returns a confirmation String like `cleared 2 messages`. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "trace_run",
        aliases: &[],
        short: "run a function with tracing: trace_run(fn_name, args) -> String",
        long: "Runs a function with step-by-step tracing enabled. Same as `!trace`. \
               Returns the trace output as a formatted String. \
               Takes `(fn_name:String, args:String)` where args is the input expression text (or empty). \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "route_list",
        aliases: &[],
        short: "list all registered HTTP routes: route_list() -> String",
        long: "Returns all registered HTTP routes as a formatted string, one per line. \
               Each line shows: METHOD /path -> `Handler.func`. \
               Returns 'No routes registered.' if none exist. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "route_add",
        aliases: &[],
        short: "register an HTTP route: route_add(method, path, handler) -> String",
        long: "Registers an HTTP route mapping a method+path to an Adapsis function handler. \
               Takes `(method:String, path:String, handler:String)`. \
               Method is GET, POST, PUT, DELETE, or PATCH. Path must start with '/'. \
               Handler is a qualified function name like 'Module.func'. \
               Upserts: updates existing route if method+path already registered. \
               Returns confirmation like 'added route GET /api/foo -> `Module.func`'. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "route_remove",
        aliases: &[],
        short: "remove an HTTP route: route_remove(method, path) -> String",
        long: "Removes an HTTP route by method and path. \
               Takes `(method:String, path:String)`. \
               Returns confirmation like 'removed route GET /api/foo (was -> `Module.func`)'. \
               Fails if no route found for the given method+path. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "undo",
        aliases: &[],
        short: "revert the last mutation: undo() -> String",
        long: "Queues an undo operation that reverts the last mutation. \
               Same as `!undo` command. The undo is processed by the API layer \
               after the current eval completes. \
               Returns confirmation that the undo has been queued. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sandbox_enter",
        aliases: &[],
        short: "enter sandbox mode: sandbox_enter() -> String",
        long: "Queues entry into sandbox mode where mutations are isolated. \
               Same as `!sandbox enter`. The sandbox is activated by the API layer \
               after the current eval completes. Use sandbox_merge() to keep changes \
               or sandbox_discard() to revert. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sandbox_merge",
        aliases: &[],
        short: "merge sandbox changes: sandbox_merge() -> String",
        long: "Queues merging of sandbox changes into the main program state. \
               Same as `!sandbox merge`. Processed by the API layer after eval completes. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sandbox_discard",
        aliases: &[],
        short: "discard sandbox changes: sandbox_discard() -> String",
        long: "Queues discarding of sandbox changes, reverting to the state before sandbox was entered. \
               Same as `!sandbox discard`. Processed by the API layer after eval completes. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "mock_set",
        aliases: &[],
        short: "register an IO mock: mock_set(operation, pattern, response) -> String",
        long: "Registers a mock IO response for testing. Same as `!mock`. \
               Takes `(operation:String, pattern:String, response:String)`. \
               Operation is the IO builtin name (e.g. 'http_get', 'llm_call'). \
               Pattern is matched against arguments (space-separated for multiple arg positions). \
               Response is the value returned when the mock matches. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "mock_clear",
        aliases: &[],
        short: "clear all IO mocks: mock_clear() -> String",
        long: "Clears all registered IO mocks. Same as `!unmock`. \
               Returns 'cleared N mocks'. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "sse_send",
        aliases: &["sse_broadcast"],
        short: "broadcast an SSE event: sse_send(event, data) -> String",
        long: "Broadcasts a Server-Sent Event payload to `/api/events` listeners as a JSON string like `{\"type\":\"name\",\"data\":\"value\"}`. Takes `(event_type:String, data:String)`. Returns `sent`. Requires `+await`. Alias: `sse_broadcast`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "module_create",
        aliases: &[],
        short: "create a module: module_create(name) -> String",
        long: "Creates a new module or confirms it already exists. Same as `+module Name`. \
               Takes `(name:String)`. Name must start with an uppercase letter. \
               Returns 'created module Name' on success, or 'module Name already exists'. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "test_run",
        aliases: &[],
        short: "run stored tests: test_run(fn_name) -> String",
        long: "Runs all stored tests for a function. Same as `+test`. \
               Takes `(fn_name:String)` — fully-qualified name like 'Module.func'. \
               Returns test results as a string with PASS/FAIL for each test case. \
               Returns 'no stored tests for fn_name' if none exist. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "fn_replace",
        aliases: &[],
        short: "replace a statement: fn_replace(target, new_code) -> String",
        long: "Replaces a statement in a function. Same as `!replace`. \
               Takes `(target:String, new_code:String)`. \
               Target is like 'Module.func.s1' (statement 1 of Module.func). \
               new_code is valid Adapsis code for the replacement statement(s). \
                Returns confirmation or fails with validation error. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "library_reload",
        aliases: &[],
        short: "reload library module(s) from disk: library_reload(name) -> String",
        long: "Reloads a module from the persistent library directory (~/.config/adapsis/modules/). \
               Takes `(name:String)` — the module name (e.g. \"MyModule\"). \
               If name is empty string \"\", reloads ALL .ax files from the library directory. \
               The existing module is removed from the program and re-parsed from disk. \
               Returns \"Reloaded ModuleName successfully\" on success, or fails with the error. \
               Useful for recovering from load errors at startup without restarting. \
               Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "run_module_startups",
        aliases: &[],
        short: "execute all module startup blocks: run_module_startups() -> String",
        long: "Executes the +startup block for every module that has one, in alphabetical order. \
               Also auto-registers module-level +source declarations (timer sources begin ticking). \
               Returns a summary of results. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "query_startups",
        aliases: &[],
        short: "list modules with startup/shutdown blocks: query_startups() -> String",
        long: "Returns a list of all modules that have +startup or +shutdown blocks, \
               with their effects and statement counts. \
               Returns \"No modules have startup or shutdown blocks.\" if none exist. \
               Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "library_errors",
        aliases: &[],
        short: "get library load/save errors: library_errors() -> String",
        long: "Returns a formatted string of all library module load/save errors from the current session. \
               Includes structured load errors (module name + error message) and general session errors. \
               Returns \"No library errors.\" if no errors have occurred. \
               Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "failure_history",
        aliases: &[],
        short: "show recent mutation failures: failure_history() -> String",
        long: "Returns the last 20 recorded mutation or validation failures from the current session as a newline-separated String. Returns \"No recent mutation failures.\" if empty. Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "clear_failure_history",
        aliases: &[],
        short: "clear recent mutation failures: clear_failure_history() -> String",
        long: "Clears the recorded failure history for the current session and returns \"cleared\". Takes no arguments. Requires `+await`.",
        category: BuiltinCategory::Io,
    },
    Builtin {
        name: "failure_patterns",
        aliases: &[],
        short: "summarize repeated failures: failure_patterns() -> String",
        long: "Analyzes recent mutation failures and groups repeated mistakes such as undefined variable errors, type mismatch errors, parse errors, and validation errors. Returns a compact summary string to help the AI avoid repeating the same mistake. Takes no arguments. Requires `+await`.",
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
        long: "Lists all known types and functions, or shows details for one symbol if you pass a name. Use it first when you need to discover what exists in the current program.",
    },
    QueryCommand {
        name: "?source",
        args: "<fn>",
        short: "show reconstructed source code of a function or type",
        long: "Shows reconstructed Adapsis source for a function or type. Use it to inspect the current definition after mutations or loading from session state.",
    },
    QueryCommand {
        name: "?callers",
        args: "<fn>",
        short: "who calls this function",
        long: "Shows which functions directly call the given function. This is useful for impact analysis before changing or removing code.",
    },
    QueryCommand {
        name: "?callees",
        args: "<fn>",
        short: "what does this function call (1 level)",
        long: "Shows the direct functions called by the given function. It is a one-level view, not a full transitive dependency tree.",
    },
    QueryCommand {
        name: "?deps",
        args: "<fn>",
        short: "direct dependencies (1 level)",
        long: "Shows the direct dependencies of a function. In practice this is the same one-level call graph view as `?callees`.",
    },
    QueryCommand {
        name: "?deps-all",
        args: "<fn>",
        short: "full transitive dependency tree",
        long: "Shows the full transitive dependency tree for a function. Use it when you need to understand everything a function ultimately depends on.",
    },
    QueryCommand {
        name: "?deps-modules",
        args: "<fn>",
        short: "dependencies grouped by module",
        long: "Shows a function's dependencies grouped by module. Use it to see cross-module coupling at a glance.",
    },
    QueryCommand {
        name: "?effects",
        args: "<fn>",
        short: "what effects does this function have",
        long: "Shows the declared effects for a function, such as `io`, `async`, or `fail`. Useful when debugging why a call requires certain effect annotations.",
    },
    QueryCommand {
        name: "?type",
        args: "<type>",
        short: "show type definition",
        long: "Shows the definition of a type by name. Use it for structs, unions, and other named type information.",
    },
    QueryCommand {
        name: "?agents",
        args: "",
        short: "show status of all agents",
        long: "Shows the status of background agents. Use it to see which agents are running, finished, or waiting to merge.",
    },
    QueryCommand {
        name: "?plan",
        args: "",
        short: "show current plan (same as !plan)",
        long: "Shows the current task plan, including step numbers and status. This is a read-only view of the same plan managed by `!plan`.",
    },
    QueryCommand {
        name: "?inbox",
        args: "",
        short: "check messages from other agents or main session",
        long: "Shows messages sent from other agents or the main session. Use it to check coordination without consuming or mutating program state.",
    },
    QueryCommand {
        name: "?tasks",
        args: "",
        short: "list all spawned async tasks and their current wait state",
        long: "Lists spawned async tasks and what each one is currently waiting on. This query needs runtime context, so it reflects live execution state.",
    },
    QueryCommand {
        name: "?library",
        args: "",
        short: "show persistent module library: dir path, loaded modules, files on disk, load/save errors",
        long: "Shows persistent module library state, including the library directory, loaded modules, files on disk, and any load or save errors. Use it when debugging library persistence.",
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
        name: "+test",
        args: "<fn>\\n  +with ...",
        short: "run test cases for a function",
        long: "Runs stored test cases for a function and records the results. Use `+with` lines to define inputs and expected outputs; for async IO code, pair it with `!mock`.\n\
Test matchers for flexible assertions:\n  +with -> expect contains(\"substring\")   — result contains substring\n  +with -> expect starts_with(\"prefix\")   — result starts with prefix\n  +with -> expect Ok                       — any Ok value\n  +with -> expect Err                      — any Err value\n  +with -> expect Err(\"msg\")              — Err containing specific message\n\
Side-effect assertions with +after (checked after the function runs, state restored):\n  +after routes contains \"/chat\"          — HTTP route was registered\n  +after modules contains \"WebChat\"       — module exists\n  +after mocks contains \"http_get\"        — mock was registered",
    },
    ActionCommand {
        name: "!eval",
        args: "<fn> [args] | <expression>",
        short: "evaluate a function/builtin or inline expression",
        long: "Evaluates a function, builtin, or inline expression and shows the result. \
               Two modes: (1) `!eval func_name arg1 arg2` calls a function with arguments. \
               (2) `!eval <expr>` evaluates an expression directly — supports arithmetic (`1 + 2`), \
               function calls (`concat(\"a\", \"b\")`), builtins (`len(\"hello\")`), \
               comparisons (`3 > 2`), struct literals (`{name: \"alice\"}`), \
               list creation (`list(1, 2, 3)`), and nested calls (`len(concat(\"a\", \"b\"))`). \
               In AdapsisOS mode, functions with more than two statements must pass `+test` before `!eval` is allowed.",
    },
    ActionCommand {
        name: "!trace",
        args: "<fn> [args]",
        short: "step-by-step execution trace",
        long: "Runs a function with tracing enabled so you can inspect execution step by step. Use it when you need to debug control flow or intermediate values.",
    },
    ActionCommand {
        name: "!replace",
        args: "<fn.sN>\\n  ...",
        short: "replace a statement in a function",
        long: "Replaces one statement inside a function with new Adapsis code. Target a specific statement like `Module.fn.sN`, then provide the replacement block below the command.",
    },
    ActionCommand {
        name: "!remove",
        args: "<Module.function | Module | TypeName>",
        short: "remove a function, type, or entire module (also removes associated routes)",
        long: "Removes a function, type, or entire module from the program. Use fully qualified names for functions, and be careful because this mutates stored program state. When removing a function or module, any HTTP routes pointing to the removed handler are also automatically removed.",
    },
    ActionCommand {
        name: "!remove route",
        args: "<METHOD> <path>",
        short: "remove an HTTP route by method and path",
        long: "Removes a registered HTTP route by its method and path. Example: !remove route POST /api/ask. The handler function is NOT removed, only the route entry.",
    },
    ActionCommand {
        name: "!unroute",
        args: "<METHOD> <path>",
        short: "remove an HTTP route by method and path (alias for !remove route)",
        long: "Shorthand for `!remove route METHOD /path`. Removes a registered HTTP route by its method and path. Example: !unroute GET /health. The handler function is NOT removed, only the route entry.",
    },
    ActionCommand {
        name: "!move",
        args: "<symbols...> <Module>",
        short: "move functions/types into a module (auto-updates call sites)",
        long: "Moves one or more functions or types into another module and updates call sites automatically. Use it for reorganization without rewriting references by hand.",
    },
    ActionCommand {
        name: "!sandbox",
        args: "[enter|merge|discard|status]",
        short: "enter/exit isolated sandbox mode for safe experimentation",
        long: "Enters a sandbox where all mutations are isolated. \
               `!sandbox` or `!sandbox enter` — snapshot current program+runtime, begin sandbox. \
               `!sandbox merge` — keep all sandbox changes (commit to session). \
               `!sandbox discard` — revert all sandbox changes (restore snapshot). \
               `!sandbox status` — report whether sandbox is active and how many mutations since entry.",
    },
    ActionCommand {
        name: "!undo",
        args: "",
        short: "revert last mutation",
        long: "Reverts the most recent mutation by restoring the previous revision. Use it when a recent change produced the wrong result.",
    },
    ActionCommand {
        name: "!roadmap",
        args: "show/add <item>/done N/remove N",
        short: "manage long-term roadmap. Persists across sessions.",
        long: "Shows or updates the long-term roadmap that persists across sessions. Use `add`, `done N`, and `remove N` to manage backlog items.",
    },
    ActionCommand {
        name: "!plan",
        args: "set/done N/fail N/show",
        short:
            "manage task plan. Steps are auto-numbered, do NOT include numbers in step text",
        long: "Shows or updates the current task plan. `set` replaces the plan, `done N` marks steps complete, and `fail N` marks them failed; step numbers are assigned automatically.",
    },
    ActionCommand {
        name: "!watch",
        args: "<fn> [args] <ms>",
        short: "poll a function periodically, alert on change",
        long: "Starts polling a function periodically and alerts when its result changes. The last argument is the interval in milliseconds.",
    },
    ActionCommand {
        name: "!agent",
        args: "<name> --scope <scope> <task>",
        short: "spawn background agent (scopes: read-only, new-only, module X, full)",
        long: "Spawns a background agent to work on a task with a chosen scope such as `read-only`, `new-only`, `module X`, or `full`. Use it for parallel work without blocking the main session.",
    },
    ActionCommand {
        name: "!mock",
        args: "<operation> \"<pattern>\" -> \"<response>\"",
        short: "register mock IO response for testing. During +test, IO calls matching the pattern return the mock instead of real IO.",
        long: "Registers a fake IO response for testing. During `+test`, if a `+await` call matches the operation and pattern strings, the mock response is returned instead of real IO.",
    },
    ActionCommand {
        name: "!unmock",
        args: "",
        short: "clear all IO mocks",
        long: "Clears all registered IO mocks. Use it when test doubles are no longer needed or are affecting later tests.",
    },
    ActionCommand {
        name: "!stub",
        args: "<function> \"<pattern>\" -> <expression>",
        short: "stub a user function — intercept calls and return a typed expression",
        long: "Registers a function stub for testing. During `+test`, if a call to the named function matches the pattern, the expression is evaluated and returned instead of the function body. The expression is raw Adapsis code: Ok(\"done\"), Err(\"fail\"), \"text\", 42, etc. Use `!unstub` to clear.",
    },
    ActionCommand {
        name: "!unstub",
        args: "",
        short: "clear all function stubs",
        long: "Clears all registered function stubs.",
    },
    ActionCommand {
        name: "!msg",
        args: "<agent> <message>",
        short: "send a message to an agent (or 'main' for the main session). Agents see messages in their feedback.",
        long: "Sends a message to another agent or to `main`. The recipient sees it in `?inbox` or agent feedback.",
    },
    ActionCommand {
        name: "!opencode",
        args: "<description>",
        short: "request Rust-level change via OpenCode (rebuild + restart)",
        long: "Requests a Rust-level runtime change through OpenCode, then rebuilds and restarts the runtime if the change succeeds. Use it for missing builtins, runtime bugs, or core language behavior changes.",
    },
    ActionCommand {
        name: "+module",
        args: "<Name>",
        short: "switch module context — subsequent +fn/+type definitions go into this module",
        long: "Switches the active module context for subsequent `+fn` and `+type` definitions. It changes where new definitions are created until you switch again.",
    },
    ActionCommand {
        name: "+doc",
        args: "\"<description>\"",
        short: "attach documentation to the preceding module or function",
        long: "Sets the documentation string for the preceding `+module` (immediately after the module name line) or `+fn` (immediately after the `+end` that closes the function). Docs appear in `?symbols` and `?source` output.",
    },
    ActionCommand {
        name: "!done",
        args: "",
        short: "signal that the current task is complete",
        long: "Signals that the current task is complete. It is rejected if there are untested functions with more than two statements.",
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

/// Pre-built set of all builtin names (sync + IO) and their aliases for O(1) lookup.
/// Lazily initialized on first access.
static ALL_BUILTIN_SET: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        let mut set = std::collections::HashSet::new();
        for b in BUILTINS {
            set.insert(b.name);
            for alias in b.aliases {
                set.insert(alias);
            }
        }
        for b in IO_BUILTINS {
            set.insert(b.name);
            for alias in b.aliases {
                set.insert(alias);
            }
        }
        set
    });

/// Pre-built set of IO builtin names and aliases for O(1) lookup.
/// Lazily initialized on first access.
static IO_BUILTIN_SET: std::sync::LazyLock<std::collections::HashSet<&'static str>> =
    std::sync::LazyLock::new(|| {
        let mut set = std::collections::HashSet::new();
        for b in IO_BUILTINS {
            set.insert(b.name);
            for alias in b.aliases {
                set.insert(alias);
            }
        }
        set
    });

/// Check if a name is a builtin (sync or IO). O(1) HashSet lookup.
pub fn is_builtin(name: &str) -> bool {
    ALL_BUILTIN_SET.contains(name)
}

/// Check if a name is an IO builtin (requires +await). O(1) HashSet lookup.
pub fn is_io_builtin(name: &str) -> bool {
    IO_BUILTIN_SET.contains(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ═════════════════════════════════════════════════════════════════════
    // Registry non-empty and well-formed
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtins_registry_not_empty() {
        assert!(!BUILTINS.is_empty(), "BUILTINS should not be empty");
        assert!(!IO_BUILTINS.is_empty(), "IO_BUILTINS should not be empty");
        assert!(!QUERIES.is_empty(), "QUERIES should not be empty");
        assert!(!ACTIONS.is_empty(), "ACTIONS should not be empty");
    }

    #[test]
    fn builtins_have_names_and_descriptions() {
        for b in BUILTINS {
            assert!(!b.name.is_empty(), "builtin name should not be empty");
            assert!(
                !b.short.is_empty(),
                "builtin '{}' should have a short description",
                b.name
            );
        }
    }

    #[test]
    fn io_builtins_have_names_and_descriptions() {
        for b in IO_BUILTINS {
            assert!(!b.name.is_empty(), "IO builtin name should not be empty");
            assert!(
                !b.short.is_empty(),
                "IO builtin '{}' should have a short description",
                b.name
            );
        }
    }

    #[test]
    fn queries_have_names() {
        for q in QUERIES {
            assert!(!q.name.is_empty(), "query name should not be empty");
            assert!(
                q.name.starts_with('?'),
                "query '{}' should start with ?",
                q.name
            );
        }
    }

    #[test]
    fn actions_have_names() {
        for a in ACTIONS {
            assert!(!a.name.is_empty(), "action name should not be empty");
            assert!(
                a.name.starts_with('!') || a.name.starts_with('+'),
                "action '{}' should start with ! or +",
                a.name
            );
        }
    }

    #[test]
    fn no_duplicate_builtin_names() {
        let mut seen = std::collections::HashSet::new();
        for b in BUILTINS {
            assert!(seen.insert(b.name), "duplicate builtin: {}", b.name);
        }
    }

    #[test]
    fn no_duplicate_io_builtin_names() {
        let mut seen = std::collections::HashSet::new();
        for b in IO_BUILTINS {
            assert!(seen.insert(b.name), "duplicate IO builtin: {}", b.name);
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Specific builtins are registered
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn core_string_builtins_registered() {
        for name in &[
            "concat",
            "len",
            "substring",
            "split",
            "join",
            "trim",
            "starts_with",
            "ends_with",
            "contains",
            "index_of",
            "char_at",
        ] {
            assert!(
                is_builtin(name),
                "core string builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn core_math_builtins_registered() {
        for name in &["abs", "min", "max", "pow", "floor", "sqrt"] {
            assert!(
                is_builtin(name),
                "core math builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn core_list_builtins_registered() {
        for name in &["list", "push", "get", "join"] {
            assert!(
                is_builtin(name),
                "core list builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn core_conversion_builtins_registered() {
        for name in &["to_string", "to_int", "to_hex"] {
            assert!(
                is_builtin(name),
                "core conversion builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn core_bitwise_builtins_registered() {
        for name in &[
            "bit_and",
            "bit_or",
            "bit_xor",
            "bit_not",
            "shl",
            "shr",
            "left_rotate",
        ] {
            assert!(
                is_builtin(name),
                "core bitwise builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn result_constructors_registered() {
        assert!(is_builtin("Ok"), "Ok should be registered");
        assert!(is_builtin("Err"), "Err should be registered");
        assert!(is_builtin("Some"), "Some should be registered");
    }

    #[test]
    fn regex_builtins_registered() {
        assert!(is_builtin("regex_match"));
        assert!(is_builtin("regex_replace"));
    }

    #[test]
    fn json_builtins_registered() {
        assert!(is_builtin("json_get"));
        assert!(is_builtin("json_array_len"));
        assert!(is_builtin("json_escape"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // IO builtins
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn io_builtins_registered() {
        for name in &[
            "http_get",
            "http_post",
            "shell_exec",
            "file_read",
            "file_write",
            "llm_call",
            "sleep",
            "print",
            "println",
        ] {
            assert!(
                is_io_builtin(name),
                "IO builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn tcp_builtins_registered() {
        for name in &[
            "tcp_connect",
            "tcp_listen",
            "tcp_accept",
            "tcp_read",
            "tcp_write",
            "tcp_close",
        ] {
            assert!(
                is_io_builtin(name),
                "TCP builtin '{name}' should be registered"
            );
        }
    }

    #[test]
    fn io_builtins_are_also_builtins() {
        // is_builtin should return true for IO builtins too
        for name in &["http_get", "shell_exec", "llm_call"] {
            assert!(
                is_builtin(name),
                "IO builtin '{name}' should also return true for is_builtin"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // is_builtin / is_io_builtin lookup
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn is_builtin_false_for_unknown() {
        assert!(!is_builtin("nonexistent_function"));
        assert!(!is_builtin(""));
        assert!(!is_builtin("my_custom_fn"));
    }

    #[test]
    fn is_io_builtin_false_for_sync_builtins() {
        assert!(!is_io_builtin("concat"));
        assert!(!is_io_builtin("len"));
        assert!(!is_io_builtin("abs"));
    }

    #[test]
    fn is_io_builtin_false_for_unknown() {
        assert!(!is_io_builtin("nonexistent"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Alias lookup
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn aliases_work_for_is_builtin() {
        // substring has alias "substr"
        assert!(is_builtin("substr"), "alias 'substr' should resolve");
        // to_string has alias "str"
        assert!(is_builtin("str"), "alias 'str' should resolve");
        // to_int has aliases "parse_int", "int"
        assert!(is_builtin("parse_int"), "alias 'parse_int' should resolve");
        assert!(is_builtin("int"), "alias 'int' should resolve");
    }

    #[test]
    fn aliases_included_in_all_builtin_names() {
        let names = all_builtin_names();
        assert!(
            names.contains(&"substr"),
            "all_builtin_names should include alias 'substr'"
        );
        assert!(
            names.contains(&"str"),
            "all_builtin_names should include alias 'str'"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // all_builtin_names
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn all_builtin_names_includes_sync_and_io() {
        let names = all_builtin_names();
        // Sync builtins
        assert!(names.contains(&"concat"));
        assert!(names.contains(&"len"));
        // IO builtins
        assert!(names.contains(&"http_get"));
        assert!(names.contains(&"shell_exec"));
    }

    #[test]
    fn all_builtin_names_no_empty_entries() {
        let names = all_builtin_names();
        for name in &names {
            assert!(
                !name.is_empty(),
                "all_builtin_names should not contain empty strings"
            );
        }
    }

    // ═════════════════════════════════════════════════════════════════════
    // Builtin categories
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn builtin_categories_correct() {
        let concat = BUILTINS.iter().find(|b| b.name == "concat").unwrap();
        assert!(
            concat.category == BuiltinCategory::String,
            "concat should be String category"
        );

        let abs = BUILTINS.iter().find(|b| b.name == "abs").unwrap();
        assert!(
            abs.category == BuiltinCategory::Math,
            "abs should be Math category"
        );

        let list = BUILTINS.iter().find(|b| b.name == "list").unwrap();
        assert!(
            list.category == BuiltinCategory::List,
            "list should be List category"
        );

        let bit_and = BUILTINS.iter().find(|b| b.name == "bit_and").unwrap();
        assert!(
            bit_and.category == BuiltinCategory::Bitwise,
            "bit_and should be Bitwise category"
        );

        let to_string = BUILTINS.iter().find(|b| b.name == "to_string").unwrap();
        assert!(
            to_string.category == BuiltinCategory::Conversion,
            "to_string should be Conversion category"
        );

        let ok = BUILTINS.iter().find(|b| b.name == "Ok").unwrap();
        assert!(
            ok.category == BuiltinCategory::Result,
            "Ok should be Result category"
        );

        let regex_match = BUILTINS.iter().find(|b| b.name == "regex_match").unwrap();
        assert!(
            regex_match.category == BuiltinCategory::Regex,
            "regex_match should be Regex category"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // format_for_prompt
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn format_for_prompt_includes_all_sections() {
        let prompt = format_for_prompt();
        assert!(
            prompt.contains("### Built-in Functions"),
            "should have builtins section"
        );
        assert!(
            prompt.contains("### IO Functions"),
            "should have IO section"
        );
        assert!(
            prompt.contains("### Queries"),
            "should have queries section"
        );
        assert!(
            prompt.contains("### Commands"),
            "should have commands section"
        );
    }

    #[test]
    fn format_for_prompt_includes_specific_builtins() {
        let prompt = format_for_prompt();
        assert!(prompt.contains("concat"), "prompt should mention concat");
        assert!(
            prompt.contains("http_get"),
            "prompt should mention http_get"
        );
        assert!(
            prompt.contains("?symbols"),
            "prompt should mention ?symbols"
        );
        assert!(prompt.contains("!eval"), "prompt should mention !eval");
    }

    #[test]
    fn format_for_prompt_includes_aliases() {
        let prompt = format_for_prompt();
        assert!(prompt.contains("substr"), "prompt should show substr alias");
    }

    // ═════════════════════════════════════════════════════════════════════
    // Specific action/query registration
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn key_actions_registered() {
        let action_names: Vec<&str> = ACTIONS.iter().map(|a| a.name).collect();
        assert!(action_names.contains(&"!eval"));
        assert!(action_names.contains(&"+test"));
        assert!(action_names.contains(&"+module"));
        assert!(action_names.contains(&"!remove"));
        assert!(action_names.contains(&"!replace"));
        assert!(action_names.contains(&"!plan"));
        assert!(action_names.contains(&"!roadmap"));
        assert!(action_names.contains(&"!done"));
        assert!(action_names.contains(&"!mock"));
        assert!(action_names.contains(&"!unmock"));
    }

    #[test]
    fn key_queries_registered() {
        let query_names: Vec<&str> = QUERIES.iter().map(|q| q.name).collect();
        assert!(query_names.contains(&"?symbols"));
        assert!(query_names.contains(&"?source"));
        assert!(query_names.contains(&"?deps"));
        assert!(query_names.contains(&"?tasks"));
    }

    // ═════════════════════════════════════════════════════════════════════
    // Query IO builtins registration
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn query_io_builtins_registered() {
        for name in &[
            "query_symbols",
            "query_symbols_detail",
            "query_source",
            "query_callers",
            "query_callees",
            "query_deps",
            "query_deps_all",
            "query_routes",
            "query_tasks",
            "query_library",
            "library_reload",
            "library_errors",
        ] {
            assert!(
                is_io_builtin(name),
                "query IO builtin '{name}' should be registered"
            );
            assert!(
                is_builtin(name),
                "query IO builtin '{name}' should also be a builtin"
            );
        }
    }

    #[test]
    fn query_io_builtins_have_descriptions() {
        for name in &[
            "query_symbols",
            "query_symbols_detail",
            "query_source",
            "query_callers",
            "query_callees",
            "query_deps",
            "query_deps_all",
            "query_routes",
            "query_tasks",
            "query_library",
            "library_reload",
            "library_errors",
        ] {
            let builtin = IO_BUILTINS.iter().find(|b| b.name == *name);
            assert!(
                builtin.is_some(),
                "IO builtin '{name}' should exist in IO_BUILTINS"
            );
            let b = builtin.unwrap();
            assert!(
                !b.short.is_empty(),
                "'{name}' should have short description"
            );
            assert!(!b.long.is_empty(), "'{name}' should have long description");
            assert!(
                b.category == BuiltinCategory::Io,
                "'{name}' should be Io category"
            );
        }
    }

    #[test]
    fn format_for_prompt_includes_query_builtins() {
        let prompt = format_for_prompt();
        assert!(
            prompt.contains("query_symbols"),
            "prompt should mention query_symbols"
        );
        assert!(
            prompt.contains("query_source"),
            "prompt should mention query_source"
        );
        assert!(
            prompt.contains("query_tasks"),
            "prompt should mention query_tasks"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // Introspection alias registration
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn introspection_aliases_resolve_as_io_builtins() {
        for alias in &[
            "symbols_list",
            "source_get",
            "callers_get",
            "callees_get",
            "deps_get",
            "routes_list",
        ] {
            assert!(
                is_io_builtin(alias),
                "introspection alias '{alias}' should resolve as IO builtin"
            );
            assert!(
                is_builtin(alias),
                "introspection alias '{alias}' should resolve as builtin"
            );
        }
    }

    #[test]
    fn introspection_aliases_included_in_all_builtin_names() {
        let names = all_builtin_names();
        for alias in &[
            "symbols_list",
            "source_get",
            "callers_get",
            "callees_get",
            "deps_get",
            "routes_list",
        ] {
            assert!(
                names.contains(alias),
                "all_builtin_names should include introspection alias '{alias}'"
            );
        }
    }

    #[test]
    fn introspection_alias_not_registered_as_unknown() {
        // Aliases should NOT be found as primary IO_BUILTINS names —
        // they are aliases on existing entries
        for alias in &[
            "symbols_list",
            "source_get",
            "callers_get",
            "callees_get",
            "deps_get",
            "routes_list",
        ] {
            let is_primary = IO_BUILTINS.iter().any(|b| b.name == *alias);
            assert!(
                !is_primary,
                "'{alias}' should be an alias, not a primary name"
            );
        }
    }

    #[test]
    fn format_for_prompt_shows_introspection_aliases() {
        let prompt = format_for_prompt();
        assert!(
            prompt.contains("symbols_list"),
            "prompt should show symbols_list alias"
        );
        assert!(
            prompt.contains("source_get"),
            "prompt should show source_get alias"
        );
        assert!(
            prompt.contains("routes_list"),
            "prompt should show routes_list alias"
        );
    }

    // ═════════════════════════════════════════════════════════════════════
    // HashSet-based lookup tests (name interning optimization)
    // ═════════════════════════════════════════════════════════════════════

    #[test]
    fn hashset_lookup_matches_linear_scan_for_builtins() {
        // Verify that the HashSet-based is_builtin matches what a linear scan would find
        for b in BUILTINS {
            assert!(
                is_builtin(b.name),
                "HashSet should find builtin '{}'",
                b.name
            );
            for alias in b.aliases {
                assert!(
                    is_builtin(alias),
                    "HashSet should find alias '{}' for '{}'",
                    alias,
                    b.name
                );
            }
        }
    }

    #[test]
    fn hashset_lookup_matches_linear_scan_for_io_builtins() {
        // Verify that the HashSet-based is_io_builtin matches what a linear scan would find
        for b in IO_BUILTINS {
            assert!(
                is_io_builtin(b.name),
                "HashSet should find IO builtin '{}'",
                b.name
            );
            for alias in b.aliases {
                assert!(
                    is_io_builtin(alias),
                    "HashSet should find IO alias '{}' for '{}'",
                    alias,
                    b.name
                );
            }
        }
    }

    #[test]
    fn hashset_rejects_unknown_names() {
        assert!(!is_builtin("totally_made_up_function"));
        assert!(!is_io_builtin("totally_made_up_function"));
        assert!(!is_builtin(""));
        assert!(!is_io_builtin(""));
    }
}
