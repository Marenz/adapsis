use super::{Builtin, BuiltinCategory};

pub(super) const READ: Builtin = Builtin {
    name: "records_read",
    aliases: &[],
    category: BuiltinCategory::Io,
    short: "read bounded record keys: records_read(request:String) -> Result<String>",
    long: "Requires await and an authorized server-bound record namespace plus independently attested caller. No production attested caller path is enabled; operator provisioning alone is insufficient. Takes strict version-1 JSON keys envelope, never principal/namespace/path. Returns records_v1 JSON with records in key order and null for absence. Catch categorical Err; no authority is inferred from record/workspace strings.",
};
pub(super) const COMMIT: Builtin = Builtin {
    name: "records_commit",
    aliases: &[],
    category: BuiltinCategory::Io,
    short: "atomic record CAS: records_commit(request_id:String,batch:String) -> Result<String>",
    long: "Requires await and explicit server-bound write grant plus independently attested caller. No production attested caller path is enabled; operator provisioning alone is insufficient. Strict version-1 conditions/puts envelope commits atomically with an idempotency receipt. Returns records_v1 receipt. Same ID/exact semantic batch replays; changed batch refuses. Commit/channel uncertainty returns record_commit_outcome_unknown: reconcile exact request/receipt, never blindly retry with a new ID. No raw database errors or implicit retries.",
};
pub(super) const RECEIPT: Builtin = Builtin {
    name: "records_receipt",
    aliases: &[],
    category: BuiltinCategory::Io,
    short: "resolve own commit receipt: records_receipt(request_id:String) -> Result<String>",
    long: "Requires await and the same server-bound write authority/context/grant and independently attested caller as commit, not merely read permission. No production attested caller path is enabled. Returns records_v1 with receipt object or null for absence. No identity/namespace/path arguments, cross-principal receipt access or automatic retry.",
};
