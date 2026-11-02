"""Bounded canonical-runtime record envelope codec; not an external JSON validator or authority check."""

@dataclass
class RecordField:
    name: str
    value: str

@dataclass
class RecordRow:
    key: str
    revision: str
    exists: bool
    fields: list[RecordField]

@union
class RecordDecode:
    RecordsDecoded: tuple[list[RecordRow]]
    RecordsMalformed: tuple[()]

def quote(value: str) -> str:
    return '"' + json_escape(value) + '"'

def format_valid(body: str) -> bool:
    if not json_valid(body):
        return False
    if json_kind(body, "format") != "string":
        return False
    return json_get(body, "format") == "records_v1"

def field(row: RecordRow, name: str) -> str:
    for item in row.fields:
        if item.name == name:
            return item.value
    return ""

def fields_are(row: RecordRow, names: list[str]) -> bool:
    if not row.exists:
        return False
    if len(row.fields) != len(names):
        return False
    for name in names:
        count: int = 0
        for item in row.fields:
            if item.name == name:
                count = count + 1
        if count != 1:
            return False
    return True

def decode(body: str, keys: list[str]) -> RecordDecode:
    if len(body) > 600000:
        return RecordsMalformed
    if not format_valid(body):
        return RecordsMalformed
    if json_kind(body, "records") != "array":
        return RecordsMalformed
    rows_json: str = json_get(body, "records")
    if json_array_len(rows_json) != len(keys):
        return RecordsMalformed
    if len(keys) > 32:
        return RecordsMalformed
    rows: list[RecordRow] = []
    index: int = 0
    for key in keys:
        path: str = "records." + to_string(index)
        fields: list[RecordField] = []
        revision: str = ""
        present: bool = False
        kind: str = json_kind(body, path)
        if kind == "object":
            present = True
            if json_kind(body, path + ".key") != "string":
                return RecordsMalformed
            if json_get(body, path + ".key") != key:
                return RecordsMalformed
            if json_kind(body, path + ".revision") != "string":
                return RecordsMalformed
            revision = json_get(body, path + ".revision")
            if not regex_match("^[1-9][0-9]*$", revision):
                return RecordsMalformed
            if json_kind(body, path + ".fields") != "array":
                return RecordsMalformed
            fields_json: str = json_get(body, path + ".fields")
            size: int = json_array_len(fields_json)
            if size < 1 or size > 32:
                return RecordsMalformed
            cursor: int = 0
            while cursor < size:
                field_path: str = path + ".fields." + to_string(cursor)
                if json_kind(body, field_path + ".name") != "string":
                    return RecordsMalformed
                if json_kind(body, field_path + ".value") != "string":
                    return RecordsMalformed
                name: str = json_get(body, field_path + ".name")
                value: str = json_get(body, field_path + ".value")
                if len(name) < 1 or len(name) > 64 or len(value) > 4096:
                    return RecordsMalformed
                for previous in fields:
                    if previous.name == name:
                        return RecordsMalformed
                item: RecordField = RecordField(name=name, value=value)
                fields = push(fields, item)
                cursor = cursor + 1
        elif kind != "null":
            return RecordsMalformed
        row: RecordRow = RecordRow(key=key, revision=revision, exists=present, fields=fields)
        rows = push(rows, row)
        index = index + 1
    return RecordsDecoded(rows)

def read_request(keys: list[str]) -> str:
    encoded: str = ""
    separator: str = ""
    for key in keys:
        encoded = encoded + separator + quote(key)
        separator = ","
    return '{"version":1,"keys":[' + encoded + ']} '

def condition(row: RecordRow) -> str:
    expected: str = '{"kind":"absent"}'
    if row.exists:
        expected = '{"kind":"revision","revision":' + quote(row.revision) + '}'
    return '{"key":' + quote(row.key) + ',"expected":' + expected + '}'

def put(row: RecordRow) -> str:
    encoded: str = ""
    separator: str = ""
    for item in row.fields:
        encoded = encoded + separator + '{"name":' + quote(item.name) + ',"value":' + quote(item.value) + '}'
        separator = ","
    return '{"key":' + quote(row.key) + ',"fields":[' + encoded + ']}'

def batch(conditions: list[RecordRow], puts: list[RecordRow]) -> str:
    encoded_conditions: str = ""
    encoded_puts: str = ""
    separator: str = ""
    for row in conditions:
        encoded_conditions = encoded_conditions + separator + condition(row)
        separator = ","
    separator = ""
    for put_row in puts:
        encoded_puts = encoded_puts + separator + put(put_row)
        separator = ","
    return '{"version":1,"conditions":[' + encoded_conditions + '],"puts":[' + encoded_puts + ']}'

def replaced(row: RecordRow, name: str, value: str) -> RecordRow:
    fields: list[RecordField] = []
    for item in row.fields:
        if item.name == name:
            replacement: RecordField = RecordField(name=name, value=value)
            fields = push(fields, replacement)
        else:
            fields = push(fields, item)
    return RecordRow(key=row.key, revision=row.revision, exists=row.exists, fields=fields)

def receipt_matches(body: str, request_id: str) -> bool:
    if not format_valid(body):
        return False
    if json_kind(body, "receipt") != "object":
        return False
    if json_kind(body, "receipt.request_id") != "string":
        return False
    if json_get(body, "receipt.request_id") != request_id:
        return False
    if json_kind(body, "receipt.revisions") != "array":
        return False
    revisions: str = json_get(body, "receipt.revisions")
    count: int = json_array_len(revisions)
    if count < 1 or count > 16:
        return False
    index: int = 0
    seen: list[str] = []
    while index < count:
        path: str = "receipt.revisions." + to_string(index)
        if json_kind(body, path + ".key") != "string":
            return False
        if json_kind(body, path + ".revision") != "string":
            return False
        key: str = json_get(body, path + ".key")
        revision: str = json_get(body, path + ".revision")
        if key == "":
            return False
        if not regex_match("^[1-9][0-9]*$", revision):
            return False
        for previous in seen:
            if previous == key:
                return False
        seen = push(seen, key)
        index = index + 1
    return True
