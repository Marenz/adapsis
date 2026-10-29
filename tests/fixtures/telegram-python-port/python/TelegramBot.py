@union
class TgSendAttempt:
    TgSent: tuple[str]
    TgRejected: tuple[str]
    TgUncertain: tuple[str]

telegram_token: str = ""

def extract_chat_id(payload: str) -> int:
    """Extract chat ID from Telegram webhook payload"""
    val: str = json_get(payload, "message.chat.id")
    if val == "":
        return 0
    return to_int(val)

def extract_user_id(payload: str) -> int:
    """Extract sender user ID from Telegram webhook payload"""
    val: str = json_get(payload, "message.from.id")
    if val == "":
        return 0
    return to_int(val)

def extract_chat_type(payload: str) -> str:
    """Extract chat type (private/group/supergroup/channel) from Telegram webhook payload"""
    val: str = json_get(payload, "message.chat.type")
    return val

def extract_text(payload: str) -> str:
    """Extract message text, falling back to a photo caption"""
    val: str = json_get(payload, "message.text")
    if val == "":
        return json_get(payload, "message.caption")
    return val

def extract_image_file_id(payload: str) -> str:
    """Extract an image file ID from either Telegram's largest photo variant or an image document."""
    photos: str = json_get(payload, "message.photo")
    if photos != "":
        n: int = json_array_len(photos)
        if n > 0:
            largest: str = json_get(photos, to_string(n - 1))
            return json_get(largest, "file_id")
    mime: str = json_get(payload, "message.document.mime_type")
    if starts_with(mime, "image/"):
        return json_get(payload, "message.document.file_id")
    return ""

def extract_video_file_id(payload: str) -> str:
    """Extract a video file ID from Telegram's message.video object or a document whose MIME type starts with video/."""
    file_id: str = json_get(payload, "message.video.file_id")
    if file_id != "":
        return file_id
    mime: str = json_get(payload, "message.document.mime_type")
    if starts_with(mime, "video/"):
        return json_get(payload, "message.document.file_id")
    return ""

def format_sender(user_id: int, name: str, text: str) -> str:
    """Build the sender-prefixed message the model sees in a group. The numeric id stays because it is stable and unspoofable; the display name is added because a model cannot address someone as @<number>, and pattern-matching a speaker from conversation history instead gets it wrong."""
    if name == "":
        return concat("[user:", to_string(user_id), "] ", text)
    return concat("[user:", to_string(user_id), " ", name, "] ", text)

def is_admin_in(user_id: int, ids: str) -> bool:
    """Check if user_id is in comma-separated ID list (testable pure version)"""
    uid_str: str = to_string(user_id)
    needle: str = concat(",", concat(uid_str, ","))
    haystack: str = concat(",", concat(ids, ","))
    return contains(haystack, needle)

def telegram_response_ok(response: str) -> bool:
    """True only when Telegram explicitly reported success. A missing ok:true - bad entities, a malformed body or a transport failure - counts as failure so the caller can fall back."""
    return index_of(response, '"ok":true') >= 0

def telegram_file_path_response(file_info: str) -> str:
    """Extract a getFile path or preserve Telegram's JSON error description when HTTP 400 bodies are returned as ordinary response text."""
    file_path: str = json_get(file_info, "result.file_path")
    if file_path != "":
        return file_path
    description: str = json_get(file_info, "description")
    if description != "":
        return concat("ERROR: ", description)
    return "ERROR: Telegram getFile returned no file path"

def parse_id_list(raw: str) -> str:
    """Normalize an ID file into a comma-separated ID string: accepts newline- and/or comma-separated IDs, trims whitespace, skips blank lines and # comments. Returns empty string if the file yields nothing (caller decides the fallback)."""
    lines: list[str] = split(raw, "\n")
    out: str = ""
    i: int = 0
    n: int = len(lines)
    while i < n:
        line: str = get(lines, i)
        cells: list[str] = split(line, ",")
        j: int = 0
        m: int = len(cells)
        while j < m:
            tok: str = trim(get(cells, j))
            if tok != "":
                if starts_with(tok, "#") == False:
                    if out == "":
                        out = tok
                    else:
                        out = concat(out, concat(",", tok))
            j = j + 1
        i = i + 1
    return out

def extract_sender_name(payload: str) -> str:
    """The sender's display name: first+last, falling back to username, else empty. Shared by source_metadata and the group sender prefix so the name the model reads inline can never drift from the name recorded for provenance."""
    first_name: str = json_get(payload, "message.from.first_name")
    last_name: str = json_get(payload, "message.from.last_name")
    username: str = json_get(payload, "message.from.username")
    speaker_name: str = first_name
    if last_name != "":
        speaker_name = concat(first_name, " ", last_name)
    if speaker_name == "":
        speaker_name = username
    return speaker_name

def speaker_principal(payload: str) -> str:
    """The sender's stable Ladybug principal. This is the ONLY identity channel: the model-facing [user:...] prefix is prose and must never be parsed back into an identity, because its format changes whenever the display format changes."""
    user_id: int = TelegramBot.extract_user_id(payload)
    if user_id == 0:
        return ""
    return concat("telegram:user:", to_string(user_id))

def source_metadata(payload: str) -> str:
    """Preserve Telegram message identity, timestamp, speaker principal/display name and attachment file id/name/MIME/duration/size for Ladybug provenance. speaker_id is what the runtime resolves the principal from."""
    message_id: str = to_string(json_get(payload, "message.message_id"))
    attachment_ref: str = TelegramBot.extract_image_file_id(payload)
    if attachment_ref == "":
        attachment_ref = TelegramBot.extract_video_file_id(payload)
    if attachment_ref == "":
        attachment_ref = json_get(payload, "message.voice.file_id")
    attachment_name: str = json_get(payload, "message.video.file_name")
    if attachment_name == "":
        attachment_name = json_get(payload, "message.document.file_name")
    attachment_mime: str = json_get(payload, "message.video.mime_type")
    if attachment_mime == "":
        attachment_mime = json_get(payload, "message.document.mime_type")
    attachment_duration: str = ""
    if json_get(payload, "message.video.duration") != "":
        attachment_duration = to_string(json_get(payload, "message.video.duration"))
    attachment_size: str = ""
    if json_get(payload, "message.video.file_size") != "":
        attachment_size = to_string(json_get(payload, "message.video.file_size"))
    if attachment_size == "":
        if json_get(payload, "message.document.file_size") != "":
            attachment_size = to_string(json_get(payload, "message.document.file_size"))
    speaker_name: str = TelegramBot.extract_sender_name(payload)
    speaker_id: str = TelegramBot.speaker_principal(payload)
    created_at_ms: int = to_int(to_string(json_get(payload, "message.date"))) * 1000
    return concat('{"message_id":"', json_escape(message_id), '","speaker_id":"', json_escape(speaker_id), '","speaker_name":"', json_escape(speaker_name), '","created_at_ms":', to_string(created_at_ms), ',"attachment_ref":"', json_escape(attachment_ref), '","attachment_name":"', json_escape(attachment_name), '","attachment_mime":"', json_escape(attachment_mime), '","attachment_duration_seconds":"', json_escape(attachment_duration), '","attachment_size_bytes":"', json_escape(attachment_size), '"}')

def send_message_body(chat_id: int, text: str, html: bool) -> str:
    """Build the sendMessage JSON body for either HTML-parsed or plain text."""
    if html:
        return concat('{"chat_id":', to_string(chat_id), ',"text":"', json_escape(text), '","parse_mode":"HTML"}')
    return concat('{"chat_id":', to_string(chat_id), ',"text":"', json_escape(text), '"}')

@effects("io", "fail")
async def send_message_part(chat_id: int, text: str, html: bool) -> TgSendAttempt:
    """Send one attempt. Only a parsed Boolean ok field classifies Sent/Rejected; transport errors or nonboolean status are Uncertain. Malformed JSON raises a terminal parse error. No resend occurs here."""
    url: str = concat("<redacted>", telegram_token, "/sendMessage")
    response: Result[str] = await http_post(url, TelegramBot.send_message_body(chat_id, text, html))
    match response:
        case Ok(body):
            status: str = to_string(json_get(body, "ok"))
            if status == "true":
                return TgSent(body)
            if status == "false":
                return TgRejected(body)
            return TgUncertain(body)
        case Err(error):
            return TgUncertain(error)

@effects("io", "fail")
async def send_message_parts(chat_id: int, parts: list[str], i: int, last: str) -> TgSendAttempt:
    """Attempt chunks in order. Only confirmed Boolean false permits one plain fallback; uncertainty or rejection of that fallback stops remaining chunks. Sent requires a confirmed attempt. The legacy last parameter cannot manufacture success and is retained only for input compatibility."""
    if i < 0 or i >= len(parts):
        return TgUncertain("no chunk at index")
    part: str = get(parts, i)
    attempt: TgSendAttempt = await TelegramBot.send_message_part(chat_id, part, True)
    match attempt:
        case TgRejected(_):
            attempt: TgSendAttempt = await TelegramBot.send_message_part(chat_id, part, False)
        case _:
            pass
    match attempt:
        case TgSent(body):
            if i + 1 >= len(parts):
                return attempt
            following: TgSendAttempt = await TelegramBot.send_message_parts(chat_id, parts, i + 1, body)
            return following
        case TgRejected(_):
            return attempt
        case TgUncertain(_):
            return attempt

@effects("io", "fail")
async def send_message(chat_id: int, text: str) -> TgSendAttempt:
    """Format and split text, then propagate the send-chain outcome unchanged without another retry. The inherited splitter is not service-limit safe (#99)."""
    html: str = TgFormat.markdown_to_html(text)
    parts: list[str] = TgFormat.split_for_telegram(html, 3800)
    outcome: TgSendAttempt = await TelegramBot.send_message_parts(chat_id, parts, 0, "")
    return outcome

@effects("io", "fail")
async def send_reply(chat_id_str: str, text: str) -> TgSendAttempt:
    """Parse the destination and propagate the message outcome unchanged. This callable contract does not certify callback delivery acknowledgement."""
    chat_id: int = to_int(chat_id_str)
    outcome: TgSendAttempt = await TelegramBot.send_message(chat_id, text)
    return outcome
