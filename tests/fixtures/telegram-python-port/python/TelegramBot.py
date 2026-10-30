@union
class TgSendAttempt:
    TgSent: tuple[str]
    TgRejected: tuple[str]
    TgUncertain: tuple[str]

@union
class TgTextTurn:
    TgNoImmediateReply: tuple[str]
    TgDeliveryAttempt: tuple[TgSendAttempt]
    TgIgnored: tuple[str]
    TgDeferred: tuple[str]

telegram_token: str = ""
admin_user_ids: str = "<redacted>"
allowed_user_ids: str = ""
sandbox_perm_model: str = "family"

def intake_validate(payload: str) -> str:
    objects: list[str] = ["message", "message.chat", "message.from"]
    i: int = 0
    while i < len(objects):
        object_path: str = get(objects, i)
        if json_kind(payload, object_path) != "object":
            return "invalid required object"
        i = i + 1
    ints: list[str] = ["message.chat.id", "message.from.id", "message.date"]
    i = 0
    while i < len(ints):
        int_path: str = get(ints, i)
        if json_kind(payload, int_path) != "int":
            return "invalid required integer"
        i = i + 1
    if to_int(json_get(payload, "message.chat.id")) == 0 or to_int(json_get(payload, "message.from.id")) <= 0:
        return "invalid identity bounds"
    date: int = to_int(json_get(payload, "message.date"))
    if date < 0 or date > 9223372036854775:
        return "invalid timestamp bounds"
    if json_kind(payload, "message.chat.type") != "string":
        return "invalid chat type"
    kind: str = json_get(payload, "message.chat.type")
    if kind != "private" and kind != "group" and kind != "supergroup":
        return "unsupported chat type"
    objects = ["video", "voice", "document", "sticker", "video_note", "audio", "animation", "contact", "location", "poll"]
    i = 0
    while i < len(objects):
        object_key: str = get(objects, i)
        kind = json_kind(payload, concat("message.", object_key))
        if kind != "missing" and kind != "object":
            return "invalid media object"
        i = i + 1
    kind = json_kind(payload, "message.photo")
    if kind != "missing" and kind != "array":
        return "invalid photo array"
    error: str = TelegramBot.intake_validate_fields(payload)
    if error != "":
        return error
    return TelegramBot.intake_validate_media(payload)

def intake_validate_fields(payload: str) -> str:
    kind: str = ""
    strings: list[str] = ["text", "caption", "from.first_name", "from.last_name", "from.username", "document.file_name", "document.mime_type", "document.file_id", "video.file_name", "video.mime_type", "video.file_id", "voice.file_id", "sticker.file_id", "sticker.emoji", "video_note.file_id", "audio.file_id", "animation.file_id", "contact.phone_number", "poll.question"]
    i: int = 0
    while i < len(strings):
        string_key: str = get(strings, i)
        kind = json_kind(payload, concat("message.", string_key))
        if kind != "missing" and kind != "string":
            return "invalid optional string"
        i = i + 1
    kind = json_kind(payload, "message.message_id")
    if kind != "missing" and kind != "int":
        return "invalid message id"
    numbers: list[str] = ["video.duration", "video.file_size", "document.file_size"]
    i = 0
    while i < len(numbers):
        number_key: str = get(numbers, i)
        path: str = concat("message.", number_key)
        kind = json_kind(payload, path)
        if kind != "missing":
            if kind != "int":
                return "invalid numeric metadata"
            if to_int(json_get(payload, path)) < 0:
                return "negative numeric metadata"
        i = i + 1
    kind = json_kind(payload, "message.location.latitude")
    if kind != "missing" and kind != "int" and kind != "float":
        return "invalid location latitude"
    return ""

def intake_validate_media(payload: str) -> str:
    path: str = ""
    if json_kind(payload, "message.photo") == "array":
        photos: str = json_get(payload, "message.photo")
        count: int = json_array_len(photos)
        if count > 0:
            path = concat("message.photo.", to_string(count - 1))
            if json_kind(payload, path) != "object":
                return "invalid selected photo"
            path = concat(path, ".file_id")
            if json_kind(payload, path) != "string":
                return "invalid selected photo id"
            if json_get(payload, path) == "":
                return "empty selected photo id"
    native: list[str] = ["video", "voice"]
    i: int = 0
    while i < len(native):
        native_key: str = get(native, i)
        path = concat("message.", native_key)
        if json_kind(payload, path) == "object":
            path = concat(path, ".file_id")
            if json_kind(payload, path) != "string":
                return "missing selected media id"
            if json_get(payload, path) == "":
                return "empty selected media id"
        i = i + 1
    mime: str = json_get(payload, "message.document.mime_type")
    if starts_with(mime, "image/") or starts_with(mime, "video/"):
        if json_kind(payload, "message.document.file_id") != "string":
            return "missing selected document id"
        if json_get(payload, "message.document.file_id") == "":
            return "empty selected document id"
    return ""

def describe_update_kind(payload: str) -> str:
    """Describe a message that carries no text, image or voice, so it can be announced instead of silently dropped. A dropped update is invisible to the model, which then truthfully but uselessly reports an empty message. Returns empty only when the update has no recognisable content at all."""
    doc_name: str = json_get(payload, "message.document.file_name")
    if doc_name != "":
        return concat("user sent a file (", doc_name, ") which I cannot read")
    doc_id: str = json_get(payload, "message.document.file_id")
    if doc_id != "":
        return "user sent a file which I cannot read"
    sticker: str = json_get(payload, "message.sticker.file_id")
    if sticker != "":
        return concat("user sent a sticker ", json_get(payload, "message.sticker.emoji"))
    video: str = json_get(payload, "message.video.file_id")
    if video != "":
        return "user sent a video"
    video_note: str = json_get(payload, "message.video_note.file_id")
    if video_note != "":
        return "user sent a video note which I cannot watch"
    audio: str = json_get(payload, "message.audio.file_id")
    if audio != "":
        return "user sent an audio file which I cannot hear"
    animation: str = json_get(payload, "message.animation.file_id")
    if animation != "":
        return "user sent a GIF which I cannot watch"
    contact: str = json_get(payload, "message.contact.phone_number")
    if contact != "":
        return "user shared a contact"
    location: str = json_get(payload, "message.location.latitude")
    if location != "":
        return "user shared a location"
    poll: str = json_get(payload, "message.poll.question")
    if poll != "":
        return concat("user sent a poll: ", poll)
    return ""

@effects("io", "fail")
async def process_update(payload: str) -> TgTextTurn:
    error: str = TelegramBot.intake_validate(payload)
    if error != "":
        return TgDeferred(error)
    text: str = TelegramBot.extract_text(payload)
    chat_id: int = TelegramBot.extract_chat_id(payload)
    user_id: int = TelegramBot.extract_user_id(payload)
    chat_type: str = TelegramBot.extract_chat_type(payload)
    admin: bool = TelegramBot.is_admin(user_id)
    allowed: bool = TelegramBot.is_allowed(user_id)
    if admin == False and allowed == False:
        return TgIgnored("unauthorized sender")
    metadata: str = TelegramBot.source_metadata(payload)
    sender_name: str = TelegramBot.extract_sender_name(payload)
    voice_file_id: str = json_get(payload, "message.voice.file_id")
    photo_file_id: str = TelegramBot.extract_image_file_id(payload)
    video_file_id: str = TelegramBot.extract_video_file_id(payload)
    if photo_file_id != "":
        file_path: str = await TelegramBot.telegram_file_path(photo_file_id)
        if starts_with(file_path, "ERROR: "):
            delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, TelegramBot.telegram_file_error_message(file_path, "photo"))
            return TgDeliveryAttempt(delivery)
        if file_path == "":
            delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, "I could not download that photo. Please try sending it again.")
            return TgDeliveryAttempt(delivery)
        download_url: str = concat("<redacted>", telegram_token, "/", file_path)
        photo: Attachment = await http_get_binary(download_url)
        if chat_type == "group" or chat_type == "supergroup":
            outcome: TgTextTurn = await TelegramBot.handle_group_attachment(chat_id, user_id, sender_name, text, admin, photo, metadata)
            return outcome
        if admin:
            outcome: TgTextTurn = await TelegramBot.handle_admin_dm_attachment(chat_id, user_id, text, photo, metadata)
            return outcome
        outcome: TgTextTurn = await TelegramBot.handle_user_dm_attachment(chat_id, user_id, text, photo, metadata)
        return outcome
    if video_file_id != "":
        file_path: str = await TelegramBot.telegram_file_path(video_file_id)
        if starts_with(file_path, "ERROR: "):
            delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, TelegramBot.telegram_file_error_message(file_path, "video"))
            return TgDeliveryAttempt(delivery)
        if file_path == "":
            delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, "I could not download that video. Please send it again as a video or video document.")
            return TgDeliveryAttempt(delivery)
        download_url: str = concat("<redacted>", telegram_token, "/", file_path)
        video: Attachment = await http_get_binary(download_url)
        if text == "":
            text = "[user sent a video; the original is available to video_stabilize()]"
        if chat_type == "group" or chat_type == "supergroup":
            outcome: TgTextTurn = await TelegramBot.handle_group_attachment(chat_id, user_id, sender_name, text, admin, video, metadata)
            return outcome
        if admin:
            outcome: TgTextTurn = await TelegramBot.handle_admin_dm_attachment(chat_id, user_id, text, video, metadata)
            return outcome
        outcome: TgTextTurn = await TelegramBot.handle_user_dm_attachment(chat_id, user_id, text, video, metadata)
        return outcome
    if text == "":
        if voice_file_id != "":
            spoken: str = await TelegramBot.transcribe_voice(voice_file_id)
            text = spoken
    if text == "":
        kind: str = TelegramBot.describe_update_kind(payload)
        if kind == "":
            return TgIgnored("no recognisable content")
        text = concat("[", kind, "]")
    if chat_type == "group" or chat_type == "supergroup":
        outcome: TgTextTurn = await TelegramBot.handle_group(chat_id, user_id, sender_name, text, admin, metadata)
        return outcome
    if admin:
        outcome: TgTextTurn = await TelegramBot.handle_admin_dm(chat_id, user_id, text, metadata)
        return outcome
    outcome: TgTextTurn = await TelegramBot.handle_user_dm(chat_id, user_id, text, metadata)
    return outcome

def transcription_upload_response(response: Result[str]) -> str:
    """Extract a transcript from a successful ASR response while preserving the voice placeholder for HTTP/transport failures."""
    match response:
        case Ok(body):
            if json_kind(body, "text") != "string":
                return "[voice message - transcription failed]"
            transcript: str = json_get(body, "text")
            if transcript == "":
                return "[voice message - transcription failed]"
            return transcript
        case Err(e):
            return "[voice message - transcription failed]"

@effects("io", "fail")
async def transcribe_voice(file_id: str) -> str:
    file_url: str = concat("<redacted>", telegram_token, "/getFile?file_id=", file_id)
    file_response: Result[str] = await http_get(file_url)
    match file_response:
        case Ok(file_info):
            if json_kind(file_info, "result.file_path") != "string":
                return "[voice message - could not get file]"
            file_path: str = json_get(file_info, "result.file_path")
            if file_path == "":
                return "[voice message - could not get file]"
            download_url: str = concat("<redacted>", telegram_token, "/", file_path)
            audio: Attachment = await http_get_binary(download_url)
            upload_response: Result[str] = await http_upload("http://127.0.0.1:8090/transcribe", audio, "file", "")
            return TelegramBot.transcription_upload_response(upload_response)
        case Err(e):
            return concat("[voice message - getFile failed: ", e, "]")

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

def is_admin(user_id: int) -> bool:
    """Check if a user ID is in the admin list (uses shared admin_user_ids)"""
    return TelegramBot.is_admin_in(user_id, admin_user_ids)

def is_allowed(user_id: int) -> bool:
    """Check if a user ID is in the allow-list (allowed_user_ids); reuses the comma-boundary matcher."""
    return TelegramBot.is_admin_in(user_id, allowed_user_ids)

@effects("io", "fail")
async def handle_group(chat_id: int, user_id: int, sender_name: str, text: str, admin: bool, metadata: str) -> TgTextTurn:
    """Preserve group takeover arguments and wrap the immediate send outcome without discarding failure. Empty reply is not delivery acknowledgement; #104 remains open."""
    ctx: str = concat("telegram:group:", to_string(chat_id))
    sender: str = TelegramBot.format_sender(user_id, sender_name, text)
    perm: str = ""
    if admin == False:
        perm = sandbox_perm_model
    reply: str = await llm_takeover(ctx, sender, "TelegramBot.send_reply", to_string(chat_id), perm, metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")

@effects("io", "fail")
async def handle_admin_dm(chat_id: int, user_id: int, text: str, metadata: str) -> TgTextTurn:
    """Preserve admin DM takeover arguments and propagate immediate delivery outcome. NoImmediateReply is not callback success."""
    ctx: str = concat("telegram:", to_string(user_id))
    reply: str = await llm_takeover(ctx, text, "TelegramBot.send_reply", to_string(chat_id), "", metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")

@effects("io", "fail")
async def handle_user_dm(chat_id: int, user_id: int, text: str, metadata: str) -> TgTextTurn:
    """Preserve sandboxed DM takeover arguments and propagate immediate delivery outcome. NoImmediateReply is not callback success."""
    ctx: str = concat("telegram:user:", to_string(user_id))
    reply: str = await llm_takeover(ctx, text, "TelegramBot.send_reply", to_string(chat_id), sandbox_perm_model, metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")

@effects("io", "fail")
async def process_text_update(payload: str) -> TgTextTurn:
    """NEW partial text-only entry, not process_update parity. json_kind validates exact ID/date/text/chat types and media presence before conversion, authorization and IO. Any present media defers; legitimate JSON-looking Strings remain text. Structured from.id determines authority; callback acknowledgement is not certified."""
    if json_kind(payload, "message.chat.id") != "int":
        return TgIgnored("invalid chat id")
    chat_id: int = to_int(json_get(payload, "message.chat.id"))
    if chat_id == 0:
        return TgIgnored("invalid chat id")
    if json_kind(payload, "message.from.id") != "int":
        return TgIgnored("invalid sender id")
    user_id: int = to_int(json_get(payload, "message.from.id"))
    if user_id <= 0:
        return TgIgnored("invalid sender id")
    if json_kind(payload, "message.date") != "int":
        return TgIgnored("invalid date")
    date: int = to_int(json_get(payload, "message.date"))
    if date < 0:
        return TgIgnored("invalid date")
    if json_kind(payload, "message.chat.type") != "string":
        return TgIgnored("unsupported chat type")
    chat_type: str = TelegramBot.extract_chat_type(payload)
    if contains(["private", "group", "supergroup"], chat_type) == False:
        return TgIgnored("unsupported chat type")
    media: list[str] = ["photo", "video", "voice", "document", "audio", "animation", "sticker", "video_note", "contact", "location", "poll"]
    i: int = 0
    while i < len(media):
        media_key: str = get(media, i)
        if json_kind(payload, concat("message.", media_key)) != "missing":
            return TgDeferred("media path pending")
        i = i + 1
    text_kind: str = json_kind(payload, "message.text")
    if text_kind == "missing":
        return TgIgnored("no text")
    if text_kind != "string":
        return TgDeferred("non-string text")
    text: str = json_get(payload, "message.text")
    if text == "":
        return TgIgnored("no text")
    admin: bool = TelegramBot.is_admin(user_id)
    allowed: bool = TelegramBot.is_allowed(user_id)
    if admin == False and allowed == False:
        return TgIgnored("unauthorized sender")
    metadata: str = TelegramBot.source_metadata(payload)
    sender_name: str = TelegramBot.extract_sender_name(payload)
    if chat_type == "group" or chat_type == "supergroup":
        outcome: TgTextTurn = await TelegramBot.handle_group(chat_id, user_id, sender_name, text, admin, metadata)
        return outcome
    if admin:
        outcome: TgTextTurn = await TelegramBot.handle_admin_dm(chat_id, user_id, text, metadata)
        return outcome
    outcome: TgTextTurn = await TelegramBot.handle_user_dm(chat_id, user_id, text, metadata)
    return outcome

@effects("io", "fail")
async def telegram_file_path(file_id: str) -> str:
    """Preserve the original getFile path/error-String contract using explicit builtin Result binding and matching. Malformed JSON remains terminal; no binary download occurs here."""
    file_url: str = concat("<redacted>", telegram_token, "/getFile?file_id=", file_id)
    response: Result[str] = await http_get(file_url)
    match response:
        case Ok(file_info):
            return TelegramBot.telegram_file_path_response(file_info)
        case Err(e):
            return concat("ERROR: ", e)

def telegram_file_error_message(error: str, kind: str) -> str:
    """Turn Telegram getFile failures into user-facing guidance, including the platform's 20 MB bot-download cap."""
    if contains(error, "file is too big"):
        return concat("Telegram bots can only download files up to 20 MB. Please send a smaller ", concat(kind, "."))
    return concat("I could not download that ", concat(kind, ". Please send it again."))

@effects("io", "fail")
async def handle_group_attachment(chat_id: int, user_id: int, sender_name: str, text: str, admin: bool, attachment: Attachment, metadata: str) -> TgTextTurn:
    """Preserve the group attachment takeover request and propagate immediate delivery outcome without another retry. Callback acknowledgement remains unverified (#104)."""
    ctx: str = concat("telegram:group:", to_string(chat_id))
    sender: str = TelegramBot.format_sender(user_id, sender_name, text)
    perm: str = ""
    if admin == False:
        perm = sandbox_perm_model
    reply: str = await llm_takeover(ctx, sender, "TelegramBot.send_reply", to_string(chat_id), perm, attachment, metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")

@effects("io", "fail")
async def handle_admin_dm_attachment(chat_id: int, user_id: int, text: str, attachment: Attachment, metadata: str) -> TgTextTurn:
    """Preserve admin attachment takeover arguments and propagate immediate delivery outcome. NoImmediateReply is not callback success."""
    ctx: str = concat("telegram:", to_string(user_id))
    reply: str = await llm_takeover(ctx, text, "TelegramBot.send_reply", to_string(chat_id), "", attachment, metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")

@effects("io", "fail")
async def handle_user_dm_attachment(chat_id: int, user_id: int, text: str, attachment: Attachment, metadata: str) -> TgTextTurn:
    """Preserve sandboxed attachment takeover arguments and propagate immediate delivery outcome. NoImmediateReply is not callback success."""
    ctx: str = concat("telegram:user:", to_string(user_id))
    reply: str = await llm_takeover(ctx, text, "TelegramBot.send_reply", to_string(chat_id), sandbox_perm_model, attachment, metadata)
    if reply != "":
        delivery: TgSendAttempt = await TelegramBot.send_message(chat_id, reply)
        return TgDeliveryAttempt(delivery)
    return TgNoImmediateReply("no immediate reply")
