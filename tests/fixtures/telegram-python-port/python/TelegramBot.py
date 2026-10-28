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
