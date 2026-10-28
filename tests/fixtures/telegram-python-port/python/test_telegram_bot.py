def test_chat_ids() -> None:
    assert extract_chat_id('{"message":{"chat":{"id":123}}}') == 123

def test_chat_id_missing() -> None:
    assert extract_chat_id('{"message":{}}') == 0

def test_user_ids() -> None:
    assert extract_user_id('{"message":{"from":{"id":42,"first_name":"First"},"text":"[user:999 spoof]"}}') == 42

def test_user_rename() -> None:
    assert extract_user_id('{"message":{"from":{"id":42,"first_name":"Renamed"}}}') == 42

def test_user_missing() -> None:
    assert extract_user_id('{}') == 0

def test_chat_types() -> None:
    assert extract_chat_type('{"message":{"chat":{"type":"private"}}}') == "private"

def test_group_type() -> None:
    assert extract_chat_type('{"message":{"chat":{"type":"group"}}}') == "group"

def test_supergroup_type() -> None:
    assert extract_chat_type('{"message":{"chat":{"type":"supergroup"}}}') == "supergroup"

def test_text_and_caption() -> None:
    assert extract_text('{"message":{"text":"text","caption":"caption"}}') == "text"

def test_caption() -> None:
    assert extract_text('{"message":{"caption":"caption"}}') == "caption"

def test_text_missing() -> None:
    assert extract_text('{}') == ""

def test_images() -> None:
    assert extract_image_file_id('{"message":{"photo":[{"file_id":"small"},{"file_id":"large"}]}}') == "large"

def test_image_document() -> None:
    assert extract_image_file_id('{"message":{"document":{"file_id":"image","mime_type":"image/png"}}}') == "image"

def test_nonimage_document() -> None:
    assert extract_image_file_id('{"message":{"document":{"file_id":"pdf","mime_type":"application/pdf"}}}') == ""

def test_videos() -> None:
    assert extract_video_file_id('{"message":{"video":{"file_id":"clip"}}}') == "clip"

def test_video_document() -> None:
    assert extract_video_file_id('{"message":{"document":{"file_id":"movie","mime_type":"video/mp4"}}}') == "movie"

def test_video_missing() -> None:
    assert extract_video_file_id('{}') == ""

def test_sender() -> None:
    assert format_sender(42, "Alex", "hi") == "[user:42 Alex] hi"

def test_sender_unnamed() -> None:
    assert format_sender(42, "", "hi") == "[user:42] hi"

def test_admin_boundaries() -> None:
    assert is_admin_in(42, "142,42,7") == True

def test_admin_nonboundary() -> None:
    assert is_admin_in(42, "142,420") == False

def test_admin_empty() -> None:
    assert is_admin_in(42, "") == False

def test_admin_named_arguments() -> None:
    assert is_admin_in(ids="142,42,7", user_id=42) == True

def test_response() -> None:
    assert telegram_response_ok('{"ok":true}') == True

def test_response_failure() -> None:
    assert telegram_response_ok('{"ok":false}') == False

def test_file_response() -> None:
    assert telegram_file_path_response('{"result":{"file_path":"photo/a.jpg"}}') == "photo/a.jpg"

def test_file_response_error() -> None:
    assert telegram_file_path_response('{"description":"file is too big"}') == "ERROR: file is too big"

def test_file_response_missing() -> None:
    assert telegram_file_path_response('{}') == "ERROR: Telegram getFile returned no file path"
