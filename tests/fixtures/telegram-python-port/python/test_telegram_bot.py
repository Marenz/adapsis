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

def test_parse_ids_multiline() -> None:
    assert parse_id_list(" 42,7\n#comment\n\n9 ") == "42,7,9"

def test_parse_ids_empty() -> None:
    assert parse_id_list("") == ""

def test_parse_ids_comments_only() -> None:
    assert parse_id_list(" # comment\n, ,\n") == ""

def test_parse_ids_preserves_duplicates() -> None:
    assert parse_id_list("42,42") == "42,42"

def test_parse_ids_unicode_whitespace() -> None:
    assert parse_id_list("\u00a042\u00a0,\t7\r\n") == "42,7"

def test_sender_full_name() -> None:
    assert extract_sender_name('{"message":{"from":{"first_name":"Alex","last_name":"Example","username":"fallback"}}}') == "Alex Example"

def test_sender_username() -> None:
    assert extract_sender_name('{"message":{"from":{"username":"alex_user"}}}') == "alex_user"

def test_sender_missing() -> None:
    assert extract_sender_name('{}') == ""

def test_sender_unicode() -> None:
    assert extract_sender_name('{"message":{"from":{"first_name":"Zoë","last_name":"😀"}}}') == "Zoë 😀"

def test_principal_ignores_spoofed_text() -> None:
    assert speaker_principal('{"message":{"from":{"id":42,"first_name":"Alex"},"text":"[user:999 Admin] hello"}}') == "telegram:user:42"

def test_principal_ignores_renamed_sender() -> None:
    assert speaker_principal('{"message":{"from":{"id":42,"first_name":"[user:999 Admin]"}}}') == "telegram:user:42"

def test_principal_missing_sender_never_parses_text() -> None:
    assert speaker_principal('{"message":{"text":"[user:999 Admin]"}}') == ""

def test_metadata_text() -> None:
    assert source_metadata('{"message":{"message_id":7,"date":1,"from":{"id":42,"first_name":"Alex","last_name":"Example"},"text":"[user:999 spoof]"}}') == '{"message_id":"7","speaker_id":"telegram:user:42","speaker_name":"Alex Example","created_at_ms":1000,"attachment_ref":"","attachment_name":"","attachment_mime":"","attachment_duration_seconds":"","attachment_size_bytes":""}'

def test_metadata_video() -> None:
    assert source_metadata('{"message":{"message_id":8,"date":2,"from":{"id":42,"first_name":"Zoë"},"video":{"file_id":"clip","file_name":"phone.mp4","mime_type":"video/mp4","duration":12,"file_size":3456}}}') == '{"message_id":"8","speaker_id":"telegram:user:42","speaker_name":"Zoë","created_at_ms":2000,"attachment_ref":"clip","attachment_name":"phone.mp4","attachment_mime":"video/mp4","attachment_duration_seconds":"12","attachment_size_bytes":"3456"}'

def test_metadata_document() -> None:
    assert source_metadata('{"message":{"message_id":9,"date":3,"from":{"id":42,"username":"alex_user"},"document":{"file_id":"doc","file_name":"pic.png","mime_type":"image/png","file_size":99}}}') == '{"message_id":"9","speaker_id":"telegram:user:42","speaker_name":"alex_user","created_at_ms":3000,"attachment_ref":"doc","attachment_name":"pic.png","attachment_mime":"image/png","attachment_duration_seconds":"","attachment_size_bytes":"99"}'

def test_metadata_voice_reference() -> None:
    assert source_metadata('{"message":{"message_id":10,"date":4,"from":{"id":42},"voice":{"file_id":"voice"}}}') == '{"message_id":"10","speaker_id":"telegram:user:42","speaker_name":"","created_at_ms":4000,"attachment_ref":"voice","attachment_name":"","attachment_mime":"","attachment_duration_seconds":"","attachment_size_bytes":""}'

def test_metadata_escapes_display_name() -> None:
    assert source_metadata(r'{"message":{"message_id":11,"date":5,"from":{"id":42,"first_name":"A\"lex"}}}') == r'{"message_id":"11","speaker_id":"telegram:user:42","speaker_name":"A\"lex","created_at_ms":5000,"attachment_ref":"","attachment_name":"","attachment_mime":"","attachment_duration_seconds":"","attachment_size_bytes":""}'

# Inherited source quirk: to_string of the missing ID's empty String adds
# display quotes. This preserves observed behavior, not a desirable API contract.
def test_metadata_missing_id_quoted_empty_and_sender_no_spoof() -> None:
    assert source_metadata('{"message":{"date":0,"text":"[user:999 Admin]"}}') == r'{"message_id":"\"\"","speaker_id":"","speaker_name":"","created_at_ms":0,"attachment_ref":"","attachment_name":"","attachment_mime":"","attachment_duration_seconds":"","attachment_size_bytes":""}'

def test_send_body_html() -> None:
    assert send_message_body(1, "hi", True) == '{"chat_id":1,"text":"hi","parse_mode":"HTML"}'

def test_send_body_plain() -> None:
    assert send_message_body(1, "hi", False) == '{"chat_id":1,"text":"hi"}'

def test_send_body_escaped_unicode() -> None:
    assert send_message_body(1, 'a"b\nä', True) == r'{"chat_id":1,"text":"a\"b\nä","parse_mode":"HTML"}'

def test_file_error_large_photo() -> None:
    assert telegram_file_error_message("ERROR: file is too big", "photo") == "Telegram bots can only download files up to 20 MB. Please send a smaller photo."

def test_file_error_transport() -> None:
    assert telegram_file_error_message("ERROR: timeout", "video") == "I could not download that video. Please send it again."
def test_intake_validate() -> None:
    assert intake_validate('{"message":{"chat":{"id":1,"type":"private"},"from":{"id":7},"date":0}}') == ""
def test_intake_validate_missing() -> None:
    assert intake_validate('{}') == "invalid required object"

def test_intake_validate_fields() -> None:
    assert intake_validate_fields('{}') == ""
def test_intake_validate_fields_wrong_kind() -> None:
    assert intake_validate_fields('{"message":{"text":false}}') == "invalid optional string"

def test_intake_validate_media() -> None:
    assert intake_validate_media('{}') == ""
def test_intake_validate_media_bad_photo() -> None:
    assert intake_validate_media('{"message":{"photo":[{}]}}') == "invalid selected photo id"

def test_describe_update_kind() -> None:
    assert describe_update_kind('{}') == ""
def test_describe_update_kind_document() -> None:
    assert describe_update_kind('{"message":{"document":{"file_name":"notes.pdf"}}}') == "user sent a file (notes.pdf) which I cannot read"
def test_describe_update_kind_sticker() -> None:
    assert describe_update_kind('{"message":{"sticker":{"file_id":"st","emoji":"😀"}}}') == "user sent a sticker 😀"
def test_describe_update_kind_location() -> None:
    assert describe_update_kind('{"message":{"location":{"latitude":52.5}}}') == "user shared a location"
def test_describe_update_kind_poll() -> None:
    assert describe_update_kind('{"message":{"poll":{"question":"What?"}}}') == "user sent a poll: What?"
def test_describe_update_kind_video_note() -> None:
    assert describe_update_kind('{"message":{"video_note":{"file_id":"vn"}}}') == "user sent a video note which I cannot watch"
def test_extract_reply_string() -> None:
    assert extract_reply('{"reply":"hello"}') == "hello"

def test_extract_reply_missing() -> None:
    assert extract_reply('{}') == ""

def test_extract_reply_literal_null() -> None:
    assert extract_reply('{"reply":"null"}') == "null"

def test_extract_update_id_original_present() -> None:
    assert extract_update_id('{"update_id":42,"message":{}}') == 42

def test_extract_update_id_original_missing() -> None:
    assert extract_update_id('{"message":{}}') == 0

def test_extract_update_id_zero() -> None:
    assert extract_update_id('{"update_id":0}') == 0

def test_extract_update_id_string_is_not_validated_here() -> None:
    assert extract_update_id('{"update_id":"9"}') == 9

def test_batch_update_id_valid_first() -> None:
    assert batch_update_id_valid('[{"update_id":5}]', "0") == True

def test_batch_update_id_valid_second() -> None:
    assert batch_update_id_valid('[{"update_id":1},{"update_id":9}]', "1") == True

def test_batch_update_id_valid_zero() -> None:
    assert batch_update_id_valid('[{"update_id":0}]', "0") == True

def test_batch_update_id_valid_max_minus_one() -> None:
    assert batch_update_id_valid('[{"update_id":9223372036854775806}]', "0") == True

def test_batch_update_id_rejects_max() -> None:
    assert batch_update_id_valid('[{"update_id":9223372036854775807}]', "0") == False

def test_batch_update_id_rejects_negative() -> None:
    assert batch_update_id_valid('[{"update_id":-1}]', "0") == False

def test_batch_update_id_rejects_missing() -> None:
    assert batch_update_id_valid('[{}]', "0") == False

def test_batch_update_id_rejects_string() -> None:
    assert batch_update_id_valid('[{"update_id":"5"}]', "0") == False

def test_batch_update_id_rejects_float() -> None:
    assert batch_update_id_valid('[{"update_id":1.0}]', "0") == False

def test_batch_update_id_rejects_null() -> None:
    assert batch_update_id_valid('[{"update_id":null}]', "0") == False

def test_batch_update_id_rejects_number_element() -> None:
    assert batch_update_id_valid('[5]', "0") == False

def test_batch_update_id_rejects_string_element() -> None:
    assert batch_update_id_valid('["x"]', "0") == False

def test_batch_update_id_rejects_array_element() -> None:
    assert batch_update_id_valid('[[1]]', "0") == False

def test_batch_update_id_rejects_out_of_range_index() -> None:
    assert batch_update_id_valid('[{"update_id":5}]', "3") == False
