def test_escape() -> None:
    assert escape_html("a < b & c > d") == "a &lt; b &amp; c &gt; d"

def test_escape_injection() -> None:
    assert escape_html("<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

def test_bold() -> None:
    assert apply_inline("**bold**") == "<b>bold</b>"

def test_link() -> None:
    assert apply_inline("[label](https://example.invalid)") == '<a href="https://example.invalid">label</a>'

def test_spoiler() -> None:
    assert apply_inline("||secret||") == "<tg-spoiler>secret</tg-spoiler>"

def test_open_tag() -> None:
    assert tag("code", False) == "<code>"

def test_close_tag() -> None:
    assert tag("code", True) == "</code>"
