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

def test_code_wrap() -> None:
    assert code_wrap("é &lt;") == "<code>é &lt;</code>"

def test_code_segments() -> None:
    assert code_segments(["a", "b", "c"], 0, "") == "a<code>b</code>c"

def test_inline_code_protects_markdown() -> None:
    assert inline_code("`**x**`") == "<code>**x**</code>"

def test_heading() -> None:
    assert convert_line("# Title") == "<b>Title</b>"

def test_quote() -> None:
    assert convert_line("&gt; quote") == "<blockquote>quote</blockquote>"

def test_unicode_bullet() -> None:
    assert convert_line("- ä") == "• ä"

def test_convert_plain_lines() -> None:
    assert convert_lines(["a", "b"], 0, False, "", []) == ["a", "b"]

def test_convert_fenced_lines() -> None:
    assert convert_lines(["```", "x", "```"], 0, False, "", []) == ["<pre>x</pre>"]

def test_convert_unclosed_fence() -> None:
    assert convert_lines(["a", "```", "x"], 0, False, "", []) == ["a", "<pre>x</pre>"]

def test_markdown_mixed() -> None:
    assert markdown_to_html("**bold** _ital_ `code`") == "<b>bold</b> <i>ital</i> <code>code</code>"

def test_markdown_escape_unicode() -> None:
    assert markdown_to_html("<script>x</script> & ä") == "&lt;script&gt;x&lt;/script&gt; &amp; ä"

def test_markdown_fence_no_inline_formatting() -> None:
    assert markdown_to_html("```\n**not bold**\n<raw>\n```") == "<pre>**not bold**\n&lt;raw&gt;</pre>"

def test_markdown_line_formats() -> None:
    assert markdown_to_html("> quoted\n- item\n# Head") == "<blockquote>quoted</blockquote>\n• item\n<b>Head</b>"

def test_markdown_unclosed_fence() -> None:
    assert markdown_to_html("```\nlast") == "<pre>last</pre>"

def test_markdown_empty() -> None:
    assert markdown_to_html("") == ""

def test_pre_wrap() -> None:
    assert pre_wrap("a\nb") == "<pre>a\nb</pre>"

def test_split_parts() -> None:
    assert split_parts(["a", "b"], 0, "", 0, 1, []) == ["a", "b"]

def test_split_parts_oversized_line_is_retained() -> None:
    assert split_parts(["solo"], 0, "", 0, 2, []) == ["solo"]

def test_split_exact_boundary() -> None:
    assert split_for_telegram("aaa\nbbb\nccc", 7) == ["aaa\nbbb", "ccc"]

def test_split_below_boundary() -> None:
    assert split_for_telegram("aaa\nbbb\nccc", 6) == ["aaa", "bbb", "ccc"]

def test_split_large_limit() -> None:
    assert split_for_telegram("aaa\nbbb\nccc", 100) == ["aaa\nbbb\nccc"]

def test_split_preserves_oversized_pre_block() -> None:
    assert split_for_telegram("<pre>a\nb</pre>\nxx", 5) == ["<pre>a\nb</pre>", "xx"]

def test_split_unicode_exact_byte_boundary() -> None:
    assert split_for_telegram("ä\nö", 5) == ["ä\nö"]

def test_split_unicode_below_byte_boundary() -> None:
    assert split_for_telegram("ä\nö", 3) == ["ä", "ö"]

def test_split_empty() -> None:
    assert split_for_telegram("", 1) == []

def test_parts_preview() -> None:
    assert parts_preview("aaa\nbbb\nccc", 7) == "aaa\nbbb || ccc"

def test_sample_lines() -> None:
    assert sample_lines() == "aaa\nbbb\nccc"

def test_sample_join7() -> None:
    assert sample_join7() == "aaa\nbbb || ccc"

def test_sample_join2() -> None:
    assert sample_join2() == "aaa || bbb || ccc"

def test_sample_pre() -> None:
    assert sample_pre() == "<pre>a\nb</pre>\nxx"

def test_sample_pre_join() -> None:
    assert sample_pre_join() == "<pre>a\nb</pre> || xx"
