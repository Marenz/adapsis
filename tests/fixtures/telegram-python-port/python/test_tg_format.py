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

# Original stored cases carried verbatim (frozen block/case index in the name).
def test_stored_escape_html_1() -> None:
    assert escape_html(text="a < b & c > d") == "a &lt; b &amp; c &gt; d"

def test_stored_escape_html_2() -> None:
    assert escape_html(text="<script>alert(1)</script>") == "&lt;script&gt;alert(1)&lt;/script&gt;"

def test_stored_escape_html_3() -> None:
    assert escape_html(text="plain text") == "plain text"

def test_stored_apply_inline_1() -> None:
    assert apply_inline(s="**b**") == "<b>b</b>"

def test_stored_apply_inline_2() -> None:
    assert apply_inline(s="__b__") == "<b>b</b>"

def test_stored_apply_inline_3() -> None:
    assert apply_inline(s="*i*") == "<i>i</i>"

def test_stored_apply_inline_4() -> None:
    assert apply_inline(s="_i_") == "<i>i</i>"

def test_stored_apply_inline_5() -> None:
    assert apply_inline(s="~~s~~") == "<s>s</s>"

def test_stored_apply_inline_6() -> None:
    assert apply_inline(s="||sp||") == "<tg-spoiler>sp</tg-spoiler>"

def test_stored_apply_inline_7() -> None:
    assert apply_inline(s="[lbl](http://x.y)") == "<a href=\"http://x.y\">lbl</a>"

def test_stored_apply_inline_8() -> None:
    assert apply_inline(s="hi _x_ y") == "hi <i>x</i> y"

def test_stored_code_segments_1() -> None:
    assert code_segments(parts=["a"], i=0, acc="") == "a"

def test_stored_inline_code_4() -> None:
    assert inline_code(s="no code here") == "no code here"

def test_stored_convert_line_1() -> None:
    assert convert_line(line="# H") == "<b>H</b>"

def test_stored_convert_line_2() -> None:
    assert convert_line(line="## H") == "<b>H</b>"

def test_stored_convert_line_3() -> None:
    assert convert_line(line="### H") == "<b>H</b>"

def test_stored_convert_line_4() -> None:
    assert convert_line(line="&gt; q") == "<blockquote>q</blockquote>"

def test_stored_convert_line_5() -> None:
    assert convert_line(line="&gt;") == "<blockquote></blockquote>"

def test_stored_convert_line_6() -> None:
    assert convert_line(line="- i") == "• i"

def test_stored_convert_line_7() -> None:
    assert convert_line(line="* i") == "• i"

def test_stored_convert_line_8() -> None:
    assert convert_line(line="plain") == "plain"

def test_stored_convert_lines_1() -> None:
    assert convert_lines(lines=["a", "b"], i=0, in_pre=False, buf="", acc=[]) == ["a", "b"]

def test_stored_markdown_to_html_1() -> None:
    assert markdown_to_html(text="**bold**") == "<b>bold</b>"

def test_stored_markdown_to_html_2() -> None:
    assert markdown_to_html(text="__bold__") == "<b>bold</b>"

def test_stored_markdown_to_html_3() -> None:
    assert markdown_to_html(text="*ital*") == "<i>ital</i>"

def test_stored_markdown_to_html_4() -> None:
    assert markdown_to_html(text="_ital_") == "<i>ital</i>"

def test_stored_markdown_to_html_5() -> None:
    assert markdown_to_html(text="~~gone~~") == "<s>gone</s>"

def test_stored_markdown_to_html_8() -> None:
    assert markdown_to_html(text="# Title") == "<b>Title</b>"

def test_stored_markdown_to_html_9() -> None:
    assert markdown_to_html(text="## Title") == "<b>Title</b>"

def test_stored_markdown_to_html_10() -> None:
    assert markdown_to_html(text="### Title") == "<b>Title</b>"

def test_stored_markdown_to_html_11() -> None:
    assert markdown_to_html(text="- item") == "• item"

def test_stored_markdown_to_html_12() -> None:
    assert markdown_to_html(text="* item") == "• item"

def test_stored_markdown_to_html_13() -> None:
    assert markdown_to_html(text="> quoted") == "<blockquote>quoted</blockquote>"

def test_stored_markdown_to_html_14() -> None:
    assert markdown_to_html(text="a < b & c") == "a &lt; b &amp; c"

def test_stored_markdown_to_html_15() -> None:
    assert markdown_to_html(text="<script>x</script>") == "&lt;script&gt;x&lt;/script&gt;"

def test_stored_markdown_to_html_16() -> None:
    assert markdown_to_html(text="[lbl](http://x.y)") == "<a href=\"http://x.y\">lbl</a>"

def test_stored_split_parts_1() -> None:
    assert split_parts(lines=["a", "b"], i=0, part="", depth=0, limit=1, acc=[]) == ["a", "b"]

def test_stored_split_parts_2() -> None:
    assert split_parts(lines=["solo"], i=0, part="", depth=0, limit=2, acc=[]) == ["solo"]

def test_stored_split_for_telegram_2() -> None:
    assert split_for_telegram(html="solo", limit=2) == ["solo"]
