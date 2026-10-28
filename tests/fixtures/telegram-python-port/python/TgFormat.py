def escape_html(text: str) -> str:
    """HTML-escape ampersand, less-than and greater-than so user text can never inject a tag. Runs first; every later step appends tags after this point, so generated tags are never re-escaped."""
    amp: str = regex_replace("&", "&amp;", text)
    lt: str = regex_replace("<", "&lt;", amp)
    return regex_replace(">", "&gt;", lt)

def apply_inline(s: str) -> str:
    """Apply inline Markdown to an already-escaped segment that is not inside a code span. Longest delimiters first - double asterisk before single, double underscore before single - so bold never degrades into italics. Emits only Telegram-allowed tags."""
    links: str = regex_replace(r"\[([^\]]+)\]\(([^)]+)\)", '<a href="$2">$1</a>', s)
    spoiler: str = regex_replace(r"\|\|(.+?)\|\|", "<tg-spoiler>$1</tg-spoiler>", links)
    strike: str = regex_replace("~~(.+?)~~", "<s>$1</s>", spoiler)
    bold1: str = regex_replace(r"\*\*(.+?)\*\*", "<b>$1</b>", strike)
    bold2: str = regex_replace("__(.+?)__", "<b>$1</b>", bold1)
    ital1: str = regex_replace(r"\*(.+?)\*", "<i>$1</i>", bold2)
    return regex_replace("_(.+?)_", "<i>$1</i>", ital1)

def tag(name: str, closing: bool) -> str:
    """Build an HTML tag from separate pieces. The source never contains a literal closing code or pre tag, because the runtime stops reading this code block at the first closing code fence it sees anywhere in the payload - including inside a string."""
    if closing:
        return concat("</", name, ">")
    return concat("<", name, ">")
