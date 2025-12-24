from __future__ import annotations

import html as _html
import re
from html.parser import HTMLParser


_WS_RE = re.compile(r"\s+")


class _HTMLToTextParser(HTMLParser):
    def __init__(self) -> None:
        # convert_charrefs=True decodes &nbsp; etc.
        super().__init__(convert_charrefs=True)
        self._parts: list[str] = []
        self._skip_depth = 0  # inside script/style/noscript

    def handle_starttag(self, tag: str, attrs):
        t = (tag or "").lower()
        if t in {"script", "style", "noscript"}:
            self._skip_depth += 1
            return
        # Add soft separators for common block-ish tags to avoid word-gluing.
        if t in {"p", "div", "br", "hr", "li", "tr", "td", "th", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_endtag(self, tag: str):
        t = (tag or "").lower()
        if t in {"script", "style", "noscript"}:
            if self._skip_depth > 0:
                self._skip_depth -= 1
            return
        if t in {"p", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"}:
            self._parts.append("\n")

    def handle_data(self, data: str):
        if self._skip_depth:
            return
        if data:
            self._parts.append(data)

    def get_text(self) -> str:
        return "".join(self._parts)


def html_to_text(html_str: str) -> str:
    """
    Best-effort HTML → plain text, safe for messy markup.
    Removes script/style/noscript, decodes entities, collapses whitespace.
    """
    if not html_str:
        return ""
    p = _HTMLToTextParser()
    try:
        p.feed(html_str)
        p.close()
        out = p.get_text()
    except Exception:
        # Fallback: strip tags very crudely (still decode entities).
        out = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\\1>", " ", html_str)
        out = re.sub(r"(?is)<[^>]+>", " ", out)
        out = _html.unescape(out)

    # Normalize whitespace, keep some newlines as separators
    out = out.replace("\r\n", "\n").replace("\r", "\n")
    out = "\n".join(_WS_RE.sub(" ", line).strip() for line in out.split("\n"))
    out = "\n".join(line for line in out.split("\n") if line)
    return out.strip()









