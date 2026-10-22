from __future__ import annotations

import re

_MD_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL = re.compile(r"https?://\S+")
_BOLD = re.compile(r"\*{1,3}([^*]+)\*{1,3}")
_QUOTE = re.compile(r"^>\s?", re.MULTILINE)
_WHITESPACE = re.compile(r"\s+")
_S_TAG = re.compile(r"(?:^|\s)/s\b", re.IGNORECASE)


def sanitize_spoken(text: str) -> str:
    """Strip Reddit markup for TTS. Keep meaning, drop /s and URLs. No length cap."""
    cleaned = _MD_LINK.sub(r"\1", text)
    cleaned = _URL.sub("link", cleaned)
    cleaned = _S_TAG.sub("", cleaned)
    cleaned = _BOLD.sub(r"\1", cleaned)
    cleaned = _QUOTE.sub("", cleaned)
    cleaned = cleaned.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    cleaned = cleaned.replace("&#x200B;", "").replace("&nbsp;", " ")
    cleaned = _WHITESPACE.sub(" ", cleaned).strip()
    return cleaned


def truncate_body(text: str) -> str:
    """Normalize comment body text. Kept for call-site compatibility; no length cap."""
    return text.strip()


def looks_unusable(author: str, body: str) -> bool:
    if not body or not body.strip():
        return True
    lowered = body.strip().lower()
    if lowered in {"[deleted]", "[removed]"}:
        return True
    if author in {"[deleted]", "[removed]", "AutoModerator"}:
        return True
    if author.lower().endswith("bot") and author.lower() != "bot":
        if lowered.startswith("i am a bot") or "this action was performed automatically" in lowered:
            return True
    return False
