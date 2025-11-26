"""
HTML Cleaner for ChatGPT Export (Phase 1)
-----------------------------------------
Strips tags, removes boilerplate, cleans whitespace.
"""

import re


def clean_html_text(text: str) -> str:
    """Normalize and sanitize text extracted from HTML."""
    if not text:
        return ""

    # Remove multiple spaces/newlines
    cleaned = " ".join(text.split())

    # Remove common ChatGPT boilerplate
    boilerplate = [
        "OpenAI",
        "ChatGPT",
        "export",
        "share",
        "copy",
        "regenerate",
        "like",
        "dislike",
    ]
    for b in boilerplate:
        if cleaned.lower() == b.lower():
            return ""

    # Remove weird unicode leftovers
    cleaned = re.sub(r"[\u200b\u200c\u200d]", "", cleaned)

    return cleaned.strip()
