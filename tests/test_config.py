"""Basic sanity tests."""
from backend.config.settings import get_settings
from backend.utils.helpers import clean_text, format_source
from backend.prompts.system import is_greeting_or_smalltalk


def test_settings_loads():
    s = get_settings()
    assert s.chunk_size > 0
    assert s.chunk_overlap >= 0


def test_clean_text():
    assert clean_text("  hello   world  ") == "hello world"


def test_format_source():
    meta = {"source": "doc.pdf", "page_number": 5, "content_type": "text"}
    result = format_source(meta)
    assert result["source"] == "doc.pdf"
    assert result["page"] == 5


def test_greeting_detection():
    assert is_greeting_or_smalltalk("Hello!") is True
    assert is_greeting_or_smalltalk("What is revenue?") is False