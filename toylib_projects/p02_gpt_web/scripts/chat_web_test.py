"""Tests for scripts/chat_web.py's text handling, with a scripted generator."""

import dataclasses

from toylib_projects.p02_gpt_web.scripts import chat_web


class ByteTokenizer:
    """Token i is byte i, so a multi-byte character spans several tokens."""

    def encode(self, text):
        return list(text.encode())

    def decode(self, ids):
        return bytes(ids).decode(errors="replace")


@dataclasses.dataclass
class Scripted:
    """Streams a fixed reply, as `generate.Generator.stream` would."""

    reply: str
    seq_len: int = 2048

    def stream(self, prompt, stop_tokens=(), **_):
        for t in self.reply.encode():
            if t in stop_tokens:
                return
            yield t


def _text(reply, stop=chat_web.STOP):
    return list(chat_web.stream_text(Scripted(reply), ByteTokenizer(), [0], stop))


class TestStreamText:
    def test_stops_before_the_users_turn(self):
        pieces = _text("Paris is the capital.\nUser: and Spain?")
        assert "".join(pieces) == "Paris is the capital."

    def test_holds_back_what_might_start_the_stop_string(self):
        """No piece ever shows part of "\\nUser:" that later turns out to be the stop."""
        pieces = _text("Yes.\nUser: more")
        assert all("\n" not in p for p in pieces)

    def test_a_newline_that_is_not_the_stop_comes_through(self):
        assert "".join(_text("one\ntwo\nUsers like it")) == "one\ntwo\nUsers like it"

    def test_never_splits_a_character(self):
        pieces = _text("café ☕ ok", stop=None)
        assert "".join(pieces) == "café ☕ ok"
        assert all("�" not in p for p in pieces)

    def test_held_back(self):
        assert chat_web.held_back("hello\nUs", "\nUser:") == 3
        assert chat_web.held_back("hello", "\nUser:") == 0


class TestTranscript:
    def test_ends_where_the_reply_begins(self):
        text = chat_web.transcript([{"role": "user", "content": " Hi "}])
        assert text.startswith(chat_web.PREAMBLE) and text.endswith("User: Hi\nAssistant:")

    def test_drops_the_oldest_turns_to_fit(self):
        messages = [
            {"role": "user", "content": "x" * 400},
            {"role": "assistant", "content": "y" * 400},
            {"role": "user", "content": "last"},
        ]
        ids = chat_web.fit(ByteTokenizer(), messages, budget=300)
        text = ByteTokenizer().decode(ids[1:])
        assert len(ids) <= 300 and "last" in text and "x" not in text
