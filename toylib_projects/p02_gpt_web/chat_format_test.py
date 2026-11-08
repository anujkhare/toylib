"""Tests for chat_format.py, on a byte-level tokenizer (no merges, 257 ids)."""

import numpy as np
import pytest

from toylib_projects.p02_gpt_web import chat_format
from toylib_projects.p02_gpt_web.tokenizer import bpe

TOK = chat_format.ChatTokenizer(bpe.Tokenizer([]))


def test_chat_tokens_follow_the_base_vocabulary():
    assert len(TOK) == 257 + 8
    assert TOK.user_start == 257 and TOK.output_end == 264
    assert TOK.decode([TOK.assistant_start, *TOK.encode("hi"), TOK.assistant_end]) == (
        "<|assistant_start|>hi<|assistant_end|>"
    )
    with pytest.raises(AttributeError):
        TOK.not_a_token


def test_bytes_per_token_leaves_out_special_tokens():
    table = TOK.bytes_per_token()
    assert len(table) == len(TOK) and (table[257:] == -1).all() and table[ord("a") + 1] == 1


def _trained_text(ids, mask):
    return TOK.decode([i for i, m in zip(ids, mask) if m])


def test_only_the_assistant_is_trained():
    conversation = {"messages": [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Ciao"},
    ]}
    ids, mask = chat_format.render_conversation(TOK, conversation)
    assert TOK.decode(ids) == (
        "<|endoftext|><|user_start|>Hi<|user_end|><|assistant_start|>Hello<|assistant_end|>"
        "<|user_start|>Bye<|user_end|><|assistant_start|>Ciao<|assistant_end|>"
    )
    assert _trained_text(ids, mask) == "Hello<|assistant_end|>Ciao<|assistant_end|>"


def test_calculator_results_are_not_trained():
    parts = [
        {"type": "text", "text": "2 x 3 = "},
        {"type": "python", "text": "2*3"},
        {"type": "python_output", "text": "6"},
        {"type": "text", "text": "6"},
    ]
    ids, mask = chat_format.render_conversation(
        TOK, {"messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": parts}]}
    )
    assert "<|output_start|>6<|output_end|>" in TOK.decode(ids)
    assert _trained_text(ids, mask) == "2 x 3 = <|python_start|>2*3<|python_end|>6<|assistant_end|>"


def test_a_system_message_joins_the_first_user_message():
    conversation = {"messages": [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Yo"},
    ]}
    ids, _ = chat_format.render_conversation(TOK, conversation)
    assert "<|user_start|>Be brief.\n\nHi<|user_end|>" in TOK.decode(ids)
    assert conversation["messages"][0]["role"] == "system"  # Not modified.


def test_conversations_are_capped():
    long = {"messages": [{"role": "user", "content": "x" * 50}, {"role": "assistant", "content": "y"}]}
    ids, mask = chat_format.render_conversation(TOK, long, max_tokens=20)
    assert len(ids) == len(mask) == 20


def test_render_for_completion_ends_at_the_assistants_turn():
    ids = chat_format.render_for_completion(TOK, [{"role": "user", "content": "Hi"}])
    assert TOK.decode(ids) == "<|endoftext|><|user_start|>Hi<|user_end|><|assistant_start|>"


@pytest.mark.parametrize("expr,result", [
    ("12/60", 0.2), ("1,000 + 2", 1002), ("2**10", None), ("'strawberry'.count('r')", 3),
    ("__import__('os')", None), ("abc", None), ("1/0", None),
])
def test_calculator(expr, result):
    assert chat_format.calculator(expr) == result


def test_tool_tokens_answer_a_closed_call_once():
    call = [TOK.python_start, *TOK.encode("6*7"), TOK.python_end]
    forced = chat_format.tool_tokens(TOK, TOK.encode("so ") + call)
    assert TOK.decode(forced) == "<|output_start|>42<|output_end|>"
    assert chat_format.tool_tokens(TOK, TOK.encode("so ") + call[:-1]) == []  # Still open.
    assert chat_format.tool_tokens(TOK, [TOK.python_end]) == []  # Never opened.
    bad = [TOK.python_start, *TOK.encode("6**7"), TOK.python_end]
    assert chat_format.tool_tokens(TOK, bad) == []


def test_tokens_round_trip_through_numpy():
    ids, _ = chat_format.render_conversation(
        TOK, {"messages": [{"role": "user", "content": "é"}, {"role": "assistant", "content": "ok"}]}
    )
    assert TOK.decode(np.array(ids, np.uint16)) == TOK.decode(ids)
