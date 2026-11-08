"""Tests for sft.py: schedules, and fine-tuning a tiny pretrained run end to end."""

import dataclasses
import json

import jax
import numpy as np
import pytest

from toylib.harness import inference
from toylib.viz import curves
from toylib_projects.p02_gpt_web import chat_format
from toylib_projects.p02_gpt_web import sft
from toylib_projects.p02_gpt_web import sft_data
from toylib_projects.p02_gpt_web import tokenize_webtext
from toylib_projects.p02_gpt_web import train
from toylib_projects.p02_gpt_web.scripts import chat_eval
from toylib_projects.p02_gpt_web.tokenizer import bpe


def test_lr_multiplier_holds_then_falls_to_zero():
    schedule = sft.lr_multiplier(max_steps=100, offset=1000)
    assert float(schedule(1000)) == pytest.approx(0.8)
    assert float(schedule(1050)) == pytest.approx(0.8)
    assert float(schedule(1075)) == pytest.approx(0.4)
    assert float(schedule(1100)) == pytest.approx(0.0)


def test_muon_momentum_ramps_over_300_steps():
    schedule = sft.muon_momentum(offset=10)
    assert [round(float(schedule(10 + s)), 4) for s in (0, 150, 300, 900)] == [0.85, 0.9, 0.95, 0.95]


def test_d20_sft_adds_the_chat_tokens_to_d20():
    run = sft.model_run(sft.RUNS["d20_sft"])
    assert run.vocab_size == 32_768 + 8
    assert dataclasses.replace(run, vocab_size=32_768) == train.RUNS["d20_nanochat"]


BASE = dataclasses.replace(
    train.Run(
        num_layers=2, num_heads=2, qkv_dim=32, batch_size=4, max_steps=3, seq_len=64,
        vocab_size=257, warmup_steps=1, num_microbatches=2, eval_interval_steps=100,
        num_eval_steps=1, save_interval_steps=100,
    ),
    **train.NANOCHAT_OPTIONS,
)
TOK = chat_format.ChatTokenizer(bpe.Tokenizer([]))


@pytest.fixture
def base_checkpoints(tmp_path):
    """A pretrained tiny run's checkpoints, with its optimizer's count at max_steps."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    bpe.Tokenizer([]).save(tokenize_webtext.tokenizer_path(257, data_dir))
    tokens = np.random.default_rng(0).integers(2, 64, 4_000, dtype=np.uint16)
    for split in ("train", "val"):
        tokens.tofile(tokenize_webtext.bin_path(split, 257, data_dir))
    exp = train.create_run_experiment(BASE, run_id="base", data_dir=data_dir, output_dir=tmp_path)
    exp.outer_loop()
    model = jax.tree.map(np.asarray, exp.model)
    exp.cleanup()
    return tmp_path / "base" / "checkpoints", model


@pytest.fixture
def sft_dir(tmp_path):
    """Packed rows of a few short conversations, for both splits."""
    rng = np.random.default_rng(0)
    conversations = []
    for i in range(40):
        q, a = "".join(rng.choice(list("abc"), 3)), "".join(rng.choice(list("xyz"), 4))
        conversations.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})
    rendered = [tuple(map(np.array, chat_format.render_conversation(TOK, c))) for c in conversations]
    rows, masks = sft_data.pack(iter(rendered), BASE.seq_len + 1)
    out = tmp_path / "sft"
    out.mkdir()
    for split in ("train", "val"):
        tokens_path, mask_path = sft_data.paths(split, len(TOK), out)
        np.save(tokens_path, rows.astype(np.uint16))
        np.save(mask_path, masks)
    return out


def _fake_tasks(name, data_dir=None):
    if name in chat_eval.CATEGORICAL:
        return [chat_eval.Problem("Q?", "A", letters=("A", "B")), chat_eval.Problem("R?", "B", letters=("A", "B"))]
    if name == "GSM8K":
        return [chat_eval.Problem("1+1?", "2")]
    return [chat_eval.Problem("def f():\n", "", entry_point="f", test="def check(f):\n    assert f() == 1\n")]


def test_fine_tunes_a_pretrained_run(base_checkpoints, sft_dir, tmp_path, monkeypatch):
    checkpoint_dir, base_model = base_checkpoints
    run = sft.SFTRun(
        base=BASE, base_checkpoint_dir=str(checkpoint_dir), max_steps=4, eval_interval_steps=2,
        eval_tokens=2 * BASE.seq_len, chatcore_interval_steps=2, save_interval_steps=100,
    )
    exp = sft.create_experiment(run, run_id="sft", data_dir=sft_dir, output_dir=tmp_path, tokenizer=TOK)
    exp.chat_prompts, exp.sampling_max_tokens, exp.chatcore_max_new_tokens = ["Hi"], 8, 8
    monkeypatch.setattr(chat_eval, "load_task", _fake_tasks)

    sft.load_base(exp, run)
    # The base rows load into the bigger tables; the chat tokens' rows are new.
    embedding = np.asarray(exp.model.embedding_layer.weights)
    np.testing.assert_array_equal(embedding[:257], base_model.embedding_layer.weights)
    assert embedding.shape[0] == 265 and np.abs(embedding[257:]).sum() > 0
    output = np.asarray(exp.model.output_layer.weights)
    np.testing.assert_array_equal(output[:, :257], base_model.output_layer.weights)
    assert sft.optimizer_counts(exp.opt_state) == {BASE.max_steps}

    exp.outer_loop()
    assert sft.optimizer_counts(exp.opt_state) == {BASE.max_steps + 4}
    scores = sft.finish(exp, tmp_path / "sft")
    exp.cleanup()

    assert set(scores["accuracy"]) == set(chat_eval.TASKS) and "ChatCORE" in scores
    assert json.loads((tmp_path / "sft" / "chatcore.json").read_text())["ChatCORE"] == scores["ChatCORE"]
    metrics = curves.read_log(tmp_path / "sft" / "logs_sft.txt")
    assert all(np.isfinite(metrics["train/loss"][1]))
    assert "chatcore/ChatCORE" in metrics and "val/bits_per_byte" in metrics
    assert inference.checkpoint_steps(tmp_path / "sft" / "checkpoints")[-1] == 3


def test_resumes_from_its_own_checkpoint(base_checkpoints, sft_dir, tmp_path):
    checkpoint_dir, _ = base_checkpoints
    run = sft.SFTRun(
        base=BASE, base_checkpoint_dir=str(checkpoint_dir), max_steps=4, eval_interval_steps=100,
        chatcore_interval_steps=0, save_interval_steps=2,
    )
    exp = sft.create_experiment(run, run_id="sft", data_dir=sft_dir, output_dir=tmp_path, tokenizer=TOK)
    assert not exp.resume()
    sft.load_base(exp, run)
    exp.training_config.max_steps = 3  # Stop early, as if the runtime ended.
    exp.outer_loop()
    exp.cleanup()

    again = sft.create_experiment(run, run_id="sft", data_dir=sft_dir, output_dir=tmp_path, tokenizer=TOK)
    assert again.resume() and again.step == 3
    assert sft.optimizer_counts(again.opt_state) == {BASE.max_steps + 3}
    again.outer_loop()
    again.cleanup()
    assert sft.optimizer_counts(again.opt_state) == {BASE.max_steps + 4}
