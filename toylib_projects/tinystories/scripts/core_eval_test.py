"""Tests for scripts/core_eval.py."""

import numpy as np
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import pytest
from pathlib import Path

from toylib_projects.tinystories.scripts.core_eval import (
    _build_fewshot_prefix,
    _encode_pair,
    _evaluate_task,
    _sum_continuation_logprob,
    load_model_from_checkpoint,
    run_sampling,
)
from toylib_projects.tinystories import decoder_only_model
from toylib_projects.tinystories.train import get_model_config


# ---------------------------------------------------------------------------
# Shared test fixtures

VOCAB_SIZE = 16
SEQ_LEN = 32


class SimpleTokenizer:
    """Character-level tokenizer: each char maps to ord(c) % VOCAB_SIZE."""

    def encode(self, text: str) -> list[int]:
        return [ord(c) % VOCAB_SIZE for c in text]

    def decode(self, ids: list[int], skip_special_tokens: bool = False) -> str:
        return "".join(chr(max(32, i)) for i in ids)


def mock_forward_favors_token_3(tokens):
    """Forward fn that assigns maximum logit to token 3, regardless of input.

    After log_softmax:
      log_probs[:, 3] ≈ 0      (near-certain)
      log_probs[:, other] ≈ -100
    """
    logits = jnp.zeros((SEQ_LEN, VOCAB_SIZE))
    return logits.at[:, 3].set(100.0)


# ---------------------------------------------------------------------------
# _encode_pair


class TestEncodePair:
    def test_normal_case(self):
        """Prefix + continuation fits; tokens, prefix_len and cont_ids are correct."""
        tok = SimpleTokenizer()
        prefix, cont = "ab", "cd"
        tokens, prefix_len, cont_ids = _encode_pair(tok, prefix, cont, SEQ_LEN)

        p_ids = tok.encode(prefix)
        c_ids = tok.encode(cont)

        assert tokens.shape == (SEQ_LEN,)
        assert prefix_len == len(p_ids)
        assert cont_ids == c_ids
        # Tokens are laid out as prefix then continuation at the front
        assert list(tokens[: len(p_ids) + len(c_ids)]) == p_ids + c_ids
        # Remainder is zero-padded
        assert all(t == 0 for t in tokens[len(p_ids) + len(c_ids) :])

    def test_truncates_prefix_when_too_long(self):
        """When prefix + cont > seq_len the prefix is truncated, cont is preserved."""
        tok = SimpleTokenizer()
        prefix = "a" * (SEQ_LEN + 5)  # 37 tokens — forces truncation
        cont = "bc"                    # 2 tokens — short enough to keep intact

        tokens, prefix_len, cont_ids = _encode_pair(tok, prefix, cont, SEQ_LEN)

        assert tokens.shape == (SEQ_LEN,)
        assert cont_ids == tok.encode(cont)
        # The whole seq_len array is used (no trailing zeros)
        assert prefix_len + len(cont_ids) == SEQ_LEN
        assert prefix_len >= 0

    def test_exact_fit(self):
        """When prefix + cont == seq_len, nothing is truncated."""
        tok = SimpleTokenizer()
        half = SEQ_LEN // 2
        prefix = "a" * half
        cont = "b" * half

        tokens, prefix_len, cont_ids = _encode_pair(tok, prefix, cont, SEQ_LEN)

        assert prefix_len == half
        assert len(cont_ids) == half
        assert tokens.shape == (SEQ_LEN,)
        # No zero padding expected
        assert all(t > 0 for t in tokens)


# ---------------------------------------------------------------------------
# _build_fewshot_prefix


MC_DATA = [
    {"query": "q1", "choices": ["a", "b", "c"], "gold": 0},
    {"query": "q2", "choices": ["d", "e", "f"], "gold": 1},
    {"query": "q3", "choices": ["g", "h", "i"], "gold": 2},
]
LM_DATA = [
    {"context": "ctx1", "continuation": "ans1"},
    {"context": "ctx2", "continuation": "ans2"},
    {"context": "ctx3", "continuation": "ans3"},
]
SCHEMA_DATA = [
    {"context_options": ["co1a", "co1b"], "continuation": "c1", "gold": 0},
    {"context_options": ["co2a", "co2b"], "continuation": "c2", "gold": 1},
    {"context_options": ["co3a", "co3b"], "continuation": "c3", "gold": 0},
]


class TestBuildFewshotPrefix:
    def test_skips_target_index(self):
        prefix = _build_fewshot_prefix(
            MC_DATA, idx=0, num_fewshot=2, task_type="multiple_choice", cont_delim=" "
        )
        assert "q1" not in prefix
        assert "q2" in prefix
        assert "q3" in prefix

    def test_respects_num_fewshot_limit(self):
        prefix = _build_fewshot_prefix(
            MC_DATA, idx=0, num_fewshot=1, task_type="multiple_choice", cont_delim=" "
        )
        assert "q2" in prefix
        assert "q3" not in prefix

    def test_zero_shot_returns_empty_string(self):
        prefix = _build_fewshot_prefix(
            MC_DATA, idx=0, num_fewshot=0, task_type="multiple_choice", cont_delim=" "
        )
        assert prefix == ""

    def test_multiple_choice_uses_gold_answer(self):
        # q2 has gold=1, so choices[1]="e" should appear, not "d" or "f"
        prefix = _build_fewshot_prefix(
            MC_DATA, idx=0, num_fewshot=1, task_type="multiple_choice", cont_delim=" "
        )
        assert "e" in prefix
        assert "d" not in prefix
        assert "f" not in prefix

    def test_language_modeling_format(self):
        prefix = _build_fewshot_prefix(
            LM_DATA, idx=0, num_fewshot=1, task_type="language_modeling", cont_delim=" "
        )
        assert "ctx2" in prefix
        assert "ans2" in prefix
        assert "ctx1" not in prefix

    def test_schema_uses_gold_context_option(self):
        # SCHEMA_DATA[1]: gold=1, so context_options[1]="co2b" should appear
        prefix = _build_fewshot_prefix(
            SCHEMA_DATA, idx=0, num_fewshot=1, task_type="schema", cont_delim=" "
        )
        assert "co2b" in prefix
        assert "co2a" not in prefix
        assert "c2" in prefix

    def test_multiple_examples_separated_by_double_newline(self):
        prefix = _build_fewshot_prefix(
            MC_DATA, idx=0, num_fewshot=2, task_type="multiple_choice", cont_delim=" "
        )
        assert "\n\n" in prefix


# ---------------------------------------------------------------------------
# _sum_continuation_logprob


class TestSumContinuationLogprob:
    def test_sums_at_correct_positions(self):
        """Sums log_probs at positions [prefix_len-1, prefix_len, ...]."""
        seq_len, vocab_size = 10, 8
        log_probs = jnp.zeros((seq_len, vocab_size))
        log_probs = log_probs.at[2, 3].set(-1.5)  # pos=2, token=3
        log_probs = log_probs.at[3, 5].set(-2.0)  # pos=3, token=5

        # prefix_len=3 → continuation starts at pos=2
        total = _sum_continuation_logprob(
            log_probs, prefix_len=3, cont_ids=[3, 5], seq_len=seq_len
        )
        assert abs(total - (-1.5 + -2.0)) < 1e-5

    def test_stops_before_seq_len_boundary(self):
        """Tokens at pos >= seq_len-1 are not included."""
        seq_len, vocab_size = 5, 4
        log_probs = jnp.full((seq_len, vocab_size), -1.0)

        # prefix_len=4 → pos values: 3 (j=0), 4 (j=1), 5 (j=2)
        # pos=4 >= seq_len-1=4 → break, so only j=0 (pos=3) contributes
        total = _sum_continuation_logprob(
            log_probs, prefix_len=4, cont_ids=[0, 1, 2], seq_len=seq_len
        )
        assert abs(total - (-1.0)) < 1e-5

    def test_empty_continuation(self):
        log_probs = jnp.ones((8, 4))
        total = _sum_continuation_logprob(log_probs, prefix_len=2, cont_ids=[], seq_len=8)
        assert total == 0.0

    def test_single_token_continuation(self):
        seq_len, vocab_size = 8, 4
        log_probs = jnp.zeros((seq_len, vocab_size))
        log_probs = log_probs.at[1, 2].set(-0.7)  # pos=1, token=2

        total = _sum_continuation_logprob(
            log_probs, prefix_len=2, cont_ids=[2], seq_len=seq_len
        )
        assert abs(total - (-0.7)) < 1e-5


# ---------------------------------------------------------------------------
# _evaluate_task
#
# Strategy: use mock_forward_favors_token_3 which assigns near-certain
# probability to token 3.  With SimpleTokenizer:
#   'c' → ord('c') % 16 = 3   (the "correct" token)
#   'd' → ord('d') % 16 = 4   (an "incorrect" token)


class TestEvaluateTask:
    TOK = SimpleTokenizer()
    FWD = staticmethod(mock_forward_favors_token_3)

    # ---- multiple_choice ------------------------------------------------

    def test_mc_all_correct(self):
        """Choice starting with token 3 ('c') beats choice starting with 4 ('d')."""
        data = [{"query": "q", "choices": ["c", "d"], "gold": 0}] * 4
        meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert acc == pytest.approx(1.0)

    def test_mc_all_wrong(self):
        """gold=1 ('d') but model always picks 'c' → 0% accuracy."""
        data = [{"query": "q", "choices": ["c", "d"], "gold": 1}] * 4
        meta = {"task_type": "multiple_choice", "num_fewshot": 0, "continuation_delimiter": " "}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert acc == pytest.approx(0.0)

    def test_mc_with_fewshot(self):
        """Few-shot context doesn't break evaluation (smoke)."""
        data = [{"query": "q", "choices": ["c", "d"], "gold": 0}] * 5
        meta = {"task_type": "multiple_choice", "num_fewshot": 2, "continuation_delimiter": " "}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert 0.0 <= acc <= 1.0

    # ---- language_modeling -----------------------------------------------

    def test_lm_all_correct(self):
        """Continuation starts with token 3 ('c') — matches model's prediction."""
        data = [{"context": "q", "continuation": "c"}] * 4
        meta = {"task_type": "language_modeling", "num_fewshot": 0, "continuation_delimiter": ""}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert acc == pytest.approx(1.0)

    def test_lm_all_wrong(self):
        """Continuation starts with token 4 ('d') — doesn't match prediction (3)."""
        data = [{"context": "q", "continuation": "d"}] * 4
        meta = {"task_type": "language_modeling", "num_fewshot": 0, "continuation_delimiter": ""}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert acc == pytest.approx(0.0)

    # ---- schema ----------------------------------------------------------

    def test_schema_returns_valid_accuracy(self):
        """Schema task runs and returns a value in [0, 1]."""
        data = [
            {"context_options": ["aa", "bb"], "continuation": "c", "gold": 0}
        ] * 4
        meta = {"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert 0.0 <= acc <= 1.0

    def test_schema_tied_scores_picks_gold_zero(self):
        """When both context options score equally, argmax picks index 0.
        So setting gold=0 should yield 100% accuracy."""
        # Both "aa" and "bb" are same length → same logprob for continuation → tie
        data = [
            {"context_options": ["aa", "bb"], "continuation": "c", "gold": 0}
        ] * 4
        meta = {"task_type": "schema", "num_fewshot": 0, "continuation_delimiter": " "}
        acc = _evaluate_task(self.FWD, self.TOK, data, meta, SEQ_LEN)
        assert acc == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# load_model_from_checkpoint + run_sampling  (require JAX compilation + I/O)


def _save_checkpoint(model, checkpoint_dir: Path, step: int) -> None:
    """Save just the model weights (mirrors what Experiment.save_checkpoint does)."""
    model_np = jax.tree.map(np.asarray, model)
    ckpt_manager = ocp.CheckpointManager(str(checkpoint_dir))
    ckpt_manager.save(step, args=ocp.args.Composite(model=ocp.args.StandardSave(model_np)))
    ckpt_manager.wait_until_finished()
    ckpt_manager.close()


@pytest.mark.expensive
class TestLoadModelFromCheckpoint:
    """Round-trip: save a model with orbax, reload with load_model_from_checkpoint."""

    DEPTH = 1
    SEQ_LEN_CKPT = 8
    VOCAB_SIZE_CKPT = 10

    @pytest.fixture
    def saved_model(self, tmp_path):
        """Create, save, and return (model, checkpoint_dir, step)."""
        model_config = get_model_config(
            depth=self.DEPTH, seq_len=self.SEQ_LEN_CKPT, vocab_size=self.VOCAB_SIZE_CKPT
        )
        model = decoder_only_model.DecoderOnlyTransformer(
            config=model_config, key=jax.random.key(0)
        )
        model.init()
        ckpt_dir = tmp_path / "checkpoints"
        ckpt_dir.mkdir()
        _save_checkpoint(model, ckpt_dir, step=100)
        return model, ckpt_dir

    def test_roundtrip_preserves_weights(self, saved_model):
        original, ckpt_dir = saved_model

        loaded, step, _ = load_model_from_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=100,
            depth=self.DEPTH,
            seq_len=self.SEQ_LEN_CKPT,
            vocab_size=self.VOCAB_SIZE_CKPT,
        )

        assert step == 100
        orig_leaves = jax.tree_util.tree_leaves(original)
        loaded_leaves = jax.tree_util.tree_leaves(loaded)
        assert len(orig_leaves) == len(loaded_leaves)
        for orig, restored in zip(orig_leaves, loaded_leaves):
            assert jnp.allclose(jnp.asarray(orig), jnp.asarray(restored)), (
                "Checkpoint round-trip should preserve every parameter exactly"
            )

    def test_auto_detects_latest_step(self, saved_model):
        _, ckpt_dir = saved_model

        _, step, _ = load_model_from_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=None,  # auto-detect
            depth=self.DEPTH,
            seq_len=self.SEQ_LEN_CKPT,
            vocab_size=self.VOCAB_SIZE_CKPT,
        )
        assert step == 100

    def test_loaded_model_produces_correct_output_shape(self, saved_model):
        _, ckpt_dir = saved_model

        loaded, _, model_config = load_model_from_checkpoint(
            checkpoint_dir=str(ckpt_dir),
            step=100,
            depth=self.DEPTH,
            seq_len=self.SEQ_LEN_CKPT,
            vocab_size=self.VOCAB_SIZE_CKPT,
        )
        tokens = jnp.zeros((model_config.seq_len,), dtype=jnp.int32)
        logits = loaded(tokens)
        assert logits.shape == (model_config.seq_len, model_config.vocab_size)


@pytest.mark.expensive
class TestRunSampling:
    """Smoke tests for run_sampling with a tiny model."""

    # Longest DEFAULT_PROMPT is ~46 chars; seq_len=64 gives headroom for char tokenizer.
    MODEL_CONFIG = decoder_only_model.ModelConfig(
        num_layers=1,
        num_heads=1,
        qkv_dim=8,
        vocab_size=128,
        seq_len=64,
    )

    @pytest.fixture
    def tiny_model(self):
        model = decoder_only_model.DecoderOnlyTransformer(
            config=self.MODEL_CONFIG, key=jax.random.key(0)
        )
        model.init()
        return model

    def test_smoke(self, tiny_model):
        """run_sampling completes without error."""
        tok = SimpleTokenizer()
        # SimpleTokenizer uses vocab_size=16 but model has vocab_size=128.
        # All encoded token IDs are < 16 < 128, so this is safe.
        run_sampling(tiny_model, self.MODEL_CONFIG, tok, max_tokens=4)

    def test_output_is_printed(self, tiny_model, capsys):
        """run_sampling prints a header and prompt/output lines."""
        tok = SimpleTokenizer()
        run_sampling(tiny_model, self.MODEL_CONFIG, tok, max_tokens=4)
        captured = capsys.readouterr()
        assert "Samples" in captured.out
        assert "Prompt" in captured.out
        assert "Output" in captured.out
