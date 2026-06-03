"""Tests for logger.py.

There is no `logger_test.py` in tinystories to fork from — this file is new.
Covers the three sinks shipped in `logger.py`: ``FileLogger``,
``StdoutLogger``, and ``WandBLogger`` (which is mocked since real wandb runs
require network + credentials).
"""

import datetime
import json
from pathlib import Path
from unittest import mock

from . import logger as logger_mod


# ──────────────────────────────────────────────────────────────────────────
# FileLogger
# ──────────────────────────────────────────────────────────────────────────


class TestFileLogger:
    def test_writes_jsonl_with_step_and_timestamp(self, tmp_path: Path) -> None:
        log = logger_mod.FileLogger(
            config_dict={"a": 1}, output_path=str(tmp_path), run_id="run-A"
        )
        log.log(step=0, metrics={"loss": 0.5})
        log.log(step=1, metrics={"loss": 0.4})
        log.close()

        path = tmp_path / "logs_run-A.txt"
        assert path.exists()
        rows = [
            json.loads(line)
            for line in path.read_text().splitlines()
            if line.strip()
        ]
        assert len(rows) == 2
        assert rows[0]["loss"] == 0.5 and rows[0]["step"] == 0
        assert rows[1]["loss"] == 0.4 and rows[1]["step"] == 1
        # Timestamp is added by the logger and should parse as ISO-ish.
        for r in rows:
            datetime.datetime.strptime(r["timestamp"], "%Y-%m-%d %H:%M:%S")

    def test_creates_output_dir_if_missing(self, tmp_path: Path) -> None:
        """Logger should mkdir its output_path so callers don't have to."""
        target = tmp_path / "nested" / "logs"
        log = logger_mod.FileLogger(
            config_dict={}, output_path=str(target), run_id="x"
        )
        log.close()
        assert target.exists()
        assert (target / "logs_x.txt").exists()

    def test_run_id_defaults_to_timestamp(self, tmp_path: Path) -> None:
        """Without run_id, filename uses an ISO-like timestamp."""
        log = logger_mod.FileLogger(config_dict={}, output_path=str(tmp_path))
        log.close()
        names = [p.name for p in tmp_path.glob("logs_*.txt")]
        assert len(names) == 1
        # Default label format is "%Y%m%dT%H%M%S" — 15 chars after "logs_" prefix.
        stem = names[0].removeprefix("logs_").removesuffix(".txt")
        datetime.datetime.strptime(stem, "%Y%m%dT%H%M%S")

    def test_context_manager_closes(self, tmp_path: Path) -> None:
        with logger_mod.FileLogger(
            config_dict={}, output_path=str(tmp_path), run_id="ctx"
        ) as log:
            log.log(step=0, metrics={"x": 1})
        # File should be closed; another open + read should work fine.
        path = tmp_path / "logs_ctx.txt"
        assert path.exists()
        assert "x" in path.read_text()

    def test_non_serializable_metric_does_not_crash(self, tmp_path: Path) -> None:
        """`default=str` fallback means callables/objects get stringified, not raise."""
        log = logger_mod.FileLogger(
            config_dict={}, output_path=str(tmp_path), run_id="weird"
        )
        log.log(step=0, metrics={"callable_metric": (lambda x: x)})
        log.close()

        text = (tmp_path / "logs_weird.txt").read_text()
        rows = [json.loads(line) for line in text.splitlines() if line.strip()]
        assert len(rows) == 1
        # The lambda survived as some string representation.
        assert isinstance(rows[0]["callable_metric"], str)


# ──────────────────────────────────────────────────────────────────────────
# StdoutLogger
# ──────────────────────────────────────────────────────────────────────────


class TestStdoutLogger:
    def test_prints_step_and_metrics(self, capsys) -> None:
        log = logger_mod.StdoutLogger(config_dict={})
        log.log(step=42, metrics={"foo": 1.0, "bar": 2.5})
        out = capsys.readouterr().out
        assert "Step 42" in out
        assert "foo" in out and "bar" in out

    def test_close_is_noop(self) -> None:
        log = logger_mod.StdoutLogger(config_dict={})
        log.close()  # must not raise


# ──────────────────────────────────────────────────────────────────────────
# WandBLogger (mocked — real wandb needs credentials)
# ──────────────────────────────────────────────────────────────────────────


class TestWandBLogger:
    def test_init_calls_wandb_with_expected_args(self) -> None:
        """Mock wandb so we exercise the logger without a real run."""
        fake_run = mock.MagicMock()
        fake_wandb = mock.MagicMock()
        fake_wandb.init.return_value = fake_run

        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log = logger_mod.WandBLogger(
                config_dict={"hp": 1},
                project_name="wm-test",
                user_name="anuj",
                run_id="abc",
            )

        fake_wandb.init.assert_called_once_with(
            entity="anuj",
            project="wm-test",
            config={"hp": 1},
            id="abc",
            resume="allow",
        )
        fake_run.define_metric.assert_called_once_with("*", step_metric="global_step")
        assert log.run is fake_run

    def test_log_forwards_with_global_step(self) -> None:
        fake_run = mock.MagicMock()
        fake_wandb = mock.MagicMock()
        fake_wandb.init.return_value = fake_run

        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log = logger_mod.WandBLogger(
                config_dict={}, project_name="p", user_name="u",
            )
            log.log(step=7, metrics={"loss": 0.1})

        # The logger stamps global_step onto the metrics dict before forwarding.
        fake_run.log.assert_called_once_with({"loss": 0.1, "global_step": 7})

    def test_close_calls_finish(self) -> None:
        fake_run = mock.MagicMock()
        fake_wandb = mock.MagicMock()
        fake_wandb.init.return_value = fake_run

        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log = logger_mod.WandBLogger(
                config_dict={}, project_name="p", user_name="u",
            )
            log.close()
        fake_run.finish.assert_called_once()
