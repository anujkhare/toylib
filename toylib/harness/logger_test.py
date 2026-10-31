"""Tests for logger.py.

Covers the three sinks shipped in `logger.py`: ``FileLogger``,
``StdoutLogger``, and ``WandBLogger`` (which is mocked since real wandb runs
require network + credentials).
"""

import datetime
import json
from pathlib import Path
from unittest import mock

import pytest

from toylib.harness import logger as logger_mod


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
# MultiLogger
# ──────────────────────────────────────────────────────────────────────────


class TestMultiLogger:
    def test_each_sink_gets_its_own_metrics(self, tmp_path: Path) -> None:
        file_log = logger_mod.FileLogger(
            config_dict={}, output_path=str(tmp_path), run_id="m"
        )
        other = mock.MagicMock(spec=logger_mod.Logger)
        log = logger_mod.MultiLogger([file_log, other])
        log.log(step=3, metrics={"loss": 0.5})
        log.close()

        # FileLogger's timestamp must not leak into what wandb receives.
        other.log.assert_called_once_with(3, {"loss": 0.5})
        other.close.assert_called_once()
        assert '"loss": 0.5' in (tmp_path / "logs_m.txt").read_text()

    def test_fetch_checkpoint_returns_the_first_hit(self) -> None:
        log = logger_mod.MultiLogger(
            [
                logger_mod.StdoutLogger(config_dict={}),
                mock.MagicMock(spec=logger_mod.Logger),
            ]
        )
        log.loggers[1].fetch_checkpoint.return_value = 12
        assert log.fetch_checkpoint("/ckpts") == 12


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
                project_name="toylib-test",
                user_name="anuj",
                run_id="abc",
            )

        fake_wandb.init.assert_called_once_with(
            entity="anuj",
            project="toylib-test",
            config={"hp": 1},
            id="abc",
            dir=None,
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

    def _logger(self, user_name="u", **kwargs):
        fake_wandb = mock.MagicMock()
        fake_wandb.errors.CommError = type("CommError", (Exception,), {})
        fake_wandb.init.return_value.id = "run"
        fake_wandb.init.return_value.offline = False
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log = logger_mod.WandBLogger(
                config_dict={}, project_name="p", user_name=user_name, **kwargs
            )
        return log, fake_wandb

    def test_blank_user_name_means_the_default_entity(self) -> None:
        _, fake_wandb = self._logger(user_name="")
        assert fake_wandb.init.call_args.kwargs["entity"] is None

    def test_checkpoints_stay_local_unless_asked(self) -> None:
        log, fake_wandb = self._logger()
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log.log_checkpoint(5, "/ckpts/5")
            assert log.fetch_checkpoint("/ckpts") is None
        fake_wandb.Artifact.assert_not_called()
        log.run.use_artifact.assert_not_called()

    def test_log_checkpoint_uploads_the_directory(self) -> None:
        log, fake_wandb = self._logger(upload_checkpoints=True)
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            log.log_checkpoint(5, "/ckpts/5")
        fake_wandb.Artifact.assert_called_once_with(
            "run-checkpoints", type="checkpoint", metadata={"step": 5}
        )
        artifact = fake_wandb.Artifact.return_value
        artifact.add_dir.assert_called_once_with("/ckpts/5")
        log.run.log_artifact.assert_called_once_with(artifact, aliases=["step-5"])

    def test_fetch_checkpoint_downloads_into_the_step_directory(self) -> None:
        log, fake_wandb = self._logger(upload_checkpoints=True)
        artifact = log.run.use_artifact.return_value
        artifact.metadata = {"step": 5}
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            assert log.fetch_checkpoint("/ckpts") == 5
        artifact.download.assert_called_once_with(root="/ckpts/5")

    def test_fetch_checkpoint_without_any_upload(self) -> None:
        log, fake_wandb = self._logger(upload_checkpoints=True)
        log.run.use_artifact.side_effect = ValueError("artifact 'x' not found in 'p'")
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            assert log.fetch_checkpoint("/ckpts") is None

    def test_fetch_checkpoint_does_not_hide_a_network_failure(self) -> None:
        log, fake_wandb = self._logger(upload_checkpoints=True)
        error = fake_wandb.errors.CommError("HTTP 503")
        log.run.use_artifact.side_effect = error
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            with pytest.raises(type(error)):
                log.fetch_checkpoint("/ckpts")

    def test_fetch_checkpoint_offline(self) -> None:
        log, fake_wandb = self._logger(upload_checkpoints=True)
        log.run.offline = True
        with mock.patch.dict("sys.modules", {"wandb": fake_wandb}):
            assert log.fetch_checkpoint("/ckpts") is None
        log.run.use_artifact.assert_not_called()
