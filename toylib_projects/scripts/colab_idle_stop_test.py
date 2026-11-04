"""Tests for colab_idle_stop.py."""

from toylib_projects.scripts import colab_idle_stop as cis


def test_parse_sessions_reads_bracketed_names():
    output = (
        "[d12b] gpu-a100-s-abc | Hardware: A100 | Shape: Standard | Variant: GPU\n"
        "[cpu-box] cpu-xyz | Hardware: None | Shape: Standard | Variant: CPU\n"
    )
    assert cis.parse_sessions(output) == ["d12b", "cpu-box"]


def test_cli_messages_are_not_sessions():
    assert cis.parse_sessions("[colab] No active sessions found on server.\n") == []
    assert cis.parse_sessions("[colab] Session 'd12' not found.\n") == []


def test_parse_probe_finds_the_line_among_other_output():
    usage = cis.parse_probe('noise\nPROBE {"gpu_percent": 97.0, "cpu_cores": 1.5}\n')
    assert usage == cis.Usage(gpu_percent=97.0, cpu_cores=1.5)
    assert cis.parse_probe("Traceback ...") is None


def test_busy_cpu_without_gpu_is_not_idle():
    assert not cis.is_idle(cis.Usage(0.0, 6.0), gpu_percent=5, cpu_cores=0.5)
    assert not cis.is_idle(cis.Usage(90.0, 0.1), gpu_percent=5, cpu_cores=0.5)
    assert cis.is_idle(cis.Usage(0.0, 0.1), gpu_percent=5, cpu_cores=0.5)


def test_stops_only_after_the_whole_window_idle():
    tracker = cis.IdleTracker(idle_seconds=300)
    assert not tracker.update("a", True, now=0)
    assert not tracker.update("a", True, now=240)
    assert tracker.update("a", True, now=300)


def test_activity_resets_the_clock():
    tracker = cis.IdleTracker(idle_seconds=300)
    tracker.update("a", True, now=0)
    tracker.update("a", False, now=200)
    assert not tracker.update("a", True, now=300)
    assert tracker.update("a", True, now=600)


def test_no_answer_counts_as_busy():
    # d12's final evaluation stopped answering for six minutes and was stopped mid-run.
    tracker = cis.IdleTracker(idle_seconds=300)
    tracker.update("a", True, now=0)
    assert not tracker.update("a", None, now=360)
    assert not tracker.update("a", True, now=420)
    assert tracker.update("a", True, now=720)


def test_forgets_sessions_that_are_gone():
    tracker = cis.IdleTracker(idle_seconds=300)
    tracker.update("a", True, now=0)
    tracker.forget_others(["b"])
    assert not tracker.update("a", True, now=300)
