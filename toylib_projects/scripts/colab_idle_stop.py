"""Stop Colab sessions that sit idle, so a finished or crashed job stops billing.

Every minute, lists the sessions the `colab` CLI knows about and asks each one how busy
its GPU and CPU are. A session that stays idle for the whole window is stopped.

    uv run python -m toylib_projects.scripts.colab_idle_stop
    uv run python -m toylib_projects.scripts.colab_idle_stop --idle-minutes 10 --dry-run

Idle means the GPU's utilization stays at or below `--gpu-percent` and the machine uses
less than `--cpu-cores` cores, sampled over a few seconds. CPU counts so that work
without the GPU, like writing token files, isn't mistaken for idling, and so does
waiting on disk. A session that doesn't answer counts as busy, so its idle clock starts
over: a job that loads the machine too much to answer is still a job.
"""

import argparse
import dataclasses
import datetime
import json
import os
import re
import signal
import subprocess
import tempfile
import time

COLAB = ["colab", "--auth=adc"]

# Runs on the session: samples CPU from /proc/stat and the GPU from nvidia-smi.
PROBE = """
import json, subprocess, time
def cpu():
    fields = [int(x) for x in open("/proc/stat").readline().split()[1:]]
    return sum(fields), fields[3]  # Total, and idle; waiting on disk counts as busy.
def gpu():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu",
            "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10).stdout
        return max(float(x) for x in out.split())
    except Exception:
        return 0.0  # No GPU, or no answer: only the CPU decides.
total0, idle0 = cpu()
gpu_percent = 0.0
for _ in range(SAMPLES):
    time.sleep(1)
    gpu_percent = max(gpu_percent, gpu())
total1, idle1 = cpu()
busy = 1 - (idle1 - idle0) / max(total1 - total0, 1)
cores = busy * len([l for l in open("/proc/stat") if l[:3] == "cpu" and l[3].isdigit()])
print("PROBE " + json.dumps({"gpu_percent": gpu_percent, "cpu_cores": cores}))
"""


@dataclasses.dataclass(frozen=True)
class Usage:
    gpu_percent: float
    cpu_cores: float


def run(args: list[str], timeout: float) -> str:
    """Run a command and return its output, killing its whole process group at the
    deadline, so a child that keeps the pipe open can't hang the poll."""
    with subprocess.Popen(
        args,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    ) as proc:
        try:
            return proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.communicate()
            return ""


def parse_sessions(output: str) -> list[str]:
    """Session names from `colab sessions`, whose lines look like `[name] host | ...`.

    The CLI's own messages also start with brackets ("[colab] No active sessions..."),
    so a session line must name its hardware too.
    """
    return re.findall(r"^\[([^\]]+)\] \S+ \| Hardware:", output, flags=re.MULTILINE)


def parse_probe(output: str) -> Usage | None:
    match = re.search(r"^PROBE (\{.*\})$", output, flags=re.MULTILINE)
    return Usage(**json.loads(match.group(1))) if match else None


def is_idle(usage: Usage, gpu_percent: float, cpu_cores: float) -> bool:
    return usage.gpu_percent <= gpu_percent and usage.cpu_cores < cpu_cores


class IdleTracker:
    """Remember when each session went idle, and say which have been idle long enough."""

    def __init__(self, idle_seconds: float):
        self.idle_seconds = idle_seconds
        self.idle_since: dict[str, float] = {}

    def update(self, session: str, idle: bool | None, now: float) -> bool:
        """Record one poll and return whether to stop the session.

        `idle` is None when the session didn't answer. That counts as busy: a machine
        too loaded to answer looks just like one that has hung.
        """
        if idle:
            self.idle_since.setdefault(session, now)
        else:
            self.idle_since.pop(session, None)
        since = self.idle_since.get(session)
        return since is not None and now - since >= self.idle_seconds

    def forget_others(self, sessions: list[str]) -> None:
        for name in set(self.idle_since) - set(sessions):
            del self.idle_since[name]


def log(message: str) -> None:
    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"{stamp}  {message}", flush=True)


def poll(args: argparse.Namespace, tracker: IdleTracker, probe_path: str) -> None:
    sessions = parse_sessions(run([*COLAB, "sessions"], timeout=60))
    tracker.forget_others(sessions)
    for name in sessions:
        output = run(
            [*COLAB, "exec", "-s", name, "-f", probe_path, "--timeout", "40"],
            timeout=60,
        )
        usage = parse_probe(output)
        idle = None if usage is None else is_idle(usage, args.gpu_percent, args.cpu_cores)
        stop = tracker.update(name, idle, time.time())
        if usage is None:
            state = "no answer"
        else:
            state = f"gpu {usage.gpu_percent:.0f}%, cpu {usage.cpu_cores:.2f} cores, " + (
                "idle" if idle else "busy"
            )
        if name in tracker.idle_since:
            state += f" for {(time.time() - tracker.idle_since[name]) / 60:.1f} min"
        log(f"{name}: {state}")
        if stop:
            if args.dry_run:
                log(f"{name}: would stop (dry run)")
                continue
            log(f"{name}: stopping, {run([*COLAB, 'stop', '-s', name], timeout=120).strip()}")
            tracker.idle_since.pop(name, None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle-minutes", type=float, default=5)
    parser.add_argument("--interval", type=float, default=60, help="Seconds between polls")
    parser.add_argument("--gpu-percent", type=float, default=5)
    parser.add_argument("--cpu-cores", type=float, default=0.5)
    parser.add_argument("--samples", type=int, default=3, help="Seconds of sampling")
    parser.add_argument("--dry-run", action="store_true", help="Report, never stop")
    parser.add_argument("--once", action="store_true", help="Poll once and exit")
    args = parser.parse_args()

    tracker = IdleTracker(60 * args.idle_minutes)
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(PROBE.replace("SAMPLES", str(args.samples)))
    log(f"Stopping sessions idle for {args.idle_minutes:g} min; polling every "
        f"{args.interval:g}s{' (dry run)' if args.dry_run else ''}.")
    try:
        while True:
            start = time.monotonic()
            poll(args, tracker, f.name)
            if args.once:
                break
            time.sleep(max(0.0, args.interval - (time.monotonic() - start)))
    finally:
        os.unlink(f.name)


if __name__ == "__main__":
    main()
