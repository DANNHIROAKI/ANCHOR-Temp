#!/usr/bin/env python3
"""Run one memory benchmark with fixed-interval /proc RSS polling.

The benchmark child and this monitor communicate through two inherited pipes.
The child announces INPUT_READY, BASELINE_READY, PREPARE_READY (when present),
and ONESHOT_DONE. The monitor captures event RSS values, polls VmRSS every
fixed interval between baseline and completion, and acknowledges every event
so the measured allocation remains live while it is sampled.

No cgroup delegation or privileged filesystem write is required. The memory
cap is a sampled soft cap: when an observed RSS exceeds it, the whole benchmark
process group is terminated.
"""

from __future__ import annotations

import argparse
import json
import os
import pathlib
import select
import signal
import subprocess
import sys
import time
from typing import Any


UNAVAILABLE_EXIT = 78
EVENT_INPUT_READY = b"I"
EVENT_BASELINE_READY = b"B"
EVENT_PREPARE_READY = b"P"
EVENT_ONESHOT_DONE = b"D"


class MemoryMeasurementUnavailable(RuntimeError):
    """Raised when the required procfs RSS source cannot be read."""


def read_vmrss_bytes(pid: int) -> int:
    """Return current resident memory from /proc/<pid>/status in bytes."""

    path = pathlib.Path("/proc") / str(pid) / "status"
    try:
        with path.open("r", encoding="ascii") as stream:
            for line in stream:
                if line.startswith("VmRSS:"):
                    fields = line.split()
                    if len(fields) != 3 or fields[2] != "kB":
                        raise MemoryMeasurementUnavailable(
                            f"unexpected VmRSS format in {path}: {line.strip()!r}"
                        )
                    value = int(fields[1])
                    if value < 0:
                        raise ValueError("negative VmRSS")
                    return value * 1024
    except (OSError, ValueError) as exc:
        raise MemoryMeasurementUnavailable(f"cannot read {path}: {exc}") from exc
    raise MemoryMeasurementUnavailable(f"VmRSS is unavailable in {path}")


def _write_report(path: pathlib.Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _acknowledge(descriptor: int) -> None:
    while True:
        try:
            if os.write(descriptor, b"1") != 1:
                raise MemoryMeasurementUnavailable("short memory-protocol ACK write")
            return
        except InterruptedError:
            continue
        except OSError as exc:
            raise MemoryMeasurementUnavailable(
                f"cannot acknowledge benchmark memory event: {exc}"
            ) from exc


def _terminate_group(
    process: subprocess.Popen[bytes], *, grace_seconds: float = 0.25
) -> str | None:
    if process.poll() is not None:
        return None
    sent = "SIGTERM"
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return None
    deadline = time.monotonic() + grace_seconds
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.01)
    if process.poll() is None:
        sent = "SIGKILL"
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    return sent


def _advance_deadline(previous: float, interval: float, now: float) -> float:
    """Advance to the first future absolute slot without catch-up sampling."""

    candidate = previous + interval
    if candidate <= now:
        candidate += (int((now - candidate) // interval) + 1) * interval
    return candidate


def _exceeds_memory_cap(rss_bytes: int, memory_cap_bytes: int) -> bool:
    return rss_bytes > memory_cap_bytes


def run(args: argparse.Namespace) -> tuple[int, dict[str, Any], bytes, bytes]:
    event_read, event_write = os.pipe()
    ack_read, ack_write = os.pipe()
    process: subprocess.Popen[bytes] | None = None
    interval = args.poll_interval_ms / 1000.0
    report: dict[str, Any] = {
        "status": "MEMORY-MEASUREMENT-UNAVAILABLE",
        "memory_measurement_backend": "procfs_vmrss_polling",
        "memory_poll_interval_ms": args.poll_interval_ms,
        "peak_is_sampled": True,
        "rss_sample_count": 0,
        "memory_cap_exceeded": False,
        "monitor_cpu_core": args.monitor_cpu_core,
        "monitor_cpu_affinity_applied": False,
    }
    samples: list[int] = []
    sample_times_ns: list[int] = []
    stdout = stderr = b""
    forced_status: str | None = None
    try:
        command = [
            *args.command,
            "--memory-event-fd",
            str(event_write),
            "--memory-ack-fd",
            str(ack_read),
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            pass_fds=(event_write, ack_read),
            start_new_session=True,
        )

        # CPU affinity is inherited across fork/exec. Pinning this monitor
        # before Popen would also constrain the child, preventing a child-side
        # numactl command from binding the benchmark to a different CPU.
        if args.monitor_cpu_core is not None:
            try:
                os.sched_setaffinity(0, {args.monitor_cpu_core})
                report["monitor_cpu_affinity_applied"] = True
            except (AttributeError, OSError) as exc:
                report["monitor_cpu_affinity_error"] = str(exc)

        os.close(event_write)
        event_write = -1
        os.close(ack_read)
        ack_read = -1

        safety_deadline = time.monotonic() + args.timeout_seconds
        next_sample_deadline: float | None = None
        baseline_ready = False
        protocol_done = False

        def sample(*, cap_is_active: bool) -> int:
            nonlocal forced_status
            assert process is not None
            value = read_vmrss_bytes(process.pid)
            now_ns = time.monotonic_ns()
            report["last_observed_rss_bytes"] = value
            if cap_is_active:
                samples.append(value)
                sample_times_ns.append(now_ns)
                report["rss_sample_count"] = len(samples)
                if _exceeds_memory_cap(value, args.memory_cap_bytes):
                    report["memory_cap_exceeded"] = True
                    report["termination_signal"] = _terminate_group(process)
                    forced_status = "MEMORY-CAP-EXCEEDED"
            return value

        while not protocol_done and forced_status is None:
            assert process is not None
            if process.poll() is not None:
                break
            now = time.monotonic()
            if now >= safety_deadline:
                report["termination_signal"] = _terminate_group(process)
                report["timeout_source"] = "procfs-wrapper-safety-timeout"
                forced_status = "TO"
                break

            wait_until = safety_deadline
            if next_sample_deadline is not None:
                wait_until = min(wait_until, next_sample_deadline)
            timeout = max(0.0, min(wait_until - now, 0.1))
            readable, _, _ = select.select([event_read], [], [], timeout)
            if readable:
                try:
                    event = os.read(event_read, 1)
                except OSError as exc:
                    raise MemoryMeasurementUnavailable(
                        f"cannot read benchmark memory event: {exc}"
                    ) from exc
                if not event:
                    if process.poll() is None:
                        raise MemoryMeasurementUnavailable(
                            "benchmark closed event pipe before ONESHOT_DONE"
                        )
                    break
                if event == EVENT_INPUT_READY:
                    report["InputMemory"] = sample(cap_is_active=False)
                    _acknowledge(ack_write)
                elif event == EVENT_BASELINE_READY:
                    report["BaselineMemory"] = sample(cap_is_active=True)
                    baseline_ready = True
                    next_sample_deadline = time.monotonic() + interval
                    if forced_status is None:
                        _acknowledge(ack_write)
                elif event == EVENT_PREPARE_READY:
                    if not baseline_ready:
                        raise MemoryMeasurementUnavailable(
                            "PREPARE_READY arrived before BASELINE_READY"
                        )
                    report["MemoryAfterPrepare"] = sample(cap_is_active=True)
                    if forced_status is None:
                        _acknowledge(ack_write)
                elif event == EVENT_ONESHOT_DONE:
                    if not baseline_ready:
                        raise MemoryMeasurementUnavailable(
                            "ONESHOT_DONE arrived before BASELINE_READY"
                        )
                    sample(cap_is_active=True)
                    if forced_status is None:
                        _acknowledge(ack_write)
                        protocol_done = True
                else:
                    raise MemoryMeasurementUnavailable(
                        f"unknown benchmark memory event byte: {event!r}"
                    )
                continue

            now = time.monotonic()
            if (
                next_sample_deadline is not None
                and now >= next_sample_deadline
                and forced_status is None
            ):
                sample(cap_is_active=True)
                next_sample_deadline = _advance_deadline(
                    next_sample_deadline, interval, time.monotonic()
                )

        if process.poll() is None:
            remaining = max(0.01, safety_deadline - time.monotonic())
            try:
                stdout, stderr = process.communicate(timeout=remaining)
            except subprocess.TimeoutExpired:
                report["termination_signal"] = _terminate_group(process)
                report["timeout_source"] = "procfs-wrapper-safety-timeout"
                forced_status = "TO"
                stdout, stderr = process.communicate()
        else:
            stdout, stderr = process.communicate()

        if samples:
            peak = max(samples)
            report["PeakRSSPollBytes"] = peak
            report["PeakMemoryTotal"] = peak
            report["rss_first_sample_monotonic_ns"] = sample_times_ns[0]
            report["rss_last_sample_monotonic_ns"] = sample_times_ns[-1]
            if report.get("InputMemory") is not None:
                report["PeakMemoryIncremental"] = max(
                    0, peak - int(report["InputMemory"])
                )
            if report.get("BaselineMemory") is not None:
                report["PeakMemoryAux"] = max(
                    0, peak - int(report["BaselineMemory"])
                )

        if forced_status is not None:
            report["status"] = forced_status
        elif process.returncode in (-signal.SIGKILL, 128 + signal.SIGKILL):
            report["status"] = "OOM"
        else:
            # A benchmark may reject its workload before the baseline. The
            # harness will preserve that benchmark-generated classification.
            report["status"] = "OK"

        if report["status"] == "TO":
            return 124, report, stdout, stderr
        if report["status"] in {"OOM", "MEMORY-CAP-EXCEEDED"}:
            return 137, report, stdout, stderr
        return int(process.returncode or 0), report, stdout, stderr
    except MemoryMeasurementUnavailable as exc:
        report["status"] = "MEMORY-MEASUREMENT-UNAVAILABLE"
        report["error_message"] = str(exc)
        if process is not None and process.poll() is None:
            report["termination_signal"] = _terminate_group(process)
            stdout, stderr = process.communicate()
        return UNAVAILABLE_EXIT, report, stdout, stderr
    finally:
        for descriptor in (event_read, event_write, ack_read, ack_write):
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
        if process is not None and process.poll() is None:
            _terminate_group(process)
            process.wait()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--memory-cap-bytes", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, required=True)
    parser.add_argument("--poll-interval-ms", type=int, default=5)
    parser.add_argument("--monitor-cpu-core", type=int)
    parser.add_argument("--report", type=pathlib.Path)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("a command is required after --")
    if args.memory_cap_bytes <= 0 or args.timeout_seconds <= 0:
        parser.error("memory cap and timeout must be positive")
    if args.poll_interval_ms <= 0:
        parser.error("poll interval must be positive")
    if args.monitor_cpu_core is not None and args.monitor_cpu_core < 0:
        parser.error("monitor CPU core must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    code, report, stdout, stderr = run(args)
    _write_report(args.report, report)
    sys.stdout.buffer.write(stdout)
    sys.stdout.buffer.flush()
    sys.stderr.buffer.write(stderr)
    sys.stderr.buffer.flush()
    return code


if __name__ == "__main__":
    raise SystemExit(main())
