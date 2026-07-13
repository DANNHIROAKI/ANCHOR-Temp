from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import pathlib
import sys

import pytest


REPOSITORY = pathlib.Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "anchor_run_procfs", REPOSITORY / "scripts" / "run_procfs.py"
)
assert SPEC is not None and SPEC.loader is not None
run_procfs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(run_procfs)


def test_procfs_reads_current_vmrss_and_absolute_deadlines() -> None:
    assert run_procfs.read_vmrss_bytes(os.getpid()) > 0
    assert run_procfs._advance_deadline(1.0, 0.005, 1.004) == 1.005
    assert math.isclose(run_procfs._advance_deadline(1.0, 0.005, 1.017), 1.02)


def test_procfs_monitor_event_protocol_and_peak_formulas() -> None:
    child = r"""
import json, os, sys, time
args = sys.argv[1:]
event_fd = int(args[args.index("--memory-event-fd") + 1])
ack_fd = int(args[args.index("--memory-ack-fd") + 1])
keep = []
def event(value):
    os.write(event_fd, value)
    if os.read(ack_fd, 1) != b"1":
        raise RuntimeError("missing ACK")
event(b"I")
keep.append(bytearray(2 * 1024 * 1024))
event(b"B")
keep.append(bytearray(8 * 1024 * 1024))
time.sleep(0.02)
event(b"P")
keep.append(bytearray(4 * 1024 * 1024))
time.sleep(0.02)
event(b"D")
print(json.dumps({"status": "OK", "bytes": sum(map(len, keep))}))
"""
    args = argparse.Namespace(
        command=[sys.executable, "-c", child],
        memory_cap_bytes=512 * 1024 * 1024,
        timeout_seconds=5.0,
        poll_interval_ms=2,
        monitor_cpu_core=None,
        report=None,
    )
    code, report, stdout, stderr = run_procfs.run(args)
    assert code == 0, stderr.decode(errors="replace")
    assert json.loads(stdout)["status"] == "OK"
    assert report["status"] == "OK"
    assert report["memory_measurement_backend"] == "procfs_vmrss_polling"
    assert report["memory_poll_interval_ms"] == 2
    assert report["peak_is_sampled"] is True
    assert report["rss_sample_count"] >= 3
    assert report["PeakRSSPollBytes"] == report["PeakMemoryTotal"]
    assert report["PeakMemoryIncremental"] == max(
        0, report["PeakMemoryTotal"] - report["InputMemory"]
    )
    assert report["PeakMemoryAux"] == max(
        0, report["PeakMemoryTotal"] - report["BaselineMemory"]
    )
    assert report["MemoryAfterPrepare"] >= report["BaselineMemory"]


def test_procfs_monitor_affinity_is_not_inherited_by_child() -> None:
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        pytest.skip("Linux CPU affinity APIs are unavailable")
    original_affinity = os.sched_getaffinity(0)
    if len(original_affinity) < 2:
        pytest.skip("the affinity inheritance test needs two available CPUs")
    monitor_core = sorted(original_affinity)[1]
    child = r"""
import json, os, sys
args = sys.argv[1:]
event_fd = int(args[args.index("--memory-event-fd") + 1])
ack_fd = int(args[args.index("--memory-ack-fd") + 1])
def event(value):
    os.write(event_fd, value)
    if os.read(ack_fd, 1) != b"1":
        raise RuntimeError("missing ACK")
event(b"I")
event(b"B")
event(b"D")
print(json.dumps({"affinity": sorted(os.sched_getaffinity(0))}))
"""
    args = argparse.Namespace(
        command=[sys.executable, "-c", child],
        memory_cap_bytes=512 * 1024 * 1024,
        timeout_seconds=5.0,
        poll_interval_ms=2,
        monitor_cpu_core=monitor_core,
        report=None,
    )
    monitor_affinity: set[int] | None = None
    try:
        code, report, stdout, stderr = run_procfs.run(args)
        monitor_affinity = os.sched_getaffinity(0)
    finally:
        os.sched_setaffinity(0, original_affinity)

    assert code == 0, stderr.decode(errors="replace")
    assert report["status"] == "OK"
    assert report["monitor_cpu_affinity_applied"] is True
    assert monitor_affinity == {monitor_core}
    child_report = json.loads(stdout)
    assert set(child_report["affinity"]) == original_affinity
    assert isinstance(report["InputMemory"], int)
    assert isinstance(report["PeakMemoryTotal"], int)
