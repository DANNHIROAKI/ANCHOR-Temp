"""Capture and validate the frozen execution environment."""

from __future__ import annotations

import json
import os
import pathlib
import platform
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from typing import Any

from .protocol import OFFICIAL_TIMEOUT_SECONDS
from .stable_hash import hash_file, stable_hash


def _read(path: str) -> str | None:
    try:
        return pathlib.Path(path).read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None


def _command(*arguments: str) -> str | None:
    try:
        return subprocess.run(
            arguments,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _cpu_model() -> str | None:
    try:
        for line in pathlib.Path("/proc/cpuinfo").read_text().splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or None


def _swap_bytes() -> int | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("SwapTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _memory_total_bytes() -> int | None:
    try:
        for line in pathlib.Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _boost_state() -> str | None:
    no_turbo = _read("/sys/devices/system/cpu/intel_pstate/no_turbo")
    if no_turbo is not None:
        return "disabled" if no_turbo == "1" else "enabled"
    boost = _read("/sys/devices/system/cpu/cpufreq/boost")
    if boost is not None:
        return "enabled" if boost == "1" else "disabled"
    return None


def capture_environment(
    *,
    benchmark_path: str | pathlib.Path,
    memory_cap_bytes: int,
    cpu_core: int,
    numa_node: int,
    code_commit: str,
    validation_report_path: str | pathlib.Path,
    memory_configuration: str,
    smt_sibling_idle_confirmed: bool,
    allocator_id: str = "system-default",
    prng_id: str = "anchor-domain-rng-v1",
    build_flags: Sequence[str] = (),
    linker_id: str | None = None,
    target_isa: str | None = None,
    third_party_commits: Mapping[str, str] | None = None,
    monitor_cpu_core: int | None = None,
    memory_poll_interval_ms: int = 5,
) -> dict[str, Any]:
    """Return a machine manifest; this operation never changes host settings."""

    executable = pathlib.Path(benchmark_path).resolve()
    if not executable.is_file():
        raise FileNotFoundError(executable)
    validation_path = pathlib.Path(validation_report_path).resolve()
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    with validation_path.open("r", encoding="utf-8") as stream:
        validation = json.load(stream)
    executable_sha256 = hash_file(executable)
    if validation.get("schema_version") != "anchor-validation-report-v1":
        raise ValueError("unsupported validation report schema")
    if validation.get("status") != "OK":
        raise ValueError("publication capture requires validation status OK")
    if validation.get("benchmark_sha256") != executable_sha256:
        raise ValueError("validation report benchmark SHA-256 does not match executable")
    compiler = shutil.which("c++") or shutil.which("g++") or shutil.which("clang++")
    numactl = shutil.which("numactl")
    governor = _read(f"/sys/devices/system/cpu/cpu{cpu_core}/cpufreq/scaling_governor")
    thp = _read("/sys/kernel/mm/transparent_hugepage/enabled")
    proc_status = _read("/proc/self/status")
    procfs_vmrss_available = bool(
        proc_status and any(line.startswith("VmRSS:") for line in proc_status.splitlines())
    )
    resolved_monitor_cpu_core = (
        int(cpu_core) if monitor_cpu_core is None else int(monitor_cpu_core)
    )
    if memory_poll_interval_ms <= 0:
        raise ValueError("memory_poll_interval_ms must be positive")
    lscpu = _command("lscpu", "--json")
    topology: Any = lscpu
    if lscpu:
        try:
            topology = json.loads(lscpu)
        except json.JSONDecodeError:
            pass
    facts: dict[str, Any] = {
        "schema_version": "anchor-machine-v1",
        "benchmark_path": str(executable),
        "memory_cap_bytes": int(memory_cap_bytes),
        "memory_measurement_backend": "procfs_vmrss_polling",
        "memory_poll_interval_ms": int(memory_poll_interval_ms),
        "memory_monitor_runner": str(
            pathlib.Path(__file__).resolve().parents[2] / "scripts" / "run_procfs.py"
        ),
        "timeout_seconds": OFFICIAL_TIMEOUT_SECONDS,
        "setup_timeout_seconds": OFFICIAL_TIMEOUT_SECONDS,
        "cpu_core": int(cpu_core),
        "monitor_cpu_core": resolved_monitor_cpu_core,
        "numa_node": int(numa_node),
        "numactl_path": numactl,
        "code_commit": code_commit,
        "build_sha256": executable_sha256,
        "validation_report_path": str(validation_path),
        "validation_report_sha256": hash_file(validation_path),
        "validation_report_status": validation["status"],
        "compiler_id": _command(compiler, "--version") if compiler else None,
        "linker_id": linker_id or _command("ld", "--version"),
        "build_flags": list(build_flags),
        "target_isa": target_isa or (_command(compiler, "-dumpmachine") if compiler else None),
        "third_party_commits": dict(sorted((third_party_commits or {}).items())),
        "memory_configuration": memory_configuration,
        "smt_sibling_idle_confirmed": bool(smt_sibling_idle_confirmed),
        "allocator_id": allocator_id,
        "prng_id": prng_id,
        "kernel_id": platform.release(),
        "host": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "cpu_model": _cpu_model(),
            "microcode": _read(
                f"/sys/devices/system/cpu/cpu{cpu_core}/microcode/version"
            ),
            "topology": topology,
            "governor": governor,
            "boost": _boost_state(),
            "transparent_hugepages": thp,
            "memory_total_bytes": _memory_total_bytes(),
            "swap_total_bytes": _swap_bytes(),
            "smt": {
                "active": _read("/sys/devices/system/cpu/smt/active"),
                "control": _read("/sys/devices/system/cpu/smt/control"),
                "target_cpu_thread_siblings_list": _read(
                    f"/sys/devices/system/cpu/cpu{cpu_core}/topology/thread_siblings_list"
                ),
                "target_cpu_core_id": _read(
                    f"/sys/devices/system/cpu/cpu{cpu_core}/topology/core_id"
                ),
                "target_cpu_package_id": _read(
                    f"/sys/devices/system/cpu/cpu{cpu_core}/topology/physical_package_id"
                ),
            },
            "procfs_status_readable": proc_status is not None,
            "procfs_vmrss_available": procfs_vmrss_available,
            "monitor_cpu_core_id": _read(
                f"/sys/devices/system/cpu/cpu{resolved_monitor_cpu_core}/topology/core_id"
            ),
            "monitor_cpu_package_id": _read(
                f"/sys/devices/system/cpu/cpu{resolved_monitor_cpu_core}/topology/physical_package_id"
            ),
            "glibc": platform.libc_ver(),
            "python": platform.python_version(),
        },
        "required_environment": {
            "single_software_thread": True,
            "cpu_governor": "performance",
            "turbo_boost": False,
            "transparent_hugepages": "never",
            "swap": False,
            "smt_sibling_idle": True,
            "separate_monitor_physical_core": True,
            "procfs_vmrss": True,
        },
    }
    identity = {key: value for key, value in facts.items() if key != "machine_id"}
    facts["machine_id"] = stable_hash("machine-manifest-v1", identity)[:16].hex()
    return facts


def environment_violations(manifest: dict[str, Any]) -> list[str]:
    """Report publication-protocol mismatches without mutating the host."""

    host = manifest.get("host", {})
    violations: list[str] = []
    if host.get("governor") != "performance":
        violations.append("CPU governor is not 'performance'")
    if host.get("boost") != "disabled":
        violations.append("Turbo/boost is not confirmed disabled")
    thp = str(host.get("transparent_hugepages") or "")
    if "[never]" not in thp and thp != "never":
        violations.append("transparent huge pages are not 'never'")
    if host.get("swap_total_bytes") != 0:
        violations.append("host swap is not confirmed disabled")
    if not host.get("procfs_vmrss_available"):
        violations.append("/proc/self/status VmRSS is unavailable")
    if manifest.get("monitor_cpu_core") == manifest.get("cpu_core"):
        violations.append("memory monitor CPU core is not separate from benchmark CPU core")
    target_core = host.get("smt", {}).get("target_cpu_core_id")
    target_package = host.get("smt", {}).get("target_cpu_package_id")
    if (
        target_core is not None
        and target_package is not None
        and host.get("monitor_cpu_core_id") == target_core
        and host.get("monitor_cpu_package_id") == target_package
    ):
        violations.append("memory monitor shares the benchmark physical core")
    if not manifest.get("numactl_path"):
        violations.append("numactl is unavailable, so CPU/NUMA binding cannot be enforced")
    if not str(manifest.get("memory_configuration", "")).strip():
        violations.append("memory frequency/channel configuration was not recorded")
    if manifest.get("smt_sibling_idle_confirmed") is not True:
        violations.append("SMT sibling idleness was not explicitly confirmed")
    return violations
