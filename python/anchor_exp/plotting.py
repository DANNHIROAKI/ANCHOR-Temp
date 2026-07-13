"""Publication-oriented plots for the twelve experiment sweeps."""

from __future__ import annotations

import pathlib
from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .aggregate import aggregate_records
from .preprocess.common import OptionalDependencyError


ALGORITHM_ORDER = ("AC", "AS", "SweepRT", "LiftedRT")
COLORS = {
    "AC": "#0072B2",
    "AS": "#009E73",
    "SweepRT": "#D55E00",
    "LiftedRT": "#CC79A7",
}
MARKERS = {"AC": "o", "AS": "s", "SweepRT": "^", "LiftedRT": "D"}


def _matplotlib() -> Any:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise OptionalDependencyError(
            "plotting requires matplotlib; install requirements-plots.txt"
        ) from exc
    return plt


def _numeric(value: Any) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if np.isfinite(result) else None


def _group_metric(
    rows: Sequence[Mapping[str, Any]], experiment_id: str, metrics: set[str]
) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
    result: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["experiment_id"] == experiment_id and row["metric"] in metrics:
            result[(str(row["algorithm"]), str(row["metric"]))].append(row)
    for values in result.values():
        values.sort(key=lambda row: float(row["x_value"]))
    return result


def _failure_points(
    statuses: Sequence[Mapping[str, Any]], experiment_id: str, mode: str
) -> dict[str, list[tuple[float, str]]]:
    result: dict[str, list[tuple[float, str]]] = defaultdict(list)
    for row in statuses:
        if (
            row["experiment_id"] != experiment_id
            or row["measurement_mode"] != mode
            or row["task"] != "oneshot"
        ):
            continue
        for status, count in row["statuses"].items():
            if status in {"OOM", "MEMORY-CAP-EXCEEDED", "TO"} and count:
                result[str(row["algorithm"])].append((float(row["x_value"]), status))
    return result


def _save(fig: Any, base: pathlib.Path, formats: Sequence[str]) -> list[pathlib.Path]:
    output: list[pathlib.Path] = []
    for suffix in formats:
        path = base.with_suffix(f".{suffix}")
        fig.savefig(path, dpi=240, bbox_inches="tight")
        output.append(path)
    return output


def _configure_axis(ax: Any, *, x_name: str, ylabel: str, log_y: bool = True) -> None:
    ax.set_xlabel(x_name)
    ax.set_ylabel(ylabel)
    if x_name in {"N", "t", "alpha_target", "alpha_realized"}:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    ax.grid(True, which="both", linewidth=0.4, alpha=0.35)


def _plot_metric_panel(
    ax: Any,
    groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]],
    *,
    metrics: Sequence[str],
    x_override: Mapping[tuple[str, Any], float] | None = None,
) -> None:
    line_styles = ["-", "--", ":"]
    for algorithm in ALGORITHM_ORDER:
        for metric_index, metric in enumerate(metrics):
            rows = groups.get((algorithm, metric), ())
            if not rows:
                continue
            x = []
            y = []
            for row in rows:
                key = (algorithm, row["x_value"])
                x_value = x_override.get(key) if x_override is not None else None
                if x_value is None:
                    x_value = float(row["x_value"])
                x.append(x_value)
                y.append(float(row["value"]))
            label = algorithm if len(metrics) == 1 else f"{algorithm} · {metric}"
            ax.plot(
                x,
                y,
                color=COLORS[algorithm],
                marker=MARKERS[algorithm],
                linestyle=line_styles[metric_index % len(line_styles)],
                linewidth=1.7,
                markersize=4.5,
                label=label,
            )


def _realized_alpha_map(records: Sequence[Mapping[str, Any]], experiment_id: str) -> dict[tuple[str, Any], float]:
    values: dict[tuple[str, Any], list[float]] = defaultdict(list)
    for record in records:
        if record.get("experiment_id") != experiment_id or record.get("status") != "OK":
            continue
        alpha = _numeric(record.get("alpha_realized"))
        if alpha is None:
            continue
        values[(str(record["algorithm"]), record["x_value"])].append(alpha)
    result: dict[tuple[str, Any], float] = {}
    for key, items in values.items():
        first = items[0]
        if any(item != first for item in items[1:]):
            raise ValueError(
                "inconsistent alpha_realized across the single time/memory pair "
                f"for {experiment_id} {key}"
            )
        result[key] = first
    return result


FAILURE_MARKERS = {"OOM": "X", "MEMORY-CAP-EXCEEDED": "D", "TO": "P"}


def _main_plots(
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    output_dir: pathlib.Path,
    formats: Sequence[str],
) -> list[pathlib.Path]:
    plt = _matplotlib()
    rows = aggregate["metrics"]
    statuses = aggregate["statuses"]
    experiment_ids = sorted({str(row["experiment_id"]) for row in rows})
    output: list[pathlib.Path] = []

    def display_title(experiment: str) -> str:
        for record in records:
            if record.get("experiment_id") == experiment:
                if record.get("dataset_id") == "CMAB-1M":
                    return f"{experiment} · N={int(record['N_total']):,}"
                break
        return experiment

    def completed_x(experiment: str, mode: str) -> set[float]:
        state: dict[float, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
        for row in statuses:
            if (
                row["experiment_id"] == experiment
                and row["measurement_mode"] == mode
                and row["task"] == "oneshot"
            ):
                state[float(row["x_value"])][str(row["algorithm"])].update(
                    row["statuses"]
                )
        return {
            x
            for x, algorithms in state.items()
            if set(algorithms) == set(ALGORITHM_ORDER)
            and all(status_set == {"OK"} for status_set in algorithms.values())
        }

    def restricted_groups(
        groups: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]], allowed: set[float]
    ) -> dict[tuple[str, str], list[Mapping[str, Any]]]:
        return {
            key: [row for row in values if float(row["x_value"]) in allowed]
            for key, values in groups.items()
        }

    for experiment_id in experiment_ids:
        experiment_rows = [row for row in rows if row["experiment_id"] == experiment_id]
        if not experiment_rows:
            continue
        x_name = str(experiment_rows[0]["x_name"])

        time_groups = _group_metric(rows, experiment_id, {"OneShotTime"})
        if time_groups:
            fig, ax = plt.subplots(figsize=(6.2, 4.0))
            _plot_metric_panel(ax, time_groups, metrics=("OneShotTime",))
            failures = _failure_points(statuses, experiment_id, "time")
            successful = [float(row["value"]) for row in experiment_rows if row["metric"] == "OneShotTime"]
            marker_y = max(successful) * 1.25 if successful else 1.0
            for algorithm, points in failures.items():
                for x_value, status in points:
                    ax.scatter(
                        [x_value],
                        [marker_y],
                        color=COLORS.get(algorithm, "black"),
                        marker=FAILURE_MARKERS[status],
                        s=45,
                        zorder=5,
                    )
            _configure_axis(ax, x_name=x_name, ylabel="One-shot time (s)")
            ax.legend(fontsize=8)
            ax.set_title(display_title(experiment_id))
            output.extend(_save(fig, output_dir / f"{experiment_id}-time", formats))
            plt.close(fig)

            if x_name == "d":
                allowed = completed_x(experiment_id, "time")
                if allowed:
                    fig, ax = plt.subplots(figsize=(6.2, 4.0))
                    _plot_metric_panel(
                        ax,
                        restricted_groups(time_groups, allowed),
                        metrics=("OneShotTime",),
                    )
                    _configure_axis(ax, x_name="d", ylabel="One-shot time (s)")
                    ax.legend(fontsize=8)
                    ax.set_title(f"{experiment_id} · all algorithms completed")
                    output.extend(
                        _save(fig, output_dir / f"{experiment_id}-time-common", formats)
                    )
                    plt.close(fig)

            alpha_map = _realized_alpha_map(records, experiment_id)
            if alpha_map and x_name in {"level", "alpha_target"}:
                fig, ax = plt.subplots(figsize=(6.2, 4.0))
                _plot_metric_panel(
                    ax,
                    time_groups,
                    metrics=("OneShotTime",),
                    x_override=alpha_map,
                )
                _configure_axis(ax, x_name="alpha_realized", ylabel="One-shot time (s)")
                ax.legend(fontsize=8)
                ax.set_title(f"{display_title(experiment_id)} · realized density")
                output.extend(
                    _save(fig, output_dir / f"{experiment_id}-time-vs-alpha-realized", formats)
                )
                plt.close(fig)

        memory_metrics = {"PeakMemoryTotal", "PeakMemoryIncremental"}
        memory_groups = _group_metric(rows, experiment_id, memory_metrics)
        if memory_groups:
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            _plot_metric_panel(
                ax,
                memory_groups,
                metrics=("PeakMemoryTotal", "PeakMemoryIncremental"),
            )
            memory_failures = _failure_points(statuses, experiment_id, "memory")
            successful_memory = [
                float(row["value"])
                for row in experiment_rows
                if row["metric"] in memory_metrics
            ]
            marker_y = max(successful_memory) * 1.25 if successful_memory else 1.0
            for algorithm, points in memory_failures.items():
                for x_value, status in points:
                    ax.scatter(
                        [x_value],
                        [marker_y],
                        color=COLORS.get(algorithm, "black"),
                        marker=FAILURE_MARKERS[status],
                        s=45,
                        zorder=5,
                    )
            _configure_axis(ax, x_name=x_name, ylabel="Peak memory (bytes)")
            ax.legend(fontsize=7, ncol=2)
            ax.set_title(display_title(experiment_id))
            output.extend(_save(fig, output_dir / f"{experiment_id}-memory", formats))
            plt.close(fig)
            alpha_map = _realized_alpha_map(records, experiment_id)
            if alpha_map and x_name in {"level", "alpha_target"}:
                fig, ax = plt.subplots(figsize=(6.6, 4.2))
                _plot_metric_panel(
                    ax,
                    memory_groups,
                    metrics=("PeakMemoryTotal", "PeakMemoryIncremental"),
                    x_override=alpha_map,
                )
                _configure_axis(ax, x_name="alpha_realized", ylabel="Peak memory (bytes)")
                ax.legend(fontsize=7, ncol=2)
                ax.set_title(f"{display_title(experiment_id)} · realized density")
                output.extend(
                    _save(
                        fig,
                        output_dir / f"{experiment_id}-memory-vs-alpha-realized",
                        formats,
                    )
                )
                plt.close(fig)
            if x_name == "d":
                allowed = completed_x(experiment_id, "memory")
                if allowed:
                    fig, ax = plt.subplots(figsize=(6.6, 4.2))
                    _plot_metric_panel(
                        ax,
                        restricted_groups(memory_groups, allowed),
                        metrics=("PeakMemoryTotal", "PeakMemoryIncremental"),
                    )
                    _configure_axis(ax, x_name="d", ylabel="Peak memory (bytes)")
                    ax.legend(fontsize=7, ncol=2)
                    ax.set_title(f"{experiment_id} · all algorithms completed")
                    output.extend(
                        _save(fig, output_dir / f"{experiment_id}-memory-common", formats)
                    )
                    plt.close(fig)

        if x_name == "shape_sigma":
            quantile_values: dict[str, dict[float, list[float]]] = defaultdict(
                lambda: defaultdict(list)
            )
            for record in records:
                if record.get("experiment_id") != experiment_id:
                    continue
                quantiles = record.get("aspect_ratio_quantiles")
                if not isinstance(quantiles, Mapping):
                    continue
                for name, value in quantiles.items():
                    numeric = _numeric(value)
                    if numeric is not None:
                        quantile_values[str(name)][float(record["x_value"])].append(numeric)
            if quantile_values:
                fig, ax = plt.subplots(figsize=(6.2, 4.0))
                for name, by_x in sorted(quantile_values.items()):
                    x = sorted(by_x)
                    y = []
                    for item in x:
                        values_at_x = by_x[item]
                        first = values_at_x[0]
                        if any(value != first for value in values_at_x[1:]):
                            raise ValueError(
                                "inconsistent aspect-ratio metadata across the single "
                                f"time/memory pair for {experiment_id} x={item}"
                            )
                        y.append(first)
                    ax.plot(x, y, marker="o", label=name)
                _configure_axis(
                    ax,
                    x_name="shape_sigma",
                    ylabel="Realized max/min side ratio",
                    log_y=True,
                )
                ax.legend(fontsize=8)
                ax.set_title(f"{experiment_id} · realized shape")
                output.extend(
                    _save(fig, output_dir / f"{experiment_id}-shape-quantiles", formats)
                )
                plt.close(fig)
    return output


def _auxiliary_plots(
    records: Sequence[Mapping[str, Any]],
    aggregate: Mapping[str, Any],
    output_dir: pathlib.Path,
    formats: Sequence[str],
) -> list[pathlib.Path]:
    plt = _matplotlib()
    rows = aggregate["metrics"]
    output: list[pathlib.Path] = []
    titles: dict[str, str] = {}
    for record in records:
        experiment_id = str(record.get("experiment_id"))
        if record.get("dataset_id") == "CMAB-1M":
            titles[experiment_id] = f"{experiment_id} · N={int(record['N_total']):,}"
    derived_rows: list[dict[str, Any]] = []
    for row in rows:
        if row["metric"] != "QueryTime" or float(row["value"]) <= 0:
            continue
        derived = dict(row)
        derived["metric"] = "Throughput"
        sample_count = float(row["t"])
        derived["value"] = sample_count / float(row["value"])
        derived_rows.append(derived)
    rows = [*rows, *derived_rows]
    panels = {
        "stages": ("PrepareTime", "QueryTime"),
        "prepared-query": ("PreparedQueryTime",),
        "count-only": ("CountOnlyTime",),
        "aux-memory": ("PeakMemoryAux", "MemoryAfterPrepare"),
        "throughput": ("Throughput",),
    }
    for experiment_id in sorted({str(row["experiment_id"]) for row in rows}):
        source_rows = [row for row in rows if row["experiment_id"] == experiment_id]
        if not source_rows:
            continue
        x_name = str(source_rows[0]["x_name"])
        for panel, metrics in panels.items():
            groups = _group_metric(rows, experiment_id, set(metrics))
            if not groups:
                continue
            fig, ax = plt.subplots(figsize=(6.6, 4.2))
            _plot_metric_panel(ax, groups, metrics=metrics)
            if panel == "throughput":
                ylabel = "Samples / second"
            else:
                ylabel = "Memory (bytes)" if "memory" in panel else "Time (s)"
            _configure_axis(ax, x_name=x_name, ylabel=ylabel)
            ax.legend(fontsize=7, ncol=2)
            ax.set_title(f"{titles.get(experiment_id, experiment_id)} · {panel}")
            output.extend(_save(fig, output_dir / f"{experiment_id}-{panel}", formats))
            plt.close(fig)
    return output


def plot_results(
    records: Sequence[Mapping[str, Any]],
    output_dir: str | pathlib.Path,
    *,
    formats: Sequence[str] = ("png", "pdf"),
) -> list[pathlib.Path]:
    destination = pathlib.Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    aggregate = aggregate_records(records)
    return [
        *_main_plots(records, aggregate, destination, formats),
        *_auxiliary_plots(records, aggregate, destination, formats),
    ]
