#!/usr/bin/env python3
"""Build an SM 8.9 scalar-FP32/DRAM Roofline from existing NCU reports only.

Each report must contain exactly one captured kernel implementing the entire
declared M x N x K SGEMM. The hardware Roofline uses NCU's instruction counters;
2*M*N*K is recorded separately as useful GEMM work. No GPU workload is launched.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
# Current x->row naive baseline. Older x->column reports remain historical inputs.
DEFAULT_BASELINE = ROOT / "results/naive-baseline-20261017"
METRICS = {
    "gpu__time_duration.sum": "ns",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained": "inst/cycle",
    "sm__cycles_elapsed.avg.per_second": "hz",
    "dram__bytes.sum.peak_sustained": "byte/cycle",
    "dram__cycles_elapsed.avg.per_second": "hz",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum.per_cycle_elapsed": "inst/cycle",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum.per_cycle_elapsed": "inst/cycle",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum.per_cycle_elapsed": "inst/cycle",
    "smsp__cycles_elapsed.avg.per_second": "hz",
    "dram__bytes.sum.per_second": "byte/s",
    "dram__bytes_read.sum": "byte",
    "dram__bytes_write.sum": "byte",
    "device__attribute_compute_capability_major": "",
    "device__attribute_compute_capability_minor": "",
}


def find_ncu(explicit):
    if explicit:
        candidates = [Path(explicit)]
    else:
        candidates = []
        for name in ("ncu.exe", "ncu", "ncu.bat"):
            found = shutil.which(name)
            if found:
                path = Path(found)
                candidates.extend([
                    path,
                    path.parent / "target/windows-desktop-win7-x64/ncu.exe",
                ])
        candidates.append(Path(
            r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2026.1.1"
            r"\target\windows-desktop-win7-x64\ncu.exe"
        ))
    for path in candidates:
        # Execute the binary directly; batch wrappers require a shell on Windows.
        if path.is_file() and path.suffix.lower() not in (".bat", ".cmd"):
            return path.resolve()
    raise ValueError("Cannot find ncu executable; pass --ncu PATH_TO_NCU_EXE.")


def run_ncu(executable, arguments, env):
    completed = subprocess.run(
        [str(executable), "--config-file", "off", *arguments],
        env=env, capture_output=True, text=True, encoding="utf-8", errors="replace",
        check=False, timeout=180,
    )
    if completed.returncode:
        raise ValueError(f"ncu failed ({completed.returncode}): "
                         f"{completed.stderr.strip() or completed.stdout.strip()}")
    return completed.stdout


def parse_raw_csv(output):
    rows = [row for row in csv.reader(io.StringIO(output.lstrip("\ufeff"))) if row]
    start = next((i for i, row in enumerate(rows)
                  if "Kernel Name" in row and "gpu__time_duration.sum" in row), None)
    if start is None:
        raise ValueError("Missing NCU raw CSV header or gpu__time_duration.sum.")
    header, *body = rows[start:]
    if len(body) != 2:
        raise ValueError("Expected one units row and exactly one captured kernel; "
                         f"found {max(0, len(body) - 1)} data rows. "
                         "Multi-kernel GEMMs cannot reuse 2*M*N*K for every row.")
    units, values = body
    if len(units) != len(header) or len(values) != len(header):
        raise ValueError("NCU CSV columns do not align.")
    if len(set(header)) != len(header):
        raise ValueError("NCU CSV contains duplicate columns.")
    columns = dict(zip(header, values))
    unit_columns = dict(zip(header, units))
    metrics = {}
    for name, expected_unit in METRICS.items():
        if name not in columns:
            raise ValueError(f"Missing required metric: {name}")
        unit = unit_columns[name].strip()
        if unit != expected_unit:
            raise ValueError(f"Unexpected unit for {name}: {unit!r}, "
                             f"expected {expected_unit!r} (use --print-units base).")
        raw_value = columns[name].strip()
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError as exc:
            raise ValueError(f"Invalid metric {name}: {raw_value!r}") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Non-finite or negative metric {name}: {raw_value!r}")
        metrics[name] = {"value": value, "unit": unit, "raw_value": raw_value}
    return columns, metrics


def profiler_command(session):
    """Keep only Launch Settings / Profiler Command Line, never Session Info."""
    launch = re.search(r"(?m)^Launch Settings\s*$", session)
    if not launch:
        return None
    lines = session[launch.end():].splitlines()
    result = []
    continuation_column = None
    for line in lines:
        if line.startswith("Profiler Command Line"):
            continuation_column = len("Profiler Command Line ")
            result.append(line[continuation_column:].rstrip())
        elif result:
            if not line.strip() or not line[:continuation_column].isspace():
                break
            result.append(line[continuation_column:].rstrip())
    return "\n".join(result) if result else None


def check_shape(command, shape):
    # NCU wraps even in the middle of an option, e.g. '-' followed by '-mode'.
    # Ordinary whitespace joins suffice for the M/N/K options in these reports.
    normalized = re.sub(r"-\s*\n\s*-", "--", command or "")
    found = {}
    for name, expected in shape.items():
        matches = re.findall(rf"(?<!\S)--{name}(?:\s+|=)(\d+)(?=\s|$)", normalized)
        if matches:
            observed = {int(value) for value in matches}
            if observed != {expected}:
                raise ValueError(f"Declared {name}={expected} disagrees with "
                                 f"report command values {sorted(observed)}.")
            found[name] = expected
    return {"matched_dimensions": found,
            "status": "matched_report_command" if len(found) == 3 else "partially_or_not_recorded",
            "assumption": "The captured kernel implements the complete declared SGEMM."}


def analyze(label, path, shape, ncu, env):
    if not path.is_file():
        raise ValueError(f"Report does not exist: {path}")
    print(f"Reading {label}: {path.name}", flush=True)
    columns, metrics = parse_raw_csv(run_ncu(ncu, [
        "--import", str(path), "--page", "raw", "--csv", "--print-units", "base",
    ], env))
    command = profiler_command(run_ncu(ncu, ["--import", str(path), "--page", "session"], env))
    value = lambda name: metrics[name]["value"]
    cc = (value("device__attribute_compute_capability_major"),
          value("device__attribute_compute_capability_minor"))
    if cc != (8, 9):
        raise ValueError(f"This scalar FP32 model is validated for SM 8.9 only; got {cc}.")
    for name in ("Kernel Name", "Block Size", "Grid Size", "device__attribute_display_name"):
        if not columns.get(name, "").strip():
            raise ValueError(f"Missing report metadata: {name}")
    compute = (2 * value("sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained")
               * value("sm__cycles_elapsed.avg.per_second"))
    bandwidth = (value("dram__bytes.sum.peak_sustained")
                 * value("dram__cycles_elapsed.avg.per_second"))
    hw_perf = sum(
        factor * value(f"smsp__sass_thread_inst_executed_op_{op}_pred_on.sum.per_cycle_elapsed")
        for op, factor in (("fadd", 1), ("fmul", 1), ("ffma", 2))
    ) * value("smsp__cycles_elapsed.avg.per_second")
    duration_s = value("gpu__time_duration.sum") / 1e9
    dram_rate = value("dram__bytes.sum.per_second")
    dram_read = value("dram__bytes_read.sum")
    dram_write = value("dram__bytes_write.sum")
    for name, number in (("compute peak", compute), ("bandwidth peak", bandwidth),
                         ("FP32 performance", hw_perf), ("duration", duration_s),
                         ("DRAM rate", dram_rate), ("DRAM bytes", dram_read + dram_write)):
        if not math.isfinite(number) or number <= 0:
            raise ValueError(f"{name} must be finite and positive; got {number}.")
    flop = 2 * shape["m"] * shape["n"] * shape["k"]
    with path.open("rb") as handle:
        digest = hashlib.file_digest(handle, "sha256").hexdigest()
    return {
        "label": label,
        "report": {"path": str(path), "sha256": digest,
                   "mtime_utc": datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat(),
                   "source_commit": None,
                   "source_commit_note": "Not recovered from the report; current Git is not substituted. See capture logs for the build fingerprint.",
                   "profiler_command_line_wrapped": command},
        "kernel": columns["Kernel Name"], "block": columns["Block Size"],
        "grid": columns["Grid Size"], "gpu": columns["device__attribute_display_name"],
        "compute_capability": "8.9", "shape_check": check_shape(command, shape),
        "metrics": metrics,
        "derived": {
            "duration_ms": duration_s * 1e3,
            "sm_clock_ghz": value("sm__cycles_elapsed.avg.per_second") / 1e9,
            "dram_clock_ghz": value("dram__cycles_elapsed.avg.per_second") / 1e9,
            "fp32_peak_gflops": compute / 1e9,
            "dram_peak_gbytes_per_second": bandwidth / 1e9,
            "ridge_ai_flop_per_byte": compute / bandwidth,
            "hw_gflops": hw_perf / 1e9,
            "hw_ai_flop_per_byte": hw_perf / dram_rate,
            "hw_peak_percent": hw_perf / compute * 100,
            "dram_bytes_per_second": dram_rate,
            "dram_read_bytes": int(dram_read), "dram_write_bytes": int(dram_write),
            "dram_total_bytes": int(dram_read + dram_write),
            "effective_gflops": flop / duration_s / 1e9,
            "effective_ai_flop_per_byte": flop / (dram_read + dram_write),
        },
    }


def plot_model(model, output):
    cache = ROOT / "build/matplotlib-cache"
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    reports = model["reports"]
    points = [sample["derived"] for sample in reports]
    xmin = min(0.25, min(p["hw_ai_flop_per_byte"] for p in points) / 4)
    xmax = max(1024, max(p["hw_ai_flop_per_byte"] for p in points) * 3)
    ymin = min(10, min(p["hw_gflops"] for p in points) / 4)
    ymax = max(20000, max(p["fp32_peak_gflops"] for p in points) * 1.6)
    xs = np.geomspace(xmin, xmax, 400)
    # Keep detailed labels outside the plot when comparing several kernels.
    # Nearby points/ceilings can otherwise hide each other's annotations.
    comparison = len(reports) > 2
    fig, ax = plt.subplots(figsize=(14 if comparison else 11,
                                    max(7.5, 1.5 * len(reports) + 1.5)))
    fig.subplots_adjust(left=0.08 if comparison else 0.10,
                        right=0.65 if comparison else 0.96, bottom=0.20, top=0.84)
    details = fig.add_axes((0.70, 0.20, 0.28, 0.64)) if comparison else None
    if details is not None:
        details.set_axis_off()
        details.text(0, 1.02, "Measured points and sample ceilings", fontsize=11,
                     weight="bold", transform=details.transAxes)
    fig.suptitle("Scalar FP32 / DRAM Roofline", fontsize=19, y=0.97, weight="bold")
    shape = model["shape"]
    gpu = reports[0]["gpu"] if len({r["gpu"] for r in reports}) == 1 else "SM 8.9 GPUs"
    fig.text(0.5, 0.91, f"{gpu}  |  SGEMM {shape['m']} x {shape['n']} x {shape['k']}",
             ha="center", fontsize=11, color="#444444")
    colors = plt.get_cmap("tab10")
    styles = {
        "naive": ("#0072B2", "o", "-"),
        "coalesced": ("#E69F00", "^", "--"),
        "shared": ("#009E73", "D", "-."),
        "cublas": ("#CC79A7", "s", ":"),
    }
    summary = []
    for i, sample in enumerate(reports):
        d = sample["derived"]
        label = sample["label"]
        color, marker, linestyle = styles.get(
            label.lower(), (colors(i % 10), ("o", "s", "^", "D")[i % 4],
                            ("-", "--", "-.", ":")[i % 4]))
        roof = np.minimum(d["fp32_peak_gflops"], d["dram_peak_gbytes_per_second"] * xs)
        ax.plot(xs, roof, color=color, linestyle=linestyle,
                linewidth=2.2, label=f"{label} sample ceiling", zorder=2)
        ax.scatter(d["hw_ai_flop_per_byte"], d["hw_gflops"], color=color,
                   marker=marker, s=75, edgecolors="white",
                   linewidths=0.8, label=f"{label} measured", zorder=4)
        if details is None:
            ax.annotate(f"{label}: {d['hw_gflops']:,.0f} GFLOP/s\n"
                        f"AI {d['hw_ai_flop_per_byte']:.1f}; {d['hw_peak_percent']:.1f}% of FP32 peak",
                        xy=(d["hw_ai_flop_per_byte"], d["hw_gflops"]),
                        xytext=(-12, -31 if i % 2 else 10), textcoords="offset points",
                        ha="right", fontsize=9, color=color)
        else:
            y = 0.93 - i * 0.96 / len(reports)
            details.plot([0, 0.12], [y, y], color=color, linestyle=linestyle,
                         linewidth=2.2, transform=details.transAxes)
            details.scatter([0.06], [y], color=color, marker=marker, s=75,
                            edgecolors="white", linewidths=0.8,
                            transform=details.transAxes, zorder=3)
            details.text(0.16, y, label, fontsize=12, color=color, weight="bold",
                         va="center", transform=details.transAxes)
            details.annotate(
                f"Measured: {d['hw_gflops']:,.1f} GFLOP/s\n"
                f"AI: {d['hw_ai_flop_per_byte']:.1f} FLOP/byte; peak: {d['hw_peak_percent']:.1f}%\n"
                f"Ceilings: {d['fp32_peak_gflops']/1000:.2f} TFLOP/s; "
                f"{d['dram_peak_gbytes_per_second']:.1f} GB/s\n"
                f"Ridge: {d['ridge_ai_flop_per_byte']:.1f} FLOP/byte",
                xy=(0, y), xycoords="axes fraction", xytext=(0, -16),
                textcoords="offset points", va="top", fontsize=9, linespacing=1.5,
                color="#444444")
        summary.append(f"{label}: {d['fp32_peak_gflops']/1000:.2f} TFLOP/s  |  "
                       f"{d['dram_peak_gbytes_per_second']:.1f} GB/s  |  "
                       f"ridge {d['ridge_ai_flop_per_byte']:.1f} FLOP/byte")
    reference = next((sample for sample in reports if sample["label"].lower() == "cublas"), reports[0])
    ridge = reference["derived"]["ridge_ai_flop_per_byte"]
    ax.axvline(ridge, color="#888888", linestyle=":", linewidth=1, zorder=1)
    ax.annotate(f"ridge {ridge:.1f} FLOP/byte", xy=(ridge, ymin * 1.5),
                xytext=(7, 0), textcoords="offset points", fontsize=9, color="#666666")
    if comparison:
        ax.text(0.025, 0.96, "Markers: measured FP32 performance\n"
                "Lines: ceilings at each sample's recorded clocks",
                transform=ax.transAxes, va="top", fontsize=9, linespacing=1.6,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95, "pad": 7})
    else:
        ax.text(0.025, 0.96, "Ceilings at each sample's recorded clocks\n" + "\n".join(summary),
                transform=ax.transAxes, va="top", fontsize=9, linespacing=1.6,
                bbox={"facecolor": "white", "edgecolor": "#dddddd", "alpha": 0.95, "pad": 7})
    ax.set(xscale="log", yscale="log", xlim=(xmin, xmax), ylim=(ymin, ymax),
           xlabel="Hardware arithmetic intensity (FLOP / DRAM byte)",
           ylabel="Hardware FP32 performance (GFLOP/s)")
    ax.grid(True, which="major", alpha=0.24)
    ax.grid(True, which="minor", alpha=0.07)
    if not comparison:
        ax.legend(loc="lower right", fontsize=9, framealpha=0.96)
    fig.text(0.10, 0.08,
             "Theoretical ceilings at recorded clocks; DRAM traffic from NCU. No sustained microbenchmark calibration.\n"
             "Points use executed FADD + FMUL + 2 x FFMA; useful 2MNK throughput is recorded separately in model.json.",
             fontsize=9, color="#555555", linespacing=1.6)
    for suffix in ("png", "svg"):
        fig.savefig(output / f"roofline.{suffix}", dpi=180, facecolor="white")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", metavar="LABEL=PATH",
                        help="Repeat for each single-kernel report; replaces defaults.")
    for dimension in ("m", "n", "k"):
        parser.add_argument(f"--{dimension}", type=int, default=2048)
    parser.add_argument("--output", type=Path, default=DEFAULT_BASELINE / "roofline")
    parser.add_argument("--ncu", help="Path to the real ncu executable, not its .bat wrapper.")
    args = parser.parse_args()
    shape = {dimension: getattr(args, dimension) for dimension in ("m", "n", "k")}
    if any(number <= 0 for number in shape.values()):
        parser.error("M, N and K must be positive.")
    inputs = args.report or [f"naive={DEFAULT_BASELINE / 'naive-p1.ncu-rep'}",
                             f"cuBLAS={DEFAULT_BASELINE / 'cublas-p1.ncu-rep'}"]
    reports = []
    for item in inputs:
        label, separator, path = item.partition("=")
        if not separator or not label.strip() or not path.strip():
            parser.error("--report must be LABEL=PATH.")
        if label in [previous[0] for previous in reports]:
            parser.error(f"Duplicate report label: {label}")
        reports.append((label, Path(path).resolve()))
    ncu = find_ncu(args.ncu)
    temp = ROOT / "build/profiler-temp"
    temp.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(TEMP=str(temp), TMP=str(temp), NV_COMPUTE_PROFILER_DISABLE_STOCK_FILE_DEPLOYMENT="1")
    model = {
        "schema_version": 1,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "import_tool": {"path": str(ncu), "version": run_ncu(ncu, ["--version"], env).strip()},
        "shape": shape, "useful_gemm_flop": 2 * shape["m"] * shape["n"] * shape["k"],
        "method": {
            "reference": "Nsight Compute 2026.1.1 SpeedOfLight_RooflineChart.section scalar FP32 / DRAM formulas",
            "compute_peak": "2 * sm__sass_thread_inst_executed_op_ffma_pred_on.sum.peak_sustained * sm__cycles_elapsed.avg.per_second",
            "bandwidth_peak": "dram__bytes.sum.peak_sustained * dram__cycles_elapsed.avg.per_second",
            "hw_performance": "(FADD + FMUL + 2*FFMA instruction sum.per_cycle_elapsed) * smsp__cycles_elapsed.avg.per_second",
            "hw_arithmetic_intensity": "hw_performance / dram__bytes.sum.per_second",
            "effective_performance": "2*M*N*K / gpu__time_duration.sum_seconds",
            "effective_arithmetic_intensity": "2*M*N*K / (dram__bytes_read.sum + dram__bytes_write.sum)",
        },
        "limitations": [
            "Theoretical ceilings use each report's recorded clocks; no sustained compute or bandwidth microbenchmark calibration.",
            "This is a scalar FP32 / DRAM model for SM 8.9; it does not model Tensor Cores, L1 or L2 roofs.",
            "Hardware FP32 instructions and useful GEMM operations differ; use matching work definitions for intensity and performance.",
            "Each input must be one kernel performing the entire declared SGEMM; one captured row alone does not establish this for arbitrary applications.",
            "NCU replay, cache policy and clock control can differ from normal benchmark timing; recorded clocks are not proof of successful clock locking.",
            "Raw metric CSV values are rounded by NCU; derived results inherit that precision.",
            "Reports do not embed the source commit; use capture logs and source fingerprints to identify the measured implementation.",
        ],
        "reports": [analyze(label, path, shape, ncu, env) for label, path in reports],
    }
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plot_model(model, output)
    (output / "model.json").write_text(json.dumps(model, ensure_ascii=False, indent=2,
                                                  allow_nan=False) + "\n", encoding="utf-8")
    print(f"Saved {output / 'model.json'}")
    print(f"Saved {output / 'roofline.png'} and roofline.svg")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.TimeoutExpired, ImportError) as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
