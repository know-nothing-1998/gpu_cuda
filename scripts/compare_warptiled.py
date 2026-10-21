"""Compare warp tiling with the unchanged tuned kernel and strict FP32 cuBLAS.

Builds isolated source snapshots, checks every candidate, and confirms the best
screened candidates twice in reversed order. No production default is changed.
Warp candidates have eight inputs; each WM-by-WN output tile is assigned to one warp.
The block shape is (BN/(TN*WNITER), BM/(TM*WMITER)), where WMITER is derived
from the warp tile. There is no independent thread-count input.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import sys

sys.dont_write_bytecode = True
from autotune import Config, load_shapes, reject_reason, sha256, write_json, write_matrix
from bench_tuned import Runner, CommandError, compiler_environment
from experiment_sources import snapshot_sources

ROOT = Path(__file__).resolve().parent.parent
WARP_PARAMS = ("BM", "BN", "BK", "WM", "WN", "TM", "TN", "WNITER")
LOCAL_TIME = timezone(timedelta(hours=8))


@dataclass(frozen=True)
class WarpConfig:
    BM: int
    BN: int
    BK: int
    WM: int
    WN: int
    TM: int
    TN: int
    WNITER: int

    @property
    def THREADS(self):
        return 32 * (self.BM // self.WM) * (self.BN // self.WN)

    def values(self):
        return {key: getattr(self, key) for key in WARP_PARAMS}

    @property
    def name(self):
        return "warp_" + "_".join(f"{key.lower()}{value}" for key, value in self.values().items())

    @property
    def canonical(self):
        inputs = ";".join(f"{key}={value}" for key, value in self.values().items())
        return inputs + f";THREADS={self.THREADS}"

    @property
    def shared_bytes(self):
        return 4 * self.BK * (self.BM + self.BN)

    def cmake_flags(self):
        return [f"-DGEMM_WARP_{key}={value}" for key, value in self.values().items()]


def warp_reject_reason(config):
    if any(type(value) is not int or not 1 <= value <= 12288 for value in config.values().values()):
        return "all warp parameters must be positive integers <= 12288"
    if config.BM % config.WM or config.BN % config.WN:
        return "warp tiles must divide block tiles"
    if config.BK % 4 or config.BN % 4:
        return "BK and BN must be multiples of four"
    if config.THREADS % 32 or config.THREADS > 1024:
        return "derived thread count must be a multiple of 32 and at most 1024"
    if config.shared_bytes > 49152:
        return "static shared memory exceeds 48 KiB"
    denominator = 32 * config.TM * config.TN * config.WNITER
    if (config.WM * config.WN) % denominator or config.WM * config.WN < denominator:
        return "WMITER must be a positive integer"
    wmiter = config.WM * config.WN // denominator
    if (config.WM % wmiter or config.WN % config.WNITER
            or (config.WM // wmiter) % config.TM
            or (config.WN // config.WNITER) % config.TN):
        return "warp subtiles must divide evenly among lanes"
    return None


def load_plan(space, matrix, cases, max_candidates=None, *, shapes=None):
    spec = json.loads(space.read_text(encoding="utf-8-sig"))
    if tuple(spec["parameter_order"]) != WARP_PARAMS:
        raise ValueError("unexpected warp parameter order")
    configs = list(dict.fromkeys(WarpConfig(*values) for values in spec["candidates"]))
    if not configs:
        raise ValueError("no warp candidates")
    for config in configs:
        reason = warp_reject_reason(config)
        if reason:
            raise ValueError(f"{config.name}: {reason}")
    if max_candidates:
        configs = configs[:max_candidates]
    shapes = load_shapes(matrix, cases) if shapes is None else list(shapes)
    if not shapes:
        raise ValueError("no matrix cases selected")
    default = Config(*spec["tuned_default"])
    previous = spec.get("tuned_previous_winners", {})
    winners = {shape["id"]: Config(*previous[shape["id"]]) if shape["id"] in previous else default
               for shape in shapes}
    for config in [default, *winners.values()]:
        reason = reject_reason(config)
        if reason:
            raise ValueError(f"{config.name}: {reason}")
    return configs, default, winners, shapes


def read_measurement(path, kernel, params, shape, source_hash, warmup, rounds):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 2 or {row["actual_kernel"] for row in rows} != {kernel, "cublas"}:
        raise ValueError("expected one custom-kernel row and one cuBLAS row")
    result = {}
    for row in rows:
        if row["status"] != "valid" or row["requested_kernel"] != kernel:
            raise ValueError("invalid benchmark status or requested kernel")
        if any(int(row[key]) for key in ("failed", "nonfinite", "cpu_failed")):
            raise ValueError("numerical validation failed")
        if int(row["cpu_compared"]) <= 0:
            raise ValueError("missing independent CPU reference")
        if row["actual_kernel"] == kernel:
            if row["reference_scope"] != "cublas_full" or int(row["compared"]) != shape["m"] * shape["n"]:
                raise ValueError("missing full comparison against cuBLAS")
            if kernel == "warptiled":
                if row.get("warptiled_config") != WarpConfig(**params).canonical:
                    raise ValueError("CSV warp parameters do not match the compiled configuration")
            elif any(row.get("tuned_" + key.lower()) != str(value) for key, value in params.items()):
                raise ValueError("CSV tuned parameters do not match the compiled configuration")
        if row["source_sha256"] != source_hash or row["build_type"] != "Release":
            raise ValueError("source fingerprint or build type mismatch")
        if row["compute_type"] != "CUBLAS_COMPUTE_32F_PEDANTIC":
            raise ValueError("cuBLAS precision mismatch")
        if any(row[key] != "fp32" for key in ("input_dtype", "accum_dtype", "output_dtype")):
            raise ValueError("dtype mismatch")
        expected = {key: shape[key] for key in ("m", "n", "k", "seed", "iters")}
        expected.update(warmup=warmup, rounds=rounds)
        if any(int(row[key]) != value for key, value in expected.items()):
            raise ValueError("benchmark dimensions, seed or timing counts do not match")
        samples = [float(value) for value in row["round_mean_ms"].split(";")]
        if len(samples) != rounds or any(not math.isfinite(value) or value <= 0 for value in samples):
            raise ValueError("invalid timing samples")
        median = statistics.median(samples)
        if not math.isclose(median, float(row["median_ms"]), rel_tol=1e-8):
            raise ValueError("CSV median disagrees with individual timing rounds")
        result[row["actual_kernel"]] = {"samples_ms": samples, "median_ms": median}
    return result


class Experiment:
    def __init__(self, args, configs, default, winners, shapes):
        self.args, self.configs, self.default = args, configs, default
        self.winners, self.shapes = winners, shapes
        stamp = datetime.now(LOCAL_TIME).strftime("%Y%m%d-%H%M%S-%f")
        prefix = "autotune-warptiled-" if getattr(args, "kernel", None) == "warptiled" else "warptiling-"
        self.out = (Path(args.output) if args.output else ROOT / "results" / (prefix + stamp)).resolve()
        self.work = ROOT / "build" / (prefix + stamp)
        self.source, self.build = self.work / "source", self.work / "cmake-build"
        self.out.mkdir(parents=True, exist_ok=False)
        self.work.mkdir(parents=True, exist_ok=False)
        self.runner = Runner(self.out, args.timeout)
        self.variants, self.records = {}, []
        self.manifest = {
            "status": "running", "kernel": "warptiled", "started": datetime.now(LOCAL_TIME).isoformat(),
            "work_dir": str(self.work), "matrix": shapes, "architecture": args.arch,
            "candidates": [config.values() for config in configs],
            "tuned_default": default.values(),
            "tuned_previous_winners": {case: config.values() for case, config in winners.items()},
            "tuned_reference_policy": "Use the recorded tuned winner for known IDs; otherwise use tuned_default.",
            "screen_rounds": args.screen_rounds, "confirmation_passes": 2,
            "confirmation_rounds": "matrix rounds", "warmup_min": 100,
            "ranking_rule": "median of round means from two complete confirmation passes",
            "source_snapshot": "source-snapshot.zip", "source_hashes": "source-sha256.json",
            "experiment_scope": "whole-kernel comparison; includes different shared layout and thread mapping",
        }
        self.checkpoint()

    def checkpoint(self):
        write_json(self.out / "manifest.json", {**self.manifest, "variants": self.variants})
        write_json(self.out / "measurements.json", self.records)

    def run(self, argv, label):
        (self.out / label).parent.mkdir(parents=True, exist_ok=True)
        return self.runner.run(argv, label).decode("utf-8", errors="replace")

    def telemetry(self, label):
        try:
            self.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit",
                      "--format=csv"], "telemetry/" + label)
        except CommandError:
            if self.runner.timed_out:
                raise

    def prepare(self):
        print("Results: " + str(self.out), flush=True)
        self.hashes = snapshot_sources(ROOT, self.source, self.out / "source-snapshot.zip")
        write_json(self.out / "source-sha256.json", self.hashes)
        shutil.copy2(self.args.space, self.out / "search-space.json")
        write_matrix(self.out / "matrix.csv", self.shapes)
        compiler_environment(self.runner, self.work)
        self.run(["nvcc", "--version"], "nvcc-version")
        self.run(["cmake", "--version"], "cmake-version")
        self.telemetry("before")

    def compile(self, kernel, config):
        name = config.name if kernel == "warptiled" else "tuned_" + config.name
        dest = self.out / "variants" / name
        dest.mkdir(parents=True, exist_ok=False)
        tuned = config if kernel == "tuned" else self.default
        warp = config if kernel == "warptiled" else self.configs[0]
        variant = {
            "kernel": kernel, "params": config.values(), "status": "building",
            "tuned_params": tuned.values(), "warp_params": warp.values(),
            "threads": config.THREADS if kernel == "warptiled" else config.threads,
            "shared_bytes": config.shared_bytes,
        }
        self.variants[name] = variant
        self.checkpoint()
        prefix = "variants/" + name + "/"
        self.run([
            "cmake", "-S", str(self.source), "-B", str(self.build),
            "-G", "NMake Makefiles" if os.name == "nt" else "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
            "-DGEMM_WARP_RUNTIME_CONFIGS=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES={self.args.arch}",
            *tuned.cmake_flags(), *warp.cmake_flags(),
        ], prefix + "configure")
        build_log = self.run(["cmake", "--build", str(self.build), "--target", "gemm"], prefix + "build")
        for header in ("build_info.h", "tuned_config.h", "warptiled_config.h"):
            shutil.copy2(self.build / "generated" / header, dest / header)
        source_match = re.search(r'GEMM_SOURCE_SHA256 "([a-f0-9]{64})"', (dest / "build_info.h").read_text())
        if not source_match:
            raise ValueError("generated source fingerprint missing")
        variant["source_sha256"] = source_match[1]
        for header_name, expected in (("tuned_config.h", tuned), ("warptiled_config.h", warp)):
            header = (dest / header_name).read_text(encoding="utf-8")
            for key, value in expected.values().items():
                if not re.search(rf"\b{key}\s*=\s*{value}\s*;", header):
                    raise ValueError(f"{header_name} {key} does not match requested parameters")
        resources = re.search(
            rf"Function properties for [^\r\n]*sgemm_{kernel}[^\r\n]*(.*?)"
            r"(?=ptxas info\s*: Function properties|ptxas info\s*: Compiling entry|\Z)",
            build_log, re.DOTALL,
        )
        resource_text = resources[1] if resources else ""
        for key, pattern in (
            ("registers_per_thread", r"Used (\d+) registers"),
            ("spill_store_bytes", r"(\d+) bytes spill stores"),
            ("spill_load_bytes", r"(\d+) bytes spill loads"),
        ):
            match = re.search(pattern, resource_text)
            variant[key] = int(match[1]) if match else None
        exe = self.build / ("gemm.exe" if os.name == "nt" else "gemm")
        saved = self.work / "bin" / name / exe.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, saved)
        variant.update(exe=str(saved), exe_sha256=sha256(saved), status="compiled")
        info = self.run([saved, "--mode", "info"], prefix + "device-info")
        fields = dict(line.split("=", 1) for line in info.splitlines() if "=" in line)
        if fields.get("source_sha256") != source_match[1] or fields.get("build_type") != "Release":
            raise ValueError("executable source/build identity mismatch")
        if fields.get("compute_capability", "").replace(".", "") != self.args.arch:
            raise ValueError("selected --arch differs from GPU compute capability")
        if fields.get("warptiled_config") != warp.canonical:
            raise ValueError("executable warp parameters mismatch")
        for key, value in tuned.values().items():
            if fields.get("tuned_" + key.lower()) != str(value):
                raise ValueError("executable tuned parameters mismatch")
        if variant["threads"] > int(fields["max_threads_per_block"]) or variant["shared_bytes"] > int(fields["shared_bytes_per_block"]):
            raise ValueError("candidate exceeds physical device limits")
        self.manifest.setdefault("device", fields)
        self.checkpoint()
        return name

    def check(self, name):
        variant = self.variants[name]
        prefix = "variants/" + name + "/"
        argv = [variant["exe"], "--mode", "check", "--kernel", variant["kernel"]]
        if "suite_cases=32 status=valid" not in self.run(argv, prefix + "check-suite"):
            raise ValueError("fixed 32-case numerical suite did not pass")
        params = variant["params"]
        for label, m, n, k in (
            ("wide", 65, 97, 33), ("tall", 97, 65, 33),
            ("tile-tail", params["BM"] + 1, params["BN"] + 3, params["BK"] + 1),
        ):
            self.run([*argv, "--m", m, "--n", n, "--k", k], prefix + "check-" + label)
        variant["status"] = "checked"
        self.checkpoint()

    def bench(self, name, shape, phase, pass_number):
        variant = self.variants[name]
        kernel = variant["kernel"]
        relative = f"variants/{name}/{shape['id']}-{phase}-{pass_number}"
        path = self.out / (relative + ".csv")
        rounds = self.args.screen_rounds if phase == "screen" else shape["rounds"]
        warmup = max(100, shape["warmup"])
        self.run([
            variant["exe"], "--mode", "bench", "--kernel", kernel,
            "--m", shape["m"], "--n", shape["n"], "--k", shape["k"], "--seed", shape["seed"],
            "--warmup", warmup, "--iters", shape["iters"], "--rounds", rounds, "--csv", str(path),
        ], relative)
        timing = read_measurement(path, kernel, variant["params"], shape, variant["source_sha256"], warmup, rounds)
        self.records.append({
            "variant": name, "case": shape["id"], "phase": phase, "pass": pass_number,
            "kernel": kernel, "source_sha256": variant["source_sha256"],
            "csv": relative + ".csv", "timing": timing,
        })
        self.checkpoint()
        return timing[kernel]["median_ms"]

    def execute(self):
        self.prepare()
        for index, config in enumerate(self.configs, 1):
            name = config.name
            print(f"[{index}/{len(self.configs)}] {name}", flush=True)
            try:
                self.compile("warptiled", config)
                self.check(name)
                for shape in self.shapes:
                    ms = self.bench(name, shape, "screen", 1)
                    print(f"  {shape['id']}: {ms:.6f} ms", flush=True)
                self.variants[name]["status"] = "valid"
            except (CommandError, ValueError) as exc:
                self.variants[name].update(status="failed", error=str(exc))
                self.checkpoint()
                if self.runner.timed_out or config == self.configs[0]:
                    raise
                print("  Candidate excluded: " + str(exc), flush=True)
            self.checkpoint()

        tuned_configs = list(dict.fromkeys([self.default, *self.winners.values()]))
        for index, config in enumerate(tuned_configs, 1):
            print(f"Tuned baseline [{index}/{len(tuned_configs)}] {config.name}", flush=True)
            name = self.compile("tuned", config)
            self.check(name)
            self.variants[name]["status"] = "valid"

        selected = {self.configs[0].name}
        per_case = {}
        for shape in self.shapes:
            ranked = [record for record in self.records
                      if record["phase"] == "screen" and record["case"] == shape["id"]
                      and self.variants[record["variant"]]["status"] == "valid"]
            ranked.sort(key=lambda record: record["timing"]["warptiled"]["median_ms"])
            per_case[shape["id"]] = [record["variant"] for record in ranked[:self.args.top_k]]
            selected.update(per_case[shape["id"]])
        warp_names = [config.name for config in self.configs if config.name in selected]
        tasks = []
        for shape in self.shapes:
            relevant = warp_names + list(dict.fromkeys([
                "tuned_" + self.default.name, "tuned_" + self.winners[shape["id"]].name,
            ]))
            tasks.extend((name, shape) for name in relevant)
        random.Random(self.args.search_seed).shuffle(tasks)
        self.manifest.update(screen_finalists=per_case, confirmed_warp_candidates=warp_names)
        self.checkpoint()
        failed = set()
        for pass_number, ordered in ((1, tasks), (2, list(reversed(tasks)))):
            for name, shape in ordered:
                if name in failed:
                    continue
                print(f"Confirm {pass_number}/2 {shape['id']} {name}", flush=True)
                self.telemetry(f"{shape['id']}-{name}-{pass_number}")
                try:
                    self.bench(name, shape, "confirm", pass_number)
                except (CommandError, ValueError) as exc:
                    failed.add(name)
                    self.variants[name].update(status="confirm_failed", error=str(exc))
                    self.checkpoint()
                    if self.runner.timed_out or self.variants[name]["kernel"] == "tuned" or name == self.configs[0].name:
                        raise
        self.write_summary(failed)
        self.telemetry("after")
        self.manifest.update(
            status="complete", finished=datetime.now(LOCAL_TIME).isoformat(),
            workspace_files_changed_during_run=[name for name, digest in self.hashes.items()
                if not (ROOT / name).is_file() or sha256(ROOT / name) != digest],
        )
        self.checkpoint()
        print("DONE: " + str(self.out / "README.md"), flush=True)

    def write_summary(self, failed):
        summary = {}
        rankings, winners = {}, {}
        for shape in self.shapes:
            rows = []
            for name, variant in self.variants.items():
                if name in failed:
                    continue
                records = sorted((record for record in self.records
                    if record["variant"] == name and record["case"] == shape["id"] and record["phase"] == "confirm"),
                    key=lambda record: record["pass"])
                if len(records) != 2 or [record["pass"] for record in records] != [1, 2]:
                    continue
                kernel = variant["kernel"]
                samples = [sample for record in records for sample in record["timing"][kernel]["samples_ms"]]
                library = [sample for record in records for sample in record["timing"]["cublas"]["samples_ms"]]
                ms, library_ms = statistics.median(samples), statistics.median(library)
                flop = 2 * shape["m"] * shape["n"] * shape["k"]
                rows.append({
                    "variant": name, "kernel": kernel, "params": variant["params"],
                    "median_ms": ms, "gflops": flop / (ms * 1e6),
                    "min_ms": min(samples), "max_ms": max(samples),
                    "pass_medians_ms": [record["timing"][kernel]["median_ms"] for record in records],
                    "cublas_median_ms": library_ms, "cublas_gflops": flop / (library_ms * 1e6),
                    "cublas_over_self": library_ms / ms,
                    "source_sha256": variant["source_sha256"], "csv_files": [record["csv"] for record in records],
                })
            baseline = next(row for row in rows if row["variant"] == "tuned_" + self.default.name)
            previous = next(row for row in rows if row["variant"] == "tuned_" + self.winners[shape["id"]].name)
            rows.sort(key=lambda row: row["median_ms"])
            for row in rows:
                row.update(default_tuned_over_self=baseline["median_ms"] / row["median_ms"],
                           prior_tuned_winner_over_self=previous["median_ms"] / row["median_ms"])
            winner = next(row for row in rows if row["kernel"] == "warptiled")
            summary[shape["id"]] = {
                "shape": {key: shape[key] for key in ("m", "n", "k")},
                "best_warptiled": winner, "tuned_default": baseline,
                "tuned_previous_winner": previous, "rankings": rows,
                "note": "Best measured from this candidate set; changes include shared layout and thread mapping."
            }
            warp_rows = [dict(row, config=row["variant"]) for row in rows if row["kernel"] == "warptiled"]
            warp_baseline = next((row for row in warp_rows if row["variant"] == self.configs[0].name), None)
            if warp_baseline is None:
                raise ValueError(f"{shape['id']}: no complete confirmation for the warptiled baseline")
            for row in warp_rows:
                row["baseline_over_self"] = warp_baseline["median_ms"] / row["median_ms"]
                row["within_3_percent_of_best"] = row["median_ms"] <= warp_rows[0]["median_ms"] * 1.03
            rankings[shape["id"]] = warp_rows
            best = warp_rows[0]
            config = WarpConfig(**best["params"])
            variant = self.variants[config.name]
            winners[shape["id"]] = {
                "shape": {key: shape[key] for key in ("m", "n", "k")}, **best,
                "exe": variant["exe"], "exe_sha256": variant["exe_sha256"],
                "source_snapshot": "source-snapshot.zip",
                "configure_argv": ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                                   f"-DCMAKE_CUDA_ARCHITECTURES={self.args.arch}",
                                   *self.default.cmake_flags(), *config.cmake_flags()],
                "scope": "best measured in this search and current source/GPU, not a global optimum",
            }
        write_json(self.out / "summary.json", summary)
        write_json(self.out / "rankings.json", rankings)
        write_json(self.out / "winners.json", winners)
        flat = [{"case": case, "rank": index, **{key: value for key, value in item.items()
                 if key not in ("params", "csv_files", "pass_medians_ms")},
                 **item["params"], "pass_medians_ms": ";".join(map(str, item["pass_medians_ms"]))}
                for case, values in rankings.items() for index, item in enumerate(values, 1)]
        with (self.out / "ranking.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        lines = [
            "# Warp-tiling comparison", "",
            "Unchanged tuned kernel versus a separate experimental warptiled kernel, strict FP32 cuBLAS.",
            "Each compiled variant passed 32 fixed correctness cases plus wide, tall and tile-tail cases.",
            "Every benchmark validates the full output against cuBLAS and checks a CPU double reference.",
            "Screening uses three rounds by default; the union of per-shape finalists and default warp",
            "configuration is confirmed on all selected shapes twice in reversed task order. The tuned",
            "default and the previous tuned winner for each shape are recompiled from the same snapshot",
            "and measured in those confirmation passes. At least 100 warmups precede each measurement.",
            "Shapes without a recorded tuned winner use the default tuned configuration as their reference.",
            "Only complete confirmation samples enter this table (median of round means).", "",
            "| Case | Tuned default GFLOP/s | Prior tuned winner GFLOP/s | Best warp GFLOP/s | Paired cuBLAS GFLOP/s | Warp / prior tuned |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for case, item in summary.items():
            best = item["best_warptiled"]
            lines.append(f"| {case} | {item['tuned_default']['gflops']:.2f} | {item['tuned_previous_winner']['gflops']:.2f} | "
                         f"{best['gflops']:.2f} | {best['cublas_gflops']:.2f} | {best['prior_tuned_winner_over_self']:.3f} |")
        lines += ["", "## Best measured warp configurations", "",
                  "Parameter order: BM / BN / BK / WM / WN / TM / TN / WNITER.",
                  "The block shape is (BN / (TN * WNITER), BM / (TM * WMITER)), where",
                  "WMITER = WM * WN / (32 * TM * TN * WNITER). THREADS is derived as",
                  "32 * (BM / WM) * (BN / WN) and recorded as metadata.", ""]
        for case, item in summary.items():
            best = item["best_warptiled"]
            params = " / ".join(str(best["params"][key]) for key in WARP_PARAMS)
            threads = WarpConfig(**best["params"]).THREADS
            lines.append(f"- {case}: `{params}`; derived THREADS={threads}; "
                         f"pass medians {best['pass_medians_ms']} ms.")
        lines += [
            "", "Results compare complete kernels. They do not isolate warp mapping from shared-memory",
            "layout or compiler instruction selection, and do not establish a global optimum.",
            "Inspect individual rounds, register/spill counts and telemetry before interpreting small differences.",
            "The default tuned launch is not switched to an experimental winner.", "",
            "The unified rankings contain only warptiled candidates. baseline_over_self uses the first",
            "warp candidate; summary.json retains the separate tuned comparisons. Winners are limited",
            "to this measured search. Configurations within 3% of the best are flagged for review.", "",
            "- [Winners and reproduction commands](winners.json)",
            "- [Confirmed warptiled rankings](rankings.json)",
            "- [Confirmed warptiled ranking table](ranking.csv)",
            "- [Selected matrix and timing settings](matrix.csv)",
            "- [Confirmed rankings and comparisons](summary.json)",
            "- [All raw measurement references](measurements.json)",
            "- [Device, compiler resources and binary identities](manifest.json)",
            "- [Executed commands](commands.jsonl)",
            "- [Captured source](source-snapshot.zip)",
            "- [Source hashes](source-sha256.json)", "",
        ]
        (self.out / "README.md").write_text("\n".join(lines), encoding="utf-8")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--space", type=Path, default=ROOT / "benchmarks/warptiling.json")
    parser.add_argument("--matrix", type=Path, default=ROOT / "benchmarks/matrix.csv")
    parser.add_argument("--cases", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arch", default="89")
    parser.add_argument("--screen-rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--max-candidates", type=int)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not re.fullmatch(r"[1-9][0-9]*", args.arch):
        parser.error("--arch must be a numeric CUDA architecture")
    if min(args.screen_rounds, args.top_k, args.timeout) <= 0 or (
        args.max_candidates is not None and args.max_candidates <= 0
    ):
        parser.error("rounds, top-k, timeout and max-candidates must be positive")
    return args


def run(args, shapes=None):
    experiment = None
    try:
        configs, default, winners, shapes = load_plan(
            args.space, args.matrix, args.cases, args.max_candidates, shapes=shapes)
        if args.dry_run:
            print(json.dumps({
                "kernel": "warptiled", "cases": shapes,
                "candidates": [{**config.values(), "threads": config.THREADS,
                                "shared_bytes": config.shared_bytes} for config in configs],
                "baseline": configs[0].values(), "rejected": [],
                "warp_candidates": [config.values() for config in configs],
                "tuned_default": default.values(),
                "tuned_previous_winners": {case: config.values() for case, config in winners.items()},
                "screen_rounds": args.screen_rounds, "confirmation_passes": 2,
                "confirmation_warmup_min": 100,
                "note": "No files, compilation or GPU operations performed.",
            }, indent=2))
            return 0
        experiment = Experiment(args, configs, default, winners, shapes)
        experiment.execute()
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        if experiment is not None:
            experiment.manifest.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", error=str(exc))
            experiment.checkpoint()
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


def main(argv=None):
    return run(arguments(argv))


if __name__ == "__main__":
    sys.exit(main())
