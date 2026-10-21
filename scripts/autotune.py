"""Autotune tuned or warptiled with the existing correctness/cuBLAS harness.

Select a kernel and --m/--n/--k, or named --cases from the benchmark matrix.
Only the isolated build is reconfigured. No kernel-body editing or GPU profiling.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
import zipfile

sys.dont_write_bytecode = True
from experiment_sources import snapshot_sources

ROOT = Path(__file__).resolve().parent.parent
PARAMS = ("BM", "BN", "BK", "TM", "TN")


@dataclass(frozen=True)
class Config:
    BM: int
    BN: int
    BK: int
    TM: int
    TN: int

    @property
    def name(self):
        return "_".join(f"{key.lower()}{getattr(self, key)}" for key in PARAMS)

    @property
    def threads(self):
        return (self.BM // self.TM) * (self.BN // self.TN)

    @property
    def shared_bytes(self):
        return 4 * self.BK * (self.BM + self.BN)

    def values(self):
        return {key: getattr(self, key) for key in PARAMS}

    def cmake_flags(self):
        return [f"-DGEMM_TUNED_{key}={getattr(self, key)}" for key in PARAMS]


def reject_reason(config, max_threads=1024, max_shared=49152):
    if any(type(v) is not int or v <= 0 for v in config.values().values()):
        return "parameters must be positive integers"
    if config.BM % config.TM or config.BN % config.TN:
        return "BM/TM and BN/TN must divide exactly"
    if config.BK % 4 or config.BN % 4:
        return "BK and BN must be multiples of four for float4 shared stores"
    if config.threads > max_threads:
        return f"threads per block exceed {max_threads}"
    if config.shared_bytes > min(49152, max_shared):
        return "static shared memory exceeds the implementation/device limit"
    return None


def load_space(path):
    spec = json.loads(path.read_text(encoding="utf-8-sig"))
    baseline = Config(*spec["baseline"])
    if reject_reason(baseline):
        raise ValueError("invalid baseline: " + reject_reason(baseline))
    raw = [baseline]
    for (bm, bn), bk, (tm, tn) in itertools.product(
        spec["block_tiles"], spec["bk"], spec["thread_tiles"]
    ):
        raw.append(Config(bm, bn, bk, tm, tn))
    raw.extend(Config(*values) for values in spec.get("extra_configs", []))
    valid, rejected = [], []
    for config in dict.fromkeys(raw):
        reason = reject_reason(config)
        if reason:
            rejected.append({"config": config.name, "params": config.values(), "reason": reason})
        else:
            valid.append(config)
    return baseline, valid, rejected


def validate_shape(shape):
    if min(shape[key] for key in ("m", "n", "k", "iters", "rounds")) <= 0:
        raise ValueError("dimensions and timing counts must be positive")
    if shape["warmup"] < 0 or not 0 <= shape["seed"] <= 2**32 - 1:
        raise ValueError("invalid warmup or seed")
    if shape["k"] > 4096:
        raise ValueError("K exceeds the current numerical harness contract (4096)")
    if max(shape["m"] * shape["n"], shape["m"] * shape["k"], shape["n"] * shape["k"]) > 2**31 - 1:
        raise ValueError("matrix exceeds the current kernel's signed int index range")


def load_shapes(path, cases):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("duplicate matrix IDs")
    if cases and set(cases) - {r["id"] for r in rows}:
        raise ValueError("unknown case IDs: " + ", ".join(sorted(set(cases) - {r["id"] for r in rows})))
    result = []
    for row in rows:
        if cases and row["id"] not in cases:
            continue
        if row["dtype"] != "fp32" or not re.fullmatch(r"[A-Za-z0-9_-]+", row["id"]):
            raise ValueError("matrix must use fp32 and filesystem-safe IDs")
        shape = {key: int(row[key]) for key in ("m", "n", "k", "seed", "warmup", "iters", "rounds")}
        validate_shape(shape)
        result.append({"id": row["id"], **shape})
    if not result:
        raise ValueError("no matrix cases selected")
    return result


def resolve_shapes(args):
    """Resolve explicit dimensions or matrix IDs before creating any run artifacts."""
    if args.m is None:
        shapes = load_shapes(args.matrix, args.cases)
    else:
        matches = [shape for shape in load_shapes(args.matrix, None)
                   if (shape["m"], shape["n"], shape["k"]) == (args.m, args.n, args.k)]
        if len(matches) > 1:
            raise ValueError("multiple matrix cases match M/N/K; select one with --cases")
        shapes = matches or [{
            "id": f"M{args.m}_N{args.n}_K{args.k}",
            "m": args.m, "n": args.n, "k": args.k,
            "seed": 42, "warmup": 100, "iters": 1000, "rounds": 5,
        }]
    for shape in shapes:
        for key in ("seed", "warmup", "iters", "rounds"):
            value = getattr(args, key)
            if value is not None:
                shape[key] = value
        validate_shape(shape)
    return shapes


def write_matrix(path, shapes):
    """Record the selected shapes and effective CLI overrides, not the whole catalog."""
    fields = ("id", "m", "n", "k", "dtype", "seed", "warmup", "iters", "rounds")
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: "fp32" if key == "dtype" else shape[key] for key in fields}
                         for shape in shapes)


def read_benchmark(path, shape, source_hash, warmup, iters, rounds):
    """Reject bad/stale measurements before they can enter any ranking."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 2 or {r["actual_kernel"] for r in rows} != {"tuned", "cublas"}:
        raise ValueError("expected one tuned row and one cuBLAS row")
    parsed = {}
    for row in rows:
        if row["status"] != "valid" or row["requested_kernel"] != "tuned":
            raise ValueError("invalid benchmark status or implementation")
        if any(int(row[key]) != 0 for key in ("failed", "nonfinite", "cpu_failed")):
            raise ValueError("numerical validation failed")
        if int(row["cpu_compared"]) <= 0:
            raise ValueError("missing independent CPU reference comparisons")
        if row["actual_kernel"] == "tuned" and (
            row["reference_scope"] != "cublas_full" or int(row["compared"]) != shape["m"] * shape["n"]
        ):
            raise ValueError("missing full output comparison against cuBLAS")
        if row["source_sha256"] != source_hash or row["build_type"] != "Release":
            raise ValueError("source identity/build type mismatch")
        if row["compute_type"] != "CUBLAS_COMPUTE_32F_PEDANTIC":
            raise ValueError("cuBLAS precision mismatch")
        if any(row[key] != "fp32" for key in ("input_dtype", "accum_dtype", "output_dtype")):
            raise ValueError("dtype mismatch")
        for key, expected in {
            **{key: shape[key] for key in ("m", "n", "k", "seed")},
            "warmup": warmup, "iters": iters, "rounds": rounds,
        }.items():
            if int(row[key]) != expected:
                raise ValueError(f"benchmark {key} mismatch")
        samples = [float(value) for value in row["round_mean_ms"].split(";")]
        if len(samples) != rounds or any(not math.isfinite(v) or v <= 0 for v in samples):
            raise ValueError("invalid timing samples")
        median = statistics.median(samples)
        if not math.isclose(median, float(row["median_ms"]), rel_tol=1e-8):
            raise ValueError("median does not match timing samples")
        parsed[row["actual_kernel"]] = {"samples_ms": samples, "median_ms": median}
    return parsed


def ranking_for(shape, configs, records, baseline_name, passes=2):
    """Rank complete confirmation runs only; never fall back to screening data."""
    ranked = []
    for config in configs:
        own = [r for r in records if r["phase"] == "confirm"
               and r["case"] == shape["id"] and r["config"] == config.name]
        if len(own) != passes or {r["pass"] for r in own} != set(range(1, passes + 1)):
            continue
        samples = [v for r in own for v in r["timing"]["tuned"]["samples_ms"]]
        library = [v for r in own for v in r["timing"]["cublas"]["samples_ms"]]
        ms, lib_ms = statistics.median(samples), statistics.median(library)
        ranked.append({
            "config": config.name, "params": config.values(), "median_ms": ms,
            "gflops": 2 * shape["m"] * shape["n"] * shape["k"] / (ms * 1e6),
            "min_ms": min(samples), "max_ms": max(samples),
            "pass_medians_ms": [r["timing"]["tuned"]["median_ms"] for r in own],
            "cublas_median_ms": lib_ms, "cublas_over_self": lib_ms / ms,
            "source_sha256": own[0]["source_sha256"],
            "csv_files": [r["csv"] for r in own],
        })
    baseline = next((r for r in ranked if r["config"] == baseline_name), None)
    if not baseline:
        raise ValueError(f"{shape['id']}: no complete confirmation for the baseline")
    ranked.sort(key=lambda r: r["median_ms"])
    for item in ranked:
        item["baseline_over_self"] = baseline["median_ms"] / item["median_ms"]
        item["within_3_percent_of_best"] = item["median_ms"] <= ranked[0]["median_ms"] * 1.03
    return ranked


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class CommandError(RuntimeError):
    pass


class CommandTimeout(RuntimeError):
    """Fatal: do not reuse a build directory whose compiler children may still run."""


class ReuseError(RuntimeError):
    """Fatal: an explicitly requested binary cache did not establish its identity."""


class Search:
    def __init__(self, args, baseline, configs, rejected, shapes):
        self.args, self.baseline, self.configs, self.shapes = args, baseline, configs, shapes
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        self.out = Path(args.output).resolve() if args.output else ROOT / "results" / ("autotune-" + stamp)
        self.work = ROOT / "build" / ("autotune-" + stamp)
        self.out.mkdir(parents=True, exist_ok=False)
        self.work.mkdir(parents=True, exist_ok=False)
        self.source, self.build = self.work / "source", self.work / "cmake-build"
        self.env = {k.upper(): v for k, v in os.environ.items()} if os.name == "nt" else dict(os.environ)
        self.records, self.variants = [], {}
        self.reuse_root, self.reused_variants = None, {}
        self.reused_source_artifacts = {}
        self.manifest = {
            "status": "running", "started": datetime.now().astimezone().isoformat(),
            "baseline": baseline.values(), "search_seed": args.search_seed,
            "selected_candidates": [c.values() for c in configs], "rejected": rejected,
            "matrix": shapes, "screen_rounds": args.screen_rounds, "top_k": args.top_k,
            "confirmation_passes": 2, "confirmation_warmup_min": 100,
            "output_dir": str(self.out), "work_dir": str(self.work),
            "requested_reuse_builds": str(args.reuse_builds.resolve()) if args.reuse_builds else None,
            "ranking_rule": "median of confirmation round means; each shape ranked separately",
        }
        self.checkpoint()

    def checkpoint(self):
        write_json(self.out / "manifest.json", {**self.manifest, "variants": self.variants})
        write_json(self.out / "measurements.json", self.records)

    def run(self, args, log, required=True):
        args = list(map(str, args))
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.monotonic()
        try:
            proc = subprocess.run(args, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, timeout=self.args.timeout)
            code, data = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired as exc:
            code, data = -1, (exc.stdout or b"") + b"\nAUTOTUNE: command timed out\n"
        except OSError as exc:
            code, data = -2, str(exc).encode("utf-8")
        log.write_bytes(data)
        with (self.out / "commands.jsonl").open("a", encoding="utf-8") as f:
            f.write(json.dumps({"argv": args, "exit": code, "elapsed_seconds": time.monotonic() - started,
                                "log": str(log.relative_to(self.out))}, ensure_ascii=False) + "\n")
        text = data.decode("utf-8", errors="replace")
        if code == -1:
            raise CommandTimeout(f"Timed out; search stopped without reusing the build directory: {log}")
        if code and required:
            raise CommandError(f"exit {code}: {log}\n{text[-1500:]}")
        return code, text

    def prepare(self):
        print("Output: " + str(self.out), flush=True)
        self.hashes = snapshot_sources(ROOT, self.source, self.out / "source-snapshot.zip")
        write_json(self.out / "source-sha256.json", self.hashes)
        shutil.copy2(self.args.space, self.out / "search-space.json")
        write_matrix(self.out / "matrix.csv", self.shapes)
        if self.args.reuse_builds:
            self.prepare_reuse()
        if os.name == "nt":
            vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft Visual Studio/Installer/vswhere.exe"
            self.env["PATH"] = str(vswhere.parent) + os.pathsep + self.env["PATH"]
            _, vs = self.run([vswhere, "-latest", "-products", "*", "-requires",
                              "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property",
                              "installationPath"], self.out / "environment/vswhere.log")
            if not vs.strip():
                raise CommandError("Visual Studio C++ x64 tools not found")
            devcmd = Path(vs.strip()) / "Common7/Tools/VsDevCmd.bat"
            setup = self.work / "compiler-env.cmd"
            setup.write_text(f'@echo off\ncall "{devcmd}" -no_logo -arch=x64 -host_arch=x64 >nul\n'
                             "if errorlevel 1 exit /b 1\nset\n", encoding="mbcs")
            proc = subprocess.run([self.env["COMSPEC"], "/d", "/c", str(setup)],
                                  env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                  timeout=60, check=True)
            for line in proc.stdout.decode("mbcs", errors="replace").splitlines():
                if "=" in line and not line.startswith("="):
                    key, value = line.split("=", 1)
                    self.env[key.upper()] = value
        temp = self.work / "temp"
        temp.mkdir()
        self.env.update(TEMP=str(temp), TMP=str(temp))
        for name, command in (("nvcc", ["nvcc", "--version"]), ("cmake", ["cmake", "--version"]),
                              ("git", ["git", "rev-parse", "HEAD"])):
            self.run(command, self.out / "environment" / (name + ".log"), required=False)
        self.telemetry("before")

    def prepare_reuse(self):
        self.reuse_root = self.args.reuse_builds.resolve()
        manifest_path = self.reuse_root / "manifest.json"
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") != "complete":
            raise ReuseError("--reuse-builds requires a completed previous search")
        previous_arch = str(previous.get("device", {}).get("compute_capability", "")).replace(".", "")
        if previous_arch != self.args.arch:
            raise ReuseError("--reuse-builds architecture differs from the previous device")
        self.archive_reused_sources("source-snapshot.zip", "source-sha256.json")
        self.reused_variants = previous["variants"]
        shutil.copy2(manifest_path, self.out / "reused-manifest.json")
        self.manifest["reuse_builds"] = {
            "previous_results": str(self.reuse_root),
            "previous_manifest_sha256": sha256(manifest_path),
            "architecture": previous_arch,
            "core_source_match": "exact CMakeLists.txt and src/** path/hash sets",
            "source_identity": (
                "Reused binaries keep their original source_sha256, including the original "
                "benchmark matrix in that compiled fingerprint. Each variant identifies its "
                "original source archive; the current matrix only defines this run's measurements."
            ),
        }
        self.checkpoint()

    def archive_reused_sources(self, archive_name, hashes_name):
        """Validate and preserve a binary's original inputs, including chained reuse."""
        key = (archive_name, hashes_name)
        if key in self.reused_source_artifacts:
            return self.reused_source_artifacts[key]
        origin_archive = (self.reuse_root / archive_name).resolve()
        origin_hashes = (self.reuse_root / hashes_name).resolve()
        if not all(p.is_relative_to(self.reuse_root) for p in (origin_archive, origin_hashes)):
            raise ReuseError("previous source evidence must be inside its results directory")
        old_hashes = json.loads(origin_hashes.read_text(encoding="utf-8"))
        is_core = lambda name: name == "CMakeLists.txt" or name.startswith("src/")
        current_core = {name: digest for name, digest in self.hashes.items() if is_core(name)}
        previous_core = {name: digest for name, digest in old_hashes.items() if is_core(name)}
        if current_core != previous_core:
            changed = sorted(name for name in current_core.keys() | previous_core.keys()
                             if current_core.get(name) != previous_core.get(name))
            raise ReuseError("--reuse-builds core source mismatch: " + ", ".join(changed))
        try:
            with zipfile.ZipFile(origin_archive) as archive:
                names = archive.namelist()
                if len(names) != len(set(names)):
                    raise ReuseError("duplicate paths in previous source archive")
                if set(names) != set(old_hashes):
                    raise ReuseError("previous source archive/manifest path mismatch")
                if any(hashlib.sha256(archive.read(name)).hexdigest() != digest
                       for name, digest in old_hashes.items()):
                    raise ReuseError("previous source archive/manifest hash mismatch")
        except zipfile.BadZipFile as exc:
            raise ReuseError("previous source archive is invalid") from exc
        if key == ("source-snapshot.zip", "source-sha256.json"):
            saved_archive = self.out / "reused-source-snapshot.zip"
            saved_hashes = self.out / "reused-source-sha256.json"
        else:
            saved_archive = self.out / "reused-origins" / (sha256(origin_archive) + ".zip")
            saved_hashes = self.out / "reused-origins" / (sha256(origin_hashes) + ".json")
            saved_archive.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin_archive, saved_archive)
        shutil.copy2(origin_hashes, saved_hashes)
        evidence = {
            "source_snapshot": saved_archive.relative_to(self.out).as_posix(),
            "source_hashes": saved_hashes.relative_to(self.out).as_posix(),
        }
        self.reused_source_artifacts[key] = evidence
        return evidence

    def reuse_binary(self, config, dest):
        previous = self.reused_variants.get(config.name)
        if not previous or previous.get("status") != "valid":
            return None
        if previous.get("params") != config.values():
            raise ReuseError("previous variant parameters mismatch: " + config.name)
        exe = Path(previous["exe"])
        if not exe.is_file() or sha256(exe) != previous.get("exe_sha256"):
            raise ReuseError("previous executable missing or hash mismatch: " + str(exe))
        original = self.reuse_root / "variants" / config.name
        info = (original / "build_info.h").read_text(encoding="utf-8")
        match = re.search(r'GEMM_SOURCE_SHA256 "([a-f0-9]{64})"', info)
        if not match or match[1] != previous.get("source_sha256"):
            raise ReuseError("previous build fingerprint mismatch: " + config.name)
        header = (original / "tuned_config.h").read_text(encoding="utf-8")
        if any(not re.search(rf"\b{key}\s*=\s*{value}\s*;", header)
               for key, value in config.values().items()):
            raise ReuseError("previous generated configuration mismatch: " + config.name)
        evidence = self.archive_reused_sources(
            previous.get("source_snapshot", "source-snapshot.zip"),
            previous.get("source_hashes", "source-sha256.json"),
        )
        artifacts = ("build_info.h", "tuned_config.h", "build.log", "configure.log")
        for name in artifacts:
            shutil.copy2(original / name, dest / name)
        saved = self.work / "bin" / config.name / exe.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, saved)
        saved_hash = sha256(saved)
        if saved_hash != previous["exe_sha256"]:
            raise ReuseError("executable changed during reuse: " + str(exe))
        variant = {
            **previous, **evidence, "exe": str(saved), "exe_sha256": saved_hash,
            "status": "compiled", "reused_from": {
                "results": str(self.reuse_root), "variant": config.name, "exe": str(exe),
                "source_sha256": previous["source_sha256"],
                "build_artifacts": {name: str(original / name) for name in artifacts},
            },
        }
        self.variants[config.name] = variant
        print("  Reusing verified executable: " + config.name, flush=True)
        return saved

    def telemetry(self, label):
        self.run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,pstate,temperature.gpu,clocks.gr,clocks.mem,power.draw,power.limit",
                  "--format=csv"], self.out / "telemetry" / (label + ".log"), required=False)

    def compile(self, config):
        dest = self.out / "variants" / config.name
        dest.mkdir(parents=True, exist_ok=True)
        reused = self.reuse_binary(config, dest)
        if reused:
            return reused
        variant = {"params": config.values(), "threads": config.threads,
                   "static_shared_bytes": config.shared_bytes, "status": "building",
                   "source_snapshot": "source-snapshot.zip", "source_hashes": "source-sha256.json"}
        self.variants[config.name] = variant
        generator = "NMake Makefiles" if os.name == "nt" else "Unix Makefiles"
        self.run(["cmake", "-S", self.source, "-B", self.build, "-G", generator,
                  "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
                  "-DGEMM_WARP_RUNTIME_CONFIGS=OFF",
                  f"-DCMAKE_CUDA_ARCHITECTURES={self.args.arch}", *config.cmake_flags()],
                 dest / "configure.log")
        _, log = self.run(["cmake", "--build", self.build, "--target", "gemm"], dest / "build.log")
        generated = self.build / "generated"
        for name in ("build_info.h", "tuned_config.h"):
            shutil.copy2(generated / name, dest / name)
        info = (generated / "build_info.h").read_text(encoding="utf-8")
        variant["source_sha256"] = re.search(r'GEMM_SOURCE_SHA256 "([a-f0-9]{64})"', info)[1]
        for key, value in config.values().items():
            header = (generated / "tuned_config.h").read_text(encoding="utf-8")
            if not re.search(rf"\b{key}\s*=\s*{value}\s*;", header):
                raise ValueError("generated tuned configuration does not match requested parameters")
        tuned_log = re.search(r"Function properties for [^\r\n]*sgemm_tuned[^\r\n]*"
                              r"(.*?)(?=ptxas info\s*: Function properties|ptxas info\s*: Compiling entry|\Z)",
                              log, re.DOTALL)
        resource_text = tuned_log[1] if tuned_log else ""
        reg = re.search(r"Used (\d+) registers.*?(\d+) bytes smem", resource_text)
        variant["ptxas_registers_per_thread"] = int(reg[1]) if reg else None
        stores = re.search(r"(\d+) bytes spill stores", resource_text)
        loads = re.search(r"(\d+) bytes spill loads", resource_text)
        variant["spill_store_bytes"] = int(stores[1]) if stores else None
        variant["spill_load_bytes"] = int(loads[1]) if loads else None
        exe = self.build / ("gemm.exe" if os.name == "nt" else "gemm")
        saved = self.work / "bin" / config.name / exe.name
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(exe, saved)
        variant.update(exe=str(saved), exe_sha256=sha256(saved), status="compiled")
        return saved

    def check(self, config, exe):
        dest = self.out / "variants" / config.name
        _, text = self.run([exe, "--mode", "check", "--kernel", "tuned"], dest / "check-suite.log")
        if "suite_cases=32 status=valid" not in text:
            raise ValueError("fixed correctness suite did not report success")
        for label, m, n, k in (("wide", 65, 97, 33), ("tall", 97, 65, 33),
                               ("tile-tail", config.BM + 1, config.BN + 3, config.BK + 1)):
            self.run([exe, "--mode", "check", "--kernel", "tuned",
                      "--m", m, "--n", n, "--k", k], dest / ("check-" + label + ".log"))
        self.variants[config.name]["status"] = "checked"

    def bench(self, config, shape, phase, pass_number):
        variant = self.variants[config.name]
        dest = self.out / "variants" / config.name
        csv_path = dest / f"{shape['id']}-{phase}-{pass_number}.csv"
        rounds = self.args.screen_rounds if phase == "screen" else shape["rounds"]
        warmup = shape["warmup"] if phase == "screen" else max(100, shape["warmup"])
        self.run([variant["exe"], "--mode", "bench", "--kernel", "tuned",
                  "--m", shape["m"], "--n", shape["n"], "--k", shape["k"], "--seed", shape["seed"],
                  "--warmup", warmup, "--iters", shape["iters"], "--rounds", rounds, "--csv", csv_path],
                 csv_path.with_suffix(".log"))
        timing = read_benchmark(csv_path, shape, variant["source_sha256"], warmup, shape["iters"], rounds)
        record = {
            "config": config.name, "case": shape["id"], "phase": phase, "pass": pass_number,
            "source_sha256": variant["source_sha256"], "timing": timing,
            "csv": str(csv_path.relative_to(self.out)).replace("\\", "/"),
        }
        self.records.append(record)
        self.checkpoint()
        return timing["tuned"]["median_ms"]

    def execute(self):
        self.prepare()
        baseline_exe = self.compile(self.baseline)
        _, info = self.run([baseline_exe, "--mode", "info"], self.out / "device-info.log")
        fields = dict(line.split("=", 1) for line in info.splitlines() if "=" in line)
        self.manifest["device"] = fields
        if fields["compute_capability"].replace(".", "") != self.args.arch:
            raise ValueError("GPU compute capability differs from --arch; rerun with the device's architecture")
        accepted = []
        for config in self.configs:
            reason = reject_reason(config, int(fields["max_threads_per_block"]), int(fields["shared_bytes_per_block"]))
            if reason:
                self.manifest["rejected"].append({"config": config.name, "params": config.values(), "reason": reason})
            else:
                accepted.append(config)
        if self.baseline not in accepted:
            raise ValueError("baseline exceeds the selected device's limits")
        self.manifest["device_accepted_candidates"] = len(accepted)
        self.checkpoint()
        for index, config in enumerate(accepted, 1):
            print(f"[{index}/{len(accepted)}] {config.name}", flush=True)
            stage = "compile"
            try:
                exe = baseline_exe if config == self.baseline else self.compile(config)
                stage = "check"
                self.check(config, exe)
                stage = "screen"
                for shape in self.shapes:
                    ms = self.bench(config, shape, "screen", 1)
                    print(f"  {shape['id']}: {ms:.6f} ms", flush=True)
                self.variants[config.name]["status"] = "valid"
            except (CommandError, ValueError) as exc:
                self.variants.setdefault(config.name, {"params": config.values()})
                self.variants[config.name].update(status=f"{stage}_failed", error=str(exc))
                print(f"  Rejected at {stage}; see logs.", flush=True)
                self.checkpoint()
                if config == self.baseline:
                    raise
            self.checkpoint()

        finalists = {}
        for shape in self.shapes:
            times = [r for r in self.records if r["phase"] == "screen" and r["case"] == shape["id"]
                     and self.variants[r["config"]]["status"] == "valid"]
            times.sort(key=lambda r: r["timing"]["tuned"]["median_ms"])
            names = {r["config"] for r in times[:self.args.top_k]} | {self.baseline.name}
            finalists[shape["id"]] = [c for c in accepted if c.name in names]
        self.manifest["finalists"] = {key: [c.name for c in values] for key, values in finalists.items()}
        final_names = {c.name for values in finalists.values() for c in values}
        final_configs = [c for c in accepted if c.name in final_names]
        self.manifest["confirmation_candidates"] = [c.name for c in final_configs]
        tasks = [(config, shape) for shape in self.shapes for config in final_configs]
        random.Random(self.args.search_seed + 1).shuffle(tasks)
        confirmation_failed = set()
        for pass_number, ordered in ((1, tasks), (2, list(reversed(tasks)))):
            for config, shape in ordered:
                if config.name in confirmation_failed:
                    continue
                print(f"Confirm {pass_number}/2 {shape['id']} {config.name}", flush=True)
                self.telemetry(f"{shape['id']}-{config.name}-confirm-{pass_number}")
                try:
                    self.bench(config, shape, "confirm", pass_number)
                except (CommandError, ValueError) as exc:
                    confirmation_failed.add(config.name)
                    self.variants[config.name].update(status="confirm_failed", error=str(exc))
                    if config == self.baseline:
                        raise
        rankings = {
            shape["id"]: ranking_for(shape, [c for c in final_configs
                                             if c.name not in confirmation_failed],
                                      self.records, self.baseline.name)
            for shape in self.shapes
        }
        write_json(self.out / "rankings.json", rankings)
        winners = {}
        for shape in self.shapes:
            winner = rankings[shape["id"]][0]
            config = Config(**winner["params"])
            winners[shape["id"]] = {
                "shape": {key: shape[key] for key in ("m", "n", "k")}, **winner,
                "exe": self.variants[config.name]["exe"],
                "configure_argv": ["cmake", "-S", ".", "-B", "build", "-DCMAKE_BUILD_TYPE=Release",
                                   f"-DCMAKE_CUDA_ARCHITECTURES={self.args.arch}", *config.cmake_flags()],
                "scope": "best measured in this search and current source/GPU, not a global optimum",
            }
        write_json(self.out / "winners.json", winners)
        flat = [{"case": case, "rank": index, **{k: v for k, v in item.items()
                  if k not in ("params", "csv_files", "pass_medians_ms")},
                 **item["params"], "pass_medians_ms": ";".join(map(str, item["pass_medians_ms"]))}
                for case, values in rankings.items() for index, item in enumerate(values, 1)]
        with (self.out / "ranking.csv").open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(flat[0]))
            writer.writeheader()
            writer.writerows(flat)
        self.telemetry("after")
        changed = [name for name, digest in self.hashes.items()
                   if not (ROOT / name).is_file() or sha256(ROOT / name) != digest]
        self.manifest.update(
            status="complete", finished=datetime.now().astimezone().isoformat(),
            workspace_files_changed_during_run=changed,
            note=("Each variant names its binary's source snapshot and original compiled fingerprint. "
                  "The current matrix defines this run's measurements; core source edits require a new search."),
        )
        self.checkpoint()
        lines = [
            "# Autotuning results", "",
            "Captured CUDA/CMake source, Release FP32, one GPU, existing CUDA Event/cuBLAS harness.",
            "Each candidate passes the 32-case suite, two rectangular boundary checks and a tile-relative",
            "(BM+1, BN+3, BK+1) tail check before screening.",
            "Every timed shape also validates output before timing. Failed candidates are excluded.",
            "Screening uses the matrix's iteration count and three rounds by default; finalists and the baseline",
            "are measured on all selected shapes twice in reversed task order with the matrix's round count",
            "and at least 100 warmups.",
            "Ranks use only confirmation samples, separately for each shape. No NCU timing is used.", "",
            "| Case | BM/BN/BK/TM/TN | ms | GFLOP/s | baseline/self | cuBLAS/self |",
            "|---|---|---:|---:|---:|---:|",
        ]
        for case, item in winners.items():
            params = "/".join(str(item["params"][key]) for key in PARAMS)
            lines.append(f"| {case} | {params} | {item['median_ms']:.6f} | {item['gflops']:.2f} | "
                         f"{item['baseline_over_self']:.3f} | {item['cublas_over_self']:.3f} |")
        lines += [
            "", "These are measured winners in the selected candidate set, not guaranteed global optima.",
            "Configurations within 3% of the best are flagged in rankings.json; this is a practical",
            "near-tie threshold, not a statistical confidence interval. Inspect raw rounds and telemetry.",
            "The ordinary project build/default launch has not been switched to any winner.", "",
            "- [Winners and reproduction commands](winners.json)",
            "- [All confirmed rankings](ranking.csv)",
            "- [Screening and confirmation measurements](measurements.json)",
            "- [Configuration status, resource reports and device](manifest.json)",
            "- [Source snapshot](source-snapshot.zip)",
            "- [Executed commands](commands.jsonl)", "",
        ]
        if self.reuse_root:
            reused_count = sum("reused_from" in variant for variant in self.variants.values())
            lines += [
                f"Reused {reused_count} verified binaries from `{self.reuse_root}`.",
                "Their original source_sha256 values are preserved, including the original matrix",
                "in the compiled fingerprint. This run's matrix and script changes do not rewrite",
                "binary identities. Every reused binary is checked and timed again.",
                "Each manifest variant links its original source_snapshot/source_hashes evidence.",
                "- [Previous run manifest](reused-manifest.json)",
                "- [Previous source snapshot](reused-source-snapshot.zip)",
                "- [Previous source hashes](reused-source-sha256.json)", "",
            ]
        if changed:
            lines += ["Workspace files changed during the search: " + ", ".join(changed),
                      "Results retain each variant's recorded binary/source identity.", ""]
        (self.out / "README.md").write_text("\n".join(lines), encoding="utf-8")
        print("DONE: " + str(self.out / "README.md"), flush=True)
        for case, winner in winners.items():
            print(f"{case}: {winner['params']}  {winner['gflops']:.2f} GFLOP/s  "
                  f"baseline/self={winner['baseline_over_self']:.3f}", flush=True)


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kernel", choices=("tuned", "warptiled"), default="tuned")
    parser.add_argument("--space", type=Path, help="Candidate JSON; defaults to the selected kernel's space")
    parser.add_argument("--matrix", type=Path, default=ROOT / "benchmarks/matrix.csv")
    parser.add_argument("--cases", nargs="+", help="Matrix IDs, e.g. P1 P2; default: all")
    for dimension in ("m", "n", "k"):
        parser.add_argument("--" + dimension, type=int, help="Single shape; specify all three dimensions")
    parser.add_argument("--seed", type=int, help="Override the input seed")
    parser.add_argument("--warmup", type=int, help="Override warmups; confirmation uses at least 100")
    parser.add_argument("--iters", type=int, help="Override GEMMs per timing round")
    parser.add_argument("--rounds", type=int, help="Override rounds per confirmation pass")
    parser.add_argument("--arch", default="89", help="CUDA architecture, default: 89")
    parser.add_argument("--output", type=Path, help="New output directory; existing directories are rejected")
    parser.add_argument("--reuse-builds", type=Path, metavar="PREVIOUS_RESULTS",
                        help="Reuse verified binaries from a completed run with identical core source")
    parser.add_argument("--search-seed", type=int, default=42)
    parser.add_argument("--screen-rounds", type=int, default=3)
    parser.add_argument("--top-k", type=int, default=3, help="Finalists per shape, plus baseline")
    parser.add_argument("--max-candidates", type=int, help="Limit count, including baseline, for a short search")
    parser.add_argument("--timeout", type=int, default=300, help="Timeout seconds per subprocess")
    parser.add_argument("--dry-run", action="store_true", help="Print candidates only; no files/build/GPU")
    args = parser.parse_args(argv)
    args.space = args.space or ROOT / "benchmarks" / (
        "tuning.json" if args.kernel == "tuned" else "warptiling.json"
    )
    if not re.fullmatch(r"[1-9][0-9]*", args.arch):
        parser.error("--arch must be a numeric CUDA architecture")
    if min(args.screen_rounds, args.top_k, args.timeout, args.max_candidates or 1) < 1:
        parser.error("counts and timeout must be positive")
    if args.max_candidates == 0:
        parser.error("--max-candidates must be positive")
    supplied = [getattr(args, key) is not None for key in ("m", "n", "k")]
    if any(supplied) and not all(supplied):
        parser.error("specify all of --m --n --k")
    if any(supplied) and args.cases:
        parser.error("--m/--n/--k cannot be combined with --cases")
    if any(supplied):
        try:
            validate_shape(dict(m=args.m, n=args.n, k=args.k, seed=42, warmup=100, iters=1000, rounds=5))
        except ValueError as exc:
            parser.error(str(exc))
    if any(value is not None and value <= 0 for value in (args.iters, args.rounds)):
        parser.error("--iters and --rounds must be positive")
    if args.warmup is not None and args.warmup < 0:
        parser.error("--warmup must be nonnegative")
    if args.seed is not None and not 0 <= args.seed <= 2**32 - 1:
        parser.error("--seed must fit an unsigned 32-bit integer")
    if args.kernel == "warptiled" and args.reuse_builds:
        parser.error("--reuse-builds currently supports only --kernel tuned")
    return args


def main(argv=None):
    args = arguments(argv)
    search = None
    try:
        shapes = resolve_shapes(args)
        if args.kernel == "warptiled":
            from compare_warptiled import run
            return run(args, shapes=shapes)
        baseline, configs, rejected = load_space(args.space)
        remaining = [c for c in configs if c != baseline]
        random.Random(args.search_seed).shuffle(remaining)
        configs = [baseline] + remaining
        if args.max_candidates:
            configs = configs[:args.max_candidates]
        if args.dry_run:
            print(json.dumps({
                "kernel": args.kernel, "cases": shapes,
                "candidates": [{**config.values(), "threads": config.threads,
                                "shared_bytes": config.shared_bytes} for config in configs],
                "baseline": baseline.values(), "rejected": rejected,
                "screen_rounds": args.screen_rounds, "confirmation_passes": 2,
                "confirmation_warmup_min": 100,
                "note": "No files, compilation or GPU operations performed.",
            }, indent=2))
            return 0
        print(f"{len(configs)} candidates, {len(rejected)} statically rejected; cases: "
              + ", ".join(s["id"] for s in shapes), flush=True)
        search = Search(args, baseline, configs, rejected, shapes)
        search.execute()
        return 0
    except (OSError, ValueError, KeyError, TypeError, CommandError, CommandTimeout,
            ReuseError, subprocess.SubprocessError) as exc:
        if search:
            search.manifest.update(status="failed", error=str(exc))
            search.checkpoint()
        print("Autotune failed: " + str(exc), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        if search:
            search.manifest.update(status="interrupted")
            search.checkpoint()
        print("Interrupted; completed logs/measurements were preserved.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
