"""Compile one tuned configuration and benchmark it against cuBLAS."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

# Keep --dry-run free of local __pycache__ writes as well as build/result files.
sys.dont_write_bytecode = True
from autotune import Config, PARAMS, read_benchmark, reject_reason, sha256


ROOT = Path(__file__).resolve().parent.parent


class CommandError(RuntimeError):
    pass


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("bm", "bn", "bk", "tm", "tn", "m", "n", "k"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--arch", default="89")
    parser.add_argument("--timeout", type=int, default=300, help="Seconds per subprocess")
    parser.add_argument("--output", type=Path, help="New result directory; never overwritten")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without creating files")
    args = parser.parse_args(argv)
    config = Config(*(getattr(args, name.lower()) for name in PARAMS))
    reason = reject_reason(config)
    if reason:
        parser.error(reason)
    if min(args.m, args.n, args.k, args.iters, args.rounds, args.timeout) <= 0:
        parser.error("dimensions, iterations, rounds and timeout must be positive")
    if args.k > 4096:
        parser.error("K exceeds the current numerical harness contract (4096)")
    if max(args.m * args.n, args.m * args.k, args.n * args.k) > 2**31 - 1:
        parser.error("matrix exceeds the current kernel's signed int index range")
    if args.warmup < 0 or not 0 <= args.seed <= 2**32 - 1:
        parser.error("warmup must be nonnegative and seed must fit an unsigned 32-bit integer")
    if not re.fullmatch(r"[1-9][0-9]*", args.arch):
        parser.error("--arch must be a numeric CUDA architecture")
    return args


def plan_for(args):
    config = Config(*(getattr(args, name.lower()) for name in PARAMS))
    shape = {name: getattr(args, name) for name in ("m", "n", "k", "seed")}
    output = args.output or ROOT / "results" / (
        "tuned-manual-" + datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    )
    output = output.resolve()
    build = ROOT / "build" / "manual-tuned"
    exe = build / ("gemm.exe" if os.name == "nt" else "gemm")
    commands = {
        "configure": [
            "cmake", "-S", str(ROOT), "-B", str(build),
            "-G", "NMake Makefiles" if os.name == "nt" else "Unix Makefiles",
            "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_TESTING=OFF",
            "-DGEMM_WARP_RUNTIME_CONFIGS=OFF",
            f"-DCMAKE_CUDA_ARCHITECTURES={args.arch}", *config.cmake_flags(),
        ],
        "build": ["cmake", "--build", str(build), "--target", "gemm"],
        "info": [str(exe), "--mode", "info"],
        "benchmark": [
            str(exe), "--mode", "bench", "--kernel", "tuned",
            "--m", str(args.m), "--n", str(args.n), "--k", str(args.k),
            "--seed", str(args.seed), "--warmup", str(args.warmup),
            "--iters", str(args.iters), "--rounds", str(args.rounds),
            "--csv", str(output / "benchmark.csv"),
        ],
    }
    return config, shape, output, build, exe, commands


class Runner:
    def __init__(self, output, timeout):
        self.output, self.timeout, self.timed_out = output, timeout, False
        self.interrupted = False
        self.env = ({key.upper(): value for key, value in os.environ.items()}
                    if os.name == "nt" else dict(os.environ))

    def run(self, argv, label, private_output=False):
        argv = list(map(str, argv))
        started = time.monotonic()
        try:
            process = subprocess.run(
                argv, cwd=ROOT, env=self.env, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, timeout=self.timeout,
            )
            code, data = process.returncode, process.stdout
        except subprocess.TimeoutExpired as exc:
            self.timed_out = True
            code, data = -1, (exc.stdout or b"") + b"\nCommand timed out.\n"
        except KeyboardInterrupt:
            self.interrupted = True
            code, data = -3, b"Command interrupted.\n"
        except OSError as exc:
            code, data = -2, str(exc).encode("utf-8")
        log = self.output / (label + ".log")
        log.write_bytes(b"Environment values omitted from logs.\n" if private_output else data)
        with (self.output / "commands.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({
                "argv": argv, "exit": code, "elapsed_seconds": time.monotonic() - started,
                "log": log.name,
            }) + "\n")
        if self.interrupted:
            raise KeyboardInterrupt
        if code:
            raise CommandError(f"{label} failed (exit {code}); see {log}")
        return data


def compiler_environment(runner, build):
    temp = build / "temp"
    if not str(temp).isascii():
        raise ValueError("The manual build directory must have an ASCII path for CUDA tools")
    temp.mkdir(parents=True, exist_ok=True)
    runner.env.update(TEMP=str(temp), TMP=str(temp))
    if os.name != "nt":
        return
    vswhere = Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / (
        "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    runner.env["PATH"] = str(vswhere.parent) + os.pathsep + runner.env.get("PATH", "")
    install = runner.run([
        vswhere, "-latest", "-products", "*",
        "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
        "-property", "installationPath",
    ], "vswhere").decode("utf-8-sig", errors="replace").strip()
    if not install:
        raise ValueError("Visual Studio C++ x64 tools were not found")
    devcmd = Path(install) / "Common7/Tools/VsDevCmd.bat"
    setup = build / "compiler-env.cmd"
    setup.write_text(
        f'@echo off\ncall "{devcmd}" -no_logo -arch=x64 -host_arch=x64 >nul\n'
        "if errorlevel 1 exit /b 1\nset\n", encoding="mbcs",
    )
    environment = runner.run(
        [runner.env["COMSPEC"], "/d", "/c", setup], "compiler-env", private_output=True
    )
    for line in environment.decode("mbcs", errors="replace").splitlines():
        if "=" in line and not line.startswith("="):
            key, value = line.split("=", 1)
            runner.env[key.upper()] = value
    # The developer prompt may replace these values.
    runner.env.update(TEMP=str(temp), TMP=str(temp))


def verify_info(text, config, source_hash, arch):
    fields = dict(line.split("=", 1) for line in text.splitlines() if "=" in line)
    if fields.get("source_sha256") != source_hash or fields.get("build_type") != "Release":
        raise ValueError("Executable source fingerprint or build type does not match the build")
    for name, expected in config.values().items():
        if fields.get("tuned_" + name.lower()) != str(expected):
            raise ValueError(f"Executable tuned {name} does not match the requested configuration")
    if fields.get("compute_capability", "").replace(".", "") != arch:
        raise ValueError("GPU compute capability differs from --arch")
    return fields


def verify_csv_config(path, config):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    tuned = next((row for row in rows if row.get("actual_kernel") == "tuned"), None)
    if tuned is None or any(
        tuned.get("tuned_" + name.lower()) != str(value)
        for name, value in config.values().items()
    ):
        raise ValueError("Benchmark CSV does not identify the requested tuned configuration")


def write_config(output, record):
    (output / "config.json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def execute(args, config, shape, output, build, exe, commands):
    output.mkdir(parents=True, exist_ok=False)
    record = {
        "status": "running", "started": datetime.now().astimezone().isoformat(),
        "parameters": config.values(), "shape": shape,
        "warmup": args.warmup, "iters": args.iters, "rounds": args.rounds,
        "architecture": args.arch, "build_directory": str(build),
    }
    write_config(output, record)
    runner = Runner(output, args.timeout)
    lock = build / ".bench-tuned.lock"
    lock_owned = False
    try:
        build.mkdir(parents=True, exist_ok=True)
        try:
            with lock.open("x", encoding="utf-8") as stream:
                stream.write(f"pid={os.getpid()}\nresults={output}\n")
            lock_owned = True
        except FileExistsError as exc:
            raise ValueError(
                f"Manual build is locked: {lock}. Another run may still be using it."
            ) from exc
        print(f"Building BM/BN/BK/TM/TN={args.bm}/{args.bn}/{args.bk}/{args.tm}/{args.tn}",
              flush=True)
        compiler_environment(runner, build)
        runner.run(commands["configure"], "configure")
        runner.run(commands["build"], "build")
        for name in ("build_info.h", "tuned_config.h"):
            shutil.copy2(build / "generated" / name, output / name)
        info_header = (output / "build_info.h").read_text(encoding="utf-8")
        match = re.search(r'GEMM_SOURCE_SHA256 "([a-f0-9]{64})"', info_header)
        if not match:
            raise ValueError("Generated build_info.h has no valid source fingerprint")
        source_hash = match[1]
        info = runner.run(commands["info"], "device-info").decode("utf-8", errors="replace")
        record.update(
            source_sha256=source_hash, executable=str(exe), executable_sha256=sha256(exe),
            device=verify_info(info, config, source_hash, args.arch),
        )
        write_config(output, record)
        print(f"Benchmarking {args.m}x{args.n}x{args.k} against cuBLAS...", flush=True)
        runner.run(commands["benchmark"], "benchmark")
        csv_path = output / "benchmark.csv"
        timing = read_benchmark(csv_path, shape, source_hash, args.warmup, args.iters, args.rounds)
        verify_csv_config(csv_path, config)
        flop = 2 * args.m * args.n * args.k
        results = {
            kernel: {**values, "gflops": flop / (values["median_ms"] * 1e6)}
            for kernel, values in timing.items()
        }
        ratio = results["cublas"]["median_ms"] / results["tuned"]["median_ms"]
        record.update(
            status="complete", finished=datetime.now().astimezone().isoformat(),
            results=results, tuned_over_cublas_gflops=ratio, cublas_over_self=ratio,
        )
        write_config(output, record)
        for kernel in ("tuned", "cublas"):
            result = results[kernel]
            print(f"{kernel:6s}  {result['median_ms']:.6f} ms  {result['gflops']:.2f} GFLOP/s")
        print(f"cuBLAS_ms / tuned_ms: {ratio:.3f}x  (tuned throughput: {ratio:.2%} of cuBLAS)")
        print("Results: " + str(output))
    except KeyboardInterrupt:
        record.update(status="interrupted")
        if runner.interrupted:
            record["build_lock_retained"] = str(lock)
        write_config(output, record)
        raise
    except Exception as exc:
        record.update(status="failed", error=str(exc))
        if runner.timed_out:
            record["build_lock_retained"] = str(lock)
        write_config(output, record)
        raise
    finally:
        # Timed-out compiler subprocesses may leave children running; do not let
        # another invocation reuse the build until the user has checked them.
        if lock_owned and not runner.timed_out and not runner.interrupted:
            lock.unlink()


def main(argv=None):
    args = arguments(argv)
    config, shape, output, build, exe, commands = plan_for(args)
    try:
        if output.exists():
            raise ValueError("Result directory already exists; choose a new --output path")
        if args.dry_run:
            print(json.dumps({
                "parameters": config.values(), "shape": shape,
                "output": str(output), "build_directory": str(build), "commands": commands,
            }, indent=2))
            return 0
        execute(args, config, shape, output, build, exe, commands)
        return 0
    except (CommandError, ValueError, OSError, KeyError, TypeError) as exc:
        print("Manual benchmark failed: " + str(exc), file=sys.stderr)
        if isinstance(exc, CommandError) and (build / ".bench-tuned.lock").exists():
            print("Build lock retained after timeout; check for remaining compiler processes "
                  "before removing it.", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        if (build / ".bench-tuned.lock").exists():
            print("Build lock retained; check for remaining processes before removing it.",
                  file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
