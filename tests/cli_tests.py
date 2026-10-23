"""CLI integration tests: rejection paths, independent oracle, CSV, profile isolation."""
import csv
import io
import math
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

exe = str(Path(sys.argv[1]).resolve())
count = 0


def invoke(args, ok, expected):
    global count
    result = subprocess.run([exe, *args], capture_output=True, text=True, timeout=60)
    output = result.stdout + result.stderr
    assert (result.returncode == 0) == ok, (args, result.returncode, output)
    assert expected in output, (args, expected, output)
    count += 1
    return output


for args, message in [
    (["--mode", "bench"], "requires --m --n --k"),
    (["--mode", "check", "--m", "1"], "specify all"),
    (["--m", "0"], "must be positive"),
    (["--m", "-1"], "unsigned integer"),
    (["--m", "2147483648"], "out of range"),
    (["--seed", "4294967296"], "out of range"),
    (["--iters", "1.5"], "unsigned integer"),
    (["--iters", "0"], "must be positive"),
    (["--rounds", "0"], "must be positive"),
    (["--mode", "other"], "unknown mode"),
    (["--kernel", "does_not_exist"], "unknown/unimplemented kernel"),
    (["--dtype", "fp16"], "unsupported dtype"),
    (["--extra", "1"], "unknown option"),
    (["--mode"], "missing value"),
    (["--seed", "1", "--seed", "2"], "duplicate option"),
    (["--mode", "check", "--m", "1", "--n", "1", "--k", "4097"], "outside the current numerical"),
    (["--mode", "profile", "--m", "1", "--n", "1", "--k", "1", "--rounds", "2"], "requires --rounds 1"),
    (["--mode", "check", "--m", "2147483647", "--n", "2147483647", "--k", "4096"], "total allocation overflow"),
]:
    invoke(args, False, message)

# Availability is part of the public interface; enabling a new implementation
# automatically opts it into the correctness, benchmark and profile checks below.
info = invoke(["--mode", "info"], True, "CUBLAS_COMPUTE_32F_PEDANTIC")
info_fields = dict(line.split("=", 1) for line in info.splitlines() if "=" in line)
source_hash = info_fields["source_sha256"]
assert re.fullmatch(r"[0-9a-fA-F]{64}", source_hash), "missing/invalid source fingerprint"


def listed_kernels(output):
    entries = [dict(field.split("=", 1) for field in shlex.split(line))
               for line in output.splitlines() if line.startswith("kernel=")]
    names = [entry["kernel"] for entry in entries]
    assert len(names) == len(set(names)), "duplicate kernel registration"
    return {entry["kernel"]: entry for entry in entries}


kernel_listing = invoke(["--mode", "list"], True, "kernel=cublas")
kernels = listed_kernels(kernel_listing)
assert {"cublas", "naive", "coalesced", "shared", "register", "tuned", "warptiled",
        "doublebuffer", "doublebuffer_opt", "asyncpipeline"} <= kernels.keys()
assert kernels["cublas"]["available"] == "true"
doublebuffer_config = dict(field.split("=", 1)
                          for field in info_fields["doublebuffer_config"].split(";"))
doublebuffer_opt_config = dict(field.split("=", 1)
                              for field in info_fields["doublebuffer_opt_config"].split(";"))
assert all(doublebuffer_opt_config[key] == value for key, value in doublebuffer_config.items())
assert doublebuffer_opt_config["SWIZZLE_A"] in {"0", "1"}
assert doublebuffer_opt_config["VECTOR_STORE_C"] in {"0", "1"}
for name in ("doublebuffer", "doublebuffer_opt", "asyncpipeline"):
    assert name + "_config " + info_fields[name + "_config"] in kernel_listing
for name, entry in kernels.items():
    assert entry["available"] in {"true", "false"}, entry
    assert entry["source"], entry
    assert info_fields[name + "_available"] == entry["available"], (name, info)


def assert_valid_csv(rows, requested, actual, compared):
    assert [row["actual_kernel"] for row in rows] == actual, rows
    for row in rows:
        assert row["requested_kernel"] == requested
        assert row["status"] == "valid" and row["failed"] == "0" and row["nonfinite"] == "0"
        assert row["compared"] == str(compared)
        assert row["cpu_scope"] == "cpu_double_full" and row["cpu_failed"] == "0"
        assert row["cpu_compared"] == str(compared)
        assert row["source_sha256"] == source_hash, "CSV fingerprint disagrees with executable info"
        for param in ("bm", "bn", "bk", "tm", "tn"):
            field = "tuned_" + param
            expected = info_fields[field] if row["actual_kernel"] == "tuned" else ""
            assert row[field] == expected, "CSV tuned configuration disagrees with executable info"
        expected_warp = info_fields["warptiled_config"] if row["actual_kernel"] == "warptiled" else ""
        assert row["warptiled_config"] == expected_warp, "CSV warp configuration mismatch"
        expected_doublebuffer = (info_fields[row["actual_kernel"] + "_config"]
                                 if row["actual_kernel"] in {"doublebuffer", "doublebuffer_opt"} else "")
        assert row["doublebuffer_config"] == expected_doublebuffer, "CSV doublebuffer configuration mismatch"
        expected_async = (info_fields["asyncpipeline_config"]
                          if row["actual_kernel"] == "asyncpipeline" else "")
        assert row["asyncpipeline_config"] == expected_async, "CSV asyncpipeline configuration mismatch"
        assert len(row["round_mean_ms"].split(";")) == 2
        assert math.isfinite(float(row["median_ms"])) and float(row["median_ms"]) > 0
    assert float(rows[-1]["cublas_over_self"]) == 1


# Different grid extents (2 x 3 blocks for 32 x 32 tiles) catch swapped m/n
# launch dimensions; all dimensions also exercise non-tile-aligned boundaries.
shape = ["--m", "33", "--n", "65", "--k", "17"]

with tempfile.TemporaryDirectory(prefix="gemm-cli-") as folder:
    dest = Path(folder) / "tiny.csv"
    args = ["--mode", "bench", "--kernel", "cublas", "--m", "2", "--n", "2", "--k", "3",
            "--seed", "0", "--warmup", "0", "--iters", "2", "--rounds", "2", "--csv", str(dest)]
    invoke(args, True, "status=valid")
    original = dest.read_bytes()
    rows = list(csv.DictReader(io.StringIO(original.decode("utf-8"))))
    assert len(rows) == 1, "cuBLAS must not be benchmarked twice"
    assert_valid_csv(rows, "cublas", ["cublas"], 4)
    assert rows[0]["reference_scope"] == "cpu_double_full", "cuBLAS must have an independent oracle"
    invoke(args, False, "CSV already exists")
    assert dest.read_bytes() == original, "existing evidence was overwritten"

    for name, entry in kernels.items():
        if entry["available"] == "false":
            # Benchmark scripts must be able to inspect an unfinished kernel
            # without accidentally executing it or falling back to cuBLAS.
            preflight = invoke(["--mode", "list", "--kernel", name], True, "kernel=" + name)
            assert listed_kernels(preflight) == kernels
            rejected_csv = Path(folder) / (name + "-unavailable.csv")
            for mode in ("check", "bench", "profile"):
                args = ["--mode", mode, "--kernel", name, *shape]
                if mode == "bench":
                    args += ["--csv", str(rejected_csv)]
                output = invoke(args, False, name + " is not implemented")
                assert "check kernel=" not in output and "bench kernel=" not in output
                assert "profile requested=" not in output
                assert not rejected_csv.exists(), "unavailable kernel wrote benchmark evidence"
            continue

        if name != "cublas":
            invoke(["--mode", "check", "--kernel", name, *shape], True,
                   "check kernel=" + name + " shape=33x65x17 pattern=random status=valid")
            custom_csv = Path(folder) / (name + ".csv")
            invoke(["--mode", "bench", "--kernel", name, *shape, "--seed", "0",
                    "--warmup", "0", "--iters", "2", "--rounds", "2", "--csv", str(custom_csv)],
                   True, "bench kernel=" + name + " status=valid")
            with custom_csv.open(encoding="utf-8", newline="") as source:
                rows = list(csv.DictReader(source))
            assert_valid_csv(rows, name, [name, "cublas"], 33 * 65)
            assert rows[0]["reference_scope"] == "cublas_full"
            assert rows[1]["reference_scope"] == "cpu_double_full"

        profile = invoke(["--mode", "profile", "--kernel", name, *shape,
                          "--warmup", "0", "--iters", "2"], True, "target_operations=2")
        assert "profile requested=" + name + " actual=" + name in profile
        assert "validation=not_run" in profile
        assert "check kernel=" not in profile and "bench kernel=" not in profile
print(f"CLI contract passed ({count} invocations).")
