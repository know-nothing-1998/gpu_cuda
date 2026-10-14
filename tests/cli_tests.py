"""CLI integration tests: rejection paths, independent oracle, CSV, profile isolation."""
import csv
import io
import math
from pathlib import Path
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
    (["--kernel", "shared"], "unknown/unimplemented kernel"),
    (["--dtype", "fp16"], "unsupported dtype"),
    (["--extra", "1"], "unknown option"),
    (["--mode"], "missing value"),
    (["--seed", "1", "--seed", "2"], "duplicate option"),
    (["--mode", "check", "--m", "1", "--n", "1", "--k", "4097"], "outside the current numerical"),
    (["--mode", "profile", "--m", "1", "--n", "1", "--k", "1", "--rounds", "2"], "requires --rounds 1"),
    (["--mode", "check", "--m", "2147483647", "--n", "2147483647", "--k", "4096"], "total allocation overflow"),
]:
    invoke(args, False, message)

# Remains useful after the user implements naive; only check unavailable if it is unavailable.
info = invoke(["--mode", "info"], True, "CUBLAS_COMPUTE_32F_PEDANTIC")
if "naive_available=false" in info:
    invoke(["--mode", "check", "--kernel", "naive"], False, "naive is not implemented")

with tempfile.TemporaryDirectory(prefix="gemm-cli-") as folder:
    dest = Path(folder) / "tiny.csv"
    args = ["--mode", "bench", "--kernel", "cublas", "--m", "2", "--n", "2", "--k", "3",
            "--seed", "0", "--warmup", "0", "--iters", "2", "--rounds", "2", "--csv", str(dest)]
    invoke(args, True, "status=valid")
    original = dest.read_bytes()
    rows = list(csv.DictReader(io.StringIO(original.decode("utf-8"))))
    assert len(rows) == 1, "cuBLAS must not be benchmarked twice"
    row = rows[0]
    assert row["reference_scope"] == "cpu_double_full", "cuBLAS must have an independent oracle"
    assert row["compared"] == "4" and row["failed"] == "0"
    assert len(row["round_mean_ms"].split(";")) == 2
    assert float(row["cublas_over_self"]) == 1
    assert math.isfinite(float(row["median_ms"])) and float(row["median_ms"]) > 0
    invoke(args, False, "CSV already exists")
    assert dest.read_bytes() == original, "existing evidence was overwritten"

profile = invoke(["--mode", "profile", "--kernel", "cublas", "--m", "2", "--n", "3", "--k", "4",
                  "--warmup", "0", "--iters", "2"], True, "target_operations=2")
assert "check kernel=" not in profile and "bench kernel=" not in profile
print(f"CLI contract passed ({count} invocations).")
