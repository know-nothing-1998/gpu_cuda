"""CPU-only contracts for the single-configuration benchmark command."""

from contextlib import redirect_stderr, redirect_stdout
import csv
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock


SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
sys.path.insert(0, str(SCRIPTS))
import bench_tuned as bench
sys.path.pop(0)


ARGS = [
    "--bm", "16", "--bn", "32", "--bk", "32", "--tm", "2", "--tn", "2",
    "--m", "65", "--n", "97", "--k", "33",
]
SOURCE_HASH = "a" * 64


def info_text(config, **changes):
    fields = {
        "source_sha256": SOURCE_HASH, "build_type": "Release",
        "compute_capability": "8.9",
        **{"tuned_" + key.lower(): str(value) for key, value in config.values().items()},
    }
    fields.update(changes)
    return "\n".join(f"{key}={value}" for key, value in fields.items())


def write_config_csv(path, config, **changes):
    fields = ["actual_kernel"] + ["tuned_" + key.lower() for key in bench.PARAMS]
    tuned = {
        "actual_kernel": "tuned",
        **{"tuned_" + key.lower(): str(value) for key, value in config.values().items()},
    }
    tuned.update(changes)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows([tuned, {"actual_kernel": "cublas"}])


class ArgumentTests(unittest.TestCase):
    def test_defaults_and_compile_time_parameter_flags(self):
        args = bench.arguments(ARGS)
        config, shape, _, build, _, commands = bench.plan_for(args)
        self.assertEqual((args.warmup, args.iters, args.rounds), (100, 1000, 5))
        self.assertEqual(shape, {"m": 65, "n": 97, "k": 33, "seed": 42})
        self.assertEqual(build, bench.ROOT / "build" / "manual-tuned")
        for flag in config.cmake_flags():
            self.assertIn(flag, commands["configure"])
            self.assertNotIn(flag, commands["benchmark"])
        self.assertEqual(commands["build"][-2:], ["--target", "gemm"])
        self.assertIn("-DCMAKE_BUILD_TYPE=Release", commands["configure"])

    def test_rejects_invalid_parameters_shapes_and_measurement_counts(self):
        for override in (
            ["--tm", "0"], ["--tm", "3"], ["--bk", "6"],
            ["--bm", "128", "--bn", "128", "--bk", "64", "--tm", "8", "--tn", "8"],
            ["--bm", "64", "--bn", "64", "--tm", "1", "--tn", "1"],
            ["--m", "0"], ["--k", "4097"], ["--m", "65536", "--n", "65536"],
            ["--seed", "-1"], ["--seed", str(2**32)], ["--warmup", "-1"],
            ["--iters", "0"], ["--rounds", "0"], ["--timeout", "0"], ["--arch", "abc"],
        ):
            with self.subTest(override=override), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.arguments(ARGS + override)
                self.assertEqual(error.exception.code, 2)

    def test_all_five_parameters_and_three_dimensions_are_required(self):
        for index in range(0, len(ARGS), 2):
            with self.subTest(missing=ARGS[index]), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    bench.arguments(ARGS[:index] + ARGS[index + 2:])
                self.assertEqual(error.exception.code, 2)


class MetadataTests(unittest.TestCase):
    def setUp(self):
        self.config = bench.Config(16, 32, 32, 2, 2)

    def test_info_must_match_requested_parameters_source_build_and_device(self):
        bench.verify_info(info_text(self.config), self.config, SOURCE_HASH, "89")
        for key, value in (
            ("tuned_bm", "64"), ("tuned_bn", "64"), ("tuned_bk", "16"),
            ("tuned_tm", "4"), ("tuned_tn", "4"),
            ("source_sha256", "b" * 64), ("build_type", "Debug"),
            ("compute_capability", "8.6"),
        ):
            with self.subTest(field=key):
                with self.assertRaises(ValueError):
                    bench.verify_info(
                        info_text(self.config, **{key: value}),
                        self.config, SOURCE_HASH, "89",
                    )

    def test_csv_requires_compiled_parameters_on_the_tuned_row(self):
        with tempfile.TemporaryDirectory(prefix="gemm-manual-csv-") as temporary:
            path = Path(temporary) / "benchmark.csv"
            write_config_csv(path, self.config)
            bench.verify_csv_config(path, self.config)
            for value in ("64", ""):
                with self.subTest(value=value):
                    write_config_csv(path, self.config, tuned_bm=value)
                    with self.assertRaises(ValueError):
                        bench.verify_csv_config(path, self.config)


class CommandTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gemm-manual-command-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patcher = mock.patch.object(bench, "ROOT", self.root)
        patcher.start()
        self.addCleanup(patcher.stop)

    def main(self, argv):
        with redirect_stdout(io.StringIO()) as out, redirect_stderr(io.StringIO()) as err:
            code = bench.main(argv)
        return code, out.getvalue(), err.getvalue()

    def test_dry_run_creates_nothing_and_does_not_query_tools_or_gpu(self):
        with mock.patch.object(
            bench.subprocess, "run", side_effect=AssertionError("No subprocess during dry-run")
        ):
            code, output, _ = self.main(ARGS + ["--dry-run"])
        self.assertEqual(code, 0)
        plan = json.loads(output)
        self.assertEqual(plan["parameters"], {"BM": 16, "BN": 32, "BK": 32, "TM": 2, "TN": 2})
        self.assertEqual(list(self.root.iterdir()), [])

    def test_existing_results_are_not_overwritten(self):
        output = self.root / "existing"
        output.mkdir()
        marker = output / "keep.txt"
        marker.write_text("keep", encoding="utf-8")
        with mock.patch.object(
            bench.subprocess, "run", side_effect=AssertionError("Do not start a rejected run")
        ):
            code, _, error = self.main(ARGS + ["--output", str(output)])
        self.assertEqual(code, 1)
        self.assertIn("already exists", error)
        self.assertEqual(marker.read_text(encoding="utf-8"), "keep")
        self.assertFalse((self.root / "build").exists())

    def test_success_records_validated_pair_and_releases_build_lock(self):
        output = self.root / "results"
        args = bench.arguments(ARGS + ["--output", str(output)])
        config, shape, _, build, exe, commands = bench.plan_for(args)
        calls = []

        def run(runner, argv, label, private_output=False):
            calls.append((label, argv))
            if label == "build":
                (build / "generated").mkdir()
                (build / "generated/build_info.h").write_text(
                    f'#define GEMM_SOURCE_SHA256 "{SOURCE_HASH}"\n', encoding="utf-8"
                )
                (build / "generated/tuned_config.h").write_text("fixture\n", encoding="utf-8")
                exe.write_bytes(b"test executable bytes; never run")
            elif label == "info" or label == "device-info":
                return info_text(config).encode("utf-8")
            elif label == "benchmark":
                write_config_csv(output / "benchmark.csv", config)
            return b""

        timing = {
            "tuned": {"median_ms": 2.0, "samples_ms": [2.0] * 5},
            "cublas": {"median_ms": 1.0, "samples_ms": [1.0] * 5},
        }
        with mock.patch.object(bench, "compiler_environment"), \
                mock.patch.object(bench.Runner, "run", run), \
                mock.patch.object(bench, "read_benchmark", return_value=timing) as validate:
            code, text, _ = self.main(ARGS + ["--output", str(output)])
        self.assertEqual(code, 0)
        validate.assert_called_once_with(
            output / "benchmark.csv", shape, SOURCE_HASH, 100, 1000, 5
        )
        self.assertEqual([label for label, _ in calls],
                         ["configure", "build", "device-info", "benchmark"])
        self.assertEqual(calls[0][1], commands["configure"])
        record = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["parameters"], config.values())
        self.assertEqual(record["source_sha256"], SOURCE_HASH)
        self.assertEqual(record["cublas_over_self"], 0.5)
        self.assertAlmostEqual(record["results"]["tuned"]["gflops"], 2 * 65 * 97 * 33 / 2e6)
        self.assertIn("cuBLAS_ms / tuned_ms: 0.500x", text)
        self.assertFalse((build / ".bench-tuned.lock").exists())

    def test_failed_build_is_fatal_and_never_starts_benchmark(self):
        output = self.root / "failed"
        failure = subprocess.CompletedProcess(["cmake"], 2, b"compiler error")
        with mock.patch.object(bench, "compiler_environment"), \
                mock.patch.object(bench.subprocess, "run", return_value=failure) as run:
            code, _, _ = self.main(ARGS + ["--output", str(output)])
        self.assertEqual(code, 1)
        self.assertEqual(run.call_count, 1)
        record = json.loads((output / "config.json").read_text(encoding="utf-8"))
        self.assertEqual(record["status"], "failed")
        self.assertFalse((output / "benchmark.csv").exists())
        self.assertFalse((self.root / "build/manual-tuned/.bench-tuned.lock").exists())

    def test_timeout_or_interrupt_retains_lock_and_records_unsuccessful_status(self):
        failures = (
            ("timeout", subprocess.TimeoutExpired("cmake", 1, output=b"partial"), 1, "failed"),
            ("interrupt", KeyboardInterrupt(), 130, "interrupted"),
        )
        for label, failure, expected_code, expected_status in failures:
            with self.subTest(failure=label), tempfile.TemporaryDirectory(
                prefix="gemm-manual-interrupted-"
            ) as temporary:
                root = Path(temporary)
                output = root / "results"
                with mock.patch.object(bench, "ROOT", root), \
                        mock.patch.object(bench, "compiler_environment"), \
                        mock.patch.object(bench.subprocess, "run", side_effect=failure):
                    code, _, _ = self.main(ARGS + ["--output", str(output)])
                self.assertEqual(code, expected_code)
                record = json.loads((output / "config.json").read_text(encoding="utf-8"))
                self.assertEqual(record["status"], expected_status)
                self.assertTrue((root / "build/manual-tuned/.bench-tuned.lock").exists())
                self.assertFalse((output / "benchmark.csv").exists())

    def test_environment_capture_is_not_written_to_logs(self):
        output = self.root / "private"
        output.mkdir()
        runner = bench.Runner(output, 1)
        secret_output = b"EXAMPLE_TOKEN=do-not-persist\nPATH=tools\n"
        process = subprocess.CompletedProcess(["cmd"], 0, secret_output)
        with mock.patch.object(bench.subprocess, "run", return_value=process):
            self.assertEqual(runner.run(["cmd"], "compiler-env", private_output=True),
                             secret_output)
        for path in output.iterdir():
            self.assertNotIn("do-not-persist", path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
