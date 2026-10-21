"""CPU-only contracts for accepting and ranking autotuning measurements."""

import copy
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
import zipfile


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "gemm_autotune_under_test", ROOT / "scripts" / "autotune.py"
)
autotune = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = autotune
SPEC.loader.exec_module(autotune)

SOURCE_HASH = "a" * 64
SHAPE = {
    "id": "P1", "m": 2048, "n": 2048, "k": 2048,
    "seed": 42, "warmup": 100, "iters": 44, "rounds": 3,
}
# The complete CSV schema emitted by src/gemm.cu.
CSV_FIELDS = (
    "requested_kernel,actual_kernel,m,n,k,input_dtype,accum_dtype,output_dtype,"
    "seed,generator,status,reference_scope,compared,max_abs,relative_frob,"
    "failed,nonfinite,cpu_scope,cpu_compared,cpu_max_abs,cpu_relative_frob,"
    "cpu_failed,warmup,iters,rounds,median_ms,min_ms,max_ms,gflops,"
    "cublas_over_self,round_mean_ms,source_sha256,build_type,cuda_architectures,"
    "gpu,cuda_runtime,cuda_driver,cublas_version,math_mode,compute_type"
).split(",")


def benchmark_rows():
    rows = []
    for kernel, samples in (("tuned", [2.0, 3.0, 4.0]), ("cublas", [1.0, 1.5, 2.0])):
        row = dict.fromkeys(CSV_FIELDS, "0")
        row.update(
            requested_kernel="tuned", actual_kernel=kernel,
            input_dtype="fp32", accum_dtype="fp32", output_dtype="fp32",
            generator="mt19937_u24_v1", status="valid",
            reference_scope="cublas_full" if kernel == "tuned" else "cpu_sampled",
            compared=str(SHAPE["m"] * SHAPE["n"]),
            cpu_scope="sampled", cpu_compared="1024",
            median_ms=str(statistics.median(samples)),
            min_ms=str(min(samples)), max_ms=str(max(samples)),
            round_mean_ms=";".join(map(str, samples)), source_sha256=SOURCE_HASH,
            build_type="Release", cuda_architectures="89",
            gpu="NVIDIA GeForce RTX 4060 Laptop GPU",
            cuda_runtime="13020", cuda_driver="13020", cublas_version="130200",
            compute_type="CUBLAS_COMPUTE_32F_PEDANTIC",
        )
        row.update({key: str(value) for key, value in SHAPE.items() if key != "id"})
        rows.append(row)
    return rows


class BenchmarkCsvTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gemm-autotune-test-")
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "benchmark.csv"

    def read(self, rows):
        with self.path.open("w", encoding="utf-8-sig", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        return autotune.read_benchmark(
            self.path, SHAPE, SOURCE_HASH,
            SHAPE["warmup"], SHAPE["iters"], SHAPE["rounds"],
        )

    def test_accepts_actual_two_row_format_and_either_row_order(self):
        for rows in (benchmark_rows(), list(reversed(benchmark_rows()))):
            with self.subTest(first=rows[0]["actual_kernel"]):
                parsed = self.read(rows)
                self.assertEqual(set(parsed), {"tuned", "cublas"})
                self.assertEqual(parsed["tuned"]["samples_ms"], [2.0, 3.0, 4.0])
                self.assertEqual(parsed["tuned"]["median_ms"], 3.0)
                self.assertEqual(parsed["cublas"]["median_ms"], 1.5)

    def test_rejects_numerical_failure_in_either_row(self):
        for index in (0, 1):
            for field, value in (
                ("status", "invalid"), ("status", "failed"),
                ("failed", "1"), ("nonfinite", "1"), ("cpu_failed", "1"),
            ):
                with self.subTest(row=index, field=field, value=value):
                    rows = benchmark_rows()
                    rows[index][field] = value
                    with self.assertRaises(ValueError):
                        self.read(rows)

    def test_requires_full_cublas_comparison_and_nonempty_cpu_checks(self):
        invalid = (
            (0, "reference_scope", "sampled"),
            (0, "compared", "0"),
            (0, "compared", str(SHAPE["m"] * SHAPE["n"] - 1)),
            (0, "cpu_compared", "0"),
            (1, "cpu_compared", "0"),
        )
        for index, field, value in invalid:
            with self.subTest(row=index, field=field, value=value):
                rows = benchmark_rows()
                rows[index][field] = value
                with self.assertRaises(ValueError):
                    self.read(rows)

    def test_rejects_stale_source_or_incompatible_build_and_precision(self):
        for index in (0, 1):
            for field, value in (
                ("source_sha256", "b" * 64), ("build_type", "Debug"),
                ("requested_kernel", "register"),
                ("compute_type", "CUBLAS_COMPUTE_32F_FAST_TF32"),
                ("input_dtype", "fp16"), ("accum_dtype", "fp16"),
                ("output_dtype", "fp16"),
            ):
                with self.subTest(row=index, field=field):
                    rows = benchmark_rows()
                    rows[index][field] = value
                    with self.assertRaises(ValueError):
                        self.read(rows)

    def test_rejects_different_shape_seed_or_measurement_settings(self):
        for index in (0, 1):
            for field in ("m", "n", "k", "seed", "warmup", "iters", "rounds"):
                with self.subTest(row=index, field=field):
                    rows = benchmark_rows()
                    rows[index][field] = str(SHAPE[field] + 1)
                    with self.assertRaises(ValueError):
                        self.read(rows)

    def test_rejects_nonfinite_zero_negative_or_incomplete_samples(self):
        for index in (0, 1):
            for samples in ("2;nan;4", "2;inf;4", "2;0;4", "2;-1;4", "2;3"):
                with self.subTest(row=index, samples=samples):
                    rows = benchmark_rows()
                    rows[index]["round_mean_ms"] = samples
                    with self.assertRaises(ValueError):
                        self.read(rows)

    def test_rejects_median_not_supported_by_samples(self):
        for median in ("nan", "0", "0.001"):
            with self.subTest(median=median):
                rows = benchmark_rows()
                rows[0]["median_ms"] = median
                with self.assertRaises(ValueError):
                    self.read(rows)

    def test_requires_exactly_one_tuned_and_one_cublas_row(self):
        rows = benchmark_rows()
        for malformed in (rows[:1], rows[1:], [rows[0], rows[0]], rows + [rows[1]]):
            with self.subTest(kernels=[row["actual_kernel"] for row in malformed]):
                with self.assertRaises(ValueError):
                    self.read(malformed)


BASELINE = autotune.Config(64, 64, 64, 4, 4)
CANDIDATE = autotune.Config(64, 64, 16, 4, 4)
OTHER = autotune.Config(16, 64, 16, 2, 4)


def measurement(config, pass_number, ms, phase="confirm", case="P1"):
    return {
        "config": config.name, "case": case, "phase": phase, "pass": pass_number,
        "source_sha256": SOURCE_HASH,
        "csv": f"{config.name}-{case}-{phase}-{pass_number}.csv",
        "timing": {
            "tuned": {"samples_ms": [ms, ms, ms], "median_ms": ms},
            "cublas": {"samples_ms": [0.5, 0.5, 0.5], "median_ms": 0.5},
        },
    }


class RankingTests(unittest.TestCase):
    def setUp(self):
        self.baseline_records = [
            measurement(BASELINE, 1, 2.0), measurement(BASELINE, 2, 2.2)
        ]

    def rank(self, records, configs=(BASELINE, CANDIDATE, OTHER)):
        return autotune.ranking_for(SHAPE, configs, records, BASELINE.name)

    def test_ranks_confirmation_only_and_keeps_shapes_separate(self):
        records = self.baseline_records + [
            measurement(CANDIDATE, 1, 1.0), measurement(CANDIDATE, 2, 1.2),
            measurement(BASELINE, 1, 0.0001, phase="screen"),
            measurement(OTHER, 1, 0.0001, phase="screen"),
            measurement(OTHER, 1, 0.0001, case="P2"),
            measurement(OTHER, 2, 0.0001, case="P2"),
        ]
        ranked = self.rank(records)
        self.assertEqual([item["config"] for item in ranked],
                         [CANDIDATE.name, BASELINE.name])
        self.assertAlmostEqual(ranked[0]["median_ms"], 1.1)
        self.assertAlmostEqual(ranked[0]["baseline_over_self"], 2.1 / 1.1)
        self.assertAlmostEqual(ranked[0]["gflops"], 2 * 2048**3 / (1.1 * 1e6))
        self.assertTrue(all("-confirm-" in path for path in ranked[0]["csv_files"]))

    def test_missing_or_duplicate_pass_cannot_win(self):
        incomplete_sets = (
            [measurement(CANDIDATE, 1, 0.0001)],
            [measurement(CANDIDATE, 1, 0.0001), measurement(CANDIDATE, 1, 0.0001)],
            [measurement(CANDIDATE, 1, 0.0001), measurement(CANDIDATE, 3, 0.0001)],
        )
        for incomplete in incomplete_sets:
            with self.subTest(passes=[record["pass"] for record in incomplete]):
                ranked = self.rank(self.baseline_records + incomplete)
                self.assertEqual([item["config"] for item in ranked], [BASELINE.name])

    def test_excluded_failed_config_cannot_reenter_from_retained_records(self):
        # execute() removes failed candidates from its eligible configuration list.
        # Their earlier measurements may still exist in the audit trail.
        records = self.baseline_records + [
            measurement(CANDIDATE, 1, 0.0001), measurement(CANDIDATE, 2, 0.0001)
        ]
        ranked = self.rank(records, configs=[BASELINE])
        self.assertEqual([item["config"] for item in ranked], [BASELINE.name])

    def test_baseline_needs_complete_confirmation_even_if_screening_succeeded(self):
        records = [
            measurement(BASELINE, 1, 2.0, phase="screen"),
            measurement(BASELINE, 1, 2.0),
            measurement(CANDIDATE, 1, 1.0), measurement(CANDIDATE, 2, 1.0),
        ]
        with self.assertRaisesRegex(ValueError, "baseline"):
            self.rank(records)


class SearchSpaceTests(unittest.TestCase):
    def test_asymmetric_block_and_reduction_tiles_are_supported(self):
        for config in (
            BASELINE, CANDIDATE, OTHER,
            autotune.Config(128, 64, 16, 8, 4),
            autotune.Config(64, 64, 16, 8, 2),
        ):
            with self.subTest(config=config):
                self.assertIsNone(autotune.reject_reason(config))
        self.assertEqual(OTHER.threads, 128)
        self.assertEqual(OTHER.shared_bytes, 5120)

    def test_rejects_invalid_types_divisibility_alignment_and_resource_usage(self):
        invalid = (
            autotune.Config(0, 64, 16, 4, 4),
            autotune.Config(64, 64, 16, 0, 4),
            autotune.Config(True, 64, 16, 1, 4),
            autotune.Config(64, 64, 16, 4.0, 4),
            autotune.Config(64, 64, 16, 3, 4),
            autotune.Config(64, 64, 6, 4, 4),
            autotune.Config(64, 66, 16, 4, 2),
            autotune.Config(64, 64, 16, 1, 1),
            autotune.Config(128, 128, 64, 8, 8),
        )
        for config in invalid:
            with self.subTest(config=config):
                self.assertIsNotNone(autotune.reject_reason(config))
        self.assertIsNotNone(autotune.reject_reason(BASELINE, max_threads=128))
        self.assertIsNotNone(autotune.reject_reason(BASELINE, max_shared=16384))
        self.assertIsNotNone(autotune.reject_reason(
            autotune.Config(128, 128, 64, 8, 8), max_shared=102400
        ))

    def test_load_space_preserves_baseline_deduplicates_and_reports_rejections(self):
        spec = {
            "baseline": [64, 64, 64, 4, 4],
            "block_tiles": [[64, 64], [16, 64], [64, 64]],
            "bk": [16, 64],
            "thread_tiles": [[4, 4], [3, 4]],
        }
        with tempfile.TemporaryDirectory(prefix="gemm-autotune-space-") as temporary:
            path = Path(temporary) / "space.json"
            path.write_text(json.dumps(spec), encoding="utf-8-sig")
            baseline, valid, rejected = autotune.load_space(path)
            self.assertEqual(baseline, BASELINE)
            self.assertEqual(valid[0], BASELINE)
            self.assertEqual(len(valid), len(set(valid)))
            self.assertIn(autotune.Config(16, 64, 16, 4, 4), valid)
            self.assertTrue(rejected)
            self.assertTrue(all(item["reason"] for item in rejected))

            invalid_spec = copy.deepcopy(spec)
            invalid_spec["baseline"] = [64, 64, 64, 0, 4]
            path.write_text(json.dumps(invalid_spec), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "baseline"):
                autotune.load_space(path)

    def test_extra_configs_are_validated_and_deduplicated_with_generated_configs(self):
        spec = {
            "baseline": [64, 64, 64, 4, 4],
            "block_tiles": [[64, 64]], "bk": [64], "thread_tiles": [[4, 4]],
            "extra_configs": [
                [64, 64, 64, 4, 4],
                [16, 64, 16, 2, 4],
                [16, 64, 16, 2, 4],
                [64, 64, 6, 4, 4],
                [128, 128, 64, 8, 8],
            ],
        }
        with tempfile.TemporaryDirectory(prefix="gemm-autotune-extra-") as temporary:
            path = Path(temporary) / "space.json"
            path.write_text(json.dumps(spec), encoding="utf-8")
            baseline, valid, rejected = autotune.load_space(path)
        self.assertEqual(baseline, BASELINE)
        self.assertEqual(valid, [BASELINE, OTHER])
        self.assertEqual(
            {item["config"] for item in rejected},
            {autotune.Config(64, 64, 6, 4, 4).name,
             autotune.Config(128, 128, 64, 8, 8).name},
        )
        self.assertTrue(all(item["reason"] for item in rejected))


class ReuseBuildTests(unittest.TestCase):
    """Exercise the reuse entry points with archived bytes, never real executables."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gemm-autotune-reuse-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.previous = self.root / "previous"
        self.previous.mkdir()
        self.source_bytes = {
            "CMakeLists.txt": b"project(test_snapshot)\n",
            "src/tuned.cu": b"// original kernel\n",
            "src/tuned.cuh": b"// original interface\n",
            "benchmarks/matrix.csv": b"id,m,n,k\nP1,2048,2048,2048\n",
        }
        self.hashes = {
            name: hashlib.sha256(data).hexdigest()
            for name, data in self.source_bytes.items()
        }
        with zipfile.ZipFile(self.previous / "source-snapshot.zip", "w") as archive:
            for name, data in self.source_bytes.items():
                archive.writestr(name, data)
        self.write_json(self.previous / "source-sha256.json", self.hashes)
        self.exe = self.previous / "gemm.exe"
        self.exe.write_bytes(b"test fixture: never execute these bytes")
        self.previous_manifest = {
            "status": "complete",
            "device": {"compute_capability": "8.9"},
            "variants": {
                BASELINE.name: {
                    "status": "valid", "params": BASELINE.values(),
                    "exe": str(self.exe),
                    "exe_sha256": hashlib.sha256(self.exe.read_bytes()).hexdigest(),
                    "source_sha256": SOURCE_HASH,
                },
            },
        }
        self.write_json(self.previous / "manifest.json", self.previous_manifest)
        artifacts = self.previous / "variants" / BASELINE.name
        artifacts.mkdir(parents=True)
        (artifacts / "build_info.h").write_text(
            f'#define GEMM_SOURCE_SHA256 "{SOURCE_HASH}"\n', encoding="utf-8"
        )
        (artifacts / "tuned_config.h").write_text(
            "\n".join(f"inline constexpr int {key} = {value};"
                      for key, value in BASELINE.values().items()),
            encoding="utf-8",
        )
        for name in ("build.log", "configure.log"):
            (artifacts / name).write_text("fixture log\n", encoding="utf-8")
        self.search_number = 0

    @staticmethod
    def write_json(path, value):
        path.write_text(json.dumps(value), encoding="utf-8")

    def search(self, current_hashes=None):
        self.search_number += 1
        args = SimpleNamespace(
            output=str(self.root / f"new-results-{self.search_number}"),
            reuse_builds=self.previous, arch="89",
            search_seed=42, screen_rounds=3, top_k=3,
        )
        with mock.patch.object(autotune, "ROOT", self.root):
            search = autotune.Search(args, BASELINE, [BASELINE], [], [SHAPE])
        search.hashes = dict(self.hashes if current_hashes is None else current_hashes)
        return search

    def test_matrix_only_change_reuses_binary_and_preserves_original_evidence(self):
        current_hashes = dict(self.hashes)
        current_hashes["benchmarks/matrix.csv"] = hashlib.sha256(
            b"id,m,n,k\nP4,4096,4096,4096\n"
        ).hexdigest()
        search = self.search(current_hashes)
        search.prepare_reuse()
        with mock.patch.object(
            search, "run", side_effect=AssertionError("Reuse must not invoke a compiler")
        ) as run:
            saved = search.compile(BASELINE)
        run.assert_not_called()
        self.assertEqual(saved.read_bytes(), self.exe.read_bytes())
        self.assertNotEqual(saved, self.exe)
        variant = search.variants[BASELINE.name]
        self.assertEqual(variant["source_sha256"], SOURCE_HASH)
        with zipfile.ZipFile(search.out / variant["source_snapshot"]) as archive:
            self.assertEqual(archive.read("benchmarks/matrix.csv"),
                             self.source_bytes["benchmarks/matrix.csv"])
        self.assertEqual(search.records, [])

    def test_core_changes_additions_and_deletions_prevent_reuse(self):
        for label in ("CMakeLists.txt", "src/tuned.cu", "added", "deleted"):
            with self.subTest(change=label):
                current_hashes = dict(self.hashes)
                if label == "deleted":
                    del current_hashes["src/tuned.cuh"]
                else:
                    path = "src/new_kernel.cu" if label == "added" else label
                    current_hashes[path] = hashlib.sha256(b"changed source").hexdigest()
                with self.assertRaisesRegex(autotune.ReuseError, "core source mismatch"):
                    self.search(current_hashes).prepare_reuse()

    def test_incomplete_or_failed_previous_search_prevents_reuse(self):
        for status in ("running", "failed", "interrupted"):
            with self.subTest(status=status):
                previous = copy.deepcopy(self.previous_manifest)
                previous["status"] = status
                self.write_json(self.previous / "manifest.json", previous)
                with self.assertRaisesRegex(autotune.ReuseError, "completed previous search"):
                    self.search().prepare_reuse()

    def test_changed_binary_is_rejected_before_compilation_or_execution(self):
        search = self.search()
        search.prepare_reuse()
        self.exe.write_bytes(b"binary changed after the previous run")
        with mock.patch.object(
            search, "run", side_effect=AssertionError("Bad reuse must fail without compiling")
        ) as run:
            with self.assertRaisesRegex(autotune.ReuseError, "hash mismatch"):
                search.compile(BASELINE)
        run.assert_not_called()
        self.assertEqual(search.records, [])

    def test_source_archive_corruption_is_rejected_even_when_hash_list_matches(self):
        with zipfile.ZipFile(self.previous / "source-snapshot.zip", "w") as archive:
            for name, data in self.source_bytes.items():
                archive.writestr(name, b"tampered" if name == "src/tuned.cu" else data)
        with self.assertRaisesRegex(autotune.ReuseError, "archive/manifest hash mismatch"):
            self.search().prepare_reuse()


if __name__ == "__main__":
    unittest.main()
