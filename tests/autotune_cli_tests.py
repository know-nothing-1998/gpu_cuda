"""CPU-only contracts for the unified tuned/warptiled autotuning interface."""

from contextlib import ExitStack, redirect_stderr, redirect_stdout
import csv
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile
from unittest import mock


ROOT = Path(__file__).resolve().parent.parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT / "scripts"))
import autotune
import compare_warptiled
from experiment_sources import snapshot_sources


P4_ARGS = ["--m", "16", "--n", "256", "--k", "128"]
CUSTOM_ARGS = ["--m", "17", "--n", "257", "--k", "129"]


class SourceSnapshotTests(unittest.TestCase):
    def test_archive_build_tree_and_hashes_share_all_required_inputs(self):
        with tempfile.TemporaryDirectory(prefix="gemm-source-snapshot-") as temporary:
            root = Path(temporary)
            required = {
                "CMakeLists.txt": b"project(snapshot_fixture)\n",
                "src/warptiled.cu": b"// kernel\r\n",
                "src/warptiled_config.h.in": b"@GEMM_WARP_BM@\n",
                "src/warptiled_variants.inc.in": b"@GEMM_WARP_VARIANT_ENTRIES@\n",
                "benchmarks/warptiling.json": b'{"candidates": []}\n',
                "benchmarks/tuning.json": b"{}\n",
                "scripts/experiment_sources.py": b"# snapshot helper\n",
                "tests/example.py": b"# contract\n",
                "third_party/NOTICE.txt": "License notice: 字节\n".encode("utf-8"),
            }
            excluded = {
                "scripts/__pycache__/cached.pyc": b"cache",
                "build/old/gemm.exe": b"binary",
                "results/previous/source-snapshot.zip": b"old experiment",
            }
            for name, data in {**required, **excluded}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)
            source = root / "build/new/source"
            archive_path = root / "results/new/source-snapshot.zip"
            archive_path.parent.mkdir(parents=True)
            hashes = snapshot_sources(root, source, archive_path)
            self.assertEqual(hashes, {name: hashlib.sha256(data).hexdigest()
                                     for name, data in required.items()})
            with zipfile.ZipFile(archive_path) as archive:
                self.assertEqual(set(archive.namelist()), set(required))
                for name, data in required.items():
                    self.assertEqual(archive.read(name), data)
                    self.assertEqual((source / name).read_bytes(), data)


class UnifiedArgumentsTests(unittest.TestCase):
    def test_defaults_preserve_tuned_entry_point(self):
        args = autotune.arguments([])
        self.assertEqual(args.kernel, "tuned")
        self.assertEqual(args.space, ROOT / "benchmarks/tuning.json")
        self.assertEqual(autotune.resolve_shapes(args),
                         autotune.load_shapes(args.matrix, None))

    def test_kernel_selects_default_candidate_space(self):
        args = autotune.arguments(["--kernel", "warptiled"])
        self.assertEqual(args.space, ROOT / "benchmarks/warptiling.json")

    def test_explicit_candidate_space_is_preserved_for_both_kernels(self):
        for kernel in ("tuned", "warptiled"):
            with self.subTest(kernel=kernel):
                args = autotune.arguments(["--kernel", kernel, "--space", "my-space.json"])
                self.assertEqual(args.space, Path("my-space.json"))

    def test_p4_dimensions_inherit_matrix_identity_and_timing(self):
        args = autotune.arguments(P4_ARGS)
        shapes = autotune.resolve_shapes(args)
        self.assertEqual(shapes, autotune.load_shapes(args.matrix, ["P4"]))
        self.assertEqual(shapes[0]["iters"], 10000)

    def test_named_case_and_explicit_dimensions_resolve_identically(self):
        named = autotune.resolve_shapes(autotune.arguments(["--cases", "P4"]))
        explicit = autotune.resolve_shapes(autotune.arguments(P4_ARGS))
        self.assertEqual(named, explicit)

    def test_unknown_shape_gets_stable_id_and_documented_defaults(self):
        shapes = autotune.resolve_shapes(autotune.arguments(CUSTOM_ARGS))
        self.assertEqual(shapes, [{
            "id": "M17_N257_K129", "m": 17, "n": 257, "k": 129,
            "seed": 42, "warmup": 100, "iters": 1000, "rounds": 5,
        }])

    def test_overrides_apply_to_named_and_explicit_shapes(self):
        for selection in (["--cases", "P1", "P4"], P4_ARGS, CUSTOM_ARGS):
            with self.subTest(selection=selection):
                args = autotune.arguments(selection + [
                    "--seed", "4294967295", "--warmup", "0",
                    "--iters", "23", "--rounds", "7",
                ])
                for shape in autotune.resolve_shapes(args):
                    self.assertEqual(
                        {key: shape[key] for key in ("seed", "warmup", "iters", "rounds")},
                        {"seed": 2**32 - 1, "warmup": 0, "iters": 23, "rounds": 7},
                    )

    def test_rejects_partial_dimensions_and_case_dimension_ambiguity(self):
        for argv in (
            ["--m", "16"], ["--n", "256", "--k", "128"],
            P4_ARGS + ["--cases", "P4"],
        ):
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    autotune.arguments(argv)
                self.assertEqual(error.exception.code, 2)

    def test_rejects_invalid_shapes_and_measurement_options(self):
        invalid = (
            ["--m", "0", "--n", "256", "--k", "128"],
            ["--m", "-1", "--n", "256", "--k", "128"],
            ["--m", "16", "--n", "256", "--k", "4097"],
            ["--m", "65536", "--n", "65536", "--k", "1"],
            ["--m", "1048576", "--n", "1", "--k", "4096"],
            ["--m", "1", "--n", "1048576", "--k", "4096"],
            ["--iters", "0"], ["--rounds", "-1"], ["--warmup", "-1"],
            ["--seed", "-1"], ["--seed", "4294967296"],
            ["--max-candidates", "0"], ["--max-candidates", "-1"],
            ["--screen-rounds", "0"], ["--top-k", "0"], ["--timeout", "0"],
        )
        for argv in invalid:
            with self.subTest(argv=argv), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    autotune.arguments(argv)
                self.assertEqual(error.exception.code, 2)

    def test_reuse_builds_remains_tuned_only(self):
        args = autotune.arguments(["--reuse-builds", "previous-run"])
        self.assertEqual(args.reuse_builds, Path("previous-run"))
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            autotune.arguments(["--kernel", "warptiled", "--reuse-builds", "previous-run"])
        self.assertEqual(error.exception.code, 2)

    def test_unknown_case_returns_error_without_starting_search(self):
        with mock.patch.object(autotune, "Search") as search, \
                redirect_stderr(io.StringIO()) as errors:
            result = autotune.main(["--cases", "DOES_NOT_EXIST", "--dry-run"])
        self.assertEqual(result, 1)
        self.assertIn("unknown case", errors.getvalue())
        search.assert_not_called()


class SelectedMatrixTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="gemm-autotune-cli-")
        self.addCleanup(temporary.cleanup)
        self.folder = Path(temporary.name)

    def test_writer_records_only_selected_shapes_and_overrides(self):
        args = autotune.arguments(P4_ARGS + ["--iters", "27", "--rounds", "9", "--seed", "7"])
        shapes = autotune.resolve_shapes(args)
        path = self.folder / "matrix.csv"
        autotune.write_matrix(path, shapes)
        self.assertEqual(autotune.load_shapes(path, None), shapes)
        with path.open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["id"], "P4")
        self.assertEqual(rows[0]["dtype"], "fp32")
        self.assertEqual(rows[0]["iters"], "27")
        self.assertEqual(rows[0]["rounds"], "9")
        self.assertEqual(rows[0]["seed"], "7")

    def test_duplicate_matching_shapes_require_named_case(self):
        first = autotune.resolve_shapes(autotune.arguments(P4_ARGS))[0]
        second = {**first, "id": "P4_ALTERNATE", "iters": 99}
        matrix = self.folder / "ambiguous.csv"
        autotune.write_matrix(matrix, [first, second])
        with self.assertRaisesRegex(ValueError, "multiple matrix cases"):
            autotune.resolve_shapes(autotune.arguments(P4_ARGS + ["--matrix", str(matrix)]))
        named = autotune.resolve_shapes(autotune.arguments([
            "--matrix", str(matrix), "--cases", "P4_ALTERNATE",
        ]))
        self.assertEqual(named, [second])

    def test_custom_shape_warp_plan_falls_back_to_tuned_default(self):
        args = autotune.arguments(["--kernel", "warptiled"] + CUSTOM_ARGS)
        shapes = autotune.resolve_shapes(args)
        configs, default, winners, planned = compare_warptiled.load_plan(
            args.space, args.matrix, args.cases, shapes=shapes,
        )
        self.assertTrue(configs)
        self.assertEqual(planned, shapes)
        self.assertEqual(winners, {"M17_N257_K129": default})


class DryRunTests(unittest.TestCase):
    def plan(self, kernel, selection):
        with tempfile.TemporaryDirectory(prefix="gemm-autotune-dry-") as folder:
            output = Path(folder) / "must-not-be-created"
            stdout, stderr = io.StringIO(), io.StringIO()
            with ExitStack() as stack:
                stack.enter_context(redirect_stdout(stdout))
                stack.enter_context(redirect_stderr(stderr))
                guards = [
                    stack.enter_context(mock.patch.object(autotune, "Search")),
                    stack.enter_context(mock.patch.object(compare_warptiled, "Experiment")),
                    stack.enter_context(mock.patch("subprocess.run")),
                    stack.enter_context(mock.patch("subprocess.Popen")),
                    stack.enter_context(mock.patch.object(Path, "mkdir")),
                    stack.enter_context(mock.patch.object(Path, "write_text")),
                ]
                result = autotune.main([
                    "--kernel", kernel, "--dry-run", "--max-candidates", "1",
                    "--output", str(output),
                ] + selection)
                for guard in guards:
                    guard.assert_not_called()
            self.assertEqual(result, 0, stderr.getvalue())
            self.assertFalse(output.exists())
            self.assertEqual(list(Path(folder).iterdir()), [])
            plan = json.loads(stdout.getvalue())
            self.assertEqual(plan["kernel"], kernel)
            return plan

    def test_both_kernels_preview_p4_without_build_or_file_effects(self):
        for kernel in ("tuned", "warptiled"):
            with self.subTest(kernel=kernel):
                plan = self.plan(kernel, P4_ARGS)
                self.assertEqual(plan["cases"][0]["id"], "P4")
                self.assertEqual(plan["cases"][0]["iters"], 10000)
                candidates = plan["candidates"] if kernel == "tuned" else plan["warp_candidates"]
                self.assertEqual(len(candidates), 1)

    def test_both_kernels_preview_custom_shape_and_overrides(self):
        for kernel in ("tuned", "warptiled"):
            with self.subTest(kernel=kernel):
                plan = self.plan(kernel, CUSTOM_ARGS + ["--iters", "33", "--rounds", "7"])
                self.assertEqual(plan["cases"], [{
                    "id": "M17_N257_K129", "m": 17, "n": 257, "k": 129,
                    "seed": 42, "warmup": 100, "iters": 33, "rounds": 7,
                }])


if __name__ == "__main__":
    unittest.main()
