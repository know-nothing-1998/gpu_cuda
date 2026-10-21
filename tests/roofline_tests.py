"""CPU-only archive/schema/CLI checks; no NCU, CUDA or matplotlib required."""

import copy
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import roofline


def sample_model():
    return {
        "schema_version": 1,
        "shape": {"m": 2048, "n": 2048, "k": 2048},
        "generated_utc": "2026-10-22T01:00:00+00:00",
        "reports": [{
            "label": "naive", "gpu": "test SM 8.9",
            "derived": {
                "fp32_peak_gflops": 10000.0,
                "dram_peak_gbytes_per_second": 250.0,
                "ridge_ai_flop_per_byte": 40.0,
                "hw_gflops": 1000.0,
                "hw_ai_flop_per_byte": 20.0,
                "hw_peak_percent": 10.0,
                "dram_write_bytes": 0,
            },
        }],
    }


class ModelTests(unittest.TestCase):
    def test_valid_archive_and_zero_optional_traffic(self):
        model = sample_model()
        before = copy.deepcopy(model)
        self.assertIs(roofline.validate_model(model), model)
        self.assertEqual(model, before)

    def test_required_plot_values_reject_invalid_numbers(self):
        for field in roofline.PLOT_FIELDS:
            for value in (None, True, "100", 0, -1, float("nan"), float("inf"), 10 ** 400):
                with self.subTest(field=field, value=str(value)[:20]):
                    model = sample_model()
                    model["reports"][0]["derived"][field] = value
                    with self.assertRaisesRegex(ValueError, "finite and positive"):
                        roofline.validate_model(model)

    def test_shape_requires_positive_integer_dimensions(self):
        for shape in (None, {}, {"m": 1, "n": 2}, {"m": True, "n": 2, "k": 3},
                      {"m": 1.5, "n": 2, "k": 3}, {"m": 0, "n": 2, "k": 3}):
            with self.subTest(shape=shape):
                model = sample_model()
                model["shape"] = shape
                with self.assertRaisesRegex(ValueError, "positive integer"):
                    roofline.validate_model(model)

    def test_schema_reports_and_metadata_rejected(self):
        for field, value in (("schema_version", 2), ("schema_version", True),
                             ("reports", []), ("reports", {}), ("reports", [None]),
                             ("comparison_note", []), ("title_note", 10)):
            with self.subTest(field=field, value=value):
                model = sample_model()
                model[field] = value
                with self.assertRaises(ValueError):
                    roofline.validate_model(model)
        for field in ("label", "gpu", "derived"):
            model = sample_model()
            del model["reports"][0][field]
            with self.assertRaises(ValueError):
                roofline.validate_model(model)

    def test_duplicate_labels_ignore_case_and_whitespace(self):
        model = sample_model()
        model["reports"].append(copy.deepcopy(model["reports"][0]))
        model["reports"][1]["label"] = " NAIVE "
        with self.assertRaisesRegex(ValueError, "Duplicate report label"):
            roofline.validate_model(model)

    def test_derived_units_must_be_consistent(self):
        for field in ("ridge_ai_flop_per_byte", "hw_peak_percent"):
            model = sample_model()
            model["reports"][0]["derived"][field] *= 1000
            with self.assertRaisesRegex(ValueError, "disagrees"):
                roofline.validate_model(model)


class WrappedCommandTests(unittest.TestCase):
    def test_fixed_width_ncu_session_preserves_argument_spaces(self):
        prefix = "Profiler Command Line "
        command = ("ncu --config-file off " + "x" * 99 + " --section LaunchStats "
                   "--replay-mode kernel --export D:\\gpu_cuda\\results\\p1 "
                   "build\\gemm.exe --mode profile --kernel warptiled "
                   "--m 2048 --n 2048 --k 2048 --warmup 20 --iters 1")
        chunks = [command[index:index + 120] for index in range(0, len(command), 120)]
        session = ("Session Info\nprivate metadata\n\nLaunch Settings\n"
                   + prefix + chunks[0] + "\n"
                   + "\n".join(" " * len(prefix) + chunk for chunk in chunks[1:])
                   + "    \n\nOther Setting value\n")
        wrapped = roofline.profiler_command(session)
        self.assertEqual(wrapped.replace("\n", ""), command)
        checked = roofline.check_shape(wrapped, {"m": 2048, "n": 2048, "k": 2048})
        self.assertEqual(checked["status"], "matched_report_command")

    def test_dimension_digits_options_and_paths_can_wrap(self):
        command = ("ncu --config-file off \n--section LaunchStats\n --replay-mode kernel "
                   "--export D:\\gpu_cuda\\r\nesults\\profile -\n-mode profile "
                   "--m 2048 --n 204\n8 --k 2048 --warmup 20    ")
        prefix = "Profiler Command Line "
        lines = command.splitlines()
        session = ("Launch Settings\n" + prefix + lines[0] + "\n"
                   + "\n".join(" " * len(prefix) + line for line in lines[1:]) + "\n\n")
        wrapped = roofline.profiler_command(session)
        self.assertEqual(wrapped, command.rstrip())
        checked = roofline.check_shape(wrapped, {"m": 2048, "n": 2048, "k": 2048})
        self.assertEqual(checked["matched_dimensions"], {"m": 2048, "n": 2048, "k": 2048})
        with self.assertRaisesRegex(ValueError, "Declared n=204 disagrees"):
            roofline.check_shape(wrapped, {"m": 2048, "n": 204, "k": 2048})


class OfflineCliTests(unittest.TestCase):
    def test_report_mode_keeps_default_shape_and_import_path(self):
        with tempfile.TemporaryDirectory(prefix="gemm-roofline-test-") as folder:
            output = Path(folder)
            with mock.patch.object(sys, "argv", ["roofline.py", "--report", "naive=example.ncu-rep",
                                                  "--output", str(output)]), \
                    mock.patch.object(roofline, "find_ncu", return_value=Path("ncu.exe")), \
                    mock.patch.object(roofline, "run_ncu", return_value="test NCU version"), \
                    mock.patch.object(roofline, "analyze", return_value=sample_model()["reports"][0]) as analyze, \
                    mock.patch.object(roofline, "plot_model"), \
                    mock.patch("sys.stdout", new=io.StringIO()):
                roofline.main()
            self.assertEqual(analyze.call_args.args[:3],
                             ("naive", Path("example.ncu-rep").resolve(),
                              {"m": 2048, "n": 2048, "k": 2048}))
            recorded = json.loads((output / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(recorded["useful_gemm_flop"], 2 * 2048 ** 3)

    def test_offline_redraw_preserves_provenance_without_ncu(self):
        with tempfile.TemporaryDirectory(prefix="gemm-roofline-test-") as folder:
            root = Path(folder)
            archive = root / "archive.json"
            model = sample_model()
            original = json.dumps(model, ensure_ascii=False, separators=(",", ":"))
            archive.write_text(original, encoding="utf-8")
            output = root / "redrawn"
            with mock.patch.object(sys, "argv", ["roofline.py", "--model", str(archive),
                                                  "--output", str(output)]), \
                    mock.patch.object(roofline, "find_ncu", side_effect=AssertionError("NCU called")), \
                    mock.patch.object(roofline, "run_ncu", side_effect=AssertionError("NCU called")), \
                    mock.patch.object(roofline, "plot_model") as plot, \
                    mock.patch("sys.stdout", new=io.StringIO()):
                roofline.main()
            plot.assert_called_once_with(model, output.resolve())
            self.assertEqual(json.loads((output / "model.json").read_text(encoding="utf-8")), model)
            self.assertEqual(archive.read_text(encoding="utf-8"), original)

    def test_redraw_in_archive_directory_preserves_original_file(self):
        with tempfile.TemporaryDirectory(prefix="gemm-roofline-test-") as folder:
            archive = Path(folder) / "model.json"
            original = json.dumps(sample_model(), separators=(",", ":"))
            archive.write_text(original, encoding="utf-8")
            with mock.patch.object(sys, "argv", ["roofline.py", "--model", str(archive)]), \
                    mock.patch.object(roofline, "plot_model") as plot, \
                    mock.patch("sys.stdout", new=io.StringIO()):
                roofline.main()
            self.assertEqual(plot.call_args.args[1], archive.parent.resolve())
            self.assertEqual(archive.read_text(encoding="utf-8"), original)

    def test_model_rejects_report_shape_and_ncu_arguments(self):
        for extra in (["--report", "naive=x.ncu-rep"], ["--m", "2048"],
                      ["--n", "2048"], ["--k", "2048"], ["--ncu", "ncu.exe"]):
            with self.subTest(extra=extra), \
                    mock.patch.object(sys, "argv", ["roofline.py", "--model", "missing.json", *extra]), \
                    mock.patch("sys.stderr", new=io.StringIO()):
                with self.assertRaises(SystemExit) as raised:
                    roofline.main()
                self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
