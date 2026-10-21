"""CPU-only tests for selecting precompiled warp configurations from gemm CLI.

Only --help and --mode list are invoked, including all rejection cases. These
branches must remain ahead of CUDA context construction in the executable.
"""

from pathlib import Path
import re
import subprocess
import sys


PARAMETERS = ("BM", "BN", "BK", "WM", "WN", "TM", "TN", "WNITER")
P4_CONFIGURATION = (16, 32, 32, 8, 16, 2, 2, 1)
BASE_ARGS = ["--mode", "list", "--kernel", "warptiled"]


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: runtime_warp_cli_tests.py path/to/gemm.exe")
    executable = str(Path(sys.argv[1]).resolve())
    invocations = 0

    def invoke(arguments, success=True, expected=None):
        nonlocal invocations
        assert arguments == ["--help"] or arguments[:2] == ["--mode", "list"]
        result = subprocess.run(
            [executable, *arguments], capture_output=True, text=True, timeout=30
        )
        output = result.stdout + result.stderr
        assert (result.returncode == 0) == success, (arguments, result.returncode, output)
        if expected is not None:
            assert expected in output, (arguments, expected, output)
        assert "check kernel=" not in output and "bench kernel=" not in output
        assert "profile requested=" not in output
        invocations += 1
        return output

    def flags(configuration):
        return [part for name, value in zip(PARAMETERS, configuration)
                for part in ("--" + name.lower(), str(value))]

    def records(output, prefix):
        values = []
        for line in output.splitlines():
            if not line.startswith(prefix + " "):
                continue
            content = line[len(prefix) + 1:]
            pairs = [field.split("=", 1) for field in content.split(";")]
            assert all(len(pair) == 2 for pair in pairs), line
            fields = dict(pairs)
            assert len(fields) == len(pairs), line
            assert set(fields) == set(PARAMETERS) | {"THREADS"}, line
            assert all(re.fullmatch(r"[1-9][0-9]*", value) for value in fields.values()), line
            configuration = tuple(int(fields[name]) for name in PARAMETERS)
            bm, bn, bk, wm, wn, tm, tn, wniter = configuration
            threads = int(fields["THREADS"])
            assert bm % wm == 0 and bn % wn == 0, line
            assert threads == 32 * (bm // wm) * (bn // wn), line
            assert threads <= 1024, line
            wmiters, remainder = divmod(wm * wn, 32 * tm * tn * wniter)
            assert wmiters > 0 and remainder == 0, line
            assert bn % (tn * wniter) == 0 and bm % (tm * wmiters) == 0, line
            assert (bn // (tn * wniter)) * (bm // (tm * wmiters)) == threads, line
            assert bk % 4 == 0, line
            values.append(configuration)
        return values

    def selection(output):
        selected = records(output, "selected_warptiled_config")
        assert len(selected) == 1, output
        return selected[0]

    help_output = invoke(["--help"])
    for name in PARAMETERS:
        assert "--" + name.lower() in help_output, help_output

    initial = invoke(BASE_ARGS, expected="kernel=warptiled")
    configurations = records(initial, "warptiled_config")
    assert configurations, initial
    assert len(configurations) == len(set(configurations)), "duplicate compiled configuration"
    default = selection(initial)
    assert default in configurations, "default must be selectable from the compiled table"
    assert selection(invoke(["--mode", "list"])) == default

    # Every advertised specialization must be resolvable, with the selected
    # configuration matching the request and the available table unchanged.
    for configuration in configurations:
        output = invoke(BASE_ARGS + flags(configuration))
        assert selection(output) == configuration, output
        assert records(output, "warptiled_config") == configurations, output

    # A build with the optional table disabled still supports its one compiled
    # default; requests for other tuples must fail rather than silently fallback.
    if P4_CONFIGURATION in configurations:
        assert selection(invoke(BASE_ARGS + flags(P4_CONFIGURATION))) == P4_CONFIGURATION
    else:
        invoke(BASE_ARGS + flags(P4_CONFIGURATION), False, "not compiled")

    # The tuple is structurally valid. Choose a BK not present in this binary so
    # this check also survives builds that select a different CMake default.
    unsupported = next(
        (16, 32, bk, 8, 16, 2, 2, 1)
        for bk in range(4, 129, 4)
        if (16, 32, bk, 8, 16, 2, 2, 1) not in configurations
    )
    invoke(BASE_ARGS + flags(unsupported), False, "not compiled")

    full_flags = flags(default)
    for index, name in enumerate(PARAMETERS):
        option = "--" + name.lower()
        invoke(BASE_ARGS + [option, str(default[index])], False, "specify all")
        invoke(BASE_ARGS + full_flags[:2 * index] + full_flags[2 * index + 2:],
               False, "specify all")
        invoke(BASE_ARGS + full_flags + [option, str(default[index])],
               False, "duplicate option")
        invoke(BASE_ARGS + [option], False, "missing value")
        for invalid, expected in (("0", "must be positive"),
                                  ("-1", "unsigned integer"),
                                  ("1.5", "unsigned integer"),
                                  ("2147483648", "out of range")):
            changed = list(full_flags)
            changed[2 * index + 1] = invalid
            invoke(BASE_ARGS + changed, False, expected)

    for kernel in ("naive", "coalesced", "shared", "register", "tuned", "cublas"):
        invoke(["--mode", "list", "--kernel", kernel, *full_flags],
               False, "requires --kernel warptiled")

    # Configuration selection is process-local; a later plain invocation must
    # still use the compiled default rather than retaining the prior selection.
    assert selection(invoke(BASE_ARGS)) == default
    print(f"Runtime warp CLI contract passed ({invocations} CPU-only invocations).")


if __name__ == "__main__":
    main()
