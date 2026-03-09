"""
Test execution via LiveCodeBench-style runs: subprocess, stdin/stdout, no exec/eval.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

from config import SUBPROCESS_TIMEOUT


def run_tests(
    task_id: str,
    code: str,
    problem: dict[str, Any],
    timeout: int | None = None,
) -> dict:
    """
    Run tests for a solution using LiveCodeBench test cases (subprocess, stdin/stdout).

    Args:
        task_id: Problem identifier (e.g. LCB question_id).
        code: Full solution code (run as script; stdin = test input).
        problem: Dict with public_test_cases (and optionally private_test_cases),
                 each a list of {"input": str, "output": str}.
        timeout: Seconds per test run (from config).

    Returns:
        dict with pass_rate (float), passed (int), total (int), error_types (list[str]).
    """
    if timeout is None:
        timeout = SUBPROCESS_TIMEOUT
    return _run_tests_lcb(code, problem, timeout)


def _run_tests_lcb(
    code: str,
    problem: dict[str, Any],
    timeout: int,
) -> dict:
    public = problem.get("public_test_cases") or []
    private = problem.get("private_test_cases") or []
    test_cases = public + private
    if not test_cases:
        return {
            "pass_rate": 0.0,
            "passed": 0,
            "total": 0,
            "error_types": ["NoTestCases"],
        }

    metadata = problem.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    func_name = metadata.get("func_name", "")

    passed = 0
    error_types: list[str] = []
    for tc in test_cases:
        test_in = tc.get("input", tc.get("input_text", ""))
        test_out = tc.get("output", tc.get("output_text", ""))
        testtype = tc.get("testtype", "stdin")
        if not isinstance(test_in, str):
            test_in = str(test_in)
        if not isinstance(test_out, str):
            test_out = str(test_out)

        if testtype == "functional":
            ok, err = _run_one_test_functional(
                code, test_in, test_out, timeout, func_name
            )
        else:
            ok, err = _run_one_test(code, test_in, test_out, timeout)
        if ok:
            passed += 1
        else:
            error_types.append(err)

    total = len(test_cases)
    pass_rate = passed / total if total > 0 else 0.0
    return {
        "pass_rate": pass_rate,
        "passed": passed,
        "total": total,
        "error_types": error_types,
    }


def _run_one_test(code: str, test_in: str, expected_out: str, timeout: int) -> tuple[bool, str]:
    """
    Run code in a subprocess with test_in on stdin; compare stdout to expected_out.

    Returns:
        (True, "") if output matches (after stripping); else (False, error_type).
    """
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir="/tmp",
    ) as f:
        f.write(code)
        tmp_path = f.name
    try:
        result = subprocess.run(
            ["python3", tmp_path],
            input=test_in,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={},
        )
        out = (result.stdout or "").strip()
        expected = expected_out.strip()
        if result.returncode != 0:
            return False, "RuntimeError"
        if out != expected:
            return False, "AssertionError"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception:
        return False, "RuntimeError"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def _run_one_test_functional(
    code: str,
    test_in: str,
    expected_out: str,
    timeout: int,
    func_name: str,
) -> tuple[bool, str]:
    """
    Run solution as a module: call func_name(*json.loads(test_in)) and compare
    printed result to expected_out. Used for LeetCode-style call-based tests.
    """
    if not func_name:
        return False, "NoFuncName"
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".py",
        delete=False,
        dir="/tmp",
    ) as f:
        f.write(code)
        solution_path = f.name
    runner_path = solution_path + "_runner.py"
    runner_dir = os.path.dirname(solution_path)
    runner_code = f"""import sys
import json
sys.path.insert(0, {repr(runner_dir)})
import importlib.util
spec = importlib.util.spec_from_file_location("solution", {repr(solution_path)})
solution = importlib.util.module_from_spec(spec)
spec.loader.exec_module(solution)
args = json.loads(sys.stdin.read())
if isinstance(args, list):
    result = getattr(solution, {repr(func_name)})(*args)
else:
    result = getattr(solution, {repr(func_name)})(args)
print(result)
"""
    try:
        with open(runner_path, "w") as f:
            f.write(runner_code)
        result = subprocess.run(
            ["python3", runner_path],
            input=test_in,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd="/tmp",
            env={},
        )
        out = (result.stdout or "").strip()
        expected = expected_out.strip()
        if result.returncode != 0:
            return False, "RuntimeError"
        if out != expected:
            return False, "AssertionError"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "TimeoutError"
    except Exception:
        return False, "RuntimeError"
    finally:
        for p in (solution_path, runner_path):
            try:
                os.unlink(p)
            except Exception:
                pass
