# Pilot evaluation history

Summary of pilot runs and the changes applied between them.

---

## Run 1: HumanEval + EvalPlus

- **Setup:** HumanEval with EvalPlus tests.
- **Outcome:** setrlimit crash during test execution; all steps reported `pass_rate=0`; all self-verification scores constant at `0.5`.
- **Decision:** **RED (invalid)** — execution environment and metrics not usable.

**Fixes applied after Run 1:**

- setrlimit set to `-1` (or equivalent) so test runner could execute.
- Strip code fences from model output before evaluation.
- Self-verification text fallback for when logprobs/JSON are unavailable.
- Data archival at start of run (trajectories moved to `trajectories_poisoned_<timestamp>` before each run).

---

## Run 2: HumanEval + EvalPlus (after fixes)

- **Setup:** Same HumanEval + EvalPlus with the above fixes.
- **Outcome:** Tests ran correctly; 363/400 steps had `pass_rate=1.0` (ceiling effect); only 7 positive labels for improvement; Feature AUC ≈ 0.633.
- **Decision:** **YELLOW (valid but benchmark too easy)** — metrics computable but little variance for learning a stopping policy.

**Fixes applied after Run 2:**

- Switched benchmark to LiveCodeBench (code_generation_lite).
- Subprocess-based evaluator (no exec/eval); timeout 5s per test.
- Datasets/version pin for reproducibility.

---

## Run 3: LCB release_v1 (2 problems)

- **Setup:** LiveCodeBench `release_v1`, first 2 problems (smoke test).
- **Outcome:** Only 1 public test case per problem (private test cases were not decoded); both problems were trivial Codeforces “A” problems, solved at iteration 0; self-verification still constant at 0.5 due to loose yes/no matching.
- **Decision:** **YELLOW (invalid — not enough test coverage)** — private tests missing, problems too easy, SV uninformative.

**Fixes applied after Run 3 (this plan):**

1. **Decode private test cases** — In `data_lcb.py`, `parse_tests` now tries `json.loads` first; on failure (for private blobs) decodes via base64 → zlib → pickle → JSON. Preserves `testtype` per test for evaluation.
2. **Filter by difficulty** — Added `LCB_MIN_DIFFICULTY = "medium"` in `config.py`; in `data_lcb.py` skip rows below that (easy/medium/hard ordering). Added `difficulty` to each problem dict.
3. **Self-verification yes/no matching** — In `repair.py`, replaced substring matching with first-word matching: only treat as yes/no when the response starts with the word “yes” or “no” (after stripping), else 0.5.
4. **Functional (LeetCode) tests** — In `evaluate.py`, support both `testtype == "stdin"` (existing subprocess stdin/stdout) and `testtype == "functional"` (wrapper script that imports solution and calls `func_name(*json.loads(input))`, then compare stdout). Problem dict includes `metadata` (e.g. `func_name`) from LCB.
5. **Competitive programming prompt** — In `repair.py` `generate_initial`, prompt updated to: “Solve the following competitive programming problem in Python. Read input from stdin and print output to stdout. Return ONLY the Python code, no explanation.”
6. **Archive only when requested** — In `run_pilot.py`, archive existing trajectory data only when `PILOT_FORCE_CLEAN=1`; otherwise keep data for crash-resume.
7. **Timeout** — `SUBPROCESS_TIMEOUT` increased from 5 to 10 seconds in `config.py` for harder LCB problems.

---

Next run (Run 4) should use the full LCB medium/hard subset with decoded private tests, functional test support, and the updated prompt and SV logic to produce a valid pilot for waste rate, ECE, and Feature AUC.
