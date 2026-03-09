# Pilot Implementation Plan: Self-Repair Termination Validation
**Goal**: Run Reflexion-style iterative repair on HumanEval[:100], 5 iterations each.
Produce three pilot metrics to decide GREEN/YELLOW/RED on full COLM paper.

---

## Three Metrics to Produce
| Metric | Measure | GREEN Threshold |
|---|---|---|
| Waste Rate | % iterations after Oracle-First stop | >25% |
| Self-Verification ECE | Calibration error vs binary solve outcome | >0.2 |
| Feature AUC | XGBoost prediction of "next iteration improves?" | >0.65 |

---

## Dependencies
````
evalplus              # sandboxed execution — NOT human-eval
litellm               # model-agnostic LLM calls
tenacity              # retry logic
python-Levenshtein
scikit-learn
xgboost
jsonlines
pandas numpy tqdm
````
Pin: `evalplus==0.3.1`

---

## File Structure
````
pilot/
  config.py
  repair.py
  evaluate.py
  features.py
  analyze.py
  run_pilot.py
  data/
    trajectories/     # one .jsonl per problem, written per-iteration (crash safe)
    results/          # pilot_summary.json
````

---

## Step 1 — config.py
````python
MODEL = "gpt-4o-mini"
MAX_ITERATIONS = 5          # iteration 0 = initial gen, 1-4 = repair
N_PROBLEMS = 100
PASS_THRESHOLD = 0.8        # Oracle-First binary solve threshold
IMPROVEMENT_THRESHOLD = 0.05
SUBPROCESS_TIMEOUT = 5
RATE_LIMIT_SLEEP = 0.5
MAX_RETRIES = 3
DATA_DIR = "data/trajectories"
RESULTS_DIR = "data/results"
COST_HARD_STOP_USD = 50.0
````

---

## Step 2 — evaluate.py
**Use EvalPlus. Never use exec() or eval() directly.**
````python
from evalplus.data import get_human_eval_plus
from evalplus.evaluate import evaluate_solution

def run_tests(task_id: str, code: str, timeout: int = 5) -> dict:
    """
    Args:
        task_id: e.g. "HumanEval/0"
        code: full function string
    Returns:
        {pass_rate: float, passed: int, total: int, error_types: list[str]}
    """
    result = evaluate_solution(task_id, code, timeout=timeout)
    passed = sum(result["results"])
    total = len(result["results"])
    error_types = [r["error_type"] for r in result["details"] if r.get("error_type")]
    return {
        "pass_rate": passed / total if total > 0 else 0.0,
        "passed": passed,
        "total": total,
        "error_types": error_types
    }
````

---

## Step 3 — repair.py

### 3a. Initial Generation (Iteration 0 — zero-shot, no repair context)
````python
def generate_initial(problem_prompt: str, model: str) -> str:
    prompt = f"Complete the following Python function:\n\n{problem_prompt}"
    return call_llm(prompt, model)
````

### 3b. Reflexion Repair Prompt (Iterations 1–4)
````python
def build_repair_prompt(problem: str, code: str, test_results: dict) -> str:
    errors = "\n".join(test_results["error_types"][:5]) or "No errors captured"
    return f"""Fix this Python function.

Problem: {problem}

Current code ({test_results['passed']}/{test_results['total']} tests pass):
```python
{code}
```

Errors: {errors}

Return ONLY the fixed function, no explanation."""
````

### 3c. Self-Verification Score — logprobs, not free-text parsing
````python
def get_self_verification_score(problem: str, code: str, model: str) -> float:
    """
    Returns P(Yes) from logprobs on 'Will this code pass all tests? Yes/No'.
    Falls back to structured JSON, then 0.5 neutral if both fail.
    """
    prompt = f"Problem:\n{problem}\n\nCode:\n```python\n{code}\n```\n\nWill this code pass all tests? Answer Yes or No only."
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=5
        )
        return extract_yes_probability(response)  # scan top_logprobs for Yes/No tokens
    except Exception:
        pass
    try:  # fallback: structured JSON
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt + ' Respond {"confidence": <float 0-1>}'}],
            max_tokens=20,
            response_format={"type": "json_object"}
        )
        import json
        return float(json.loads(response.choices[0].message.content)["confidence"])
    except Exception:
        return 0.5  # neutral — log this
````

### 3d. LLM Wrapper with retry + cost tracking
````python
import litellm, time
from tenacity import retry, stop_after_attempt, wait_exponential

running_cost_usd = 0.0

@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(prompt: str, model: str) -> str:
    global running_cost_usd
    if running_cost_usd >= COST_HARD_STOP_USD:
        raise RuntimeError(f"Cost hard stop hit: ${running_cost_usd:.2f}")
    time.sleep(RATE_LIMIT_SLEEP)
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    running_cost_usd += litellm.completion_cost(response)
    return response.choices[0].message.content
````

### 3e. Repair Loop
````python
def run_repair_loop(task_id: str, problem: str, model: str) -> list[dict]:
    """
    Iteration 0 = initial gen. Iterations 1-4 = repair.
    Saves .jsonl after every iteration. Skips if already complete.
    """
    out_path = f"{DATA_DIR}/{task_id.replace('/', '_')}.jsonl"
    if _is_complete(out_path):
        return _load_jsonl(out_path)

    trajectory, code = [], generate_initial(problem, model)

    for iteration in range(MAX_ITERATIONS):
        test_results = run_tests(task_id, code)
        sv_score = get_self_verification_score(problem, code, model)
        prev_code = trajectory[-1]["code"] if trajectory else ""

        step = {
            "problem_id": task_id,
            "iteration": iteration,
            "code": code,
            "pass_rate": test_results["pass_rate"],
            "passed": test_results["passed"],
            "total": test_results["total"],
            "error_types": test_results["error_types"],
            "patch_delta": compute_ast_levenshtein(prev_code, code),
            "self_verification_score": sv_score,
            "features": extract_features(trajectory, iteration),
            "timestamp": time.time()
        }
        trajectory.append(step)
        _append_jsonl(out_path, step)  # write immediately

        if iteration < MAX_ITERATIONS - 1:
            code = call_llm(build_repair_prompt(problem, code, test_results), model)

    return trajectory
````

---

## Step 4 — features.py
**Importable standalone. Downstream papers reuse this module.**
````python
import ast, Levenshtein
import numpy as np
from scipy.stats import entropy
from config import IMPROVEMENT_THRESHOLD

# EXTENSION POINT — add feature groups here for downstream papers

def ast_normalize(code: str) -> str:
    """Strip comments, normalize whitespace via AST round-trip."""
    try:
        return ast.unparse(ast.parse(code))
    except SyntaxError:
        return code.strip()

def compute_ast_levenshtein(code_a: str, code_b: str) -> int:
    """AST-normalized edit distance. Raw string distance is too noisy."""
    return Levenshtein.distance(ast_normalize(code_a), ast_normalize(code_b))

def extract_features(trajectory: list[dict], current_idx: int) -> dict:
    """
    Args:
        trajectory: steps recorded so far (before current_idx)
        current_idx: current iteration number (0-indexed)
    Returns:
        Feature dict for predicting whether next iteration improves pass_rate
    """
    if current_idx == 0 or not trajectory:
        return _zero_features()

    curr = trajectory[-1]
    prev = trajectory[-2] if len(trajectory) >= 2 else None
    older = trajectory[-3] if len(trajectory) >= 3 else None

    pass_rate = curr["pass_rate"]
    pass_rate_delta = pass_rate - (prev["pass_rate"] if prev else 0.0)
    pass_rate_delta_2 = pass_rate - (older["pass_rate"] if older else 0.0)

    error_types = curr["error_types"]
    error_vec = np.array([error_types.count(e) for e in
                          ["SyntaxError","TypeError","AssertionError","TimeoutError","RuntimeError"]],
                         dtype=float)
    error_vec_norm = error_vec / error_vec.sum() if error_vec.sum() > 0 else error_vec
    error_ent = float(entropy(error_vec_norm + 1e-9))

    patch_lev = curr["patch_delta"]
    is_dup = int(patch_lev < 5)
    is_osc = int(older is not None and
                 compute_ast_levenshtein(curr["code"], older["code"]) < 10)

    no_improve = 0
    for step in reversed(trajectory):
        if step["pass_rate"] < pass_rate - IMPROVEMENT_THRESHOLD:
            break
        no_improve += 1

    return {
        # EXTENSION POINT — pass rate features
        "pass_rate": pass_rate,
        "pass_rate_delta": pass_rate_delta,
        "pass_rate_delta_2": pass_rate_delta_2,
        # EXTENSION POINT — error features
        "error_type_entropy": error_ent,
        "syntax_error_count": int(error_vec[0]),
        "assertion_error_count": int(error_vec[2]),
        "timeout_count": int(error_vec[3]),
        # EXTENSION POINT — patch features
        "patch_levenshtein": patch_lev,
        "is_duplicate": is_dup,
        "is_oscillating": is_osc,
        # EXTENSION POINT — history features
        "iteration_number": current_idx,
        "consecutive_no_improvement": no_improve,
        "max_pass_rate_so_far": max(s["pass_rate"] for s in trajectory),
        # EXTENSION POINT — self-verification
        "self_verification_score": curr["self_verification_score"],
    }

def _zero_features() -> dict:
    return {k: 0 for k in ["pass_rate","pass_rate_delta","pass_rate_delta_2",
                             "error_type_entropy","syntax_error_count",
                             "assertion_error_count","timeout_count",
                             "patch_levenshtein","is_duplicate","is_oscillating",
                             "iteration_number","consecutive_no_improvement",
                             "max_pass_rate_so_far","self_verification_score"]}
````

---

## Step 5 — analyze.py

### Metric 1 — Waste Rate
````python
def compute_waste_rate(trajectories: list[list[dict]]) -> float:
    waste = []
    for traj in trajectories:
        oracle = next((s["iteration"] for s in traj
                       if s["pass_rate"] >= PASS_THRESHOLD), MAX_ITERATIONS - 1)
        waste.append(((MAX_ITERATIONS - 1) - oracle) / (MAX_ITERATIONS - 1))
    return float(np.mean(waste))
````

### Metric 2 — ECE (binary ground truth — NOT continuous pass_rate)
````python
def compute_ece(trajectories: list[list[dict]], n_bins: int = 15) -> float:
    """
    confidence = self_verification_score
    outcome = int(pass_rate >= PASS_THRESHOLD)  ← binary, not continuous
    """
    confs, outcomes = [], []
    for traj in trajectories:
        for step in traj:
            confs.append(step["self_verification_score"])
            outcomes.append(int(step["pass_rate"] >= PASS_THRESHOLD))

    confs, outcomes = np.array(confs), np.array(outcomes)
    ece = 0.0
    for lo, hi in zip(np.linspace(0,1,n_bins), np.linspace(1/n_bins,1,n_bins)):
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0: continue
        ece += (mask.sum()/len(confs)) * abs(confs[mask].mean() - outcomes[mask].mean())
    return float(ece)
````

### Metric 3 — Feature AUC
````python
def compute_feature_auc(trajectories: list[list[dict]]) -> tuple[float, float]:
    """
    Label: did_improve = pass_rate[t+1] - pass_rate[t] >= IMPROVEMENT_THRESHOLD
    class_weight balanced — label imbalance is expected and handled.
    Returns (mean_auc, std_auc) from 5-fold CV.
    """
    X, y = [], []
    for traj in trajectories:
        for i in range(len(traj) - 1):
            X.append(list(traj[i]["features"].values()))
            y.append(int(traj[i+1]["pass_rate"] - traj[i]["pass_rate"]
                         >= IMPROVEMENT_THRESHOLD))

    X, y = np.array(X), np.array(y)
    n_pos = int(y.sum())
    print(f"  Labels: {n_pos} positive / {len(y)} total")
    if n_pos < 20:
        print("  WARNING: <20 positive labels — treat AUC as directional only")

    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score
    clf = XGBClassifier(
        scale_pos_weight=max((len(y)-n_pos)/max(n_pos,1), 1),
        n_estimators=100, max_depth=4, random_state=42, eval_metric="auc"
    )
    aucs = cross_val_score(clf, X, y, cv=5, scoring="roc_auc")
    return float(aucs.mean()), float(aucs.std())
````

### Summary Output
````python
def print_summary(waste_rate: float, ece: float, auc_mean: float, auc_std: float):
    rows = [
        ("Waste Rate",            f"{waste_rate:.1%}", ">25%",  waste_rate > 0.25),
        ("Self-Verification ECE", f"{ece:.3f}",        ">0.200", ece > 0.2),
        (f"Feature AUC (±{auc_std:.3f})", f"{auc_mean:.3f}", ">0.650", auc_mean > 0.65),
    ]
    print("\n" + "="*58)
    print(f"{'Metric':<28} {'Value':>8}  {'Threshold':>9}  {'Status'}")
    print("-"*58)
    for name, val, thresh, green in rows:
        print(f"{name:<28} {val:>8}  {thresh:>9}  {'GREEN' if green else 'RED'}")
    print("="*58)
    greens = sum(r[3] for r in rows)
    print({3: "\nDECISION: GREEN — Full COLM study, start immediately",
           2: "\nDECISION: YELLOW — Characterization paper, target EMNLP/TMLR",
           1: "\nDECISION: RED — Kill topic, move to Topic 2 or 5",
           0: "\nDECISION: RED — Kill topic, move to Topic 2 or 5"}[greens])
````

---

## Step 6 — run_pilot.py
````python
from evalplus.data import get_human_eval_plus
from tqdm import tqdm
import os, json
from config import *
from repair import run_repair_loop
from analyze import compute_waste_rate, compute_ece, compute_feature_auc, print_summary

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    problems = list(get_human_eval_plus().items())[:N_PROBLEMS]
    trajectories = [run_repair_loop(tid, p["prompt"], MODEL)
                    for tid, p in tqdm(problems, desc="Repair loops")]

    waste = compute_waste_rate(trajectories)
    ece = compute_ece(trajectories)
    auc_mean, auc_std = compute_feature_auc(trajectories)

    print_summary(waste, ece, auc_mean, auc_std)
    with open(f"{RESULTS_DIR}/pilot_summary.json", "w") as f:
        json.dump({"waste_rate": waste, "ece": ece,
                   "auc_mean": auc_mean, "auc_std": auc_std}, f, indent=2)

if __name__ == "__main__":
    main()
````

---

## Run Order
````bash
pip install evalplus==0.3.1 litellm tenacity python-Levenshtein \
            scikit-learn xgboost jsonlines pandas numpy tqdm scipy
export OPENAI_API_KEY=sk-...
python run_pilot.py
# → data/results/pilot_summary.json
````

---

## Decision Table
| Greens | Action |
|---|---|
| 3/3 | Full RepairStop-1K collection → COLM March 31 |
| 2/3 | Empirical characterization + benchmark → EMNLP/TMLR |
| 0–1 | Kill → move to Code Reasoning Topic 2 or Critic Separation Topic 5 |

---

## Known Risks
| Risk | Mitigation |
|---|---|
| <20 positive AUC labels | Log n_pos; treat result as directional signal only |
| logprobs unavailable | JSON fallback → 0.5 neutral; log occurrences |
| evalplus API change | Pin `evalplus==0.3.1` |
| Cost overrun | Hard stop at $50 in call_llm() |