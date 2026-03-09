"""
LLM repair loop: initial generation, Reflexion-style repair, self-verification, trajectory I/O.
Uses litellm only; no vendor SDKs.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import litellm
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    COST_HARD_STOP_USD,
    DATA_DIR,
    LLM_TIMEOUT_SEC,
    MAX_ITERATIONS,
    OPENROUTER_API_BASE,
    RATE_LIMIT_SLEEP,
)
from evaluate import run_tests
from features import compute_ast_levenshtein, extract_features

logger = logging.getLogger(__name__)

running_cost_usd = 0.0


def _log_call(
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    """Log per-call model, tokens, and cost."""
    logger.info(
        "llm_call model=%s prompt_tokens=%s completion_tokens=%s cost_usd=%.6f",
        model,
        prompt_tokens,
        completion_tokens,
        cost_usd,
    )


def strip_code_fences(text: str) -> str:
    """
    Extract raw Python from LLM output that may be wrapped in markdown fences or prose.

    Args:
        text: Raw completion (may contain ```python...``` or ```...```, or prose before code).

    Returns:
        Clean Python source string.
    """
    text = text.strip()
    # Match ```python ... ``` or ``` ... ``` (optional language tag)
    pattern = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
    blocks = pattern.findall(text)
    if blocks:
        # Take longest block (most likely the full function)
        code = max(blocks, key=len).strip()
        if code:
            return code
    # No fences: look for first def / from / import
    for start in ("def ", "from ", "import "):
        idx = text.find(start)
        if idx != -1:
            return text[idx:].strip()
    return text


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def call_llm(prompt: str, model: str) -> str | None:
    """
    Call LLM with retry, rate limit, and cost tracking.

    Args:
        prompt: User message content.
        model: Model name (e.g. gpt-4o-mini).

    Returns:
        Assistant message content, or None when the provider returns null/empty content.

    Raises:
        RuntimeError: If running cost exceeds COST_HARD_STOP_USD.
    """
    global running_cost_usd
    if running_cost_usd >= COST_HARD_STOP_USD:
        raise RuntimeError(f"Cost hard stop hit: ${running_cost_usd:.2f}")
    time.sleep(RATE_LIMIT_SLEEP)
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            timeout=LLM_TIMEOUT_SEC,
            api_base=OPENROUTER_API_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
    except Exception as e:
        logger.exception("call_llm failed: %s", e)
        logger.warning("call_llm retry due to exception")
        raise
    try:
        content = response.choices[0].message.content
        if isinstance(content, str):
            content = content.strip() or None
        if content is None:
            _log_call(model, 0, 0, 0.0)
            return None
        cost = 0.0
        running_cost_usd += cost
        usage = getattr(response, "usage", None)
        if usage is not None and hasattr(usage, "prompt_tokens"):
            prompt_tokens = getattr(usage, "prompt_tokens", 0)
            completion_tokens = getattr(usage, "completion_tokens", 0)
        elif isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
        else:
            prompt_tokens = completion_tokens = 0
        _log_call(model, prompt_tokens, completion_tokens, cost)
        return content
    except (IndexError, AttributeError, TypeError) as e:
        logger.warning("llm_bad_response_shape %s", e)
        _log_call(model, 0, 0, 0.0)
        return None


def generate_initial(problem_prompt: str, model: str) -> str | None:
    """
    Zero-shot initial code generation (iteration 0).

    Args:
        problem_prompt: Full problem prompt.
        model: Model name.

    Returns:
        Raw completion (code string), or None if the provider returns null/empty content.
    """
    prompt = (
        "Solve the following competitive programming problem in Python.\n"
        "Read input from stdin and print output to stdout.\n"
        "Return ONLY the Python code, no explanation.\n\n"
        f"{problem_prompt}"
    )
    return call_llm(prompt, model)


def build_repair_prompt(problem: str, code: str, test_results: dict[str, Any]) -> str:
    """
    Build Reflexion-style repair prompt with errors and current code.

    Args:
        problem: Problem description.
        code: Current code.
        test_results: dict with passed, total, error_types.

    Returns:
        Prompt string.
    """
    errors = "\n".join(test_results["error_types"][:5]) or "No errors captured"
    return f"""Fix this Python function.

Problem: {problem}

Current code ({test_results['passed']}/{test_results['total']} tests pass):
```python
{code}
```

Errors: {errors}

Return ONLY the fixed function, no explanation."""


def _extract_yes_probability(response: Any) -> float:
    """
    Extract P(Yes) from logprobs (top_logprobs). Handles tokenizer-dependent Yes/No tokens.

    Args:
        response: litellm completion response with choices[0].logprobs.

    Returns:
        Probability of "Yes" in [0, 1], or 0.5 if not determinable.
    """
    import math
    try:
        choice = response.choices[0]
        logprobs = getattr(choice, "logprobs", None) or getattr(
            choice.message, "logprobs", None
        )
        if not logprobs:
            return 0.5
        content = getattr(logprobs, "content", None)
        if not content:
            return 0.5
        yes_logprob = None
        no_logprob = None
        for item in content if isinstance(content, list) else [content]:
            top = getattr(item, "top_logprobs", None) or (item if isinstance(item, dict) else None)
            if top is None:
                continue
            tokens_with_lp: list[tuple[str, float]] = []
            if isinstance(top, list):
                for t in top:
                    token = t.get("token", getattr(t, "token", "")) if isinstance(t, dict) else getattr(t, "token", "")
                    lp = t.get("logprob", getattr(t, "logprob", -999)) if isinstance(t, dict) else getattr(t, "logprob", -999)
                    tokens_with_lp.append((str(token), float(lp)))
            elif isinstance(top, dict):
                for token, lp in top.items():
                    tok = token if isinstance(token, str) else getattr(token, "token", "")
                    lp_val = lp if isinstance(lp, (int, float)) else getattr(lp, "logprob", -999)
                    tokens_with_lp.append((str(tok), float(lp_val)))
            for tok, lp in tokens_with_lp:
                t = tok.strip().lower()
                if t in ("yes", "yes."):
                    yes_logprob = lp if yes_logprob is None else max(yes_logprob, lp)
                elif t in ("no", "no."):
                    no_logprob = lp if no_logprob is None else max(no_logprob, lp)
        if yes_logprob is not None and no_logprob is not None:
            p_yes = math.exp(yes_logprob)
            p_no = math.exp(no_logprob)
            return float(p_yes / (p_yes + p_no))
        if yes_logprob is not None:
            return 1.0
        if no_logprob is not None:
            return 0.0
    except Exception as e:
        logger.debug("extract_yes_probability failed: %s", e)
    return 0.5


def get_self_verification_score(problem: str, code: str, model: str) -> float:
    """
    Return P(Yes) from logprobs for 'Will this code pass all tests? Yes/No'.
    Falls back to structured JSON confidence, then 0.5 if both fail.

    Args:
        problem: Problem description.
        code: Code string.
        model: Model name.

    Returns:
        Score in [0, 1].
    """
    prompt = (
        f"Problem:\n{problem}\n\nCode:\n```python\n{code}\n```\n\n"
        "Will this code pass all tests? Answer Yes or No only."
    )
    try:
        response = litellm.completion(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1,
            logprobs=True,
            top_logprobs=5,
            timeout=LLM_TIMEOUT_SEC,
            api_base=OPENROUTER_API_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        return _extract_yes_probability(response)
    except Exception as e:
        logger.debug("self-verification logprobs path failed: %s", e)
    try:
        response = litellm.completion(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": prompt + ' Respond {"confidence": <float 0-1>}',
                }
            ],
            max_tokens=20,
            response_format={"type": "json_object"},
            timeout=LLM_TIMEOUT_SEC,
            api_base=OPENROUTER_API_BASE,
            api_key=os.environ.get("OPENROUTER_API_KEY"),
        )
        raw = None
        try:
            raw = response.choices[0].message.content
        except (IndexError, AttributeError, TypeError):
            pass
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            logger.debug("self-verification JSON content missing or empty")
        else:
            return float(json.loads(raw)["confidence"])
    except Exception as e:
        logger.info("self-verification JSON fallback failed: %s", e)
    # Text-based fallback: plain Yes/No response (no logprobs/JSON)
    try:
        response = call_llm(prompt, model)
        raw = (response or "").strip().lower()
        first_word = raw.split()[0].rstrip(".,!") if raw.split() else ""
        if first_word == "yes":
            return 1.0
        if first_word == "no":
            return 0.0
    except Exception as e:
        logger.debug("self-verification text fallback failed: %s", e)
    return 0.5


def _is_complete(out_path: str) -> bool:
    """Return True if trajectory file exists and has MAX_ITERATIONS steps."""
    import os
    if not os.path.isfile(out_path):
        return False
    with open(out_path) as f:
        count = sum(1 for _ in f)
    return count >= MAX_ITERATIONS


def _load_jsonl(out_path: str) -> list[dict[str, Any]]:
    """Load trajectory from JSONL file."""
    import json
    traj = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            traj.append(json.loads(line))
    return traj


def _append_jsonl(out_path: str, step: dict[str, Any]) -> None:
    """Append one step as a single JSON line. No overwrite."""
    import os
    with open(out_path, "a") as f:
        f.write(json.dumps(step) + "\n")


def run_repair_loop(
    task_id: str,
    problem: str,
    model: str,
    problem_dict: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """
    Run Reflexion-style repair loop: iteration 0 = initial gen, 1..4 = repair.
    Saves one JSONL line after every iteration (crash-safe). Skips if already complete.

    Args:
        task_id: e.g. LCB question_id or HumanEval/0.
        problem: Problem prompt (text).
        model: Model name.
        problem_dict: Full problem dict for evaluation (e.g. LCB: public_test_cases, etc.). Required for run_tests.

    Returns:
        List of step dicts (frozen schema: problem_id, iteration, code, pass_rate, ...).
        Steps may include optional llm_null_response: true when the LLM returned null
        and the previous code was reused for that iteration.
    """
    import os
    if problem_dict is None:
        problem_dict = {}
    out_path = f"{DATA_DIR}/{task_id.replace('/', '_')}.jsonl"
    if _is_complete(out_path):
        return _load_jsonl(out_path)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    trajectory: list[dict[str, Any]] = []
    raw = generate_initial(problem, model)
    if raw is None:
        logger.warning("llm_null_content problem_id=%s stage=initial", task_id)
        raw = generate_initial(problem, model)
    if raw is None:
        logger.error("llm_null_content problem_id=%s stage=initial retries_exhausted", task_id)
        raise RuntimeError(f"Initial LLM returned None for task_id={task_id}")
    code = strip_code_fences(raw)

    next_step_llm_null = False
    for iteration in range(MAX_ITERATIONS):
        test_results = run_tests(task_id, code, problem_dict)
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
            "timestamp": time.time(),
            "llm_null_response": next_step_llm_null,
        }
        trajectory.append(step)
        _append_jsonl(out_path, step)

        if iteration < MAX_ITERATIONS - 1:
            raw = call_llm(
                build_repair_prompt(problem, code, test_results),
                model,
            )
            if raw is None:
                logger.warning(
                    "llm_null_content problem_id=%s iteration=%s", task_id, iteration
                )
                raw = call_llm(
                    build_repair_prompt(problem, code, test_results),
                    model,
                )
            if raw is None:
                logger.warning(
                    "llm_null_content problem_id=%s iteration=%s retries_exhausted",
                    task_id,
                    iteration,
                )
                next_step_llm_null = True
            else:
                code = strip_code_fences(raw)
                next_step_llm_null = False

    return trajectory
