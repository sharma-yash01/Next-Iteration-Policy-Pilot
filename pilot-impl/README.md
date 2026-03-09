# Self-Repair Termination Pilot

Reflexion-style iterative repair on LiveCodeBench problems (5 iterations per problem). Produces three pilot metrics for a GREEN/YELLOW/RED decision on a full COLM study.

## Pilot 1 Results (Nemotron-3-nano, LCB medium+)

Model: `openrouter/nvidia/nemotron-3-nano-30b-a3b:free` on 80 LiveCodeBench medium/hard problems.

| Metric | Threshold | Result | Status |
|---|---|---|---|
| Waste Rate | >25% | 5.9% | RED |
| Self-Verification ECE | >0.2 | 0.46 (artifactual) | Misleading GREEN |
| Feature AUC | >0.65 | NaN (1/320 positives) | RED |

**Decision: RED.** Root causes:
- 93.75% of problems never solved (model too weak for medium+ competitive programming)
- 5% solved immediately at iteration 0; only 1 problem (1.25%) showed gradual repair
- Self-verification score stuck at 0.5 fallback for all 400 steps (model did not return logprobs)
- Bimodal regime: instant-solve or never-solve, no intermediate repair trajectories to learn from

## Pilot 2 (Current Run): gpt-oss-120b, LCB easy+

Config changes in `config.py`:
- Model: `openrouter/openai/gpt-oss-120b:free` (120B MoE, 5.1B active; top open-source coding model)
- Difficulty: `LCB_MIN_DIFFICULTY = "easy"` (include easy problems to generate gradual repair trajectories)

**Goal**: Determine whether a stronger model on easier problems produces the intermediate-improvement trajectories needed for a stopping-policy paper.

## Setup

```bash
cd pilot-impl
pip install -r requirements.txt
export OPENROUTER_API_KEY=sk-or-v1-...
python run_pilot.py
```

## Output

| Path | Description |
|---|---|
| `data/trajectories/*.jsonl` | Per-problem trajectory (schema: problem_id, iteration, code, pass_rate, error_types, patch_delta, self_verification_score, features, timestamp) |
| `data/results/pilot_summary.json` | Pilot metrics and decision inputs |

## Decision Thresholds

- **Waste Rate** > 25% → GREEN
- **Self-Verification ECE** > 0.2 → GREEN
- **Feature AUC** > 0.65 → GREEN

3/3 GREEN → full COLM study; 2/3 → YELLOW (characterization paper, EMNLP/TMLR); 0-1 → RED (kill topic).
