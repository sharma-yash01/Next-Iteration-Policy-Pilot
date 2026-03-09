# Self-Repair Termination Pilot

Reflexion-style iterative repair on HumanEval[:100] (5 iterations per problem). Produces three pilot metrics for a GREEN/YELLOW/RED decision on a full COLM study.

## Setup

```bash
cd pilot-impl
pip install -r requirements.txt
export OPENAI_API_KEY=sk-...
```

## Run

```bash
python run_pilot.py
```

- **Trajectories**: one `.jsonl` per problem under `data/trajectories/` (written after every iteration; crash-safe).
- **Summary**: `data/results/pilot_summary.json` with `waste_rate`, `ece`, `auc_mean`, `auc_std`.

## Output paths

| Path | Description |
|------|-------------|
| `data/trajectories/*.jsonl` | Per-problem trajectory (frozen schema: problem_id, iteration, code, pass_rate, error_types, patch_delta, self_verification_score, features, timestamp) |
| `data/results/pilot_summary.json` | Pilot metrics and decision inputs |

## Decision thresholds

- **Waste Rate** > 25% → GREEN
- **Self-Verification ECE** > 0.2 → GREEN
- **Feature AUC** > 0.65 → GREEN

3/3 GREEN → full COLM; 2/3 → YELLOW (EMNLP/TMLR); 0–1 → RED.
