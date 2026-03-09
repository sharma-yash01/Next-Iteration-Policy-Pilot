# Next-Iteration-Policy-Pilot
Pilot experiment to see if next iteration prediction policy for Code/SWE Agents is something worth researching and implementing

## Free OpenRouter Models for Pilot v2

All models below are **free** on OpenRouter (`$0/M input & output tokens`).
Rate limits: ~20 req/min, ~200 req/day per model.

| Model ID (for `config.py`) | Params | Active | Context | Why |
|---|---|---|---|---|
| `stepfun/step-3.5-flash:free` | 196B MoE | 11B | 256K | **#1 free coding model** on OpenRouter rankings |
| `qwen/qwen3-coder:free` | 480B MoE | 35B | 262K | Purpose-built for agentic coding, tool use, repo-scale context |
| `qwen/qwen3-next-80b-a3b-instruct:free` | 80B MoE | 3B | 262K | Fast instruct model, good for code + reasoning without thinking traces |
| `meta-llama/llama-3.3-70b-instruct:free` | 70B | 70B | 128K | Proven baseline, strong multilingual + code |
| `openai/gpt-oss-120b:free` | 117B MoE | 5.1B | 131K | Current pick; upstream rate-limits can be an issue |
| `openai/gpt-oss-20b:free` | 21B MoE | 3.6B | 131K | Smaller sibling, lower latency, same OpenAI lineage |

**Recommendation:** If `gpt-oss-120b:free` keeps hitting upstream 429s, switch to **`stepfun/step-3.5-flash:free`** or **`qwen/qwen3-coder:free`** — both have higher capacity and strong coding benchmarks.
