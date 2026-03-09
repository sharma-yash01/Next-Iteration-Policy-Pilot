"""All constants for the pilot. Do not hardcode these elsewhere."""

MODEL = "stepfun/step-3.5-flash:free"
# OpenRouter: use with api_base + OPENROUTER_API_KEY env so model id has no prefix.
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
MAX_ITERATIONS = 5  # iteration 0 = initial gen, 1-4 = repair
N_PROBLEMS = 80  # number of problems to run (LCB: first N from release)
# LiveCodeBench: release_v1 (400) through release_v6 (1055)
LCB_RELEASE = "release_v1"
# Only include problems with this difficulty or harder (easy | medium | hard)
LCB_MIN_DIFFICULTY = "easy"
PASS_THRESHOLD = 0.8  # Oracle-First binary solve threshold
IMPROVEMENT_THRESHOLD = 0.05
SUBPROCESS_TIMEOUT = 10
RATE_LIMIT_SLEEP = 3.5  # ~17 req/min, under OpenRouter free tier 20 req/min
# Max seconds for a single completion request; prevents indefinite hangs.
LLM_TIMEOUT_SEC = 120
MAX_RETRIES = 3
DATA_DIR = "data/trajectories"
RESULTS_DIR = "data/results"
COST_HARD_STOP_USD = 50.0
