## Topic 1 (CR T4): Self-Repair Termination — Learned Stopping Policies for LLM Code Fixing Loops

**Score: 87/100** *(revised from 83)*

- Publication Probability (28/30) *(+2)*
- Lab Fit (22/25)
- Feasibility (18/20)
- Program Potential (10/15)
- Novelty (9/10) *(+2)*

### Hook & Motivation

Production code agents (Cursor, Aider, Amazon Q, Devin) use arbitrary iteration limits (3-5 rounds) with no learned stopping policies. ChatRepair's iterative repair framework demonstrates the problem: LLM-based tools "may generate many repeated or similar patches that were already determined to be incorrect, wasting dollar cost in API access or time in GPU execution"—with **65% of patches being duplicates**[arxiv:2304.00385]. Recent work on iterative repair strategies compares fixed allocation patterns (10×1, 8-2, 5×2, 4-3-3, 2×5, 1×10) but provides no dynamic, per-problem stopping prediction. No principled framework exists for learning when self-repair loops should terminate based on trajectory features and repair convergence patterns.

### Research Question

Given a multi-iteration LLM code repair trajectory (code deltas, test failures, self-verification signals), can we predict whether the next repair iteration will improve correctness? Can a learned termination policy reduce wasted compute by >30% while maintaining or improving pass@k compared to fixed-iteration baselines?

### Target Venue & Timeline

**COLM 2026** | 8-9 pages | **Deadline: March 31, 2026** | Premier LM venue for agent reasoning and test-time compute optimization

*Alternative*: **NeurIPS 2026** (May 15) if safety/verification framing emphasized; **EMNLP 2026** (ARR June) for empirical systems focus

### Gap Analysis

**What EXISTS**:

- **Reflexion** (Shinn et al., NeurIPS 2023): Verbal reinforcement learning where LLMs reflect on task feedback, maintaining episodic memory. Achieved 91% pass@1 on HumanEval vs GPT-4's 80% using Actor-Evaluator-Self-Reflection architecture. **Uses fixed iteration limits—no learned stopping.**
- **Self-Refine** (Madaan et al., NeurIPS 2023): Iterative refinement with self-feedback across 7 tasks. Same LLM provides feedback and refines output iteratively. **Stops after fixed iterations or when model self-declares "no further refinement needed"—not learned stopping.**
- **ChatRepair** (Xia & Zhang, ISSTA 2024): Conversational APR achieving 114/48 correct fixes on Defects4J 1.2/2.0 for $0.42/bug. **Critical empirical finding: 65% of patches are duplicates**. Uses fixed conversation length limits.
- **REx** (Tang et al., NeurIPS 2024): Thompson Sampling for *branch selection* in refinement trees—solves explore-exploit for **which program to refine next**, NOT when to stop refining. Reduces LLM calls by 1.5-5× across APPS, ARC, loop invariants.
- **PAG** (Liu et al., 2025): Unified policy/verifier with selective revision; no explicit stopping predictor trained on repair trajectories.
- **ReVeal** (Zhao et al., 2025): Multi-turn RL for code generation + verification; stops at correctness or T_max—no learned iteration-level stopping.
- **SETS** (2025): Scales test-time compute but no learned iteration-level stopping policy.
- **BEACON** (Oct 2025): Bayesian optimal stopping for Best-of-N sampling—NOT iterative refinement with feedback.
- **GenGuard** (Jul 2024): Token-level stopping—NOT iteration-level.
- **"LLMs vs Halting Problem"** (Sultan et al., Jan 2026): Evaluates *global program termination* on SV-Comp—NOT self-repair loop stopping with test feedback.
- **Spiess et al.** (ICSE 2025): Shows code LLMs have high Expected Calibration Error (0.09-0.73): "intrinsic LLM confidences are poor predictors of code correctness"—motivates learned stopping beyond self-verification.
- **Lahiri et al.** (May 2025): Compares 7 fixed strategies (10×1, 4-3-3, 2×5, 1×10); no learned dynamic stopping.

**What DOES NOT exist**:

- Learned prediction of iteration value based on trajectory features
- Classifier trained on {code_i, test results_i, error types_i, patch delta_{i-1→i}} → "next iteration improves pass@k ≥5%"
- Systematic study of repair convergence patterns (when do fixes plateau?)
- Public benchmark with oracle stopping annotations for code repair trajectories

**Confidence**: 80% genuine gap (ChatRepair empirically shows 65% duplicate waste; REx solves branch selection, not stopping)

### Contribution to Literature (Pujar et al. 2025 Survey Alignment)

**Primary CTA Addressed**:

- **CTA-6** (Rank 1 relevance): *"Structured way to analyze errors in agents to overcome repetitions, avoid waste of resources"*
    - Survey identifies this as critical gap for agent efficiency
    - Our work: First learned mechanism to detect when repair loops have exhausted value
    - Directly addresses repetition detection (65% duplicates in ChatRepair) with trajectory-based stopping

**Secondary CTAs**:

- **CTA-4** (Rank 2): *"Investigate whether execution serves as deterministic check that mitigates model rigidity"*
    - Our stopping classifier uses execution feedback (test pass/fail, error types) as primary signal
    - Experiment: Does execution filtering reduce entropy of repair trajectories over iterations?
- **CTA-7** (Rank 3): *"Develop error recovery benchmarks evaluating agents' ability to recover from mistakes"*
    - RepairStop-1K: First benchmark with oracle stopping annotations across 5K repair trajectories

**Paper Positioning Statements**:

*Introduction*: "Production code agents waste compute on repetitive failed patches (ChatRepair: 65% duplicates; Xia & Zhang, 2024) yet lack structured mechanisms to detect and stop error repetition. Recent surveys identify this as a critical gap for agent efficiency (Pujar et al., 2025 CTA-6)."

*Benchmark Section*: "RepairStop-1K responds to CTA-7's call for error recovery benchmarks (Pujar et al., 2025), providing the first public dataset with oracle stopping annotations for code repair trajectories."

*Method Section*: "Our approach tests CTA-4's hypothesis (Pujar et al., 2025) that execution feedback serves as deterministic check, using test pass/fail signals as primary features for stopping prediction."

### Complete Research Lineage

**Foundation Papers (Read First)**:

1. **Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning"**, NeurIPS 2023 - https://arxiv.org/abs/2303.11366
    - Standard for self-repair loops; establishes failure modes (loops, hallucinated fixes)
2. **Madaan et al., "Self-Refine: Iterative Refinement with Self-Feedback"**, NeurIPS 2023 - https://openreview.net/pdf?id=S37hOerQLB
    - Pioneering iterative self-correction with task-specific stopping indicators
3. **Chen et al., "REx: Code Repair with LLMs Gives an Exploration-Exploitation Tradeoff"**, NeurIPS 2024 - https://neurips.cc/virtual/2024/poster/93642
    - Exploration-exploitation for branch selection (but not termination prediction)

**Self-Verification & Multi-Turn Correction**:

1. **Liu et al., "PAG: Multi-Turn Reinforced LLM Self-Correction with Policy as Generative Verifier"**, 2025 - https://arxiv.org/abs/2506.10406
    - Unified policy/verifier with selective revision; no explicit stopping predictor
2. **Zhao et al., "ReVeal: Self-Evolving Code Agents via Iterative Generation-Verification"**, 2025 - https://arxiv.org/abs/2506.11442
    - Multi-turn RL for code generation + verification; stops at correctness or T_max
3. **Anonymous, "SETS: Leveraging Self-Verification and Self-Correction for Scale"**, 2025 - https://arxiv.org/html/2501.19306v3
    - Scales test-time compute but no learned iteration-level stopping
4. **Emergent Mind Survey, "Self-Verification-Based LLMs"**, 2026 - https://www.emergentmind.com/topics/self-verification-based-llms
    - Catalogs RL frameworks for self-verification; notes scalability challenges

**Optimal Stopping & Verification Economics**:

1. **"BEACON: Bayesian Optimal Stopping for Efficient LLM Sampling"**, Oct 2025 - https://arxiv.org/abs/2510.15945
    - Best-of-N stopping (independent draws, not repair trajectories)
2. **Lu et al., "When Does Verification Pay Off? A Closer Look at LLMs as Solution Verifiers"**, Dec 2025 - https://arxiv.org/abs/2512.02304
    - Verifier gain metric for rejection sampling; no repair-loop formulation
3. **Sultan et al., "LLMs versus the Halting Problem: Revisiting Program Termination Prediction"**, Jan 2026 - https://arxiv.org/abs/2601.18987
    - Static program termination on SV-Comp; provides rhetorical contrast to your work

**Token-Level Stopping (Different Granularity)**:

1. **"When to Stop? Towards Efficient Code Generation in LLMs with Excess Token Prevention (GenGuard)"**, Jul 2024 - https://arxiv.org/abs/2407.20042
    - Token-level stopping; orthogonal to iteration-level

**Calibration & Confidence (Critical for Stopping)**:

1. **Spiess et al., "Calibration and Correctness of Language Models for Code"**, ICSE 2025 - https://www.software-lab.org/publications/icse2025_calibration.pdf
    - Shows code LLMs poorly calibrated (ECE 0.09-0.73); motivates learned stopping beyond intrinsic confidence
2. **"Learning to Route LLMs with Confidence Tokens"**, OpenReview - https://openreview.net/forum?id=U08mUogGDM
    - Lightweight confidence probes; directly applicable to stop classifier architecture
3. **"Correctness Assessment of Code Using Internal Representations"**, Jan 2025 - https://arxiv.org/abs/2501.12934
    - Uses hidden states to predict correctness; baseline for feature extraction

**Iterative Repair Strategies (Empirical Context)**:

1. **Lahiri et al., "Optimizing Iterative Program Repair with Instruction-Tuned Models"**, May 2025 - https://arxiv.org/abs/2505.02931
    - Compares 7 fixed strategies (10×1, 4-3-3, 2×5, 1×10); no learned dynamic stopping
2. **Xia & Zhang, "ChatRepair: Fixing Bugs with ChatGPT"**, ISSTA 2024 - https://arxiv.org/abs/2304.00385
    - Iterative repair with conversation; notes waste from repeated incorrect patches (65% duplicates)

**Production Systems (Empirical Context)**:

1. **Aider, Cursor, Amazon Q, Devin, OpenHands** - all use fixed iteration caps (3-5)

### Method

**1. Problem Formalization**

- Define **Self-Repair Termination Prediction**:
    - **Input**: Trajectory at iteration $t$: $\tau_t = {(\text{code}_i, \text{test results}*i, \text{error types}i, \Delta\text{code}{i-1 \to i})}*{i=1}^t$ + self-verification text/scores
    - **Output**: Binary label "next iteration yields improved pass@k by ≥5%" OR value estimate $\mathbb{E}[\Delta \text{pass@k} | \text{continue}]$
- **Oracle definitions**:
    - **Oracle-First**: Stop at first iteration where pass@k ≥ 80% (primary metric)
    - **Oracle-Plateau**: Stop when next 2 iterations yield <5% improvement (secondary analysis)
    - **Oracle-Cost**: Minimize total iterations while reaching ≥80% pass@k
- Metrics:
    - **Waste rate**: % iterations beyond Oracle-First stopping point
    - **Regret vs oracle**: gap in pass@k at fixed compute budget (|iterations_policy - iterations_oracle|)
    - **Compute savings**: % reduction in LLM API calls/FLOPs vs fixed-3 baseline
    - **Stopping accuracy**: precision/recall/F1 and AUC for binary improvement prediction

**2. Dataset Construction: RepairStop-1K Benchmark**

- Collect **multi-iteration repair trajectories** on:
    - HumanEval (164), MBPP (400), APPS-Intro (436) → target: 1,000 problems × avg 5 iterations = 5K trajectory steps
- Repair strategies:
    - Reflexion-style (verbal self-reflection on failures)
    - ChatRepair-style (test feedback in conversation)
    - ReVeal-style (with test generation)
    - Basic retry-on-fail (re-prompt with error message)
- Log per iteration:
    - **Code**: Full function, AST structure, LOC, cyclomatic complexity
    - **Test execution**: Pass/fail counts per test, error categories (SyntaxError, TypeError, AssertionError, timeout, etc.)
    - **Patch analysis**: Line-level diff, Levenshtein distance, AST edit distance, semantic similarity via CodeBERT
    - **Self-verification**: Model confidence score, confidence token probabilities, explanation text/length, semantic similarity to "correct" vs "incorrect" exemplars
    - **Coverage**: Lines/branches covered, delta from previous iteration
    - **Metadata**: Iteration number, cumulative time, token counts
- **Labels**:
    - Oracle stopping points (Oracle-First, Oracle-Plateau, Oracle-Cost)
    - Binary "next iteration improves pass@k by ≥5%" at each step
    - Final outcome: solved (≥80% pass@k) or failed
    - Eventually-fixed flag
- **Benchmark release**: JSON logs, evaluation scripts, leaderboard at [repository URL]
- **Novelty claim**: First public benchmark with oracle stopping annotations for code repair trajectories (addresses Pujar et al. 2025 CTA-7)

**3. Termination Policy Learning**

- **Feature engineering** (30-50 features total):
    - **A. Code features** (8-10 features): AST edit distance from previous iteration, semantic embedding similarity (CodeBERT cosine), cyclomatic complexity delta, lines/functions changed, number of changed lines
    - **B. Test execution features** (10-12 features): Current pass rate (% tests passing), pass rate trajectory (Δpass_{t-1→t}, Δpass_{t-2→t}), error type distribution (syntax/type/assertion/runtime/timeout %), newly passing tests, newly failing tests
    - **C. Self-verification features** (6-8 features): Model confidence score from self-verification prompt, confidence token probabilities, explanation length (tokens), calibration (|confidence - actual_pass_rate|), semantic similarity to "correct" vs "incorrect" exemplars
    - **D. Historical features** (8-10 features): Iteration count, consecutive iterations with no improvement, consecutive no-change patches (Levenshtein distance <5), oscillation detection (semantic similarity to code from 2 iterations ago >0.9), max pass rate achieved so far, convergence velocity (moving average of Δpass@k over last 2 iterations)
    - **E. Coverage features** (3-5 features): % lines covered, % branches covered, coverage delta from previous iteration
- **Models**:
    - **Lightweight classifiers**: XGBoost, Random Forest, 2-layer MLP (≤10M params) over trajectory features
    - **Hidden-state probe**: Extract LLM layer-16 hidden states at iteration t, train linear classifier (following "Learning to Route with Confidence Tokens")
    - **Small transformer**: 100M param encoder over trajectory history (≤100M params)
- **Baselines**:
    - **Fixed-3, Fixed-5** (production systems)
    - **Stop-on-success**: Stop when self-verification predicts success
    - **Stop-on-plateau**: Stop after 2 iterations with <5% improvement in pass rate
    - **Stop-on-duplicate**: Stop after 2 consecutive identical patches (Levenshtein <5)
    - **Complexity heuristic**: Stop if cyclomatic complexity unchanged for 2 iterations OR increases >20%
    - **Pass-rate-plateau**: Stop if pass@k improvement <5% for 2 iterations
- **Training**:
    - Primary: Supervised learning on Oracle-First labels (80/20 train/val split, stratified by task difficulty)
    - Class balancing: weighted loss or SMOTE for minority "stop" class
    - **Calibration baseline**: Measure LLM self-verification Expected Calibration Error (ECE) using Spiess et al. methodology; report as baseline for improvement; compare stopping accuracy of learned classifier vs calibrated self-verification (Platt scaling on verifier scores)
    - Cross-dataset generalization: train on HumanEval+MBPP, test on APPS
    - Future work: RL fine-tuning with reward $R = \Delta \text{pass@k} - \lambda \cdot C_{\text{iteration}}$ (penalize wasted iterations)

**4. Evaluation**

- **Primary experiments**:
    - **Pareto curves**: pass@k vs compute cost (iterations × model FLOPs OR cumulative # LLM calls)
    - Success rate at fixed budget (e.g., 100 LLM calls)
    - Stopping accuracy: precision/recall/F1 for "should stop" prediction
    - Regret vs oracle policies
- **Ablations**:
    - **Feature groups**: Code-only, test-only, self-verification-only, history-only, coverage-only, full feature set
    - **Model architecture**: XGBoost vs Random Forest vs MLP vs transformer vs hidden-state probe
    - **Repair strategy generalization**: Train on Reflexion, test on ReVeal/ChatRepair
    - **Model size**: GPT-3.5 vs GPT-4 vs Claude-3.5 (7B, 13B, 34B base LLMs)
    - **Task difficulty**: HumanEval-Easy (pass@1 >50%) vs APPS-Hard problems
- **Error analysis**:
    - Confusion matrix for stop/continue decisions (false positives: stop too early vs false negatives: waste iterations)
    - Where does stopping fail? (easy vs hard problems, specific error types, early vs late iterations, oscillating solutions, plateaus misidentified as progress)
    - Compare learned policy to calibrated self-verification: when does learned classifier outperform Platt scaling?
    - Qualitative analysis of failure modes
- **Generalization tests**:
    - Cross-dataset (train HumanEval, test MBPP/APPS)
    - Unseen repair strategies (train on Reflexion+ChatRepair, test on new method)
    - Model size generalization
    - Domain shift: train on algorithmic code, test on web/data science code

### Lab Fit

- **Cursor**: Composer multi-turn editing efficiency, inline agent iteration budgets—direct production applicability
- **Anthropic**: Claude Code repair loops, MCP workflow optimization, alignment-relevant sandbagging concerns
- **OpenAI**: Codex agent runtime optimization, post-training compute allocation, compute budget enforcement
- **DeepMind**: AlphaCode verification & repair pipelines, repair strategy optimization
- **Aider / Replit / Sourcegraph**: Immediate integration into existing repair systems, direct A/B testing opportunity

### Feasibility

**Timeline**: 10 weeks total (fits COLM March 31 deadline)

- Week 1: **Pilot validation (CRITICAL: GO/NO-GO decision)**
- Weeks 2-4: Full RepairStop-1K dataset collection (if GREEN)
- Weeks 5-7: Model training + ablations
- Weeks 8-9: Cross-dataset generalization + error analysis
- Week 10: Writing + figures

**Compute**: ~$3K

- Trajectory collection: $1.5K (1,000 problems × 5 iterations × $0.30/problem)
- Classifier training: $500 (lightweight models, CPU-bound)
- Ablations + generalization: $1K

**Key Risk**: Trajectory features may be weakly predictive; simple heuristics (pass-rate plateau, duplicate detection) could be near-optimal

- **Mitigation**:
    - **Pilot validation** (Week 1): Measure waste rate AND self-verification ECE AND feature AUC. If waste >25% AND ECE >0.2 AND feature AUC >0.65 → GREEN LIGHT for full study
    - If prediction hard OR heuristics work well, pivot to **characterization paper** (waste rates, duplication analysis, convergence patterns, benchmark contribution) — still publishable at EMNLP/TMLR

**Data Dependencies**: None (uses public benchmarks: HumanEval, MBPP, APPS)

**Model Dependencies**: Access to GPT-3.5/GPT-4/Claude API for trajectory collection

### Leverage for Research Program

- **RepairStop-1K Benchmark** reusable for:
    - Topic 5 (Critic Separation in Code Verification) — solver-critic stopping
    - Topic 9 (Budget Allocation Across Gen-Verify-Repair) — optimal cost splits
    - Topic 7 (Multi-Agent Minimal Configs) — stopping policy for agent coordination
- Infrastructure (repair trajectory logging + feature extraction) extends to:
    - Other repair strategies (non-code domains: math reasoning, dialog)
    - Multi-agent minimal configs
    - Any iterative refinement task
- Establishes **iteration-level stopping** as first-class code reasoning primitive
- **Research narrative**: Complements REx (which branch?) with RepairStop (when stop?)
- **Neuro-symbolic extension** (future work): Augment RepairStop with formal verification signals (Z3, static analyzers) to predict when *provable correctness* obviates further repair
- **Future extensions**: Multi-objective stopping (balance correctness, efficiency, code quality), transfer learning across domains (code → math → dialog)

### Critical Next Step: 1-Week Pilot (REQUIRED)

**Experiment**:

```python
# Run Reflexion on HumanEval[:100] for 5 iterations each
for problem in HumanEval[:100]:
	trajectory = []
	for iteration in range(5):
		patch = reflexion_repair(problem, trajectory)
		pass_at_k = evaluate_tests(patch, tests)
		features = extract_all_features(patch, trajectory, tests)
		self_verification_score = model_confidence(patch, problem)
		# NEW
		trajectory.append({
			'code': patch,
			'pass_at_k': pass_at_k,
			'features': features,
			'self_verification': self_verification_score
		})
# Analyze:
# 1. Waste rate: % iterations after first pass@k ≥80% (Oracle-First)
# 2. Self-verification ECE: calibration of model confidence vs actual correctness
# 3. Feature correlations with "next iteration improves pass@k ≥5%"
# 4. Baseline performance: fixed-3 vs pass-rate-plateau vs oracle regret
# 5. Duplicate rate: % consecutive identical patches
```

**Decision Criteria**:

- **GREEN LIGHT** (proceed to full COLM submission):
    - Waste rate >25% (substantial room for improvement) AND
    - Self-verification ECE >0.2 (poorly calibrated, justifies learned stopping) AND
    - Features show predictive signal (AUC >0.65 for "next iteration improves")
- **YELLOW LIGHT** (reframe as characterization + benchmark paper):
    - Waste rate 15-25% OR features weakly predictive (AUC 0.55-0.65) →
    - Pivot to empirical characterization paper at EMNLP/TMLR
    - Contribution: RepairStop-1K benchmark + waste analysis + oracle comparisons
    - No learned stopping models, just empirical characterization
- **RED LIGHT** (abandon, move to Topic 2 or Topic 5):
    - Simple heuristics (pass-rate plateau, duplicate detection) within 5% of oracle performance OR
    - Waste rate <15% (problem not significant) OR
    - Features AUC <0.55 (trajectory signals uninformative)

**Timeline**: 1 week pilot → Decision by Feb 17, 2026 → If GREEN, full study starts immediately

**DO NOT commit 10 weeks to full COLM study without this validation.**

### Recruitment Impact Analysis

**If GREEN pilot + COLM acceptance**:

- **Cursor**: 65% → 80% (+15pp)
    - Directly solves Composer multi-turn efficiency
    - Addresses iteration budget in inline agent loops
- **Anthropic**: 60% → 70% (+10pp)
    - Claude Code repair loop optimization
    - Production-oriented thinking about compute allocation
- **OpenAI**: 55% → 65% (+10pp)
    - Codex agent runtime optimization
    - Demonstrates understanding of real deployment constraints
- **DeepMind**: 50% → 60% (+10pp)
    - AlphaCode repair pipeline relevance
    - Rigorous experimental methodology

**If YELLOW/RED pilot**:

- Returns to baseline odds (no boost)
- Time cost: 1 week (acceptable exploratory cost)

**Assessment**: Strong positive impact IF pilot validates GREEN criteria. Production-oriented thinking (waste reduction, API cost optimization) aligns perfectly with Research Engineer positioning. Main risk: characterization-only outcome (YELLOW) less compelling than learned solution for recruitment.

### Production Validation (Stretch Goal)

**A/B Testing Partnership**: Collaborate with Aider/Cursor to deploy learned stopping policies in production code agents, measuring:

- Real-world compute savings (API costs reduced, latency reduction)
- User satisfaction (correct solutions per session, time to resolution)
- Edge cases where learned policy underperforms fixed limits

**Expected Outcome**: 20-30% API cost savings with <5% success rate reduction

**Value for Paper**: Section on "Production Deployment" showing real-world impact beyond benchmarks

### Success Criteria

**COLM Acceptance Criteria**:

- Pilot validation (GREEN scenario achieved)
- RepairStop-1K benchmark contribution (novel artifact addressing CTA-7)
- Learned stopping policy outperforms baselines by ≥10% waste reduction
- Addresses Pujar et al. 2025 CTAs (6, 4, 7) with explicit citations in intro/method/benchmark sections
- Cross-dataset generalization demonstrated (train HumanEval, test MBPP/APPS)
- Ablations show learned features outperform simple heuristics

**Recruitment Value**: Direct production applicability + rigorous methodology + survey alignment + novel benchmark contribution

**Timeline Constraint**: 10 weeks total → March 31 COLM deadline feasible with 1-week pilot decision by Feb 17