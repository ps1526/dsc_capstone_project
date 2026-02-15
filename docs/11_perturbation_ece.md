# Weight Perturbation + ECE Integration for RRMC

**Branch:** `blob`  
**Created:** 2026-02-13  
**Status:** Experimental

## Overview

This document describes the perturbation-based ECE integration implemented in the `blob` branch, which combines weight perturbation simulation with Expected Calibration Error (ECE) for enhanced uncertainty quantification in RRMC.

## Motivation

Traditional RRMC uses self-revision mutual information (MI) as an uncertainty signal. However, MI alone may not capture **calibration quality** — whether the model's confidence matches its accuracy. This branch adds:

1. **Perturbed ensemble sampling** to simulate weight noise (since API models don't expose weights)
2. **ECE computation** per state to measure calibration error
3. **Combined MI+ECE** uncertainty for stopping decisions and question selection

## Architecture

### Output-Space Perturbation

Since OpenRouter API provides no direct weight access, we simulate weight perturbation through **output-space ensemble diversity**:

```
Temperature Variation → Simulates softmax temperature / logit noise
Prompt Perturbation  → Creates output distribution variance
Embedding Noise      → Post-hoc noise on answer representations
```

This approximates the effect of weight perturbation by measuring output distribution spread.

### Key Components

#### 1. `rrmc/core/perturbation.py`

Core perturbation infrastructure:

- **`GaussianNoiseStrategy`**: Normal distribution noise N(0, σ²)
- **`LaplaceNoiseStrategy`**: Double exponential (heavier tails)
- **`PerturbedEnsembleSampler`**: Generates N perturbed samples using:
  - Temperature schedule (0.3 to 1.2)
  - Prompt variants (deterministic prefixes)
  - Post-hoc embedding noise
- **`compute_ensemble_ece()`**: ECE from perturbed sample confidences

#### 2. `rrmc/core/mi_estimator_ece.py`

Combined MI+ECE estimation:

- **`RobustMIWithECE`**: Extends `RobustMI` with ECE
- **`CombinedUncertaintyEstimate`**: Stores MI, ECE, and combined score
- **Weighting**: `combined_score = α * MI + β * ECE`

Default: α=0.5, β=0.5 (equal weight)

#### 3. `rrmc/methods/stopping_rules.py`

New stopping rules:

- **`PerturbedMIECEStopping`**: Uses combined MI+ECE
- **`ECEOnlyStopping`**: ECE-only baseline (α=0, β=1)

#### 4. `rrmc/methods/question_selector_ece.py`

ECE-enhanced question selection:

- **`ECEQuestionSelector`**: Extends VoI with ECE
- Scores questions by: `E[MI_reduction] + β * E[ECE_reduction]`
- Simulates outcomes and measures expected calibration improvement

## Configuration

### Experiment Config Example

```yaml
# configs/experiments/blob/blob_dc_gaussian.yaml
experiment_name: blob/blob_dc_gaussian
task: dc

# Perturbation settings
perturbation:
  enabled: true
  noise_type: gaussian  # or laplace
  noise_scale: 0.1
  n_perturbed_samples: 8
  temperature_range: [0.3, 1.2]
  use_prompt_perturbation: true
  use_embedding_noise: true

# Combined uncertainty
combined_uncertainty:
  use_ece: true
  alpha: 0.5  # MI weight
  beta: 0.5   # ECE weight
  ece_bins: 10

# Methods to compare
methods:
  - fixed_turns
  - robust_mi
  - perturbed_mi_ece
  - ece_only
```

### Method Config Example

```yaml
# configs/methods/perturbed_mi_ece.yaml
threshold: 0.3
k_samples: 8
alpha: 0.5
beta: 0.5
perturbation:
  noise_type: gaussian
  noise_scale: 0.1
  temperature_range: [0.3, 1.2]
```

## Usage

### Running Experiments

```bash
# Gaussian noise on DC task
python run.py blob/blob_dc_gaussian --n_train 30 --n_test 20

# Laplace noise on DC task
python run.py blob/blob_dc_laplace --n_train 30 --n_test 20

# GN diagnostic experiment
python run.py blob/blob_gn_gaussian --n_puzzles 50

# Full comparison
python run.py blob/blob_comparison --n_train 40 --n_test 30
```

### Analyzing Results

```bash
# Generate analysis plots and tables
python scripts/analyze_ece_results.py results analysis_output

# Outputs:
#   analysis_output/summary.txt          - Performance table
#   analysis_output/ece_vs_mi.png        - ECE vs MI scatter
#   analysis_output/noise_sensitivity.png - Performance by noise type
```

## Implementation Details

### ECE Computation per State

At each turn t, we:

1. Generate perturbed ensemble (N samples with noise)
2. Cluster samples into semantic equivalence classes
3. Compute confidence per sample = cluster frequency
4. Compute ECE over (confidence, accuracy) pairs

This gives us **ECE_t** as a state feature alongside **MI_t**.

### Stopping Decision

```
Stop if: α * MI + β * ECE ≤ τ
```

Where τ is calibrated using Clopper-Pearson on train states.

### Question Selection

For each candidate question q:

1. Simulate R outcomes (NPC responses)
2. Compute MI and ECE in each hypothetical future
3. Score q by expected uncertainty reduction:
   ```
   score(q) = E[α * ΔMI + β * ΔECE | ask(q)]
   ```

Select question with highest score.

## Key Design Decisions

### Why Temperature/Prompt Perturbation?

OpenRouter API has no weight access. Our approximation:

- **Temperature variation**: Simulates softmax temperature ≈ logit-level noise
- **Prompt perturbation**: Deterministic prefixes create distribution variance
- **Embedding noise**: Post-hoc noise on representations

This is conceptually similar to dropout or weight noise at inference time, measuring output uncertainty without direct weight manipulation.

### ECE as State Feature

ECE is computed **per state** (each turn), not just post-hoc:

- Tracks calibration quality throughout the episode
- Combined with MI for richer uncertainty signal
- Can be weighted adaptively (α, β tunable)

### Noise Type Selection

- **Gaussian**: Symmetric, standard choice
- **Laplace**: Heavier tails, better for epistemic uncertainty

Hypothesis: Laplace captures model disagreement better than Gaussian.

## Experiments and Hypotheses

### H1: Perturbation improves ECE

**Claim**: Perturbed ensembles have better-calibrated confidences.

**Test**: Compare ECE before/after perturbation on calibration states.

### H2: MI+ECE outperforms MI-only

**Claim**: Combined uncertainty better predicts errors.

**Test**: Compare stopping accuracy: `perturbed_mi_ece` vs `robust_mi` vs `ece_only`.

### H3: ECE-based questions improve calibration

**Claim**: Selecting questions by ECE reduction asks better calibration-improving questions.

**Test**: Compare question quality (manual eval on 10 examples) and downstream accuracy.

### H4: Laplace > Gaussian

**Claim**: Laplace noise (heavier tails) better captures epistemic uncertainty.

**Test**: Compare `blob_dc_laplace` vs `blob_dc_gaussian` on accuracy and ECE.

## Expected Outcomes

### Success Criteria

1. **ECE values decrease** as investigation progresses (model gets more calibrated)
2. **MI+ECE correlation** is positive but not perfect (captures different aspects)
3. **Combined stopping** achieves ≥ accuracy of MI-only with ≤ turns
4. **ECE-based questions** produce sensible, calibration-improving queries

### Metrics to Track

- Accuracy per method
- Average turns per method
- ECE per state (mean, std, trajectory)
- MI-ECE Spearman correlation
- Question quality (manual inspection)
- Noise sensitivity (performance vs noise scale/type)

## Limitations

### Not True Weight Perturbation

This is an **approximation**. True weight noise would:

- Perturb all layer weights
- Affect hidden representations directly
- Require model access

Our approach only perturbs the output distribution, which is a looser proxy.

### Calibration is On-Policy

ECE is computed using model samples (not a held-out validation set). This means:

- ECE may overestimate calibration quality
- Ground truth needed for accurate ECE (not always available)
- Fallback: use cluster confidence as proxy

### Computational Cost

Perturbed ensemble adds API calls:

- N perturbed samples per state (default: 8)
- R simulations per candidate question (default: 3)
- Total: ~8x cost vs baseline

Use smaller N for production or cache states.

## Future Work

### Post-MVP Improvements

1. **Adaptive weighting**: Learn α, β from calibration data
2. **State-dependent ECE**: Weight ECE more when MI is low
3. **DPP slate selection**: Use DPP for diverse perturbed samples
4. **True weight perturbation**: If local models added later

### Integration with Existing Features

- **Trajectory ensemble**: Combine with perturbation for double uncertainty
- **AllSuspectsWrapper**: Force all suspects before stopping
- **VoI/DQS**: Extend with ECE scoring

## References

### Related Work

- **Weight Perturbation**: Bayesian dropout, weight noise for uncertainty
- **ECE**: Guo et al. (2017) "On Calibration of Modern Neural Networks"
- **Ensemble Uncertainty**: Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty Estimation"

### RRMC Docs

- `docs/00_proposal.md`: Original RRMC specification
- `docs/01_literature.md`: Curated literature map
- `docs/02_implementation.md`: Implementation guide

## Contributing

To extend this work:

1. Add new noise strategies: Implement `NoiseStrategy` subclass
2. Add new ECE variants: Extend `compute_perturbed_ece()`
3. Add new question scorers: Extend `ECEQuestionSelector`
4. Run ablations: Vary α, β, noise_scale in configs

## Appendix: Code Map

```
rrmc/core/perturbation.py          - Noise strategies, perturbed sampler
rrmc/core/mi_estimator_ece.py      - Combined MI+ECE estimation
rrmc/methods/stopping_rules.py     - PerturbedMIECEStopping, ECEOnlyStopping
rrmc/methods/question_selector_ece.py - ECE question selection
rrmc/evaluation/metrics.py         - PerturbedECEResult, compute_perturbed_ece
pipeline.py                         - Config parsing for perturbation/ECE
configs/experiments/blob/           - Blob experiment configs
configs/methods/perturbed_mi_ece.yaml - Method config
scripts/analyze_ece_results.py     - Analysis and plotting
```

## Contact

For questions about this branch:

- See `docs/README.md` for general RRMC documentation
- Check `prompt/` for implementation history
- Run `python run.py --list` to see available experiments
