"""
Evaluation Metrics for RRMC.

Provides publishable reporting metrics:
- Risk-coverage curves
- ECE (Expected Calibration Error)
- Bootstrap confidence intervals
- Homogeneity diagnostics (effective rank, repetition, prompt sensitivity)
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from scipy import stats as scipy_stats


# =========================================================================
# Risk-Coverage Curves
# =========================================================================

@dataclass
class RiskCoveragePoint:
    """A single point on the risk-coverage curve."""
    threshold: float
    coverage: float  # fraction of states with score <= threshold
    error_rate: float  # error rate among covered states


@dataclass
class RiskCoverageCurve:
    """Full risk-coverage curve."""
    points: List[RiskCoveragePoint]
    auc: float  # Area under the risk-coverage curve


def compute_risk_coverage_curve(
    scores: List[float],
    errors: List[int],
    n_thresholds: int = 50,
) -> RiskCoverageCurve:
    """
    Compute risk-coverage curve over varying thresholds.

    For each threshold tau, compute:
    - coverage = fraction of states with score <= tau
    - error_rate = error rate among those covered states

    Args:
        scores: MI or uncertainty scores per state.
        errors: Binary error indicators (1=wrong, 0=correct).
        n_thresholds: Number of threshold values to evaluate.

    Returns:
        RiskCoverageCurve with points and AUC.
    """
    if not scores or not errors:
        return RiskCoverageCurve(points=[], auc=0.0)

    scores_arr = np.array(scores)
    errors_arr = np.array(errors)
    n = len(scores_arr)

    # Generate threshold values spanning the range of scores
    thresholds = np.linspace(scores_arr.min(), scores_arr.max(), n_thresholds)

    points = []
    for tau in thresholds:
        mask = scores_arr <= tau
        n_covered = mask.sum()
        if n_covered == 0:
            continue
        coverage = n_covered / n
        error_rate = errors_arr[mask].mean()
        points.append(RiskCoveragePoint(
            threshold=float(tau),
            coverage=float(coverage),
            error_rate=float(error_rate),
        ))

    # Compute AUC via trapezoidal rule (coverage on x, error on y)
    auc = 0.0
    if len(points) >= 2:
        coverages = [p.coverage for p in points]
        error_rates = [p.error_rate for p in points]
        auc = float(np.trapz(error_rates, coverages))

    return RiskCoverageCurve(points=points, auc=auc)


# =========================================================================
# Expected Calibration Error (ECE)
# =========================================================================

@dataclass
class ECEResult:
    """Expected Calibration Error result."""
    ece: float
    n_bins: int
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]


@dataclass
class PerturbedECEResult:
    """ECE result for perturbed ensemble."""
    ece: float
    n_bins: int
    bin_accuracies: List[float]
    bin_confidences: List[float]
    bin_counts: List[int]
    n_samples: int
    cluster_entropy: float
    homogeneity_score: float


def compute_ece(
    scores: List[float],
    errors: List[int],
    n_bins: int = 10,
) -> ECEResult:
    """
    Compute Expected Calibration Error on visited states.

    Interprets 1 - score as "confidence" (lower MI => higher confidence).
    Groups states into bins by confidence and computes the gap between
    average confidence and average accuracy per bin.

    Args:
        scores: MI or uncertainty scores per state.
        errors: Binary error indicators (1=wrong, 0=correct).
        n_bins: Number of calibration bins.

    Returns:
        ECEResult with per-bin statistics.
    """
    if not scores or not errors:
        return ECEResult(ece=0.0, n_bins=n_bins, bin_accuracies=[], bin_confidences=[], bin_counts=[])

    scores_arr = np.array(scores)
    errors_arr = np.array(errors)

    # Convert scores to confidence: normalise to [0,1] then invert
    if scores_arr.max() > scores_arr.min():
        conf = 1.0 - (scores_arr - scores_arr.min()) / (scores_arr.max() - scores_arr.min())
    else:
        conf = np.ones_like(scores_arr)

    accuracy = 1.0 - errors_arr

    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0

    for i in range(n_bins):
        mask = (conf >= bin_boundaries[i]) & (conf < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (conf == bin_boundaries[i + 1])
        count = mask.sum()
        bin_counts.append(int(count))
        if count > 0:
            avg_acc = accuracy[mask].mean()
            avg_conf = conf[mask].mean()
            bin_accs.append(float(avg_acc))
            bin_confs.append(float(avg_conf))
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)

    ece /= len(scores_arr)

    return ECEResult(
        ece=float(ece),
        n_bins=n_bins,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
    )


def compute_perturbed_ece(
    confidences: List[float],
    accuracies: List[float],
    n_bins: int = 10,
) -> PerturbedECEResult:
    """
    Compute ECE for perturbed ensemble samples.
    
    Unlike standard ECE which derives confidence from MI scores,
    this uses explicit confidence values (e.g., cluster frequencies)
    from the perturbed ensemble.
    
    Args:
        confidences: Confidence scores per sample (0-1 scale)
        accuracies: Accuracy per sample (0 or 1)
        n_bins: Number of calibration bins
    
    Returns:
        PerturbedECEResult with per-bin statistics
    """
    if not confidences or not accuracies:
        return PerturbedECEResult(
            ece=0.0, n_bins=n_bins, bin_accuracies=[], bin_confidences=[],
            bin_counts=[], n_samples=0, cluster_entropy=0.0, homogeneity_score=0.0
        )
    
    conf_arr = np.array(confidences)
    acc_arr = np.array(accuracies)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    
    for i in range(n_bins):
        mask = (conf_arr >= bin_boundaries[i]) & (conf_arr < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (conf_arr == bin_boundaries[i + 1])
        
        count = mask.sum()
        bin_counts.append(int(count))
        
        if count > 0:
            avg_acc = acc_arr[mask].mean()
            avg_conf = conf_arr[mask].mean()
            bin_accs.append(float(avg_acc))
            bin_confs.append(float(avg_conf))
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
    
    ece /= len(conf_arr)
    
    # Compute cluster entropy (Shannon entropy of confidence distribution)
    conf_unique, conf_counts = np.unique(conf_arr, return_counts=True)
    probs = conf_counts / len(conf_arr)
    entropy = -np.sum(probs * np.log(probs + 1e-10))
    
    # Homogeneity score: max confidence frequency
    homogeneity = float(np.max(conf_counts) / len(conf_arr))
    
    return PerturbedECEResult(
        ece=float(ece),
        n_bins=n_bins,
        bin_accuracies=bin_accs,
        bin_confidences=bin_confs,
        bin_counts=bin_counts,
        n_samples=len(conf_arr),
        cluster_entropy=float(entropy),
        homogeneity_score=homogeneity,
    )


# =========================================================================
# Bootstrap Confidence Intervals
# =========================================================================

@dataclass
class BootstrapCI:
    """Bootstrap confidence interval for a statistic."""
    mean: float
    ci_lower: float
    ci_upper: float
    std: float
    n_bootstrap: int


def bootstrap_ci(
    values: List[float],
    statistic: str = "mean",
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> BootstrapCI:
    """
    Compute bootstrap confidence interval for a statistic.

    Args:
        values: Sample values.
        statistic: "mean" or "median".
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed.

    Returns:
        BootstrapCI with mean, lower, upper bounds.
    """
    if not values:
        return BootstrapCI(mean=0.0, ci_lower=0.0, ci_upper=0.0, std=0.0, n_bootstrap=n_bootstrap)

    rng = np.random.RandomState(seed)
    arr = np.array(values)
    n = len(arr)

    stat_fn = np.mean if statistic == "mean" else np.median
    observed = float(stat_fn(arr))

    boot_stats = []
    for _ in range(n_bootstrap):
        resample = arr[rng.randint(0, n, size=n)]
        boot_stats.append(float(stat_fn(resample)))

    boot_stats = np.array(boot_stats)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(boot_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(boot_stats, 100 * (1 - alpha / 2)))

    return BootstrapCI(
        mean=observed,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        std=float(np.std(boot_stats)),
        n_bootstrap=n_bootstrap,
    )


def bootstrap_accuracy_ci(
    correct: List[bool],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Bootstrap CI for accuracy."""
    return bootstrap_ci(
        [float(c) for c in correct],
        statistic="mean",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


def bootstrap_turns_ci(
    turns: List[int],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> BootstrapCI:
    """Bootstrap CI for average turns used."""
    return bootstrap_ci(
        [float(t) for t in turns],
        statistic="mean",
        n_bootstrap=n_bootstrap,
        confidence=confidence,
    )


# =========================================================================
# Homogeneity Diagnostics
# =========================================================================

@dataclass
class HomogeneityDiagnostics:
    """Diagnostics for answer distribution homogeneity."""
    effective_rank: float
    max_cluster_frequency: float
    repetition_rate: float
    prompt_sensitivity_index: Optional[float]
    n_samples: int


def compute_effective_rank(cluster_sizes: Dict[int, int]) -> float:
    """
    Compute effective rank of the cluster distribution.

    Effective rank = exp(H(p)) where H is Shannon entropy.
    A uniform distribution over k clusters gives effective rank k.
    A collapsed distribution gives effective rank 1.

    Args:
        cluster_sizes: Mapping cluster_id -> count.

    Returns:
        Effective rank (>= 1).
    """
    if not cluster_sizes:
        return 1.0

    total = sum(cluster_sizes.values())
    if total == 0:
        return 1.0

    probs = np.array([c / total for c in cluster_sizes.values()])
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def compute_repetition_rate(answers: List[str]) -> float:
    """
    Compute repetition rate: fraction of answers identical to another.

    Args:
        answers: List of answer strings.

    Returns:
        Repetition rate in [0, 1]. 0 means all unique, 1 means all same.
    """
    if len(answers) <= 1:
        return 0.0
    unique = set(answers)
    return 1.0 - len(unique) / len(answers)


def compute_prompt_sensitivity_index(
    variant_estimates: Dict[str, Any],
) -> Optional[float]:
    """
    Compute prompt sensitivity index from multi-variant MI estimates.

    Measures the spread of MI values across prompt variants:
    PSI = std(MI values) / mean(MI values).
    High PSI indicates the model is sensitive to prompt wording.

    Args:
        variant_estimates: Dict mapping variant name -> MIEstimate
                          (or any object with .mi attribute).

    Returns:
        Prompt sensitivity index, or None if insufficient data.
    """
    mi_values = []
    for est in variant_estimates.values():
        if hasattr(est, "mi"):
            mi_values.append(est.mi)

    if len(mi_values) < 2:
        return None

    mean_mi = np.mean(mi_values)
    if mean_mi == 0:
        return 0.0
    return float(np.std(mi_values) / mean_mi)


def compute_homogeneity_diagnostics(
    answers: List[str],
    cluster_sizes: Dict[int, int],
    variant_estimates: Optional[Dict[str, Any]] = None,
) -> HomogeneityDiagnostics:
    """
    Compute full homogeneity diagnostics.

    Args:
        answers: All answer strings from MI sampling.
        cluster_sizes: Cluster size distribution.
        variant_estimates: Optional variant -> MIEstimate dict.

    Returns:
        HomogeneityDiagnostics.
    """
    total = sum(cluster_sizes.values()) if cluster_sizes else len(answers)
    max_freq = max(cluster_sizes.values()) / total if cluster_sizes and total > 0 else 1.0

    return HomogeneityDiagnostics(
        effective_rank=compute_effective_rank(cluster_sizes),
        max_cluster_frequency=max_freq,
        repetition_rate=compute_repetition_rate(answers),
        prompt_sensitivity_index=compute_prompt_sensitivity_index(variant_estimates) if variant_estimates else None,
        n_samples=len(answers),
    )


# =========================================================================
# Aggregation: generate a full metrics report
# =========================================================================

@dataclass
class MetricsReport:
    """Full metrics report for a method evaluation."""
    accuracy_ci: BootstrapCI
    turns_ci: BootstrapCI
    risk_coverage: Optional[RiskCoverageCurve] = None
    ece: Optional[ECEResult] = None
    homogeneity: Optional[HomogeneityDiagnostics] = None
    mi_error_correlation: Optional[Dict[str, float]] = None
    perturbed_ece: Optional[PerturbedECEResult] = None


def generate_metrics_report(
    correct_flags: List[bool],
    turns_used: List[int],
    mi_scores: Optional[List[float]] = None,
    errors: Optional[List[int]] = None,
    answers: Optional[List[str]] = None,
    cluster_sizes: Optional[Dict[int, int]] = None,
    variant_estimates: Optional[Dict[str, Any]] = None,
    n_bootstrap: int = 1000,
) -> MetricsReport:
    """
    Generate a full metrics report from evaluation results.

    Args:
        correct_flags: Per-episode correctness.
        turns_used: Per-episode turns used.
        mi_scores: MI scores at decision points (for calibration states).
        errors: Binary error flags at decision points.
        answers: Answer strings from MI sampling.
        cluster_sizes: Cluster distribution for homogeneity.
        variant_estimates: Multi-variant MI estimates.
        n_bootstrap: Bootstrap resamples.

    Returns:
        MetricsReport with all available metrics.
    """
    accuracy_ci = bootstrap_accuracy_ci(correct_flags, n_bootstrap=n_bootstrap)
    turns_ci = bootstrap_turns_ci(turns_used, n_bootstrap=n_bootstrap)

    risk_coverage = None
    ece_result = None
    mi_corr = None
    if mi_scores and errors and len(mi_scores) == len(errors):
        risk_coverage = compute_risk_coverage_curve(mi_scores, errors)
        ece_result = compute_ece(mi_scores, errors)
        if len(set(mi_scores)) >= 2 and len(set(errors)) >= 2:
            try:
                rho, pval = scipy_stats.spearmanr(mi_scores, errors)
                mi_corr = {"rho": float(rho), "pvalue": float(pval)}
            except Exception:
                pass

    homogeneity = None
    if answers and cluster_sizes:
        homogeneity = compute_homogeneity_diagnostics(
            answers, cluster_sizes, variant_estimates
        )

    return MetricsReport(
        accuracy_ci=accuracy_ci,
        turns_ci=turns_ci,
        risk_coverage=risk_coverage,
        ece=ece_result,
        homogeneity=homogeneity,
        mi_error_correlation=mi_corr,
    )
