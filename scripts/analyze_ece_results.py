"""
Analyze ECE Results from Blob Experiments.

Generates plots and analysis of perturbation+ECE experiments.
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import numpy as np

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib/seaborn not installed. Plots disabled.")


def load_results(results_dir: str, pattern: str = "blob") -> Dict[str, Any]:
    """
    Load results from blob experiments.
    
    Args:
        results_dir: Results directory
        pattern: File pattern to match
    
    Returns:
        Dict mapping method -> results
    """
    results_path = Path(results_dir)
    all_results = {}
    
    # Search for blob experiment results
    for method_dir in results_path.glob("baseline/dc/*"):
        if not method_dir.is_dir():
            continue
        
        method_name = method_dir.name
        
        # Find most recent blob result
        blob_files = sorted(method_dir.glob("*blob*.json"))
        if blob_files:
            with open(blob_files[-1], 'r') as f:
                data = json.load(f)
                all_results[method_name] = data
    
    return all_results


def analyze_ece_vs_mi(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Compute correlation between ECE and MI across states.
    
    Args:
        results: Results dict
    
    Returns:
        Dict with correlation statistics
    """
    mi_scores = []
    ece_scores = []
    
    for method, data in results.items():
        if not isinstance(data, list):
            continue
        
        for episode in data:
            decisions = episode.get("decisions", [])
            for decision in decisions:
                if "mi" in decision and "ece" in decision:
                    mi_scores.append(decision["mi"])
                    ece_scores.append(decision["ece"])
    
    if len(mi_scores) < 2:
        return {"correlation": 0.0, "n_samples": 0}
    
    from scipy import stats
    corr, pval = stats.pearsonr(mi_scores, ece_scores)
    
    return {
        "correlation": float(corr),
        "p_value": float(pval),
        "n_samples": len(mi_scores),
    }


def plot_ece_vs_mi(results: Dict[str, Any], output_path: str):
    """
    Plot ECE vs MI scatter.
    
    Args:
        results: Results dict
        output_path: Output image path
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return
    
    mi_scores = []
    ece_scores = []
    methods = []
    
    for method, data in results.items():
        if not isinstance(data, list):
            continue
        
        for episode in data:
            decisions = episode.get("decisions", [])
            for decision in decisions:
                if "mi" in decision and "ece" in decision:
                    mi_scores.append(decision["mi"])
                    ece_scores.append(decision["ece"])
                    methods.append(method)
    
    if not mi_scores:
        print("No ECE/MI data found")
        return
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=mi_scores, y=ece_scores, hue=methods, alpha=0.6)
    plt.xlabel("Mutual Information (nats)")
    plt.ylabel("Expected Calibration Error")
    plt.title("ECE vs MI Across Methods")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved plot to {output_path}")


def plot_noise_sensitivity(
    results_dir: str,
    noise_types: List[str] = ["gaussian", "laplace"],
    output_path: str = "noise_sensitivity.png"
):
    """
    Plot performance across noise types/scales.
    
    Args:
        results_dir: Results directory
        noise_types: Noise types to compare
        output_path: Output path
    """
    if not HAS_MATPLOTLIB:
        print("Skipping plot: matplotlib not available")
        return
    
    # Load results for different noise configs
    accuracies = {}
    for noise in noise_types:
        pattern = f"blob_dc_{noise}"
        results = load_results(results_dir, pattern)
        
        for method, data in results.items():
            if method not in accuracies:
                accuracies[method] = {}
            
            if isinstance(data, list):
                correct = sum(1 for ep in data if ep.get("correctness", False))
                acc = correct / len(data) if data else 0.0
                accuracies[method][noise] = acc
    
    if not accuracies:
        print("No noise sensitivity data found")
        return
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(noise_types))
    width = 0.15
    
    for i, (method, noise_accs) in enumerate(accuracies.items()):
        accs = [noise_accs.get(n, 0.0) for n in noise_types]
        ax.bar(x + i * width, accs, width, label=method)
    
    ax.set_xlabel("Noise Type")
    ax.set_ylabel("Accuracy")
    ax.set_title("Performance Across Noise Types")
    ax.set_xticks(x + width * len(accuracies) / 2)
    ax.set_xticklabels(noise_types)
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved noise sensitivity plot to {output_path}")


def generate_summary_table(results: Dict[str, Any]) -> str:
    """
    Generate summary table of results.
    
    Args:
        results: Results dict
    
    Returns:
        Formatted table string
    """
    lines = []
    lines.append("=" * 80)
    lines.append("BLOB EXPERIMENT SUMMARY")
    lines.append("=" * 80)
    lines.append(f"{'Method':<25} {'Accuracy':>10} {'Avg Turns':>10} {'Avg ECE':>10}")
    lines.append("-" * 80)
    
    for method, data in sorted(results.items()):
        if not isinstance(data, list):
            continue
        
        correct = sum(1 for ep in data if ep.get("correctness", False))
        acc = correct / len(data) if data else 0.0
        
        turns = [ep.get("round", 0) for ep in data]
        avg_turns = np.mean(turns) if turns else 0.0
        
        # Extract ECE from decisions
        ece_values = []
        for ep in data:
            for dec in ep.get("decisions", []):
                if "ece" in dec:
                    ece_values.append(dec["ece"])
        avg_ece = np.mean(ece_values) if ece_values else 0.0
        
        lines.append(f"{method:<25} {acc:>10.2%} {avg_turns:>10.1f} {avg_ece:>10.3f}")
    
    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    """Main analysis script."""
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "results"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "analysis_output"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Analyzing results from: {results_dir}")
    print(f"Output directory: {output_dir}")
    
    # Load results
    results = load_results(results_dir)
    
    if not results:
        print("No blob results found")
        return
    
    print(f"Found {len(results)} methods")
    
    # Generate summary
    summary = generate_summary_table(results)
    print("\n" + summary)
    
    with open(os.path.join(output_dir, "summary.txt"), "w") as f:
        f.write(summary)
    
    # Analyze ECE vs MI correlation
    print("\nAnalyzing ECE vs MI correlation...")
    corr_stats = analyze_ece_vs_mi(results)
    print(f"  Correlation: {corr_stats['correlation']:.4f} (p={corr_stats['p_value']:.4f})")
    print(f"  N samples: {corr_stats['n_samples']}")
    
    # Generate plots
    if HAS_MATPLOTLIB:
        print("\nGenerating plots...")
        plot_ece_vs_mi(results, os.path.join(output_dir, "ece_vs_mi.png"))
        plot_noise_sensitivity(results_dir, output_path=os.path.join(output_dir, "noise_sensitivity.png"))
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
