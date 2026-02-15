"""
MI Estimator with ECE Integration for RRMC.

Combines Robust MI with ECE for enhanced uncertainty quantification
using perturbed ensemble sampling.
"""

from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from .llm import LLMWrapper
from .clustering import SemanticClusterer
from .mi_estimator import SelfRevisionMI, RobustMI, MIEstimate
from .perturbation import (
    PerturbedEnsembleSampler,
    PerturbationConfig,
    compute_ensemble_ece,
)


@dataclass
class CombinedUncertaintyEstimate:
    """Combined MI + ECE uncertainty estimate."""
    mi: float
    ece: float
    combined_score: float
    mi_weight: float  # alpha
    ece_weight: float  # beta
    n_samples: int
    mi_estimate: Optional[MIEstimate] = None
    ece_details: Optional[Dict[str, Any]] = None
    perturbed_samples: Optional[List] = None


class RobustMIWithECE:
    """
    Robust MI estimator with ECE integration.
    
    Combines:
    1. Robust MI across prompt variants (existing)
    2. ECE from perturbed ensemble (new)
    3. Weighted combination for unified uncertainty
    """
    
    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 8,
        temperature: float = 0.7,
        variants: Optional[List[str]] = None,
        regime: str = "normal",
        # ECE-specific params
        use_ece: bool = True,
        alpha: float = 0.5,  # MI weight
        beta: float = 0.5,   # ECE weight
        perturbation_config: Optional[PerturbationConfig] = None,
        ece_bins: int = 10,
    ):
        """
        Initialize combined MI+ECE estimator.
        
        Args:
            llm: LLM wrapper
            clusterer: Semantic clusterer
            k_samples: Number of samples for MI
            temperature: Base sampling temperature
            variants: Prompt variants for robust MI
            regime: Decoding regime
            use_ece: Whether to compute and use ECE
            alpha: Weight for MI in combined score (0-1)
            beta: Weight for ECE in combined score (0-1)
            perturbation_config: Config for perturbed sampling
            ece_bins: Number of bins for ECE computation
        """
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.temperature = temperature
        self.variants = variants or ["base", "skeptical"]
        self.regime = regime
        
        # ECE parameters
        self.use_ece = use_ece
        self.alpha = alpha
        self.beta = beta
        self.ece_bins = ece_bins
        
        # Initialize robust MI (base estimator)
        self.robust_mi = RobustMI(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            temperature=temperature,
            use_diversity_sampling=True,
            regime=regime,
        )
        
        # Initialize perturbed sampler if using ECE
        if self.use_ece:
            if perturbation_config is None:
                perturbation_config = PerturbationConfig(
                    n_samples=k_samples,
                    noise_scale=0.1,
                    temperature_range=(0.3, 1.2),
                )
            self.perturbed_sampler = PerturbedEnsembleSampler(
                llm=llm,
                clusterer=clusterer,
                config=perturbation_config,
            )
        else:
            self.perturbed_sampler = None
    
    def estimate(
        self,
        task_type: str,
        state: Dict[str, Any],
        ground_truth: Optional[Any] = None,
    ) -> CombinedUncertaintyEstimate:
        """
        Estimate combined MI + ECE uncertainty for a state.
        
        Args:
            task_type: Task type (DC, SP, GN)
            state: Current environment state
            ground_truth: Optional ground truth for ECE accuracy
        
        Returns:
            CombinedUncertaintyEstimate with MI, ECE, and combined score
        """
        # Step 1: Compute Robust MI (existing approach)
        robust_mi_value, mi_estimates = self.robust_mi.estimate(
            task_type=task_type,
            state=state,
            variants=self.variants,
        )
        
        # Get best MI estimate for details
        best_variant = max(mi_estimates.keys(), key=lambda k: mi_estimates[k].mi)
        best_mi_estimate = mi_estimates[best_variant]
        
        # Step 2: Compute ECE if enabled
        ece_value = 0.0
        ece_details = None
        perturbed_samples = None
        
        if self.use_ece and self.perturbed_sampler is not None:
            # Build prompt from state
            prompt = self._get_answer_prompt(task_type, state)
            messages = [{"role": "user", "content": prompt}]
            
            # Generate perturbed ensemble
            perturbed_samples = self.perturbed_sampler.sample_perturbed(
                messages=messages,
                max_tokens=256,
                parallel=True,
            )
            
            # Compute ensemble ECE
            ece_details = compute_ensemble_ece(
                samples=perturbed_samples,
                clusterer=self.clusterer,
                task_type=task_type,
                ground_truth=ground_truth,
                n_bins=self.ece_bins,
            )
            ece_value = ece_details["ece"]
        
        # Step 3: Combine MI and ECE
        # Normalize both to similar scales (MI is typically 0-2 nats, ECE is 0-1)
        # We'll keep them as-is and weight appropriately
        combined_score = self.alpha * robust_mi_value + self.beta * ece_value
        
        return CombinedUncertaintyEstimate(
            mi=robust_mi_value,
            ece=ece_value,
            combined_score=combined_score,
            mi_weight=self.alpha,
            ece_weight=self.beta,
            n_samples=self.k_samples,
            mi_estimate=best_mi_estimate,
            ece_details=ece_details,
            perturbed_samples=perturbed_samples,
        )
    
    def compute_combined_uncertainty(
        self,
        mi_value: float,
        ece_value: float,
    ) -> float:
        """
        Compute combined uncertainty score from MI and ECE.
        
        Args:
            mi_value: Mutual information value
            ece_value: Expected calibration error
        
        Returns:
            Combined uncertainty score
        """
        return self.alpha * mi_value + self.beta * ece_value
    
    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        """
        Get best answer from the ensemble.
        
        Uses the robust MI's answer selection logic.
        
        Args:
            task_type: Task type
            state: Current state
        
        Returns:
            Best answer string
        """
        return self.robust_mi.get_best_answer(task_type, state)
    
    def _get_answer_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
        """Get answer prompt for task type."""
        if task_type == "DC":
            choices = state.get("choices", {})
            choices_str = ""
            if isinstance(choices, dict) and choices:
                formatted = [f"{k}: {v}" for k, v in choices.items()]
                choices_str = "Choices:\n" + "\n".join(formatted) + "\n\n"
            return f"""Based on the investigation, identify the murderer.

Case: {state.get('initial_info', '')}

Interrogation: {state.get('history_string', '')}

{choices_str}Answer ONLY with a single letter (A, B, C, D, or E):"""
        elif task_type == "SP":
            return f"""Explain the hidden story.

Situation: {state.get('surface', '')}

Clues: {state.get('history_string', '')}

Explanation:"""
        else:  # GN
            history = state.get('history_string', '')
            return f"""Let's play a game of guessing number.
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once.

Previous guesses and feedback:
{history}

You have finished all the rounds of interaction, please give your final answer based on the guesses and feedback above:
Guess: [number]"""


def compute_ece_only(
    llm: LLMWrapper,
    clusterer: SemanticClusterer,
    task_type: str,
    state: Dict[str, Any],
    ground_truth: Optional[Any] = None,
    perturbation_config: Optional[PerturbationConfig] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute ECE without MI (ECE-only baseline).
    
    Args:
        llm: LLM wrapper
        clusterer: Semantic clusterer
        task_type: Task type
        state: Current state
        ground_truth: Optional ground truth
        perturbation_config: Perturbation configuration
        n_bins: Number of ECE bins
    
    Returns:
        ECE details dict
    """
    if perturbation_config is None:
        perturbation_config = PerturbationConfig(
            n_samples=8,
            noise_scale=0.1,
            temperature_range=(0.3, 1.2),
        )
    
    sampler = PerturbedEnsembleSampler(
        llm=llm,
        clusterer=clusterer,
        config=perturbation_config,
    )
    
    # Build prompt
    estimator_temp = RobustMIWithECE(
        llm=llm,
        clusterer=clusterer,
        use_ece=False,
    )
    prompt = estimator_temp._get_answer_prompt(task_type, state)
    messages = [{"role": "user", "content": prompt}]
    
    # Generate perturbed samples
    samples = sampler.sample_perturbed(
        messages=messages,
        max_tokens=256,
        parallel=True,
    )
    
    # Compute ECE
    ece_details = compute_ensemble_ece(
        samples=samples,
        clusterer=clusterer,
        task_type=task_type,
        ground_truth=ground_truth,
        n_bins=n_bins,
    )
    
    return ece_details
