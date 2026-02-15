"""
Perturbation infrastructure for RRMC.

Simulates weight perturbation through output-space ensemble perturbation
using temperature variation, prompt perturbation, and embedding noise.
"""

import re
import numpy as np
from typing import List, Dict, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

from .llm import LLMWrapper, LLMResponse
from .clustering import SemanticClusterer


@dataclass
class PerturbedSample:
    """A single perturbed sample from the ensemble."""
    content: str
    raw_content: str
    temperature: float
    prompt_variant: int
    embedding_noise_scale: float
    prompt_tokens: int = 0
    completion_tokens: int = 0


@dataclass
class PerturbationConfig:
    """Configuration for perturbation sampling."""
    n_samples: int = 8
    noise_type: str = "gaussian"  # gaussian or laplace
    noise_scale: float = 0.1
    temperature_range: Tuple[float, float] = (0.3, 1.2)
    use_prompt_perturbation: bool = True
    use_embedding_noise: bool = True
    seed: int = 42


class NoiseStrategy(ABC):
    """Abstract base class for noise injection strategies."""
    
    @abstractmethod
    def generate_noise(self, size: int, scale: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate noise array."""
        pass
    
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass


class GaussianNoiseStrategy(NoiseStrategy):
    """Gaussian (normal) noise injection."""
    
    def generate_noise(self, size: int, scale: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate Gaussian noise N(0, scale^2)."""
        rng = np.random.RandomState(seed)
        return rng.normal(0, scale, size=size)
    
    def name(self) -> str:
        return "gaussian"


class LaplaceNoiseStrategy(NoiseStrategy):
    """Laplace (double exponential) noise injection."""
    
    def generate_noise(self, size: int, scale: float, seed: Optional[int] = None) -> np.ndarray:
        """Generate Laplace noise with scale parameter."""
        rng = np.random.RandomState(seed)
        return rng.laplace(0, scale, size=size)
    
    def name(self) -> str:
        return "laplace"


class PerturbedEnsembleSampler:
    """
    Perturbed ensemble sampler that simulates weight perturbation
    through output-space diversity.
    
    Combines:
    1. Temperature variation (simulates logit noise)
    2. Prompt perturbation (deterministic seed-based)
    3. Post-hoc embedding noise (on answer representations)
    """
    
    # Prompt perturbation templates
    PROMPT_VARIANTS = [
        "",  # No change (base)
        "Think carefully and ",
        "After careful consideration, ",
        "Upon reflection, ",
        "Considering all factors, ",
    ]
    
    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        config: Optional[PerturbationConfig] = None,
    ):
        """
        Initialize perturbed ensemble sampler.
        
        Args:
            llm: LLM wrapper for generation
            clusterer: Semantic clusterer for grouping
            config: Perturbation configuration
        """
        self.llm = llm
        self.clusterer = clusterer
        self.config = config or PerturbationConfig()
        
        # Initialize noise strategy
        if self.config.noise_type == "gaussian":
            self.noise_strategy = GaussianNoiseStrategy()
        elif self.config.noise_type == "laplace":
            self.noise_strategy = LaplaceNoiseStrategy()
        else:
            raise ValueError(f"Unknown noise type: {self.config.noise_type}")
        
        # Generate deterministic temperature schedule
        self.temperatures = self._generate_temperatures()
    
    def _generate_temperatures(self) -> List[float]:
        """Generate evenly spaced temperatures for diversity."""
        lo, hi = self.config.temperature_range
        if self.config.n_samples == 1:
            return [(lo + hi) / 2]
        step = (hi - lo) / (self.config.n_samples - 1)
        return [lo + i * step for i in range(self.config.n_samples)]
    
    def _perturb_prompt(self, messages: List[Dict[str, str]], variant_idx: int) -> List[Dict[str, str]]:
        """
        Apply prompt perturbation to introduce output variance.
        
        Args:
            messages: Original message list
            variant_idx: Which variant to use (cycled)
        
        Returns:
            Perturbed messages
        """
        if not self.config.use_prompt_perturbation or not messages:
            return messages
        
        # Get variant prefix
        variant = self.PROMPT_VARIANTS[variant_idx % len(self.PROMPT_VARIANTS)]
        if not variant:
            return messages
        
        # Apply to last user message
        perturbed = messages.copy()
        for i in range(len(perturbed) - 1, -1, -1):
            if perturbed[i]["role"] == "user":
                perturbed[i] = {
                    "role": "user",
                    "content": variant + perturbed[i]["content"]
                }
                break
        
        return perturbed
    
    def sample_perturbed(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int = 512,
        parallel: bool = True,
    ) -> List[PerturbedSample]:
        """
        Generate N perturbed samples using temperature + prompt variation.
        
        Args:
            messages: Input messages
            max_tokens: Maximum tokens per sample
            parallel: Whether to run in parallel
        
        Returns:
            List of PerturbedSample objects
        """
        n = self.config.n_samples
        
        if not parallel or n <= 1:
            return self._sample_sequential(messages, max_tokens)
        else:
            return self._sample_parallel(messages, max_tokens)
    
    def _sample_sequential(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[PerturbedSample]:
        """Sequential sampling (for debugging or small n)."""
        samples = []
        for i in range(self.config.n_samples):
            temp = self.temperatures[i]
            perturbed_msgs = self._perturb_prompt(messages, i)
            
            response = self.llm.generate(
                messages=perturbed_msgs,
                temperature=temp,
                max_tokens=max_tokens,
            )
            
            samples.append(PerturbedSample(
                content=response.content,
                raw_content=response.raw_content,
                temperature=temp,
                prompt_variant=i % len(self.PROMPT_VARIANTS),
                embedding_noise_scale=self.config.noise_scale,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            ))
        
        return samples
    
    def _sample_parallel(
        self,
        messages: List[Dict[str, str]],
        max_tokens: int,
    ) -> List[PerturbedSample]:
        """Parallel sampling for efficiency."""
        def generate_one(i: int) -> PerturbedSample:
            temp = self.temperatures[i]
            perturbed_msgs = self._perturb_prompt(messages, i)
            
            response = self.llm.generate(
                messages=perturbed_msgs,
                temperature=temp,
                max_tokens=max_tokens,
            )
            
            return PerturbedSample(
                content=response.content,
                raw_content=response.raw_content,
                temperature=temp,
                prompt_variant=i % len(self.PROMPT_VARIANTS),
                embedding_noise_scale=self.config.noise_scale,
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
            )
        
        samples = [None] * self.config.n_samples
        max_workers = getattr(self.llm, 'max_workers', 8)
        
        with ThreadPoolExecutor(max_workers=min(self.config.n_samples, max_workers)) as executor:
            futures = {executor.submit(generate_one, i): i for i in range(self.config.n_samples)}
            for future in as_completed(futures):
                idx = futures[future]
                try:
                    samples[idx] = future.result()
                except Exception as e:
                    print(f"Perturbed sample {idx} failed: {e}")
                    # Create dummy sample
                    samples[idx] = PerturbedSample(
                        content="",
                        raw_content="",
                        temperature=self.temperatures[idx],
                        prompt_variant=idx % len(self.PROMPT_VARIANTS),
                        embedding_noise_scale=self.config.noise_scale,
                    )
        
        return samples
    
    def apply_embedding_noise(
        self,
        embeddings: np.ndarray,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        Apply noise to embeddings (post-hoc perturbation).
        
        Args:
            embeddings: Array of shape (n_samples, embedding_dim)
            seed: Random seed
        
        Returns:
            Noised embeddings
        """
        if not self.config.use_embedding_noise:
            return embeddings
        
        noise = self.noise_strategy.generate_noise(
            size=embeddings.shape,
            scale=self.config.noise_scale,
            seed=seed or self.config.seed,
        )
        
        return embeddings + noise


def compute_ensemble_ece(
    samples: List[PerturbedSample],
    clusterer: SemanticClusterer,
    task_type: str,
    ground_truth: Optional[Any] = None,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute ECE for a perturbed ensemble.
    
    For each sample, we derive a "confidence" from the cluster frequency
    and compute accuracy (if ground truth is available).
    
    Args:
        samples: List of perturbed samples
        clusterer: Semantic clusterer
        task_type: Task type (DC, SP, GN)
        ground_truth: Optional ground truth for accuracy
        n_bins: Number of ECE bins
    
    Returns:
        Dict with ece, bin_accuracies, bin_confidences, bin_counts
    """
    if not samples:
        return {
            "ece": 0.0,
            "n_bins": n_bins,
            "bin_accuracies": [],
            "bin_confidences": [],
            "bin_counts": [],
            "n_samples": 0,
        }
    
    # Extract contents and cluster
    contents = [s.content for s in samples]
    cluster_result = clusterer.cluster(contents, task_type)
    
    # Compute confidence for each sample as its cluster frequency
    cluster_ids = cluster_result.cluster_ids
    cluster_sizes = cluster_result.cluster_sizes
    total = len(samples)
    
    confidences = np.array([cluster_sizes[cid] / total for cid in cluster_ids])
    
    # Compute accuracy if ground truth available
    if ground_truth is not None:
        accuracies = np.array([
            int(_check_correctness(s.content, ground_truth, task_type, clusterer))
            for s in samples
        ])
    else:
        # No ground truth: assume uniform accuracy for ECE structure
        accuracies = np.ones(len(samples))
    
    # Compute ECE over bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_accs = []
    bin_confs = []
    bin_counts = []
    ece = 0.0
    
    for i in range(n_bins):
        mask = (confidences >= bin_boundaries[i]) & (confidences < bin_boundaries[i + 1])
        if i == n_bins - 1:
            mask = mask | (confidences == bin_boundaries[i + 1])
        
        count = mask.sum()
        bin_counts.append(int(count))
        
        if count > 0:
            avg_acc = accuracies[mask].mean()
            avg_conf = confidences[mask].mean()
            bin_accs.append(float(avg_acc))
            bin_confs.append(float(avg_conf))
            ece += count * abs(avg_acc - avg_conf)
        else:
            bin_accs.append(0.0)
            bin_confs.append(0.0)
    
    ece /= total
    
    return {
        "ece": float(ece),
        "n_bins": n_bins,
        "bin_accuracies": bin_accs,
        "bin_confidences": bin_confs,
        "bin_counts": bin_counts,
        "n_samples": total,
        "cluster_entropy": clusterer.compute_entropy(cluster_result),
    }


def _check_correctness(
    prediction: str,
    ground_truth: Any,
    task_type: str,
    clusterer: SemanticClusterer,
) -> bool:
    """
    Check if prediction is correct.
    
    Args:
        prediction: Predicted answer string
        ground_truth: Ground truth answer
        task_type: Task type
        clusterer: Clusterer for parsing
    
    Returns:
        True if correct, False otherwise
    """
    if task_type == "DC":
        # Parse suspect index
        pred_idx = clusterer._extract_dc_answer(prediction)
        gt_idx = int(ground_truth) if not isinstance(ground_truth, int) else ground_truth
        return pred_idx == gt_idx
    elif task_type == "GN":
        # Parse 4-digit number
        pred_num = clusterer._extract_gn_answer(prediction)
        gt_num = re.sub(r'[^0-9]', '', str(ground_truth))[:4]
        return pred_num == gt_num
    else:  # SP
        # Use char-level F1 threshold
        from .calibration import _char_f1
        f1 = _char_f1(prediction.lower(), str(ground_truth).lower())
        return f1 > 0.5


def compute_homogeneity_score(
    samples: List[PerturbedSample],
    clusterer: SemanticClusterer,
    task_type: str,
) -> float:
    """
    Compute homogeneity score for perturbed ensemble.
    
    High score (near 1.0) indicates mode collapse.
    
    Args:
        samples: Perturbed samples
        clusterer: Semantic clusterer
        task_type: Task type
    
    Returns:
        Homogeneity score in [0, 1]
    """
    if not samples:
        return 1.0
    
    contents = [s.content for s in samples]
    cluster_result = clusterer.cluster(contents, task_type)
    
    if cluster_result.n_clusters == 0:
        return 1.0
    
    max_size = max(cluster_result.cluster_sizes.values())
    return max_size / len(samples)
