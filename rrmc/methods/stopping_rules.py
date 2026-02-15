"""
Baseline Stopping Rules for RRMC Comparison.

Implements various uncertainty-based stopping rules:
- Fixed turns (naive)
- Self-consistency
- Verbalized confidence
- Semantic entropy
- MI-only (without robustness)
- Robust MI with risk-controlled threshold
"""

import re
import json
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np

from ..core.llm import LLMWrapper
from ..core.clustering import SemanticClusterer
from ..core.mi_estimator import SelfRevisionMI, RobustMI, MIEstimate
from ..core.mi_estimator_ece import RobustMIWithECE, CombinedUncertaintyEstimate
from ..core.calibration import StoppingController


@dataclass
class StoppingDecision:
    """Result of stopping decision."""
    should_stop: bool
    reason: str
    score: float  # Uncertainty score
    prediction: Optional[str] = None  # Predicted answer if stopping
    metadata: Optional[Dict[str, Any]] = None  # Extra info (e.g. remaining suspects)


class BaseStoppingRule(ABC):
    """Base class for stopping rules."""

    def __init__(self, max_turns: int = 25):
        self.max_turns = max_turns

    @abstractmethod
    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        """
        Decide whether to stop and answer.

        Args:
            task_type: Task type (DC, SP, GN)
            state: Current environment state
            turn: Current turn number

        Returns:
            StoppingDecision with decision and score
        """
        pass

    @abstractmethod
    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        """Get the best answer given current state."""
        pass


class FixedTurnsStopping(BaseStoppingRule):
    """
    Naive baseline: Ask for fixed number of turns, then answer.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        fixed_turns: int = 10,
        max_turns: int = 25,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.fixed_turns = fixed_turns
        self._last_raw_answer: Optional[str] = None

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        should_stop = turn >= self.fixed_turns or turn >= self.max_turns
        return StoppingDecision(
            should_stop=should_stop,
            reason="fixed_turns" if turn >= self.fixed_turns else "max_turns",
            score=float(turn),
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        prompt = self._get_answer_prompt(task_type, state)
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )
        self._last_raw_answer = response.raw_content or response.content
        return self._parse_answer(response.content, task_type)

    def _get_answer_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
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
            # Use AR-Bench's format - game context + final_guess_prompt
            history = state.get('history_string', '')
            return f"""Let's play a game of guessing number.
The game rule is: I have a 4-digit secret number in mind, all digits of the number is unique such that all digits from 0 to 9 can only be present once.

Previous guesses and feedback:
{history}

You have finished all the rounds of interaction, please give your final answer based on the guesses and feedback above:
Guess: [number]"""

    def _parse_answer(self, response: str, task_type: str) -> str:
        if task_type == "DC":
            match = re.search(r'\b([A-E])\b', response.upper())
            if match:
                return match.group(1)
            # Preserve raw response so env can match suspect names/digits
            return response.strip()
        elif task_type == "GN":
            # Look for explicit "Guess: XXXX" or "Final Answer: XXXX" pattern first
            explicit_patterns = [
                r'(?:final\s+)?(?:answer|guess)\s*[:\s]+(\d{4})',
                r'(?:my\s+)?(?:final\s+)?guess\s+(?:is|:)\s*(\d{4})',
            ]
            for pattern in explicit_patterns:
                matches = re.findall(pattern, response.lower())
                if matches:
                    # Take the last explicit match
                    candidate = matches[-1]
                    if len(set(candidate)) == 4:
                        return candidate
            
            # Extract all 4-digit numbers and return the LAST one with unique digits
            # (model analyzes previous guesses first, then states final answer)
            all_digits = re.findall(r'\d{4}', response)
            valid_candidates = [c for c in all_digits if len(set(c)) == 4]
            if valid_candidates:
                return valid_candidates[-1]  # Return LAST valid one
            
            # Fallback: extract any 4 digits and try to make unique
            digits = re.sub(r'[^0-9]', '', response)
            if len(digits) >= 4:
                # Try to find 4 unique digits from extracted
                unique = []
                for d in digits:
                    if d not in unique:
                        unique.append(d)
                    if len(unique) == 4:
                        return ''.join(unique)
            # Last resort: random valid guess
            import random
            available = [str(i) for i in range(10)]
            random.shuffle(available)
            return ''.join(available[:4])
        return response.strip()[:500]


class SelfConsistencyStopping(BaseStoppingRule):
    """
    Self-consistency stopping rule.

    Answer when majority cluster frequency >= threshold.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 10,
        consistency_threshold: float = 0.8,
        max_turns: int = 25,
        temperature: float = 0.7,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.consistency_threshold = consistency_threshold
        self.temperature = temperature
        self._last_answers: List[str] = []
        self._last_raw_samples: List[str] = []

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=0.0,
                prediction=self.get_best_answer(task_type, state),
            )

        # Sample k answers (parallel)
        prompt = self._get_answer_prompt(task_type, state)
        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.k_samples,
            temperature=self.temperature,
            max_tokens=256,
            parallel=True,
        )
        self._last_raw_samples = [r.raw_content or r.content for r in responses]
        answers = [self._parse_answer(r.content, task_type) for r in responses]

        self._last_answers = answers

        # Cluster answers
        cluster_result = self.clusterer.cluster(answers, task_type)

        # Compute max cluster frequency
        max_freq = max(cluster_result.cluster_sizes.values()) / len(answers)
        score = 1 - max_freq  # Lower score = more confident

        should_stop = max_freq >= self.consistency_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="high_consistency" if should_stop else "low_consistency",
            score=score,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        if self._last_answers:
            cluster_result = self.clusterer.cluster(self._last_answers, task_type)
            max_cluster = max(
                cluster_result.cluster_sizes,
                key=lambda k: cluster_result.cluster_sizes[k]
            )
            return cluster_result.representative_texts[max_cluster]

        # Fallback
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)

    def _get_answer_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
        return FixedTurnsStopping(self.llm)._get_answer_prompt(task_type, state)

    def _parse_answer(self, response: str, task_type: str) -> str:
        return FixedTurnsStopping(self.llm)._parse_answer(response, task_type)


class VerbalizedConfidenceStopping(BaseStoppingRule):
    """
    Verbalized confidence stopping rule.

    Ask model to rate confidence 1-10, stop if >= threshold.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        confidence_threshold: float = 8.0,
        max_turns: int = 25,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.confidence_threshold = confidence_threshold
        self._last_answer: str = ""

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=0.0,
                prediction=self.get_best_answer(task_type, state),
            )

        prompt = self._get_confidence_prompt(task_type, state)
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=256,
        )

        # Parse confidence and answer
        confidence, answer = self._parse_confidence_response(response.content, task_type)
        self._last_answer = answer

        score = 10 - confidence  # Lower score = more confident
        should_stop = confidence >= self.confidence_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="high_confidence" if should_stop else "low_confidence",
            score=score,
            prediction=answer if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        if self._last_answer:
            return self._last_answer
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)

    def _get_confidence_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
        base_prompt = FixedTurnsStopping(self.llm)._get_answer_prompt(task_type, state)
        return f"""{base_prompt}

Also rate your confidence from 1-10 (10 = completely certain).

Format your response as:
Answer: [your answer]
Confidence: [1-10]"""

    def _parse_confidence_response(
        self,
        response: str,
        task_type: str,
    ) -> Tuple[float, str]:
        # Extract confidence
        conf_match = re.search(r'Confidence[:\s]+(\d+)', response, re.I)
        confidence = float(conf_match.group(1)) if conf_match else 5.0

        # Extract answer
        answer = FixedTurnsStopping(self.llm)._parse_answer(response, task_type)

        return confidence, answer


class SemanticEntropyStopping(BaseStoppingRule):
    """
    Semantic entropy stopping rule.

    Stop when entropy over answer clusters <= threshold.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 10,
        entropy_threshold: float = 0.5,
        max_turns: int = 25,
        temperature: float = 0.7,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.entropy_threshold = entropy_threshold
        self.temperature = temperature
        self._last_answers: List[str] = []

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        # Sample k answers (parallel)
        prompt = self._get_answer_prompt(task_type, state)
        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.k_samples,
            temperature=self.temperature,
            max_tokens=256,
            parallel=True,
        )
        answers = [self._parse_answer(r.content, task_type) for r in responses]

        self._last_answers = answers

        # Cluster and compute entropy
        cluster_result = self.clusterer.cluster(answers, task_type)
        entropy = self.clusterer.compute_entropy(cluster_result)

        should_stop = entropy <= self.entropy_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="low_entropy" if should_stop else "high_entropy",
            score=entropy,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        if self._last_answers:
            cluster_result = self.clusterer.cluster(self._last_answers, task_type)
            max_cluster = max(
                cluster_result.cluster_sizes,
                key=lambda k: cluster_result.cluster_sizes[k]
            )
            return cluster_result.representative_texts[max_cluster]
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)

    def _get_answer_prompt(self, task_type: str, state: Dict[str, Any]) -> str:
        return FixedTurnsStopping(self.llm)._get_answer_prompt(task_type, state)

    def _parse_answer(self, response: str, task_type: str) -> str:
        return FixedTurnsStopping(self.llm)._parse_answer(response, task_type)


class MIOnlyStopping(BaseStoppingRule):
    """
    MI-only stopping rule (without robustness).

    Uses single-variant self-revision MI with fixed threshold.
    Supports min_turns to prevent premature stopping before
    enough evidence has been gathered.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        mi_threshold: float = 0.3,
        k_samples: int = 6,
        max_turns: int = 25,
        temperature: float = 0.7,
        min_turns: int = 0,
    ):
        super().__init__(max_turns)
        self.mi_estimator = SelfRevisionMI(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            temperature=temperature,
        )
        self.mi_threshold = mi_threshold
        self.min_turns = min_turns
        self._last_estimate: Optional[MIEstimate] = None

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        # Estimate MI (single variant) — always compute for diagnostics
        estimate = self.mi_estimator.estimate(
            task_type=task_type,
            state=state,
            revision_variant="base",
        )
        self._last_estimate = estimate

        # Before min_turns, continue investigating regardless of MI
        if turn < self.min_turns:
            return StoppingDecision(
                should_stop=False,
                reason="below_min_turns",
                score=estimate.mi,
                prediction=None,
            )

        should_stop = estimate.mi <= self.mi_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="low_mi" if should_stop else "high_mi",
            score=estimate.mi,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        if self._last_estimate and self._last_estimate.revised_answers:
            # Return majority from revised answers
            answers = self._last_estimate.revised_answers
            cluster_result = self.mi_estimator.clusterer.cluster(answers, task_type)
            max_cluster = max(
                cluster_result.cluster_sizes,
                key=lambda k: cluster_result.cluster_sizes[k]
            )
            return cluster_result.representative_texts[max_cluster]
        return ""


class RobustMIStopping(BaseStoppingRule):
    """
    Robust MI stopping rule with risk-controlled threshold.

    Uses multiple prompt variants and calibrated threshold.
    Supports min_turns to prevent premature stopping before
    enough evidence has been gathered.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        threshold: float = 0.3,
        k_samples: int = 6,
        max_turns: int = 25,
        temperature: float = 0.7,
        variants: Optional[List[str]] = None,
        use_diversity_sampling: bool = True,
        regime: str = "normal",
        min_turns: int = 0,
    ):
        super().__init__(max_turns)
        self.robust_mi = RobustMI(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            temperature=temperature,
            use_diversity_sampling=use_diversity_sampling,
            regime=regime,
        )
        self.threshold = threshold
        self.variants = variants
        self.regime = regime
        self.min_turns = min_turns
        self._last_mi: float = 0.0
        self._last_estimates: Dict[str, MIEstimate] = {}

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        # Estimate robust MI — always compute for diagnostics
        robust_mi, estimates = self.robust_mi.estimate(
            task_type=task_type,
            state=state,
            variants=self.variants,
        )
        self._last_mi = robust_mi
        self._last_estimates = estimates

        # Before min_turns, continue investigating regardless of MI
        if turn < self.min_turns:
            return StoppingDecision(
                should_stop=False,
                reason="below_min_turns",
                score=robust_mi,
                prediction=None,
            )

        should_stop = robust_mi <= self.threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="low_robust_mi" if should_stop else "high_robust_mi",
            score=robust_mi,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(
        self,
        task_type: str,
        state: Dict[str, Any],
    ) -> str:
        # Reuse cached estimates from the last MI call when available
        if self._last_estimates:
            all_revised = []
            for est in self._last_estimates.values():
                all_revised.extend(est.revised_answers)
            if all_revised:
                cluster_result = self.robust_mi.mi_estimator.clusterer.cluster(all_revised, task_type)
                if cluster_result.cluster_sizes:
                    max_cluster = max(
                        cluster_result.cluster_sizes,
                        key=lambda k: cluster_result.cluster_sizes[k]
                    )
                    return cluster_result.representative_texts.get(max_cluster, all_revised[0])
        return self.robust_mi.get_best_answer(task_type, state)

    def update_threshold(self, new_threshold: float):
        """Update the stopping threshold (after calibration)."""
        self.threshold = new_threshold


class KnowNoStopping(BaseStoppingRule):
    """
    KnowNo-style set-size gate stopping rule.

    Sample k answers, cluster them, and stop when the number of
    distinct clusters (the "prediction set size") is <= set_size_threshold.
    Inspired by Ren et al. (2023) "Robots That Ask For Help".
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 10,
        set_size_threshold: int = 1,
        max_turns: int = 25,
        temperature: float = 0.7,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.set_size_threshold = set_size_threshold
        self.temperature = temperature
        self._last_answers: List[str] = []

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        prompt = FixedTurnsStopping(self.llm)._get_answer_prompt(task_type, state)
        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.k_samples,
            temperature=self.temperature,
            max_tokens=256,
            parallel=True,
        )
        answers = [FixedTurnsStopping(self.llm)._parse_answer(r.content, task_type) for r in responses]
        self._last_answers = answers

        cluster_result = self.clusterer.cluster(answers, task_type)
        set_size = cluster_result.n_clusters

        should_stop = set_size <= self.set_size_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="small_set" if should_stop else "large_set",
            score=float(set_size),
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        if self._last_answers:
            cluster_result = self.clusterer.cluster(self._last_answers, task_type)
            max_cluster = max(
                cluster_result.cluster_sizes,
                key=lambda k: cluster_result.cluster_sizes[k],
            )
            return cluster_result.representative_texts[max_cluster]
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)


class CIPLiteStopping(BaseStoppingRule):
    """
    C-IP-lite: Conformal set-size stopping rule.

    Like KnowNo, but the set-size threshold is calibrated on a held-out
    set via conformal prediction. During evaluation the calibrated
    threshold is used directly.

    At runtime:
    - Sample k answers and cluster.
    - Compute prediction set size.
    - Stop if set_size <= calibrated_threshold.

    The threshold can be set via `update_threshold()` after calibration.
    """

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 10,
        set_size_threshold: int = 2,
        max_turns: int = 25,
        temperature: float = 0.7,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.set_size_threshold = set_size_threshold
        self.temperature = temperature
        self._last_answers: List[str] = []

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        prompt = FixedTurnsStopping(self.llm)._get_answer_prompt(task_type, state)
        responses = self.llm.sample_n(
            messages=[{"role": "user", "content": prompt}],
            n=self.k_samples,
            temperature=self.temperature,
            max_tokens=256,
            parallel=True,
        )
        answers = [FixedTurnsStopping(self.llm)._parse_answer(r.content, task_type) for r in responses]
        self._last_answers = answers

        cluster_result = self.clusterer.cluster(answers, task_type)
        set_size = cluster_result.n_clusters

        should_stop = set_size <= self.set_size_threshold

        return StoppingDecision(
            should_stop=should_stop,
            reason="conformal_small_set" if should_stop else "conformal_large_set",
            score=float(set_size),
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        if self._last_answers:
            cluster_result = self.clusterer.cluster(self._last_answers, task_type)
            max_cluster = max(
                cluster_result.cluster_sizes,
                key=lambda k: cluster_result.cluster_sizes[k],
            )
            return cluster_result.representative_texts[max_cluster]
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)

    def update_threshold(self, new_threshold: int):
        """Update calibrated set-size threshold."""
        self.set_size_threshold = new_threshold


class UoTLiteStopping(BaseStoppingRule):
    """
    UoT-lite: Uncertainty of Thought stopping rule.

    Simplified version of UoT (Hu et al., 2024):
    1. Generate a set of plausible answers.
    2. Stop if only 1 plausible answer remains.
    3. Otherwise, generate a question designed to split the answer set
       (the question is logged but not actually used for selection in
       this stopping-only implementation).
    """

    UOT_ANSWER_SET_PROMPT = """Given this case/puzzle:
{context}

Previous questions and answers:
{history}

Generate a numbered list of plausible answers. Each answer should be a distinct possibility consistent with the evidence so far.
Format:
1. [answer 1]
2. [answer 2]
...

List only plausible answers:"""

    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        k_samples: int = 6,
        max_turns: int = 25,
        temperature: float = 0.7,
    ):
        super().__init__(max_turns)
        self.llm = llm
        self.clusterer = clusterer
        self.k_samples = k_samples
        self.temperature = temperature
        self._last_answer_set: List[str] = []

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )

        # Generate answer set
        context = self._get_context(task_type, state)
        prompt = self.UOT_ANSWER_SET_PROMPT.format(
            context=context,
            history=state.get("history_string", ""),
        )

        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )

        answer_set = self._parse_answer_set(response.content)
        self._last_answer_set = answer_set

        # Cluster the answer set to find distinct plausible answers
        if len(answer_set) >= 2:
            cluster_result = self.clusterer.cluster(answer_set, task_type)
            n_plausible = cluster_result.n_clusters
        else:
            n_plausible = len(answer_set)

        should_stop = n_plausible <= 1

        return StoppingDecision(
            should_stop=should_stop,
            reason="single_answer" if should_stop else "multiple_answers",
            score=float(n_plausible),
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
        )

    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        if self._last_answer_set:
            # Cluster and return majority
            cluster_result = self.clusterer.cluster(self._last_answer_set, task_type)
            if cluster_result.cluster_sizes:
                max_cluster = max(
                    cluster_result.cluster_sizes,
                    key=lambda k: cluster_result.cluster_sizes[k],
                )
                return cluster_result.representative_texts[max_cluster]
            return self._last_answer_set[0]
        return FixedTurnsStopping(self.llm).get_best_answer(task_type, state)

    def _get_context(self, task_type: str, state: Dict[str, Any]) -> str:
        if task_type == "DC":
            return state.get("initial_info", "")
        elif task_type == "SP":
            return state.get("surface", "")
        else:
            return state.get("rules", "")

    @staticmethod
    def _parse_answer_set(text: str) -> List[str]:
        """Parse numbered list of answers."""
        lines = text.strip().split("\n")
        answers = []
        for line in lines:
            # Match numbered items: "1. answer" or "1) answer"
            match = re.match(r'^\s*\d+[\.\)]\s*(.+)', line)
            if match:
                answer = match.group(1).strip()
                if answer:
                    answers.append(answer)
        return answers if answers else [text.strip()]


class AllSuspectsWrapper(BaseStoppingRule):
    """
    Wrapper that enforces questioning all suspects before allowing stop.

    Wraps any existing stopping rule and overrides should_stop() to
    return False until all suspects have been questioned at least once.
    Only applies to DC (detective_cases) task.
    """

    def __init__(
        self,
        inner_rule: BaseStoppingRule,
        enabled: bool = True,
    ):
        super().__init__(inner_rule.max_turns)
        self.inner_rule = inner_rule
        self.enabled = enabled

    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        # Only apply to DC task when enabled
        if self.enabled and task_type.upper() in ("DC", "DETECTIVE_CASES"):
            suspects = set(state.get("suspect_names", []))
            if suspects:
                questioned = self._get_questioned_suspects(state)
                remaining = suspects - questioned

                if remaining:
                    return StoppingDecision(
                        should_stop=False,
                        reason="not_all_suspects_questioned",
                        score=float(len(remaining)),
                        prediction=None,
                        metadata={"remaining_suspects": sorted(remaining)},
                    )

        # All suspects questioned (or disabled/not DC) — delegate to inner rule
        return self.inner_rule.should_stop(task_type, state, turn)

    @staticmethod
    def _get_questioned_suspects(state: Dict[str, Any]) -> set:
        """Extract suspects that have been questioned from history."""
        history = state.get("history", [])
        return {r.get("suspect") for r in history if r.get("suspect")}

    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        return self.inner_rule.get_best_answer(task_type, state)


class PerturbedMIECEStopping(BaseStoppingRule):
    """
    Stopping rule using perturbed ensemble with combined MI + ECE.
    
    Combines:
    1. Robust MI across prompt variants
    2. ECE from perturbed ensemble
    3. Weighted combination as uncertainty signal
    
    Stop when: combined_score <= threshold
    """
    
    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        threshold: float = 0.3,
        k_samples: int = 8,
        max_turns: int = 25,
        temperature: float = 0.7,
        variants: Optional[List[str]] = None,
        alpha: float = 0.5,  # MI weight
        beta: float = 0.5,   # ECE weight
        perturbation_config: Optional[Any] = None,
        ece_bins: int = 10,
        regime: str = "normal",
    ):
        """
        Initialize perturbed MI+ECE stopping rule.
        
        Args:
            llm: LLM wrapper
            clusterer: Semantic clusterer
            threshold: Stopping threshold for combined score
            k_samples: Number of samples
            max_turns: Maximum turns
            temperature: Base temperature
            variants: Prompt variants for MI
            alpha: Weight for MI in combined score
            beta: Weight for ECE in combined score
            perturbation_config: Perturbation configuration
            ece_bins: Number of ECE bins
            regime: Decoding regime
        """
        super().__init__(max_turns)
        
        self.mi_ece_estimator = RobustMIWithECE(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            temperature=temperature,
            variants=variants,
            regime=regime,
            use_ece=True,
            alpha=alpha,
            beta=beta,
            perturbation_config=perturbation_config,
            ece_bins=ece_bins,
        )
        
        self.threshold = threshold
        self.alpha = alpha
        self.beta = beta
        self._last_estimate: Optional[CombinedUncertaintyEstimate] = None
    
    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )
        
        # Estimate combined MI + ECE
        ground_truth = state.get("ground_truth")  # If available during eval
        estimate = self.mi_ece_estimator.estimate(
            task_type=task_type,
            state=state,
            ground_truth=ground_truth,
        )
        self._last_estimate = estimate
        
        should_stop = estimate.combined_score <= self.threshold
        
        return StoppingDecision(
            should_stop=should_stop,
            reason="low_combined_uncertainty" if should_stop else "high_combined_uncertainty",
            score=estimate.combined_score,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
            metadata={
                "mi": estimate.mi,
                "ece": estimate.ece,
                "alpha": estimate.mi_weight,
                "beta": estimate.ece_weight,
            },
        )
    
    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        """Get best answer using MI+ECE estimator."""
        return self.mi_ece_estimator.get_best_answer(task_type, state)
    
    def update_threshold(self, new_threshold: float):
        """Update the stopping threshold (after calibration)."""
        self.threshold = new_threshold


class ECEOnlyStopping(BaseStoppingRule):
    """
    ECE-only stopping rule (baseline without MI).
    
    Stop when: ECE <= threshold
    """
    
    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        ece_threshold: float = 0.15,
        k_samples: int = 8,
        max_turns: int = 25,
        perturbation_config: Optional[Any] = None,
        ece_bins: int = 10,
    ):
        """
        Initialize ECE-only stopping rule.
        
        Args:
            llm: LLM wrapper
            clusterer: Semantic clusterer
            ece_threshold: Stopping threshold for ECE
            k_samples: Number of samples
            max_turns: Maximum turns
            perturbation_config: Perturbation configuration
            ece_bins: Number of ECE bins
        """
        super().__init__(max_turns)
        
        # Use MI+ECE estimator but with beta=1, alpha=0
        self.mi_ece_estimator = RobustMIWithECE(
            llm=llm,
            clusterer=clusterer,
            k_samples=k_samples,
            use_ece=True,
            alpha=0.0,  # No MI
            beta=1.0,   # ECE only
            perturbation_config=perturbation_config,
            ece_bins=ece_bins,
        )
        
        self.ece_threshold = ece_threshold
        self._last_estimate: Optional[CombinedUncertaintyEstimate] = None
    
    def should_stop(
        self,
        task_type: str,
        state: Dict[str, Any],
        turn: int,
    ) -> StoppingDecision:
        if turn >= self.max_turns:
            return StoppingDecision(
                should_stop=True,
                reason="max_turns",
                score=float('inf'),
                prediction=self.get_best_answer(task_type, state),
            )
        
        # Estimate ECE only
        ground_truth = state.get("ground_truth")
        estimate = self.mi_ece_estimator.estimate(
            task_type=task_type,
            state=state,
            ground_truth=ground_truth,
        )
        self._last_estimate = estimate
        
        should_stop = estimate.ece <= self.ece_threshold
        
        return StoppingDecision(
            should_stop=should_stop,
            reason="low_ece" if should_stop else "high_ece",
            score=estimate.ece,
            prediction=self.get_best_answer(task_type, state) if should_stop else None,
            metadata={"ece": estimate.ece},
        )
    
    def get_best_answer(self, task_type: str, state: Dict[str, Any]) -> str:
        """Get best answer using ECE estimator."""
        return self.mi_ece_estimator.get_best_answer(task_type, state)
