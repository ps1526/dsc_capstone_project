"""
ECE-Based Question Selector for RRMC.

Extends VoI (Value of Information) approach to score questions
by expected ECE and MI reduction.
"""

import re
import json
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..core.llm import LLMWrapper
from ..core.clustering import SemanticClusterer
from ..core.mi_estimator_ece import RobustMIWithECE, compute_ece_only
from ..core.perturbation import PerturbationConfig


@dataclass
class CandidateQuestionECE:
    """A candidate question with ECE-based scoring."""
    suspect: Optional[str]
    question: str
    expected_mi_reduction: float = 0.0
    expected_ece_reduction: float = 0.0
    combined_score: float = 0.0
    simulated_mis: List[float] = field(default_factory=list)
    simulated_eces: List[float] = field(default_factory=list)


@dataclass
class QuestionSelectionResultECE:
    """Result of ECE question selection."""
    selected: CandidateQuestionECE
    candidates: List[CandidateQuestionECE]
    current_mi: float
    current_ece: float


# Prompt templates for candidate generation (DC task)
CANDIDATE_GENERATION_DC = """You are a detective investigating a murder case.

Case Background:
{initial_info}

Interrogation History:
{history}

Suspects: {suspect_names}

Generate {m} distinct questions you could ask the suspects to help identify the murderer.
For each question, specify which suspect to ask and why this question is informative.

Output as a JSON list:
[
  {{"suspect": "<name>", "question": "<question text>"}},
  ...
]

Output ONLY the JSON list, nothing else."""

SIMULATE_RESPONSE_DC = """You will play the role of a suspect being interrogated.

Your Name: {suspect_name}
Case Background:
{initial_info}

Interrogation History:
{history}

A detective asks you: "{question}"

Provide a plausible one-sentence response as this suspect would answer.
Response:"""


class ECEQuestionSelector:
    """
    ECE-enhanced question selector.
    
    Scores candidate questions by expected reduction in combined MI+ECE.
    Simulates outcomes and measures expected uncertainty improvement.
    """
    
    def __init__(
        self,
        llm: LLMWrapper,
        clusterer: SemanticClusterer,
        robust_mi_ece: Optional[RobustMIWithECE] = None,
        m_candidates: int = 5,
        r_simulations: int = 3,
        k_samples: int = 4,
        alpha: float = 0.5,  # MI weight
        beta: float = 0.5,   # ECE weight
        score_type: str = "combined",  # mi, ece, or combined
        perturbation_config: Optional[PerturbationConfig] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize ECE question selector.
        
        Args:
            llm: LLM wrapper
            clusterer: Semantic clusterer
            robust_mi_ece: Optional pre-configured MI+ECE estimator
            m_candidates: Number of candidate questions to generate
            r_simulations: Number of simulated outcomes per candidate
            k_samples: Number of samples for MI/ECE estimation
            alpha: Weight for MI in scoring
            beta: Weight for ECE in scoring
            score_type: How to score questions (mi, ece, combined)
            perturbation_config: Perturbation configuration
            temperature: Generation temperature
        """
        self.llm = llm
        self.clusterer = clusterer
        self.m_candidates = m_candidates
        self.r_simulations = r_simulations
        self.k_samples = k_samples
        self.alpha = alpha
        self.beta = beta
        self.score_type = score_type
        self.temperature = temperature
        
        # Initialize MI+ECE estimator
        if robust_mi_ece is None:
            self.robust_mi_ece = RobustMIWithECE(
                llm=llm,
                clusterer=clusterer,
                k_samples=k_samples,
                alpha=alpha,
                beta=beta,
                perturbation_config=perturbation_config,
            )
        else:
            self.robust_mi_ece = robust_mi_ece
    
    def select_question(
        self,
        task_type: str,
        state: Dict[str, Any],
        current_mi: float = 0.0,
        current_ece: float = 0.0,
    ) -> QuestionSelectionResultECE:
        """
        Select the best question using ECE-based scoring.
        
        Args:
            task_type: Task type (DC, SP, GN)
            state: Current environment state
            current_mi: Current MI value
            current_ece: Current ECE value
        
        Returns:
            QuestionSelectionResultECE with selected question and candidates
        """
        if task_type == "DC":
            return self._select_dc_question(state, current_mi, current_ece)
        else:
            raise NotImplementedError(f"ECE question selection not implemented for {task_type}")
    
    def _select_dc_question(
        self,
        state: Dict[str, Any],
        current_mi: float,
        current_ece: float,
    ) -> QuestionSelectionResultECE:
        """Select question for DC task."""
        # Step 1: Generate candidate questions
        candidates = self._generate_candidate_questions_dc(state)
        
        if not candidates:
            # Fallback: create a default question
            suspects = state.get("suspect_names", [])
            suspect = suspects[0] if suspects else "the suspect"
            candidates = [CandidateQuestionECE(
                suspect=suspect,
                question="Can you tell me about your whereabouts?",
            )]
        
        # Step 2: Score each candidate
        for candidate in candidates:
            self._score_candidate_dc(candidate, state, current_mi, current_ece)
        
        # Step 3: Select best candidate
        if self.score_type == "mi":
            best = max(candidates, key=lambda c: c.expected_mi_reduction)
        elif self.score_type == "ece":
            best = max(candidates, key=lambda c: c.expected_ece_reduction)
        else:  # combined
            best = max(candidates, key=lambda c: c.combined_score)
        
        return QuestionSelectionResultECE(
            selected=best,
            candidates=candidates,
            current_mi=current_mi,
            current_ece=current_ece,
        )
    
    def _generate_candidate_questions_dc(
        self,
        state: Dict[str, Any],
    ) -> List[CandidateQuestionECE]:
        """Generate M candidate questions for DC task."""
        initial_info = state.get("initial_info", "")
        history = state.get("history_string", "")
        suspect_names = state.get("suspect_names", [])
        
        prompt = CANDIDATE_GENERATION_DC.format(
            initial_info=initial_info,
            history=history,
            suspect_names=", ".join(suspect_names),
            m=self.m_candidates,
        )
        
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=512,
        )
        
        # Parse JSON response
        candidates = self._parse_candidate_questions(response.content, suspect_names)
        return candidates[:self.m_candidates]
    
    def _parse_candidate_questions(
        self,
        response: str,
        suspect_names: List[str],
    ) -> List[CandidateQuestionECE]:
        """Parse candidate questions from JSON response."""
        candidates = []
        
        # Try to extract JSON
        try:
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                for item in data:
                    if isinstance(item, dict) and "question" in item:
                        suspect = item.get("suspect", "")
                        # Validate suspect name
                        if suspect not in suspect_names and suspect_names:
                            suspect = suspect_names[0]
                        candidates.append(CandidateQuestionECE(
                            suspect=suspect,
                            question=item["question"],
                        ))
        except (json.JSONDecodeError, KeyError):
            pass
        
        # Fallback: extract questions from lines
        if not candidates:
            lines = response.strip().split("\n")
            for line in lines:
                if "?" in line:
                    question = line.strip()
                    # Remove numbering
                    question = re.sub(r'^\d+[\.\)]\s*', '', question)
                    if question:
                        candidates.append(CandidateQuestionECE(
                            suspect=suspect_names[0] if suspect_names else None,
                            question=question,
                        ))
        
        return candidates
    
    def _score_candidate_dc(
        self,
        candidate: CandidateQuestionECE,
        state: Dict[str, Any],
        current_mi: float,
        current_ece: float,
    ):
        """
        Score a candidate question by simulating outcomes.
        
        For each of R simulations:
        1. Simulate NPC response
        2. Compute MI and ECE in that hypothetical future
        3. Measure reduction from current values
        """
        simulated_mis = []
        simulated_eces = []
        
        for _ in range(self.r_simulations):
            # Simulate NPC response
            simulated_answer = self._simulate_npc_response_dc(
                state=state,
                suspect=candidate.suspect,
                question=candidate.question,
            )
            
            # Create hypothetical future state
            future_state = self._create_future_state_dc(
                state=state,
                question=candidate.question,
                answer=simulated_answer,
                suspect=candidate.suspect,
            )
            
            # Estimate MI and ECE in future state
            estimate = self.robust_mi_ece.estimate(
                task_type="DC",
                state=future_state,
            )
            
            simulated_mis.append(estimate.mi)
            simulated_eces.append(estimate.ece)
        
        # Compute expected reductions
        avg_mi = sum(simulated_mis) / len(simulated_mis) if simulated_mis else current_mi
        avg_ece = sum(simulated_eces) / len(simulated_eces) if simulated_eces else current_ece
        
        mi_reduction = max(0, current_mi - avg_mi)
        ece_reduction = max(0, current_ece - avg_ece)
        
        # Combined score
        combined = self.alpha * mi_reduction + self.beta * ece_reduction
        
        candidate.expected_mi_reduction = mi_reduction
        candidate.expected_ece_reduction = ece_reduction
        candidate.combined_score = combined
        candidate.simulated_mis = simulated_mis
        candidate.simulated_eces = simulated_eces
    
    def _simulate_npc_response_dc(
        self,
        state: Dict[str, Any],
        suspect: Optional[str],
        question: str,
    ) -> str:
        """Simulate NPC response to a question."""
        initial_info = state.get("initial_info", "")
        history = state.get("history_string", "")
        
        prompt = SIMULATE_RESPONSE_DC.format(
            suspect_name=suspect or "Unknown",
            initial_info=initial_info,
            history=history,
            question=question,
        )
        
        response = self.llm.generate(
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=128,
        )
        
        return response.content.strip()
    
    def _create_future_state_dc(
        self,
        state: Dict[str, Any],
        question: str,
        answer: str,
        suspect: Optional[str],
    ) -> Dict[str, Any]:
        """Create hypothetical future state after asking a question."""
        future_state = state.copy()
        
        # Add to history
        history = future_state.get("history", []).copy()
        history.append({
            "question": question,
            "answer": answer,
            "suspect": suspect,
        })
        future_state["history"] = history
        
        # Update history string
        history_lines = []
        for h in history:
            if "suspect" in h:
                history_lines.append(
                    f"Q to {h['suspect']}: {h['question']} A: {h['answer']}"
                )
            else:
                history_lines.append(f"Q: {h['question']} A: {h['answer']}")
        future_state["history_string"] = "\n".join(history_lines)
        
        return future_state
