"""
RRMC Pipeline.

Orchestrates calibration, evaluation, and comparison of stopping methods.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

from config import Config, get_method_config
from rrmc.core.llm import LLMWrapper, load_api_key_from_env_file
from rrmc.core.environment import TaskType
from rrmc.core.clustering import SemanticClusterer
from rrmc.core.mi_estimator import RobustMI
from rrmc.evaluation.evaluator import RRMCEvaluator, print_comparison_table
from rrmc.methods import get_method
from rrmc.methods.stopping_rules import AllSuspectsWrapper
from rrmc.methods.trajectory_ensemble import TrajectoryEnsemble
from rrmc.methods.question_selector import VoIQuestionSelector
from rrmc.methods.dqs import DQSQuestionSelector


class Pipeline:
    """
    RRMC experiment pipeline.

    Handles the full workflow:
    1. Load configuration
    2. Initialize LLM and evaluator
    3. Run calibration (if enabled)
    4. Evaluate methods
    5. Generate comparison report
    """

    def __init__(self, config: Config):
        """
        Initialize pipeline with configuration.

        Args:
            config: Config object with experiment settings
        """
        self.config = config
        self.llm = None
        self.evaluator = None
        self.train_evaluator = None
        self.results = {}

    def run(self) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Returns:
            Results dictionary with calibration and evaluation results
        """
        self._init_llm()
        self._init_evaluator()

        # Estimate API calls and set up progress bar
        estimated_calls = self._estimate_api_calls()
        verbose = self.config.get("verbose", True)
        
        self.pbar = tqdm(
            total=estimated_calls,
            desc="API calls",
            unit="call",
            disable=not verbose,
        )
        self.llm.progress_callback = self.pbar.update

        try:
            if self.config.get("calibrate", False):
                return self._run_calibration_pipeline()
            else:
                return self._run_comparison_pipeline()
        finally:
            self.pbar.close()

    def _estimate_api_calls(self) -> int:
        """
        Estimate total API calls for progress bar.
        
        This is an upper bound estimate based on:
        - n_puzzles × max_turns × (1 + k_samples) for sampling methods
        - Additional calls for calibration phase if enabled
        """
        max_turns = self.config.get("max_turns", 25)
        k_samples = self.config.get("k_samples", 4)
        task_str = self.config.get("task", "dc")
        # DC/SP use an NPC LLM call per question; GN is symbolic
        npc_calls = 1 if task_str in ("dc", "sp") else 0
        
        if self.config.get("calibrate", False):
            n_train = self.config.get("n_train", 20)
            n_test = self.config.get("n_test", 10)
            methods = self.config.get("methods", ["fixed_turns", "self_consistency", "mi_only"])
            n_methods = len(methods) + 1  # +1 for calibrated RRMC
            
            # Calibration: n_train puzzles, each turn has question + MI sampling
            # MI sampling: k_samples × n_variants
            n_variants = len(self.config.get("variants", ["base", "skeptical"]))
            calib_calls = n_train * max_turns * (1 + npc_calls + k_samples * n_variants)
            
            # Evaluation: n_test puzzles × n_methods × (calls per method)
            # Rough estimate: each method averages ~(1 + k_samples) calls per turn
            eval_calls = n_test * n_methods * max_turns * (1 + npc_calls + k_samples)
            
            return calib_calls + eval_calls
        else:
            n_puzzles = self.config.get("n_puzzles", 5)
            methods = self.config.get("methods", ["fixed_turns"])
            n_methods = len(methods)
            
            # Each method: n_puzzles × max_turns × (1 + avg_samples)
            # fixed_turns: 1 call/turn, others: 1 + k_samples calls/turn
            avg_samples = k_samples if n_methods > 1 else 0
            return n_puzzles * n_methods * max_turns * (1 + npc_calls + avg_samples)

    def _init_llm(self):
        """Initialize LLM wrapper."""
        verbose = self.config.get("verbose", True)

        # Get API key
        api_key = self.config.get("api_key", "")
        if not api_key or api_key.startswith("${"):
            # Try environment variable
            api_key = os.environ.get("OPENROUTER_API_KEY", "")

        if not api_key:
            # Try env file
            env_file = self.config.get("env_file", "openrouter.env")
            if os.path.exists(env_file):
                api_key = load_api_key_from_env_file(env_file)

        if not api_key:
            raise ValueError("No API key found. Set OPENROUTER_API_KEY or provide in config.")

        model = self.config.get("policy_model", "qwen/qwen3-8b")
        base_url = self.config.get("base_url", None)
        max_workers = self.config.get("max_workers", 8)
        provider = self.config.get("provider", None)
        
        if verbose:
            provider_str = f", provider={provider.get('order', ['auto'])[0]}" if provider else ""
            print(f"Initializing LLM: {model} (max_workers={max_workers}{provider_str})")

        self.llm = LLMWrapper(
            api_key=api_key,
            model=model,
            base_url=base_url,
            default_temperature=self.config.get("temperature", 0.7),
            default_top_p=self.config.get("top_p", 0.95),
            max_workers=max_workers,
            provider=provider,
        )

    def _init_evaluator(self):
        """Initialize evaluator(s)."""
        verbose = self.config.get("verbose", True)

        # Map task string to TaskType
        task_str = self.config.get("task", "dc")
        task_type = {
            "dc": TaskType.DC,
            "sp": TaskType.SP,
            "gn": TaskType.GN,
        }[task_str]

        # Determine data paths
        base_dir = Path(__file__).parent
        data_subset = self.config.get("data_subset", "test")

        test_path = self.config.get("test_data") or str(
            base_dir / "AR-Bench" / "data" / task_str / f"{data_subset}.json"
        )

        if not os.path.exists(test_path):
            raise FileNotFoundError(f"Test data not found: {test_path}")

        if verbose:
            print(f"Task: {task_type.value}")
            print(f"Data: {test_path}")

        max_workers = self.config.get("max_workers", 8)

        # Create question selector if enabled (DC only)
        question_selector = None

        if self.config.get("use_dqs", False) and task_str in ("dc", "gn"):
            # DQS (Deliberative Question Selection)
            # DC: LLM-based branch evaluation
            # GN: deterministic entropy-based scoring (no extra LLM calls)
            question_selector = DQSQuestionSelector(
                llm=self.llm,
                m_candidates=self.config.get("dqs_m_candidates", 5),
                r_simulations=self.config.get("dqs_r_simulations", 2),
                temperature=self.config.get("dqs_temperature", 0.7),
                eval_temperature=self.config.get("dqs_eval_temperature", 0.3),
            )
            if verbose:
                mode = "entropy" if task_str == "gn" else "LLM-eval"
                print(f"DQS question selection ({mode}): M={question_selector.m_candidates}")

        elif self.config.get("use_voi", False) and task_str == "dc":
            # VoI (Value of Information) – MI-based branch scoring
            clusterer = SemanticClusterer(use_entailment=False)
            robust_mi = RobustMI(
                llm=self.llm,
                clusterer=clusterer,
                k_samples=self.config.get("k_samples", 4),
                temperature=self.config.get("temperature", 0.7),
                use_diversity_sampling=self.config.get("use_diversity_sampling", True),
                regime=self.config.get("regime", "normal"),
            )
            question_selector = VoIQuestionSelector(
                llm=self.llm,
                clusterer=clusterer,
                robust_mi=robust_mi,
                m_candidates=self.config.get("voi_m_candidates", 5),
                r_simulations=self.config.get("voi_r_simulations", 2),
                k_samples=self.config.get("k_samples", 4),
                variants=self.config.get("variants", ["base", "skeptical"]),
                use_dpp=self.config.get("use_dpp", False),
                dpp_slate_size=self.config.get("dpp_slate_size", 3),
            )
            if verbose:
                print(f"VoI question selection: M={question_selector.m_candidates}, R={question_selector.r_simulations}")

        self.evaluator = RRMCEvaluator(
            llm=self.llm,
            data_path=test_path,
            task_type=task_type,
            max_turns=self.config.get("max_turns", 25),
            output_dir=self.config.get("output_dir", "results"),
            regime=self.config.get("regime", "normal"),
            max_workers=max_workers,
            question_selector=question_selector,
        )

        # Initialize train evaluator if calibrating
        if self.config.get("calibrate", False):
            train_path = self.config.get("train_data") or str(
                base_dir / "AR-Bench" / "data" / task_str / "train.json"
            )

            if not os.path.exists(train_path):
                # Fall back to test data
                train_path = test_path

            if train_path != test_path:
                self.train_evaluator = RRMCEvaluator(
                    llm=self.llm,
                    data_path=train_path,
                    task_type=task_type,
                    max_turns=self.config.get("max_turns", 25),
                    output_dir=self.config.get("output_dir", "results"),
                    regime=self.config.get("regime", "normal"),
                    max_workers=max_workers,
                )
            else:
                self.train_evaluator = self.evaluator

    def _run_calibration_pipeline(self) -> Dict[str, Any]:
        """Run calibration + evaluation pipeline."""
        verbose = self.config.get("verbose", True)

        # Get puzzle indices
        n_train = self.config.get("n_train", 20)
        n_test = self.config.get("n_test", 10)

        n_train_available = len(self.train_evaluator.env)
        n_test_available = len(self.evaluator.env)

        # Handle same train/test file
        if self.train_evaluator is self.evaluator:
            max_idx = n_train_available
            train_indices = list(range(0, min(n_train, max_idx // 2)))
            test_indices = list(range(max_idx // 2, max_idx // 2 + min(n_test, max_idx // 2)))
        else:
            train_indices = list(range(min(n_train, n_train_available)))
            test_indices = list(range(min(n_test, n_test_available)))

        if verbose:
            print(f"Train puzzles: {len(train_indices)}")
            print(f"Test puzzles: {len(test_indices)}")
            print(f"Target error rate: {self.config.get('target_error', 0.10):.1%}")
            print(f"Regime: {self.config.get('regime', 'normal')}")

        # Get variants
        variants = self.config.get("variants", ["base", "skeptical"])
        if isinstance(variants, str):
            variants = [v.strip() for v in variants.split(",")]

        # Phase 1: Collect calibration states
        calibrator = self.train_evaluator.collect_calibration_states(
            puzzle_indices=train_indices,
            k_samples=self.config.get("k_samples", 4),
            variants=variants,
            verbose=verbose,
        )

        # Phase 2: Calibrate threshold
        calibration_result = self.train_evaluator.calibrate_threshold(
            calibrator=calibrator,
            target_error=self.config.get("target_error", 0.10),
            verbose=verbose,
        )

        # Save calibration
        output_dir = self.config.get("output_dir", "results")
        os.makedirs(output_dir, exist_ok=True)

        task_str = self.config.get("task", "dc")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        calibration_path = os.path.join(output_dir, f"calibration_{task_str}_{timestamp}.json")
        calibrator.save(calibration_path)

        if verbose:
            print(f"Calibration saved to: {calibration_path}")

        # Get calibrated threshold
        calibrated_threshold = calibration_result.threshold
        if calibrated_threshold == -float('inf'):
            calibrated_threshold = 0.1
            if verbose:
                print(f"Using fallback threshold: {calibrated_threshold}")

        # Phase 3: Evaluate methods
        if verbose:
            print(f"\n{'='*60}")
            print("EVALUATION PHASE")
            print(f"{'='*60}")
            print(f"Using calibrated threshold τ = {calibrated_threshold:.4f}")

        eval_results = {}

        # Calibrated RRMC
        calibrated_rule = get_method(
            "robust_mi",
            llm=self.llm,
            clusterer=self.evaluator.clusterer,
            threshold=calibrated_threshold,
            k_samples=self.config.get("k_samples", 4),
            max_turns=self.config.get("max_turns", 25),
            variants=variants,
            regime=self.config.get("regime", "normal"),
        )
        eval_results["rrmc_calibrated"] = self.evaluator.evaluate_method(
            stopping_rule=calibrated_rule,
            puzzle_indices=test_indices,
            verbose=verbose,
        )

        # Compare against other methods
        methods = self.config.get("methods", ["fixed_turns", "self_consistency", "mi_only"])
        for method_name in methods:
            if method_name == "robust_mi":
                continue  # Already evaluated as rrmc_calibrated

            method_cfg = get_method_config(method_name)
            stopping_rule = self._create_method(method_name, method_cfg)
            eval_results[method_name] = self.evaluator.evaluate_method(
                stopping_rule=stopping_rule,
                puzzle_indices=test_indices,
                verbose=verbose,
            )

        # Save results
        self.evaluator._save_results(eval_results)

        # Get MI-error correlation
        mi_corr = getattr(calibrator, 'mi_error_correlation', None)

        results = {
            "calibration": {
                "threshold": calibration_result.threshold,
                "n_states": calibration_result.n_states,
                "n_covered": calibration_result.n_covered,
                "empirical_error": calibration_result.empirical_error,
                "ucb_error": calibration_result.ucb_error,
                "mi_error_correlation": mi_corr,
            },
            "evaluation": eval_results,
        }

        # Print comparison table
        print_comparison_table(eval_results, mi_error_correlation=mi_corr)

        if verbose:
            self._print_calibration_summary(results["calibration"])

        self._print_token_usage()
        return results

    def _run_comparison_pipeline(self) -> Dict[str, Any]:
        """Run simple comparison pipeline (no calibration)."""
        verbose = self.config.get("verbose", True)

        # Get number of puzzles (supports both n_puzzles and n_test)
        n_puzzles = self.config.get("n_puzzles", 5)
        n_available = len(self.evaluator.env)
        puzzle_indices = list(range(min(n_puzzles, n_available)))

        methods = self.config.get("methods", ["fixed_turns", "self_consistency", "mi_only", "robust_mi"])
        if isinstance(methods, str):
            methods = [m.strip() for m in methods.split(",")]

        if verbose:
            print(f"Evaluating {len(puzzle_indices)} puzzles")
            print(f"Methods: {methods}")
            print("\n" + "=" * 60)
            print("STARTING EVALUATION")
            print("=" * 60)

        # Check if trajectory ensemble is enabled
        ensemble_cfg = self.config.get("trajectory_ensemble", None)
        ensemble_enabled = False
        ensemble = None
        if ensemble_cfg is not None:
            if isinstance(ensemble_cfg, dict):
                ensemble_enabled = ensemble_cfg.get("enabled", False)
            else:
                # Config object
                ensemble_enabled = getattr(ensemble_cfg, "enabled", False)

        if ensemble_enabled:
            # Extract ensemble parameters
            if isinstance(ensemble_cfg, dict):
                ecfg = ensemble_cfg
            else:
                ecfg = ensemble_cfg.to_dict() if hasattr(ensemble_cfg, "to_dict") else {}

            ensemble = TrajectoryEnsemble(
                n_trajectories=ecfg.get("n_trajectories", 6),
                consensus_threshold=ecfg.get("consensus_threshold", 0.5),
                early_consensus=ecfg.get("early_consensus", True),
                temperature_range=tuple(ecfg.get("temperature_range", [0.5, 1.0])),
            )

        results = {}
        for method_name in methods:
            method_cfg = get_method_config(method_name)
            stopping_rule = self._create_method(method_name, method_cfg)

            if ensemble_enabled and ensemble is not None:
                results[method_name] = self.evaluator.evaluate_method_ensemble(
                    stopping_rule=stopping_rule,
                    ensemble=ensemble,
                    puzzle_indices=puzzle_indices,
                    verbose=verbose,
                )
            else:
                results[method_name] = self.evaluator.evaluate_method(
                    stopping_rule=stopping_rule,
                    puzzle_indices=puzzle_indices,
                    verbose=verbose,
                )

        # Save and print
        self.evaluator._save_results(results)
        print_comparison_table(results)
        self._print_token_usage()

        return {"evaluation": results}

    def _create_method(self, method_name: str, method_cfg: Dict[str, Any]):
        """Create a stopping method with merged config.
        
        Priority: experiment config > method config > defaults
        """
        # Base kwargs
        kwargs = {
            "llm": self.llm,
            "max_turns": self.config.get("max_turns", 25),
        }

        # Add clusterer for methods that need it
        if method_name in ["self_consistency", "semantic_entropy", "mi_only", "robust_mi", 
                           "knowno", "cip_lite", "uot_lite"]:
            kwargs["clusterer"] = self.evaluator.clusterer

        # Method-specific settings (experiment config overrides method config)
        if method_name == "fixed_turns":
            # Allow experiment config to override fixed_turns
            kwargs["fixed_turns"] = self.config.get(
                "fixed_turns", method_cfg.get("fixed_turns", 10)
            )
        elif method_name == "self_consistency":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 8)
            )
            kwargs["consistency_threshold"] = self.config.get(
                "consistency_threshold", method_cfg.get("consistency_threshold", 0.7)
            )
        elif method_name == "semantic_entropy":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 8)
            )
            kwargs["entropy_threshold"] = self.config.get(
                "entropy_threshold", method_cfg.get("entropy_threshold", 0.5)
            )
        elif method_name == "verbalized_confidence":
            kwargs["confidence_threshold"] = self.config.get(
                "confidence_threshold", method_cfg.get("confidence_threshold", 8.0)
            )
        elif method_name == "mi_only":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 6)
            )
            kwargs["mi_threshold"] = self.config.get(
                "mi_threshold", method_cfg.get("mi_threshold", 0.3)
            )
            kwargs["min_turns"] = self.config.get(
                "min_turns", method_cfg.get("min_turns", 0)
            )
        elif method_name == "robust_mi":
            kwargs["k_samples"] = self.config.get("k_samples", method_cfg.get("k_samples", 4))
            kwargs["threshold"] = self.config.get(
                "threshold", method_cfg.get("threshold", 0.3)
            )
            kwargs["use_diversity_sampling"] = self.config.get(
                "use_diversity_sampling",
                method_cfg.get("use_diversity_sampling", True)
            )
            kwargs["regime"] = self.config.get("regime", method_cfg.get("regime", "normal"))
            variants = self.config.get("variants", method_cfg.get("variants", ["base", "skeptical"]))
            if isinstance(variants, str):
                variants = [v.strip() for v in variants.split(",")]
            kwargs["variants"] = variants
            kwargs["min_turns"] = self.config.get(
                "min_turns", method_cfg.get("min_turns", 0)
            )
        elif method_name == "knowno":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 10)
            )
            kwargs["set_size_threshold"] = self.config.get(
                "set_size_threshold", method_cfg.get("set_size_threshold", 1)
            )
        elif method_name == "cip_lite":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 10)
            )
            kwargs["set_size_threshold"] = self.config.get(
                "set_size_threshold", method_cfg.get("set_size_threshold", 2)
            )
        elif method_name == "uot_lite":
            kwargs["k_samples"] = self.config.get(
                "k_samples", method_cfg.get("k_samples", 6)
            )
        elif method_name == "perturbed_mi_ece":
            # Perturbed MI+ECE method
            kwargs["k_samples"] = self.config.get("k_samples", method_cfg.get("k_samples", 8))
            kwargs["threshold"] = self.config.get("threshold", method_cfg.get("threshold", 0.3))
            kwargs["temperature"] = self.config.get("temperature", method_cfg.get("temperature", 0.7))
            
            # Variants for MI
            variants = self.config.get("variants", method_cfg.get("variants", ["base", "skeptical"]))
            if isinstance(variants, str):
                variants = [v.strip() for v in variants.split(",")]
            kwargs["variants"] = variants
            
            # MI and ECE weights
            combined_unc = self.config.get("combined_uncertainty", method_cfg.get("combined_uncertainty", {}))
            if isinstance(combined_unc, dict):
                kwargs["alpha"] = combined_unc.get("alpha", method_cfg.get("alpha", 0.5))
                kwargs["beta"] = combined_unc.get("beta", method_cfg.get("beta", 0.5))
                kwargs["ece_bins"] = combined_unc.get("ece_bins", method_cfg.get("ece_bins", 10))
            else:
                kwargs["alpha"] = getattr(combined_unc, "alpha", 0.5)
                kwargs["beta"] = getattr(combined_unc, "beta", 0.5)
                kwargs["ece_bins"] = getattr(combined_unc, "ece_bins", 10)
            
            # Perturbation config
            pert_cfg = self.config.get("perturbation", method_cfg.get("perturbation", {}))
            if pert_cfg:
                from rrmc.core.perturbation import PerturbationConfig
                if isinstance(pert_cfg, dict):
                    kwargs["perturbation_config"] = PerturbationConfig(
                        n_samples=kwargs["k_samples"],
                        noise_type=pert_cfg.get("noise_type", "gaussian"),
                        noise_scale=pert_cfg.get("noise_scale", 0.1),
                        temperature_range=tuple(pert_cfg.get("temperature_range", [0.3, 1.2])),
                        use_prompt_perturbation=pert_cfg.get("use_prompt_perturbation", True),
                        use_embedding_noise=pert_cfg.get("use_embedding_noise", True),
                    )
            
            kwargs["regime"] = self.config.get("regime", method_cfg.get("regime", "normal"))
            
        elif method_name == "ece_only":
            # ECE-only method
            kwargs["k_samples"] = self.config.get("k_samples", method_cfg.get("k_samples", 8))
            kwargs["ece_threshold"] = self.config.get("ece_threshold", method_cfg.get("ece_threshold", 0.15))
            kwargs["ece_bins"] = self.config.get("ece_bins", method_cfg.get("ece_bins", 10))
            
            # Perturbation config
            pert_cfg = self.config.get("perturbation", method_cfg.get("perturbation", {}))
            if pert_cfg:
                from rrmc.core.perturbation import PerturbationConfig
                if isinstance(pert_cfg, dict):
                    kwargs["perturbation_config"] = PerturbationConfig(
                        n_samples=kwargs["k_samples"],
                        noise_type=pert_cfg.get("noise_type", "gaussian"),
                        noise_scale=pert_cfg.get("noise_scale", 0.1),
                        temperature_range=tuple(pert_cfg.get("temperature_range", [0.3, 1.2])),
                        use_prompt_perturbation=pert_cfg.get("use_prompt_perturbation", True),
                        use_embedding_noise=pert_cfg.get("use_embedding_noise", True),
                    )

        stopping_rule = get_method(method_name, **kwargs)

        # Wrap with AllSuspectsWrapper if configured (DC only)
        if self.config.get("require_all_suspects", False):
            stopping_rule = AllSuspectsWrapper(
                inner_rule=stopping_rule,
                enabled=True,
            )

        return stopping_rule

    def _print_calibration_summary(self, calibration: Dict[str, Any]):
        """Print calibration summary."""
        print(f"\nCalibration Summary:")
        print(f"  Threshold (τ): {calibration['threshold']:.4f}")
        print(f"  States collected: {calibration['n_states']}")
        print(f"  States covered: {calibration['n_covered']}")
        print(f"  Empirical error: {calibration['empirical_error']:.2%}")
        if calibration.get("mi_error_correlation"):
            corr = calibration["mi_error_correlation"]
            print(f"  MI-Error ρ: {corr['rho']:.4f} (p={corr['pvalue']:.4f})")

    def _print_token_usage(self):
        """Print API token usage."""
        if self.llm:
            usage = self.llm.get_token_usage()
            print(f"\nTotal API usage:")
            print(f"  Prompt tokens: {usage['prompt_tokens']:,}")
            print(f"  Completion tokens: {usage['completion_tokens']:,}")
            print(f"  Total tokens: {usage['total_tokens']:,}")
