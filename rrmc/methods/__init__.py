"""
RRMC Stopping Methods.

Provides stopping rules and a registry for config-based instantiation.
"""

from typing import Any, Dict, List

from .stopping_rules import (
    BaseStoppingRule,
    FixedTurnsStopping,
    SelfConsistencyStopping,
    VerbalizedConfidenceStopping,
    SemanticEntropyStopping,
    MIOnlyStopping,
    RobustMIStopping,
    KnowNoStopping,
    CIPLiteStopping,
    UoTLiteStopping,
    AllSuspectsWrapper,
    PerturbedMIECEStopping,
    ECEOnlyStopping,
)


# Method registry mapping names to classes
METHODS: Dict[str, type] = {
    "fixed_turns": FixedTurnsStopping,
    "self_consistency": SelfConsistencyStopping,
    "verbalized_confidence": VerbalizedConfidenceStopping,
    "semantic_entropy": SemanticEntropyStopping,
    "mi_only": MIOnlyStopping,
    "robust_mi": RobustMIStopping,
    "knowno": KnowNoStopping,
    "cip_lite": CIPLiteStopping,
    "uot_lite": UoTLiteStopping,
    "perturbed_mi_ece": PerturbedMIECEStopping,
    "ece_only": ECEOnlyStopping,
}


def get_method(name: str, **kwargs) -> BaseStoppingRule:
    """
    Get a stopping rule instance by name.

    Args:
        name: Method name (e.g., "robust_mi", "fixed_turns")
        **kwargs: Arguments to pass to the constructor

    Returns:
        Instantiated stopping rule

    Raises:
        ValueError: If method name is not recognized
    """
    if name not in METHODS:
        available = list(METHODS.keys())
        raise ValueError(f"Unknown method: {name}. Available: {available}")

    return METHODS[name](**kwargs)


def list_methods() -> List[str]:
    """List available method names."""
    return list(METHODS.keys())
