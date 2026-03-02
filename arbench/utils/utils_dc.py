"""
Crime Detection Game utilities for AR-Bench

This module provides utilities for the crime detection/mystery solving game including
answer choice handling, text formatting, and evaluation functions.
"""
import re
import json
from typing import Dict, List, Optional, Any, Tuple


# Constants for crime detection game
ANSWER_CHOICES = ["A", "B", "C", "D"]
CHOICE_TO_INDEX = {"A": 0, "B": 1, "C": 2, "D": 3}


def convert_initial_info_to_string(initial_info: Dict[str, Any]) -> str:
    """
    Convert initial crime information dictionary to formatted string.
    
    Args:
        initial_info: Dictionary containing crime scene information
        
    Returns:
        Formatted string representation of the crime information
    """
    result = ""
    for key, value in initial_info.items():
        if isinstance(value, list):
            result += f"{key}: {', '.join(map(str, value))}\n"
        else:
            result += f"{key}: {value}\n"
    return result.strip()


def extract_answer_choice(response: str) -> Optional[str]:
    """
    Extract answer choice (A, B, C, D) from response text.
    
    Args:
        response: Response text that may contain an answer choice
        
    Returns:
        Extracted answer choice or None if not found
    """
    # Look for patterns like "Answer: A", "Choice: B", or just "A"
    patterns = [
        r"(?:Answer|Choice|Option):\s*([ABCD])",
        r"\b([ABCD])\b",
        r"answer\s+is\s+([ABCD])",
        r"choose\s+([ABCD])",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE)
        if match:
            choice = match.group(1).upper()
            if choice in ANSWER_CHOICES:
                return choice
    
    return None


def format_choices(choices: List[str]) -> str:
    """
    Format a list of choices as A, B, C, D options.
    
    Args:
        choices: List of choice strings
        
    Returns:
        Formatted string with lettered choices
    """
    if len(choices) != 4:
        raise ValueError("Crime detection game requires exactly 4 choices")
    
    formatted = ""
    for i, choice in enumerate(choices):
        formatted += f"{ANSWER_CHOICES[i]}. {choice}\n"
    return formatted.strip()


def calculate_accuracy(predictions: List[str], ground_truths: List[str]) -> float:
    """
    Calculate accuracy for crime detection predictions.
    
    Args:
        predictions: List of predicted answer choices
        ground_truths: List of ground truth answer choices
        
    Returns:
        Accuracy as float between 0 and 1
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have same length")
    
    if len(predictions) == 0:
        return 0.0
    
    correct = sum(1 for pred, truth in zip(predictions, ground_truths) if pred == truth)
    return correct / len(predictions)


def is_valid_choice(choice: str) -> bool:
    """
    Check if a choice is valid (A, B, C, or D).
    
    Args:
        choice: Choice string to validate
        
    Returns:
        True if choice is valid, False otherwise
    """
    return choice.upper() in ANSWER_CHOICES


def choice_to_index(choice: str) -> int:
    """
    Convert choice letter to index.
    
    Args:
        choice: Choice letter (A, B, C, D)
        
    Returns:
        Index (0, 1, 2, 3)
        
    Raises:
        ValueError: If choice is not valid
    """
    choice_upper = choice.upper()
    if choice_upper not in CHOICE_TO_INDEX:
        raise ValueError(f"Invalid choice: {choice}")
    return CHOICE_TO_INDEX[choice_upper]


def index_to_choice(index: int) -> str:
    """
    Convert index to choice letter.
    
    Args:
        index: Index (0, 1, 2, 3)
        
    Returns:
        Choice letter (A, B, C, D)
        
    Raises:
        ValueError: If index is not valid
    """
    if index not in range(4):
        raise ValueError(f"Invalid index: {index}")
    return ANSWER_CHOICES[index]


def extract_reasoning(response: str) -> str:
    """
    Extract reasoning text from response, excluding the final answer choice.
    
    Args:
        response: Full response text
        
    Returns:
        Reasoning text without the answer choice
    """
    # Try to find answer pattern and extract everything before it
    patterns = [
        r"(.+?)(?:Answer|Choice|Option):\s*[ABCD]",
        r"(.+?)(?:The answer is|I choose|My answer is)\s*[ABCD]",
    ]
    
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # If no answer pattern found, return the full response
    return response.strip()


class CrimeDetectionSearchTreeNode:
    """Node class for crime detection search tree."""
    
    def __init__(self, state: Dict[str, Any], action: str = None, parent: 'CrimeDetectionSearchTreeNode' = None):
        self.state = state
        self.action = action
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0.0
        
    def add_child(self, child_node: 'CrimeDetectionSearchTreeNode') -> None:
        """Add a child node to this node."""
        self.children.append(child_node)
        child_node.parent = self
        
    def is_leaf(self) -> bool:
        """Check if this node is a leaf node."""
        return len(self.children) == 0
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation."""
        return {
            "state": self.state,
            "action": self.action,
            "visits": self.visits,
            "value": self.value,
            "children": [child.to_dict() for child in self.children]
        }
