"""
20 Questions Game utilities for AR-Bench

This module provides utilities for the 20 questions game including scoring functions,
text parsing, keypoint handling, and related exceptions.
"""
import re
from collections import Counter
from typing import List, Tuple, Optional



class NotQuestionError(Exception):
    """Exception raised when a question is expected but not found."""
    
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "There is not a question"


def f1_score(prediction: List[str], ground_truth: List[str], **kwargs) -> float:
    """
    Calculate F1 score between prediction and ground truth lists.
    
    Args:
        prediction: Predicted list of items
        ground_truth: Ground truth list of items
        **kwargs: Additional keyword arguments (ignored)
        
    Returns:
        F1 score as float
    """
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1



def contains_number(input_string: str) -> bool:
    """
    Check if input string contains any digit.
    
    Args:
        input_string: String to check
        
    Returns:
        True if string contains digits, False otherwise
    """
    return bool(re.search(r"\d", input_string))


def place_keypoints(keypoints: List[str]) -> str:
    """
    Format keypoints as numbered list.
    
    Args:
        keypoints: List of keypoint strings
        
    Returns:
        Formatted string with numbered keypoints
    """
    text = ""
    for i, key in enumerate(keypoints):
        text += f"{i+1}. {key}\n"
    return text


def parse_keypoints(input_string: str) -> List[int]:
    """
    Parse keypoint numbers from "Hit point:" formatted string.
    
    Args:
        input_string: String containing "Hit point: x, y, z" format
        
    Returns:
        List of extracted integer keypoint numbers
        
    Raises:
        ValueError: If parsing fails
    """
    # Extract numbers following "Hit point:"
    matches = re.findall(r"Hit point: ([\d, ]+)", input_string)

    # Parse numbers from matched string
    numbers = []
    if matches:
        try:
            numbers = [int(num.strip()) for num in matches[0].split(",") if num.strip()]
        except:
            print(f" ==> input_string: {input_string}, matches: {matches}")
            raise ValueError
    return numbers


def parse_match_info(text: str) -> Tuple[Optional[str], int]:
    """
    Parse match information from formatted text.
    
    Args:
        text: Text containing "Match point:" and "Match count:" information
        
    Returns:
        Tuple of (match_point_string, match_count_integer)
    """
    # Match pattern for "Match point:" followed by "Match count:"
    match_pattern = r"Match point:\s*([\d,\s]*?),\s*Match count:\s*(\d+)"

    # Search for matching pattern
    match = re.search(match_pattern, text)

    if match:
        # Extract match point and count
        match_point = match.group(1).strip()
        match_count = int(match.group(2).strip())
        return match_point, match_count
    else:
        return None, 0


def extract_questioner_respond(text: str) -> bool:
    """
    Check if text contains questioner response format.
    
    Args:
        text: Text to check
        
    Returns:
        True if text contains "question:" or "answer:", False otherwise
    """
    return "question:" in text.lower() or "answer:" in text.lower()
