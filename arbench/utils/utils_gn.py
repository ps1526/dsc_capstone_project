"""
Guess Number Game utilities for AR-Bench

This module provides utilities for the guess number game including number generation,
guess extraction, comparison logic, and related exceptions.
"""
import random
import re
from collections import Counter
from typing import List, Tuple


class NotNumberError(Exception):
    """Exception raised when a number is expected but not found."""
    
    def __init__(self):
        pass

    def __str__(self) -> str:
        return "There is not a number"


def generate_unique_four_digit_number() -> List[int]:
    """
    Generate a unique four-digit number with all different digits.
    
    Returns:
        List of 4 integers representing the digits
    """
    digits = list(range(10))
    random.shuffle(digits)
    
    # Ensure the first digit is not 0
    if digits[0] == 0:
        # If first digit is 0, swap with any non-zero digit
        for i in range(1, 10):
            if digits[i] != 0:
                digits[0], digits[i] = digits[i], digits[0]
                break
    
    # Return the first four digits
    return digits[:4]


def extract_and_convert_guess(s: str) -> List[int]:
    """
    Extract a 4-digit number from a string and convert to list of integers.
    
    Args:
        s: Input string that may contain a 4-digit number
        
    Returns:
        List of 4 integers if found, empty list otherwise
    """
    matches = re.findall(r"\d{4}", s)

    if matches:
        # Extract the last matched number string
        number_string = matches[-1]
        # Convert to list of integers
        number_list = [int(char) for char in number_string]
        return number_list
    else:
        # Return empty list if no 4-digit number found
        return []


def compare_guess(list1: List[int], list2: List[int]) -> Tuple[int, int, int]:

    # Convert input to list of integers if needed
    if isinstance(list1, str):
        list1 = [int(d) for d in list1]
    if isinstance(list2, str):
        list2 = [int(d) for d in list2]

    if len(list1) != len(list2):
        raise ValueError("Lists must have the same length")

    # Count elements in same positions
    same_position_count = sum(a == b for a, b in zip(list1, list2))

    # Count frequency of elements in both lists
    list1_counts = Counter(list1)
    list2_counts = Counter(list2)

    # Calculate total same elements (including same and different positions)
    same_elements = sum((list1_counts & list2_counts).values())

    # Same elements but different positions = total same - same position
    same_element_different_position_count = same_elements - same_position_count

    # Calculate final score (same position worth 2 points, different position worth 1 point)
    total_score = same_position_count * 2 + same_element_different_position_count

    return same_position_count, same_element_different_position_count, total_score
