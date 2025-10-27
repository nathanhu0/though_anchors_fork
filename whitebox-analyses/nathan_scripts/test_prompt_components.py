#!/usr/bin/env python3
"""
Test script for parsing prompt components with aggregated structure.
"""

import os
import sys
import json
from typing import List, Tuple, Dict
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import get_problem_text_sentences
from attention_analysis.attn_supp_funcs import get_raw_tokens, get_sentence_token_boundaries


def parse_prompt_components_aggregated(full_text: str, model_name: str) -> Dict:
    """
    Parse prompt into aggregated components: prompt_prefix, problem_sentences, cot_prefix.

    Returns:
        Dict with component information including token boundaries
    """
    # Get raw tokens for precise boundaries
    tokens = get_raw_tokens(full_text, model_name)

    # Find key markers in the text
    problem_match = re.search(r'\bProblem:\s*', full_text)
    solution_match = re.search(r'\bSolution:\s*', full_text)
    think_match = re.search(r'<think>\s*', full_text)

    if not all([problem_match, solution_match, think_match]):
        raise ValueError("Could not find required markers in text")

    # 1. PROMPT PREFIX: Everything from BOS to end of "Problem: "
    prompt_prefix_end_char = problem_match.end()
    prompt_prefix_text = full_text[:prompt_prefix_end_char]
    prompt_prefix_tokens = len(get_raw_tokens(prompt_prefix_text, model_name))

    # 2. PROBLEM SENTENCES: Between "Problem: " and "Solution:"
    problem_start_char = problem_match.end()
    problem_end_char = solution_match.start()
    problem_content = full_text[problem_start_char:problem_end_char].strip()

    # Split problem content into sentences
    problem_sentences = split_problem_sentences(problem_content)

    # Get token boundaries for each problem sentence
    problem_sentence_boundaries = []
    current_char_pos = problem_start_char
    current_token_pos = prompt_prefix_tokens

    for sentence in problem_sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        # Find this sentence in the text
        sentence_start_char = full_text.find(sentence, current_char_pos)
        if sentence_start_char == -1:
            print(f"Warning: Could not find sentence '{sentence[:50]}...' in text")
            continue

        sentence_end_char = sentence_start_char + len(sentence)

        # Calculate token boundaries
        text_up_to_sentence_start = full_text[:sentence_start_char]
        text_up_to_sentence_end = full_text[:sentence_end_char]

        sentence_start_token = len(get_raw_tokens(text_up_to_sentence_start, model_name))
        sentence_end_token = len(get_raw_tokens(text_up_to_sentence_end, model_name))

        problem_sentence_boundaries.append({
            'text': sentence,
            'char_range': (sentence_start_char, sentence_end_char),
            'token_range': (sentence_start_token, sentence_end_token)
        })

        current_char_pos = sentence_end_char
        current_token_pos = sentence_end_token

    # 3. COT PREFIX: "Solution: " + "<think>\n" + everything until model response starts
    cot_start_char = solution_match.start()

    # Find where the actual response starts (after <think>\n)
    think_end_char = think_match.end()
    cot_end_char = think_end_char

    cot_prefix_text = full_text[cot_start_char:cot_end_char]

    # Token boundaries for CoT prefix
    text_up_to_cot_start = full_text[:cot_start_char]
    text_up_to_cot_end = full_text[:cot_end_char]

    cot_start_token = len(get_raw_tokens(text_up_to_cot_start, model_name))
    cot_end_token = len(get_raw_tokens(text_up_to_cot_end, model_name))

    return {
        'prompt_prefix': {
            'text': prompt_prefix_text,
            'char_range': (0, prompt_prefix_end_char),
            'token_range': (0, prompt_prefix_tokens)
        },
        'problem_sentences': problem_sentence_boundaries,
        'cot_prefix': {
            'text': cot_prefix_text,
            'char_range': (cot_start_char, cot_end_char),
            'token_range': (cot_start_token, cot_end_token)
        },
        'response_start_token': cot_end_token,
        'total_tokens': len(tokens)
    }


def split_problem_sentences(problem_text: str) -> List[str]:
    """Split problem text into sentences, handling mathematical notation carefully."""
    # Handle common issues in math problems

    # First, protect mathematical expressions and special cases
    problem_text = problem_text.strip()

    # Simple sentence splitting that handles most math problems
    # Split on sentence endings followed by whitespace and capital letters
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', problem_text)

    # Also handle cases where sentences end with periods but the next sentence starts with lowercase
    # (common in math problems with variable names)
    refined_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence:
            refined_sentences.append(sentence)

    # If we only got one sentence but it's very long, try splitting on other cues
    if len(refined_sentences) == 1 and len(refined_sentences[0]) > 200:
        # Try splitting on common problem delimiters
        long_sentence = refined_sentences[0]
        # Look for patterns like "Given that...", "If...", etc.
        additional_splits = re.split(r'\s+(?=(?:Given|If|Find|Determine|Calculate|What|How)\s)', long_sentence)
        if len(additional_splits) > 1:
            refined_sentences = [s.strip() for s in additional_splits if s.strip()]

    return refined_sentences


def test_prompt_parsing(problem_num: int, model_name: str = "llama-8b"):
    """Test prompt parsing on a specific problem."""
    print(f"=== Testing Prompt Parsing for Problem {problem_num} ===")

    # Get the full text
    full_text, response_sentences = get_problem_text_sentences(problem_num, True, model_name)

    try:
        # Parse components
        components = parse_prompt_components_aggregated(full_text, model_name)

        print(f"Total text length: {len(full_text)} characters")
        print(f"Total tokens: {components['total_tokens']}")
        print(f"Response starts at token: {components['response_start_token']}")
        print()

        # Show prompt prefix
        prompt_prefix = components['prompt_prefix']
        print("PROMPT PREFIX:")
        print(f"  Tokens {prompt_prefix['token_range'][0]}-{prompt_prefix['token_range'][1]}")
        print(f"  Text: {repr(prompt_prefix['text'][:100])}...")
        print()

        # Show problem sentences
        print("PROBLEM SENTENCES:")
        for i, sentence_info in enumerate(components['problem_sentences']):
            token_start, token_end = sentence_info['token_range']
            text = sentence_info['text'][:80].replace('\n', '\\n')
            if len(sentence_info['text']) > 80:
                text += "..."
            print(f"  {i}: tokens {token_start:3d}-{token_end:3d} | {text}")
        print()

        # Show CoT prefix
        cot_prefix = components['cot_prefix']
        print("COT PREFIX:")
        print(f"  Tokens {cot_prefix['token_range'][0]}-{cot_prefix['token_range'][1]}")
        print(f"  Text: {repr(cot_prefix['text'])}")
        print()

        # Verify continuity
        print("CONTINUITY CHECK:")
        expected_next_token = prompt_prefix['token_range'][1]
        for i, sentence_info in enumerate(components['problem_sentences']):
            actual_start = sentence_info['token_range'][0]
            actual_end = sentence_info['token_range'][1]
            if actual_start >= expected_next_token:  # Allow for small gaps due to whitespace
                print(f"  ✓ Problem sentence {i}: tokens {actual_start}-{actual_end}")
                expected_next_token = actual_end
            else:
                print(f"  ⚠ Problem sentence {i}: tokens {actual_start}-{actual_end} (overlap with previous)")

        cot_start = cot_prefix['token_range'][0]
        if cot_start >= expected_next_token:
            print(f"  ✓ CoT prefix: tokens {cot_start}-{cot_prefix['token_range'][1]}")
        else:
            print(f"  ⚠ CoT prefix: tokens {cot_start}-{cot_prefix['token_range'][1]} (overlap)")

        return components

    except Exception as e:
        print(f"Error parsing problem {problem_num}: {e}")
        return None


if __name__ == "__main__":
    # Test on our sample problems
    test_problems = [1591, 2050, 330]

    for problem_num in test_problems:
        try:
            components = test_prompt_parsing(problem_num)
            print("\n" + "="*80 + "\n")
        except Exception as e:
            print(f"Failed to test problem {problem_num}: {e}")
            print("\n" + "="*80 + "\n")