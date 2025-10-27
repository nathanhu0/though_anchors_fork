#!/usr/bin/env python3
"""
Visualize prompt structure and boundaries for extending suppression analysis.
"""

import os
import sys
import json
from typing import List, Tuple
import re

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attention_analysis.receiver_head_funcs import get_problem_text_sentences, get_model_rollouts_root
from attention_analysis.attn_supp_funcs import get_sentence_token_boundaries
from utils import split_solution_keep_spacing


def parse_prompt_components(full_text: str) -> List[Tuple[str, str, Tuple[int, int]]]:
    """
    Parse the full prompt text into logical components.

    Returns:
        List of (component_type, text, (start_char, end_char)) tuples
    """
    components = []
    current_pos = 0

    # 1. Find system instruction (everything before "Problem:")
    problem_match = re.search(r'\bProblem:\s*', full_text)
    if problem_match:
        system_text = full_text[:problem_match.start()].strip()
        if system_text:
            components.append(("system_instruction", system_text, (current_pos, problem_match.start())))

        # 2. Problem marker
        problem_start = problem_match.start()
        problem_marker_end = problem_match.end()
        components.append(("problem_marker", full_text[problem_start:problem_marker_end], (problem_start, problem_marker_end)))
        current_pos = problem_marker_end

    # 3. Find problem content (between "Problem:" and "Solution:")
    solution_match = re.search(r'\bSolution:\s*', full_text)
    if solution_match and problem_match:
        problem_content = full_text[problem_marker_end:solution_match.start()].strip()
        if problem_content:
            # Split problem content into sentences
            problem_sentences = split_problem_into_sentences(problem_content)
            sentence_start = problem_marker_end
            for sentence in problem_sentences:
                sentence = sentence.strip()
                if sentence:
                    # Find this sentence in the original text
                    sentence_pos = full_text.find(sentence, sentence_start)
                    if sentence_pos != -1:
                        components.append(("problem_sentence", sentence, (sentence_pos, sentence_pos + len(sentence))))
                        sentence_start = sentence_pos + len(sentence)

        # 4. Solution marker
        solution_start = solution_match.start()
        solution_marker_end = solution_match.end()
        components.append(("solution_marker", full_text[solution_start:solution_marker_end], (solution_start, solution_marker_end)))
        current_pos = solution_marker_end

    # 5. Find thinking start marker
    think_match = re.search(r'<think>\s*', full_text[current_pos:])
    if think_match:
        think_start = current_pos + think_match.start()
        think_end = current_pos + think_match.end()
        components.append(("think_marker", full_text[think_start:think_end], (think_start, think_end)))
        current_pos = think_end

    # 6. Everything after <think> is the actual response content
    if current_pos < len(full_text):
        response_content = full_text[current_pos:]
        if response_content.strip():
            components.append(("response_content", response_content, (current_pos, len(full_text))))

    return components


def split_problem_into_sentences(problem_text: str) -> List[str]:
    """Split problem text into sentences, handling mathematical notation carefully."""
    # Simple sentence splitting - could be made more sophisticated
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', problem_text)
    return [s.strip() for s in sentences if s.strip()]


def visualize_problem_structure(problem_num: int, model_name: str = "llama-8b", is_correct: bool = True):
    """Visualize the prompt structure for a given problem."""
    print(f"=== Problem {problem_num} Structure ===")

    # Get the full text
    full_text, response_sentences = get_problem_text_sentences(problem_num, is_correct, model_name)

    print(f"Total text length: {len(full_text)} characters")
    print(f"Response sentences: {len(response_sentences)}")
    print()

    # Parse prompt components
    components = parse_prompt_components(full_text)

    print("PROMPT COMPONENTS:")
    print("-" * 50)

    for i, (comp_type, text, (start, end)) in enumerate(components):
        if comp_type == "response_content":
            # Don't print the full response content, just show it exists
            print(f"{i:2d}. {comp_type:20s} | chars {start:4d}-{end:4d} | [Response content - {len(text)} chars]")
            break
        else:
            # Show first 100 chars of component
            display_text = text.replace('\n', '\\n')[:100]
            if len(text) > 100:
                display_text += "..."
            print(f"{i:2d}. {comp_type:20s} | chars {start:4d}-{end:4d} | {display_text}")

    print()
    print("RESPONSE SENTENCE BREAKDOWN:")
    print("-" * 50)

    # Find where response content starts
    response_start = None
    for comp_type, text, (start, end) in components:
        if comp_type == "response_content":
            response_start = start
            break

    if response_start:
        # Get token boundaries for response sentences
        response_boundaries = get_sentence_token_boundaries(full_text, response_sentences, model_name)

        for i, sentence in enumerate(response_sentences[:5]):  # Show first 5 sentences
            if i < len(response_boundaries):
                token_start, token_end = response_boundaries[i]
                display_sentence = sentence.replace('\n', '\\n')[:80]
                if len(sentence) > 80:
                    display_sentence += "..."
                print(f"{i:2d}. tokens {token_start:3d}-{token_end:3d} | {display_sentence}")

        if len(response_sentences) > 5:
            print(f"... and {len(response_sentences) - 5} more response sentences")

    print()
    return components, response_sentences


def get_prompt_token_boundaries(full_text: str, components: List[Tuple[str, str, Tuple[int, int]]], model_name: str):
    """Convert character boundaries to token boundaries for prompt components."""
    from attention_analysis.attn_supp_funcs import get_raw_tokens

    # Get all tokens for the full text
    tokens = get_raw_tokens(full_text, model_name)

    # For each component, find which tokens it spans
    component_boundaries = []

    # This is a simplified approach - we'd need to implement proper char->token mapping
    for comp_type, text, (char_start, char_end) in components:
        if comp_type != "response_content":
            # For now, estimate token boundaries (this would need refinement)
            approx_token_start = len(get_raw_tokens(full_text[:char_start], model_name))
            approx_token_end = len(get_raw_tokens(full_text[:char_end], model_name))
            component_boundaries.append((comp_type, (approx_token_start, approx_token_end)))

    return component_boundaries


def show_raw_tokens(problem_num: int, model_name: str = "llama-8b", is_correct: bool = True):
    """Show the actual raw tokens to see special tokens like user/assistant markers."""
    from attention_analysis.attn_supp_funcs import get_raw_tokens

    full_text, _ = get_problem_text_sentences(problem_num, is_correct, model_name)
    tokens = get_raw_tokens(full_text, model_name)

    print(f"=== RAW TOKENS for Problem {problem_num} ===")
    print(f"Total tokens: {len(tokens)}")
    print()

    # Show first 50 tokens to see the prompt structure
    for i, token in enumerate(tokens[:50]):
        print(f"{i:3d}: {repr(token)}")

    print("\n... [truncated] ...\n")

    # Show tokens around the think marker
    think_text = "<think>"
    for i, token in enumerate(tokens):
        if think_text in str(token):
            start_idx = max(0, i-5)
            end_idx = min(len(tokens), i+10)
            print(f"Tokens around '<think>' (indices {start_idx}-{end_idx}):")
            for j in range(start_idx, end_idx):
                marker = " <-- THINK" if j == i else ""
                print(f"{j:3d}: {repr(tokens[j])}{marker}")
            break

    return tokens


if __name__ == "__main__":
    # Test with a few problems
    test_problems = [1591]  # Just one for now to see the token structure

    for problem_num in test_problems:
        try:
            # Show raw tokens first
            tokens = show_raw_tokens(problem_num)
            print("\n" + "="*80 + "\n")

            # Then show our component parsing
            components, sentences = visualize_problem_structure(problem_num)

            # Show token boundaries too
            print("ESTIMATED TOKEN BOUNDARIES:")
            print("-" * 50)
            full_text, _ = get_problem_text_sentences(problem_num, True, "llama-8b")
            token_boundaries = get_prompt_token_boundaries(full_text, components, "llama-8b")
            for comp_type, (start, end) in token_boundaries:
                print(f"{comp_type:20s} | tokens {start:3d}-{end:3d}")

            print("\n" + "="*80 + "\n")

        except Exception as e:
            print(f"Error with problem {problem_num}: {e}")
            print()