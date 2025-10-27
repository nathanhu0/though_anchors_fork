"""
Detailed attention interaction analysis with token-level granularity.
Supports both suppression and amplification interventions.
Returns probability changes and KL divergences at token level without sentence aggregation.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Tuple
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkld import pkld
from attention_analysis.receiver_head_funcs import get_problem_text_sentences, get_model_rollouts_root
from attention_analysis.attn_supp_funcs import get_sentence_token_boundaries, get_raw_tokens
from nh_utils import parse_prompt_components_aggregated, calculate_kl_divergence_full_tensor, calculate_tv_distance_full_tensor, extract_token_probabilities_full
from pytorch_models.model_config import model2layers_heads
# Removed unused logits_funcs imports - now using full logits directly
from pytorch_models.model_loader import get_deepseek_r1
from pytorch_models.analysis import analyze_text, extract_attention_and_logits

# Import existing helper function
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from prep_suppression_mtxs import get_all_problem_numbers


def analyze_component_intervention(boundaries, baseline_logits, text, model_name, device_map, n_tokens, token_ids, cumulative=False, amplify=False, amplify_factor=2.0):
    """
    Run intervention analysis on a list of token boundaries using full logits tensors.

    Args:
        boundaries: List of (start_token, end_token) tuples defining components to intervene on
        baseline_logits: Full logits tensor (seq_len, vocab_size) from model without intervention
        text: Full input text
        model_name: Model identifier (e.g., "llama-8b")
        device_map: Device mapping for model loading
        n_tokens: Total number of tokens in the sequence
        token_ids: List of actual token IDs in the sequence
        cumulative: If True, intervene on all tokens from 0 to end of component i
                   If False, intervene only on tokens within component i
        amplify: If True, amplify attention instead of suppressing
        amplify_factor: Factor to amplify attention by

    Returns:
        Dict containing:
            - kl_matrix_t1/t06: (n_components, n_tokens) KL divergence at each position
            - tv_matrix_t1/t06: (n_components, n_tokens) Total variation distance
            - prob_after_t1/t06: (n_components, n_tokens) Probability of actual tokens after intervention
    """
    n_components = len(boundaries)
    results = {key: np.full((n_components, n_tokens), np.nan) for key in
               ['kl_matrix_t1', 'kl_matrix_t06', 'tv_matrix_t1', 'tv_matrix_t06', 'prob_after_t1', 'prob_after_t06']}

    layers, heads = model2layers_heads(model_name)
    layers_to_mask = {i: list(range(heads)) for i in range(layers)}

    for i, (start, end) in enumerate(tqdm(boundaries, desc="Processing components", leave=False)):
        token_range = [0 if cumulative else start, end]

        # Get full logits directly (suppress verbose output)
        intervened_result = analyze_text(
            text=text, model_name=model_name,
            return_logits=True, device_map=device_map,
            token_range_to_mask=token_range, layers_to_mask=layers_to_mask,
            amplify=amplify, amplify_factor=amplify_factor,
            verbose=False
        )
        intervened_logits = intervened_result["logits"]  # (seq_len, vocab_size)

        # Vectorized KL/TV calculation for both temperatures
        kl_t1 = calculate_kl_divergence_full_tensor(baseline_logits, intervened_logits, temperature=1.0)
        kl_t06 = calculate_kl_divergence_full_tensor(baseline_logits, intervened_logits, temperature=0.6)
        tv_t1 = calculate_tv_distance_full_tensor(baseline_logits, intervened_logits, temperature=1.0)
        tv_t06 = calculate_tv_distance_full_tensor(baseline_logits, intervened_logits, temperature=0.6)

        # Extract token probabilities after intervention
        prob_after_t1 = extract_token_probabilities_full(intervened_logits, token_ids, temperature=1.0)
        prob_after_t06 = extract_token_probabilities_full(intervened_logits, token_ids, temperature=0.6)

        # Store results - shift KL/TV by 1 to align with token positions
        assert kl_t1.shape[0] == n_tokens, f"KL shape {kl_t1.shape[0]} != {n_tokens}"

        # Shift KL/TV indices by 1: kl_t1[0] (predicting token 1) goes to position 1
        results['kl_matrix_t1'][i, 1:] = kl_t1[:-1].cpu().numpy()  # Drop last prediction
        results['kl_matrix_t06'][i, 1:] = kl_t06[:-1].cpu().numpy()
        results['tv_matrix_t1'][i, 1:] = tv_t1[:-1].cpu().numpy()
        results['tv_matrix_t06'][i, 1:] = tv_t06[:-1].cpu().numpy()
        # Position 0 stays 0 for KL/TV (no prediction for first token)

        # Probabilities stay aligned with token positions
        results['prob_after_t1'][i, :] = prob_after_t1
        results['prob_after_t06'][i, :] = prob_after_t06

    return results


# Removed extract_baseline_probabilities - now using extract_token_probabilities_full directly


@pkld()
def analyze_intervention_detailed(problem_num, model_name="llama-8b", is_correct=True, device_map="auto", cumulative=False, amplify=False, amplify_factor=2.0):
    """
    Analyze attention intervention effects at token level without sentence aggregation.

    Args:
        problem_num: Problem number to analyze
        model_name: Model to use
        is_correct: Whether to use correct solution
        device_map: Device mapping for model
        cumulative: If True, intervene on all tokens up to sentence i (cumulative intervention)
                   If False, intervene only on sentence i (individual intervention)
        amplify: If True, amplify attention instead of suppressing (default: False)
        amplify_factor: Factor to amplify attention by (default: 2.0)

    Returns:
        dict: Analysis results containing:
            - prompt_intervention: dict with matrices for prompt components (kl_matrix_t1/t06, tv_matrix_t1/t06, prob_after_t1/t06)
            - response_intervention: dict with matrices for response sentences (kl_matrix_t1/t06, tv_matrix_t1/t06, prob_after_t1/t06)
            - prob_before_t1: (n_tokens,) - baseline probability of each actual token at T=1.0
            - prob_before_t06: (n_tokens,) - baseline probability of each actual token at T=0.6
            - metadata: dict with sentences, boundaries, prompt info, etc.
    """
    print(f"Loading problem {problem_num} with {model_name}...")

    # Get text and sentence boundaries
    text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    # Get prompt component boundaries
    prompt_info = parse_prompt_components_aggregated(text, model_name)
    prompt_boundaries = prompt_info['prompt_boundaries']
    prompt_names = prompt_info['prompt_names']

    # Get model configuration for proper layer/head masking
    layers, heads = model2layers_heads(model_name)
    layers_to_mask = {i: list(range(heads)) for i in range(layers)}

    n_sentences = len(sentences)
    n_prompt_components = len(prompt_boundaries)

    print(f"  {n_prompt_components} prompt components, {n_sentences} response sentences")

    # Get baseline logits and token info in one call
    print("Computing baseline logits...")
    baseline_result = analyze_text(
        text=text,
        model_name=model_name,
        return_logits=True,
        device_map=device_map
    )
    baseline_logits = baseline_result["logits"]  # (seq_len, vocab_size)
    token_ids = baseline_result["tokens"]
    n_tokens = len(token_ids)

    # Extract baseline probabilities for actual tokens at both temperatures
    prob_before_t1 = extract_token_probabilities_full(baseline_logits, token_ids, temperature=1.0)
    prob_before_t06 = extract_token_probabilities_full(baseline_logits, token_ids, temperature=0.6)

    print("Starting intervention analysis...")

    # Run prompt component analysis
    print("Analyzing prompt components...")
    prompt_results = analyze_component_intervention(
        prompt_boundaries, baseline_logits, text, model_name,
        device_map, n_tokens, token_ids, cumulative, amplify, amplify_factor
    )

    # Run response sentence analysis
    print("Analyzing response sentences...")
    response_results = analyze_component_intervention(
        sentence_boundaries, baseline_logits, text, model_name,
        device_map, n_tokens, token_ids, cumulative, amplify, amplify_factor
    )


    # Clean up GPU memory
    torch.cuda.empty_cache()

    # Extract prompt component texts
    prompt_texts = [prompt_info['prompt_prefix']['text']]
    for sentence_info in prompt_info['problem_sentences']:
        prompt_texts.append(sentence_info['text'])
    prompt_texts.append(prompt_info['cot_prefix']['text'])

    metadata = {
        'sentences': sentences,
        'sentence_boundaries': sentence_boundaries,
        'prompt_names': prompt_names,
        'prompt_boundaries': prompt_boundaries,
        'prompt_texts': prompt_texts,
        'token_ids': token_ids,
        'problem_num': problem_num,
        'model_name': model_name,
        'is_correct': is_correct
    }

    print("Analysis complete!")
    return {
        'prompt_intervention': prompt_results,
        'response_intervention': response_results,
        'prob_before_t1': prob_before_t1,
        'prob_before_t06': prob_before_t06,
        'metadata': metadata
    }


# Results are automatically cached by @pkld decorator


def run_multiple_problems(problem_nums, model_name="llama-8b", is_correct=True, device_map="auto"):
    """Run analysis on multiple problems."""
    results = {}

    for i, problem_num in enumerate(problem_nums):
        print(f"\n{'='*60}")
        print(f"Processing problem {problem_num} ({i+1}/{len(problem_nums)})")
        print(f"{'='*60}")

        try:
            result = analyze_suppression_detailed(
                problem_num=problem_num,
                model_name=model_name,
                is_correct=is_correct,
                device_map=device_map
            )
            results[problem_num] = result

            print(f"  ✓ Completed: {result['response_suppression']['kl_matrix_t1'].shape[0]} response sentences, {result['prompt_suppression']['kl_matrix_t1'].shape[0]} prompt components, {result['response_suppression']['kl_matrix_t1'].shape[1]} tokens")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[problem_num] = None

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed attention intervention analysis")
    parser.add_argument("--model-name", type=str, default="llama-8b", help="Model name")
    parser.add_argument("--max-problems", type=int, default=-1, help="Limit number of problems (-1 for all)")
    parser.add_argument("--cumulative", action="store_true", help="Intervene on all tokens up to sentence i (instead of just sentence i)")
    parser.add_argument("--correct-only", action="store_true", help="Only process correct solutions")
    parser.add_argument("--incorrect-only", action="store_true", help="Only process incorrect solutions")
    parser.add_argument("--amplify", action="store_true", help="Amplify attention instead of suppressing")
    parser.add_argument("--amplify-factor", type=float, default=2.0, help="Factor to amplify attention by (default: 2.0)")

    args = parser.parse_args()

    # Validate mutually exclusive flags
    if args.correct_only and args.incorrect_only:
        parser.error("Cannot specify both --correct-only and --incorrect-only")

    # Get all problems using existing helper function
    if args.correct_only:
        all_problems = get_all_problem_numbers(args.model_name, include_incorrect=False)
        print(f"Processing CORRECT solutions only")
    elif args.incorrect_only:
        all_problems = get_all_problem_numbers(args.model_name, include_incorrect=True)
        # Filter to only incorrect solutions
        all_problems = [(prob_num, is_correct) for prob_num, is_correct in all_problems if not is_correct]
        print(f"Processing INCORRECT solutions only")
    else:
        all_problems = get_all_problem_numbers(args.model_name, include_incorrect=True)
        print(f"Processing both correct and incorrect solutions")

    if args.max_problems > 0:
        all_problems = all_problems[:args.max_problems]

    print(f"Found {len(all_problems)} problems to process")

    # Run analysis on all problems
    results = {}
    successful = 0

    for i, (problem_num, is_correct) in enumerate(all_problems):
        status = "correct" if is_correct else "incorrect"
        print(f"\n{'='*60}")
        print(f"Processing problem {problem_num} ({status}) - {i+1}/{len(all_problems)}")
        print(f"{'='*60}")

        try:
            result = analyze_intervention_detailed(
                problem_num=problem_num,
                model_name=args.model_name,
                is_correct=is_correct,
                device_map="auto",
                cumulative=args.cumulative,
                amplify=args.amplify,
                amplify_factor=args.amplify_factor
            )
            results[(problem_num, is_correct)] = result
            successful += 1

            print(f"  ✓ Completed: {result['response_intervention']['kl_matrix_t1'].shape[0]} response sentences, {result['prompt_intervention']['kl_matrix_t1'].shape[0]} prompt components, {result['response_intervention']['kl_matrix_t1'].shape[1]} tokens")

        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")
            results[(problem_num, is_correct)] = None
    print(f"\n{'='*60}")
    print(f"Completed {successful}/{len(all_problems)} problems successfully")

    print(f"\nResults automatically cached by @pkld decorator")