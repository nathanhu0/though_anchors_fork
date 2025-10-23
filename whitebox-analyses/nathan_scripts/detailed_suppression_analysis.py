"""
Detailed suppression analysis with token-level granularity.
Returns probability changes and KL divergences at token level without sentence aggregation.
"""

import os
import sys
import numpy as np
import torch
from pathlib import Path
from typing import Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pkld import pkld
from attention_analysis.receiver_head_funcs import get_problem_text_sentences, get_model_rollouts_root
from attention_analysis.attn_supp_funcs import get_sentence_token_boundaries, get_raw_tokens, calculate_kl_divergence_sparse
from pytorch_models.model_config import model2layers_heads
from attention_analysis.logits_funcs import analyze_text_get_p_logits, decompress_logits_for_position, compress_logits_top_p
from pytorch_models.model_loader import get_deepseek_r1
from pytorch_models.analysis import analyze_text, extract_attention_and_logits

# Import existing helper function
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))
from prep_suppression_mtxs import get_all_problem_numbers


def calculate_tv_distance_sparse(
    baseline_data: Tuple[np.ndarray, np.ndarray],
    suppressed_data: Tuple[np.ndarray, np.ndarray],
    temperature: float = 0.6,
) -> float:
    """
    Calculate total variation distance between two sparse probability distributions.

    Args:
        baseline_data: Tuple of (indices, logits) for baseline distribution
        suppressed_data: Tuple of (indices, logits) for suppressed distribution
        temperature: Temperature for softmax conversion

    Returns:
        float: Total variation distance (0.5 * sum(|p - q|))
    """
    b_idxs, b_logits = baseline_data
    s_idxs, s_logits = suppressed_data

    if b_idxs is None or b_logits is None or s_idxs is None or s_logits is None:
        return np.nan

    # Match calculate_kl_divergence_sparse exactly
    import torch.nn.functional as F

    # Ensure logits are float32 for stable softmax
    b_logits = b_logits.astype(np.float32)
    s_logits = s_logits.astype(np.float32)

    union_indices = np.union1d(b_idxs, s_idxs)
    union_size = len(union_indices)

    idx_to_union_pos = {idx: pos for pos, idx in enumerate(union_indices)}

    min_logit_val = -1e9  # Approx -inf for softmax
    b_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)
    s_logits_union = np.full(union_size, min_logit_val, dtype=np.float32)

    for idx, logit in zip(b_idxs, b_logits):
        b_logits_union[idx_to_union_pos[idx]] = logit
    for idx, logit in zip(s_idxs, s_logits):
        s_logits_union[idx_to_union_pos[idx]] = logit

    b_logits_tensor = torch.from_numpy(b_logits_union)
    s_logits_tensor = torch.from_numpy(s_logits_union)

    # Use log_softmax then exp like KL function does
    log_p = F.log_softmax(b_logits_tensor / temperature, dim=0)
    log_q = F.log_softmax(s_logits_tensor / temperature, dim=0)

    # Check for issues like KL function
    if (
        torch.isinf(log_p).any()
        or torch.isnan(log_p).any()
        or torch.isinf(log_q).any()
        or torch.isnan(log_q).any()
    ):
        return np.nan

    p_dist = torch.exp(log_p)  # Equivalent to softmax(logits/T)
    q_dist = torch.exp(log_q)

    if torch.isnan(p_dist).any() or torch.isnan(q_dist).any():
        return np.nan

    # Total variation distance: 0.5 * sum(|p - q|)
    tv_terms = torch.abs(p_dist - q_dist)
    tv = 0.5 * torch.sum(tv_terms)

    if torch.isnan(tv):
        return np.nan

    return float(tv.item())


def get_compressed_logits_with_preloaded_model(
    model, tokenizer, text, model_name,
    token_range_to_mask=None, layers_to_mask=None,
    p_nucleus=0.9999, max_k=100
):
    """Get compressed logits using a pre-loaded model."""
    result = extract_attention_and_logits(
        model=model,
        tokenizer=tokenizer,
        text=text,
        model_name=model_name,
        return_logits=True,
        token_range_to_mask=token_range_to_mask,
        mask_layers=layers_to_mask,
    )

    logits = result["logits"]
    # Apply the same compression as analyze_text_get_p_logits
    compressed = compress_logits_top_p(logits, p_nucleus, max_k=max_k)
    return compressed


@pkld
def analyze_suppression_detailed(problem_num, model_name="llama-8b", is_correct=True, device_map="auto", cumulative=False):
    """
    Analyze suppression effects at token level without sentence aggregation.

    Args:
        problem_num: Problem number to analyze
        model_name: Model to use
        is_correct: Whether to use correct solution
        device_map: Device mapping for model
        cumulative: If True, suppress all tokens up to sentence i (cumulative suppression)
                   If False, suppress only sentence i (individual suppression)

    Returns:
        dict: Analysis results containing:
            - kl_matrix_t1: (n_sentences, n_tokens) - KL divergence at T=1.0 per token when each sentence suppressed
            - kl_matrix_t06: (n_sentences, n_tokens) - KL divergence at T=0.6 per token when each sentence suppressed
            - tv_matrix_t1: (n_sentences, n_tokens) - Total variation at T=1.0 per token when each sentence suppressed
            - tv_matrix_t06: (n_sentences, n_tokens) - Total variation at T=0.6 per token when each sentence suppressed
            - prob_before_t1: (n_tokens,) - baseline probability of each actual token at T=1.0
            - prob_before_t06: (n_tokens,) - baseline probability of each actual token at T=0.6
            - prob_after_t1: (n_sentences, n_tokens) - probability of each actual token when sentences suppressed at T=1.0
            - prob_after_t06: (n_sentences, n_tokens) - probability of each actual token when sentences suppressed at T=0.6
            - metadata: dict with sentences, boundaries, etc.
    """
    print(f"Loading problem {problem_num} with {model_name}...")

    # Get text and sentence boundaries
    text, sentences = get_problem_text_sentences(problem_num, is_correct, model_name)
    sentence_boundaries = get_sentence_token_boundaries(text, sentences, model_name)

    # Get model configuration for proper layer/head masking
    layers, heads = model2layers_heads(model_name)
    layers_to_mask = {i: list(range(heads)) for i in range(layers)}

    n_sentences = len(sentences)

    print(f"  {n_sentences} sentences")

    # Get baseline probabilities using existing functions
    print("Computing baseline probabilities...")
    baseline_compressed_logits = analyze_text_get_p_logits(
        text=text,
        model_name=model_name,
        p_nucleus=0.9999,
        device_map=device_map
    )

    # Get token information from a separate call to get the token IDs
    token_info = analyze_text(
        text=text,
        model_name=model_name,
        return_logits=False,
        device_map=device_map
    )
    token_ids = token_info["tokens"]
    n_tokens = len(token_ids)

    # Extract probabilities from compressed logits for actual tokens at both temperatures
    # Note: logits at position i predict token at position i+1
    prob_before_t1 = np.zeros(n_tokens)
    prob_before_t06 = np.zeros(n_tokens)

    for i, token_id in enumerate(token_ids):
        if i == 0:
            # First token has no previous position to predict from
            prob_before_t1[i] = np.nan
            prob_before_t06[i] = np.nan
            continue

        # Decompress logits from position i-1 to get prediction for token i
        indices, logits = decompress_logits_for_position(baseline_compressed_logits, i-1)
        if indices is not None and token_id in indices:
            # Find the index of our token in the compressed data
            token_idx = np.where(indices == token_id)[0]
            if len(token_idx) > 0:
                # Convert logit to probability using softmax at both temperatures
                logits_tensor = torch.from_numpy(logits)
                probs_t1 = torch.softmax(logits_tensor / 1.0, dim=0)
                probs_t06 = torch.softmax(logits_tensor / 0.6, dim=0)

                prob_before_t1[i] = probs_t1[token_idx[0]].item()
                prob_before_t06[i] = probs_t06[token_idx[0]].item()
        # If token not found in top-p, probability is very small (essentially 0)

    # Initialize result matrices for both temperatures
    kl_matrix_t1 = np.zeros((n_sentences, n_tokens))  # T=1.0 (raw)
    kl_matrix_t06 = np.zeros((n_sentences, n_tokens))  # T=0.6 (sampling)
    tv_matrix_t1 = np.zeros((n_sentences, n_tokens))  # T=1.0
    tv_matrix_t06 = np.zeros((n_sentences, n_tokens))  # T=0.6
    prob_after_t1 = np.zeros((n_sentences, n_tokens))  # Probability of actual tokens at T=1.0
    prob_after_t06 = np.zeros((n_sentences, n_tokens))  # Probability of actual tokens at T=0.6

    print("Starting suppression analysis...")

    for sent_idx in range(n_sentences):
        print(f"  Suppressing sentence {sent_idx+1}/{n_sentences}")

        # Get token range based on suppression mode
        if sent_idx < len(sentence_boundaries):
            if cumulative:
                # Suppress all tokens from start up to END of sentence i
                token_start = 0
                _, token_end = sentence_boundaries[sent_idx]
                token_range = list(range(token_start, token_end))
                print(f"    Suppressing tokens 0-{token_end} (cumulative up to sentence {sent_idx})")
            else:
                # Suppress only sentence i
                token_start, token_end = sentence_boundaries[sent_idx]
                token_range = list(range(token_start, token_end))
                print(f"    Suppressing tokens {token_start}-{token_end} (sentence {sent_idx} only)")
        else:
            print(f"    Warning: No boundary for sentence {sent_idx}, skipping")
            continue

        # Get suppressed probabilities using existing functions
        suppressed_compressed_logits = analyze_text_get_p_logits(
            text=text,
            model_name=model_name,
            p_nucleus=0.9999,
            token_range_to_mask=token_range,
            layers_to_mask=layers_to_mask,
            device_map=device_map
        )

        # Extract probabilities only for the actual tokens and store at both temperatures
        for token_pos, token_id in enumerate(token_ids):
            if token_pos == 0:
                # First token has no previous position to predict from
                prob_after_t1[sent_idx, token_pos] = np.nan
                prob_after_t06[sent_idx, token_pos] = np.nan
                continue

            if token_pos < n_tokens:
                # Decompress logits from position token_pos-1 to get prediction for token_pos
                indices, logits = decompress_logits_for_position(suppressed_compressed_logits, token_pos-1)
                if indices is not None and token_id in indices:
                    # Find the index of our token in the compressed data
                    token_idx = np.where(indices == token_id)[0]
                    if len(token_idx) > 0:
                        # Convert logit to probability using softmax at both temperatures
                        logits_tensor = torch.from_numpy(logits)
                        probs_t1 = torch.softmax(logits_tensor / 1.0, dim=0)
                        probs_t06 = torch.softmax(logits_tensor / 0.6, dim=0)

                        prob_after_t1[sent_idx, token_pos] = probs_t1[token_idx[0]].item()
                        prob_after_t06[sent_idx, token_pos] = probs_t06[token_idx[0]].item()
                # If token not found in top-p, probability is very small (essentially 0)

        # Calculate KL divergence and TV distance for each token position
        for token_pos in range(n_tokens):
            # Get compressed logits for this position from both baseline and suppressed
            baseline_indices, baseline_logits = decompress_logits_for_position(baseline_compressed_logits, token_pos)
            suppressed_indices, suppressed_logits = decompress_logits_for_position(suppressed_compressed_logits, token_pos)

            if baseline_indices is not None and suppressed_indices is not None:
                baseline_data = (baseline_indices, baseline_logits)
                suppressed_data = (suppressed_indices, suppressed_logits)

                # Compute KL and TV at both temperatures
                # T=1.0 (raw model distributions)
                kl_div_t1 = calculate_kl_divergence_sparse(baseline_data, suppressed_data, temperature=1.0)
                tv_dist_t1 = calculate_tv_distance_sparse(baseline_data, suppressed_data, temperature=1.0)

                # T=0.6 (sampling temperature)
                kl_div_t06 = calculate_kl_divergence_sparse(baseline_data, suppressed_data, temperature=0.6)
                tv_dist_t06 = calculate_tv_distance_sparse(baseline_data, suppressed_data, temperature=0.6)

                kl_matrix_t1[sent_idx, token_pos] = kl_div_t1
                tv_matrix_t1[sent_idx, token_pos] = tv_dist_t1
                kl_matrix_t06[sent_idx, token_pos] = kl_div_t06
                tv_matrix_t06[sent_idx, token_pos] = tv_dist_t06
            else:
                kl_matrix_t1[sent_idx, token_pos] = np.nan
                tv_matrix_t1[sent_idx, token_pos] = np.nan
                kl_matrix_t06[sent_idx, token_pos] = np.nan
                tv_matrix_t06[sent_idx, token_pos] = np.nan

    # Clean up GPU memory
    torch.cuda.empty_cache()

    metadata = {
        'sentences': sentences,
        'sentence_boundaries': sentence_boundaries,
        'token_ids': token_ids,
        'problem_num': problem_num,
        'model_name': model_name,
        'is_correct': is_correct
    }

    print("Analysis complete!")
    return {
        'kl_matrix_t1': kl_matrix_t1,
        'kl_matrix_t06': kl_matrix_t06,
        'tv_matrix_t1': tv_matrix_t1,
        'tv_matrix_t06': tv_matrix_t06,
        'prob_before_t1': prob_before_t1,
        'prob_before_t06': prob_before_t06,
        'prob_after_t1': prob_after_t1,
        'prob_after_t06': prob_after_t06,
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

            print(f"  ✓ Completed: {result['kl_matrix_t1'].shape[0]} sentences, {result['kl_matrix_t1'].shape[1]} tokens")

        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[problem_num] = None

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Detailed suppression analysis")
    parser.add_argument("--model-name", type=str, default="llama-8b", help="Model name")
    parser.add_argument("--max-problems", type=int, default=-1, help="Limit number of problems (-1 for all)")
    parser.add_argument("--cumulative", action="store_true", help="Suppress all tokens up to sentence i (instead of just sentence i)")
    parser.add_argument("--correct-only", action="store_true", help="Only process correct solutions")
    parser.add_argument("--incorrect-only", action="store_true", help="Only process incorrect solutions")

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
            result = analyze_suppression_detailed(
                problem_num=problem_num,
                model_name=args.model_name,
                is_correct=is_correct,
                device_map="auto",
                cumulative=args.cumulative
            )
            results[(problem_num, is_correct)] = result
            successful += 1

            print(f"  ✓ Completed: {result['kl_matrix_t1'].shape[0]} sentences, {result['kl_matrix_t1'].shape[1]} tokens")

        except Exception as e:
            print(f"  ✗ Failed: {type(e).__name__}: {e}")
            results[(problem_num, is_correct)] = None

    print(f"\n{'='*60}")
    print(f"Completed {successful}/{len(all_problems)} problems successfully")

    print(f"\nResults automatically cached by @pkld decorator")