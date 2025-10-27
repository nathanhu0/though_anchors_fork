import os
import sys
import re
import numpy as np
import torch
import torch.nn.functional as F
from typing import List, Tuple, Dict

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from attention_analysis.attn_supp_funcs import get_raw_tokens

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

    # Create simple boundary list for suppression analysis
    prompt_boundaries = [
        (0, prompt_prefix_tokens),  # prompt_prefix
    ]
    prompt_names = ['prompt_prefix']

    # Add problem sentences
    for i, sentence_info in enumerate(problem_sentence_boundaries):
        prompt_boundaries.append(sentence_info['token_range'])
        prompt_names.append(f'problem_sentence_{i}')

    # Add cot_prefix
    prompt_boundaries.append((cot_start_token, cot_end_token))
    prompt_names.append('cot_prefix')

    return {
        'prompt_boundaries': prompt_boundaries,
        'prompt_names': prompt_names,
        'response_start_token': cot_end_token,
        'total_tokens': len(tokens),
        # Keep detailed info for other uses
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
        }
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


def extract_baseline_probabilities(baseline_compressed_logits, token_ids):
    """
    Extract baseline probabilities for actual tokens at both temperatures.

    Args:
        baseline_compressed_logits: Compressed logits from model without suppression
        token_ids: List of actual token IDs in the sequence

    Returns:
        Tuple of (prob_before_t1, prob_before_t06) arrays
    """
    import torch
    from attention_analysis.logits_funcs import decompress_logits_for_position

    n_tokens = len(token_ids)
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

    return prob_before_t1, prob_before_t06


def calculate_kl_divergence_full_tensor(baseline_logits, suppressed_logits, temperature=0.6):
    """KL(baseline || suppressed) at each position. Returns (seq_len,) tensor."""
    # Convert to tensors if needed
    if isinstance(baseline_logits, np.ndarray):
        baseline_logits = torch.from_numpy(baseline_logits)
    if isinstance(suppressed_logits, np.ndarray):
        suppressed_logits = torch.from_numpy(suppressed_logits)

    # Remove batch dimension if present
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits.squeeze(0)
    if suppressed_logits.dim() == 3:
        suppressed_logits = suppressed_logits.squeeze(0)

    log_p = F.log_softmax(baseline_logits / temperature, dim=-1)
    log_q = F.log_softmax(suppressed_logits / temperature, dim=-1)
    return F.kl_div(log_q, log_p, reduction='none', log_target=True).sum(dim=-1)


def calculate_tv_distance_full_tensor(baseline_logits, suppressed_logits, temperature=0.6):
    """TV distance at each position. Returns (seq_len,) tensor."""
    # Convert to tensors if needed
    if isinstance(baseline_logits, np.ndarray):
        baseline_logits = torch.from_numpy(baseline_logits)
    if isinstance(suppressed_logits, np.ndarray):
        suppressed_logits = torch.from_numpy(suppressed_logits)

    # Remove batch dimension if present
    if baseline_logits.dim() == 3:
        baseline_logits = baseline_logits.squeeze(0)
    if suppressed_logits.dim() == 3:
        suppressed_logits = suppressed_logits.squeeze(0)

    p = F.softmax(baseline_logits / temperature, dim=-1)
    q = F.softmax(suppressed_logits / temperature, dim=-1)
    return 0.5 * (p - q).abs().sum(dim=-1)


def extract_token_probabilities_full(logits, token_ids, temperature=1.0):
    """Extract actual token probabilities at given temperature. Returns (n_tokens,) array."""
    # Convert to tensor if needed
    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)

    # Remove batch dimension if present
    if logits.dim() == 3:
        logits = logits.squeeze(0)  # [1, seq_len, vocab_size] -> [seq_len, vocab_size]

    probs = F.softmax(logits / temperature, dim=-1)  # (seq_len, vocab_size)
    token_probs = torch.zeros(len(token_ids))

    # Position i in logits predicts token_ids[i+1], but we want prob of token_ids[i]
    # So position i-1 predicts token_ids[i]
    for i, token_id in enumerate(token_ids):
        if i == 0:
            token_probs[i] = float('nan')  # No prediction for first token
        else:
            # Ensure token_id is a scalar integer
            if isinstance(token_id, (torch.Tensor, np.ndarray)):
                token_id = int(token_id.item())
            token_probs[i] = probs[i-1, token_id].item()

    return token_probs.numpy()


