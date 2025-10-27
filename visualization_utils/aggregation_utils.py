import numpy as np
import sys
import os
from typing import Dict, Any, List, Tuple, Optional
from tqdm import tqdm
# Add parent directories to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
sys.path.append(os.path.join(parent_dir, 'whitebox-analyses'))
sys.path.append(os.path.join(parent_dir, 'whitebox-analyses', 'nathan_scripts'))

from detailed_interaction_analysis import analyze_intervention_detailed

def aggregate_to_boundaries(
    value_matrix: np.ndarray,
    target_boundaries: List[Tuple[int, int]],
    causal_mask: Optional[np.ndarray] = None,
    aggregation_method: str = 'mean'
) -> np.ndarray:
    """
    Aggregate token-level values to target boundaries (e.g., sentences).

    This is the core aggregation function that works for any source→target mapping.

    Args:
        value_matrix: (n_sources, n_tokens) matrix of token-level values
        target_boundaries: List of (start, end) tuples defining target boundaries
        causal_mask: Optional (n_sources, n_tokens) mask for valid positions
        aggregation_method: How to aggregate within boundaries ('mean', 'sum', 'max')

    Returns:
        (n_sources, n_targets) aggregated interaction matrix
    """
    n_sources, n_tokens = value_matrix.shape
    n_targets = len(target_boundaries)

    interaction_matrix = np.zeros((n_sources, n_targets))

    for source_idx in range(n_sources):
        for target_idx, (start_token, end_token) in enumerate(target_boundaries):
            # Get values for this source→target interaction
            target_values = value_matrix[source_idx, start_token:end_token]

            # Apply causal mask if provided
            if causal_mask is not None:
                causal_valid = causal_mask[source_idx, start_token:end_token]
                valid_mask = causal_valid & ~np.isnan(target_values) & ~np.isinf(target_values)
                valid_values = target_values[valid_mask]
            else:
                valid_mask = ~np.isnan(target_values) & ~np.isinf(target_values)
                valid_values = target_values[valid_mask]

            # Aggregate valid values
            if len(valid_values) > 0:
                if aggregation_method == 'mean':
                    interaction_matrix[source_idx, target_idx] = np.mean(valid_values)
                elif aggregation_method == 'sum':
                    interaction_matrix[source_idx, target_idx] = np.sum(valid_values)
                elif aggregation_method == 'max':
                    interaction_matrix[source_idx, target_idx] = np.max(valid_values)
                else:
                    raise ValueError(f"Unknown aggregation_method: {aggregation_method}")
            else:
                interaction_matrix[source_idx, target_idx] = 0

    return interaction_matrix

def create_sentence_to_sentence_matrix(result: Dict[str, Any], metric: str = 'kl_matrix_t1') -> np.ndarray:
    """Create sentence→sentence interaction matrix using pre-computed results."""
    response_data = result['response_intervention']
    sentence_boundaries = result['metadata']['sentence_boundaries']

    # Use pre-computed matrix without additional causal masking
    # Note: The intervention data already respects causality (sentence i suppressed → measure effects on later tokens)
    # If we need stricter causal masking in the future, we could add it here via the causal_mask parameter
    value_matrix = response_data[metric]
    return aggregate_to_boundaries(value_matrix, sentence_boundaries)

def create_prompt_to_sentence_matrix(result: Dict[str, Any], metric: str = 'kl_matrix_t1') -> np.ndarray:
    """Create prompt→sentence interaction matrix using pre-computed results."""
    prompt_data = result['prompt_intervention']
    sentence_boundaries = result['metadata']['sentence_boundaries']

    # Use pre-computed matrix
    value_matrix = prompt_data[metric]
    return aggregate_to_boundaries(value_matrix, sentence_boundaries)

def calculate_dependency_scores(
    interaction_matrix: np.ndarray,
    min_distance: int = 3,
    interaction_type: str = 'sentence_to_sentence'
) -> np.ndarray:
    """Calculate per-target dependency scores from interaction matrix."""
    n_targets = interaction_matrix.shape[1]
    dependency_scores = np.zeros(n_targets)

    if interaction_type == 'sentence_to_sentence':
        # For each target sentence, find max dependency on far-earlier sentences
        for target_idx in range(n_targets):
            max_dependency = 0
            for source_idx in range(target_idx):
                if target_idx - source_idx >= min_distance:
                    effect = interaction_matrix[source_idx, target_idx]
                    max_dependency = max(max_dependency, effect)
            dependency_scores[target_idx] = max_dependency

    elif interaction_type == 'prompt_to_sentence':
        # For each target sentence, find max dependency on any prompt component
        for target_idx in range(n_targets):
            max_dependency = 0
            for source_idx in range(interaction_matrix.shape[0]):
                effect = interaction_matrix[source_idx, target_idx]
                max_dependency = max(max_dependency, effect)
            dependency_scores[target_idx] = max_dependency

    return dependency_scores

def calculate_effect_scores(
    interaction_matrix: np.ndarray,
    min_distance: int = 3,
    interaction_type: str = 'sentence_to_sentence'
) -> np.ndarray:
    """Calculate per-source effect scores from interaction matrix."""
    n_sources = interaction_matrix.shape[0]
    effect_scores = np.zeros(n_sources)

    if interaction_type == 'sentence_to_sentence':
        # For each source sentence, find max effect on far-future sentences
        for source_idx in range(n_sources):
            max_effect = 0
            actual_min_distance = max(1, min_distance)
            for target_idx in range(source_idx + actual_min_distance, interaction_matrix.shape[1]):
                effect = interaction_matrix[source_idx, target_idx]
                max_effect = max(max_effect, effect)
            effect_scores[source_idx] = max_effect

    elif interaction_type == 'prompt_to_sentence':
        # For each prompt component, find max effect on any reasoning sentence
        for source_idx in range(n_sources):
            max_effect = 0
            for target_idx in range(interaction_matrix.shape[1]):
                effect = interaction_matrix[source_idx, target_idx]
                max_effect = max(max_effect, effect)
            effect_scores[source_idx] = max_effect

    return effect_scores

def get_available_metrics(result: Dict[str, Any]) -> List[str]:
    """Get available metrics from the result."""
    metrics = []

    # Check response intervention data
    response_data = result.get('response_intervention', {})
    for metric in ['kl_matrix_t1', 'kl_matrix_t06', 'tv_matrix_t1', 'tv_matrix_t06']:
        if metric in response_data:
            metrics.append(metric)

    return metrics

def get_available_interaction_types(result: Dict[str, Any]) -> List[str]:
    """Get available interaction types from the result."""
    types = []

    if 'response_intervention' in result:
        types.append('sentence_to_sentence')

    if 'prompt_intervention' in result:
        types.append('prompt_to_sentence')

    return types

def get_sentences_for_interaction_type(
    result: Dict[str, Any],
    interaction_type: str
) -> Tuple[List[str], List[str]]:
    """Get source and target sentences for interaction type."""
    target_sentences = result['metadata']['sentences']  # Always reasoning sentences

    if interaction_type == 'sentence_to_sentence':
        return target_sentences, target_sentences

    elif interaction_type == 'prompt_to_sentence':
        source_sentences = result['metadata']['prompt_texts']
        return source_sentences, target_sentences

    else:
        raise ValueError(f"Unknown interaction_type: {interaction_type}")

def is_intervention_cached(
    problem_num: int,
    model_name: str = "llama-8b",
    is_correct: bool = True,
    cumulative: bool = False,
    amplify: bool = False,
    amplify_factor: float = 2.0,
    device_map: str = "auto"
) -> bool:
    """Check if intervention analysis results are cached."""
    from pkld.utils import get_cache_fp

    # Get the cache filepath using the same parameters as analyze_intervention_detailed
    temp_cache_path = get_cache_fp(
        analyze_intervention_detailed,
        args=(),  # Empty args
        kwargs={
            'problem_num': problem_num,
            'model_name': model_name,
            'is_correct': is_correct,
            'device_map': device_map,
            'cumulative': cumulative,
            'amplify': amplify,
            'amplify_factor': amplify_factor
        },
        max_fn_len=128
    )

    filename = temp_cache_path.name
    correct_cache_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'whitebox-analyses', 'nathan_scripts', '.pkljar',
        'detailed_interaction_analysis', 'analyze_intervention_detailed'
    )
    correct_cache_path = os.path.join(correct_cache_dir, filename)

    return os.path.exists(correct_cache_path)

def load_intervention(
    model_name: str = 'llama-8b',
    problem_num: int = 330,
    cumulative: bool = False,
    is_correct: bool = True,
    amplify: bool = True,
    amplify_factor: float = 5.0
) -> Dict[str, Any]:
    """
    Load intervention data and compute all derived metrics.

    Returns enhanced result with sentence-to-sentence interaction stats for all metrics.
    """
    # Load base intervention data
    result = analyze_intervention_detailed(
        problem_num=problem_num,
        model_name=model_name,
        is_correct=is_correct,
        cumulative=cumulative,
        amplify=amplify,
        amplify_factor=amplify_factor
    )

    # Compute NLL changes from probability data
    prob_before_t1 = result['prob_before_t1']
    prob_before_t06 = result['prob_before_t06']

    # Response NLL changes
    response_prob_after_t1 = result['response_intervention']['prob_after_t1']
    response_prob_after_t06 = result['response_intervention']['prob_after_t06']

    eps = 1e-10
    result['response_intervention']['nll_changes_t1'] = (
        np.log(np.maximum(prob_before_t1, eps))[None, :] -
        np.log(np.maximum(response_prob_after_t1, eps))
    )
    result['response_intervention']['nll_changes_t06'] = (
        np.log(np.maximum(prob_before_t06, eps))[None, :] -
        np.log(np.maximum(response_prob_after_t06, eps))
    )

    # Prompt NLL changes if available
    if 'prompt_intervention' in result:
        prompt_prob_after_t1 = result['prompt_intervention']['prob_after_t1']
        prompt_prob_after_t06 = result['prompt_intervention']['prob_after_t06']

        result['prompt_intervention']['nll_changes_t1'] = (
            np.log(np.maximum(prob_before_t1, eps))[None, :] -
            np.log(np.maximum(prompt_prob_after_t1, eps))
        )
        result['prompt_intervention']['nll_changes_t06'] = (
            np.log(np.maximum(prob_before_t06, eps))[None, :] -
            np.log(np.maximum(prompt_prob_after_t06, eps))
        )

    # Add sentence-to-sentence interaction matrices for all metrics
    sentence_interactions = {}
    for metric in ['kl_matrix_t1', 'kl_matrix_t06', 'tv_matrix_t1', 'tv_matrix_t06', 'nll_changes_t1', 'nll_changes_t06']:
        if metric in result['response_intervention']:
            sentence_interactions[f'sentence_{metric}'] = create_sentence_to_sentence_matrix(result, metric)

    # Add prompt-to-sentence interaction matrices if available
    if 'prompt_intervention' in result:
        for metric in ['kl_matrix_t1', 'kl_matrix_t06', 'tv_matrix_t1', 'tv_matrix_t06', 'nll_changes_t1', 'nll_changes_t06']:
            if metric in result['prompt_intervention']:
                sentence_interactions[f'prompt_{metric}'] = create_prompt_to_sentence_matrix(result, metric)

    result['sentence_interactions'] = sentence_interactions

    return result

def load_all_cached_interventions(
    model_name: str = 'llama-8b',
    include_incorrect: bool = True,
    cumulative: bool = False,
    amplify: bool = False,
    amplify_factor: float = 2.0
) -> Dict[tuple, Dict[str, Any]]:
    """
    Load all cached intervention results for a specific intervention configuration.

    Args:
        model_name: Model to load for
        include_incorrect: Whether to include incorrect solutions
        cumulative: Whether to use cumulative interventions
        amplify: Whether to amplify (True) or suppress (False)
        amplify_factor: Amplification/suppression factor

    Returns:
        Dict with structure: {(problem_num, is_correct): result_dict}
    """
    # Import here to avoid circular imports
    sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'whitebox-analyses', 'scripts'))
    from prep_suppression_mtxs import get_all_problem_numbers

    results = {}

    print(f"Loading intervention type: cumulative={cumulative}, amplify={amplify}, factor={amplify_factor}")

    for problem_num, is_correct in tqdm(get_all_problem_numbers(model_name, include_incorrect)):
        # Check if cached
        if is_intervention_cached(
            problem_num=problem_num,
            model_name=model_name,
            is_correct=is_correct,
            cumulative=cumulative,
            amplify=amplify,
            amplify_factor=amplify_factor
        ):
            try:
                result = load_intervention(
                    model_name=model_name,
                    problem_num=problem_num,
                    cumulative=cumulative,
                    is_correct=is_correct,
                    amplify=amplify,
                    amplify_factor=amplify_factor
                )
                problem_key = (problem_num, is_correct)
                results[problem_key] = result

            except Exception as e:
                print(f"Error loading problem {problem_num}, is_correct={is_correct}: {e}")

    print(f"Loaded {len(results)} problems")
    return results

