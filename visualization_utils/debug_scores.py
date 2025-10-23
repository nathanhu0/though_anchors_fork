#!/usr/bin/env python3

import sys
sys.path.append('..')
sys.path.append('../whitebox-analyses')
sys.path.append('../whitebox-analyses/nathan_scripts')

from text_viz import create_sentence_interaction_matrix, calculate_dependency_scores
from masking_utils import create_causal_mask_and_distances
from detailed_suppression_analysis import analyze_suppression_detailed
import numpy as np

def debug_scores(problem_num=1591, min_distance=3):
    """Debug the score calculations to verify hover logic."""

    # Load data
    result = analyze_suppression_detailed(
        problem_num=problem_num,
        model_name="llama-8b",
        is_correct=True,
        cumulative=False
    )

    # Add masking info
    masking_info = create_causal_mask_and_distances(result)
    result.update(masking_info)

    # Calculate interaction matrix and scores
    interaction_matrix = create_sentence_interaction_matrix(result)
    dependency_scores = calculate_dependency_scores(interaction_matrix, min_distance)

    print(f"Interaction matrix shape: {interaction_matrix.shape}")
    print(f"Dependency scores shape: {dependency_scores.shape}")

    # Find the sentence with highest dependency
    max_dep_idx = np.argmax(dependency_scores)
    max_dep_value = dependency_scores[max_dep_idx]

    print(f"\nSentence {max_dep_idx} has highest dependency: {max_dep_value:.6f}")

    # Find which earlier sentence contributed this max value
    best_source = -1
    best_value = -1
    for source_sent in range(max_dep_idx):
        if max_dep_idx - source_sent >= min_distance:
            effect = interaction_matrix[source_sent, max_dep_idx]
            if effect > best_value:
                best_value = effect
                best_source = source_sent

    print(f"This came from sentence {best_source} -> {max_dep_idx}: {best_value:.6f}")
    print(f"Match check: {abs(max_dep_value - best_value) < 1e-10}")

    # Show a few values for verification
    print(f"\nSample interaction values (first 5 -> sentence {max_dep_idx}):")
    for i in range(min(5, max_dep_idx)):
        val = interaction_matrix[i, max_dep_idx]
        print(f"  {i} -> {max_dep_idx}: {val:.6f}")

    return interaction_matrix, dependency_scores

if __name__ == "__main__":
    debug_scores()