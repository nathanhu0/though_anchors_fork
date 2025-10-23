#!/usr/bin/env python3
"""
Example usage of the text visualization utilities.
"""

import sys
import os
sys.path.append('..')
sys.path.append('../whitebox-analyses')
sys.path.append('../whitebox-analyses/nathan_scripts')

from text_viz import visualize_result
from masking_utils import create_causal_mask_and_distances
from detailed_suppression_analysis import analyze_suppression_detailed

def quick_viz(problem_num, is_correct=True, metric='kl_matrix_t1'):
    """Create visualization for a problem quickly."""

    # Load analysis result
    result = analyze_suppression_detailed(
        problem_num=problem_num,
        model_name="llama-8b",
        is_correct=is_correct,
        cumulative=False
    )

    # Add masking info
    masking_info = create_causal_mask_and_distances(result)
    result.update(masking_info)

    # Create visualization
    output_file = f"problem_{problem_num}_{'correct' if is_correct else 'incorrect'}_{metric}.html"
    visualize_result(result, output_path=output_file, metric=metric)

    print(f"Visualization created: {output_file}")
    print(f"Open in browser: file://{os.path.abspath(output_file)}")

    return output_file

if __name__ == "__main__":
    # Example: visualize problem 1591 with KL divergence
    quick_viz(1591, is_correct=True, metric='kl_matrix_t1')

    # Example: visualize with total variation instead
    # quick_viz(1591, is_correct=True, metric='tv_matrix_t1')