#!/usr/bin/env python3

import sys
import os
sys.path.append('..')
sys.path.append('../whitebox-analyses')
sys.path.append('../whitebox-analyses/nathan_scripts')

from text_viz import visualize_result
from detailed_suppression_analysis import analyze_suppression_detailed

def test_visualization(problem_num=1591, model_name="llama-8b", is_correct=True):
    """Test the visualization with cached data."""

    print(f"Loading analysis for problem {problem_num}...")

    # Load the analysis result (should be cached)
    result = analyze_suppression_detailed(
        problem_num=problem_num,
        model_name=model_name,
        is_correct=is_correct,
        device_map="auto",
        cumulative=False
    )

    # Add the masking info that was in your notebook
    from masking_utils import create_causal_mask_and_distances
    masking_info = create_causal_mask_and_distances(result)
    result.update(masking_info)

    print(f"Data loaded: {len(result['metadata']['sentences'])} sentences")

    # Create visualization
    output_path = f"problem_{problem_num}_viz.html"
    visualize_result(result, output_path=output_path, metric='kl_matrix_t1')

    print(f"Visualization saved to: {output_path}")
    print(f"Open in browser: file://{os.path.abspath(output_path)}")

    return output_path

if __name__ == "__main__":
    test_visualization()