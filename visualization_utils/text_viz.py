import numpy as np
import json
import os
import sys
from typing import List, Dict, Any, Tuple, Optional

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from visualization_utils.aggregation_utils import (
    create_sentence_to_sentence_matrix,
    create_prompt_to_sentence_matrix,
    calculate_dependency_scores,
    calculate_effect_scores,
    get_sentences_for_interaction_type,
    get_available_metrics,
    get_available_interaction_types
)


def extract_problem_text_from_prompt(prompt_texts: List[str], full_prompt: str = "") -> str:
    """Extract problem text from prompt components or full prompt."""
    if prompt_texts and len(prompt_texts) > 1:
        # Usually the problem text is in the problem_sentences (after prompt_prefix)
        # Skip the first element (prompt prefix) and last element (cot prefix)
        problem_parts = prompt_texts[1:-1] if len(prompt_texts) > 2 else prompt_texts[1:]
        return " ".join(problem_parts)
    elif full_prompt:
        # Try to extract from full prompt if available
        # Look for text between common markers
        if "Problem:" in full_prompt:
            start = full_prompt.find("Problem:") + len("Problem:")
            end = full_prompt.find("\n\n", start) if "\n\n" in full_prompt[start:] else len(full_prompt)
            return full_prompt[start:end].strip()
        # Otherwise return first part before "Let's think"
        elif "Let's think" in full_prompt:
            return full_prompt[:full_prompt.find("Let's think")].strip()
    return "Problem text not available in metadata"

def normalize_scores(scores: np.ndarray, clip_percentile: float = 95) -> np.ndarray:
    """Normalize scores to 0-1 range with optional clipping of outliers."""
    if len(scores) == 0:
        return scores

    # Clip extreme outliers
    upper_clip = np.percentile(scores, clip_percentile)
    scores_clipped = np.clip(scores, 0, upper_clip)

    # Normalize to 0-1
    score_min, score_max = scores_clipped.min(), scores_clipped.max()
    if score_max > score_min:
        return (scores_clipped - score_min) / (score_max - score_min)
    else:
        return np.zeros_like(scores_clipped)

def create_html_visualization(sentences: List[str],
                            dependency_scores: np.ndarray,
                            effect_scores: np.ndarray,
                            sentence_matrix: np.ndarray,
                            title: str = "Sentence Dependencies",
                            problem_info: Dict[str, Any] = None,
                            min_distance: int = 3,
                            prompt_matrix: Optional[np.ndarray] = None) -> str:
    """
    Create HTML visualization with hover interactions and mode switching.

    Default mode: sentence-to-sentence interactions
    Hover mode: prompt-to-sentence effects (if prompt_matrix provided)

    Args:
        sentences: List of reasoning sentence strings (what we visualize)
        dependency_scores: Sentence dependency scores (from sentence_matrix)
        effect_scores: Sentence effect scores (from sentence_matrix)
        sentence_matrix: (n_sentences, n_sentences) sentence-to-sentence matrix
        title: Title for the visualization
        problem_info: Dict with problem metadata
        min_distance: Distance threshold for far dependencies
        prompt_matrix: Optional (n_prompt_components, n_sentences) prompt-to-sentence matrix

    Returns:
        HTML string
    """
    n_sentences = len(sentences)

    # Calculate global max for color scaling from all matrices
    all_values = [dependency_scores, effect_scores, sentence_matrix.flatten()]
    if prompt_matrix is not None:
        all_values.append(prompt_matrix.flatten())

    all_values = np.concatenate(all_values)
    valid_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]
    positive_values = valid_values[valid_values > 0]
    global_min = 0  # Always start from 0 (white)
    global_max = np.max(positive_values) if len(positive_values) > 0 else 1

    # Convert to JSON for JavaScript
    sentences_json = json.dumps(sentences)
    dependency_scores_json = json.dumps(dependency_scores.tolist())
    effect_scores_json = json.dumps(effect_scores.tolist())
    sentence_matrix_json = json.dumps(sentence_matrix.tolist())

    # Optional prompt matrix
    has_prompt_matrix = prompt_matrix is not None
    prompt_matrix_json = json.dumps(prompt_matrix.tolist()) if has_prompt_matrix else "null"

    # Color bar values with both interpretations
    colorbar_min = "0"
    colorbar_max_log = f"{global_max:.3e}"
    colorbar_mid_log = f"{global_max/2:.3e}"

    # Convert to multiplicative effect: exp(log_change)
    colorbar_max_mult = f"{np.exp(global_max):.2f}x"
    colorbar_mid_mult = f"{np.exp(global_max/2):.2f}x"

    # Problem info section
    problem_section = ""
    if problem_info:
        # Extract problem text from metadata
        prompt_texts = problem_info.get('prompt_texts', [])
        full_prompt = problem_info.get('full_prompt', '')
        problem_text = extract_problem_text_from_prompt(prompt_texts, full_prompt)
        
        problem_section = f"""
        <div class="problem-info">
            <h2>Problem {problem_info.get('problem_num', 'N/A')} ({'Correct' if problem_info.get('is_correct', False) else 'Incorrect'} Solution)</h2>
            <div style="background-color: #f9f9f9; padding: 15px; border-left: 4px solid #2196F3; margin: 10px 0; border-radius: 4px;">
                <strong>Question:</strong> {problem_text}
            </div>
            <p><strong>Model:</strong> {problem_info.get('model_name', 'N/A')}</p>
            <p><strong>Total Sentences:</strong> {len(sentences)}</p>
            <p><strong>Metric:</strong> {problem_info.get('metric', 'N/A')}</p>
        </div>
        """

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{title}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
            background-color: #fafafa;
        }}
        .sentence {{
            display: inline;
            padding: 2px 4px;
            margin: 1px;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s ease;
            border: 2px solid transparent;
        }}
        .sentence:hover {{
            opacity: 0.8;
        }}
        .sentence.hovered {{
            border: 2px solid #333;
            box-shadow: 0 0 8px rgba(0,0,0,0.3);
            font-weight: bold;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .problem-info {{
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 15px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .problem-info h2 {{
            margin-top: 0;
            color: #333;
        }}
        .problem-info p {{
            margin: 5px 0;
        }}
        .text-container {{
            width: 100%;
            font-size: 16px;
            background-color: #fff;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .legend {{
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .colorbar-container {{
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .colorbar {{
            height: 20px;
            width: 300px;
            background: linear-gradient(to right, #ffffff, #ffcccc, #ff9999, #ff6666, #ff3333, #ff0000);
            border: 1px solid #ccc;
            border-radius: 4px;
            margin: 10px 0;
        }}
        .colorbar-labels {{
            display: flex;
            justify-content: space-between;
            width: 300px;
            font-size: 11px;
            color: #666;
            font-family: monospace;
        }}
        .colorbar-blue {{
            background: linear-gradient(to right, #ffffff, #ccccff, #9999ff, #6666ff, #3333ff, #0000ff);
        }}
        .mode-controls {{
            margin: 20px 0;
            padding: 15px;
            background-color: #fff;
            border: 1px solid #ddd;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .mode-button {{
            padding: 8px 16px;
            margin: 0 5px;
            border: 2px solid #ddd;
            border-radius: 5px;
            background-color: #f8f8f8;
            cursor: pointer;
            transition: all 0.2s ease;
        }}
        .mode-button.active {{
            border-color: #2196F3;
            background-color: #e3f2fd;
            font-weight: bold;
        }}
        .mode-button:hover {{
            background-color: #e8e8e8;
        }}
        .distance-control {{
            margin: 10px 0;
        }}
        .distance-control input {{
            width: 60px;
            padding: 4px;
            margin: 0 5px;
            border: 1px solid #ddd;
            border-radius: 3px;
        }}
        .mode-indicator {{
            font-weight: bold;
            margin: 10px 0;
            padding: 8px 12px;
            background-color: #e8f4f8;
            border-left: 4px solid #2196F3;
            border-radius: 4px;
        }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    {problem_section}

    <div class="legend">
        <strong>Instructions:</strong><br>
        • <strong>Default view:</strong> Sentences colored by max dependency on far-earlier sentences<br>
        • <strong>Hover view:</strong> Shows how suppressing the hovered sentence affects other sentences<br>
        • <strong>Interpretation:</strong> Only positive dependencies shown (white = no dependency, dark = strong dependency)
    </div>

    <div class="mode-controls">
        <strong>Visualization Mode:</strong><br>
        <button class="mode-button active" id="mode1Btn" onclick="setMode(1)">
            Mode 1: Dependency on Earlier (RED)
        </button>
        <button class="mode-button" id="mode2Btn" onclick="setMode(2)">
            Mode 2: Effect on Future (BLUE)
        </button>
        <div class="distance-control">
            <strong>Min Distance:</strong>
            <input type="number" id="distanceInput" value="{min_distance}" min="0" max="20" onchange="updateDistance()">
            <span>sentences</span>
        </div>
    </div>

    <div class="colorbar-container">
        <strong>Color Scale (Dependency Strength):</strong>
        <div class="colorbar" id="colorbar"></div>
        <div class="colorbar-labels">
            <span>{colorbar_min}</span>
            <span>{colorbar_mid_log}</span>
            <span>{colorbar_max_log}</span>
        </div>
        <div style="font-size: 11px; color: #666; margin-top: 5px;">
            Log prob change: {colorbar_min} to {colorbar_max_log} | Multiplicative effect: 1.0x to {colorbar_max_mult}
        </div>
        <div class="mode-indicator" id="modeIndicator">
            Mode 1: Max dependency on far-earlier sentences (log prob changes)
        </div>
    </div>

    <div class="text-container" id="textContainer">
    </div>

    <script>
        const sentences = {sentences_json};
        const dependencyScores = {dependency_scores_json};
        const effectScores = {effect_scores_json};
        const sentenceMatrix = {sentence_matrix_json};
        const promptMatrix = {prompt_matrix_json};
        const hasPromptMatrix = {str(has_prompt_matrix).lower()};
        const globalMin = {global_min};
        const globalMax = {global_max};

        let currentMode = 1; // 1 = dependency (red), 2 = effect (blue)
        let currentHover = -1;
        let minDistance = {min_distance};
        let currentDependencyScores = [];
        let currentEffectScores = [];


        function colorFromScore(score, isBlue = false) {{
            // Clip negative values to white (no dependency)
            if (score <= 0) {{
                return 'rgb(255, 255, 255)';  // White for zero or negative
            }}

            // Normalize positive values only
            let normalized;
            if (globalMax <= 0) {{
                normalized = 0;
            }} else {{
                normalized = score / globalMax;  // 0 to 1 for positive values
            }}
            const intensity = Math.min(1, Math.max(0, normalized));

            if (isBlue) {{
                const blueGreen = Math.round(255 - intensity * 155);  // 255 to 100
                return `rgb(${{blueGreen}}, ${{blueGreen}}, 255)`;
            }} else {{
                const redGreen = Math.round(255 - intensity * 155);  // 255 to 100
                return `rgb(255, ${{redGreen}}, ${{redGreen}})`;
            }}
        }}

        function updateColors(hoverIndex = -1) {{
            const container = document.getElementById('textContainer');
            const modeIndicator = document.getElementById('modeIndicator');

            sentences.forEach((sentence, i) => {{
                const span = document.getElementById(`sentence-${{i}}`);

                // Remove/add hovered class
                span.classList.toggle('hovered', i === hoverIndex);

                if (hoverIndex === -1) {{
                    // Default coloring: sentence-to-sentence interactions only
                    if (currentMode === 1) {{
                        span.style.backgroundColor = colorFromScore(currentDependencyScores[i], false);
                    }} else {{
                        span.style.backgroundColor = colorFromScore(currentEffectScores[i], true);
                    }}
                }} else {{
                    // Hover coloring: show prompt-to-sentence if available, else sentence-to-sentence
                    if (currentMode === 1) {{
                        // Mode 1 hover: Show how prompt + other sentences affect the HOVERED sentence
                        let maxEffect = 0;

                        // Check prompt effects if available
                        if (hasPromptMatrix) {{
                            for (let promptIdx = 0; promptIdx < promptMatrix.length; promptIdx++) {{
                                const promptEffect = promptMatrix[promptIdx][hoverIndex];
                                maxEffect = Math.max(maxEffect, promptEffect);
                            }}
                        }}

                        // Check sentence effects
                        for (let sentIdx = 0; sentIdx < sentenceMatrix.length; sentIdx++) {{
                            if (sentIdx !== hoverIndex) {{
                                const sentEffect = sentenceMatrix[sentIdx][hoverIndex];
                                maxEffect = Math.max(maxEffect, sentEffect);
                            }}
                        }}

                        span.style.backgroundColor = colorFromScore(maxEffect, false);
                    }} else {{
                        // Mode 2 hover: Show how HOVERED sentence affects OTHER sentences
                        const interactionScore = sentenceMatrix[hoverIndex][i];
                        span.style.backgroundColor = colorFromScore(interactionScore, true);
                    }}
                }}
            }});

            // Update mode indicator
            if (hoverIndex === -1) {{
                if (currentMode === 1) {{
                    modeIndicator.textContent = `Mode 1: Max dependency on sentences >=${{minDistance}} positions earlier`;
                }} else {{
                    modeIndicator.textContent = `Mode 2: Max effect on sentences >=${{minDistance}} positions later`;
                }}
            }} else {{
                if (currentMode === 1) {{
                    modeIndicator.textContent = `Hover: How much sentence ${{hoverIndex + 1}} depends on earlier sentences`;
                }} else {{
                    modeIndicator.textContent = `Hover: How sentence ${{hoverIndex + 1}} affects later sentences`;
                }}
            }}
        }}

        function setMode(mode) {{
            currentMode = mode;

            // Update button styles
            document.getElementById('mode1Btn').classList.toggle('active', mode === 1);
            document.getElementById('mode2Btn').classList.toggle('active', mode === 2);

            // Update colorbar
            const colorbar = document.getElementById('colorbar');
            if (mode === 1) {{
                colorbar.className = 'colorbar';
            }} else {{
                colorbar.className = 'colorbar colorbar-blue';
            }}

            // Refresh colors
            updateColors(currentHover);
        }}

        function updateDistance() {{
            minDistance = parseInt(document.getElementById('distanceInput').value);

            // Recalculate scores with new distance
            currentDependencyScores = calculateDependencyScores(interactionMatrix, minDistance);
            currentEffectScores = calculateEffectScores(interactionMatrix, minDistance);

            // Refresh the visualization
            updateColors(currentHover);
        }}

        function initializeVisualization() {{
            // First, calculate the initial scores using the batch version
            recalculateScores();

            const container = document.getElementById('textContainer');

            sentences.forEach((sentence, i) => {{
                const span = document.createElement('span');
                span.id = `sentence-${{i}}`;
                span.className = 'sentence';
                span.textContent = sentence + ' ';

                span.addEventListener('mouseenter', () => {{
                    currentHover = i;
                    updateColors(i);
                }});

                span.addEventListener('mouseleave', () => {{
                    if (currentHover === i) {{
                        currentHover = -1;
                        updateColors();
                    }}
                }});

                container.appendChild(span);
            }});

            // Initial coloring
            updateColors();
        }}

        // Initialize when page loads
        document.addEventListener('DOMContentLoaded', initializeVisualization);
    </script>
</body>
</html>
"""
    return html

def visualize_result(result: Dict[str, Any],
                    output_path: str = "sentence_viz.html",
                    metric: str = 'kl_matrix_t1',
                    min_distance: int = 3) -> str:
    """
    Main function to create visualization from analysis result.

    Default mode: sentence-to-sentence interactions
    Hover mode: prompt-to-sentence effects (if available)

    Args:
        result: Result dict from detailed_interaction_analysis
        output_path: Where to save the HTML file
        metric: Which metric to use ('kl_matrix_t1', 'tv_matrix_t1', etc.)
        min_distance: Minimum distance for dependency calculation

    Returns:
        Path to generated HTML file
    """
    # Always get reasoning sentences (what we visualize)
    target_sentences = result['metadata']['sentences']

    # Create sentence-to-sentence matrix (default mode)
    sentence_matrix = create_sentence_to_sentence_matrix(result, metric)
    sentence_dependency_scores = calculate_dependency_scores(sentence_matrix, min_distance, 'sentence_to_sentence')
    sentence_effect_scores = calculate_effect_scores(sentence_matrix, min_distance, 'sentence_to_sentence')

    # Create prompt-to-sentence matrix if available (hover mode)
    prompt_matrix = None
    if 'prompt_intervention' in result:
        prompt_matrix = create_prompt_to_sentence_matrix(result, metric)

    # Create title and problem info
    problem_num = result['metadata']['problem_num']
    is_correct = result['metadata']['is_correct']
    model_name = result['metadata']['model_name']
    prompt_texts = result['metadata'].get('prompt_texts', [])
    full_prompt = result['metadata'].get('full_prompt', '')
    title = f"Sentence Dependencies - {metric}"

    problem_info = {
        'problem_num': problem_num,
        'is_correct': is_correct,
        'model_name': model_name,
        'metric': metric,
        'prompt_texts': prompt_texts,
        'full_prompt': full_prompt
    }

    # Generate HTML with both matrices
    html = create_html_visualization(
        target_sentences, sentence_dependency_scores, sentence_effect_scores,
        sentence_matrix, title, problem_info, min_distance, prompt_matrix
    )

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path

def visualize_batch_results(
    batch_results: Dict[tuple, Dict[str, Any]],
    output_path: str = "batch_sentence_viz.html",
    metric: str = 'kl_matrix_t1',
    min_distance: int = 3
) -> str:
    """
    Create interactive visualization with problem selector for batch results.

    Args:
        batch_results: Results from load_all_cached_interventions()
        output_path: Where to save HTML
        metric: Which metric to use
        min_distance: Minimum distance for dependency calculation

    Returns:
        Path to generated HTML file
    """
    if not batch_results:
        raise ValueError("No results provided")

    # Get available metrics from first result
    first_result = next(iter(batch_results.values()))
    available_metrics = get_available_metrics(first_result)
    # Add NLL metrics if computed
    if 'nll_changes_t1' in first_result.get('response_intervention', {}):
        available_metrics.extend(['nll_changes_t1', 'nll_changes_t06'])

    # Prepare data for all problems with all metrics
    problems_data = {}
    available_problems = []

    for problem_key, result in batch_results.items():
        problem_num, is_correct = problem_key

        # Get reasoning sentences and prompt text
        target_sentences = result['metadata']['sentences']
        prompt_texts = result['metadata'].get('prompt_texts', [])
        full_prompt = result['metadata'].get('full_prompt', '')

        # Store matrices for all available metrics
        sentence_matrices = {}
        prompt_matrices = {}

        for curr_metric in available_metrics:
            # Create sentence-to-sentence matrix
            sentence_matrices[curr_metric] = create_sentence_to_sentence_matrix(result, curr_metric).tolist()

            # Create prompt-to-sentence matrix if available
            if 'prompt_intervention' in result and curr_metric in result['prompt_intervention']:
                prompt_matrices[curr_metric] = create_prompt_to_sentence_matrix(result, curr_metric).tolist()
            else:
                prompt_matrices[curr_metric] = None

        model_name = result['metadata']['model_name']
        has_prompt_data = 'prompt_intervention' in result

        problems_data[f"{problem_num}_{is_correct}"] = {
            'problem_num': problem_num,
            'is_correct': is_correct,
            'model_name': model_name,
            'sentences': target_sentences,
            'prompt_texts': prompt_texts,
            'full_prompt': full_prompt,
            'sentence_matrices': sentence_matrices,
            'prompt_matrices': prompt_matrices,
            'has_prompt_matrix': has_prompt_data
        }

        available_problems.append({
            'key': f"{problem_num}_{is_correct}",
            'label': f"Problem {problem_num} ({'Correct' if is_correct else 'Incorrect'})",
            'problem_num': problem_num,
            'is_correct': is_correct
        })

    # Sort problems by number, then by correctness
    available_problems.sort(key=lambda x: (x['problem_num'], not x['is_correct']))

    # Get global color scaling across all problems and metrics
    all_values = []
    for data in problems_data.values():
        for curr_metric in available_metrics:
            sentence_matrix = data['sentence_matrices'][curr_metric]
            all_values.extend([item for sublist in sentence_matrix for item in sublist])
            prompt_matrix = data['prompt_matrices'][curr_metric]
            if prompt_matrix:
                all_values.extend([item for sublist in prompt_matrix for item in sublist])

    all_values = np.array(all_values)
    valid_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]
    positive_values = valid_values[valid_values > 0]
    global_min = 0
    # Use 99th percentile to clip outliers
    global_max = np.percentile(positive_values, 99) if len(positive_values) > 0 else 1

    # Convert to JSON for JavaScript
    problems_json = json.dumps(problems_data)
    available_problems_json = json.dumps(available_problems)
    available_metrics_json = json.dumps(available_metrics)

    # Get first problem text from metadata
    if available_problems:
        first_key = available_problems[0]['key']
        first_data = problems_data[first_key]
        problem_text = extract_problem_text_from_prompt(
            first_data.get('prompt_texts', []),
            first_data.get('full_prompt', '')
        )
    else:
        problem_text = "No problems available"

    html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Batch Sentence Dependencies - {metric}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 20px;
            background-color: #f8f9fa;
        }}

        .header {{
            text-align: center;
            margin-bottom: 30px;
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .problem-selector {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}

        .problem-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            max-height: 200px;
            overflow-y: auto;
        }}

        .problem-btn {{
            padding: 8px 12px;
            border: 2px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
            font-size: 14px;
        }}

        .problem-btn:hover {{
            border-color: #007bff;
            background: #f8f9ff;
        }}

        .problem-btn.active {{
            border-color: #007bff;
            background: #007bff;
            color: white;
        }}

        .controls {{
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            display: flex;
            flex-wrap: wrap;
            gap: 20px;
            align-items: center;
        }}

        .mode-buttons {{
            display: flex;
            gap: 10px;
        }}

        .mode-btn {{
            padding: 8px 16px;
            border: 2px solid #ddd;
            background: white;
            border-radius: 4px;
            cursor: pointer;
            transition: all 0.2s;
        }}

        .mode-btn.active {{
            border-color: #dc3545;
            background: #dc3545;
            color: white;
        }}

        .mode-btn.mode2.active {{
            border-color: #007bff;
            background: #007bff;
        }}
        
        .agg-btn {{
            background: white;
            border: 1px solid #ccc;
            border-radius: 4px;
            cursor: pointer;
            padding: 4px 8px;
            transition: all 0.2s;
        }}
        
        .agg-btn:hover {{
            background: #f0f0f0;
        }}
        
        .agg-btn.active {{
            background: #4CAF50;
            color: white;
            border-color: #4CAF50;
        }}

        .distance-control {{
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .colorbar {{
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 10px;
        }}

        .colorbar-gradient {{
            width: 200px;
            height: 20px;
            background: linear-gradient(to right, white, #dc3545);
            border: 1px solid #ddd;
            border-radius: 4px;
        }}

        .text-container {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            line-height: 1.6;
        }}

        .sentence {{
            display: inline;
            margin: 2px;
            padding: 3px 6px;
            border-radius: 3px;
            cursor: pointer;
            transition: all 0.2s;
            border: 1px solid transparent;
        }}

        .sentence:hover {{
            border-color: #333;
            transform: scale(1.02);
        }}

        .sentence.hovered {{
            outline: 3px solid #333;
            outline-offset: 1px;
        }}

        .problem-info {{
            background: #e9ecef;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 20px;
            font-family: monospace;
            font-size: 14px;
            max-height: 150px;
            overflow-y: auto;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Batch Sentence Dependencies</h1>
        <p>Interactive visualization with problem and metric selection</p>
    </div>

    <div class="problem-selector">
        <h3>Select Problem:</h3>
        <div class="problem-grid" id="problemGrid">
        </div>
    </div>

    <div class="controls">
        <div class="metric-selector">
            <label for="metricSelect">Metric:</label>
            <select id="metricSelect" onchange="selectMetric(this.value)" style="padding: 6px; font-size: 14px;">
            </select>
        </div>

        <div class="mode-buttons">
            <button class="mode-btn active" onclick="setMode(1)">
                Mode 1: Dependencies (RED)
            </button>
            <button class="mode-btn mode2" onclick="setMode(2)">
                Mode 2: Effects (BLUE)
            </button>
        </div>

        <div class="distance-control">
            <label for="distanceSlider">Min Gap (sentences between):</label>
            <input type="range" id="distanceSlider" min="0" max="20" value="{min_distance}"
                   oninput="updateDistance(this.value)">
            <span id="distanceValue">{min_distance}</span>
        </div>
        
        <div style="margin-top: 10px;">
            <label style="font-size: 14px;">
                <input type="checkbox" id="adaptiveScaling" onchange="toggleAdaptiveScaling(this.checked)">
                Adaptive color scaling
            </label>
            <span id="scalingInfo" style="font-size: 12px; color: #666; margin-left: 10px;"></span>
        </div>
        
        <div style="margin-top: 10px;">
            <label style="font-size: 14px;">Aggregation:</label>
            <button id="aggMax" class="agg-btn active" onclick="setAggregation('max')" style="padding: 4px 8px; margin: 0 5px;">MAX</button>
            <button id="aggSum" class="agg-btn" onclick="setAggregation('sum')" style="padding: 4px 8px;">SUM</button>
        </div>
    </div>

    <div class="problem-info" id="problemInfo">
        <h3>Problem Question:</h3>
        <div id="problemText">{problem_text}</div>
    </div>

    <div id="modeIndicator" style="text-align: center; margin-bottom: 10px; font-weight: bold; color: #666;">
        Mode 1: Max dependency on sentences with ≥{min_distance} gap
    </div>

    <div class="colorbar" style="margin: 10px auto; width: 300px;">
        <span>0</span>
        <div class="colorbar-gradient" style="display: inline-block; width: 200px; height: 20px;"></div>
        <span id="maxValue">{global_max:.3e}</span>
        <span id="percentileLabel" style="font-size: 11px; color: #666;">(99th percentile)</span>
    </div>

    <div class="text-container" id="textContainer">
    </div>

    <script>
        const problemsData = {problems_json};
        const availableProblems = {available_problems_json};
        const availableMetrics = {available_metrics_json};
        const globalMin = {global_min};
        let globalMax = {global_max};  // Made mutable for metric changes

        let currentProblemKey = availableProblems[0].key;
        let currentMetric = '{metric}';
        let currentMode = 1;
        let currentHover = -1;
        let minDistance = {min_distance};
        let currentDependencyScores = [];
        let currentEffectScores = [];
        let useAdaptiveScaling = false;
        let adaptiveMax = globalMax;
        let aggregationMode = 'max'; // 'max' or 'sum'

        function initializeProblemGrid() {{
            const grid = document.getElementById('problemGrid');
            availableProblems.forEach(problem => {{
                const btn = document.createElement('button');
                btn.className = 'problem-btn';
                btn.textContent = problem.label;
                btn.onclick = () => selectProblem(problem.key);
                if (problem.key === currentProblemKey) {{
                    btn.classList.add('active');
                }}
                grid.appendChild(btn);
            }});
        }}

        function selectProblem(problemKey) {{
            currentProblemKey = problemKey;

            // Update active button
            document.querySelectorAll('.problem-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});
            event.target.classList.add('active');

            // Update problem info
            const problemData = problemsData[problemKey];
            const problemText = extractProblemText(problemData.prompt_texts, problemData.full_prompt);
            const problemInfoDiv = document.getElementById('problemInfo');
            problemInfoDiv.innerHTML = `<h3>Problem ${{problemData.problem_num}} (${{problemData.is_correct ? 'Correct' : 'Incorrect'}} Solution):</h3><div id="problemText">${{problemText}}</div>`;

            // Recalculate and update visualization
            recalculateScores();
            renderSentences();
            if (useAdaptiveScaling) {{
                computeAdaptiveMax();
            }}
            updateColors();
        }}

        function calculateDependencyScores(interactionMatrix, minDist) {{
            const nSentences = interactionMatrix[0].length;
            const scores = new Array(nSentences).fill(0);

            for (let targetSent = 0; targetSent < nSentences; targetSent++) {{
                let aggregatedValue = 0;
                if (aggregationMode === 'sum') {{
                    // Sum all dependencies meeting distance criteria
                    for (let suppressedSent = 0; suppressedSent < targetSent; suppressedSent++) {{
                        const gap = targetSent - suppressedSent - 1;
                        if (gap >= minDist) {{
                            const effect = interactionMatrix[suppressedSent][targetSent];
                            // Filter out NaN and invalid values
                            if (!isNaN(effect) && isFinite(effect)) {{
                                aggregatedValue += effect;
                            }}
                        }}
                    }}
                }} else {{
                    // Max dependency (default)
                    for (let suppressedSent = 0; suppressedSent < targetSent; suppressedSent++) {{
                        const gap = targetSent - suppressedSent - 1;
                        if (gap >= minDist) {{
                            const effect = interactionMatrix[suppressedSent][targetSent];
                            // Filter out NaN and invalid values
                            if (!isNaN(effect) && isFinite(effect)) {{
                                aggregatedValue = Math.max(aggregatedValue, effect);
                            }}
                        }}
                    }}
                }}
                scores[targetSent] = aggregatedValue;
            }}
            return scores;
        }}

        function calculateEffectScores(interactionMatrix, minDist) {{
            const nSentences = interactionMatrix.length;
            const scores = new Array(nSentences).fill(0);

            for (let sourceSent = 0; sourceSent < nSentences; sourceSent++) {{
                let aggregatedValue = 0;
                // Start at sourceSent + minDist + 1 to ensure gap of at least minDist
                if (aggregationMode === 'sum') {{
                    for (let targetSent = sourceSent + minDist + 1; targetSent < interactionMatrix[0].length; targetSent++) {{
                        const effect = interactionMatrix[sourceSent][targetSent];
                        // Filter out NaN and invalid values
                        if (!isNaN(effect) && isFinite(effect)) {{
                            aggregatedValue += effect;
                        }}
                    }}
                }} else {{
                    // Max effect (default)
                    for (let targetSent = sourceSent + minDist + 1; targetSent < interactionMatrix[0].length; targetSent++) {{
                        const effect = interactionMatrix[sourceSent][targetSent];
                        // Filter out NaN and invalid values
                        if (!isNaN(effect) && isFinite(effect)) {{
                            aggregatedValue = Math.max(aggregatedValue, effect);
                        }}
                    }}
                }}
                scores[sourceSent] = aggregatedValue;
            }}
            return scores;
        }}

        function computeAdaptiveMax() {{
            if (!useAdaptiveScaling) {{
                adaptiveMax = globalMax;
                document.getElementById('scalingInfo').textContent = '';
                document.getElementById('percentileLabel').textContent = '(99th percentile)';
                return;
            }}
            
            const problemData = problemsData[currentProblemKey];
            const sentenceMatrix = problemData.sentence_matrices[currentMetric];
            const promptMatrix = problemData.prompt_matrices[currentMetric];
            let relevantValues = [];
            
            if (currentHover >= 0) {{
                // Hovering on a response sentence - collect the displayed max values for each sentence
                if (currentMode === 1) {{
                    // Dependencies mode - each sentence shows its max dependency
                    problemData.sentences.forEach((_, i) => {{
                        if (i < currentHover) {{
                            // This sentence shows how much the hovered sentence depends on it
                            relevantValues.push(sentenceMatrix[i][currentHover]);
                        }} else if (i === currentHover) {{
                            // The hovered sentence shows its total dependency score
                            relevantValues.push(currentDependencyScores[i]);
                        }}
                        // Later sentences show 0, don't include
                    }});
                    // Also include prompt dependencies shown
                    if (promptMatrix) {{
                        for (let i = 0; i < promptMatrix.length; i++) {{
                            relevantValues.push(promptMatrix[i][currentHover]);
                        }}
                    }}
                }} else {{
                    // Effects mode - each sentence shows the effect of hovered sentence
                    problemData.sentences.forEach((_, i) => {{
                        relevantValues.push(sentenceMatrix[currentHover][i]);
                    }});
                }}
            }} else if (currentHover < -1) {{
                // Hovering on a prompt component - collect its effects on all response sentences
                const promptIdx = -(currentHover + 100);
                if (promptMatrix && currentMode === 2) {{
                    for (let i = 0; i < sentenceMatrix[0].length; i++) {{
                        relevantValues.push(promptMatrix[promptIdx][i]);
                    }}
                }}
            }} else {{
                // No hover - collect the max scores that are displayed for each sentence
                if (currentMode === 1) {{
                    // Each sentence shows its dependency score
                    relevantValues = [...currentDependencyScores];
                }} else {{
                    // Each sentence shows its effect score
                    relevantValues = [...currentEffectScores];
                }}
            }}
            
            // Filter out zeros and NaNs
            relevantValues = relevantValues.filter(v => v > 0 && !isNaN(v));
            
            if (relevantValues.length > 0) {{
                // Use 99th percentile of relevant values
                relevantValues.sort((a, b) => a - b);
                const idx = Math.ceil(0.99 * relevantValues.length) - 1;
                adaptiveMax = relevantValues[Math.min(idx, relevantValues.length - 1)];
                
                document.getElementById('scalingInfo').textContent = `(Normalizing on ${{relevantValues.length}} displayed values)`;
                document.getElementById('percentileLabel').textContent = '(adaptive)';
            }} else {{
                adaptiveMax = 1;
                document.getElementById('scalingInfo').textContent = '(No values in current view)';
                document.getElementById('percentileLabel').textContent = '(adaptive)';
            }}
            
            document.getElementById('maxValue').textContent = adaptiveMax.toExponential(3);
        }}

        function colorFromScore(score, isBlue = false) {{
            // Handle NaN, undefined, or invalid values
            if (!isFinite(score) || score <= 0) {{
                return 'rgb(255, 255, 255)';
            }}

            const maxToUse = useAdaptiveScaling ? adaptiveMax : globalMax;
            
            // Safety check for maxToUse
            if (!isFinite(maxToUse) || maxToUse <= 0) {{
                console.warn('Invalid maxToUse:', maxToUse);
                return 'rgb(255, 255, 255)';
            }}
            
            const normalizedScore = Math.min(score / maxToUse, 1);

            if (isBlue) {{
                // Blue mode - lighter max intensity (80 instead of 0)
                const blueGreen = Math.round(255 - (175 * normalizedScore));  // Range: 255 to 80
                return `rgb(${{blueGreen}}, ${{blueGreen}}, 255)`;
            }} else {{
                // Red mode - lighter max intensity (80 instead of 0)  
                const redGreen = Math.round(255 - (175 * normalizedScore));  // Range: 255 to 80
                return `rgb(255, ${{redGreen}}, ${{redGreen}})`;
            }}
        }}

        function toggleAdaptiveScaling(checked) {{
            useAdaptiveScaling = checked;
            computeAdaptiveMax();
            updateColors(currentHover);
        }}
        
        function setAggregation(mode) {{
            aggregationMode = mode;
            
            // Update button states
            document.getElementById('aggMax').classList.toggle('active', mode === 'max');
            document.getElementById('aggSum').classList.toggle('active', mode === 'sum');
            
            // Recalculate scores with new aggregation
            recalculateScores();
            
            // Update adaptive max if needed
            if (useAdaptiveScaling) {{
                computeAdaptiveMax();
            }}
            
            updateColors(currentHover);
        }}

        function updateColors(hoverIndex = -1) {{
            const problemData = problemsData[currentProblemKey];
            // Use the current metric from sentence_matrices
            const sentenceMatrix = problemData.sentence_matrices[currentMetric];
            const promptMatrix = problemData.prompt_matrices[currentMetric];
            const hasPromptMatrix = problemData.has_prompt_matrix;
            const modeIndicator = document.getElementById('modeIndicator');
            
            // Recompute adaptive max if needed
            if (useAdaptiveScaling) {{
                computeAdaptiveMax();
            }}
            
            // Check if hovering over prompt component (negative index)
            const isPromptHover = hoverIndex < -1;
            const promptHoverIdx = isPromptHover ? -(hoverIndex + 100) : -1;

            // Update prompt component colors
            if (problemData.prompt_texts) {{
                problemData.prompt_texts.forEach((_, i) => {{
                    const promptSpan = document.getElementById(`prompt-${{i}}`);
                    if (promptSpan) {{
                        if (isPromptHover && i === promptHoverIdx) {{
                            // Hovering over this prompt component - highlight it
                            promptSpan.style.backgroundColor = '#ffe6e6';
                        }} else if (!isPromptHover && hoverIndex >= 0 && hasPromptMatrix && promptMatrix) {{
                            // Hovering over a response sentence - show how much it depends on this prompt
                            const promptEffect = promptMatrix[i][hoverIndex];
                            if (currentMode === 1) {{
                                promptSpan.style.backgroundColor = colorFromScore(promptEffect, false);
                            }} else {{
                                promptSpan.style.backgroundColor = 'white';
                            }}
                        }} else {{
                            // Default state
                            promptSpan.style.backgroundColor = 'white';
                        }}
                    }}
                }});
            }}

            // Update response sentence colors
            problemData.sentences.forEach((sentence, i) => {{
                const span = document.getElementById(`sentence-${{i}}`);

                span.classList.toggle('hovered', i === hoverIndex);

                if (hoverIndex === -1 && !isPromptHover) {{
                    // No hover - show default scores
                    if (currentMode === 1) {{
                        span.style.backgroundColor = colorFromScore(currentDependencyScores[i], false);
                    }} else {{
                        span.style.backgroundColor = colorFromScore(currentEffectScores[i], true);
                    }}
                }} else if (isPromptHover) {{
                    // Hovering over a prompt component - show its effect on response sentences
                    if (hasPromptMatrix && promptMatrix && currentMode === 2) {{
                        const promptEffect = promptMatrix[promptHoverIdx][i];
                        span.style.backgroundColor = colorFromScore(promptEffect, true);
                    }} else {{
                        span.style.backgroundColor = 'white';
                    }}
                }} else {{
                    // Hovering over a response sentence
                    if (i === hoverIndex) {{
                        // The hovered sentence itself - keep it white (no self-interaction)
                        span.style.backgroundColor = 'white';
                    }} else if (currentMode === 1) {{
                        // Mode 1: Show what the HOVERED sentence depends on
                        let dependencyScore = 0;
                        
                        if (i < hoverIndex) {{
                            // Show how much the hovered sentence depends on this earlier sentence
                            dependencyScore = sentenceMatrix[i][hoverIndex];
                        }}
                        // Later sentences: dependencyScore stays 0

                        span.style.backgroundColor = colorFromScore(dependencyScore, false);
                    }} else {{
                        // Mode 2: Show effect of hovered sentence on others
                        const interactionScore = sentenceMatrix[hoverIndex][i];
                        span.style.backgroundColor = colorFromScore(interactionScore, true);
                    }}
                }}
            }});

            if (hoverIndex === -1 && !isPromptHover) {{
                const aggText = aggregationMode === 'sum' ? 'Total' : 'Max';
                if (currentMode === 1) {{
                    const gapText = minDistance === 0 ? "adjacent sentences" : 
                                   minDistance === 1 ? "sentences with ≥1 gap" : 
                                   `sentences with ≥${{minDistance}} gap`;
                    modeIndicator.textContent = `Mode 1: ${{aggText}} dependency on ${{gapText}}`;
                }} else {{
                    const gapText = minDistance === 0 ? "adjacent sentences" : 
                                   minDistance === 1 ? "sentences with ≥1 gap" : 
                                   `sentences with ≥${{minDistance}} gap`;
                    modeIndicator.textContent = `Mode 2: ${{aggText}} effect on ${{gapText}}`;
                }}
            }} else if (isPromptHover) {{
                if (currentMode === 1) {{
                    modeIndicator.textContent = `Hover: Prompt component ${{promptHoverIdx + 1}} (dependencies not shown in this mode)`;
                }} else {{
                    // Calculate total effect of this prompt component
                    let totalEffect = 0;
                    if (hasPromptMatrix && promptMatrix) {{
                        for (let i = 0; i < sentenceMatrix[0].length; i++) {{
                            const effect = promptMatrix[promptHoverIdx][i];
                            if (!isNaN(effect) && isFinite(effect) && effect > 0) {{
                                totalEffect += effect;
                            }}
                        }}
                    }}
                    modeIndicator.textContent = `Hover: How prompt component ${{promptHoverIdx + 1}} affects response sentences (total: ${{totalEffect.toExponential(2)}})`;
                }}
            }} else {{
                if (currentMode === 1) {{
                    const score = currentDependencyScores[hoverIndex];
                    const scoreText = isNaN(score) ? 'NaN' : score.toExponential(2);
                    const aggText = aggregationMode === 'sum' ? 'Total' : 'Max';
                    modeIndicator.textContent = `Hover: How much sentence ${{hoverIndex + 1}} depends on earlier sentences (${{aggText}}: ${{scoreText}})`;
                }} else {{
                    const score = currentEffectScores[hoverIndex];
                    const scoreText = isNaN(score) ? 'NaN' : score.toExponential(2);
                    const aggText = aggregationMode === 'sum' ? 'Total' : 'Max';
                    modeIndicator.textContent = `Hover: How sentence ${{hoverIndex + 1}} affects later sentences (${{aggText}}: ${{scoreText}})`;
                }}
            }}
        }}

        function updateColorbar() {{
            const colorbarGradient = document.querySelector('.colorbar-gradient');
            if (currentMode === 1) {{
                // Red gradient for dependency mode
                colorbarGradient.style.background = 'linear-gradient(to right, white, rgb(255, 80, 80))';
            }} else {{
                // Blue gradient for effect mode
                colorbarGradient.style.background = 'linear-gradient(to right, white, rgb(80, 80, 255))';
            }}
        }}

        function setMode(mode) {{
            currentMode = mode;

            document.querySelectorAll('.mode-btn').forEach(btn => {{
                btn.classList.remove('active');
            }});

            if (mode === 1) {{
                document.querySelector('.mode-btn:first-child').classList.add('active');
            }} else {{
                document.querySelector('.mode-btn.mode2').classList.add('active');
            }}

            updateColorbar();
            updateColors(currentHover);
        }}

        function updateDistance(value) {{
            minDistance = parseInt(value);
            document.getElementById('distanceValue').textContent = minDistance;
            recalculateScores();
            if (useAdaptiveScaling) {{
                computeAdaptiveMax();
            }}
            updateColors(currentHover);
        }}

        function recalculateScores() {{
            const problemData = problemsData[currentProblemKey];
            const sentenceMatrix = problemData.sentence_matrices[currentMetric];
            currentDependencyScores = calculateDependencyScores(sentenceMatrix, minDistance);
            currentEffectScores = calculateEffectScores(sentenceMatrix, minDistance);
        }}

        function renderSentences() {{
            const problemData = problemsData[currentProblemKey];
            const container = document.getElementById('textContainer');
            container.innerHTML = '';
            
            // Add prompt components first if they exist
            if (problemData.prompt_texts && problemData.prompt_texts.length > 0) {{
                const promptDiv = document.createElement('div');
                promptDiv.style.cssText = 'border: 2px solid #2196F3; padding: 10px; margin-bottom: 15px; background-color: white; border-radius: 5px;';
                
                const promptLabel = document.createElement('div');
                promptLabel.textContent = 'PROMPT:';
                promptLabel.style.cssText = 'font-weight: bold; color: #2196F3; margin-bottom: 10px;';
                promptDiv.appendChild(promptLabel);
                
                problemData.prompt_texts.forEach((promptText, i) => {{
                    const span = document.createElement('span');
                    span.id = `prompt-${{i}}`;
                    span.className = 'sentence prompt-sentence';
                    span.textContent = promptText + ' ';
                    span.style.cssText = 'background-color: white; padding: 2px 4px; margin: 2px; border-radius: 3px; display: inline-block;';
                    span.onmouseenter = () => {{
                        currentHover = -100 - i;  // Use negative values for prompt hover
                        updateColors(currentHover);
                    }};
                    span.onmouseleave = () => {{
                        currentHover = -1;
                        updateColors();
                    }};
                    promptDiv.appendChild(span);
                }});
                
                container.appendChild(promptDiv);
            }}
            
            // Add response sentences
            const responseDiv = document.createElement('div');
            responseDiv.style.cssText = 'border: 2px solid #4CAF50; padding: 10px; background-color: white; border-radius: 5px;';
            
            const responseLabel = document.createElement('div');
            responseLabel.textContent = 'RESPONSE:';
            responseLabel.style.cssText = 'font-weight: bold; color: #4CAF50; margin-bottom: 10px;';
            responseDiv.appendChild(responseLabel);

            problemData.sentences.forEach((sentence, i) => {{
                const span = document.createElement('span');
                span.className = 'sentence';
                span.id = `sentence-${{i}}`;
                span.textContent = sentence + ' ';
                span.onmouseenter = () => {{
                    currentHover = i;
                    updateColors(i);
                }};
                span.onmouseleave = () => {{
                    currentHover = -1;
                    updateColors();
                }};
                responseDiv.appendChild(span);
            }});
            
            container.appendChild(responseDiv);
        }}

        function extractProblemText(promptTexts, fullPrompt) {{
            // Extract problem text from metadata
            if (promptTexts && promptTexts.length > 1) {{
                // Skip first (prompt prefix) and last (cot prefix) elements
                const problemParts = promptTexts.length > 2 
                    ? promptTexts.slice(1, -1) 
                    : promptTexts.slice(1);
                return problemParts.join(" ");
            }} else if (fullPrompt) {{
                // Try to extract from full prompt
                if (fullPrompt.includes("Problem:")) {{
                    const start = fullPrompt.indexOf("Problem:") + "Problem:".length;
                    const end = fullPrompt.indexOf("\\n\\n", start);
                    return fullPrompt.substring(start, end !== -1 ? end : fullPrompt.length).trim();
                }} else if (fullPrompt.includes("Let's think")) {{
                    return fullPrompt.substring(0, fullPrompt.indexOf("Let's think")).trim();
                }}
            }}
            return "Problem text not available in metadata";
        }}

        function initializeMetricSelector() {{
            const select = document.getElementById('metricSelect');
            availableMetrics.forEach(metric => {{
                const option = document.createElement('option');
                option.value = metric;
                // Special handling for normalized_kl
                if (metric === 'normalized_kl') {{
                    option.textContent = 'NORMALIZED KL (% contribution)';
                }} else {{
                    option.textContent = metric.toUpperCase().replace(/_/g, ' ');
                }}
                if (metric === currentMetric) {{
                    option.selected = true;
                }}
                select.appendChild(option);
            }});
        }}

        function selectMetric(metric) {{
            currentMetric = metric;
            
            // Recalculate the global max for the new metric
            updateGlobalMaxForMetric();
            
            // Update the display
            recalculateScores();
            if (useAdaptiveScaling) {{
                computeAdaptiveMax();
            }} else {{
                document.getElementById('maxValue').textContent = globalMax.toExponential(3);
            }}
            updateColors(currentHover);
        }}
        
        function updateGlobalMaxForMetric(percentile = 99) {{
            // Collect all values for current metric
            let allValues = [];
            for (const key in problemsData) {{
                const data = problemsData[key];
                const sentenceMatrix = data.sentence_matrices[currentMetric];
                const promptMatrix = data.prompt_matrices[currentMetric];
                
                // Collect from sentence matrix
                for (let i = 0; i < sentenceMatrix.length; i++) {{
                    for (let j = 0; j < sentenceMatrix[i].length; j++) {{
                        if (sentenceMatrix[i][j] > 0 && !isNaN(sentenceMatrix[i][j])) {{
                            allValues.push(sentenceMatrix[i][j]);
                        }}
                    }}
                }}
                
                // Collect from prompt matrix if exists
                if (promptMatrix) {{
                    for (let i = 0; i < promptMatrix.length; i++) {{
                        for (let j = 0; j < promptMatrix[i].length; j++) {{
                            if (promptMatrix[i][j] > 0 && !isNaN(promptMatrix[i][j])) {{
                                allValues.push(promptMatrix[i][j]);
                            }}
                        }}
                    }}
                }}
            }}
            
            // Calculate percentile to clip outliers
            if (allValues.length > 0) {{
                allValues.sort((a, b) => a - b);
                const index = Math.ceil((percentile / 100) * allValues.length) - 1;
                globalMax = allValues[Math.min(index, allValues.length - 1)];
                
                // Show statistics in console for debugging
                console.log(`Metric: ${{currentMetric}}`);
                console.log(`Total values: ${{allValues.length}}`);
                console.log(`Min: ${{allValues[0].toExponential(2)}}`);
                console.log(`Median: ${{allValues[Math.floor(allValues.length/2)].toExponential(2)}}`);
                console.log(`95th percentile: ${{globalMax.toExponential(2)}}`);
                console.log(`Max: ${{allValues[allValues.length-1].toExponential(2)}}`);
            }} else {{
                globalMax = 1;
            }}
        }}

        function initializeVisualization() {{
            initializeProblemGrid();
            initializeMetricSelector();
            updateColorbar();
            recalculateScores();
            renderSentences();
            updateColors();
        }}

        // Initialize when page loads
        window.onload = initializeVisualization;
    </script>
</body>
</html>
"""

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path

if __name__ == "__main__":
    import sys
    import os
    # Add parent directory for imports
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'whitebox-analyses', 'nathan_scripts'))
    from visualization_utils.aggregation_utils import load_all_cached_interventions

    # Default: load suppression results
    print("Loading llama-8b suppression results...")

    results = load_all_cached_interventions(
        model_name='llama-8b',
        cumulative=False,
        amplify=False,
        amplify_factor=2.0
    )

    if not results:
        print("No cached results found!")
        print("Make sure you have run the intervention analysis first.")
        exit(1)

    print(f"Found {len(results)} problems")

    # Create visualization
    output_path = visualize_batch_results(results)

    import os
    abs_path = os.path.abspath(output_path)
    print(f"\nVisualization created: {abs_path}")
    print(f"\nTo view, open in browser or run:")
    print(f"python -m http.server 8000")
    print(f"Then visit: http://localhost:8000/{output_path}")