import numpy as np
import json
import os
from typing import List, Dict, Any

def create_sentence_interaction_matrix(result: Dict[str, Any]) -> np.ndarray:
    """
    Create sentence-to-sentence interaction matrix based on log probability changes.

    Args:
        result: Result dict from detailed_suppression_analysis

    Returns:
        (n_sentences, n_sentences) matrix where entry [i,j] is the average log prob change
        in sentence j when sentence i is suppressed: log(prob_original) - log(prob_suppressed)
    """
    sentences = result['metadata']['sentences']
    sentence_boundaries = result['metadata']['sentence_boundaries']
    prob_before = result['prob_before_t1']  # Shape: (n_tokens,)
    prob_after = result['prob_after_t1']    # Shape: (n_sentences, n_tokens)
    causal_mask = result['causal_mask']     # Shape: (n_sentences, n_tokens)

    n_sentences = len(sentences)
    interaction_matrix = np.zeros((n_sentences, n_sentences))

    # Small epsilon to avoid log(0)
    eps = 1e-10
    prob_before_safe = np.maximum(prob_before, eps)
    prob_after_safe = np.maximum(prob_after, eps)

    # Calculate log probability changes: log(original) - log(suppressed)
    log_prob_changes = np.log(prob_before_safe)[None, :] - np.log(prob_after_safe)  # (n_sentences, n_tokens)

    # For each suppressed sentence i, calculate average effect on each target sentence j
    for i in range(n_sentences):
        for j in range(n_sentences):
            if j < len(sentence_boundaries):
                start_token, end_token = sentence_boundaries[j]

                # Get log prob changes for tokens in target sentence j when sentence i suppressed
                causal_valid = causal_mask[i, start_token:end_token]
                sentence_changes = log_prob_changes[i, start_token:end_token]

                # Only use changes where causal mask is True and values are valid
                valid_mask = causal_valid & ~np.isnan(sentence_changes) & ~np.isinf(sentence_changes)
                valid_changes = sentence_changes[valid_mask]

                if len(valid_changes) > 0:
                    interaction_matrix[i, j] = np.mean(valid_changes)  # Average within sentence
                else:
                    interaction_matrix[i, j] = 0

    return interaction_matrix

def calculate_dependency_scores(interaction_matrix: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Calculate per-sentence dependency scores from interaction matrix.

    For sentence j, find the maximum log prob change from suppressing far-earlier sentences i.
    """
    n_sentences = interaction_matrix.shape[0]
    sentence_scores = np.zeros(n_sentences)

    # For each target sentence, find max dependency on earlier sentences
    for target_sent in range(n_sentences):
        max_dependency = 0

        # Look at effects of suppressing earlier sentences on this target sentence
        for suppressed_sent in range(target_sent):
            if target_sent - suppressed_sent >= min_distance:  # Far-away earlier sentence
                effect = interaction_matrix[suppressed_sent, target_sent]
                max_dependency = max(max_dependency, effect)

        sentence_scores[target_sent] = max_dependency

    return sentence_scores

def calculate_effect_scores(interaction_matrix: np.ndarray, min_distance: int = 3) -> np.ndarray:
    """
    Calculate per-sentence effect scores from interaction matrix.

    For sentence i, find the maximum log prob change on far-future sentences j.
    """
    n_sentences = interaction_matrix.shape[0]
    sentence_scores = np.zeros(n_sentences)

    # For each source sentence, find max effect on future sentences
    for source_sent in range(n_sentences):
        max_effect = 0

        # Ensure we never consider self-interaction (minimum distance is 1)
        actual_min_distance = max(1, min_distance)
        # Look at effects of suppressing this sentence on future sentences
        for target_sent in range(source_sent + actual_min_distance, n_sentences):
            effect = interaction_matrix[source_sent, target_sent]
            max_effect = max(max_effect, effect)

        sentence_scores[source_sent] = max_effect

    return sentence_scores

def load_problem_text(problem_num: int) -> str:
    """Load the problem text from selected_problems.json."""
    try:
        # Look for selected_problems.json in parent directory
        problems_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'selected_problems.json')
        if not os.path.exists(problems_path):
            # Try current directory
            problems_path = 'selected_problems.json'

        with open(problems_path, 'r') as f:
            problems = json.load(f)

        # Find the problem by ID
        for problem in problems:
            if problem.get('problem_idx') == f'problem_{problem_num}':
                return problem.get('problem', 'Problem text not found')

        return f'Problem {problem_num} not found in selected_problems.json'
    except Exception as e:
        return f'Error loading problem text: {e}'

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
                            interaction_matrix: np.ndarray,
                            title: str = "Sentence Dependencies",
                            problem_info: Dict[str, Any] = None,
                            min_distance: int = 3) -> str:
    """
    Create HTML visualization with hover interactions and mode switching.

    Args:
        sentences: List of sentence strings
        dependency_scores: Scores for dependency on earlier sentences
        effect_scores: Scores for effect on future sentences
        interaction_matrix: (n_sentences, n_sentences) interaction matrix
        title: Title for the visualization
        problem_info: Dict with problem metadata (problem_num, is_correct, etc.)
        min_distance: Distance threshold for far dependencies

    Returns:
        HTML string
    """
    n_sentences = len(sentences)

    # Calculate global max for color scaling (only positive values matter)
    all_values = np.concatenate([dependency_scores, effect_scores, interaction_matrix.flatten()])
    valid_values = all_values[~np.isnan(all_values) & ~np.isinf(all_values)]
    positive_values = valid_values[valid_values > 0]
    global_min = 0  # Always start from 0 (white)
    global_max = np.max(positive_values) if len(positive_values) > 0 else 1

    # Keep completely raw scores - NO clipping or normalization
    dependency_scores_raw = dependency_scores
    effect_scores_raw = effect_scores
    interaction_raw = interaction_matrix

    # Convert to JSON for JavaScript (using raw values)
    sentences_json = json.dumps(sentences)
    dependency_scores_json = json.dumps(dependency_scores_raw.tolist())
    effect_scores_json = json.dumps(effect_scores_raw.tolist())
    interaction_json = json.dumps(interaction_raw.tolist())

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
        # Load the actual problem text
        problem_text = load_problem_text(problem_info.get('problem_num', 0))
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
        const interactionMatrix = {interaction_json};
        const globalMin = {global_min};
        const globalMax = {global_max};

        let currentMode = 1; // 1 = dependency (red), 2 = effect (blue)
        let currentHover = -1;
        let minDistance = {min_distance};
        let currentDependencyScores = [];
        let currentEffectScores = [];

        function calculateDependencyScores(interactionMatrix, minDist) {{
            const nSentences = interactionMatrix.length;
            const scores = new Array(nSentences).fill(0);

            for (let targetSent = 0; targetSent < nSentences; targetSent++) {{
                let maxDependency = 0;
                for (let suppressedSent = 0; suppressedSent < targetSent; suppressedSent++) {{
                    if (targetSent - suppressedSent >= minDist) {{
                        const effect = interactionMatrix[suppressedSent][targetSent];
                        maxDependency = Math.max(maxDependency, effect);
                    }}
                }}
                scores[targetSent] = maxDependency;
            }}
            return scores;
        }}

        function calculateEffectScores(interactionMatrix, minDist) {{
            const nSentences = interactionMatrix.length;
            const scores = new Array(nSentences).fill(0);

            for (let sourceSent = 0; sourceSent < nSentences; sourceSent++) {{
                let maxEffect = 0;
                // Ensure we never consider self-interaction (minimum distance is 1)
                const actualMinDist = Math.max(1, minDist);
                for (let targetSent = sourceSent + actualMinDist; targetSent < nSentences; targetSent++) {{
                    const effect = interactionMatrix[sourceSent][targetSent];
                    maxEffect = Math.max(maxEffect, effect);
                }}
                scores[sourceSent] = maxEffect;
            }}
            return scores;
        }}

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
                    // Default coloring based on current mode and current distance
                    if (currentMode === 1) {{
                        span.style.backgroundColor = colorFromScore(currentDependencyScores[i], false);
                    }} else {{
                        span.style.backgroundColor = colorFromScore(currentEffectScores[i], true);
                    }}
                }} else {{
                    // Hover coloring - show ALL interactions (not filtered by distance)
                    if (currentMode === 1) {{
                        // Mode 1 hover: Show how OTHER sentences affect the HOVERED sentence
                        const interactionScore = interactionMatrix[i][hoverIndex];
                        span.style.backgroundColor = colorFromScore(interactionScore, false);
                    }} else {{
                        // Mode 2 hover: Show how HOVERED sentence affects OTHER sentences
                        const interactionScore = interactionMatrix[hoverIndex][i];
                        span.style.backgroundColor = colorFromScore(interactionScore, true);
                    }}
                }}
            }});

            // Update mode indicator
            if (hoverIndex === -1) {{
                if (currentMode === 1) {{
                    modeIndicator.textContent = `Mode 1: Max dependency on sentences ≥${{minDistance}} positions earlier`;
                }} else {{
                    modeIndicator.textContent = `Mode 2: Max effect on sentences ≥${{minDistance}} positions later`;
                }}
            }} else {{
                if (currentMode === 1) {{
                    modeIndicator.textContent = `Hover: How ALL other sentences contribute to sentence ${{hoverIndex + 1}}'s dependencies`;
                }} else {{
                    modeIndicator.textContent = `Hover: How sentence ${{hoverIndex + 1}} affects ALL other sentences`;
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

        function recalculateScores() {{
            // Calculate scores with current distance
            currentDependencyScores = calculateDependencyScores(interactionMatrix, minDistance);
            currentEffectScores = calculateEffectScores(interactionMatrix, minDistance);
        }}

        function initializeVisualization() {{
            // First, calculate the initial scores
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

    Args:
        result: Result dict from detailed_suppression_analysis
        output_path: Where to save the HTML file
        metric: Which metric to use for interactions
        min_distance: Minimum distance for dependency calculation

    Returns:
        Path to generated HTML file
    """
    sentences = result['metadata']['sentences']

    # First create the sentence-to-sentence interaction matrix (log prob changes)
    interaction_matrix = create_sentence_interaction_matrix(result)

    # Then calculate scores by applying masking and maxing to the interaction matrix
    dependency_scores = calculate_dependency_scores(interaction_matrix, min_distance)
    effect_scores = calculate_effect_scores(interaction_matrix, min_distance)

    # Create title and problem info
    problem_num = result['metadata']['problem_num']
    is_correct = result['metadata']['is_correct']
    model_name = result['metadata']['model_name']
    title = f"Sentence Dependencies - {metric}"

    problem_info = {
        'problem_num': problem_num,
        'is_correct': is_correct,
        'model_name': model_name,
        'metric': metric
    }

    # Generate HTML
    html = create_html_visualization(
        sentences, dependency_scores, effect_scores, interaction_matrix,
        title, problem_info, min_distance
    )

    # Save to file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return output_path