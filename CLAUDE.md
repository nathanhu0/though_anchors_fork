# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Dependencies
Activate virtual environment and install Python dependencies:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Important**: Always activate the virtual environment before running any Python commands:
```bash
source .venv/bin/activate
```

### Main Scripts
This repository contains four main scripts for LLM reasoning analysis:

1. **Generate rollouts**:
   ```bash
   python generate_rollouts.py -m "deepseek/deepseek-r1-distill-qwen-14b" -b correct -r default
   ```
   Creates reasoning rollouts dataset. Key parameters:
   - `-m/--model`: LLM model to use
   - `-b/--base_solution_type`: correct/incorrect base solutions
   - `-r/--rollout_type`: default/forced_answer rollouts
   - `-sp/--split`: train/test dataset split

2. **Analyze rollouts**:
   ```bash
   python analyze_rollouts.py
   ```
   Processes generated rollouts, calculates importance metrics (forced answer, resampling, counterfactual importance)

3. **Step attribution**:
   ```bash
   python step_attribution.py
   ```
   Computes sentence-to-sentence counterfactual importance scores

4. **Generate plots**:
   ```bash
   python plots.py
   ```
   Creates figures and visualizations from the analysis

### Miscellaneous Scripts
- `misc-experiments/`: Contains experimental analysis scripts
- `misc-scripts/push_hf_dataset.py`: Uploads dataset to HuggingFace

## Architecture

### Core Analysis Pipeline
The repository implements a multi-step pipeline for analyzing LLM reasoning:
1. **Rollout Generation** → 2. **Analysis & Metrics** → 3. **Attribution** → 4. **Visualization**

### Key Components

**Data Processing**:
- `utils.py`: Core utilities for reasoning trace analysis, answer extraction, and chunk processing
- `prompts.py`: LLM prompts including `DAG_PROMPT` for auto-labeling sentence functions
- `selected_problems.json`: Curated MATH problems (25%-75% accuracy range) sorted by sentence length

**Importance Calculation**:
- **Forced Answer Importance**: Measures influence on final answer
- **Resampling Importance**: Evaluates consistency across multiple generations
- **Counterfactual Importance**: Assesses impact of removing specific sentences

**White-box Analysis**:
- `whitebox-analyses/attention_analysis/`: Attention pattern analysis and receiver heads
- `whitebox-analyses/pytorch_models/`: Model implementations for attention suppression
- `whitebox-analyses/scripts/`: Analysis scripts

### Attention Suppression Pipeline

**Core Function**: `get_suppression_KL_matrix()` - whitebox-analyses/attention_analysis/attn_supp_funcs.py:29

**Pipeline Flow**:
1. **Data Loading**: `get_problem_text_sentences()` → Load reasoning text and sentence boundaries
2. **Baseline Generation**: `analyze_text_get_p_logits()` → Get normal probability distributions
3. **Suppression Loop**: For each sentence, mask its tokens and re-run model
4. **KL Divergence**: Compare baseline vs suppressed distributions at every token position
5. **Aggregation**: Average KL scores within sentence boundaries to build influence matrix

**Attention Masking Mechanism**:
- **Hook Installation**: Replaces forward methods in all attention modules (e.g., 48 layers × 40 heads for qwen-14b)
- **Token Masking**: Sets attention weights to `-inf` for specified token ranges
- **Effect**: Masked sentences become completely invisible to the model across ALL layers
- **Result**: Measures counterfactual importance - "What if this sentence never existed?"

**Key Functions**:
- `analyze_text_get_p_logits()` - logits_funcs.py:189: Runs model with/without masking
- `apply_qwen_attn_mask_hooks()` - hooks.py:270: Installs attention masking
- `compress_logits_top_p()` - logits_funcs.py:73: Compresses probability distributions for efficiency
- `calculate_kl_divergence_sparse()` - attn_supp_funcs.py:132: Computes KL divergence between distributions

**Usage**:
```bash
# Single problem analysis
python whitebox-analyses/scripts/plot_suppression_matrix.py --problem-num 1591

# Batch processing
python whitebox-analyses/scripts/prep_suppression_mtxs.py
```

### Interactive Sentence Dependency Visualization

**Location**: `visualization_utils/` - Interactive HTML/JS system for exploring sentence dependencies

**Core Function**: `visualize_result()` - visualization_utils/text_viz.py:507

**Quick Start**:
```bash
cd visualization_utils
source ../.venv/bin/activate
python example_usage.py
# Open generated HTML file in browser or use HTTP server:
python -m http.server 8001
```

**Features**:
- **Dual Visualization Modes**:
  - **Mode 1 (RED)**: Shows how much each sentence depends on earlier sentences
  - **Mode 2 (BLUE)**: Shows how much each sentence affects future sentences
- **Interactive Controls**:
  - **Distance Slider**: Configure minimum sentence distance (0-20) for dependency calculation
  - **Hover Interactions**: Shows ALL sentence-to-sentence interactions regardless of distance
  - **Mode Switching**: Toggle between dependency and effect views
- **Metrics Display**:
  - **Log Probability Changes**: `log(prob_original) - log(prob_suppressed)`
  - **Multiplicative Effects**: `exp(log_change)` showing probability ratios
  - **Color Bar**: Real values with scientific notation
- **Problem Context**: Displays original math problem question with reasoning analysis

**Key Functions**:
- `create_sentence_interaction_matrix()` - text_viz.py:6: Converts token-level suppression effects to sentence-level
- `calculate_dependency_scores()` - text_viz.py:54: Computes max dependency on far-earlier sentences
- `calculate_effect_scores()` - text_viz.py:78: Computes max effect on far-future sentences
- `load_problem_text()` - text_viz.py:100: Loads questions from selected_problems.json

**Usage Example**:
```python
from visualization_utils.text_viz import visualize_result
from visualization_utils.masking_utils import create_causal_mask_and_distances

# Load analysis result (cached)
result = analyze_suppression_detailed(problem_num=1591, model_name="llama-8b", is_correct=True)
masking_info = create_causal_mask_and_distances(result)
result.update(masking_info)

# Create interactive visualization
visualize_result(result, output_path="sentence_dependencies.html", min_distance=3)
```

**Interpretation**:
- **White**: No dependency (zero or negative log prob changes)
- **Light colors**: Weak dependencies
- **Dark colors**: Strong dependencies (suppression significantly hurts probability)
- **Hover mode**: Shows complete interaction picture ignoring distance filters

### Caching System
The codebase extensively uses the `pkld` decorator for function result caching:
- Automatically caches expensive computations to disk
- Handles unhashable arguments (numpy arrays, dicts, lists)
- Cache files stored in `.pkljar/` directories
- Clear cache with `function_name.clear()`

### Environment Setup
Requires API keys in `.env` file:
- `OPENROUTER_API_KEY`
- `NOVITA_API_KEY`
- `TOGETHER_API_KEY`
- `FIREWORKS_API_KEY`

## Key Data Structures

**Reasoning Traces**: JSON objects containing:
- `chunks`: Individual reasoning sentences/steps
- `chunks_labeled`: Sentences with function tags (e.g., uncertainty management)
- Importance scores for each chunk
- Answer extraction and verification results

**ImportanceArgs Class**: Configuration for importance calculations including similarity thresholds, vocabulary settings, and smoothing parameters.

## Development Notes

- No formal testing framework - this is a research codebase focused on analysis
- Heavy use of matplotlib/seaborn for visualization
- Multiprocessing support for parallel analysis
- Integration with sentence transformers for semantic similarity
- Support for multiple LLM providers (OpenRouter, Novita, Together, Fireworks)

## Code Style Preferences

**User prefers clean, minimal code that achieves desired functionality:**
- Avoid over-engineering and unnecessary complexity
- OOP or helper functions are acceptable ONLY if they significantly reduce code complexity
- Keep implementations simple and direct