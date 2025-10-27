 git clone https://huggingface.co/datasets/uzaymacar/math-rollouts /nlp/scr/nathu/thought-anchors/math-rollouts


 Perfect! The structure looks exactly as expected. Here are the commands to start exploring the attention suppression functionality:

  Quick Test Commands

  source .venv/bin/activate

  # 1. Test a single problem suppression matrix (simplest start)
  python whitebox-analyses/scripts/plot_suppression_matrix.py --problem-num 1591

#12802451-54
ebatch kl_supp_cor slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b --correct-only"
ebatch kl_supp_inc slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b --incorrect-only"
ebatch kl_supp_cumm_cor slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b --cumulative --correct-only"
ebatch kl_supp_cumm_inc slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b --cumulative --incorrect-only"

#12840730
ebatch supp_prompt slconf_sphinx_b "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b"

#12840731
ebatch cum_supp_prompt slconf_sphinx_b "python whitebox-analyses/nathan_scripts/detailed_suppression_analysis.py --model-name llama-8b --cumulative"


python -m pdb whitebox-analyses/nathan_scripts/detailed_interaction_analysis.py --model-name llama-8b --correct-only --amplify --amplify-factor 5 --max-problems 1

12841334
ebatch supp_prompt slconf_sphinx_b "python whitebox-analyses/nathan_scripts/detailed_interaction_analysis.py --model-name llama-8b"

#12841858, 99, 60
ebatch supp_prompt slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_interaction_analysis.py --model-name llama-8b --cumulative"
ebatch supp_prompt slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_interaction_analysis.py --model-name llama-8b --amplify --amplify-factor 2"
ebatch supp_prompt slconf_sphinx "python whitebox-analyses/nathan_scripts/detailed_interaction_analysis.py --model-name llama-8b --amplify --amplify-factor 5"


python visualization_utils/text_viz.py