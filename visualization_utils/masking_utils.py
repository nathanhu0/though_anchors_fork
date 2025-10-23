import numpy as np

def create_causal_mask_and_distances(detailed_results):
    """Create causal masks and distance matrices from detailed analysis results."""

    sentence_boundaries = detailed_results['metadata']['sentence_boundaries']
    n_sentences, n_tokens = detailed_results['kl_matrix_t1'].shape

    # Map each token position to its sentence index (-1 if unassigned)
    in_cot_mask = np.zeros(n_tokens, dtype=bool)
    token_to_sentence = np.full(n_tokens, -1, dtype=int)

    for sent_idx, (start, end) in enumerate(sentence_boundaries):
        if end <= n_tokens:
            token_to_sentence[start:end] = sent_idx
            in_cot_mask[start:end] = True
        if sent_idx == n_sentences - 1:
            token_to_sentence[end:] = sent_idx + 1  # Assign remaining tokens to last sentence

    # Distance between last masked sentence and each token
    sentence_distances = token_to_sentence - np.arange(n_sentences)[:, None]

    # Token distances, number of tokens between last masked token and each token
    sentence_ends = np.array([sb[1] for sb in sentence_boundaries])
    token_distances = np.arange(n_tokens)[None, :] + 1 - sentence_ends[:, None]

    assert ((sentence_distances > 0) == (token_distances > 0)).all()
    causal_mask = sentence_distances > 0

    return {
        'causal_mask': causal_mask,
        'sentence_distances': sentence_distances,
        'token_distances': token_distances,
        'in_cot_mask': in_cot_mask
    }