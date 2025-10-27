#!/usr/bin/env python3
"""Test script to verify model loading caching behavior."""

import sys
import os
import time

# Add paths
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'whitebox-analyses'))

from pytorch_models.model_loader import get_deepseek_r1

def test_model_loading_times():
    """Test how long model loading takes on repeated calls."""
    model_name = "llama-8b"
    device_map = "auto"

    print(f"Testing model loading caching for {model_name}")
    print("=" * 60)

    # Test 5 consecutive loads
    for i in range(5):
        print(f"\nCall {i+1}/5:")

        start_time = time.time()
        model, tokenizer = get_deepseek_r1(
            model_name=model_name,
            float32=False,
            device_map=device_map
        )
        load_time = time.time() - start_time

        print(f"  Load time: {load_time:.2f}s")
        print(f"  Model type: {type(model)}")
        print(f"  Model device: {next(model.parameters()).device}")

        # Clean up reference (like analyze_text does)
        del model, tokenizer

    print("\n" + "=" * 60)
    print("Expected behavior:")
    print("  - First call: Slow (10-30s) - actual model loading")
    print("  - Subsequent calls: Fast (<1s) - cached model returned")

if __name__ == "__main__":
    test_model_loading_times()