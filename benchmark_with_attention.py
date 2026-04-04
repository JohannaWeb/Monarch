#!/usr/bin/env python3
"""Benchmark Monarch paging with attention extraction working."""

import sys
sys.path.insert(0, 'src')

from inference import MonarchInference
from monarch_paging import MonarchPagingConfig

def test_attention_extraction():
    """Verify that attention weights are extracted."""
    print("=" * 60)
    print("Testing Attention Extraction (fp16 TinyLlama)")
    print("=" * 60)

    inference = MonarchInference(model_path="models/tinyllama_fp16")
    paging = MonarchPagingConfig(
        window_size=512,
        max_hot_tokens=1024,
        attention_promote_threshold=0.15,
        sticky_threshold=3,
    )

    prompt = "Project Falcon is a decentralized platform. " * 5

    response, metrics, trace = inference.generate_monarch_v3_with_metrics(
        prompt=prompt,
        paging=paging,
        max_new_tokens=50,
        verbose=False,
        collect_trace=True,
    )

    print(f"\nGenerated: {metrics.get('generated_tokens', 0):.0f} tokens")
    print(f"Tokens/sec: {metrics.get('tokens_per_sec', 0):.2f}")
    print(f"Peak VRAM: {metrics.get('peak_vram_mb', 0):.1f} MB")
    print(f"Avg attention score: {metrics.get('avg_attention_score', 0):.4f}")
    print(f"Promotions: {metrics.get('promotions', 0):.0f}")
    print(f"Cold tokens: {metrics.get('cold_tokens', 0):.0f}")
    print(f"Desired hot: {metrics.get('desired_hot_tokens', 0):.0f}")

    # Check if attention is working
    if metrics.get('avg_attention_score', 0) > 0:
        print("\n✅ ATTENTION EXTRACTION WORKING")
    else:
        print("\n❌ Attention scores still 0 - check model configuration")

    print(f"\nResponse preview: {response[:150]}")

if __name__ == "__main__":
    test_attention_extraction()
