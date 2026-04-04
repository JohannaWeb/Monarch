#!/bin/bash
# Full benchmark: Standard vs Monarch-v3 with working attention extraction

set -e

echo "========================================"
echo "Benchmarking TinyLlama fp16 (Attention Working)"
echo "========================================"

# Test short sequence
echo ""
echo "1. SHORT SEQUENCE (100 prompt tokens, 100 new tokens)"
echo "---"
python3 src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt "Project Falcon is a decentralized community platform for LGBT+ individuals. Explain its mission. " \
  --max-new-tokens 100 \
  --repeats 1 \
  --promotion-threshold 0.15 \
  --sticky-threshold 3 \
  --window-size 512 \
  --max-hot-tokens 1024 \
  --page-size 16 \
  --json > benchmarks/short_seq_fp16.json

# Extract results
python3 << 'EOF'
import json
with open('benchmarks/short_seq_fp16.json') as f:
    data = json.load(f)
    std = data['standard']
    m3 = data['monarch-v3']
    delta = data['delta']

print(f"Standard:   {std['tokens_per_sec']:.2f} tok/s, {std['peak_vram_mb']:.0f} MB")
print(f"Monarch-v3: {m3['tokens_per_sec']:.2f} tok/s, {m3['peak_vram_mb']:.0f} MB")
print(f"  Avg attention: {m3['avg_attention_score']:.4f}")
print(f"  Promotions: {m3['promotions']:.0f}")
print(f"  Cold tokens: {m3['cold_tokens']:.0f}")
print(f"Delta: {delta['tokens_per_sec_pct']:+.1f}% throughput, {delta['peak_vram_mb_pct']:+.1f}% VRAM")
EOF

# Test medium sequence
echo ""
echo "2. MEDIUM SEQUENCE (300 prompt tokens, 100 new tokens)"
echo "---"
python3 -c "print('Project Falcon is a decentralized community platform. ' * 30)" > /tmp/med_prompt.txt
python3 src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt-file /tmp/med_prompt.txt \
  --max-new-tokens 100 \
  --repeats 1 \
  --promotion-threshold 0.15 \
  --sticky-threshold 3 \
  --window-size 512 \
  --max-hot-tokens 1024 \
  --page-size 16 \
  --json > benchmarks/med_seq_fp16.json

python3 << 'EOF'
import json
with open('benchmarks/med_seq_fp16.json') as f:
    data = json.load(f)
    std = data['standard']
    m3 = data['monarch-v3']
    delta = data.get('delta', {})

print(f"Standard:   {std['tokens_per_sec']:.2f} tok/s, {std['peak_vram_mb']:.0f} MB")
print(f"Monarch-v3: {m3['tokens_per_sec']:.2f} tok/s, {m3['peak_vram_mb']:.0f} MB")
print(f"  Avg attention: {m3['avg_attention_score']:.4f}")
print(f"  Promotions: {m3['promotions']:.0f}")
print(f"  Cold tokens: {m3['cold_tokens']:.0f}")
if delta:
    print(f"Delta: {delta['tokens_per_sec_pct']:+.1f}% throughput, {delta['peak_vram_mb_pct']:+.1f}% VRAM")
EOF

echo ""
echo "========================================"
echo "Benchmark complete! Results saved to benchmarks/"
echo "========================================"
