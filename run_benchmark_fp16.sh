#!/bin/bash
# Comprehensive benchmark once TinyLlama fp16 training is complete

set -e

mkdir -p benchmarks

echo "=========================================================================="
echo "MONARCH PAGING BENCHMARK - TinyLlama fp16 (Attention Extraction Working)"
echo "=========================================================================="

# SHORT SEQUENCE TEST
echo ""
echo "[1/2] SHORT SEQUENCE TEST (22 prompt, 100 generated)"
python3 src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt "Project Falcon is an LGBT+ decentralized community platform. Explain its mission." \
  --max-new-tokens 100 \
  --repeats 1 \
  --promotion-threshold 0.15 \
  --sticky-threshold 3 \
  --window-size 512 \
  --max-hot-tokens 1024 \
  --json > benchmarks/short_seq.json 2>&1

echo ""
echo "Results (short sequence):"
python3 << 'EOF'
import json
with open('benchmarks/short_seq.json') as f:
    d = json.load(f)
    s = d['standard']
    m = d['monarch-v3']

print(f"Standard:   {s['tokens_per_sec']:.2f} tok/s  |  {s['peak_vram_mb']:.0f} MB")
print(f"Monarch-v3: {m['tokens_per_sec']:.2f} tok/s  |  {m['peak_vram_mb']:.0f} MB")
print(f"  ↳ avg_attention_score: {m['avg_attention_score']:.6f}")
print(f"  ↳ promotions: {m['promotions']:.0f}")
print(f"  ↳ desired_hot: {m['desired_hot_tokens']:.0f}  cold: {m['cold_tokens']:.0f}")
if 'delta' in d:
    delta = d['delta']
    print(f"\nDelta: {delta['tokens_per_sec_pct']:+.1f}% throughput  |  {delta['peak_vram_mb_pct']:+.1f}% VRAM")
EOF

# MEDIUM SEQUENCE TEST
echo ""
echo "[2/2] MEDIUM SEQUENCE TEST (300 prompt, 100 generated)"
python3 -c "print('Project Falcon is an LGBT+ decentralized community platform. ' * 30)" > /tmp/bench_prompt.txt
python3 src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt-file /tmp/bench_prompt.txt \
  --max-new-tokens 100 \
  --repeats 1 \
  --promotion-threshold 0.15 \
  --sticky-threshold 3 \
  --window-size 512 \
  --max-hot-tokens 1024 \
  --json > benchmarks/med_seq.json 2>&1

echo ""
echo "Results (medium sequence):"
python3 << 'EOF'
import json
with open('benchmarks/med_seq.json') as f:
    d = json.load(f)
    s = d['standard']
    m = d['monarch-v3']

print(f"Standard:   {s['tokens_per_sec']:.2f} tok/s  |  {s['peak_vram_mb']:.0f} MB")
print(f"Monarch-v3: {m['tokens_per_sec']:.2f} tok/s  |  {m['peak_vram_mb']:.0f} MB")
print(f"  ↳ avg_attention_score: {m['avg_attention_score']:.6f}")
print(f"  ↳ promotions: {m['promotions']:.0f}")
print(f"  ↳ desired_hot: {m['desired_hot_tokens']:.0f}  cold: {m['cold_tokens']:.0f}")
if 'delta' in d:
    delta = d['delta']
    print(f"\nDelta: {delta['tokens_per_sec_pct']:+.1f}% throughput  |  {delta['peak_vram_mb_pct']:+.1f}% VRAM")
EOF

echo ""
echo "=========================================================================="
echo "Benchmarks complete! Results saved to: benchmarks/"
echo "=========================================================================="
