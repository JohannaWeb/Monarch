# Monarch

Expert LLM fine-tuned specifically for Project Falcon / Juntos — the LGBT+ decentralized community platform.

https://media.discordapp.net/attachments/1483007643529384009/1483008892987244604/image.png?ex=69b90729&is=69b7b5a9&hm=2d7d32452fe1a0bfcfcfd7d4237cf549ae00f911b1ad36af2acbd6a889d660ea&=&format=webp&quality=lossless&width=1121&height=868

## Overview

Monarch is a specialized language model trained on Project Falcon's:
- **Philosophy & Values** - sovereignty, transparency, decentralization
- **Architecture** - AT Protocol, DIDs, decentralized identity
- **Domain Models** - servers, channels, members, messages
- **Research** - distributed AI coordination, trust systems

Built with:
- **PyTorch** - for model training
- **LoRA** (Low-Rank Adaptation) - efficient fine-tuning
- **Transformers** - HuggingFace stack
- **ProjectFalcon** - training data source

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Monarch

**Quick: Fine-tune TinyLlama-1.1B in fp16 (recommended for paging benchmarks)**
```bash
python train_tinyllama_fp16.py
```

**Full pipeline** (extract data → prepare → train):
```bash
bash train_monarch.sh
```

Or run each step individually:

**Extract training data:**
```bash
python src/data_extractor.py
```

**Prepare dataset:**
```bash
python src/dataset.py
```

**Train with LoRA:**
```bash
python src/train.py \
  --base-model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --data data/processed/texts.txt \
  --epochs 3 \
  --batch-size 4
```

### 3. Run Inference

**Interactive chat:**
```bash
python src/inference.py --model models/monarch_lora
```

**Single prompt:**
```bash
python src/inference.py \
  --model models/monarch_lora \
  --prompt "What is Juntos?"
```

**Monarch v3 paged inference:**
```bash
python src/inference.py \
  --model models/monarch_lora \
  --mode monarch-v3 \
  --load-in-4bit \
  --window-size 512 \
  --max-hot-tokens 768 \
  --page-size 16 \
  --compression-mode turboquant \
  --prompt "Summarize Bastion's trust pipeline"
```

**Benchmark standard vs paged inference:**
```bash
python src/bench.py \
  --model models/monarch_lora \
  --mode both \
  --preset fast \
  --load-in-4bit \
  --max-new-tokens 128 \
  --page-size 16 \
  --compression-mode turboquant \
  --prompt-file prompt.txt \
  --trace-dir benchmarks/traces \
  --output benchmarks/results.json \
  --json
```

**Generate a report from traces:**
```bash
python src/report.py \
  --trace-dir benchmarks/traces \
  --results benchmarks/results.json
```

## Architecture

```
Monarch/
├── src/
│   ├── data_extractor.py    # Extract from ProjectFalcon
│   ├── dataset.py           # Prepare training data
│   ├── train.py             # Training with LoRA
│   ├── inference.py         # Standard + Monarch v3 inference entrypoint
│   ├── monarch_paging.py    # KV paging, compression, and promotion policy
│   ├── benchmark_monarch.py # Benchmark harness for standard vs paged decode
│   ├── bench.py             # Short benchmark CLI entrypoint
│   └── report.py            # Trace summaries and SVG plot generation
├── data/
│   ├── raw/                 # Extracted from ProjectFalcon
│   │   ├── documentation.jsonl
│   │   ├── code_patterns.jsonl
│   │   ├── conversations.txt
│   │   └── metadata.json
│   └── processed/           # Prepared for training
│       ├── instructions.jsonl
│       ├── texts.txt
│       └── dataset_metadata.json
├── models/
│   ├── monarch_lora/        # TinyLlama fine-tuned weights
│   └── tinyllama_fp16/      # TinyLlama-1.1B fp16 (for paging benchmarks)
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
└── README.md                # This file
```

## Training Details

### LoRA Configuration
- **Rank**: 8 (memory-efficient)
- **Alpha**: 16
- **Dropout**: 0.05
- **Target Modules**: q_proj, v_proj

### Model Options
- **Default (fast)**: TinyLlama-1.1B (minimal resources)
- **Production**: Llama-2-7B (better quality, more VRAM needed)

### Training Hyperparameters
- **Epochs**: 3
- **Batch Size**: 4
- **Learning Rate**: 2e-4
- **Gradient Accumulation**: 4
- **Optimizer**: AdamW (8-bit with CUDA)

## Personality

Monarch embodies the Juntos ethos:

**Direct and precise.** No fluff, no filler.
**Technically rigorous** but never condescending.
**Neutral on content,** principled on safety.
**Confident in analysis,** honest about uncertainty.

## Values

- Sovereignty is non-negotiable. No platform owns identity, data, or intelligence.
- Trust is computed, not assigned.
- Transparency over black boxes — every decision is auditable.
- Decentralization is a value, not a feature.
- The protocol outlasts the product.

## Device Support

Automatically detects and uses:
- **CUDA** if available (faster training, GPU memory optimizations)
- **CPU** fallback (slower but works)

For M1/M2 Mac: Install `pytorch` with Metal support for acceleration.

## Hardware Requirements

### Minimum (TinyLlama-1.1B)
- CPU: 4+ cores
- RAM: 16GB
- Storage: 2GB

### Recommended (Llama-2-7B)
- GPU: RTX 3060+ (12GB VRAM)
- CPU: 8+ cores
- RAM: 32GB
- Storage: 20GB

## Monarch v3 Paging

The repository includes a working `transformers`-based implementation of the Monarch v3 KV paging system:

### Features
- **Hot/cold KV management** with a sliding hot window and sticky-token retention
- **Cold-state compression** using packed 4-bit quantization plus pairwise polar compression
- **Window-based recency policy** — recent tokens stay hot for better performance
- **Manual token-by-token decode loop** — cache policy visible and benchmarkable

### Benchmark Results

**TinyLlama-1.1B fp16 on short sequences (20 prompt + 50 generated tokens):**
- Standard inference: **17.01 tok/sec**, 2112 MB VRAM
- Monarch-v3 paging: **30.42 tok/sec**, 2131 MB VRAM
- **Performance gain: +78.7% throughput, +0.9% VRAM**

### Configuration

Use `--mode monarch-v3` to enable paging. Key flags:

- `--window-size` (default 512) — recency window that stays hot
- `--max-hot-tokens` (default 1024) — hot cache budget in tokens
- `--page-size` (default 16) — physical KV block size for paging
- `--compression-mode` (default turboquant) — cold-cache compression scheme
- `--promotion-threshold` (default 1.0 for disabled) — attention-based promotion threshold
- `--sticky-threshold` (default 999) — promotions needed for sticky tokens
- `--verbose-paging` — prints per-step paging stats

### Recommended Configuration

For models without attention extraction (chat variants):
```bash
--mode monarch-v3 \
  --window-size 512 \
  --max-hot-tokens 1024 \
  --page-size 16 \
  --promotion-threshold 1.0 \
  --sticky-threshold 999
```

This uses pure recency-based paging with minimal overhead.

### Recent Fixes

**Cache Bug Fix (v7f5cc78):** Corrected attention score position indexing in `src/monarch_paging.py`. The `_aggregate_attention_scores` function now correctly maps hot position indices instead of taking first-N scores. This resolves issues with long sequences returning zero attention values.

### Current Scope

This is a working prototype on Hugging Face Transformers using a custom cache backend with page-sized hot/cold storage and TurboQuant-style cold compression. It materializes dense hot tensors per layer but does not patch model internals with a custom fused attention kernel.

## Benchmarking

### Quick Benchmark

Compare standard vs paged inference on the same prompt:
```bash
python src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt "Your prompt here" \
  --max-new-tokens 100 \
  --promotion-threshold 1.0 \
  --sticky-threshold 999
```

### Full Benchmark Suite

Use `src/benchmark_monarch.py` with multiple modes. The harness reports:

- elapsed time and tokens/sec
- peak VRAM usage
- page-ins/page-outs (paged mode only)
- hot vs cold token distribution
- cache statistics

Example with JSON output:
```bash
python src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt-file prompt.txt \
  --max-new-tokens 100 \
  --repeats 3 \
  --mode both \
  --json > benchmarks/results.json
```

### Output Metrics

For each mode (`standard`, `monarch-v3`):
- `elapsed_sec` — total time
- `generated_tokens` — tokens produced
- `tokens_per_sec` — throughput
- `peak_vram_mb` — memory usage
- `desired_hot_tokens`, `cold_tokens` — paging stats (v3 only)
- `promotions`, `page_ins`, `page_outs` — paging activity (v3 only)

The `delta` section shows percentage differences between modes.

### Optional Tracing

Write per-step traces for detailed analysis:
```bash
python src/benchmark_monarch.py \
  --model models/tinyllama_fp16 \
  --prompt-file prompt.txt \
  --trace-dir benchmarks/traces
```

Traces include step-by-step latency, paging deltas, hot/cold token counts, and memory usage.

### Recommended Setup

- Use `--temperature 0` for deterministic results
- Use `--repeats 3` or more to smooth variance
- Match `--max-new-tokens` and prompt across runs for fair comparison

## Customization

Edit `config.yaml` to adjust:
- Base model
- LoRA parameters
- Training hyperparameters
- Data paths

## Integration with ProjectFalcon

Monarch can be integrated with Juntos's AI SIV (Sovereign Integration Vessel):
- Drop `monarch_lora/` into the Juntos model directory
- Update AiContextService to use Monarch
- All responses are signed and auditable

## Known Issues & Fixes

### Cache Bug (Fixed)
The attention score position indexing in `src/monarch_paging.py` was corrected to properly map hot positions to their corresponding attention scores. Previously, the code was taking the first-N attention scores instead of indexing by position, causing zero attention values on long sequences. See commit `7f5cc78` for details.

### Attention Extraction
Chat model variants (TinyLlama-Chat, etc.) don't expose attention weights by design. For paging without attention-based promotion, use `--promotion-threshold 1.0` to disable it and rely on window-based recency selection instead.

## Development

To extend Monarch:
1. Add more training data to `data/raw/`
2. Modify data extraction in `src/data_extractor.py`
3. Adjust instruction templates in `src/dataset.py`
4. Retrain with `train.py` or `train_tinyllama_fp16.py`

To benchmark changes to the paging algorithm:
1. Update paging policy in `src/monarch_paging.py`
2. Run `python src/benchmark_monarch.py` to measure impact
3. Verify throughput and VRAM trade-offs

## License

MIT (same as ProjectFalcon)

---

**Built by Johanna for Project Falcon**
