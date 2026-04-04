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
Full pipeline (extract data → prepare → train):
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
│   └── monarch_lora/        # Trained LoRA weights
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

The repository now includes an experimental `transformers`-based implementation of the Monarch v3 white paper:

- **Hot/cold KV management** with a sliding hot window and sticky-token retention
- **Cold-state compression** using packed 4-bit value quantization plus pairwise polar compression for keys
- **Attention-weighted promotion** based on prompt and decode-time attention signals
- **Manual token-by-token decode loop** so cache policy is visible and benchmarkable

Use `--mode monarch-v3` to enable it. Useful flags:

- `--load-in-4bit` loads the base model with bitsandbytes 4-bit weights when CUDA is available
- `--window-size` sets the recency window that stays hot
- `--max-hot-tokens` caps the hot cache budget
- `--page-size` controls the physical KV block size used for page-in/page-out
- `--compression-mode` selects `turboquant` or the older `legacy` cold-cache compressor
- `--promotion-threshold` controls when attention makes a token promotable/sticky
- `--sticky-threshold` sets how many promotions are needed before a token becomes permanently hot
- `--verbose-paging` prints per-step paging counters during generation

Current scope: this implementation is a working prototype on top of Hugging Face Transformers. It now uses a custom cache backend with page-sized hot/cold storage and a default internal TurboQuant-style cold-page compressor, but it still materializes dense hot tensors per layer and does not patch model internals with a custom fused attention kernel.

## Benchmarking

Use `src/bench.py` or `src/benchmark_monarch.py` to compare `standard` and `monarch-v3` on the same prompt. The harness reports average metrics such as:

- elapsed seconds
- generated tokens
- decode tokens per second
- peak VRAM in MB
- page-ins and page-outs for `monarch-v3`
- resident vs desired hot token counts

If you pass `--trace-dir`, `monarch-v3` also writes per-step JSONL traces with:

- decode step latency
- per-step tokens/sec
- page-in/page-out deltas
- promotion deltas
- hot/cold token counts
- hot page count
- running peak VRAM

If you run with `--mode both`, the JSON output now includes:

- `standard`
- `monarch-v3`
- `delta`

You can also write the aggregate result bundle to disk with `--output benchmarks/results.json`.

Presets:

- `--preset fast` for a quick single-pass benchmark
- `--preset accurate` for longer, more stable comparisons

Use `src/report.py --trace-dir benchmarks/traces` to print:

- tokens/sec
- latency per token
- memory retention stats
- promotion frequency

It also emits a static dashboard at `benchmarks/traces/report/index.html` plus simple SVG plots for:

- token index vs latency
- attention score decay
- page hits vs misses

Recommended benchmark setup:

- use `--temperature 0` for deterministic comparisons
- keep the same prompt and `--max-new-tokens` across modes
- use `--repeats 3` or more to smooth out noise

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

## Development

To extend Monarch:
1. Add more training data to `data/raw/`
2. Modify data extraction in `src/data_extractor.py`
3. Adjust instruction templates in `src/dataset.py`
4. Retrain with `train.py`

## License

MIT (same as ProjectFalcon)

---

**Built by Johanna for Project Falcon**
