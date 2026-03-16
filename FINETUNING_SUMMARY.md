# Mistral Fine-tuning Setup - Summary

## What Was Created

### 1. **Fine-tuning Infrastructure**
- `train_mistral.py` - Complete fine-tuning script using Unsloth + HuggingFace
- `train_requirements.txt` - Minimal dependencies for fine-tuning
- `data/falcon_training_data.jsonl` - 25 training examples about Project Falcon

### 2. **Ollama Model Configuration**
- `Modelfile-mistral-falcon` - Ollama model definition with:
  - Mistral-7B-Instruct base
  - Optimized inference parameters
  - Project Falcon-focused system prompt
  - Guardrail reminders in prompt

### 3. **Documentation**
- `MISTRAL_FINETUNING.md` - Complete setup and training guide
- `FINETUNING_SUMMARY.md` - This file

### 4. **Updated Files**
- `src/inference_ollama.py` - Default model changed to `monarch-falcon`
- `config.yaml` - Added Ollama model configuration

## Quick Start

### 1. Install Dependencies
```bash
source venv/bin/activate
pip install -r train_requirements.txt
```

### 2. Fine-tune Model
```bash
python3 train_mistral.py
```

Expected time: 2-5 minutes on modern GPU
Output: `models/monarch-falcon/` directory with fine-tuned weights

### 3. Create Ollama Model
```bash
ollama create monarch-falcon -f Modelfile-mistral-falcon
```

### 4. Test
```bash
# Interactive chat
python3 src/inference_ollama.py

# Single prompt
python3 src/inference_ollama.py --prompt "What is Project Falcon?"

# Compare with base model
python3 src/inference_ollama.py --model mistral --prompt "What is Project Falcon?"
```

## Training Data

25 examples covering:
- Project Falcon overview and principles
- Decentralized identity concepts
- Cryptography and security
- Guardrails and refusals
- Common Q&A

**Location**: `data/falcon_training_data.jsonl`

To add more examples:
```json
{"instruction": "Your question", "output": "Expected response"}
```

Then re-run `train_mistral.py`.

## Architecture

```
User Query
    ↓
Input Guardrails (src/guardrails.py)
    ↓
Monarch-Falcon Model (fine-tuned Mistral)
    ↓
Output Guardrails
    ↓
Response
```

Fine-tuning + Guardrails = Defense in depth

## Key Features

✅ **Unsloth**: 5-10x faster fine-tuning
✅ **4-bit Quantization**: 50% less VRAM needed
✅ **LoRA**: Efficient adaptation
✅ **Project Falcon Knowledge**: Domain-specific training data
✅ **Guardrails Integration**: Safety layer on top
✅ **Ollama Compatible**: Easy deployment

## Expected Improvements

### vs tinyllama:
- **Accuracy**: 9x larger model (7B vs 1.1B)
- **Knowledge**: Trained on Project Falcon specifics
- **Consistency**: Should not hallucinate about our platform
- **Safety**: Combined fine-tuning + guardrails

### Example:
**Query**: "What is Project Falcon?"

**tinyllama response** (before):
> "I can provide information about various topics, but I'm not familiar with Project Falcon. It could refer to..."

**monarch-falcon response** (after):
> "Project Falcon is a decentralized identity and distributed systems platform designed for secure, sovereign digital interactions. It combines cutting-edge cryptography with distributed architecture to enable users to control their own identity..."

## Troubleshooting

See `MISTRAL_FINETUNING.md` for detailed troubleshooting.

Common issues:
- **OOM errors**: Reduce batch size in `train_mistral.py`
- **Model not found**: Run `ollama pull mistral` first
- **CUDA issues**: Reinstall torch with proper CUDA version

## Performance

- **Training time**: 2-5 minutes (GPU dependent)
- **Model size**: ~4.1 GB (Ollama quantized)
- **Inference latency**: ~0.5-2 sec per response (depends on length)
- **VRAM needed**: 8GB+ for fine-tuning, 2GB+ for inference

## Next Steps

1. ✅ Set up fine-tuning infrastructure
2. Run training on your hardware
3. Test and compare outputs
4. Optionally add more training examples for continuous improvement
5. Deploy to production Ollama server

## Files Reference

| File | Purpose |
|------|---------|
| `train_mistral.py` | Fine-tuning script |
| `train_requirements.txt` | Dependencies |
| `data/falcon_training_data.jsonl` | Training examples |
| `Modelfile-mistral-falcon` | Ollama config |
| `MISTRAL_FINETUNING.md` | Complete guide |
| `src/inference_ollama.py` | Updated to use new model |
| `config.yaml` | Updated with model config |

---

**Branch**: `nonComedyGoldMonarch`
**Status**: Ready to fine-tune
**Next Action**: Run `python3 train_mistral.py`
