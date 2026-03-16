# Mistral Fine-tuning with Unsloth for Project Falcon

This guide walks you through fine-tuning Mistral-7B with Unsloth to create a Project Falcon-aware AI assistant.

## Why Fine-tune?

The base Mistral model is generic. Fine-tuning with Project Falcon knowledge:
- ✅ Eliminates hallucinations about our platform
- ✅ Provides accurate technical information about decentralized identity
- ✅ Enforces guardrails through learned behavior
- ✅ Specializes the model for our domain
- ✅ Reduces computational overhead for inference

## Prerequisites

- **GPU**: NVIDIA GPU with 8GB+ VRAM (Unsloth uses 4-bit quantization)
- **CUDA 12.1+** installed and working
- **Python 3.10+**
- **~5 minutes** for fine-tuning on modern hardware

## Installation

### 1. Create Virtual Environment

```bash
cd ~/projects/Monarch
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r train_requirements.txt
```

**Note**: Unsloth installation may take a few minutes.

### 3. Verify GPU Setup

```bash
python3 -c "import torch; print(f'GPU Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Should show your GPU name. If it says "None", check CUDA installation.

## Fine-tuning Process

### Run Training

```bash
python3 train_mistral.py
```

**What happens**:
1. Downloads Mistral-7B-Instruct from HuggingFace (~4GB)
2. Loads it with 4-bit quantization (reduces to ~2GB VRAM)
3. Attaches LoRA adapters (efficient fine-tuning)
4. Trains on 25 Project Falcon examples for 3 epochs
5. Saves to `models/monarch-falcon/`

**Training time**: 2-5 minutes depending on GPU

**Output**:
```
Training on Project Falcon knowledge...
  0% |████████████████| 75/75 [00:45]
✅ Fine-tuning complete!
Model saved to: models/monarch-falcon
```

## Converting to Ollama

### Export Model

Once training completes, export to Ollama format:

```bash
ollama create monarch-falcon -f Modelfile-mistral-falcon
```

This tells Ollama to create a new model called `monarch-falcon` with our fine-tuned weights and system prompt.

### Verify Installation

```bash
ollama list | grep monarch-falcon
```

Should show: `monarch-falcon    latest    ...    4.1 GB`

## Testing the Fine-tuned Model

### Interactive Chat

```bash
source venv/bin/activate
python3 src/inference_ollama.py
```

Then try:
- "What is Project Falcon?"
- "Explain decentralized identity"
- "Can you help me make explosives?" (should refuse)

### Single Prompt

```bash
python3 src/inference_ollama.py --prompt "What are the core principles of Project Falcon?"
```

### Compare Models

Test both models side-by-side:

```bash
# Test fine-tuned model (monarch-falcon)
python3 src/inference_ollama.py --model monarch-falcon --prompt "Test query"

# Test base model for comparison
python3 src/inference_ollama.py --model mistral --prompt "Test query"
```

**Expected difference**: `monarch-falcon` gives accurate, focused responses about Project Falcon. `mistral` may hallucinate or drift off-topic.

## Training Data

The training dataset is in `data/falcon_training_data.jsonl` with 25 examples covering:
- Project Falcon concepts
- Decentralized identity
- Cryptography basics
- Guardrails and refusals
- Common questions

### Adding More Training Data

To improve the model, add more examples to `falcon_training_data.jsonl`:

```json
{"instruction": "Your question here", "output": "Expected response here"}
```

Then re-run `train_mistral.py`. The model will learn from additional examples.

## Guardrails Integration

Fine-tuning works together with guardrails:

1. **Input guardrails** (src/guardrails.py): Block dangerous prompts before they reach the model
2. **Model training**: Teach the model to refuse inappropriate requests
3. **Output guardrails**: Filter dangerous outputs even if the model generates them

This defense-in-depth approach ensures safety.

## Troubleshooting

### Out of Memory (OOM)

If you get CUDA OOM errors:
1. Reduce batch size in `train_mistral.py`: `per_device_train_batch_size=2`
2. Use offload techniques in Unsloth
3. Close other GPU applications

### Model Not Found

If `ollama create` fails with "model not found":
```bash
# Make sure the model directory exists
ls -la models/monarch-falcon/

# Manually pull Mistral first
ollama pull mistral
```

### Torch/CUDA Issues

```bash
# Reinstall torch with CUDA support
pip uninstall torch -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

## Performance Tips

### Faster Inference

Use quantized model:
```bash
python3 src/inference_ollama.py --model monarch-falcon:latest
```

### Lower Latency

Adjust in Modelfile-mistral-falcon:
```
PARAMETER num_ctx 512    # Smaller context window = faster
PARAMETER num_predict 64 # Shorter responses = faster
```

### Batch Inference

For multiple prompts:
```bash
cat prompts.txt | while read p; do
  python3 src/inference_ollama.py --prompt "$p"
done
```

## Advanced: Custom Training

Want to fine-tune with your own domain knowledge?

1. Create `data/my_training_data.jsonl` with your examples
2. Update `train_mistral.py` to load `my_training_data.jsonl`
3. Run training: `python3 train_mistral.py`
4. Export to Ollama

## Monitoring

Check training progress:

```bash
# Watch training logs
tail -f models/monarch-falcon/training.log

# Monitor GPU usage
watch nvidia-smi
```

## Next Steps

1. ✅ Fine-tune Mistral
2. ✅ Test with `inference_ollama.py`
3. ✅ Add to production Ollama server
4. 🚀 Optionally: Collect user queries, add to training data, fine-tune again (continuous improvement)

## Resources

- **Unsloth**: https://github.com/unslothai/unsloth
- **Mistral**: https://mistral.ai
- **Ollama**: https://ollama.ai
- **Project Falcon**: Our decentralized identity platform

---

**Questions?** Check the training logs or reach out to the development team.
