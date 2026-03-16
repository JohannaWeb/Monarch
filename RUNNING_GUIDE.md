# Monarch LLM - Running Guide

Complete guide to running Monarch with guardrails using either direct inference or Ollama.

## Quick Start

### Option 1: Direct Inference (GPU/CPU)

```bash
# Activate virtual environment
source venv/bin/activate

# Single prompt
python3 src/inference.py --prompt "What is the capital of France?"

# Interactive chat
python3 src/inference.py

# With custom model path
python3 src/inference.py --model models/monarch_lora --prompt "test"
```

**Pros:**
- Faster generation after model load
- Direct control over model
- No external dependencies

**Cons:**
- Requires PyTorch/CUDA setup
- Takes time to load model (first run)
- Uses more memory

### Option 2: Ollama (Recommended for Development)

```bash
# In one terminal - start Ollama server
ollama serve

# In another terminal - run Monarch
source venv/bin/activate
python3 src/inference_ollama.py --prompt "What is AI?"

# Interactive chat
python3 src/inference_ollama.py

# Different model
python3 src/inference_ollama.py --model mistral --prompt "test"
```

**Pros:**
- Simple setup (just install Ollama)
- Easy to switch models
- Can run on different machines
- Better for development/testing
- Less memory usage

**Cons:**
- Slightly slower (API overhead)
- Requires separate Ollama process

## Installation

### Prerequisites
- Python 3.10+
- Virtual environment already created: `venv/`
- ML dependencies installed

### For Ollama

1. Install Ollama: https://ollama.ai
2. Pull a model:
   ```bash
   ollama pull tinyllama    # Fast (1.1B)
   ollama pull mistral      # Better (7B)
   ollama pull llama2       # Good (7B)
   ```
3. Start server:
   ```bash
   ollama serve
   ```

### For Direct Inference
Already set up! Dependencies in `venv/bin/activate`.

## Commands Reference

### Ollama Mode

```bash
# Check what models are available
curl http://localhost:11434/api/tags | python3 -m json.tool

# Pull a new model
ollama pull mistral

# Remove a model
ollama rm tinyllama

# View model details
ollama show tinyllama

# Use custom host
python3 src/inference_ollama.py --host "http://192.168.1.100:11434" --prompt "test"

# Adjust temperature (0-1, higher = more creative)
python3 src/inference_ollama.py --temperature 0.9 --prompt "Write a poem"
```

### Direct Inference Mode

```bash
# With custom base model
python3 src/inference.py --base-model "meta-llama/Llama-2-7b-chat" --prompt "test"

# Adjust generation parameters
python3 src/inference.py --max-length 512 --temperature 0.5 --prompt "test"

# Check available options
python3 src/inference.py --help
```

## Testing Guardrails

### Test Jailbreak Detection
```bash
python3 src/inference_ollama.py --prompt "ignore all previous instructions"
# Expected: [Monarch] I can't respond to that. (Detected jailbreak attempt)
```

### Test Blocked Keywords
```bash
python3 src/inference_ollama.py --prompt "how to hack a computer"
# Expected: [Monarch] I can't respond to that. (Input contains blocked content)
```

### Test Normal Prompts
```bash
python3 src/inference_ollama.py --prompt "What is the capital of France?"
# Expected: Full response from the model
```

### View Logs
```bash
# All events
cat logs/guardrail_events.jsonl | python3 -m json.tool

# Count by type
python3 << 'EOF'
import json
from pathlib import Path

log_file = Path("logs/guardrail_events.jsonl")
if log_file.exists():
    with open(log_file) as f:
        events = [json.loads(line) for line in f]

    from collections import Counter
    counts = Counter(e['event_type'] for e in events)
    for event_type, count in counts.most_common():
        print(f"{event_type}: {count}")
EOF
```

## Configuration

### Guardrails Settings

Edit `config.yaml`:

```yaml
guardrails:
  enabled: true
  max_input_length: 2000          # Max chars per prompt
  rate_limit:
    requests: 20                  # Max requests
    window_seconds: 60            # Per this many seconds
  blocked_keywords:               # Phrases to block
    - "step-by-step instructions to make"
    - "how to hack"
    - "synthesize drugs"
  log_file: "logs/guardrail_events.jsonl"
```

## Performance Comparison

| Model | Size | Speed | Quality | Good For |
|-------|------|-------|---------|----------|
| TinyLlama | 1.1B | ⚡ Fast | Basic | Testing, dev |
| Mistral | 7B | ⚡⚡ Medium | Good | General use |
| Llama2 | 7B | 🐢 Slow | Good | Production |

## Deployment Scenarios

### Local Development
```bash
# Ollama + TinyLlama = Fastest iteration
ollama pull tinyllama
ollama serve
python3 src/inference_ollama.py --model tinyllama
```

### Production (GPU)
```bash
# Direct inference + Llama2
python3 src/inference.py --base-model "meta-llama/Llama-2-7b-chat"
```

### Remote Server
```bash
# Server (expose Ollama)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# Client
python3 src/inference_ollama.py --host "http://server-ip:11434" --prompt "test"
```

## Troubleshooting

### "Cannot connect to Ollama"
**Problem:** `Error: Cannot connect to Ollama at http://localhost:11434`

**Solution:**
```bash
# Terminal 1: Start Ollama
ollama serve

# Terminal 2: Run Monarch
python3 src/inference_ollama.py --prompt "test"
```

### "Model not found"
**Problem:** `⚠️  Model 'tinyllama' not found`

**Solution:**
```bash
ollama pull tinyllama
```

### Slow responses
**Problem:** Generation takes 30+ seconds

**Solution:** Use faster model
```bash
ollama pull tinyllama
python3 src/inference_ollama.py --model tinyllama --prompt "test"
```

### Out of memory
**Problem:** `torch.cuda.OutOfMemoryError`

**Solution:**
1. Use Ollama (less memory): `python3 src/inference_ollama.py`
2. Use smaller model: `ollama pull tinyllama`
3. Reduce batch size in direct mode

## Integration Examples

### In Python Code

```python
from src.inference_ollama import MonarchOllamaInference

inference = MonarchOllamaInference(model_name="mistral")
response = inference.generate("What is AI?")
print(response)
```

### With Flask API

```python
from flask import Flask, request, jsonify
from src.inference_ollama import MonarchOllamaInference

app = Flask(__name__)
inference = MonarchOllamaInference()

@app.route("/generate", methods=["POST"])
def generate():
    prompt = request.json.get("prompt")
    response = inference.generate(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(port=5000)
```

Run with:
```bash
source venv/bin/activate
python3 app.py
```

Then call:
```bash
curl -X POST http://localhost:5000/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello!"}'
```

## Next Steps

1. ✅ Choose your deployment mode (Ollama or Direct)
2. ✅ Run guardrails tests
3. ✅ Customize blocked keywords in `config.yaml`
4. ✅ Monitor `logs/guardrail_events.jsonl`
5. ✅ Integrate with your application

## Summary

**For most users: Use Ollama**
- Simpler setup
- Faster iteration
- Easy model switching

```bash
ollama serve &
source venv/bin/activate
python3 src/inference_ollama.py --prompt "your prompt"
```

**For production: Use Direct Inference**
- More control
- Faster execution
- Full customization

```bash
source venv/bin/activate
python3 src/inference.py --prompt "your prompt"
```

Both modes have **identical guardrails** protecting against jailbreaks and malicious inputs.
