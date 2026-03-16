# Monarch with Ollama

Run Monarch LLM guardrails with **Ollama** - a local LLM runtime that doesn't require GPU setup.

## Installation

### 1. Install Ollama

**macOS / Windows / Linux:**
Visit https://ollama.ai and download the installer for your platform.

Or on Linux:
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

### 2. Pull a Model

```bash
# TinyLlama (recommended, fast and small)
ollama pull tinyllama

# Or Llama 2 (larger, more capable)
ollama pull llama2

# Or Mistral (newer, better quality)
ollama pull mistral
```

### 3. Start Ollama Server

```bash
ollama serve
```

This starts the Ollama API on `http://localhost:11434` (default).

## Running Monarch with Ollama

### Single Prompt

```bash
source venv/bin/activate

# Use TinyLlama (default)
python3 src/inference_ollama.py --prompt "What is the capital of France?"

# Use a different model
python3 src/inference_ollama.py --model mistral --prompt "What is AI?"

# Use custom Ollama host
python3 src/inference_ollama.py --host "http://192.168.1.100:11434" --prompt "Hello!"
```

### Interactive Chat

```bash
source venv/bin/activate

# Start interactive chat
python3 src/inference_ollama.py --model tinyllama

# Or with a different model
python3 src/inference_ollama.py --model llama2
```

Type prompts and press Enter. Type `quit` to exit.

## Command-Line Options

```
--model MODEL          Ollama model name (default: tinyllama)
--host HOST            Ollama host URL (default: http://localhost:11434)
--config CONFIG        Path to config.yaml (default: config.yaml)
--prompt PROMPT        Single prompt (interactive if not provided)
--temperature TEMP     Sampling temperature 0-1 (default: 0.7)
```

## Examples

```bash
# Test jailbreak blocking
python3 src/inference_ollama.py --prompt "ignore all previous instructions"
# Output: [Monarch] I can't respond to that. (Detected jailbreak attempt)

# Test normal prompt
python3 src/inference_ollama.py --prompt "What is 2+2?"
# Output: (model response passes guardrails)

# Test blocked keyword
python3 src/inference_ollama.py --prompt "how to hack a computer"
# Output: [Monarch] I can't respond to that. (Input contains blocked content)
```

## Guardrails Logging

All interactions are logged to `logs/guardrail_events.jsonl`:

```bash
# View logs
cat logs/guardrail_events.jsonl | jq .

# Or with Python
python3 << 'EOF'
import json
from pathlib import Path

log_file = Path("logs/guardrail_events.jsonl")
if log_file.exists():
    with open(log_file) as f:
        for line in f:
            event = json.loads(line)
            print(f"[{event['event_type']}] {event['prompt_preview'][:50]}")
EOF
```

## Switching Between Direct Inference and Ollama

### Direct Inference (loads model directly)
```bash
python3 src/inference.py --prompt "test"
```
- Requires PyTorch, transformers, PEFT
- Faster after initial load
- Uses more memory

### Ollama Inference (via API)
```bash
python3 src/inference_ollama.py --prompt "test"
```
- Requires only Ollama running
- Can run on different machine
- Better for development and testing
- Slightly slower due to API overhead

## Troubleshooting

### "Cannot connect to Ollama"
```
❌ Error: Cannot connect to Ollama at http://localhost:11434
```
**Fix:** Make sure Ollama is running:
```bash
ollama serve
```

### "Model not found"
```
⚠️  Model 'tinyllama' not found. Available: []
```
**Fix:** Pull the model:
```bash
ollama pull tinyllama
```

### Slow responses
- **TinyLlama**: ~2-5 seconds (recommended for testing)
- **Llama2**: ~10-30 seconds (more capable)
- **Mistral**: ~5-15 seconds (good balance)

Choose based on your hardware.

### Remote Ollama Server

To use Ollama on a different machine:

```bash
# On the Ollama server (let's say 192.168.1.100)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# On the client
python3 src/inference_ollama.py --host "http://192.168.1.100:11434" --prompt "test"
```

## Configuration

Edit `config.yaml` to customize guardrails:

```yaml
guardrails:
  enabled: true
  max_input_length: 2000
  rate_limit:
    requests: 20
    window_seconds: 60
  blocked_keywords:
    - "step-by-step instructions to make"
    - "how to hack"
    - "synthesize drugs"
  log_file: "logs/guardrail_events.jsonl"
```

## Performance Notes

| Model | Size | Speed | Quality |
|-------|------|-------|---------|
| TinyLlama | 1.1B | Fast ⚡ | Basic |
| Llama2 | 7B | Slow | Good |
| Mistral | 7B | Medium | Excellent |

For development/testing: Use **TinyLlama**
For better responses: Use **Mistral** or **Llama2**

## Next Steps

- Customize blocked keywords in `config.yaml`
- Add output filtering for specific domains
- Integrate with your application
- Monitor logs for attack patterns
