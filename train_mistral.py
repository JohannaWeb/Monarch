#!/usr/bin/env python3
"""Fine-tune Mistral-7B with Project Falcon knowledge using Unsloth."""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import TrainingArguments, TextIteratorStreamer
from unsloth import FastLanguageModel, get_chat_template
from trl import SFTTrainer

# Configuration
MODEL_NAME = "mistral-7b-instruct-v0.1"
MAX_SEQ_LENGTH = 2048
DTYPE = torch.bfloat16
LOAD_IN_4BIT = True
LORA_RANK = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
OUTPUT_DIR = "models/monarch-falcon"
DATA_FILE = "data/falcon_training_data.jsonl"

def load_training_data(data_file):
    """Load JSONL training data."""
    data = []
    with open(data_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    print(f"Loaded {len(data)} training examples")
    return data

def format_chat_template(examples):
    """Format examples for chat template."""
    texts = []
    for example in examples:
        text = f"""<s>[INST] {example['instruction']} [/INST]
{example['output']}</s>"""
        texts.append(text)
    return {"text": texts}

def setup_model():
    """Load and configure Mistral with Unsloth."""
    print("Loading Mistral-7B with Unsloth...")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-instruct-v0.1",
        max_seq_length=MAX_SEQ_LENGTH,
        dtype=DTYPE,
        load_in_4bit=LOAD_IN_4BIT,
    )

    print("Setting up LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    return model, tokenizer

def main():
    """Fine-tune Mistral on Project Falcon knowledge."""

    # Load training data
    training_data = load_training_data(DATA_FILE)
    dataset = Dataset.from_dict({
        "text": [
            f"[INST] {ex['instruction']} [/INST]\n{ex['output']}"
            for ex in training_data
        ]
    })

    print(f"Training dataset size: {len(dataset)}")

    # Setup model
    model, tokenizer = setup_model()

    # Training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=100,
        learning_rate=2e-4,
        bf16=True,
        logging_steps=10,
        save_steps=len(dataset) // 4,
        save_total_limit=3,
        push_to_hub=False,
    )

    # Trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        args=training_args,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    print("Starting fine-tuning...")
    trainer.train()

    # Save model
    print(f"Saving fine-tuned model to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("\n✅ Fine-tuning complete!")
    print(f"Model saved to: {OUTPUT_DIR}")
    print("\nNext steps:")
    print("1. Convert to Ollama format")
    print("2. Test with src/inference_ollama.py --model monarch-falcon")

if __name__ == "__main__":
    main()
