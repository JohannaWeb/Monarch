#!/usr/bin/env python3
"""Fine-tune TinyLlama-1.1B in fp16 for Monarch with attention extraction support."""

import json

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)


def load_training_data(data_path: str = "data/falcon_training_data.jsonl") -> Dataset:
    """Load JSONL training data."""
    examples = []
    with open(data_path, "r") as f:
        for line in f:
            ex = json.loads(line)
            text = f"<s>[INST] {ex['instruction']} [/INST] {ex['output']} </s>"
            examples.append({"text": text})

    dataset = Dataset.from_dict({"text": [ex["text"] for ex in examples]})
    return dataset.map(
        lambda x: {"input_ids": tokenizer(x["text"], truncation=True, max_length=512)["input_ids"]},
        remove_columns=["text"],
    )


def main():
    global tokenizer  # For use in load_training_data

    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    output_dir = "models/tinyllama_fp16"

    print(f"Loading {model_id} in fp16 (no quantization)...")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Load in fp16, NOT quantized - this allows attention extraction
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    print("[OK] Model loaded in fp16")

    # Setup LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    # Load data
    train_dataset = load_training_data()
    print(f"[OK] Loaded {len(train_dataset)} training examples")

    # Training args - quick iteration
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=2,
        warmup_steps=10,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        learning_rate=2e-4,
        save_strategy="epoch",
        logging_steps=1,
        bf16=False,  # Use fp16
        fp16=True,
        optim="paged_adamw_32bit",
    )

    # Train with standard Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    print("Starting training...")
    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save config
    config = {
        "base_model": model_id,
        "dtype": "float16",
        "quantized": False,
        "attention_extraction": True,
    }
    with open(f"{output_dir}/monarch_config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"[OK] Model saved to {output_dir}")
    print(f"[OK] Ready for benchmarking with attention extraction!")


if __name__ == "__main__":
    main()
