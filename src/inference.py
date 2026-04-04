#!/usr/bin/env python3
"""Run inference with Monarch, including the v3 paged Transformers path."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

from monarch_paging import MonarchPagingConfig, MonarchTransformersCache


class MonarchInference:
    """Run inference with Monarch."""

    def __init__(
        self,
        model_path: str = "models/monarch_lora",
        base_model: Optional[str] = None,
        load_in_4bit: bool = False,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.load_in_4bit = load_in_4bit and torch.cuda.is_available()

        config_path = self.model_path / "monarch_config.json"
        if config_path.exists() and not base_model:
            with open(config_path, "r") as handle:
                config = json.load(handle)
                base_model = config["base_model"]

        self.base_model = base_model or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.tokenizer = None
        self.model = None

        self.load_model()

    def load_model(self) -> None:
        """Load base model and LoRA weights."""
        print(f"Loading base model: {self.base_model}")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_kwargs = {
            "trust_remote_code": True,
        }

        if self.load_in_4bit:
            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = (
                torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model,
            **model_kwargs,
        )

        print("[OK] Base model loaded")
        print(f"Loading LoRA weights from {self.model_path}")

        self.model = PeftModel.from_pretrained(base, self.model_path)
        if not self.load_in_4bit:
            self.model.to(self.device)
        self.model.eval()

        print("[OK] LoRA weights loaded")

    def _cuda_peak_memory_mb(self) -> float:
        if not torch.cuda.is_available():
            return 0.0
        return torch.cuda.max_memory_allocated() / (1024 * 1024)

    def benchmark_standard(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> Tuple[str, Dict[str, float]]:
        """Run standard generation and return text plus benchmark metrics."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        prompt_token_count = len(self.tokenizer(prompt, truncation=True, max_length=4096)["input_ids"])
        start = time.perf_counter()
        response = self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        elapsed = time.perf_counter() - start
        output_token_count = len(self.tokenizer(response)["input_ids"])
        generated_tokens = max(0, output_token_count - prompt_token_count)
        metrics = {
            "elapsed_sec": elapsed,
            "prompt_tokens": float(prompt_token_count),
            "generated_tokens": float(generated_tokens),
            "tokens_per_sec": generated_tokens / elapsed if elapsed > 0 else 0.0,
            "peak_vram_mb": self._cuda_peak_memory_mb(),
        }
        return response, metrics

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate a response with standard Transformers generation."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def generate_monarch_v3(
        self,
        prompt: str,
        paging: MonarchPagingConfig,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = False,
    ) -> str:
        response, _, _ = self.generate_monarch_v3_with_metrics(
            prompt=prompt,
            paging=paging,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            verbose=verbose,
        )
        return response

    def generate_monarch_v3_with_metrics(
        self,
        prompt: str,
        paging: MonarchPagingConfig,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        verbose: bool = False,
        collect_trace: bool = False,
    ) -> Tuple[str, Dict[str, float], List[Dict[str, float]]]:
        """Generate using Monarch's paged context controller."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=8192,
        ).to(self.device)

        prefill_start = time.perf_counter()
        with torch.no_grad():
            cache = MonarchTransformersCache(
                config=paging,
                num_hidden_layers=self.model.config.num_hidden_layers,
            )
            prefill = self.model(
                **inputs,
                past_key_values=cache,
                use_cache=True,
                output_attentions=True,
                return_dict=True,
            )
        prefill_elapsed = time.perf_counter() - prefill_start

        cache.finalize_prefill(
            input_ids=inputs["input_ids"],
            attentions=prefill.attentions,
        )

        generated_ids = inputs["input_ids"][0].detach().tolist()
        trace: List[Dict[str, float]] = []
        next_token = self._sample_token(
            prefill.logits[:, -1, :],
            temperature=temperature,
            top_p=top_p,
        )

        decode_start = time.perf_counter()
        previous_page_ins = 0
        previous_page_outs = 0
        previous_promotions = 0
        for step_idx in range(max_new_tokens):
            model_input = torch.tensor([[next_token]], device=self.device)

            step_start = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(
                    input_ids=model_input,
                    past_key_values=cache,
                    use_cache=True,
                    output_attentions=True,
                    return_dict=True,
                )
            step_elapsed = time.perf_counter() - step_start

            cache.complete_decode_step(
                token_id=next_token,
                attentions=outputs.attentions,
            )
            generated_ids.append(next_token)
            cache_stats = cache.summary()

            if collect_trace:
                trace.append(
                    {
                        "step": float(step_idx + 1),
                        "sequence_length": float(len(generated_ids)),
                        "token_id": float(next_token),
                        "step_sec": step_elapsed,
                        "step_tokens_per_sec": 1.0 / step_elapsed if step_elapsed > 0 else 0.0,
                        "desired_hot_tokens": float(cache_stats["desired_hot_tokens"]),
                        "resident_hot_tokens": float(cache_stats["resident_hot_tokens"]),
                        "cold_tokens": float(cache_stats["cold_tokens"]),
                        "hot_pages": float(cache_stats["hot_pages"]),
                        "sticky_tokens": float(cache_stats["sticky_tokens"]),
                        "avg_attention_score": float(cache_stats["avg_attention_score"]),
                        "avg_importance_ema": float(cache_stats["avg_importance_ema"]),
                        "promotions_total": float(cache_stats["promotions"]),
                        "promotions_delta": float(cache_stats["promotions"] - previous_promotions),
                        "page_ins_total": float(cache_stats["page_ins"]),
                        "page_ins_delta": float(cache_stats["page_ins"] - previous_page_ins),
                        "page_outs_total": float(cache_stats["page_outs"]),
                        "page_outs_delta": float(cache_stats["page_outs"] - previous_page_outs),
                        "page_hit": 1.0 if cache_stats["page_ins"] == previous_page_ins else 0.0,
                        "page_miss": 1.0 if cache_stats["page_ins"] > previous_page_ins else 0.0,
                        "peak_vram_mb": self._cuda_peak_memory_mb(),
                    }
                )
                previous_page_ins = cache_stats["page_ins"]
                previous_page_outs = cache_stats["page_outs"]
                previous_promotions = cache_stats["promotions"]

            if verbose:
                print(f"[monarch-v3] step={len(generated_ids)} stats={cache_stats}")

            if next_token == self.tokenizer.eos_token_id:
                break

            next_token = self._sample_token(
                outputs.logits[:, -1, :],
                temperature=temperature,
                top_p=top_p,
            )

        decode_elapsed = time.perf_counter() - decode_start
        response = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        generated_tokens = max(0, len(generated_ids) - inputs["input_ids"].shape[1])
        cache_stats = cache.summary()
        metrics: Dict[str, float] = {
            "prefill_sec": prefill_elapsed,
            "decode_sec": decode_elapsed,
            "elapsed_sec": prefill_elapsed + decode_elapsed,
            "prompt_tokens": float(inputs["input_ids"].shape[1]),
            "generated_tokens": float(generated_tokens),
            "tokens_per_sec": generated_tokens / decode_elapsed if decode_elapsed > 0 else 0.0,
            "peak_vram_mb": self._cuda_peak_memory_mb(),
        }
        metrics.update({key: float(value) for key, value in cache_stats.items()})
        return response, metrics, trace

    def _sample_token(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_p: float,
    ) -> int:
        """Sample one token from model logits."""
        if temperature <= 0.0:
            return int(torch.argmax(logits, dim=-1).item())

        scaled = logits / max(temperature, 1e-5)
        probs = torch.softmax(scaled, dim=-1)

        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        nucleus_mask = cumulative <= top_p
        nucleus_mask[..., 0] = True

        filtered_probs = torch.where(nucleus_mask, sorted_probs, torch.zeros_like(sorted_probs))
        filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
        sampled = torch.multinomial(filtered_probs, num_samples=1)
        return int(sorted_indices.gather(-1, sampled).item())

    def chat(
        self,
        mode: str,
        paging: MonarchPagingConfig,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        verbose: bool,
    ) -> None:
        """Interactive chat with Monarch."""
        print("\nMonarch - Interactive Mode")
        print(f"Decode mode: {mode}")
        print("Type 'quit' to exit\n")

        while True:
            prompt = input("You: ").strip()
            if prompt.lower() == "quit":
                print("Goodbye!")
                break
            if not prompt:
                continue

            if mode == "monarch-v3":
                response = self.generate_monarch_v3(
                    prompt,
                    paging=paging,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    verbose=verbose,
                )
            else:
                response = self.generate(
                    prompt,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
            print(f"\nMonarch: {response}\n")


def main():
    """Main inference entry point."""
    parser = argparse.ArgumentParser(description="Run Monarch inference")
    parser.add_argument(
        "--model",
        default="models/monarch_lora",
        help="Path to Monarch LoRA model",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model (if different from config)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt to generate from (interactive if not provided)",
    )
    parser.add_argument(
        "--mode",
        default="standard",
        choices=["standard", "monarch-v3"],
        help="Inference mode",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling threshold",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the base model with bitsandbytes 4-bit quantization when CUDA is available",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=512,
        help="Recent-token window retained in hot VRAM in monarch-v3 mode",
    )
    parser.add_argument(
        "--max-hot-tokens",
        type=int,
        default=768,
        help="Maximum number of hot tokens kept in-memory in monarch-v3 mode",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=16,
        help="Physical KV page size used by the paged cache backend",
    )
    parser.add_argument(
        "--compression-mode",
        default="turboquant",
        choices=["turboquant", "legacy"],
        help="Cold-page compression scheme for monarch-v3",
    )
    parser.add_argument(
        "--promotion-threshold",
        type=float,
        default=0.12,
        help="Attention threshold for promotion and sticky tracking",
    )
    parser.add_argument(
        "--sticky-threshold",
        type=int,
        default=2,
        help="Promotion count before a token becomes sticky",
    )
    parser.add_argument(
        "--importance-decay",
        type=float,
        default=0.92,
        help="Decay applied to historical attention importance",
    )
    parser.add_argument(
        "--initial-sticky-tokens",
        type=int,
        default=16,
        help="Number of prompt tokens seeded into the sticky set by prefill attention",
    )
    parser.add_argument(
        "--verbose-paging",
        action="store_true",
        help="Print per-step paging statistics in monarch-v3 mode",
    )

    args = parser.parse_args()

    paging = MonarchPagingConfig(
        window_size=args.window_size,
        max_hot_tokens=args.max_hot_tokens,
        page_size=args.page_size,
        compression_mode=args.compression_mode,
        attention_promote_threshold=args.promotion_threshold,
        sticky_threshold=args.sticky_threshold,
        importance_decay=args.importance_decay,
        initial_sticky_tokens=args.initial_sticky_tokens,
    )

    inference = MonarchInference(
        model_path=args.model,
        base_model=args.base_model,
        load_in_4bit=args.load_in_4bit,
    )

    if args.prompt:
        if args.mode == "monarch-v3":
            response = inference.generate_monarch_v3(
                args.prompt,
                paging=paging,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                verbose=args.verbose_paging,
            )
        else:
            response = inference.generate(
                args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
        print(f"\n{response}")
    else:
        inference.chat(
            mode=args.mode,
            paging=paging,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            verbose=args.verbose_paging,
        )


if __name__ == "__main__":
    main()
