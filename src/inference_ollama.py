#!/usr/bin/env python3
"""Run inference with Monarch via Ollama."""

import requests
import yaml
import argparse
import sys
from pathlib import Path
from typing import Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from guardrails import MonarchGuardrails


class MonarchOllamaInference:
    """Run inference with Monarch via Ollama API."""

    def __init__(
        self,
        model_name: str = "monarch-falcon",
        ollama_host: str = "http://localhost:11434",
        config_file: str = "config.yaml",
    ):
        """Initialize Ollama inference.

        Args:
            model_name: Name of the model in Ollama (default: monarch-falcon, fine-tuned Mistral)
            ollama_host: URL of Ollama API (default: http://localhost:11434)
            config_file: Path to config.yaml
        """
        self.model_name = model_name
        self.ollama_host = ollama_host.rstrip("/")
        self.api_url = f"{self.ollama_host}/api/generate"

        # Load guardrails config from config.yaml
        config_file_path = Path(config_file)
        if config_file_path.exists():
            with open(config_file_path, 'r') as f:
                full_config = yaml.safe_load(f)
                guardrails_config = full_config.get("guardrails", {})
        else:
            guardrails_config = {}

        self.guardrails = MonarchGuardrails(guardrails_config)

        print(f"Ollama Inference initialized")
        print(f"  Model: {self.model_name}")
        print(f"  Host: {self.ollama_host}")
        self._check_ollama_connection()

    def _check_ollama_connection(self) -> None:
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(
                f"{self.ollama_host}/api/tags",
                timeout=5
            )
            if response.status_code == 200:
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                if self.model_name in model_names or any(self.model_name in m for m in model_names):
                    print(f"[OK] Model '{self.model_name}' found in Ollama")
                else:
                    print(f"⚠️  Model '{self.model_name}' not found. Available: {model_names}")
            else:
                print(f"⚠️  Could not list models from Ollama")
        except requests.exceptions.ConnectionError:
            print(f"❌ Error: Cannot connect to Ollama at {self.ollama_host}")
            print(f"   Make sure Ollama is running: ollama serve")
            sys.exit(1)

    def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """Generate response from prompt via Ollama."""
        # 1. Input check
        result = self.guardrails.check_input(prompt)
        if not result.allowed:
            self.guardrails.log_event("BLOCKED_INPUT", prompt, result.reason)
            return f"[Monarch] I can't respond to that. ({result.reason})"

        # 2. Generate via Ollama
        try:
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }

            response = requests.post(
                self.api_url,
                json=payload,
                timeout=300,
            )

            if response.status_code != 200:
                error_msg = f"Ollama error: {response.status_code}"
                self.guardrails.log_event("BLOCKED_OUTPUT", prompt, error_msg)
                return f"[Monarch] Error generating response: {response.status_code}"

            result_data = response.json()
            response_text = result_data.get("response", "")

        except requests.exceptions.ConnectionError:
            error_msg = f"Cannot connect to Ollama at {self.ollama_host}"
            self.guardrails.log_event("BLOCKED_OUTPUT", prompt, error_msg)
            return f"[Monarch] {error_msg}"
        except requests.exceptions.Timeout:
            error_msg = "Ollama request timeout"
            self.guardrails.log_event("BLOCKED_OUTPUT", prompt, error_msg)
            return f"[Monarch] {error_msg}"
        except Exception as e:
            error_msg = str(e)
            self.guardrails.log_event("BLOCKED_OUTPUT", prompt, error_msg)
            return f"[Monarch] Error: {error_msg}"

        # 3. Output check
        result = self.guardrails.check_output(response_text)
        if not result.allowed:
            self.guardrails.log_event("BLOCKED_OUTPUT", prompt, result.reason)
            return "[Monarch] I generated a response I can't share."

        self.guardrails.log_event("ALLOWED", prompt, None)
        return response_text

    def chat(self) -> None:
        """Interactive chat with Monarch via Ollama."""
        print("\nMonarch (via Ollama) - Interactive Mode")
        print("Type 'quit' to exit\n")

        while True:
            prompt = input("You: ").strip()

            if prompt.lower() == 'quit':
                print("Goodbye!")
                break

            if not prompt:
                continue

            response = self.generate(prompt)
            print(f"\nMonarch: {response}\n")


def main():
    """Main inference entry point for Ollama."""
    parser = argparse.ArgumentParser(description="Run Monarch inference via Ollama")
    parser.add_argument("--model", default="tinyllama",
                        help="Ollama model name (default: tinyllama)")
    parser.add_argument("--host", default="http://localhost:11434",
                        help="Ollama host URL (default: http://localhost:11434)")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml file")
    parser.add_argument("--prompt", default=None,
                        help="Prompt to generate from (interactive if not provided)")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Temperature for sampling")

    args = parser.parse_args()

    inference = MonarchOllamaInference(
        model_name=args.model,
        ollama_host=args.host,
        config_file=args.config,
    )

    if args.prompt:
        response = inference.generate(
            args.prompt,
            temperature=args.temperature,
        )
        print(f"\n{response}")
    else:
        inference.chat()


if __name__ == "__main__":
    main()
