#!/usr/bin/env python3
"""
Merge a LoRA adapter into the base model for inference.

Usage:
    python -m finefacts.package._scripts.merge_adapter \
        --adapter_dir output/phase3/models/qwen3-8b/final \
        --base_model Qwen/Qwen3-8B \
        --output_dir output/phase3/models/qwen3-8b-merged
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter into base model")
    parser.add_argument("--adapter_dir", required=True)
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    try:
        from unsloth import FastLanguageModel
        use_unsloth = True
    except ImportError:
        use_unsloth = False

    print(f"Loading base model: {args.base_model}", file=sys.stderr)
    print(f"Loading adapter from: {args.adapter_dir}", file=sys.stderr)

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.adapter_dir,
            max_seq_length=10240,
            dtype=None,
            load_in_4bit=False,
        )
        print("Merging adapter...", file=sys.stderr)
        model.save_pretrained_merged(args.output_dir, tokenizer)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel

        base = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype="auto", trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            args.base_model, trust_remote_code=True
        )
        model = PeftModel.from_pretrained(base, args.adapter_dir)

        print("Merging adapter...", file=sys.stderr)
        model = model.merge_and_unload()
        model.save_pretrained(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)

    print(f"Merged model saved to {args.output_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
