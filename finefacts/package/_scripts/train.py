#!/usr/bin/env python3
"""
Fine-tune an open-source LLM for fact extraction using LoRA.

Uses Unsloth + TRL SFTTrainer for memory-efficient training.
Parameterized by model name so the same script works for all candidates.

Usage:
    python -m finefacts.package._scripts.train \
        --model_name Qwen/Qwen3-8B \
        --train_file output/phase3/training_data/train.jsonl \
        --dev_file output/phase3/training_data/dev.jsonl \
        --output_dir output/phase3/models/qwen3-8b

    # Smaller model with lower LoRA rank
    python -m finefacts.package._scripts.train \
        --model_name Qwen/Qwen3-1.7B \
        --lora_r 32 \
        --batch_size 8
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def get_model_short_name(model_name):
    """Extract short name for directory naming."""
    return model_name.split("/")[-1].lower()


def load_dataset_from_jsonl(path):
    """Load a JSONL file of conversations into a HuggingFace Dataset."""
    from datasets import Dataset

    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return Dataset.from_list(records)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for fact extraction")
    parser.add_argument("--model_name", required=True, help="HuggingFace model name")
    parser.add_argument("--train_file", required=True, help="Path to train.jsonl")
    parser.add_argument("--dev_file", required=True, help="Path to dev.jsonl")
    parser.add_argument("--output_dir", required=True, help="Output directory for model")
    parser.add_argument("--pretokenized_dir", default=None,
                        help="Path to pre-tokenized dataset (from pretokenize.py). "
                        "Skips tokenization if provided.")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=None,
                        help="LoRA alpha (default: 2x lora_r)")
    parser.add_argument("--max_seq_length", type=int, default=10240)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.lora_alpha is None:
        args.lora_alpha = args.lora_r * 2

    print(f"Model:         {args.model_name}", file=sys.stderr)
    print(f"LoRA rank:     {args.lora_r}", file=sys.stderr)
    print(f"LoRA alpha:    {args.lora_alpha}", file=sys.stderr)
    print(f"Batch size:    {args.batch_size} x {args.gradient_accumulation} = "
          f"{args.batch_size * args.gradient_accumulation}", file=sys.stderr)
    print(f"Max seq len:   {args.max_seq_length}", file=sys.stderr)
    print(f"Output:        {args.output_dir}", file=sys.stderr)

    # Try Unsloth first, fall back to standard transformers + PEFT
    # Skip Unsloth for GPT-OSS models (incompatible FusedActivation API)
    skip_unsloth = "gpt-oss" in args.model_name.lower()
    if skip_unsloth:
        use_unsloth = False
        print("Skipping Unsloth for GPT-OSS model (API incompatibility)", file=sys.stderr)
    else:
        try:
            from unsloth import FastLanguageModel
            use_unsloth = True
            print("Using Unsloth for accelerated training", file=sys.stderr)
        except ImportError:
            use_unsloth = False
            print("Unsloth not available, using standard transformers + PEFT", file=sys.stderr)

    # Load model
    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.model_name,
            max_seq_length=args.max_seq_length,
            dtype=None,  # auto-detect
            load_in_4bit=False,  # full bf16 on B200
        )
        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import LoraConfig, get_peft_model

        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, trust_remote_code=True
        )

        # GPT-OSS models use MXFP4 quantization which doesn't support training;
        # dequantize first
        load_kwargs = dict(torch_dtype="auto", trust_remote_code=True)
        if "gpt-oss" in args.model_name.lower():
            try:
                from transformers import Mxfp4Config
                load_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
                print("Dequantizing MXFP4 model for training", file=sys.stderr)
            except ImportError:
                print("Warning: Mxfp4Config not available, loading without dequantization",
                      file=sys.stderr)

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name, **load_kwargs
        )

        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
            ],
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.gradient_checkpointing_enable()

    # Ensure pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load datasets
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig

    use_pretokenized = args.pretokenized_dir is not None

    if use_pretokenized:
        from datasets import load_from_disk
        print(f"Loading pre-tokenized datasets from {args.pretokenized_dir}...", file=sys.stderr)
        train_dataset = load_from_disk(os.path.join(args.pretokenized_dir, "train"))
        dev_dataset = load_from_disk(os.path.join(args.pretokenized_dir, "dev"))
        print(f"  Train: {len(train_dataset):,} examples (pre-tokenized)", file=sys.stderr)
        print(f"  Dev:   {len(dev_dataset):,} examples (pre-tokenized)", file=sys.stderr)
        formatting_func = None
    else:
        print("Loading datasets...", file=sys.stderr)
        train_dataset = load_dataset_from_jsonl(args.train_file)
        dev_dataset = load_dataset_from_jsonl(args.dev_file)
        print(f"  Train: {len(train_dataset):,} examples", file=sys.stderr)
        print(f"  Dev:   {len(dev_dataset):,} examples", file=sys.stderr)

        # Format function for SFTTrainer
        # Unsloth requires this to return a list of strings
        def formatting_func(examples):
            """Convert conversations to chat-templated strings."""
            convos_list = examples["conversations"]
            # Handle both single example (list of dicts) and batched (list of lists)
            if convos_list and isinstance(convos_list[0], dict):
                text = tokenizer.apply_chat_template(
                    convos_list, tokenize=False, add_generation_prompt=False
                )
                return [text]
            texts = []
            for convos in convos_list:
                text = tokenizer.apply_chat_template(
                    convos, tokenize=False, add_generation_prompt=False
                )
                texts.append(text)
            return texts

    os.makedirs(args.output_dir, exist_ok=True)

    # Common training arguments
    common_args = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        bf16=True,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        eval_strategy="steps",
        eval_steps=args.save_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=args.seed,
        report_to="none",
    )

    if use_pretokenized:
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        training_args = TrainingArguments(**common_args)

        trainer = Trainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            data_collator=collator,
            args=training_args,
        )
    else:
        sft_config = SFTConfig(
            **common_args,
            max_seq_length=args.max_seq_length,
            packing=False,
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=dev_dataset,
            formatting_func=formatting_func,
            args=sft_config,
        )

    # Train
    print("Starting training...", file=sys.stderr)
    trainer.train()

    # Save final model
    final_dir = os.path.join(args.output_dir, "final")
    print(f"Saving final model to {final_dir}", file=sys.stderr)
    trainer.save_model(final_dir)
    tokenizer.save_pretrained(final_dir)

    # Save training summary
    summary = {
        "model_name": args.model_name,
        "short_name": get_model_short_name(args.model_name),
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "epochs": args.epochs,
        "batch_size": args.batch_size * args.gradient_accumulation,
        "learning_rate": args.lr,
        "max_seq_length": args.max_seq_length,
        "train_examples": len(train_dataset),
        "dev_examples": len(dev_dataset),
        "best_eval_loss": trainer.state.best_metric,
    }
    with open(os.path.join(args.output_dir, "training_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Best eval loss: {trainer.state.best_metric:.4f}", file=sys.stderr)
    print("Training complete.", file=sys.stderr)


if __name__ == "__main__":
    main()
