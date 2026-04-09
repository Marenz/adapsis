#!/usr/bin/env python3
"""Fine-tune Gemma 4 on Adapsis training data using QLoRA.

Uses unsloth for fast LoRA fine-tuning on a single RTX 3090.
Loads the base Gemma 4 model, applies QLoRA adapters, trains on
curated conversation examples, then exports to GGUF for llama.cpp.

Usage:
    python3 tools/finetune_gemma4.py --base google/gemma-4-e4b-it --epochs 3
    python3 tools/finetune_gemma4.py --base google/gemma-4-31b-a4b-it --epochs 3
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser(description="Fine-tune Gemma 4 on Adapsis data")
    parser.add_argument(
        "--base", default="google/gemma-4-e4b-it", help="Base model from HuggingFace"
    )
    parser.add_argument(
        "--training-data",
        default=os.path.expanduser("~/.config/adapsis/training/curated-examples.jsonl"),
        help="JSONL training data file",
    )
    parser.add_argument(
        "--output",
        default="adapsis-gemma4-lora",
        help="Output directory for LoRA adapter",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--export-gguf", action="store_true", help="Export to GGUF after training"
    )
    parser.add_argument("--gguf-quant", default="q4_k_m", help="GGUF quantization type")
    args = parser.parse_args()

    # Load training data
    print(f"Loading training data from {args.training_data}")
    conversations = []
    with open(args.training_data) as f:
        for line in f:
            d = json.loads(line)
            conversations.append(d["messages"])
    print(f"Loaded {len(conversations)} training examples")

    if len(conversations) < 10:
        print("WARNING: Very few training examples. Consider generating more first.")

    # Import ML libraries
    print("Loading model and tokenizer...")
    try:
        from unsloth import FastLanguageModel
    except ImportError:
        print("unsloth not installed. Install with: uv pip install --system unsloth")
        print("Falling back to standard transformers + peft...")
        from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
        from peft import LoraConfig, get_peft_model
        from trl import SFTTrainer

        use_unsloth = False
    else:
        use_unsloth = True

    if use_unsloth:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=args.base,
            max_seq_length=args.max_seq_length,
            dtype=None,  # auto-detect
            load_in_4bit=True,
        )

        model = FastLanguageModel.get_peft_model(
            model,
            r=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_alpha=args.lora_rank,
            lora_dropout=0,
            bias="none",
            use_gradient_checkpointing="unsloth",
        )
    else:
        from transformers import BitsAndBytesConfig
        import torch

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForCausalLM.from_pretrained(
            args.base,
            quantization_config=bnb_config,
            device_map="auto",
        )
        tokenizer = AutoTokenizer.from_pretrained(args.base)

        lora_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_rank,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()

    # Format training data for the trainer
    from datasets import Dataset

    def format_conversation(messages):
        """Convert messages list to the model's chat template."""
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    formatted = [{"text": format_conversation(conv)} for conv in conversations]
    dataset = Dataset.from_list(formatted)
    print(f"Dataset: {len(dataset)} examples")

    # Training
    from trl import SFTTrainer
    from transformers import TrainingArguments

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=10,
        logging_steps=5,
        save_steps=50,
        save_total_limit=2,
        fp16=True,
        optim="adamw_8bit" if use_unsloth else "adamw_torch",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
    )

    print("Starting training...")
    trainer.train()

    # Save
    print(f"Saving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Export to GGUF if requested
    if args.export_gguf:
        print(f"Exporting to GGUF ({args.gguf_quant})...")
        if use_unsloth:
            model.save_pretrained_gguf(
                f"{args.output}-gguf",
                tokenizer,
                quantization_method=args.gguf_quant,
            )
            print(f"GGUF saved to {args.output}-gguf/")
        else:
            print("GGUF export requires unsloth. Merge and convert manually.")

    print("Done!")


if __name__ == "__main__":
    main()
