#!/usr/bin/env python3
"""Fine-tune a LoRA adapter on Qwen3.5-4B for Forge code generation.

Uses unsloth for efficient 4-bit QLoRA training on a single RTX 3090.
"""

import json
from pathlib import Path

from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer, SFTConfig


def main():
    data_path = Path(__file__).parent / "forge_training_data.jsonl"
    output_dir = Path(__file__).parent / "forge-lora-qwen3.5-4b"

    # Load training data
    print("Loading training data...")
    examples = []
    with open(data_path) as f:
        for line in f:
            examples.append(json.loads(line))
    print(f"Loaded {len(examples)} examples")

    # Load model with 4-bit quantization
    print("Loading Qwen3.5-4B with 4-bit quantization...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="Qwen/Qwen3.5-4B",
        max_seq_length=4096,
        dtype=None,  # auto-detect
        load_in_4bit=True,
    )

    # Apply LoRA
    print("Applying LoRA adapter...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=32,  # LoRA rank
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # memory efficient
        random_state=42,
    )

    # Format data for chat template
    print("Formatting dataset...")

    def format_example(example):
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_example, remove_columns=["messages"])

    print(f"Dataset size: {len(dataset)}")
    print(f"Sample (first 300 chars): {dataset[0]['text'][:300]}...")

    # Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=5,
            num_train_epochs=10,  # small dataset, need multiple epochs
            learning_rate=2e-4,
            bf16=True,
            logging_steps=1,
            save_strategy="epoch",
            save_total_limit=2,
            seed=42,
            max_seq_length=4096,
            dataset_text_field="text",
            packing=True,  # pack short examples together
        ),
    )

    trainer_stats = trainer.train()
    print(f"\nTraining complete!")
    print(f"  Total steps: {trainer_stats.global_step}")
    print(f"  Training loss: {trainer_stats.training_loss:.4f}")

    # Save LoRA adapter
    print(f"\nSaving LoRA adapter to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Export to GGUF for llama.cpp
    gguf_dir = Path(__file__).parent / "forge-lora-qwen3.5-4b-gguf"
    print(f"\nExporting merged model to GGUF at {gguf_dir}...")
    try:
        model.save_pretrained_gguf(
            str(gguf_dir),
            tokenizer,
            quantization_method="q4_k_m",
        )
        print(f"GGUF export complete: {gguf_dir}")

        # Copy GGUF to models directory
        for gguf_file in gguf_dir.glob("*.gguf"):
            target = Path.home() / "models" / f"Qwen3.5-4B-Forge-LoRA-Q4_K_M.gguf"
            import shutil

            shutil.copy2(gguf_file, target)
            print(f"Copied to {target}")
            break
    except Exception as e:
        print(f"GGUF export failed (can do manually later): {e}")

    print("\nDone! To use the fine-tuned model:")
    print("  1. Update llama-server.service to point to the new GGUF")
    print(
        "  2. systemctl --user daemon-reload && systemctl --user restart llama-server"
    )
    print("  3. cargo run -- run --task 'your task'")


if __name__ == "__main__":
    main()
