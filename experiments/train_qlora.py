"""
IRC-Bench v5: QLoRA Fine-tuning (O9)
=====================================
Fine-tune Llama 3.2 1B for open-world IER.

Usage:
  python train_qlora.py --train
  python train_qlora.py --eval
"""

import argparse
import json
import os
import time
import torch
from pathlib import Path

V5_DIR = Path(__file__).parent
DATA_DIR = V5_DIR.parent / "data" / "benchmark"
MODELS_DIR = V5_DIR / "models"
RESULTS_DIR = V5_DIR / "results"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
OUTPUT_DIR = MODELS_DIR / "qlora_llama1b"

# Optimized config
BATCH_SIZE = 8
GRAD_ACCUM = 4
EPOCHS = 2
MAX_SEQ = 128
LORA_R = 16
LR = 2e-4


def build_dataset():
    """Build train/eval datasets for QLoRA."""
    from datasets import Dataset

    with open(DATA_DIR / "irc_bench_v5_train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "irc_bench_v5_dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)

    def format_sample(sample):
        return {
            "text": f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou identify implicitly referenced entities.<|eot_id|><|start_header_id|>user<|end_header_id|>\nWhat entity is implicitly referenced? Answer with only the entity name.\n\nText: {sample['implicit_text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n{sample['entity']}<|eot_id|>"
        }

    train_formatted = [format_sample(s) for s in train_data]
    dev_formatted = [format_sample(s) for s in dev_data[:500]]  # Subsample dev

    return Dataset.from_list(train_formatted), Dataset.from_list(dev_formatted)


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    print(f"Loading {BASE_MODEL} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float32,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R, lora_alpha=32, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    train_ds, dev_ds = build_dataset()
    print(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=False,  # RTX 2060 has issues with fp16 + 4bit
        logging_steps=50,
        save_strategy="epoch",
        max_length=MAX_SEQ,
        dataset_text_field="text",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))


def evaluate():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print("Loading model for evaluation...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
    )
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
    )
    model = PeftModel.from_pretrained(base, str(OUTPUT_DIR))
    tokenizer = AutoTokenizer.from_pretrained(str(OUTPUT_DIR))
    model.eval()

    with open(DATA_DIR / "irc_bench_v5_test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    predictions = []
    correct = 0

    for i, sample in enumerate(test_data):
        prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou identify implicitly referenced entities.<|eot_id|><|start_header_id|>user<|end_header_id|>\nWhat entity is implicitly referenced? Answer with only the entity name.\n\nText: {sample['implicit_text']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n"

        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        pred = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

        if pred.lower() == sample["entity"].lower():
            correct += 1

        predictions.append({
            "uid": sample["uid"],
            "gold_entity": sample["entity"],
            "gold_type": sample.get("entity_type", ""),
            "prediction": pred,
            "implicit_text": sample["implicit_text"],
        })

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(test_data)} exact={correct}/{i+1} ({100*correct/(i+1):.1f}%)")

    pred_path = RESULTS_DIR / "O9_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    print(f"\nExact match: {correct}/{len(test_data)} ({100*correct/len(test_data):.1f}%)")
    print(f"Saved: {pred_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        train()
        evaluate()


if __name__ == "__main__":
    main()
