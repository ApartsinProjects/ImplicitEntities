"""
O10: QLoRA fine-tune Llama 3.1 8B + evaluate on IRC-Bench v5 test set.
Runs on vast.ai GPU via GPU2Vast.
"""
import json
import sys
import time
import torch
from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODEL_OUTPUT = Path("qlora_llama8b_tmp")

BASE_MODEL = "meta-llama/Llama-3.1-8B-Instruct"

TRAIN_BATCH = 32
GRAD_ACCUM = 1
EPOCHS = 2
MAX_SEQ = 192
LORA_R = 16
LR = 2e-4
EVAL_BATCH = 64


def log(msg):
    print(msg, flush=True)


def build_prompt(implicit_text):
    return (
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n"
        "You identify implicitly referenced entities.<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n"
        "What entity is implicitly referenced? Answer with only the entity name.\n\n"
        f"Text: {implicit_text}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n"
    )


def build_dataset():
    from datasets import Dataset

    with open(DATA_DIR / "irc_bench_v5_train.json", encoding="utf-8") as f:
        train_data = json.load(f)
    with open(DATA_DIR / "irc_bench_v5_dev.json", encoding="utf-8") as f:
        dev_data = json.load(f)

    def format_sample(sample):
        return {
            "text": build_prompt(sample["implicit_text"]) + f"{sample['entity']}<|eot_id|>"
        }

    train_formatted = [format_sample(s) for s in train_data]
    dev_formatted = [format_sample(s) for s in dev_data[:500]]

    return Dataset.from_list(train_formatted), Dataset.from_list(dev_formatted)


def train():
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer, SFTConfig

    log(f"Loading {BASE_MODEL} (4-bit)...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
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
    sys.stdout.flush()

    train_ds, dev_ds = build_dataset()
    log(f"Train: {len(train_ds)}, Dev: {len(dev_ds)}")

    training_args = SFTConfig(
        output_dir=str(MODEL_OUTPUT),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=TRAIN_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        bf16=True,
        logging_steps=25,
        save_strategy="no",
        max_length=MAX_SEQ,
        dataset_text_field="text",
        report_to="none",
        warmup_steps=50,
    )

    trainer = SFTTrainer(
        model=model, args=training_args,
        train_dataset=train_ds, eval_dataset=dev_ds,
        processing_class=tokenizer,
    )

    t0 = time.time()
    trainer.train()
    elapsed = time.time() - t0
    log(f"Training time: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    trainer.save_model(str(MODEL_OUTPUT))
    tokenizer.save_pretrained(str(MODEL_OUTPUT))
    return model, tokenizer, elapsed


def evaluate(model=None, tokenizer=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    if model is None:
        log("\nLoading model for evaluation...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
        )
        base = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, quantization_config=bnb_config, device_map="auto",
        )
        model = PeftModel.from_pretrained(base, str(MODEL_OUTPUT))
        tokenizer = AutoTokenizer.from_pretrained(str(MODEL_OUTPUT))

    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    with open(DATA_DIR / "irc_bench_v5_test.json", encoding="utf-8") as f:
        test_data = json.load(f)

    entity_kb = json.load(open(DATA_DIR / "entity_kb.json", encoding="utf-8"))
    alias_map = {}
    for ename, edata in entity_kb.items():
        key = ename.lower().strip()
        aliases = {key}
        for a in edata.get("aliases", []):
            aliases.add(a.lower().strip())
        alias_map[key] = aliases

    n = len(test_data)
    log(f"Evaluating {n} test samples (batch_size={EVAL_BATCH})...")

    predictions = []
    correct = 0
    alias_correct = 0

    t0 = time.time()
    for batch_start in range(0, n, EVAL_BATCH):
        batch_end = min(batch_start + EVAL_BATCH, n)
        batch_samples = test_data[batch_start:batch_end]

        prompts = [build_prompt(s["implicit_text"]) for s in batch_samples]
        inputs = tokenizer(
            prompts, return_tensors="pt", truncation=True,
            max_length=MAX_SEQ, padding=True,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs, max_new_tokens=30, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        for i, sample in enumerate(batch_samples):
            prompt_len = inputs["attention_mask"][i].sum().item()
            pred_tokens = outputs[i][prompt_len:]
            pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

            gold_lower = sample["entity"].lower().strip()
            pred_lower = pred.lower().strip()

            is_exact = gold_lower == pred_lower
            is_alias = is_exact or (gold_lower in alias_map and pred_lower in alias_map[gold_lower])

            if is_exact:
                correct += 1
            if is_alias:
                alias_correct += 1

            predictions.append({
                "uid": sample["uid"],
                "gold_entity": sample["entity"],
                "gold_type": sample.get("entity_type", ""),
                "gold_qid": sample.get("entity_qid", ""),
                "prediction": pred,
                "implicit_text": sample["implicit_text"],
                "explicit_text": sample.get("explicit_text", ""),
            })

        done = batch_end
        if done % 500 < EVAL_BATCH or done == n:
            elapsed = time.time() - t0
            rate = done / elapsed
            eta = (n - done) / rate if rate > 0 else 0
            log(f"  {done}/{n} exact={correct}/{done} ({100*correct/done:.1f}%) "
                f"alias={alias_correct}/{done} ({100*alias_correct/done:.1f}%) "
                f"ETA: {eta/60:.1f}min")

    elapsed = time.time() - t0
    metrics = {
        "exp_id": "O10",
        "n_test": n,
        "model": BASE_MODEL,
        "adapter": "qlora_llama8b",
        "train_batch": TRAIN_BATCH,
        "grad_accum": GRAD_ACCUM,
        "effective_batch": TRAIN_BATCH * GRAD_ACCUM,
        "epochs": EPOCHS,
        "lora_r": LORA_R,
        "max_seq": MAX_SEQ,
        "exact_match": round(correct / n, 4),
        "alias_match": round(alias_correct / n, 4),
        "exact_correct": correct,
        "alias_correct": alias_correct,
        "eval_time_seconds": round(elapsed, 1),
    }

    with open(RESULTS_DIR / "O10_predictions.json", "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    with open(RESULTS_DIR / "O10_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log(f"\nExact match: {correct}/{n} ({100*correct/n:.1f}%)")
    log(f"Alias match: {alias_correct}/{n} ({100*alias_correct/n:.1f}%)")
    log(f"Eval time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log(f"Saved: results/O10_predictions.json, results/O10_metrics.json")


if __name__ == "__main__":
    model, tokenizer, train_time = train()
    evaluate(model, tokenizer)
