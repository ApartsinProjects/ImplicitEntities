"""
LoRA fine-tuning script for Llama/Mistral/Phi models on the IRC implicit entity
recognition task.

Fine-tunes a local HuggingFace model with LoRA (or QLoRA with 4-bit quantization)
to generate entity names given implicit reference texts. Supports training,
evaluation, dry-run diagnostics, and adapter merging.

Supported base models:
  - meta-llama/Llama-3.2-1B-Instruct   (1B,  ~3 GB VRAM)
  - meta-llama/Llama-3.2-3B-Instruct   (3B,  ~8 GB VRAM)
  - microsoft/Phi-3.5-mini-instruct     (3.8B, ~10 GB VRAM)
  - mistralai/Mistral-7B-Instruct-v0.3  (7B,  ~16 GB VRAM)

Usage:
    python train_lora_llm.py --base-model meta-llama/Llama-3.2-1B-Instruct --epochs 3
    python train_lora_llm.py --base-model microsoft/Phi-3.5-mini-instruct --quantize 4bit
    python train_lora_llm.py --mode eval --model-path experiments/trained_models/lora_xxx
    python train_lora_llm.py --dry-run
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path


# ============================================================================
#  Constants and paths
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "benchmark_v2"
BENCH_V3 = DATA_DIR / "IRC_Bench_v3.csv"
# Train/test loaded from partition column
DESCRIPTIONS_JSON = Path(__file__).parent / "entity_descriptions" / "veterans_v2_descriptions.json"
RESULTS_DIR = Path(__file__).parent / "results"
MODELS_DIR = Path(__file__).parent / "trained_models"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model family detection patterns
MODEL_FAMILIES = {
    "llama": ["llama", "Llama"],
    "mistral": ["mistral", "Mistral"],
    "phi": ["phi", "Phi"],
}

# Approximate VRAM requirements in GB (fp16 / 4-bit)
VRAM_ESTIMATES = {
    "meta-llama/Llama-3.2-1B-Instruct": (3.0, 1.5),
    "meta-llama/Llama-3.2-3B-Instruct": (8.0, 4.0),
    "microsoft/Phi-3.5-mini-instruct": (10.0, 5.0),
    "mistralai/Mistral-7B-Instruct-v0.3": (16.0, 8.0),
}

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "datasets",
]

OPTIONAL_PACKAGES = [
    ("bitsandbytes", "Required for 4-bit quantization (QLoRA)"),
]


# ============================================================================
#  Data loading
# ============================================================================

@dataclass
class Sample:
    uid: str
    text: str
    entity: str
    entity_type: str
    domain: str
    variant: str
    eval_mode: str = ""
    partition: str = ""


def load_csv_samples(csv_path: Path) -> list[Sample]:
    """Load samples from benchmark CSV."""
    samples = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            samples.append(Sample(
                uid=row["uid"],
                text=row["text"],
                entity=row["entity"],
                entity_type=row.get("entity_type", ""),
                domain=row.get("domain", ""),
                variant=row.get("variant", ""),
                eval_mode=row.get("eval_mode", ""),
                partition=row.get("partition", ""),
            ))
    return samples


def load_entity_descriptions(path: Path) -> dict:
    """Load entity description lookup."""
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ============================================================================
#  Chat template formatting per model family
# ============================================================================

def detect_model_family(model_name: str) -> str:
    """Detect model family from model name string."""
    name_lower = model_name.lower()
    if "llama" in name_lower:
        return "llama"
    if "mistral" in name_lower:
        return "mistral"
    if "phi" in name_lower:
        return "phi"
    return "generic"


def format_prompt_for_training(text: str, entity_type: str, entity: str,
                                family: str) -> str:
    """
    Format a single sample as a complete training conversation string.

    For SFTTrainer, we produce the full conversation text. The trainer uses
    the tokenizer's chat_template to handle special tokens, so we provide
    a structured message list that gets formatted via apply_chat_template.
    """
    # We return a message list; the actual formatting happens in the dataset
    # builder using the tokenizer's chat_template.
    # This function is used as a fallback for raw-text formatting.
    system_msg = "You are an expert at identifying implicitly referenced entities."
    user_msg = (
        f'Text: "{text}"\n'
        f"Type: {entity_type}\n"
        f"What entity is implicitly described?"
    )
    assistant_msg = entity

    if family == "llama":
        return (
            f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
            f"{system_msg}<|eot_id|>"
            f"<|start_header_id|>user<|end_header_id|>\n\n"
            f"{user_msg}<|eot_id|>"
            f"<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{assistant_msg}<|eot_id|>"
        )
    elif family == "mistral":
        return (
            f"<s>[INST] {system_msg}\n\n{user_msg} [/INST]{assistant_msg}</s>"
        )
    elif family == "phi":
        return (
            f"<|system|>\n{system_msg}<|end|>\n"
            f"<|user|>\n{user_msg}<|end|>\n"
            f"<|assistant|>\n{assistant_msg}<|end|>"
        )
    else:
        # Generic ChatML
        return (
            f"<|im_start|>system\n{system_msg}<|im_end|>\n"
            f"<|im_start|>user\n{user_msg}<|im_end|>\n"
            f"<|im_start|>assistant\n{assistant_msg}<|im_end|>"
        )


def build_messages(text: str, entity_type: str, entity: str = None) -> list[dict]:
    """Build a structured message list for chat template formatting."""
    messages = [
        {"role": "system", "content": "You are an expert at identifying implicitly referenced entities."},
        {"role": "user", "content": (
            f'Text: "{text}"\n'
            f"Type: {entity_type}\n"
            f"What entity is implicitly described?"
        )},
    ]
    if entity is not None:
        messages.append({"role": "assistant", "content": entity})
    return messages


# ============================================================================
#  Dataset builder for HuggingFace datasets
# ============================================================================

def build_hf_dataset(samples: list[Sample], tokenizer, max_seq_length: int,
                     max_samples: int = None):
    """
    Build a HuggingFace Dataset from samples using the tokenizer's chat template.
    Returns a datasets.Dataset with a 'text' column of formatted conversations.
    """
    from datasets import Dataset

    if max_samples and max_samples < len(samples):
        import random
        random.seed(42)
        samples = random.sample(samples, max_samples)

    texts = []
    family = detect_model_family(tokenizer.name_or_path)

    for sample in samples:
        messages = build_messages(sample.text, sample.entity_type, sample.entity)
        try:
            formatted = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
        except Exception:
            # Fallback to manual formatting if chat template fails
            formatted = format_prompt_for_training(
                sample.text, sample.entity_type, sample.entity, family
            )
        texts.append(formatted)

    return Dataset.from_dict({"text": texts})


# ============================================================================
#  Dependency checking
# ============================================================================

def check_dependencies(quantize: str = None) -> dict:
    """Check that required packages are installed. Returns status dict."""
    status = {}

    for pkg in REQUIRED_PACKAGES:
        try:
            __import__(pkg)
            status[pkg] = True
        except ImportError:
            status[pkg] = False

    for pkg, reason in OPTIONAL_PACKAGES:
        try:
            __import__(pkg)
            status[pkg] = True
        except ImportError:
            status[pkg] = False

    # Check GPU
    try:
        import torch
        status["cuda_available"] = torch.cuda.is_available()
        if status["cuda_available"]:
            status["gpu_name"] = torch.cuda.get_device_name(0)
            status["gpu_vram_gb"] = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        else:
            status["gpu_name"] = None
            status["gpu_vram_gb"] = 0.0
    except ImportError:
        status["cuda_available"] = False
        status["gpu_name"] = None
        status["gpu_vram_gb"] = 0.0

    return status


def print_dependency_report(status: dict, quantize: str = None):
    """Print formatted dependency report."""
    print("\n  Dependency Check")
    print("  " + "=" * 50)
    missing = []
    for pkg in REQUIRED_PACKAGES:
        ok = status.get(pkg, False)
        mark = "OK" if ok else "MISSING"
        print(f"    {pkg:20s} : {mark}")
        if not ok:
            missing.append(pkg)

    for pkg, reason in OPTIONAL_PACKAGES:
        ok = status.get(pkg, False)
        mark = "OK" if ok else "MISSING"
        suffix = "" if ok else f"  ({reason})"
        print(f"    {pkg:20s} : {mark}{suffix}")
        if not ok and quantize == "4bit":
            missing.append(pkg)

    print()
    print(f"    CUDA available     : {'Yes' if status['cuda_available'] else 'No'}")
    if status["cuda_available"]:
        print(f"    GPU                : {status['gpu_name']}")
        print(f"    VRAM               : {status['gpu_vram_gb']:.1f} GB")
    print("  " + "=" * 50)

    if missing:
        print(f"\n  Install missing packages:")
        print(f"    pip install {' '.join(missing)}")
    if not status["cuda_available"]:
        print("\n  WARNING: No GPU detected.")
        print("  The 1B model can train on CPU (slowly). Larger models need a GPU.")
        print("  Consider using a cloud GPU (Colab, RunPod, Lambda Labs, etc.).")

    return missing


# ============================================================================
#  Model loading
# ============================================================================

def load_model_and_tokenizer(model_name: str, quantize: str = None,
                              adapter_path: str = None):
    """
    Load base model with optional quantization and optional LoRA adapter.
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    print(f"\n  Loading model: {model_name}")
    if quantize:
        print(f"  Quantization: {quantize}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Quantization config
    bnb_config = None
    if quantize == "4bit":
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        except Exception as e:
            print(f"  WARNING: Could not create 4-bit config: {e}")
            print("  Falling back to fp16.")
            bnb_config = None

    # Determine dtype
    if bnb_config is not None:
        dtype = None  # BnB handles dtype
    elif torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        dtype = torch.bfloat16
    elif torch.cuda.is_available():
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        attn_implementation="eager",  # Safe fallback; flash_attention_2 may not be available
    )

    # Load adapter if specified
    if adapter_path:
        print(f"  Loading LoRA adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)

    param_count = sum(p.numel() for p in model.parameters())
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: {param_count / 1e6:.1f}M total, {trainable_count / 1e6:.1f}M trainable")

    return model, tokenizer


def apply_lora_config(model, lora_r: int = 16, lora_alpha: int = 32,
                      lora_dropout: float = 0.05):
    """Apply LoRA adapter to the model and return the PeftModel."""
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

    # Prepare for quantized training if needed
    if getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False):
        model = prepare_model_for_kbit_training(model)

    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, lora_config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"\n  LoRA applied: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Target modules: {target_modules}")
    print(f"  Trainable parameters: {trainable / 1e6:.2f}M / {total / 1e6:.1f}M "
          f"({100 * trainable / total:.2f}%)")

    return model


# ============================================================================
#  Training
# ============================================================================

def train(args):
    """Run LoRA fine-tuning."""
    import torch
    from transformers import TrainingArguments
    from trl import SFTTrainer, SFTConfig

    print("\n" + "=" * 60)
    print("  LoRA Fine-Tuning for IRC Implicit Entity Recognition")
    print("=" * 60)

    # Load data
    print(f"\n  Loading training data from: {BENCH_V3}")
    train_samples = [s for s in load_csv_samples(BENCH_V3) if s.partition == "train"]
    print(f"  Training samples: {len(train_samples)}")

    if args.max_samples:
        print(f"  Limiting to {args.max_samples} samples")

    # Domain/type breakdown
    domains = {}
    types = {}
    for s in train_samples:
        domains[s.domain] = domains.get(s.domain, 0) + 1
        types[s.entity_type] = types.get(s.entity_type, 0) + 1
    print(f"  Domains: {dict(sorted(domains.items()))}")
    print(f"  Entity types: {dict(sorted(types.items()))}")

    # Load model
    model, tokenizer = load_model_and_tokenizer(args.base_model, args.quantize)

    # Apply LoRA
    model = apply_lora_config(
        model,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Build dataset
    print("\n  Building training dataset...")
    train_dataset = build_hf_dataset(
        train_samples, tokenizer, args.max_seq_length, args.max_samples
    )
    print(f"  Dataset size: {len(train_dataset)} samples")

    # Output directory
    model_short = args.base_model.replace("/", "_").replace("-", "_")
    output_dir = MODELS_DIR / f"lora_{model_short}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Compute steps
    effective_batch = args.batch_size * args.gradient_accumulation
    steps_per_epoch = math.ceil(len(train_dataset) / effective_batch)
    total_steps = steps_per_epoch * args.epochs

    print(f"\n  Training config:")
    print(f"    Output dir          : {output_dir}")
    print(f"    Epochs              : {args.epochs}")
    print(f"    Batch size          : {args.batch_size}")
    print(f"    Gradient accum.     : {args.gradient_accumulation}")
    print(f"    Effective batch     : {effective_batch}")
    print(f"    Steps/epoch         : {steps_per_epoch}")
    print(f"    Total steps         : {total_steps}")
    print(f"    Learning rate       : {args.lr}")
    print(f"    Max seq length      : {args.max_seq_length}")

    # Determine training precision
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    fp16 = torch.cuda.is_available() and not bf16

    # Training arguments
    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        bf16=bf16,
        fp16=fp16,
        logging_steps=max(1, steps_per_epoch // 10),
        save_strategy="epoch",
        save_total_limit=2,
        max_length=args.max_seq_length,
        dataset_text_field="text",
        packing=False,
        report_to="none",
        seed=42,
        optim="adamw_torch",
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    # Train
    print("\n  Starting training...")
    t0 = time.time()
    train_result = trainer.train()
    elapsed = time.time() - t0

    print(f"\n  Training complete in {elapsed / 60:.1f} minutes")
    print(f"  Final loss: {train_result.training_loss:.4f}")

    # Save adapter
    print(f"\n  Saving LoRA adapter to: {output_dir}")
    trainer.model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    # Save training metadata
    metadata = {
        "base_model": args.base_model,
        "quantize": args.quantize,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_seq_length": args.max_seq_length,
        "gradient_accumulation": args.gradient_accumulation,
        "train_samples": len(train_dataset),
        "training_loss": train_result.training_loss,
        "training_time_sec": elapsed,
    }
    with open(output_dir / "training_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"  Metadata saved to: {output_dir / 'training_metadata.json'}")
    print(f"\n  To evaluate, run:")
    print(f"    python train_lora_llm.py --mode eval --model-path {output_dir}")

    return str(output_dir)


# ============================================================================
#  Evaluation
# ============================================================================

def normalize_entity(name: str) -> str:
    """Normalize entity name for matching."""
    if not name:
        return ""
    name = name.lower().strip()
    # Remove articles
    for article in ["the ", "a ", "an "]:
        if name.startswith(article):
            name = name[len(article):]
    # Remove punctuation
    name = re.sub(r"[^\w\s]", "", name)
    # Collapse whitespace
    name = re.sub(r"\s+", " ", name).strip()
    return name


def jaccard_similarity(a: str, b: str) -> float:
    """Compute Jaccard token overlap between two strings."""
    tokens_a = set(normalize_entity(a).split())
    tokens_b = set(normalize_entity(b).split())
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def match_entity(prediction: str, ground_truth: str,
                 jaccard_threshold: float = 0.5) -> tuple[bool, str]:
    """
    Match prediction against ground truth using tiered matching:
      1. Exact match (normalized)
      2. Jaccard token overlap >= threshold
    Returns (matched, tier).
    """
    pred_norm = normalize_entity(prediction)
    gt_norm = normalize_entity(ground_truth)

    if not pred_norm:
        return False, "none"

    # Exact
    if pred_norm == gt_norm:
        return True, "exact"

    # Containment (one contains the other)
    if pred_norm in gt_norm or gt_norm in pred_norm:
        return True, "alias"

    # Jaccard
    if jaccard_similarity(prediction, ground_truth) >= jaccard_threshold:
        return True, "jaccard"

    return False, "none"


def parse_generation(text: str) -> list[str]:
    """Parse generated text to extract entity predictions."""
    if not text:
        return []

    # Clean up common artifacts
    text = text.strip()

    # If the response contains multiple lines, parse each as a guess
    guesses = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove numbering
        cleaned = re.sub(r"^(\d+[\.\)]\s*|[-\*]\s*)", "", line).strip()
        # Remove surrounding quotes
        cleaned = cleaned.strip("\"'")
        # Remove trailing punctuation
        cleaned = cleaned.rstrip(".,;:!")
        if cleaned and len(cleaned) > 1:
            guesses.append(cleaned)

    # Fallback: treat entire text as one guess
    if not guesses and text.strip() and len(text.strip()) > 1:
        guesses = [text.strip()[:200]]

    return guesses


@dataclass
class EvalResult:
    uid: str
    text_snippet: str
    ground_truth: str
    entity_type: str
    predictions: list[str]
    match_rank: int  # 0 = no match, 1 = first guess correct, etc.
    match_tier: str  # exact, alias, jaccard, none


def generate_predictions(model, tokenizer, samples: list[Sample],
                         max_new_tokens: int = 64, batch_size: int = 4,
                         temperature: float = 0.1) -> list[EvalResult]:
    """Generate predictions for test samples and evaluate."""
    import torch

    model.eval()
    results = []
    family = detect_model_family(tokenizer.name_or_path)

    print(f"\n  Generating predictions for {len(samples)} samples...")
    t0 = time.time()

    for i in range(0, len(samples), batch_size):
        batch = samples[i:i + batch_size]

        for sample in batch:
            messages = build_messages(sample.text, sample.entity_type)

            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            except Exception:
                # Fallback manual formatting (without the answer)
                prompt = format_prompt_for_training(
                    sample.text, sample.entity_type, "", family
                )

            inputs = tokenizer(
                prompt, return_tensors="pt", truncation=True,
                max_length=512, padding=False
            )

            # Move to model device
            device = next(model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=max(temperature, 0.01),
                    do_sample=temperature > 0,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            # Decode only the new tokens
            input_len = inputs["input_ids"].shape[1]
            generated_ids = outputs[0][input_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

            # Parse predictions
            predictions = parse_generation(generated_text)

            # Evaluate against ground truth
            match_rank = 0
            match_tier = "none"
            for rank, pred in enumerate(predictions, 1):
                matched, tier = match_entity(pred, sample.entity)
                if matched:
                    match_rank = rank
                    match_tier = tier
                    break

            results.append(EvalResult(
                uid=sample.uid,
                text_snippet=sample.text[:120],
                ground_truth=sample.entity,
                entity_type=sample.entity_type,
                predictions=predictions[:10],
                match_rank=match_rank,
                match_tier=match_tier,
            ))

        # Progress
        done = min(i + batch_size, len(samples))
        if done % 50 == 0 or done == len(samples):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            print(f"    {done}/{len(samples)} ({rate:.1f} samples/sec)")

    return results


def compute_metrics(results: list[EvalResult], max_k: int = 10) -> dict:
    """Compute Hit@K, Global MRR, and Filtered MRR from evaluation results."""
    n = len(results)
    if n == 0:
        return {}

    hits_at = {}
    for k in [1, 3, 5, 10]:
        if k > max_k:
            continue
        count = sum(1 for r in results if 0 < r.match_rank <= k)
        hits_at[f"Hit@{k}"] = count / n

    # Global MRR
    reciprocal_ranks = []
    for r in results:
        if r.match_rank > 0:
            reciprocal_ranks.append(1.0 / r.match_rank)
        else:
            reciprocal_ranks.append(0.0)
    global_mrr = sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0

    # Filtered MRR (only matched samples)
    filtered_rr = [rr for rr in reciprocal_ranks if rr > 0]
    filtered_mrr = sum(filtered_rr) / len(filtered_rr) if filtered_rr else 0.0

    # Match tier distribution
    tier_counts = {"exact": 0, "alias": 0, "jaccard": 0, "none": 0}
    for r in results:
        tier_counts[r.match_tier] = tier_counts.get(r.match_tier, 0) + 1

    # Per-type breakdown
    type_metrics = {}
    types_seen = set(r.entity_type for r in results)
    for etype in sorted(types_seen):
        type_results = [r for r in results if r.entity_type == etype]
        type_n = len(type_results)
        type_hit1 = sum(1 for r in type_results if 0 < r.match_rank <= 1) / type_n
        type_metrics[f"Hit@1_{etype}"] = type_hit1

    metrics = {
        "n_samples": n,
        "n_matched": sum(1 for r in results if r.match_rank > 0),
        **hits_at,
        "Global_MRR": global_mrr,
        "Filtered_MRR": filtered_mrr,
        **{f"tier_{k}": v for k, v in tier_counts.items()},
        **type_metrics,
    }
    return metrics


def print_metrics(metrics: dict, label: str = ""):
    """Pretty-print evaluation metrics."""
    print(f"\n  {'=' * 55}")
    if label:
        print(f"  RESULTS: {label}")
    print(f"  {'=' * 55}")
    n = metrics.get("n_samples", 0)
    matched = metrics.get("n_matched", 0)
    print(f"  Samples: {n}  |  Matched: {matched} ({100 * matched / max(n, 1):.1f}%)")
    print()

    for k in ["Hit@1", "Hit@3", "Hit@5", "Hit@10"]:
        if k in metrics:
            print(f"  {k:12s}: {metrics[k]:.4f}  ({metrics[k] * 100:.1f}%)")
    print()
    print(f"  {'Global MRR':12s}: {metrics.get('Global_MRR', 0):.4f}")
    print(f"  {'Filtered MRR':12s}: {metrics.get('Filtered_MRR', 0):.4f}")
    print()
    print(f"  Match tiers: exact={metrics.get('tier_exact', 0)}, "
          f"alias={metrics.get('tier_alias', 0)}, "
          f"jaccard={metrics.get('tier_jaccard', 0)}, "
          f"none={metrics.get('tier_none', 0)}")

    # Per-type Hit@1 breakdown
    type_keys = [k for k in metrics if k.startswith("Hit@1_")]
    if type_keys:
        print(f"\n  Per-type Hit@1:")
        for k in sorted(type_keys):
            etype = k.replace("Hit@1_", "")
            print(f"    {etype:20s}: {metrics[k]:.4f}  ({metrics[k] * 100:.1f}%)")

    print(f"  {'=' * 55}")


def save_eval_results(results: list[EvalResult], metrics: dict,
                      output_dir: Path, label: str):
    """Save evaluation results and metrics to files."""
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    # Save detailed results
    results_file = output_dir / f"eval_{label}_{timestamp}.json"
    results_data = []
    for r in results:
        results_data.append({
            "uid": r.uid,
            "text_snippet": r.text_snippet,
            "ground_truth": r.ground_truth,
            "entity_type": r.entity_type,
            "predictions": r.predictions,
            "match_rank": r.match_rank,
            "match_tier": r.match_tier,
        })

    with open(results_file, "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "timestamp": timestamp,
            "metrics": metrics,
            "results": results_data,
        }, f, indent=2, ensure_ascii=False)

    print(f"  Results saved to: {results_file}")
    return results_file


def evaluate(args):
    """Run evaluation on test set with a fine-tuned model."""
    print("\n" + "=" * 60)
    print("  LoRA Evaluation for IRC Implicit Entity Recognition")
    print("=" * 60)

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"  ERROR: Model path not found: {model_path}")
        sys.exit(1)

    # Load metadata to get base model name
    meta_file = model_path / "training_metadata.json"
    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
        base_model = metadata.get("base_model", args.base_model)
        quantize = metadata.get("quantize", args.quantize)
        print(f"  Base model (from metadata): {base_model}")
    else:
        base_model = args.base_model
        quantize = args.quantize
        if not base_model:
            print("  ERROR: No training_metadata.json found and --base-model not specified.")
            sys.exit(1)
        print(f"  Base model (from args): {base_model}")

    # Load test data
    print(f"\n  Loading test data from: {BENCH_V3}")
    test_samples = [s for s in load_csv_samples(BENCH_V3) if s.partition == "test"]
    print(f"  Test samples: {len(test_samples)}")

    if args.max_samples:
        test_samples = test_samples[:args.max_samples]
        print(f"  Limited to {len(test_samples)} samples")

    # Load model with adapter
    model, tokenizer = load_model_and_tokenizer(
        base_model, quantize, adapter_path=str(model_path)
    )

    # Generate and evaluate
    results = generate_predictions(
        model, tokenizer, test_samples,
        max_new_tokens=64,
        batch_size=1,  # Single sample generation for reliability
        temperature=0.1,
    )

    # Compute and print metrics
    metrics = compute_metrics(results)
    label = f"lora_{model_path.name}"
    print_metrics(metrics, label)

    # Per-domain breakdown
    domains = set(s.domain for s in test_samples)
    for domain in sorted(domains):
        domain_uids = {s.uid for s in test_samples if s.domain == domain}
        domain_results = [r for r in results if r.uid in domain_uids]
        if domain_results:
            domain_metrics = compute_metrics(domain_results)
            print(f"\n  Domain: {domain} ({len(domain_results)} samples)")
            for k in ["Hit@1", "Hit@3", "Hit@5"]:
                if k in domain_metrics:
                    print(f"    {k}: {domain_metrics[k]:.4f}")

    # Save results
    save_eval_results(results, metrics, RESULTS_DIR, label)

    # Print some example predictions
    print(f"\n  Sample predictions (first 10):")
    print(f"  {'-' * 80}")
    for r in results[:10]:
        status = "HIT" if r.match_rank > 0 else "MISS"
        pred_str = r.predictions[0] if r.predictions else "<empty>"
        print(f"  [{status}] GT: {r.ground_truth}")
        print(f"         Pred: {pred_str}")
        print(f"         Text: {r.text_snippet[:80]}...")
        print()


# ============================================================================
#  Adapter merging
# ============================================================================

def merge_adapter(args):
    """Merge LoRA adapter weights into the base model for faster inference."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch

    model_path = Path(args.model_path)
    meta_file = model_path / "training_metadata.json"

    if meta_file.exists():
        with open(meta_file) as f:
            metadata = json.load(f)
        base_model = metadata.get("base_model", args.base_model)
    else:
        base_model = args.base_model

    if not base_model:
        print("  ERROR: Cannot determine base model. Provide --base-model or ensure metadata exists.")
        sys.exit(1)

    print(f"\n  Merging adapter into base model: {base_model}")
    print(f"  Adapter path: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
    )

    model = PeftModel.from_pretrained(model, str(model_path))
    model = model.merge_and_unload()

    output_dir = model_path.parent / f"merged_{model_path.name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"  Saving merged model to: {output_dir}")
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    print("  Done.")


# ============================================================================
#  Dry-run diagnostics
# ============================================================================

def dry_run(args):
    """Print dataset stats, check dependencies, estimate VRAM, print training plan."""
    print("\n" + "=" * 60)
    print("  DRY RUN: LoRA Fine-Tuning Diagnostics")
    print("=" * 60)

    # Check dependencies
    status = check_dependencies(args.quantize)
    missing = print_dependency_report(status, args.quantize)

    # Dataset stats
    print(f"\n  Dataset Stats")
    print("  " + "=" * 50)

    if BENCH_V3.exists():
        train_samples = [s for s in load_csv_samples(BENCH_V3) if s.partition == "train"]
        print(f"    Training file  : {BENCH_V3}")
        print(f"    Training samples: {len(train_samples)}")

        domains = {}
        types = {}
        entities = set()
        text_lengths = []
        for s in train_samples:
            domains[s.domain] = domains.get(s.domain, 0) + 1
            types[s.entity_type] = types.get(s.entity_type, 0) + 1
            entities.add(s.entity.lower())
            text_lengths.append(len(s.text))

        print(f"    Unique entities : {len(entities)}")
        print(f"    Domains         : {dict(sorted(domains.items()))}")
        print(f"    Entity types    : {dict(sorted(types.items()))}")
        avg_len = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        print(f"    Avg text length : {avg_len:.0f} chars "
              f"(min={min(text_lengths, default=0)}, max={max(text_lengths, default=0)})")
    else:
        print(f"    Training file NOT FOUND: {BENCH_V3}")

    if BENCH_V3.exists():
        test_samples = [s for s in load_csv_samples(BENCH_V3) if s.partition == "test"]
        print(f"\n    Test file       : {BENCH_V3}")
        print(f"    Test samples    : {len(test_samples)}")
        test_entities = set(s.entity.lower() for s in test_samples)
        if BENCH_V3.exists():
            train_entities = set(s.entity.lower() for s in train_samples)
            unseen = test_entities - train_entities
            print(f"    Unseen entities : {len(unseen)} / {len(test_entities)}")
    else:
        print(f"    Test file NOT FOUND: {BENCH_V3}")

    # VRAM estimation
    print(f"\n  VRAM Estimation")
    print("  " + "=" * 50)

    model_name = args.base_model
    fp16_vram, q4_vram = VRAM_ESTIMATES.get(model_name, (0, 0))
    use_q4 = args.quantize == "4bit"
    estimated_vram = q4_vram if use_q4 else fp16_vram

    # LoRA overhead is small (~10-50MB depending on r)
    lora_overhead = args.lora_r * 0.005  # rough estimate in GB
    total_estimated = estimated_vram + lora_overhead

    print(f"    Model           : {model_name}")
    print(f"    Quantization    : {args.quantize or 'none (fp16/bf16)'}")
    if fp16_vram > 0:
        print(f"    Base VRAM (fp16): ~{fp16_vram:.1f} GB")
        print(f"    Base VRAM (4bit): ~{q4_vram:.1f} GB")
        print(f"    LoRA overhead   : ~{lora_overhead:.2f} GB")
        print(f"    Estimated total : ~{total_estimated:.1f} GB")
    else:
        print(f"    (No VRAM estimate available for this model)")

    available_vram = status.get("gpu_vram_gb", 0)
    if available_vram > 0:
        fits = "YES" if available_vram >= total_estimated else "NO"
        print(f"    Available VRAM  : {available_vram:.1f} GB")
        print(f"    Fits in VRAM?   : {fits}")
        if available_vram < total_estimated and not use_q4 and q4_vram > 0:
            if available_vram >= q4_vram + lora_overhead:
                print(f"    TIP: Use --quantize 4bit to reduce VRAM to ~{q4_vram + lora_overhead:.1f} GB")
    elif not status.get("cuda_available", False):
        print(f"    No GPU detected. Training will be very slow on CPU.")
        if model_name in VRAM_ESTIMATES:
            if VRAM_ESTIMATES[model_name][0] <= 3.0:
                print(f"    The 1B model can technically run on CPU for small datasets.")
            else:
                print(f"    This model size requires a GPU. Consider cloud GPU services.")

    # Training plan
    effective_batch = args.batch_size * args.gradient_accumulation
    n_train = args.max_samples if args.max_samples else (len(train_samples) if BENCH_V3.exists() else 0)
    steps_per_epoch = math.ceil(n_train / effective_batch) if n_train > 0 else 0
    total_steps = steps_per_epoch * args.epochs

    print(f"\n  Training Plan")
    print("  " + "=" * 50)
    print(f"    LoRA r          : {args.lora_r}")
    print(f"    LoRA alpha      : {args.lora_alpha}")
    print(f"    LoRA dropout    : {args.lora_dropout}")
    print(f"    Epochs          : {args.epochs}")
    print(f"    Batch size      : {args.batch_size}")
    print(f"    Gradient accum. : {args.gradient_accumulation}")
    print(f"    Effective batch : {effective_batch}")
    print(f"    Learning rate   : {args.lr}")
    print(f"    Max seq length  : {args.max_seq_length}")
    print(f"    Training samples: {n_train}")
    print(f"    Steps per epoch : {steps_per_epoch}")
    print(f"    Total steps     : {total_steps}")
    print(f"    Grad checkpoint : {args.gradient_checkpointing}")

    output_name = model_name.replace("/", "_").replace("-", "_")
    print(f"\n    Output dir      : {MODELS_DIR / f'lora_{output_name}'}")

    if missing:
        print(f"\n  BLOCKED: Missing dependencies. Install them first.")
    else:
        print(f"\n  Ready to train. Remove --dry-run to start.")

    print("=" * 60)


# ============================================================================
#  CLI
# ============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tuning for IRC implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train_lora_llm.py --dry-run
  python train_lora_llm.py --base-model meta-llama/Llama-3.2-1B-Instruct --epochs 3
  python train_lora_llm.py --base-model microsoft/Phi-3.5-mini-instruct --quantize 4bit
  python train_lora_llm.py --mode eval --model-path experiments/trained_models/lora_xxx
  python train_lora_llm.py --mode merge --model-path experiments/trained_models/lora_xxx
        """,
    )

    parser.add_argument(
        "--mode", type=str, default="train",
        choices=["train", "eval", "merge"],
        help="Mode: train, eval (run evaluation on test set), merge (merge adapter into base)"
    )
    parser.add_argument(
        "--base-model", type=str,
        default="meta-llama/Llama-3.2-1B-Instruct",
        help="HuggingFace model name (default: Llama-3.2-1B-Instruct)"
    )
    parser.add_argument(
        "--model-path", type=str, default=None,
        help="Path to trained LoRA adapter (for eval/merge modes)"
    )
    parser.add_argument(
        "--quantize", type=str, default=None, choices=["4bit"],
        help="Quantization mode (4bit for QLoRA)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--lora-r", type=int, default=16, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument("--max-samples", type=int, default=None, help="Limit training samples (for quick tests)")
    parser.add_argument("--max-seq-length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument(
        "--gradient-accumulation", type=int, default=4,
        help="Gradient accumulation steps"
    )
    parser.add_argument(
        "--gradient-checkpointing", action="store_true",
        help="Enable gradient checkpointing to reduce VRAM at cost of speed"
    )
    parser.add_argument("--dry-run", action="store_true", help="Print diagnostics without training")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.dry_run:
        dry_run(args)
        return

    if args.mode == "train":
        train(args)
    elif args.mode == "eval":
        if not args.model_path:
            print("ERROR: --model-path required for eval mode")
            sys.exit(1)
        evaluate(args)
    elif args.mode == "merge":
        if not args.model_path:
            print("ERROR: --model-path required for merge mode")
            sys.exit(1)
        merge_adapter(args)


if __name__ == "__main__":
    main()
