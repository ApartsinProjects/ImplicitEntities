"""
IRC-Bench v5: Open-World Experiments (O1-O10)
=============================================
Given implicit text, generate the entity name.
Uses OpenAI Batch API for GPT models, OpenRouter for Llama.

Usage:
  python run_open_world.py --exp O1          # Single experiment
  python run_open_world.py --exp O1,O2,O3,O4 # Batch API experiments
  python run_open_world.py --status BATCH_ID
  python run_open_world.py --retrieve BATCH_ID --exp O1
"""

import argparse
import asyncio
import csv
import json
import os
import re
import sys
import time
from pathlib import Path

from openai_batch_client import OpenAIBatchClient

V5_DIR = Path(__file__).parent
DATA_DIR = V5_DIR.parent / "data" / "benchmark"
RESULTS_DIR = V5_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
BATCH_DIR = Path(__file__).parent / "batches"
BATCH_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════
#  EXPERIMENT CONFIGS
# ═══════════════════════════════════════════════════════════════

ZS_SYSTEM = "You are an entity recognition expert. Given a text that implicitly references a named entity without mentioning it, identify what entity is being referenced."

ZS_PROMPT = """What named entity is implicitly referenced in this text? The entity is never mentioned by name.

Text: "{text}"

Think about the contextual cues (dates, places, events, people, roles) and identify the specific named entity being referenced.

Answer with ONLY the entity name (canonical Wikipedia name), nothing else."""

FS_EXAMPLES = [
    {"text": "I remember that Sunday morning in December '41. We were listening to the radio when the news broke about the attack on the naval base in Hawaii. That's when everything changed.", "entity": "Attack on Pearl Harbor"},
    {"text": "I enlisted right out of high school and went to boot camp in San Diego. As an aircraft mechanic, I was sent to the Pacific.", "entity": "United States Marine Corps"},
    {"text": "After the surrender, we flew into the main islands. I landed in the bay and spent six months there for the occupation. The capital was flattened by the B-29s.", "entity": "Tokyo"},
    {"text": "In late 1941, I was set to ship out from San Francisco. A friend ran up saying they're bombing the base in Hawaii.", "entity": "Attack on Pearl Harbor"},
    {"text": "Growing up in that bustling metropolis with towering skyscrapers, I was immersed in a vibrant culture.", "entity": "New York City"},
]

FS_PROMPT = """What named entity is implicitly referenced in this text? The entity is never mentioned by name.

Examples:
{examples}

Now identify the entity in this text:
Text: "{text}"

Answer with ONLY the entity name (canonical Wikipedia name), nothing else."""

EXPERIMENTS = {
    "O1": {"model": "gpt-4o", "mode": "zs", "api": "batch"},
    "O2": {"model": "gpt-4o", "mode": "fs", "api": "batch"},
    "O3": {"model": "gpt-4.1-mini", "mode": "zs", "api": "batch"},
    "O4": {"model": "gpt-4.1-mini", "mode": "fs", "api": "batch"},
    "O5": {"model": "meta-llama/llama-3.1-8b-instruct", "mode": "zs", "api": "openrouter"},
    "O6": {"model": "meta-llama/llama-3.1-8b-instruct", "mode": "fs", "api": "openrouter"},
    "O7": {"model": "meta-llama/llama-3.2-1b-instruct", "mode": "zs", "api": "openrouter"},
    "O8": {"model": "meta-llama/llama-3.2-1b-instruct", "mode": "fs", "api": "openrouter"},
}


def load_test_data():
    with open(DATA_DIR / "irc_bench_v5_test.json", encoding="utf-8") as f:
        return json.load(f)


def build_prompt(text, mode):
    if mode == "zs":
        return [
            {"role": "system", "content": ZS_SYSTEM},
            {"role": "user", "content": ZS_PROMPT.format(text=text)},
        ]
    else:
        examples_str = "\n".join(
            f'Text: "{ex["text"]}"\nEntity: {ex["entity"]}\n'
            for ex in FS_EXAMPLES
        )
        return [
            {"role": "system", "content": ZS_SYSTEM},
            {"role": "user", "content": FS_PROMPT.format(examples=examples_str, text=text)},
        ]


def submit_batch(exp_id):
    """Submit OpenAI Batch API job for an experiment."""
    config = EXPERIMENTS[exp_id]
    test_data = load_test_data()
    print(f"Experiment {exp_id}: {config['model']} {config['mode']}")
    print(f"Test samples: {len(test_data)}")

    ts = time.strftime("%Y%m%d_%H%M%S")
    jsonl_path = BATCH_DIR / f"v5_{exp_id}_{ts}.jsonl"
    file_map = {}

    with open(jsonl_path, "w", encoding="utf-8") as f:
        for i, sample in enumerate(test_data):
            custom_id = f"{exp_id}_{i:05d}"
            file_map[custom_id] = sample["uid"]
            messages = build_prompt(sample["implicit_text"], config["mode"])

            line = json.dumps({
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": config["model"],
                    "messages": messages,
                    "temperature": 0.0,
                    "max_tokens": 100,
                },
            })
            f.write(line + "\n")

    map_path = BATCH_DIR / f"v5_{exp_id}_map_{ts}.json"
    with open(map_path, "w", encoding="utf-8") as f:
        json.dump(file_map, f, indent=2)

    # Submit
    client = OpenAIBatchClient()
    with open(jsonl_path, "rb") as f:
        upload = client.client.files.create(file=f, purpose="batch")

    batch = client.client.batches.create(
        input_file_id=upload.id,
        endpoint="/v1/chat/completions",
        completion_window="24h",
        metadata={"job": f"v5_{exp_id}", "model": config["model"]},
    )

    print(f"Batch ID: {batch.id}")
    print(f"Status: {batch.status}")

    # Save batch info
    info_path = RESULTS_DIR / f"{exp_id}_batch_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump({
            "exp_id": exp_id,
            "batch_id": batch.id,
            "model": config["model"],
            "mode": config["mode"],
            "n_samples": len(test_data),
            "map_path": str(map_path),
            "timestamp": ts,
        }, f, indent=2)

    return batch.id


def retrieve_batch(batch_id, exp_id):
    """Retrieve batch results and save predictions."""
    client = OpenAIBatchClient()
    batch = client.client.batches.retrieve(batch_id)

    if batch.status != "completed":
        print(f"Not complete: {batch.status}")
        rc = batch.request_counts
        print(f"Progress: {rc.completed}/{rc.total} failed={rc.failed}")
        return

    # Download
    output = client.client.files.content(batch.output_file_id)
    raw_path = RESULTS_DIR / f"{exp_id}_raw_results.jsonl"
    raw_path.write_bytes(output.read())

    # Load file map
    info_path = RESULTS_DIR / f"{exp_id}_batch_info.json"
    with open(info_path, encoding="utf-8") as f:
        info = json.load(f)
    map_path = info["map_path"]
    with open(map_path, encoding="utf-8") as f:
        file_map = json.load(f)

    # Load test data for gold labels
    test_data = load_test_data()
    uid_to_gold = {s["uid"]: s for s in test_data}

    # Parse results
    predictions = []
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            result = json.loads(line)
            custom_id = result["custom_id"]
            uid = file_map.get(custom_id, "")
            gold = uid_to_gold.get(uid, {})

            try:
                pred = result["response"]["body"]["choices"][0]["message"]["content"].strip()
            except (KeyError, IndexError):
                pred = ""

            predictions.append({
                "uid": uid,
                "gold_entity": gold.get("entity", ""),
                "gold_type": gold.get("entity_type", ""),
                "gold_qid": gold.get("entity_qid", ""),
                "prediction": pred,
                "implicit_text": gold.get("implicit_text", ""),
                "explicit_text": gold.get("explicit_text", ""),
            })

    # Save predictions
    pred_path = RESULTS_DIR / f"{exp_id}_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    # Quick accuracy check
    exact = sum(1 for p in predictions if p["prediction"].lower().strip() == p["gold_entity"].lower().strip())
    print(f"Predictions: {len(predictions)}")
    print(f"Exact match: {exact}/{len(predictions)} ({100*exact/len(predictions):.1f}%)")
    print(f"Saved: {pred_path}")


async def run_openrouter(exp_id):
    """Run experiment via OpenRouter (for Llama models)."""
    from openrouter_client import batch_call

    config = EXPERIMENTS[exp_id]
    test_data = load_test_data()
    print(f"Experiment {exp_id}: {config['model']} {config['mode']}")
    print(f"Test samples: {len(test_data)}")

    prompts = [build_prompt(s["implicit_text"], config["mode"]) for s in test_data]

    print("Running via OpenRouter...")
    responses = await batch_call(
        prompts, model=config["model"], temperature=0.0, max_tokens=100,
        concurrency=10, progress_every=200,
    )

    predictions = []
    for sample, resp in zip(test_data, responses):
        predictions.append({
            "uid": sample["uid"],
            "gold_entity": sample["entity"],
            "gold_type": sample["entity_type"],
            "gold_qid": sample["entity_qid"],
            "prediction": (resp or "").strip(),
            "implicit_text": sample["implicit_text"],
            "explicit_text": sample["explicit_text"],
        })

    pred_path = RESULTS_DIR / f"{exp_id}_predictions.json"
    with open(pred_path, "w", encoding="utf-8") as f:
        json.dump(predictions, f, indent=2, ensure_ascii=False)

    exact = sum(1 for p in predictions if p["prediction"].lower().strip() == p["gold_entity"].lower().strip())
    print(f"Exact match: {exact}/{len(predictions)} ({100*exact/len(predictions):.1f}%)")
    print(f"Saved: {pred_path}")


def check_status(batch_id):
    client = OpenAIBatchClient()
    batch = client.client.batches.retrieve(batch_id)
    rc = batch.request_counts
    print(f"Status: {batch.status} | {rc.completed}/{rc.total} failed={rc.failed}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", type=str, help="Experiment IDs (comma-separated)")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--status", type=str)
    parser.add_argument("--retrieve", type=str)
    args = parser.parse_args()

    if args.status:
        check_status(args.status)
        return

    if args.retrieve and args.exp:
        retrieve_batch(args.retrieve, args.exp)
        return

    if not args.exp:
        parser.print_help()
        return

    exp_ids = [e.strip() for e in args.exp.split(",")]

    for exp_id in exp_ids:
        config = EXPERIMENTS[exp_id]

        if config["api"] == "batch":
            if args.submit:
                submit_batch(exp_id)
        elif config["api"] == "openrouter":
            asyncio.run(run_openrouter(exp_id))


if __name__ == "__main__":
    main()
