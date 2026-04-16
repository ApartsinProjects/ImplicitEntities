"""
Improved LLM prompt experiments for implicit entity recognition.

Implements prompt improvements from the improvement plan:
  B1 - Direct (no context): Skip background generation, go straight to inference.
  B2 - Structured output: Prompt asks for exactly 3 entity names, one per line.
  B3 - Few-shot: Include 3 worked examples in the prompt.
  B5 - Chain-of-thought: "Let's think step by step..."

Usage:
    python run_improved_llm.py --improvement structured --dataset veterans_t2e_v2
    python run_improved_llm.py --improvement fewshot --dataset veterans_t2e_v2
    python run_improved_llm.py --improvement cot --dataset veterans_t2e_v2
    python run_improved_llm.py --improvement direct --dataset veterans_t2e_v2
    python run_improved_llm.py --improvement all --dataset veterans_t2e_v2
    python run_improved_llm.py --improvement all --dry-run
"""

import asyncio
import argparse
import time
from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from openrouter_client import batch_call, estimate_cost
from run_experiments import (
    load_dataset,
    Sample,
    evaluate_predictions,
    compute_metrics,
    print_metrics,
    save_results,
    parse_ranked_guesses,
    RESULTS_DIR,
)

# ═══════════════════════════════════════════════════════════════════════════
#  FEW-SHOT EXAMPLES (B3)
# ═══════════════════════════════════════════════════════════════════════════

FEW_SHOT_EXAMPLES = [
    {
        "text": "a colossal figure holding a torch high, a beacon of hope",
        "entity_type": "Place",
        "answer": "Statue of Liberty",
    },
    {
        "text": "he spoke of a dream, his voice rising like a hymn",
        "entity_type": "Person",
        "answer": "Martin Luther King Jr.",
    },
    {
        "text": "surprise military strike on the major naval base in the Pacific",
        "entity_type": "Event",
        "answer": "Pearl Harbor",
    },
]


def format_few_shot_block() -> str:
    """Format the few-shot examples into a prompt block."""
    lines = ["Here are some examples:\n"]
    for ex in FEW_SHOT_EXAMPLES:
        lines.append(f'Text: "{ex["text"]}"')
        lines.append(f"Entity type: {ex['entity_type']}")
        lines.append(f"Answer:\n1. {ex['answer']}\n")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS PER IMPROVEMENT
# ═══════════════════════════════════════════════════════════════════════════

def build_structured_prompts(samples: list[Sample]) -> list[list[dict]]:
    """
    B2 - Structured output: strict format, exactly 3 entity names, no descriptions.
    """
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You identify implicitly referenced entities in text. "
                "You MUST respond with EXACTLY 3 entity names, one per line. "
                "No numbering, no descriptions, no explanations. "
                "Just the bare entity names, nothing else."
            )},
            {"role": "user", "content": (
                f"What {entity_type_hint} is implicitly described in this text?\n\n"
                f'"{s.text}"'
            )},
        ])
    return prompts


def build_fewshot_prompts(samples: list[Sample]) -> list[list[dict]]:
    """
    B3 - Few-shot: include 3 worked examples before the query.
    """
    few_shot_block = format_few_shot_block()
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage, identify the specific entity that is being implicitly described. "
                "Provide your top 3 guesses ranked by confidence.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <entity name>\n"
                "2. <entity name>\n"
                "3. <entity name>\n\n"
                "Do NOT add explanations, just the entity names.\n\n"
                f"{few_shot_block}"
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n'
                f"Entity type: {entity_type_hint}\n\n"
                "What entity is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def build_cot_prompts(samples: list[Sample]) -> list[list[dict]]:
    """
    B5 - Chain-of-thought: step-by-step reasoning before guessing.
    """
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Let's think step by step about what entity is described.\n\n"
                "First, identify the key clues in the text (time period, location, "
                "actions, cultural markers, notable characteristics).\n"
                "Then, reason about which specific entity matches those clues.\n"
                "Finally, provide your top 3 guesses.\n\n"
                "IMPORTANT: After your reasoning, you MUST end with EXACTLY this format:\n"
                "---\n"
                "1. <entity name>\n"
                "2. <entity name>\n"
                "3. <entity name>"
            )},
            {"role": "user", "content": (
                f"Let's think step by step about what {entity_type_hint} is "
                f"implicitly described in this text:\n\n"
                f'"{s.text}"'
            )},
        ])
    return prompts


def build_direct_prompts(samples: list[Sample]) -> list[list[dict]]:
    """
    B1 - Direct (no context): skip background generation, go straight to inference.
    Uses the same inference prompt as baseline but without the contextualization step.
    """
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                f"Given a passage, identify the specific '{entity_type_hint}' that is "
                "being implicitly described. "
                "Provide your top 3 guesses ranked by confidence.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>\n\n"
                "Do NOT add explanations, just the entity names."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"What {entity_type_hint} is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def parse_cot_guesses(response: str) -> list[str]:
    """
    Parse CoT response: look for the final structured guesses after '---' separator.
    Falls back to standard parse_ranked_guesses if no separator found.
    """
    if not response:
        return []

    # Try to find the structured section after ---
    if "---" in response:
        parts = response.rsplit("---", 1)
        if len(parts) == 2:
            structured_part = parts[1].strip()
            guesses = parse_ranked_guesses(structured_part)
            if guesses:
                return guesses

    # Fallback: parse the whole response
    return parse_ranked_guesses(response)


# ═══════════════════════════════════════════════════════════════════════════
#  IMPROVEMENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════

IMPROVEMENTS = {
    "structured": {
        "label": "B2: Structured Output",
        "build_prompts": build_structured_prompts,
        "parse_fn": parse_ranked_guesses,
        "max_tokens": 100,
        "temperature": 0.2,
    },
    "fewshot": {
        "label": "B3: Few-Shot Examples",
        "build_prompts": build_fewshot_prompts,
        "parse_fn": parse_ranked_guesses,
        "max_tokens": 150,
        "temperature": 0.2,
    },
    "cot": {
        "label": "B5: Chain-of-Thought",
        "build_prompts": build_cot_prompts,
        "parse_fn": parse_cot_guesses,
        "max_tokens": 500,
        "temperature": 0.3,
    },
    "direct": {
        "label": "B1: Direct (No Context)",
        "build_prompts": build_direct_prompts,
        "parse_fn": parse_ranked_guesses,
        "max_tokens": 150,
        "temperature": 0.2,
    },
}


# ═══════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ═══════════════════════════════════════════════════════════════════════════

async def smoke_test(
    samples: list[Sample],
    model: str,
    improvement_key: str,
    concurrency: int = 5,
    n_samples: int = 3,
) -> bool:
    """
    Run a quick smoke test with n_samples to verify API connectivity,
    prompt format, and response parsing BEFORE launching the full batch.
    """
    cfg = IMPROVEMENTS[improvement_key]
    test_samples = samples[:n_samples]
    print(f"\n  [SMOKE TEST] {cfg['label']} with {n_samples} samples on {model}...")

    prompts = cfg["build_prompts"](test_samples)
    responses = await batch_call(
        prompts, model=model,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        concurrency=concurrency,
        progress_every=999,
    )

    null_count = sum(1 for r in responses if r is None)
    if null_count == n_samples:
        print(f"  [SMOKE TEST] FAIL: All {n_samples} calls returned None.")
        return False

    parsed_ok = 0
    for i, resp in enumerate(responses):
        guesses = cfg["parse_fn"](resp or "")
        gold = test_samples[i].entity
        status = "OK" if guesses else "EMPTY"
        print(f"    [{status}] Gold: \"{gold}\"")
        print(f"          Response: \"{(resp or '')[:150]}\"")
        print(f"          Parsed: {guesses[:3]}")
        if guesses:
            parsed_ok += 1

    if parsed_ok == 0:
        print(f"  [SMOKE TEST] FAIL: Parsed 0/{n_samples} responses.")
        return False

    print(f"  [SMOKE TEST] PASS: {parsed_ok}/{n_samples} parsed successfully.\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════
#  SINGLE IMPROVEMENT RUN
# ═══════════════════════════════════════════════════════════════════════════

async def run_improvement(
    samples: list[Sample],
    model: str,
    improvement_key: str,
    concurrency: int = 10,
    dataset_name: str = "",
    timestamp: str = "",
) -> dict:
    """Run a single improvement variant. Returns metrics dict."""
    cfg = IMPROVEMENTS[improvement_key]
    label = cfg["label"]

    print(f"\n{'#' * 60}")
    print(f"  IMPROVEMENT: {label}")
    print(f"  Dataset: {dataset_name} ({len(samples)} samples)")
    print(f"  Model: {model}")
    print(f"{'#' * 60}")

    t0 = time.time()

    # Build prompts
    prompts = cfg["build_prompts"](samples)

    # Run inference
    print(f"\n  [{label}] Running inference...")
    responses = await batch_call(
        prompts, model=model,
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        concurrency=concurrency,
        progress_every=50,
    )

    # Parse predictions
    all_predictions = []
    for resp in responses:
        guesses = cfg["parse_fn"](resp or "")
        all_predictions.append(guesses)

    # Evaluate
    eval_results = await evaluate_predictions(
        samples, all_predictions, model=model, concurrency=concurrency,
    )

    metrics = compute_metrics(eval_results)
    print_metrics(metrics, f"{dataset_name} / {label} / {model}")

    elapsed = time.time() - t0
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["improvement"] = improvement_key
    print(f"\n  Elapsed: {elapsed:.1f}s")

    # Save results
    method_name = f"llm_{improvement_key}"
    save_results(eval_results, metrics, dataset_name, method_name, model, timestamp)

    return metrics


# ═══════════════════════════════════════════════════════════════════════════
#  COMPARISON TABLE
# ═══════════════════════════════════════════════════════════════════════════

def print_comparison_table(all_metrics: dict[str, dict]):
    """Print a comparison table of all improvement runs."""
    print(f"\n\n{'=' * 75}")
    print(f"  IMPROVEMENT COMPARISON")
    print(f"{'=' * 75}")
    header = (
        f"  {'Improvement':<30s} {'Hit@1':>7s} {'Hit@3':>7s} "
        f"{'MRR':>7s} {'Time':>7s} {'Matched':>8s}"
    )
    print(header)
    print(f"  {'-' * 70}")

    for key, m in all_metrics.items():
        if "error" in m:
            print(f"  {key:<30s} ERROR: {m['error'][:30]}")
        else:
            label = IMPROVEMENTS.get(key, {}).get("label", key)
            h1 = f"{m.get('Hit@1', 0):.3f}"
            h3 = f"{m.get('Hit@3', 0):.3f}"
            mrr = f"{m.get('Global_MRR', 0):.3f}"
            t = f"{m.get('elapsed_seconds', 0):.0f}s"
            matched = f"{m.get('n_matched', 0)}/{m.get('n_samples', 0)}"
            print(f"  {label:<30s} {h1:>7s} {h3:>7s} {mrr:>7s} {t:>7s} {matched:>8s}")

    print(f"{'=' * 75}")


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

async def main():
    parser = argparse.ArgumentParser(
        description="Run improved LLM prompt experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Improvements:\n"
            "  structured  B2: Strict output format (3 entity names, no descriptions)\n"
            "  fewshot     B3: Few-shot examples in the prompt\n"
            "  cot         B5: Chain-of-thought reasoning\n"
            "  direct      B1: Direct inference (no context step)\n"
            "  all         Run all improvements and compare\n\n"
            "Examples:\n"
            "  python run_improved_llm.py --improvement structured --dataset veterans_t2e_v2\n"
            "  python run_improved_llm.py --improvement all --dataset veterans_t2e_v2 --max-samples 50\n"
            "  python run_improved_llm.py --improvement all --dry-run\n"
        ),
    )
    parser.add_argument(
        "--improvement", default="all",
        choices=["structured", "fewshot", "cot", "direct", "all"],
        help="Which improvement to run (default: all)",
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e_v2",
        help="Dataset name (default: veterans_t2e_v2)",
    )
    parser.add_argument(
        "--model", default="google/gemini-2.0-flash-001:floor",
        help="OpenRouter model ID (default: google/gemini-2.0-flash-001:floor)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples for testing, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show cost estimate without running",
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Determine which improvements to run
    if args.improvement == "all":
        improvement_keys = list(IMPROVEMENTS.keys())
    else:
        improvement_keys = [args.improvement]

    # Load dataset
    print(f"\n  Loading dataset: {args.dataset}")
    samples, unique_entities = load_dataset(args.dataset)
    if not samples:
        print("  ERROR: No samples loaded.")
        sys.exit(1)

    if args.max_samples and args.max_samples < len(samples):
        print(f"  Limiting to {args.max_samples} samples (of {len(samples)})")
        samples = samples[:args.max_samples]

    # Dry run: show cost estimates
    if args.dry_run:
        print(f"\n  DRY RUN: Cost estimates for {len(samples)} samples")
        print(f"  {'=' * 50}")
        for key in improvement_keys:
            cfg = IMPROVEMENTS[key]
            avg_prompt_tokens = 200
            if key == "fewshot":
                avg_prompt_tokens = 350  # few-shot examples add tokens
            elif key == "cot":
                avg_prompt_tokens = 250
            est = estimate_cost(
                n_prompts=len(samples),
                avg_prompt_tokens=avg_prompt_tokens,
                avg_completion_tokens=cfg["max_tokens"] // 2,
                model=args.model,
            )
            print(f"  {cfg['label']:<30s}: ${est['estimated_cost_usd']:.4f} "
                  f"({est['n_prompts']} prompts)")
        print(f"  {'=' * 50}")
        return

    print(f"\n{'=' * 60}")
    print(f"  Improved LLM Experiments")
    print(f"  Dataset: {args.dataset} ({len(samples)} samples)")
    print(f"  Model: {args.model}")
    print(f"  Improvements: {improvement_keys}")
    print(f"  Time: {timestamp}")
    print(f"{'=' * 60}")

    all_metrics: dict[str, dict] = {}

    for key in improvement_keys:
        try:
            # Smoke test first
            smoke_ok = await smoke_test(
                samples, args.model, key,
                concurrency=min(args.concurrency, 5),
            )
            if not smoke_ok:
                print(f"  SMOKE TEST FAILED for {key}. Skipping.")
                all_metrics[key] = {"error": "smoke_test_failed"}
                continue

            metrics = await run_improvement(
                samples=samples,
                model=args.model,
                improvement_key=key,
                concurrency=args.concurrency,
                dataset_name=args.dataset,
                timestamp=timestamp,
            )
            all_metrics[key] = metrics

        except Exception as e:
            print(f"\n  ERROR in {key}: {e}")
            import traceback
            traceback.print_exc()
            all_metrics[key] = {"error": str(e)}

    # Print comparison table if multiple improvements ran
    if len(all_metrics) > 1:
        print_comparison_table(all_metrics)

    # Save combined summary
    import json
    summary_path = RESULTS_DIR / f"{timestamp}_improved_llm_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "model": args.model,
            "dataset": args.dataset,
            "n_samples": len(samples),
            "improvements": improvement_keys,
            "results": all_metrics,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Combined summary: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
