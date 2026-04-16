"""
Ablation experiments for implicit entity recognition.

Tests the effect of individual design choices by selectively removing or
modifying components of the LLM-based method:

  a) no_type        : Remove entity type conditioning from the inference prompt
  b) no_context     : Skip Step 1 (background generation), go directly to inference
  c) no_type_no_ctx : Both removed (simplest possible prompt)
  d) cot            : Add chain-of-thought reasoning to the inference prompt
  e) few_shot       : Include 3 hardcoded examples in the prompt
  f) json_output    : Request structured JSON output instead of free text

Usage:
    python run_ablations.py --dataset veterans_t2e --ablation no_type
    python run_ablations.py --dataset veterans_t2e --ablation all
    python run_ablations.py --dry-run
    python run_ablations.py --dataset veterans_t2e --ablation all --max-samples 20
"""

import asyncio
import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call
from run_experiments import (
    Sample,
    load_dataset,
    get_all_dataset_names,
    parse_ranked_guesses,
    evaluate_predictions,
    compute_metrics,
    print_metrics,
    save_results,
    RESULTS_DIR,
)

# ── Ablation definitions ──────────────────────────────────────────────────

ABLATIONS = {
    "baseline": "Full two-step method with entity type (for comparison)",
    "no_type": "Remove entity type conditioning from inference prompt",
    "no_context": "Skip Step 1 (background generation), go directly to inference",
    "no_type_no_ctx": "Both type and context removed (simplest possible prompt)",
    "cot": "Add chain-of-thought reasoning to inference prompt",
    "few_shot": "Include 3 hardcoded examples in the prompt",
    "json_output": "Request structured JSON output instead of free text",
}

DEFAULT_MODEL = "google/gemini-2.0-flash-001"

# ── Few-shot examples (curated from veterans and twitter datasets) ────────

FEW_SHOT_EXAMPLES = [
    {
        "text": (
            "The day is remembered vividly, December 7th, 1941. At just 15 years old, "
            "a curious teenager sat in an apartment in Manhattan, captivated by the sounds "
            "of the radio. The symphony played softly when suddenly, the broadcast was "
            "interrupted with shocking news that a naval base in the Pacific had been bombed."
        ),
        "entity_type": "places",
        "answer": "Pearl Harbor",
    },
    {
        "text": (
            "What Marvel and James Gunn managed to do in a movie introducing 5 new "
            "characters and a whole other side of the MCU is incredibly impressive"
        ),
        "entity_type": "Movie",
        "answer": "Guardians of the Galaxy",
    },
    {
        "text": (
            "Growing up in a bustling metropolis filled with towering skyscrapers, I was "
            "immersed in vibrant culture, yet my roots were European. My parents were "
            "immigrants; my father hailed from France, and my mother was from Switzerland. "
            "By the time I graduated high school in 1943, I was accepted into a prestigious "
            "university known for its rich history and distinguished faculty."
        ),
        "entity_type": "organizations",
        "answer": "Harvard",
    },
]


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def build_context_prompts(samples: list[Sample]) -> list[list[dict]]:
    """Step 1: Generate contextual background (same as baseline)."""
    prompts = []
    for s in samples:
        prompts.append([
            {"role": "system", "content": (
                "You are a knowledgeable historian and cultural analyst. "
                "Given a text passage, generate a brief historical and situational background "
                "that could help identify any implicitly referenced entities. "
                "Focus on time periods, locations, events, cultural markers, and roles mentioned. "
                "Be concise (2-4 sentences)."
            )},
            {"role": "user", "content": f'Text: "{s.text}"\n\nProvide relevant background context.'},
        ])
    return prompts


def build_baseline_inference(samples: list[Sample], contexts: list[str]) -> list[list[dict]]:
    """Baseline inference prompt (same as run_experiments.py)."""
    prompts = []
    for s, ctx in zip(samples, contexts):
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        ctx_text = ctx if ctx else "(no additional context available)"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage and background context, identify the specific "
                f"'{entity_type_hint}' that is being implicitly described. "
                "Provide your top 3 guesses ranked by confidence.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>\n\n"
                "Do NOT add explanations, just the entity names."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"Background context: {ctx_text}\n\n"
                f"What {entity_type_hint} is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def build_no_type_inference(samples: list[Sample], contexts: list[str]) -> list[list[dict]]:
    """Ablation (a): Remove entity type conditioning."""
    prompts = []
    for s, ctx in zip(samples, contexts):
        ctx_text = ctx if ctx else "(no additional context available)"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage and background context, identify the specific entity "
                "that is being implicitly described. "
                "Provide your top 3 guesses ranked by confidence.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>\n\n"
                "Do NOT add explanations, just the entity names."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"Background context: {ctx_text}\n\n"
                "What entity is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def build_no_context_inference(samples: list[Sample]) -> list[list[dict]]:
    """Ablation (b): Skip context, use entity type."""
    prompts = []
    for s in samples:
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage, identify the specific "
                f"'{entity_type_hint}' that is being implicitly described. "
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


def build_no_type_no_ctx_inference(samples: list[Sample]) -> list[list[dict]]:
    """Ablation (c): No type, no context (simplest possible prompt)."""
    prompts = []
    for s in samples:
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage, identify the specific entity that is being implicitly "
                "described. Provide your top 3 guesses ranked by confidence.\n\n"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>\n\n"
                "Do NOT add explanations, just the entity names."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                "What entity is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def build_cot_inference(samples: list[Sample], contexts: list[str]) -> list[list[dict]]:
    """Ablation (d): Add chain-of-thought reasoning."""
    prompts = []
    for s, ctx in zip(samples, contexts):
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        ctx_text = ctx if ctx else "(no additional context available)"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage and background context, identify the specific "
                f"'{entity_type_hint}' that is being implicitly described.\n\n"
                "Let's think step by step:\n"
                "1. First, identify the key clues and contextual markers in the text.\n"
                "2. Consider what time period, location, and domain these clues point to.\n"
                "3. Narrow down to the most likely entities.\n\n"
                "After your reasoning, provide your top 3 guesses.\n\n"
                "IMPORTANT: End your response with EXACTLY this format:\n"
                "ANSWER:\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>"
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"Background context: {ctx_text}\n\n"
                f"What {entity_type_hint} is implicitly described? "
                "Think step by step, then give your top 3 guesses."
            )},
        ])
    return prompts


def build_few_shot_inference(samples: list[Sample], contexts: list[str]) -> list[list[dict]]:
    """Ablation (e): Include 3 hardcoded few-shot examples."""
    examples_block = ""
    for i, ex in enumerate(FEW_SHOT_EXAMPLES, 1):
        examples_block += (
            f"Example {i}:\n"
            f'  Text: "{ex["text"]}"\n'
            f"  Entity type: {ex['entity_type']}\n"
            f"  Answer:\n"
            f"  1. {ex['answer']}\n\n"
        )

    prompts = []
    for s, ctx in zip(samples, contexts):
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        ctx_text = ctx if ctx else "(no additional context available)"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage and background context, identify the specific "
                f"'{entity_type_hint}' that is being implicitly described. "
                "Provide your top 3 guesses ranked by confidence.\n\n"
                "Here are some examples:\n\n"
                f"{examples_block}"
                "IMPORTANT: Respond in EXACTLY this format (one guess per line):\n"
                "1. <your first guess>\n"
                "2. <your second guess>\n"
                "3. <your third guess>\n\n"
                "Do NOT add explanations, just the entity names."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"Background context: {ctx_text}\n\n"
                f"What {entity_type_hint} is implicitly described? Give top 3 guesses."
            )},
        ])
    return prompts


def build_json_inference(samples: list[Sample], contexts: list[str]) -> list[list[dict]]:
    """Ablation (f): Request structured JSON output."""
    prompts = []
    for s, ctx in zip(samples, contexts):
        entity_type_hint = s.entity_type if s.entity_type else "entity"
        ctx_text = ctx if ctx else "(no additional context available)"
        prompts.append([
            {"role": "system", "content": (
                "You are an expert at identifying implicitly referenced entities in text. "
                "Given a passage and background context, identify the specific "
                f"'{entity_type_hint}' that is being implicitly described.\n\n"
                "IMPORTANT: Respond with ONLY valid JSON in this exact format:\n"
                '{"guesses": ["first guess", "second guess", "third guess"]}\n\n'
                "Do NOT add any text before or after the JSON."
            )},
            {"role": "user", "content": (
                f'Text: "{s.text}"\n\n'
                f"Background context: {ctx_text}\n\n"
                f"What {entity_type_hint} is implicitly described? "
                "Return your top 3 guesses as JSON."
            )},
        ])
    return prompts


def parse_json_guesses(response: str) -> list[str]:
    """Parse JSON-formatted guesses from the json_output ablation."""
    if not response:
        return []
    # Try to extract JSON from response
    try:
        # Find JSON object in response
        match = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if match:
            data = json.loads(match.group())
            guesses = data.get("guesses", [])
            return [g.strip() for g in guesses if isinstance(g, str) and g.strip()]
    except (json.JSONDecodeError, AttributeError):
        pass
    # Fallback to regular parsing
    return parse_ranked_guesses(response)


def parse_cot_guesses(response: str) -> list[str]:
    """Parse guesses from chain-of-thought response (after ANSWER: marker)."""
    if not response:
        return []
    # Try to find the ANSWER section
    parts = re.split(r'ANSWER\s*:', response, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parse_ranked_guesses(parts[-1])
    # Fallback: parse from end of response
    lines = response.strip().split("\n")
    # Take last lines that look like numbered items
    tail = "\n".join(lines[-5:])
    return parse_ranked_guesses(tail)


# ═══════════════════════════════════════════════════════════════════════════
#  ABLATION RUNNERS
# ═══════════════════════════════════════════════════════════════════════════

async def run_ablation(
    ablation_name: str,
    samples: list[Sample],
    model: str,
    concurrency: int = 10,
) -> list[list[str]]:
    """
    Run a specific ablation variant of the LLM method.
    Returns list of ranked predictions per sample.
    """
    needs_context = ablation_name in ("baseline", "no_type", "cot", "few_shot", "json_output")

    contexts = [""] * len(samples)
    if needs_context:
        print(f"\n  [{ablation_name}] Step 1: Generating contextual background...")
        ctx_prompts = build_context_prompts(samples)
        raw_contexts = await batch_call(
            ctx_prompts, model=model, temperature=0.3, max_tokens=200,
            concurrency=concurrency, progress_every=50,
        )
        contexts = [c or "" for c in raw_contexts]
    else:
        print(f"\n  [{ablation_name}] Skipping context generation (ablation).")

    # Build inference prompts based on ablation type
    print(f"\n  [{ablation_name}] Step 2: Inferring entities...")
    if ablation_name == "baseline":
        inf_prompts = build_baseline_inference(samples, contexts)
    elif ablation_name == "no_type":
        inf_prompts = build_no_type_inference(samples, contexts)
    elif ablation_name == "no_context":
        inf_prompts = build_no_context_inference(samples)
    elif ablation_name == "no_type_no_ctx":
        inf_prompts = build_no_type_no_ctx_inference(samples)
    elif ablation_name == "cot":
        inf_prompts = build_cot_inference(samples, contexts)
    elif ablation_name == "few_shot":
        inf_prompts = build_few_shot_inference(samples, contexts)
    elif ablation_name == "json_output":
        inf_prompts = build_json_inference(samples, contexts)
    else:
        raise ValueError(f"Unknown ablation: {ablation_name}")

    # Adjust max_tokens for CoT (needs more room for reasoning)
    max_tokens = 400 if ablation_name == "cot" else 150

    responses = await batch_call(
        inf_prompts, model=model, temperature=0.2, max_tokens=max_tokens,
        concurrency=concurrency, progress_every=50,
    )

    # Parse responses with ablation-specific parsers
    all_predictions = []
    for resp in responses:
        if ablation_name == "json_output":
            guesses = parse_json_guesses(resp or "")
        elif ablation_name == "cot":
            guesses = parse_cot_guesses(resp or "")
        else:
            guesses = parse_ranked_guesses(resp or "")
        all_predictions.append(guesses)

    return all_predictions


# ── Cost estimation ───────────────────────────────────────────────────────

def estimate_ablation_cost(n_samples: int, ablation_name: str) -> dict:
    """Estimate cost for an ablation run using the default model."""
    # Per-sample token estimates by ablation
    estimates = {
        "baseline":       {"input": 920, "output": 262},
        "no_type":        {"input": 880, "output": 262},
        "no_context":     {"input": 500, "output": 112},   # no step 1
        "no_type_no_ctx": {"input": 460, "output": 112},   # no step 1, no type
        "cot":            {"input": 980, "output": 412},   # longer output
        "few_shot":       {"input": 1400, "output": 262},  # examples in prompt
        "json_output":    {"input": 920, "output": 262},
    }
    est = estimates.get(ablation_name, {"input": 920, "output": 262})

    # Add evaluation overhead (~120 tokens input, ~12 output per sample for alias check)
    total_input = int(n_samples * (est["input"] + 120))
    total_output = int(n_samples * (est["output"] + 12))

    # Default model pricing (Gemini Flash)
    input_per_m = 0.10
    output_per_m = 0.40
    cost = (total_input / 1_000_000) * input_per_m + (total_output / 1_000_000) * output_per_m

    return {
        "n_samples": n_samples,
        "est_input_tokens": total_input,
        "est_output_tokens": total_output,
        "est_cost_usd": cost,
    }


def print_ablation_cost_table(dataset_names: list[str], ablation_names: list[str], max_samples: int):
    """Print cost estimate table."""
    print(f"\n{'=' * 70}")
    print("  COST ESTIMATE (dry run)")
    print(f"{'=' * 70}")

    total_cost = 0.0
    header = f"  {'Ablation':<20s} {'Samples':>8s} {'Input':>10s} {'Output':>10s} {'Cost':>10s}"
    print(header)
    print(f"  {'-' * 67}")

    for ds_name in dataset_names:
        samples, _ = load_dataset(ds_name)
        n = min(len(samples), max_samples) if max_samples > 0 else len(samples)
        print(f"  Dataset: {ds_name} ({n} samples)")

        for abl in ablation_names:
            est = estimate_ablation_cost(n, abl)
            total_cost += est["est_cost_usd"]
            print(f"    {abl:<18s} {n:>8d} {est['est_input_tokens']:>10,} "
                  f"{est['est_output_tokens']:>10,} ${est['est_cost_usd']:>8.4f}")

    print(f"  {'-' * 67}")
    print(f"  {'TOTAL':<20s} {'':>8s} {'':>10s} {'':>10s} ${total_cost:>8.4f}")
    print(f"{'=' * 70}")
    return total_cost


def print_ablation_comparison(all_results: dict):
    """Print comparison table across ablations."""
    print(f"\n\n{'=' * 80}")
    print("  ABLATION COMPARISON")
    print(f"{'=' * 80}")
    header = (f"  {'Dataset':<18s} {'Ablation':<18s} {'Hit@1':>7s} {'Hit@3':>7s} "
              f"{'MRR':>7s} {'Match%':>7s} {'Time':>7s}")
    print(header)
    print(f"  {'-' * 77}")

    for key, m in all_results.items():
        ds, abl = key
        if "error" in m:
            print(f"  {ds:<18s} {abl:<18s} ERROR: {m['error'][:25]}")
        else:
            h1 = f"{m.get('Hit@1', 0):.3f}"
            h3 = f"{m.get('Hit@3', 0):.3f}"
            mrr = f"{m.get('Global_MRR', 0):.3f}"
            n_s = m.get("n_samples", 1)
            n_m = m.get("n_matched", 0)
            match_pct = f"{100 * n_m / max(n_s, 1):.1f}%"
            t = f"{m.get('elapsed_seconds', 0):.0f}s"
            print(f"  {ds:<18s} {abl:<18s} {h1:>7s} {h3:>7s} {mrr:>7s} {match_pct:>7s} {t:>7s}")

    print(f"{'=' * 80}")

    # Delta table vs baseline
    baselines = {ds: m for (ds, abl), m in all_results.items() if abl == "baseline" and "error" not in m}
    if baselines:
        print(f"\n  DELTA vs BASELINE (Hit@1)")
        print(f"  {'-' * 50}")
        for (ds, abl), m in all_results.items():
            if abl == "baseline" or "error" in m:
                continue
            if ds in baselines:
                base_h1 = baselines[ds].get("Hit@1", 0)
                this_h1 = m.get("Hit@1", 0)
                delta = this_h1 - base_h1
                sign = "+" if delta >= 0 else ""
                print(f"  {ds:<18s} {abl:<18s} {sign}{delta:.3f} ({sign}{delta*100:.1f}pp)")
        print(f"  {'-' * 50}")


async def main():
    parser = argparse.ArgumentParser(
        description="Run ablation experiments for implicit entity recognition",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Ablation options:\n"
            "  baseline       Full two-step method (for comparison)\n"
            "  no_type        Remove entity type from inference prompt\n"
            "  no_context     Skip background context generation\n"
            "  no_type_no_ctx Both removed (simplest prompt)\n"
            "  cot            Add chain-of-thought reasoning\n"
            "  few_shot       Include 3 hardcoded examples\n"
            "  json_output    Request JSON output format\n"
            "  all            Run all ablations\n\n"
            "Examples:\n"
            "  python run_ablations.py --dataset veterans_t2e --ablation no_type\n"
            "  python run_ablations.py --dataset veterans_t2e --ablation all\n"
            "  python run_ablations.py --dry-run\n"
        ),
    )
    parser.add_argument(
        "--dataset", default="veterans_t2e",
        help="Dataset name or 'all' (default: veterans_t2e)",
    )
    parser.add_argument(
        "--ablation", default="all",
        help="Ablation name or 'all' (default: all)",
    )
    parser.add_argument(
        "--model", default=DEFAULT_MODEL,
        help=f"OpenRouter model ID (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--concurrency", type=int, default=10,
        help="Max concurrent API requests (default: 10)",
    )
    parser.add_argument(
        "--max-samples", type=int, default=0,
        help="Limit number of samples for testing, 0 for all (default: 0)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Estimate token count and cost without making API calls",
    )
    args = parser.parse_args()

    # Resolve datasets
    if args.dataset == "all":
        dataset_names = get_all_dataset_names()
    else:
        dataset_names = [args.dataset]

    # Resolve ablations
    if args.ablation == "all":
        ablation_names = list(ABLATIONS.keys())
    else:
        ablation_names = [a.strip() for a in args.ablation.split(",") if a.strip()]
        for a in ablation_names:
            if a not in ABLATIONS:
                print(f"  ERROR: Unknown ablation '{a}'. Options: {list(ABLATIONS.keys())}")
                return

    print(f"\n{'=' * 60}")
    print("  Ablation Experiments")
    print(f"  Datasets:   {dataset_names}")
    print(f"  Ablations:  {ablation_names}")
    print(f"  Model:      {args.model}")
    if args.max_samples:
        print(f"  Max samples: {args.max_samples}")
    for abl in ablation_names:
        print(f"    - {abl:<18s} {ABLATIONS[abl]}")
    print(f"{'=' * 60}")

    # Cost estimation
    total_est = print_ablation_cost_table(dataset_names, ablation_names, args.max_samples)

    if args.dry_run:
        print("\n  DRY RUN complete. No API calls made.")
        return

    if total_est > 0.50:
        print(f"\n  *** WARNING: Estimated total cost is ${total_est:.4f}. ***")
        print("  Consider using --max-samples to test with a subset first.")
        print("  Press Ctrl+C within 5 seconds to abort...")
        try:
            await asyncio.sleep(5)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n  Aborted by user.")
            return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    all_results = {}

    for ds_name in dataset_names:
        samples, unique_entities = load_dataset(ds_name)
        if not samples:
            print(f"  ERROR: No samples in dataset '{ds_name}', skipping.")
            continue

        if args.max_samples and args.max_samples < len(samples):
            print(f"  Limiting to {args.max_samples} samples (of {len(samples)})")
            samples = samples[:args.max_samples]

        for abl_name in ablation_names:
            print(f"\n{'#' * 60}")
            print(f"  ABLATION: {abl_name}")
            print(f"  DATASET: {ds_name} ({len(samples)} samples)")
            print(f"  Description: {ABLATIONS[abl_name]}")

            est = estimate_ablation_cost(len(samples), abl_name)
            print(f"  Estimated cost: ${est['est_cost_usd']:.4f}")
            print(f"{'#' * 60}")

            t0 = time.time()
            try:
                predictions = await run_ablation(
                    abl_name, samples, args.model, args.concurrency,
                )
                eval_results = await evaluate_predictions(
                    samples, predictions, model=args.model, concurrency=args.concurrency,
                )
                metrics = compute_metrics(eval_results)
                elapsed = time.time() - t0
                metrics["elapsed_seconds"] = round(elapsed, 1)
                metrics["ablation"] = abl_name

                print_metrics(metrics, f"{ds_name} / ablation={abl_name}")

                save_results(
                    eval_results, metrics, ds_name,
                    f"ablation_{abl_name}", args.model, timestamp,
                )
                all_results[(ds_name, abl_name)] = metrics

            except Exception as e:
                elapsed = time.time() - t0
                print(f"\n  ERROR running ablation {abl_name} on {ds_name}: {e}")
                import traceback
                traceback.print_exc()
                all_results[(ds_name, abl_name)] = {
                    "error": str(e),
                    "elapsed_seconds": round(elapsed, 1),
                }

    # Comparison table
    if all_results:
        print_ablation_comparison(all_results)

    # Save combined summary
    summary_path = RESULTS_DIR / f"{timestamp}_ablation_summary.json"
    serializable = {}
    for (ds, abl), m in all_results.items():
        serializable[f"{ds}/{abl}"] = m
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": timestamp,
            "experiment": "ablation_study",
            "model": args.model,
            "datasets": dataset_names,
            "ablations": ablation_names,
            "max_samples": args.max_samples,
            "results": serializable,
        }, f, indent=2, ensure_ascii=False)
    print(f"\n  Summary saved: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
