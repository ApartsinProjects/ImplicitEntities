"""
Comprehensive gold label audit across ALL benchmark variants.
Uses cross-model consensus + LLM validation.

For each variant:
  1. Load all available prediction files
  2. Find samples where BEST model fails (match_tier == 'none')
  3. Cross-model consensus: if 3+ models agree on different entity, flag gold
  4. LLM audit: ask LLM to validate gold correctness + suggest alternatives
  5. Apply fixes: correct labels, add alternatives, remove non-entities
  6. Save per-variant and combined fixed benchmark

Usage:
    python audit_all_golds.py                  # Full audit all variants
    python audit_all_golds.py --variant bench_twitter_t2e  # Single variant
    python audit_all_golds.py --skip-llm       # Cross-model consensus only (free)
    python audit_all_golds.py --dry-run
"""
import asyncio
import argparse
import csv
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

sys.path.insert(0, str(Path(__file__).parent))
from openrouter_client import batch_call

DATA_DIR = Path(__file__).parent.parent / "data"
BENCH_DIR = DATA_DIR / "benchmark_v2"
BATTERY_DIR = Path(__file__).parent / "results" / "battery_20260416_114734"

VARIANTS = ["bench_veterans_t2e", "bench_veterans_e2t", "bench_twitter_t2e", "bench_twitter_e2t"]

NON_ENTITY_PATTERNS = [
    r"^\d{4}$", r"^\d{4}s$", r"^[a-z]{1,2}$",
    r"^scr-\d+", r"^m-\d+", r"^spec \d+",
]


def is_non_entity(entity: str) -> bool:
    ent = entity.strip().lower()
    for pattern in NON_ENTITY_PATTERNS:
        if re.match(pattern, ent):
            return True
    return False


def normalize(s: str) -> str:
    return re.sub(r'\s+', ' ', s.strip().lower())


def load_predictions_for_variant(variant: str) -> dict:
    """Load all LLM prediction files for a variant. Returns {model: {uid: row}}."""
    models = {}
    for f in BATTERY_DIR.glob(f"{variant}_*_predictions.csv"):
        method = f.stem.replace(f"{variant}_", "").replace("_predictions", "")
        if "embedding" in method:
            continue  # skip weak embedding predictions
        rows = list(csv.DictReader(open(f, encoding="utf-8")))
        models[method] = {r["uid"]: r for r in rows}
    return models


def load_variant_data(variant: str) -> list[dict]:
    """Load benchmark variant CSV."""
    variant_map = {
        "bench_veterans_t2e": BENCH_DIR / "variants" / "veterans_t2e.csv",
        "bench_veterans_e2t": BENCH_DIR / "variants" / "veterans_e2t.csv",
        "bench_twitter_t2e": BENCH_DIR / "variants" / "twitter_t2e.csv",
        "bench_twitter_e2t": BENCH_DIR / "variants" / "twitter_e2t.csv",
    }
    path = variant_map.get(variant)
    if not path or not path.exists():
        return []
    return list(csv.DictReader(open(path, encoding="utf-8")))


def cross_model_consensus(models: dict, variant_rows: list) -> dict:
    """
    Find samples where multiple models agree on a prediction different from gold.
    Returns {uid: {gold, consensus_pred, n_agree, all_preds, gold_in_any_top3}}.
    """
    uid_to_gold = {r["uid"]: r["entity"] for r in variant_rows}
    all_uids = set(uid_to_gold.keys())

    results = {}
    for uid in all_uids:
        gold = uid_to_gold.get(uid, "")
        gold_norm = normalize(gold)

        preds_per_model = {}
        gold_matched_any = False

        for method, data in models.items():
            if uid not in data:
                continue
            r = data[uid]
            # Check if gold matched in this model
            if r.get("match_tier", "none") != "none":
                gold_matched_any = True

            # Collect top-3 predictions
            for i in range(1, 4):
                pred = r.get(f"pred_{i}", "").strip()
                if pred:
                    preds_per_model.setdefault(method, []).append(pred)

        if gold_matched_any:
            continue  # at least one model matched gold, skip

        # No model matched gold. Find consensus.
        all_top1 = {}
        for method, data in models.items():
            if uid in data:
                p1 = data[uid].get("pred_1", "").strip().lower()
                if p1:
                    all_top1[method] = p1

        if not all_top1:
            continue

        pred_counts = Counter(all_top1.values())
        most_common, count = pred_counts.most_common(1)[0]

        # Also collect all unique predictions across top-3
        all_preds_flat = set()
        for method, pred_list in preds_per_model.items():
            for p in pred_list:
                all_preds_flat.add(p)

        results[uid] = {
            "gold": gold,
            "gold_type": next((r.get("entity_type", "") for r in variant_rows if r["uid"] == uid), ""),
            "consensus_pred": most_common,
            "n_agree": count,
            "n_models": len(all_top1),
            "all_top1": dict(all_top1),
            "all_unique_preds": list(all_preds_flat)[:10],
        }

    return results


async def llm_audit_samples(
    samples_to_check: list[dict],
    variant_rows: list[dict],
    concurrency: int = 15,
) -> list[dict]:
    """Use LLM to validate gold labels and suggest alternatives."""
    uid_to_row = {r["uid"]: r for r in variant_rows}

    prompts = []
    check_list = []
    for item in samples_to_check:
        uid = item["uid"] if isinstance(item, dict) else item
        info = item if isinstance(item, dict) else {"uid": uid}
        row = uid_to_row.get(uid if isinstance(uid, str) else info.get("uid", ""))
        if not row:
            continue

        consensus = info.get("consensus_pred", "")
        all_preds = info.get("all_unique_preds", [])

        prompt = [
            {"role": "system", "content": (
                "You are a data quality auditor for an NLP dataset. "
                "Evaluate if the gold entity label is correct for the given text. "
                "Respond in JSON format only."
            )},
            {"role": "user", "content": (
                f"Text: \"{row['text'][:300]}\"\n"
                f"Gold entity: \"{row['entity']}\"\n"
                f"Gold type: {row.get('entity_type', '')}\n"
                f"Model consensus prediction: \"{consensus}\"\n"
                f"All model predictions: {all_preds[:6]}\n\n"
                f"Respond as JSON:\n"
                f"{{\"gold_correct\": true/false, \"is_real_entity\": true/false, "
                f"\"best_entity\": \"...\", \"alternatives\": [\"...\", \"...\"], "
                f"\"explanation\": \"...\"}}"
            )},
        ]
        prompts.append(prompt)
        check_list.append({"uid": row["uid"], "gold": row["entity"], "type": row.get("entity_type", "")})

    if not prompts:
        return []

    print(f"    LLM auditing {len(prompts)} samples...")
    responses = await batch_call(
        prompts, model="google/gemini-2.0-flash-001:floor",
        temperature=0.1, max_tokens=250, concurrency=concurrency,
        progress_every=100,
    )

    judgments = []
    for i, (info, resp) in enumerate(zip(check_list, responses)):
        if not resp:
            continue
        try:
            json_match = re.search(r'\{[^{}]+\}', resp, re.DOTALL)
            j = json.loads(json_match.group()) if json_match else {}
        except (json.JSONDecodeError, AttributeError):
            j = {}

        j["uid"] = info["uid"]
        j["gold_entity"] = info["gold"]
        j["gold_type"] = info["type"]
        judgments.append(j)

    return judgments


async def audit_variant(variant: str, skip_llm: bool = False, concurrency: int = 15):
    """Full audit of a single variant."""
    print(f"\n{'='*60}")
    print(f"  AUDITING: {variant}")
    print(f"{'='*60}")

    rows = load_variant_data(variant)
    if not rows:
        print(f"  No data found for {variant}")
        return {}

    models = load_predictions_for_variant(variant)
    print(f"  Samples: {len(rows)}, Models: {len(models)} ({', '.join(models.keys())})")

    # Step 1: Non-entity detection
    non_entities = [r for r in rows if is_non_entity(r["entity"])]
    print(f"  Non-entity golds: {len(non_entities)}")

    # Step 2: Cross-model consensus
    consensus = cross_model_consensus(models, rows)
    n_consensus_3 = sum(1 for c in consensus.values() if c["n_agree"] >= 3)
    n_consensus_2 = sum(1 for c in consensus.values() if c["n_agree"] >= 2)
    print(f"  All models fail: {len(consensus)} samples")
    print(f"    2+ model consensus: {n_consensus_2}")
    print(f"    3+ model consensus: {n_consensus_3}")

    # Step 3: LLM audit (on consensus failures)
    judgments = []
    if not skip_llm:
        # Audit samples with 2+ consensus
        to_check = [
            {"uid": uid, **info}
            for uid, info in consensus.items()
            if info["n_agree"] >= 2
        ]
        if to_check:
            judgments = await llm_audit_samples(to_check, rows, concurrency)
            gold_wrong = sum(1 for j in judgments if not j.get("gold_correct", True))
            not_real = sum(1 for j in judgments if not j.get("is_real_entity", True))
            print(f"  LLM audit ({len(judgments)} checked): {gold_wrong} wrong, {not_real} not real entity")

    return {
        "variant": variant,
        "total_samples": len(rows),
        "non_entities": [{"uid": r["uid"], "entity": r["entity"]} for r in non_entities],
        "consensus": {uid: info for uid, info in consensus.items()},
        "judgments": judgments,
    }


def apply_all_fixes(all_audits: dict):
    """Apply fixes from all variant audits to produce final benchmark."""
    print(f"\n{'='*60}")
    print(f"  APPLYING FIXES TO FULL BENCHMARK")
    print(f"{'='*60}")

    # Load full benchmark
    full_path = BENCH_DIR / "irc_benchmark_v2_full.csv"
    rows = list(csv.DictReader(open(full_path, encoding="utf-8")))
    print(f"  Loaded {len(rows)} samples")

    # Collect all fixes
    uid_corrections = {}  # uid -> new entity
    uid_alternatives = {}  # uid -> list of alternatives
    uids_to_remove = set()

    for variant, audit in all_audits.items():
        # Non-entities
        for ne in audit.get("non_entities", []):
            uids_to_remove.add(ne["uid"])

        # LLM judgments
        for j in audit.get("judgments", []):
            uid = j["uid"]
            if not j.get("is_real_entity", True):
                uids_to_remove.add(uid)
            elif not j.get("gold_correct", True):
                best = j.get("best_entity", "")
                alts = j.get("alternatives", [])
                if best:
                    uid_corrections[uid] = best
                if alts:
                    uid_alternatives[uid] = alts

        # High-confidence consensus corrections (3+ models, no LLM yet)
        for uid, info in audit.get("consensus", {}).items():
            if uid not in uid_corrections and uid not in uids_to_remove:
                if info["n_agree"] >= 3:
                    # Add consensus prediction as alternative (not correction)
                    existing = uid_alternatives.get(uid, [])
                    if info["consensus_pred"] not in [a.lower() for a in existing]:
                        uid_alternatives.setdefault(uid, []).append(info["consensus_pred"])

    print(f"  Corrections: {len(uid_corrections)}")
    print(f"  Alternatives added: {len(uid_alternatives)}")
    print(f"  Removed: {len(uids_to_remove)}")

    # Apply
    fixed_rows = []
    for r in rows:
        uid = r["uid"]
        if uid in uids_to_remove:
            continue

        alts = uid_alternatives.get(uid, [])
        r["alternative_entities"] = "|".join(alts) if alts else ""

        if uid in uid_corrections:
            r["entity_original"] = r["entity"]
            r["entity"] = uid_corrections[uid]
            r["gold_corrected"] = "yes"
        else:
            r["entity_original"] = ""
            r["gold_corrected"] = "no"

        fixed_rows.append(r)

    # Save
    out_path = BENCH_DIR / "irc_benchmark_v2_audited.csv"
    fieldnames = list(fixed_rows[0].keys()) if fixed_rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    # Save audit report
    report = {
        "total_before": len(rows),
        "total_after": len(fixed_rows),
        "removed": len(uids_to_remove),
        "corrections": len(uid_corrections),
        "alternatives_added": len(uid_alternatives),
        "per_variant": {},
    }
    for variant, audit in all_audits.items():
        n_ne = len(audit.get("non_entities", []))
        n_cons = len(audit.get("consensus", {}))
        n_judged = len(audit.get("judgments", []))
        n_wrong = sum(1 for j in audit.get("judgments", []) if not j.get("gold_correct", True))
        report["per_variant"][variant] = {
            "non_entities": n_ne,
            "all_models_fail": n_cons,
            "llm_audited": n_judged,
            "gold_wrong": n_wrong,
        }

    report_path = BENCH_DIR / "full_audit_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_path.name} ({len(fixed_rows)} samples)")
    print(f"  Report: {report_path.name}")

    # Summary table
    print(f"\n  Per-variant summary:")
    print(f"  {'Variant':<25s} {'Non-ent':>8s} {'AllFail':>8s} {'Audited':>8s} {'Wrong':>8s}")
    print(f"  {'-'*57}")
    for v, stats in report["per_variant"].items():
        print(f"  {v:<25s} {stats['non_entities']:>8d} {stats['all_models_fail']:>8d} "
              f"{stats['llm_audited']:>8d} {stats['gold_wrong']:>8d}")


async def main():
    parser = argparse.ArgumentParser(description="Audit gold labels across all benchmark variants")
    parser.add_argument("--variant", default="all", help="Single variant or 'all'")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM audit (consensus only, free)")
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    variants = VARIANTS if args.variant == "all" else [args.variant]

    if args.dry_run:
        for v in variants:
            rows = load_variant_data(v)
            models = load_predictions_for_variant(v)
            print(f"  {v}: {len(rows)} samples, {len(models)} models")
        n_total = sum(len(load_variant_data(v)) for v in variants)
        print(f"\n  Total: {n_total} samples to audit")
        print(f"  Estimated LLM cost: ~${n_total * 0.0002:.2f}")
        return

    all_audits = {}
    for v in variants:
        audit = await audit_variant(v, args.skip_llm, args.concurrency)
        all_audits[v] = audit

    apply_all_fixes(all_audits)


if __name__ == "__main__":
    asyncio.run(main())
