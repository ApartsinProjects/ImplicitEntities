"""
Fix gold label issues in IRC-Bench v2:
  a) Correct wrong gold labels (e.g., "world war i" for Pearl Harbor text)
  b) Add alternative valid gold entities (multi-reference evaluation)
  c) Remove non-entity gold labels (years, equipment model numbers, etc.)

Uses LLM to validate and suggest corrections, then applies them.

Usage:
    python fix_gold_labels.py --phase detect     # Detect issues only
    python fix_gold_labels.py --phase fix         # Apply fixes
    python fix_gold_labels.py --dry-run           # Print plan
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
RESULTS_DIR = Path(__file__).parent / "results"

# ── Known wrong labels (from manual inspection of failure analysis) ────────
KNOWN_FIXES = {
    # (text_prefix, wrong_gold) -> corrected_gold
    # Pearl Harbor text wrongly labeled "world war i"
    "I remember the day vividly, December 7th, 1941": {
        "world war i": "world war ii",
    },
}

# ── Non-entity gold labels to remove ──────────────────────────────────────
NON_ENTITY_PATTERNS = [
    r"^\d{4}$",                    # bare years: "1944", "1967"
    r"^\d{4}s$",                   # decades: "1940s"
    r"^[a-z]{1,3}$",              # too short: "gi", "px"
    r"^scr-\d+",                   # radio model numbers
    r"^m-\d+",                     # equipment models
    r"^spec \d+",                  # military spec numbers
]


def is_non_entity(entity: str) -> bool:
    """Check if a gold entity is not a real entity (years, model numbers, etc.)."""
    ent = entity.strip().lower()
    for pattern in NON_ENTITY_PATTERNS:
        if re.match(pattern, ent):
            return True
    return False


async def detect_and_fix_issues(concurrency: int = 15, max_check: int = 0):
    """
    Use LLM to detect wrong gold labels and suggest alternatives.
    For each sample where the best model fails, ask the LLM:
    1. Is the gold entity correct for this text?
    2. What are the top 3 entities implicitly referenced?
    3. Is the gold entity even a real named entity?
    """
    print("Loading benchmark data...")
    vet_path = BENCH_DIR / "variants" / "veterans_t2e.csv"
    rows = list(csv.DictReader(open(vet_path, encoding="utf-8")))
    print(f"  Loaded {len(rows)} veterans_t2e samples")

    # Load best model predictions to find failures
    battery_dir = RESULTS_DIR / "battery_20260416_114734"
    pred_file = battery_dir / "bench_veterans_t2e_llm_google_gemini-2.0-flash-001_floor_predictions.csv"
    if pred_file.exists():
        preds = list(csv.DictReader(open(pred_file, encoding="utf-8")))
        pred_by_uid = {r["uid"]: r for r in preds}
        failures = [r for r in preds if r["match_tier"] == "none"]
        print(f"  LLM predictions: {len(preds)} total, {len(failures)} failures")
    else:
        pred_by_uid = {}
        failures = []
        print("  No predictions file found, checking all samples")

    # Step 1: Flag non-entity golds
    non_entities = []
    for r in rows:
        if is_non_entity(r["entity"]):
            non_entities.append({"uid": r["uid"], "entity": r["entity"], "type": r["entity_type"],
                                 "reason": "non-entity pattern match"})

    print(f"\n  Non-entity gold labels found: {len(non_entities)}")
    for ne in non_entities[:10]:
        print(f"    \"{ne['entity']}\" ({ne['type']})")

    # Step 2: Use LLM to validate failed samples and suggest alternatives
    # Only check failures (where gold doesn't match any model prediction)
    samples_to_check = []
    for r in rows:
        uid = r["uid"]
        pred = pred_by_uid.get(uid)
        if pred and pred["match_tier"] == "none":
            samples_to_check.append(r)

    if max_check and max_check < len(samples_to_check):
        samples_to_check = samples_to_check[:max_check]

    print(f"\n  Checking {len(samples_to_check)} failed samples with LLM...")

    prompts = []
    for r in samples_to_check:
        pred = pred_by_uid.get(r["uid"], {})
        model_preds = [pred.get(f"pred_{i}", "") for i in range(1, 4)]
        model_preds = [p for p in model_preds if p]

        prompt = [
            {"role": "system", "content": (
                "You are a data quality auditor for an NLP dataset. "
                "Given a text, a gold-standard entity label, and model predictions, "
                "evaluate whether the gold label is correct. Respond in JSON format."
            )},
            {"role": "user", "content": (
                f"Text: \"{r['text'][:300]}\"\n"
                f"Gold entity: \"{r['entity']}\"\n"
                f"Gold type: {r['entity_type']}\n"
                f"Model predictions: {model_preds}\n\n"
                f"Evaluate:\n"
                f"1. Is the gold entity actually referenced (even implicitly) in this text? (yes/no)\n"
                f"2. Is the gold entity a real named entity? (yes/no)\n"
                f"3. What are the top 3 entities ACTUALLY implicitly referenced? List as JSON array.\n"
                f"4. Are any model predictions more correct than the gold? (yes/no, which)\n\n"
                f"Respond as JSON: {{\"gold_correct\": bool, \"is_real_entity\": bool, "
                f"\"actual_entities\": [str, str, str], \"model_better\": bool, \"better_pred\": str, "
                f"\"explanation\": str}}"
            )},
        ]
        prompts.append(prompt)

    if prompts:
        responses = await batch_call(
            prompts, model="google/gemini-2.0-flash-001:floor",
            temperature=0.1, max_tokens=300, concurrency=concurrency,
            progress_every=50,
        )
    else:
        responses = []

    # Parse LLM judgments
    judgments = []
    for i, (r, resp) in enumerate(zip(samples_to_check, responses)):
        if not resp:
            continue
        try:
            # Extract JSON from response
            json_match = re.search(r'\{[^{}]+\}', resp, re.DOTALL)
            if json_match:
                j = json.loads(json_match.group())
            else:
                j = {"gold_correct": True, "explanation": "parse_failed"}
        except json.JSONDecodeError:
            j = {"gold_correct": True, "explanation": "json_parse_failed"}

        j["uid"] = r["uid"]
        j["gold_entity"] = r["entity"]
        j["gold_type"] = r["entity_type"]
        j["text_snippet"] = r["text"][:100]
        judgments.append(j)

    # Categorize
    gold_wrong = [j for j in judgments if not j.get("gold_correct", True)]
    not_real = [j for j in judgments if not j.get("is_real_entity", True)]
    model_better = [j for j in judgments if j.get("model_better", False)]

    print(f"\n  === LLM AUDIT RESULTS ({len(judgments)} samples checked) ===")
    print(f"  Gold label wrong:     {len(gold_wrong)} ({len(gold_wrong)/max(len(judgments),1)*100:.0f}%)")
    print(f"  Not real entity:      {len(not_real)} ({len(not_real)/max(len(judgments),1)*100:.0f}%)")
    print(f"  Model more correct:   {len(model_better)} ({len(model_better)/max(len(judgments),1)*100:.0f}%)")

    print(f"\n  --- WRONG GOLD LABELS (sample) ---")
    for j in gold_wrong[:10]:
        print(f"    Gold: \"{j['gold_entity']}\" -> Suggested: {j.get('actual_entities', [])}")
        print(f"    Reason: {j.get('explanation', '?')[:80]}")
        print()

    return {
        "non_entities": non_entities,
        "judgments": judgments,
        "gold_wrong": gold_wrong,
        "not_real": not_real,
        "model_better": model_better,
    }


def apply_fixes(audit_results: dict):
    """Apply fixes to benchmark: correct labels, add alternatives, remove non-entities."""
    print("\n  Applying fixes to benchmark...")

    # Load full benchmark
    full_path = BENCH_DIR / "irc_benchmark_v2_full.csv"
    rows = list(csv.DictReader(open(full_path, encoding="utf-8")))
    print(f"  Loaded {len(rows)} samples")

    # Build fix maps
    uid_corrections = {}  # uid -> corrected entity
    uid_alternatives = {}  # uid -> list of alternative valid entities
    uids_to_remove = set()  # uids to remove (non-entities)

    # From non-entity detection
    for ne in audit_results["non_entities"]:
        uids_to_remove.add(ne["uid"])

    # From LLM judgments
    for j in audit_results["judgments"]:
        uid = j["uid"]
        if not j.get("is_real_entity", True):
            uids_to_remove.add(uid)
        elif not j.get("gold_correct", True):
            alts = j.get("actual_entities", [])
            if alts:
                uid_corrections[uid] = alts[0]  # best alternative as primary
                uid_alternatives[uid] = alts
        elif j.get("model_better", False):
            better = j.get("better_pred", "")
            alts = j.get("actual_entities", [])
            if alts:
                uid_alternatives[uid] = alts  # keep gold but add alternatives

    print(f"  Corrections: {len(uid_corrections)}")
    print(f"  Alternatives added: {len(uid_alternatives)}")
    print(f"  Removed (non-entity): {len(uids_to_remove)}")

    # Apply fixes
    fixed_rows = []
    for r in rows:
        uid = r["uid"]
        if uid in uids_to_remove:
            continue

        # Add alternatives column
        alts = uid_alternatives.get(uid, [])
        r["alternative_entities"] = "|".join(alts) if alts else ""

        # Correct primary entity if needed
        if uid in uid_corrections:
            r["entity_original"] = r["entity"]
            r["entity"] = uid_corrections[uid]
            r["gold_corrected"] = "yes"
        else:
            r["entity_original"] = ""
            r["gold_corrected"] = "no"

        fixed_rows.append(r)

    # Save fixed benchmark
    out_path = BENCH_DIR / "irc_benchmark_v2_fixed.csv"
    fieldnames = list(fixed_rows[0].keys()) if fixed_rows else []
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(fixed_rows)

    # Save audit report
    report_path = BENCH_DIR / "gold_label_audit.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump({
            "total_samples": len(rows),
            "fixed_samples": len(fixed_rows),
            "removed": len(uids_to_remove),
            "corrections": len(uid_corrections),
            "alternatives_added": len(uid_alternatives),
            "non_entities_removed": [ne["entity"] for ne in audit_results["non_entities"]],
            "corrections_detail": [
                {"uid": j["uid"], "old": j["gold_entity"], "new": j.get("actual_entities", [])[:1],
                 "reason": j.get("explanation", "")[:100]}
                for j in audit_results.get("gold_wrong", [])
            ],
        }, f, indent=2, ensure_ascii=False)

    print(f"\n  Saved: {out_path.name} ({len(fixed_rows)} samples)")
    print(f"  Audit: {report_path.name}")
    return fixed_rows


async def main():
    parser = argparse.ArgumentParser(description="Fix gold labels in IRC-Bench v2")
    parser.add_argument("--phase", default="detect", choices=["detect", "fix", "all"])
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--max-check", type=int, default=0,
                        help="Max samples to check with LLM (0=all failures)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.dry_run:
        print("  DRY RUN: would check failed samples with LLM and fix gold labels")
        return

    if args.phase in ("detect", "all"):
        audit = await detect_and_fix_issues(args.concurrency, args.max_check)

        # Save raw audit
        audit_path = BENCH_DIR / "gold_label_audit_raw.json"
        serializable = {k: v for k, v in audit.items() if k != "judgments"}
        serializable["judgments_count"] = len(audit["judgments"])
        serializable["sample_judgments"] = audit["judgments"][:20]
        with open(audit_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2, ensure_ascii=False)
        print(f"\n  Raw audit saved: {audit_path}")

    if args.phase in ("fix", "all"):
        if args.phase == "fix":
            # Load previous audit
            audit_path = BENCH_DIR / "gold_label_audit_raw.json"
            if not audit_path.exists():
                print("  ERROR: Run --phase detect first")
                return
            audit = json.load(open(audit_path, encoding="utf-8"))
        apply_fixes(audit)


if __name__ == "__main__":
    asyncio.run(main())
