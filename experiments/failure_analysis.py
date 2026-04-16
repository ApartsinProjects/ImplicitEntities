"""
Comprehensive Failure Analysis across ALL experiment prediction files.
Analyzes cross-method failures, prediction quality, false negatives,
entity difficulty, and generates recommendations.
"""

import csv
import json
import os
import re
from collections import defaultdict, Counter
from pathlib import Path
from difflib import SequenceMatcher

RESULTS_DIR = Path(r"E:\Projects\ImplicitEntities\experiments\results")
OUTPUT_DIR = Path(r"E:\Projects\ImplicitEntities\experiments")

# ─────────────────────────────────────────────────────────
# 1. Load ALL prediction CSVs
# ─────────────────────────────────────────────────────────

def load_all_predictions():
    """Load all prediction CSVs and parse their metadata from filenames."""
    files = sorted(RESULTS_DIR.glob("*_predictions.csv"))
    print(f"Found {len(files)} prediction CSV files\n")

    all_data = []
    for f in files:
        fname = f.stem  # without .csv
        # Parse metadata from filename
        # Pattern: TIMESTAMP_DATASET_METHOD_MODEL_predictions
        # Remove the trailing _predictions
        parts = fname.replace("_predictions", "")

        rows = []
        with open(f, "r", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                rows.append(row)

        info = {
            "file": f.name,
            "path": str(f),
            "basename": parts,
            "rows": rows,
            "count": len(rows),
        }

        # Determine dataset and method from filename
        info["dataset"], info["method"], info["variant"] = parse_filename(parts)

        all_data.append(info)
        print(f"  Loaded {f.name}: {len(rows)} rows [{info['dataset']} / {info['method']} / {info['variant']}]")

    print()
    return all_data


def parse_filename(basename):
    """Extract dataset, method, variant from filename basename."""
    # Remove timestamp prefix (first token before _)
    parts = basename.split("_", 1)
    if len(parts) > 1:
        rest = parts[1]
    else:
        rest = parts[0]

    # Detect dataset
    dataset = "unknown"
    if "e2t_twitter" in rest:
        dataset = "e2t_twitter"
    elif "e2t_veterans" in rest:
        dataset = "e2t_veterans"
    elif "twitter" in rest and "e2t" not in rest:
        dataset = "twitter"
    elif "veterans_t2e" in rest or "veterans" in rest:
        dataset = "veterans_t2e"

    # Detect method
    method = "unknown"
    variant = "default"
    if "ablation_no_type_no_ctx" in rest:
        method = "ablation"
        variant = "no_type_no_ctx"
    elif "ablation_no_context" in rest:
        method = "ablation"
        variant = "no_context"
    elif "ablation_no_type" in rest:
        method = "ablation"
        variant = "no_type"
    elif "multimodel_meta-llama" in rest:
        method = "llm"
        variant = "llama-3.1-8b"
    elif "multimodel_mistralai" in rest:
        method = "llm"
        variant = "mistral-7b"
    elif "emb_all-MiniLM" in rest:
        method = "embedding"
        variant = "all-MiniLM-L6-v2"
    elif "emb_all-mpnet" in rest:
        method = "embedding"
        variant = "all-mpnet-base-v2"
    elif "emb_bge-small" in rest:
        method = "embedding"
        variant = "bge-small-en-v1.5"
    elif "emb_multi-qa" in rest:
        method = "embedding"
        variant = "multi-qa-MiniLM-L6"
    elif "_hybrid_" in rest:
        method = "hybrid"
    elif "_embedding_" in rest:
        method = "embedding"
    elif "_llm_" in rest:
        method = "llm"

    return dataset, method, variant


def is_match(row):
    """Check if a row counts as a successful match."""
    tier = row.get("match_tier", "none")
    return tier != "none" and tier != ""


# ─────────────────────────────────────────────────────────
# 2. Cross-method failure analysis
# ─────────────────────────────────────────────────────────

def cross_method_analysis(all_data):
    """For each dataset, compare LLM vs embedding vs hybrid."""
    print("=" * 70)
    print("CROSS-METHOD FAILURE ANALYSIS")
    print("=" * 70)

    results = {}

    # Group by dataset, find the 3 core methods (default variant only)
    dataset_methods = defaultdict(dict)
    for info in all_data:
        if info["variant"] == "default" and info["method"] in ("llm", "embedding", "hybrid"):
            key = (info["dataset"], info["method"])
            dataset_methods[info["dataset"]][info["method"]] = info

    for dataset in sorted(dataset_methods.keys()):
        methods = dataset_methods[dataset]
        if len(methods) < 2:
            continue

        print(f"\n{'-' * 60}")
        print(f"Dataset: {dataset}")
        print(f"Methods available: {sorted(methods.keys())}")
        print(f"{'-' * 60}")

        # Build uid-indexed lookup
        uid_results = defaultdict(dict)
        uid_data = {}
        for mname, info in methods.items():
            for row in info["rows"]:
                uid = row["uid"]
                uid_results[uid][mname] = is_match(row)
                if uid not in uid_data:
                    uid_data[uid] = row  # Store one copy of the row for context

        total = len(uid_results)
        all_fail = 0
        any_succeed = 0
        union_succeed = 0
        universal_failures = []
        disagreements = []

        method_names = sorted(methods.keys())

        for uid, mresults in uid_results.items():
            successes = [m for m in method_names if mresults.get(m, False)]
            failures = [m for m in method_names if not mresults.get(m, False)]

            if len(successes) == 0:
                all_fail += 1
                universal_failures.append(uid)
            else:
                any_succeed += 1
                union_succeed += 1

            if 0 < len(successes) < len(method_names):
                disagreements.append((uid, successes, failures))

        # Per-method accuracy
        method_acc = {}
        for mname in method_names:
            info = methods[mname]
            matched = sum(1 for r in info["rows"] if is_match(r))
            method_acc[mname] = matched / len(info["rows"]) if info["rows"] else 0

        print(f"Total samples: {total}")
        print(f"All methods fail: {all_fail} ({100*all_fail/total:.1f}%)")
        print(f"At least one succeeds: {any_succeed} ({100*any_succeed/total:.1f}%)")
        print(f"Union ceiling (best ensemble): {union_succeed}/{total} = {100*union_succeed/total:.1f}%")
        print(f"Per-method accuracy:")
        for m in method_names:
            print(f"  {m}: {100*method_acc[m]:.1f}%")

        # Show examples of universal failures
        print(f"\nUniversal failures (all methods wrong) - showing up to 10:")
        shown = 0
        for uid in universal_failures[:10]:
            row = uid_data[uid]
            text_preview = row["text"][:100].replace("\n", " ")
            pred1_vals = {m: methods[m]["rows"] for m in method_names}
            # Get pred_1 from each method
            preds = {}
            for m in method_names:
                for r in methods[m]["rows"]:
                    if r["uid"] == uid:
                        preds[m] = r.get("pred_1", "N/A")
                        break
            print(f"  [{uid}] gold='{row['gold_entity']}' type={row['entity_type']}")
            print(f"    text: {text_preview}...")
            for m in method_names:
                print(f"    {m} pred_1: {preds.get(m, 'N/A')}")
            shown += 1

        # Show examples of disagreements
        print(f"\nDisagreements (one right, others wrong) - showing up to 10:")
        for uid, successes, failures in disagreements[:10]:
            row = uid_data[uid]
            text_preview = row["text"][:80].replace("\n", " ")
            preds = {}
            for m in method_names:
                for r in methods[m]["rows"]:
                    if r["uid"] == uid:
                        preds[m] = r.get("pred_1", "N/A")
                        break
            print(f"  [{uid}] gold='{row['gold_entity']}' | won: {successes} | lost: {failures}")
            for m in method_names:
                marker = "OK" if m in successes else "FAIL"
                print(f"    {m} [{marker}] pred_1: {preds.get(m, 'N/A')}")

        results[dataset] = {
            "total": total,
            "all_fail": all_fail,
            "any_succeed": any_succeed,
            "union_ceiling_pct": round(100 * union_succeed / total, 2) if total else 0,
            "method_accuracy": {m: round(100 * method_acc[m], 2) for m in method_names},
            "num_disagreements": len(disagreements),
            "universal_failure_count": all_fail,
            "universal_failure_uids": universal_failures[:20],
        }

    return results


# ─────────────────────────────────────────────────────────
# 3. Prediction quality analysis
# ─────────────────────────────────────────────────────────

def normalize(s):
    """Lowercase, strip articles, punctuation for comparison."""
    s = s.lower().strip()
    s = re.sub(r'^(the|a|an)\s+', '', s)
    s = re.sub(r'[^\w\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s


def classify_prediction(gold, pred):
    """Classify a failed prediction into categories."""
    if not pred or pred.strip() == "":
        return "empty"

    gold_n = normalize(gold)
    pred_n = normalize(pred)

    if not gold_n or not pred_n:
        return "empty"

    # Check if prediction is more specific than gold
    # pred contains gold as substring, or gold is a short form
    if gold_n in pred_n and len(pred_n) > len(gold_n) * 1.3:
        return "more_specific"

    # Check if gold is more specific than pred
    if pred_n in gold_n and len(gold_n) > len(pred_n) * 1.3:
        return "more_general"

    # Check similarity
    ratio = SequenceMatcher(None, gold_n, pred_n).ratio()
    if ratio > 0.6:
        return "near_match"

    # Check if they share significant words
    gold_words = set(gold_n.split())
    pred_words = set(pred_n.split())
    if gold_words and pred_words:
        overlap = gold_words & pred_words
        # Remove very common words
        stopwords = {"the", "a", "an", "of", "in", "on", "at", "to", "for", "and", "or", "is", "was"}
        meaningful_overlap = overlap - stopwords
        if meaningful_overlap and len(meaningful_overlap) >= 1:
            return "related"

    return "completely_wrong"


def prediction_quality_analysis(all_data):
    """Analyze the quality of failed predictions."""
    print("\n" + "=" * 70)
    print("PREDICTION QUALITY ANALYSIS")
    print("=" * 70)

    results = {}

    for info in all_data:
        failures = [r for r in info["rows"] if not is_match(r)]
        if not failures:
            continue

        categories = Counter()
        examples = defaultdict(list)

        for row in failures:
            gold = row.get("gold_entity", "")
            pred1 = row.get("pred_1", "")
            cat = classify_prediction(gold, pred1)
            categories[cat] += 1
            if len(examples[cat]) < 3:
                examples[cat].append({
                    "gold": gold,
                    "pred_1": pred1,
                    "uid": row["uid"],
                })

        total_fail = len(failures)
        label = f"{info['dataset']}/{info['method']}/{info['variant']}"

        print(f"\n{label} ({total_fail} failures):")
        for cat in ["more_specific", "near_match", "related", "more_general", "completely_wrong", "empty"]:
            count = categories.get(cat, 0)
            pct = 100 * count / total_fail if total_fail else 0
            print(f"  {cat}: {count} ({pct:.1f}%)")
            for ex in examples.get(cat, []):
                print(f"    e.g. gold='{ex['gold']}' vs pred='{ex['pred_1']}'")

        results[label] = {
            "total_failures": total_fail,
            "categories": dict(categories),
        }

    return results


# ─────────────────────────────────────────────────────────
# 4. False negative detection
# ─────────────────────────────────────────────────────────

def enhanced_match(gold, pred):
    """Check if gold and pred should match with better normalization."""
    if not gold or not pred:
        return False

    g = normalize(gold)
    p = normalize(pred)

    if not g or not p:
        return False

    # Exact after normalization
    if g == p:
        return True

    # One contains the other
    if g in p or p in g:
        return True

    # High similarity
    if SequenceMatcher(None, g, p).ratio() > 0.85:
        return True

    # Abbreviation check: compare initials
    g_words = g.split()
    p_words = p.split()
    if len(g_words) > 1 and len(p_words) == 1 and len(p_words[0]) <= 5:
        initials = "".join(w[0] for w in g_words if w)
        if initials == p_words[0]:
            return True
    if len(p_words) > 1 and len(g_words) == 1 and len(g_words[0]) <= 5:
        initials = "".join(w[0] for w in p_words if w)
        if initials == g_words[0]:
            return True

    # Spelling variants: remove common suffixes
    for suffix in ["ing", "tion", "ment", "ed", "er", "est", "ly", "s"]:
        g_stripped = g.rstrip(suffix) if g.endswith(suffix) else g
        p_stripped = p.rstrip(suffix) if p.endswith(suffix) else p
        if g_stripped and p_stripped and SequenceMatcher(None, g_stripped, p_stripped).ratio() > 0.9:
            return True

    return False


def false_negative_detection(all_data):
    """Check if any pred_1 through pred_10 SHOULD match gold but matching failed."""
    print("\n" + "=" * 70)
    print("FALSE NEGATIVE DETECTION")
    print("=" * 70)

    results = {}

    for info in all_data:
        failures = [r for r in info["rows"] if not is_match(r)]
        if not failures:
            continue

        false_negs = []
        article_issues = []
        abbreviation_issues = []
        spelling_issues = []
        containment_issues = []

        for row in failures:
            gold = row.get("gold_entity", "")
            for i in range(1, 11):
                pred = row.get(f"pred_{i}", "")
                if not pred:
                    continue
                if enhanced_match(gold, pred):
                    g_norm = normalize(gold)
                    p_norm = normalize(pred)

                    issue_type = "general_normalization"

                    # Classify the type of mismatch
                    g_lower = gold.lower().strip()
                    p_lower = pred.lower().strip()

                    # Article difference
                    if re.sub(r'^(the|a|an)\s+', '', g_lower) == re.sub(r'^(the|a|an)\s+', '', p_lower):
                        issue_type = "article_difference"
                        article_issues.append({"uid": row["uid"], "gold": gold, "pred": pred, "rank": i})
                    # Containment
                    elif g_norm in p_norm or p_norm in g_norm:
                        issue_type = "containment"
                        containment_issues.append({"uid": row["uid"], "gold": gold, "pred": pred, "rank": i})
                    # High similarity but not identical
                    elif SequenceMatcher(None, g_norm, p_norm).ratio() > 0.85:
                        issue_type = "spelling_variant"
                        spelling_issues.append({"uid": row["uid"], "gold": gold, "pred": pred, "rank": i})

                    false_negs.append({
                        "uid": row["uid"],
                        "gold": gold,
                        "pred": pred,
                        "pred_rank": i,
                        "issue_type": issue_type,
                    })
                    break  # Only count once per sample

        label = f"{info['dataset']}/{info['method']}/{info['variant']}"
        total_fail = len(failures)
        total_fn = len(false_negs)

        print(f"\n{label}:")
        print(f"  Total failures: {total_fail}")
        print(f"  False negatives (should have matched): {total_fn} ({100*total_fn/total_fail:.1f}% of failures)" if total_fail else "  No failures")
        print(f"    Article differences (the/a/an): {len(article_issues)}")
        print(f"    Containment (substring): {len(containment_issues)}")
        print(f"    Spelling variants: {len(spelling_issues)}")

        if false_negs:
            print(f"  Examples (up to 5):")
            for fn in false_negs[:5]:
                print(f"    gold='{fn['gold']}' vs pred_{fn['pred_rank']}='{fn['pred']}' [{fn['issue_type']}]")

        current_matches = sum(1 for r in info["rows"] if is_match(r))
        improved = current_matches + total_fn
        total = len(info["rows"])

        print(f"  Current accuracy: {current_matches}/{total} = {100*current_matches/total:.1f}%")
        print(f"  With better normalization: {improved}/{total} = {100*improved/total:.1f}% (+{100*total_fn/total:.1f}pp)")

        results[label] = {
            "total_failures": total_fail,
            "false_negatives": total_fn,
            "additional_matches_pct": round(100 * total_fn / total, 2) if total else 0,
            "article_issues": len(article_issues),
            "containment_issues": len(containment_issues),
            "spelling_issues": len(spelling_issues),
            "current_accuracy_pct": round(100 * current_matches / total, 2) if total else 0,
            "improved_accuracy_pct": round(100 * improved / total, 2) if total else 0,
            "examples": false_negs[:10],
        }

    return results


# ─────────────────────────────────────────────────────────
# 5. Entity difficulty ranking
# ─────────────────────────────────────────────────────────

def entity_difficulty_ranking(all_data):
    """Rank entities by difficulty across all methods."""
    print("\n" + "=" * 70)
    print("ENTITY DIFFICULTY RANKING")
    print("=" * 70)

    # Track success/failure per gold entity across all experiments
    entity_stats = defaultdict(lambda: {"attempts": 0, "successes": 0, "entity_type": "", "datasets": set(), "methods": set()})

    for info in all_data:
        for row in info["rows"]:
            gold = row.get("gold_entity", "").strip()
            if not gold:
                continue
            key = gold.lower()
            entity_stats[key]["attempts"] += 1
            if is_match(row):
                entity_stats[key]["successes"] += 1
            entity_stats[key]["entity_type"] = row.get("entity_type", "")
            entity_stats[key]["datasets"].add(info["dataset"])
            entity_stats[key]["methods"].add(info["method"])
            entity_stats[key]["display_name"] = gold

    # Compute success rate and categorize
    entities = []
    for key, stats in entity_stats.items():
        rate = stats["successes"] / stats["attempts"] if stats["attempts"] else 0
        entities.append({
            "entity": stats.get("display_name", key),
            "entity_lower": key,
            "entity_type": stats["entity_type"],
            "attempts": stats["attempts"],
            "successes": stats["successes"],
            "success_rate": rate,
            "datasets": list(stats["datasets"]),
            "methods": list(stats["methods"]),
        })

    # Sort by success rate (hardest first)
    entities.sort(key=lambda x: (x["success_rate"], -x["attempts"]))

    always_easy = [e for e in entities if e["success_rate"] > 0.8]
    sometimes_hard = [e for e in entities if 0.3 <= e["success_rate"] <= 0.8]
    always_hard = [e for e in entities if e["success_rate"] < 0.3]

    print(f"\nTotal unique entities: {len(entities)}")
    print(f"  Always easy (>80% success): {len(always_easy)}")
    print(f"  Sometimes hard (30-80%): {len(sometimes_hard)}")
    print(f"  Always hard (<30%): {len(always_hard)}")

    print(f"\nHardest entities (lowest success rate, min 3 attempts):")
    shown = 0
    for e in entities:
        if e["attempts"] >= 3 and shown < 20:
            print(f"  '{e['entity']}' [{e['entity_type']}]: {e['successes']}/{e['attempts']} = {100*e['success_rate']:.0f}% | datasets: {e['datasets']}")
            shown += 1

    print(f"\nEasiest entities (highest success rate, min 3 attempts):")
    for e in sorted(entities, key=lambda x: (-x["success_rate"], -x["attempts"]))[:15]:
        if e["attempts"] >= 3:
            print(f"  '{e['entity']}' [{e['entity_type']}]: {e['successes']}/{e['attempts']} = {100*e['success_rate']:.0f}%")

    # By entity type
    type_stats = defaultdict(lambda: {"total": 0, "matched": 0})
    for e in entities:
        t = e["entity_type"]
        type_stats[t]["total"] += e["attempts"]
        type_stats[t]["matched"] += e["successes"]

    print(f"\nSuccess rate by entity type:")
    for t, s in sorted(type_stats.items(), key=lambda x: x[1]["matched"]/max(x[1]["total"],1)):
        rate = s["matched"] / s["total"] if s["total"] else 0
        print(f"  {t}: {s['matched']}/{s['total']} = {100*rate:.1f}%")

    return {
        "total_unique_entities": len(entities),
        "always_easy_count": len(always_easy),
        "sometimes_hard_count": len(sometimes_hard),
        "always_hard_count": len(always_hard),
        "hardest_entities": [
            {"entity": e["entity"], "type": e["entity_type"], "success_rate": round(e["success_rate"], 3),
             "attempts": e["attempts"], "successes": e["successes"]}
            for e in entities if e["attempts"] >= 3
        ][:30],
        "easiest_entities": [
            {"entity": e["entity"], "type": e["entity_type"], "success_rate": round(e["success_rate"], 3),
             "attempts": e["attempts"], "successes": e["successes"]}
            for e in sorted(entities, key=lambda x: (-x["success_rate"], -x["attempts"])) if e["attempts"] >= 3
        ][:30],
        "by_entity_type": {
            t: {"total": s["total"], "matched": s["matched"],
                "rate_pct": round(100 * s["matched"] / s["total"], 2) if s["total"] else 0}
            for t, s in type_stats.items()
        },
    }


# ─────────────────────────────────────────────────────────
# 6. Recommendations
# ─────────────────────────────────────────────────────────

def generate_recommendations(cross_results, quality_results, fn_results, difficulty_results):
    """Generate actionable recommendations based on findings."""
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)

    recs = []

    # 1. Check false negative rates
    total_fn_gain = 0
    total_fn_count = 0
    for label, data in fn_results.items():
        total_fn_gain += data.get("additional_matches_pct", 0)
        total_fn_count += 1
    avg_fn_gain = total_fn_gain / total_fn_count if total_fn_count else 0

    if avg_fn_gain > 1:
        rec = f"NORMALIZATION IMPROVEMENT (avg +{avg_fn_gain:.1f}pp): Improve entity matching with better normalization. Key issues: article stripping (the/a/an), substring containment matching, spelling variant handling. This is the lowest-hanging fruit."
        recs.append(rec)
        print(f"\n1. {rec}")

    # 2. Check union ceiling vs best method
    for dataset, data in cross_results.items():
        ceiling = data.get("union_ceiling_pct", 0)
        best_method_acc = max(data.get("method_accuracy", {}).values()) if data.get("method_accuracy") else 0
        gap = ceiling - best_method_acc
        if gap > 3:
            rec = f"ENSEMBLE FOR {dataset} (+{gap:.1f}pp): Union ceiling is {ceiling:.1f}% vs best single method at {best_method_acc:.1f}%. An ensemble or voting strategy combining methods could capture this gap."
            recs.append(rec)
            print(f"\n2. {rec}")

    # 3. Check prediction quality
    total_near = 0
    total_specific = 0
    total_related = 0
    total_wrong = 0
    total_analyzed = 0
    for label, data in quality_results.items():
        cats = data.get("categories", {})
        total_near += cats.get("near_match", 0)
        total_specific += cats.get("more_specific", 0)
        total_related += cats.get("related", 0)
        total_wrong += cats.get("completely_wrong", 0)
        total_analyzed += data.get("total_failures", 0)

    if total_analyzed > 0:
        near_pct = 100 * total_near / total_analyzed
        spec_pct = 100 * total_specific / total_analyzed
        related_pct = 100 * total_related / total_analyzed
        wrong_pct = 100 * total_wrong / total_analyzed

        rec = f"FAILURE BREAKDOWN: Of {total_analyzed} failures, {near_pct:.1f}% are near-matches, {spec_pct:.1f}% are more-specific, {related_pct:.1f}% are related, and {wrong_pct:.1f}% are completely wrong. Focus on the {100-wrong_pct:.1f}% that show partial understanding."
        recs.append(rec)
        print(f"\n3. {rec}")

    # 4. Entity type difficulty
    type_data = difficulty_results.get("by_entity_type", {})
    hardest_type = None
    lowest_rate = 100
    for t, s in type_data.items():
        if s["rate_pct"] < lowest_rate and s["total"] >= 10:
            lowest_rate = s["rate_pct"]
            hardest_type = t

    if hardest_type:
        rec = f"HARDEST ENTITY TYPE: '{hardest_type}' at {lowest_rate:.1f}% success. Consider type-specific prompting or specialized entity databases for this category."
        recs.append(rec)
        print(f"\n4. {rec}")

    # 5. Universal failures
    total_universal = sum(d.get("all_fail", 0) for d in cross_results.values())
    total_samples = sum(d.get("total", 0) for d in cross_results.values())
    if total_samples > 0:
        univ_pct = 100 * total_universal / total_samples
        rec = f"HARD FLOOR: {total_universal}/{total_samples} samples ({univ_pct:.1f}%) fail ALL methods. These represent the genuine difficulty ceiling of the task. Many are ambiguous texts or extremely obscure entities."
        recs.append(rec)
        print(f"\n5. {rec}")

    # 6. Realistic ceiling estimate
    # The ceiling is: current best + false negative recovery + ensemble gap
    best_overall = 0
    for dataset, data in cross_results.items():
        best = max(data.get("method_accuracy", {}).values()) if data.get("method_accuracy") else 0
        if best > best_overall:
            best_overall = best

    ceiling_est = min(100, best_overall + avg_fn_gain + 5)  # conservative ensemble boost
    rec = f"REALISTIC CEILING ESTIMATE: Current best single method ~{best_overall:.1f}%. With normalization fixes (+~{avg_fn_gain:.1f}pp) and ensemble methods (+~5pp), realistic ceiling is approximately {ceiling_est:.0f}%. The remaining gap requires fundamentally different approaches (knowledge graphs, retrieval augmentation)."
    recs.append(rec)
    print(f"\n6. {rec}")

    return recs


# ─────────────────────────────────────────────────────────
# Report generation
# ─────────────────────────────────────────────────────────

def write_markdown_report(cross_results, quality_results, fn_results, difficulty_results, recommendations):
    """Write human-readable markdown report."""
    lines = []
    lines.append("# Failure Analysis Report")
    lines.append("")
    lines.append("Comprehensive analysis of all 22 prediction files across datasets, methods, and entity types.")
    lines.append("")

    # Cross-method
    lines.append("## 1. Cross-Method Failure Analysis")
    lines.append("")
    for dataset, data in sorted(cross_results.items()):
        lines.append(f"### Dataset: {dataset}")
        lines.append("")
        lines.append(f"- Total samples: {data['total']}")
        lines.append(f"- All methods fail: {data['all_fail']} ({100*data['all_fail']/data['total']:.1f}%)")
        lines.append(f"- At least one succeeds: {data['any_succeed']} ({100*data['any_succeed']/data['total']:.1f}%)")
        lines.append(f"- **Union ceiling (best ensemble): {data['union_ceiling_pct']:.1f}%**")
        lines.append(f"- Method accuracy:")
        for m, acc in sorted(data['method_accuracy'].items()):
            lines.append(f"  - {m}: {acc:.1f}%")
        lines.append(f"- Disagreements between methods: {data['num_disagreements']}")
        lines.append("")

    # Prediction quality
    lines.append("## 2. Prediction Quality Analysis")
    lines.append("")
    lines.append("Categories of failed predictions (pred_1 vs gold):")
    lines.append("")
    lines.append("| Experiment | Failures | Near Match | More Specific | Related | Completely Wrong | Empty |")
    lines.append("|---|---|---|---|---|---|---|")
    for label, data in sorted(quality_results.items()):
        cats = data.get("categories", {})
        t = data["total_failures"]
        lines.append(f"| {label} | {t} | {cats.get('near_match',0)} | {cats.get('more_specific',0)} | {cats.get('related',0)} | {cats.get('completely_wrong',0)} | {cats.get('empty',0)} |")
    lines.append("")

    # False negatives
    lines.append("## 3. False Negative Detection")
    lines.append("")
    lines.append("Predictions that SHOULD have matched gold with better normalization:")
    lines.append("")
    lines.append("| Experiment | Failures | False Negatives | Current Acc | Improved Acc | Gain |")
    lines.append("|---|---|---|---|---|---|")
    for label, data in sorted(fn_results.items()):
        lines.append(f"| {label} | {data['total_failures']} | {data['false_negatives']} | {data['current_accuracy_pct']:.1f}% | {data['improved_accuracy_pct']:.1f}% | +{data['additional_matches_pct']:.1f}pp |")
    lines.append("")

    # Entity difficulty
    lines.append("## 4. Entity Difficulty Ranking")
    lines.append("")
    lines.append(f"- Total unique entities: {difficulty_results['total_unique_entities']}")
    lines.append(f"- Always easy (>80% success): {difficulty_results['always_easy_count']}")
    lines.append(f"- Sometimes hard (30-80%): {difficulty_results['sometimes_hard_count']}")
    lines.append(f"- Always hard (<30%): {difficulty_results['always_hard_count']}")
    lines.append("")

    lines.append("### Hardest Entities (min 3 attempts)")
    lines.append("")
    lines.append("| Entity | Type | Success Rate | Attempts |")
    lines.append("|---|---|---|---|")
    for e in difficulty_results["hardest_entities"][:20]:
        lines.append(f"| {e['entity']} | {e['type']} | {100*e['success_rate']:.0f}% | {e['attempts']} |")
    lines.append("")

    lines.append("### Success Rate by Entity Type")
    lines.append("")
    lines.append("| Type | Matched | Total | Rate |")
    lines.append("|---|---|---|---|")
    for t, s in sorted(difficulty_results["by_entity_type"].items(), key=lambda x: x[1]["rate_pct"]):
        lines.append(f"| {t} | {s['matched']} | {s['total']} | {s['rate_pct']:.1f}% |")
    lines.append("")

    # Recommendations
    lines.append("## 5. Recommendations")
    lines.append("")
    for i, rec in enumerate(recommendations, 1):
        lines.append(f"{i}. {rec}")
        lines.append("")

    report_path = OUTPUT_DIR / "failure_analysis_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {report_path}")
    return str(report_path)


def write_json_report(cross_results, quality_results, fn_results, difficulty_results, recommendations):
    """Write structured JSON report."""
    report = {
        "cross_method_analysis": cross_results,
        "prediction_quality": quality_results,
        "false_negative_detection": fn_results,
        "entity_difficulty": difficulty_results,
        "recommendations": recommendations,
    }

    json_path = OUTPUT_DIR / "failure_analysis.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"JSON data written to {json_path}")
    return str(json_path)


# ─────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────

def main():
    print("COMPREHENSIVE FAILURE ANALYSIS")
    print("=" * 70)

    all_data = load_all_predictions()

    cross_results = cross_method_analysis(all_data)
    quality_results = prediction_quality_analysis(all_data)
    fn_results = false_negative_detection(all_data)
    difficulty_results = entity_difficulty_ranking(all_data)
    recommendations = generate_recommendations(cross_results, quality_results, fn_results, difficulty_results)

    write_markdown_report(cross_results, quality_results, fn_results, difficulty_results, recommendations)
    write_json_report(cross_results, quality_results, fn_results, difficulty_results, recommendations)

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
