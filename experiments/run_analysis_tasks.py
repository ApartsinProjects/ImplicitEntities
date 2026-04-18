"""
Tasks 5, 6, 9: Alias-aware evaluation, per-type/domain breakdown, statistical significance.
"""
import json
import os
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter

BASE = str(Path(__file__).resolve().parent)
RESULTS = f"{BASE}/results"
DATA = str(Path(__file__).resolve().parent.parent / "data" / "benchmark")

# ── Load data ──────────────────────────────────────────────────────────────

def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

entity_kb = load_json(f"{DATA}/entity_kb.json")
test_data = load_json(f"{DATA}/irc_bench_v5_test.json")

# Build uid -> metadata maps from test data
uid_to_type = {}
uid_to_domain = {}
for item in test_data:
    uid = item["uid"]
    uid_to_type[uid] = item.get("entity_type", "") or item.get("gold_type", "")
    ref = item.get("transcript_ref", "").replace("\\", "/")
    uid_to_domain[uid] = ref.split("/")[0] if ref else "unknown"

# Build alias lookup: entity_name_lower -> set of aliases (all lowered)
alias_map = {}  # gold_entity_lower -> set of all acceptable forms (lowered)
for ename, edata in entity_kb.items():
    key = ename.lower().strip()
    aliases = {key}
    for a in edata.get("aliases", []):
        aliases.add(a.lower().strip())
    alias_map[key] = aliases

OPEN_EXPERIMENTS = ["O1", "O2", "O3", "O4", "O5", "O6", "O7", "O8"]
CLOSED_EXPERIMENTS = ["C1", "C2", "C3"]
ALL_EXPERIMENTS = OPEN_EXPERIMENTS + CLOSED_EXPERIMENTS

# ── Task 5: Alias-Aware Evaluation ────────────────────────────────────────

def jaccard_tokens(a, b):
    ta = set(a.lower().split())
    tb = set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def match_tier(gold, pred):
    """Return matching tier (1-4) or 0 if no match."""
    if not pred or not gold:
        return 0
    g = gold.lower().strip()
    p = pred.lower().strip()
    # Tier 1: exact
    if g == p:
        return 1
    # Tier 2: alias match
    if g in alias_map:
        if p in alias_map[g]:
            return 2
    # Tier 3: containment
    if g in p or p in g:
        return 3
    # Tier 4: Jaccard >= 0.5
    if jaccard_tokens(g, p) >= 0.5:
        return 4
    return 0

def evaluate_experiment(exp_name):
    path = f"{RESULTS}/{exp_name}_predictions.json"
    data = load_json(path)
    is_closed = exp_name.startswith("C")

    results = {
        "experiment": exp_name,
        "total": len(data),
        "tier_counts": Counter(),
        "hit_at_1": 0,
    }
    if is_closed:
        results["hit_at_3"] = 0
        results["hit_at_5"] = 0
        results["hit_at_10"] = 0
        results["mrr_sum"] = 0.0

    per_item = []  # for significance tests later

    for item in data:
        gold = item["gold_entity"]
        pred = item.get("prediction", "")
        uid = item["uid"]

        tier = match_tier(gold, pred)
        results["tier_counts"][tier] += 1
        hit1 = 1 if tier > 0 else 0
        results["hit_at_1"] += hit1

        item_result = {"uid": uid, "hit1": hit1, "tier": tier}

        if is_closed and "top_10" in item:
            top10 = item.get("top_10", [])
            # Find first match in top_10
            first_rank = None
            for rank_idx, cand in enumerate(top10):
                if match_tier(gold, cand) > 0:
                    first_rank = rank_idx + 1
                    break
            if first_rank is not None:
                results["mrr_sum"] += 1.0 / first_rank
                if first_rank <= 3:
                    results["hit_at_3"] += 1
                if first_rank <= 5:
                    results["hit_at_5"] += 1
                if first_rank <= 10:
                    results["hit_at_10"] += 1
            item_result["first_rank"] = first_rank

        per_item.append(item_result)

    n = results["total"]
    summary = {
        "experiment": exp_name,
        "total": n,
        "hit_at_1": results["hit_at_1"],
        "hit_at_1_pct": round(100.0 * results["hit_at_1"] / n, 2) if n else 0,
        "tier_distribution": {str(k): v for k, v in sorted(results["tier_counts"].items())},
    }
    if is_closed:
        summary["hit_at_3"] = results.get("hit_at_3", 0)
        summary["hit_at_3_pct"] = round(100.0 * results["hit_at_3"] / n, 2) if n else 0
        summary["hit_at_5"] = results.get("hit_at_5", 0)
        summary["hit_at_5_pct"] = round(100.0 * results["hit_at_5"] / n, 2) if n else 0
        summary["hit_at_10"] = results.get("hit_at_10", 0)
        summary["hit_at_10_pct"] = round(100.0 * results["hit_at_10"] / n, 2) if n else 0
        summary["mrr"] = round(results["mrr_sum"] / n, 4) if n else 0

    return summary, per_item

print("=" * 70)
print("TASK 5: Alias-Aware Evaluation")
print("=" * 70)

all_eval = {}
all_per_item = {}  # exp -> list of per-item results

for exp in ALL_EXPERIMENTS:
    path = f"{RESULTS}/{exp}_predictions.json"
    if not os.path.exists(path):
        print(f"  SKIP {exp}: file not found")
        continue
    summary, per_item = evaluate_experiment(exp)
    all_eval[exp] = summary
    all_per_item[exp] = {r["uid"]: r for r in per_item}

    line = f"  {exp:4s}  Hit@1={summary['hit_at_1_pct']:6.2f}%"
    if "mrr" in summary:
        line += f"  Hit@3={summary['hit_at_3_pct']:.2f}%  Hit@5={summary['hit_at_5_pct']:.2f}%  Hit@10={summary['hit_at_10_pct']:.2f}%  MRR={summary['mrr']:.4f}"
    tiers = summary["tier_distribution"]
    line += f"  Tiers: " + ", ".join(f"T{k}={v}" for k, v in sorted(tiers.items()))
    print(line)

save_json(f"{RESULTS}/alias_eval_all.json", all_eval)
print(f"\nSaved: {RESULTS}/alias_eval_all.json")

# ── Task 6: Per-Type and Per-Domain Breakdown ─────────────────────────────

print("\n" + "=" * 70)
print("TASK 6: Per-Type and Per-Domain Breakdown")
print("=" * 70)

# Load all prediction files, compute per-item hit1
def compute_breakdowns():
    type_results = {}   # exp -> type -> {hit, total}
    domain_results = {} # exp -> domain -> {hit, total}

    for exp in ALL_EXPERIMENTS:
        path = f"{RESULTS}/{exp}_predictions.json"
        if not os.path.exists(path):
            continue
        data = load_json(path)

        type_results[exp] = defaultdict(lambda: {"hit": 0, "total": 0})
        domain_results[exp] = defaultdict(lambda: {"hit": 0, "total": 0})

        for item in data:
            uid = item["uid"]
            gold = item["gold_entity"]
            pred = item.get("prediction", "")
            hit = 1 if match_tier(gold, pred) > 0 else 0

            etype = item.get("gold_type", "") or uid_to_type.get(uid, "")
            if not etype:
                etype = "Unknown"
            domain = uid_to_domain.get(uid, "unknown")

            type_results[exp][etype]["hit"] += hit
            type_results[exp][etype]["total"] += 1
            domain_results[exp][domain]["hit"] += hit
            domain_results[exp][domain]["total"] += 1

    return type_results, domain_results

type_results, domain_results = compute_breakdowns()

# Convert to serializable and compute percentages
def make_breakdown_json(results):
    out = {}
    for exp, groups in results.items():
        out[exp] = {}
        for grp, vals in groups.items():
            h, t = vals["hit"], vals["total"]
            out[exp][grp] = {
                "hit_at_1": h,
                "total": t,
                "hit_at_1_pct": round(100.0 * h / t, 2) if t else 0
            }
    return out

type_json = make_breakdown_json(type_results)
domain_json = make_breakdown_json(domain_results)

save_json(f"{RESULTS}/type_breakdown.json", type_json)
save_json(f"{RESULTS}/domain_breakdown.json", domain_json)

# Print type table
all_types = sorted(set(t for exp_data in type_json.values() for t in exp_data))
print("\n--- Hit@1 (%) by Entity Type ---")
header = f"{'Exp':6s}" + "".join(f"{t:>14s}" for t in all_types)
print(header)
print("-" * len(header))
for exp in ALL_EXPERIMENTS:
    if exp not in type_json:
        continue
    row = f"{exp:6s}"
    for t in all_types:
        if t in type_json[exp]:
            row += f"{type_json[exp][t]['hit_at_1_pct']:13.2f}%"
        else:
            row += f"{'N/A':>14s}"
    print(row)

# Print domain table
all_domains = sorted(set(d for exp_data in domain_json.values() for d in exp_data))
print("\n--- Hit@1 (%) by Domain ---")
header = f"{'Exp':6s}" + "".join(f"{d:>16s}" for d in all_domains)
print(header)
print("-" * len(header))
for exp in ALL_EXPERIMENTS:
    if exp not in domain_json:
        continue
    row = f"{exp:6s}"
    for d in all_domains:
        if d in domain_json[exp]:
            row += f"{domain_json[exp][d]['hit_at_1_pct']:15.2f}%"
        else:
            row += f"{'N/A':>16s}"
    print(row)

print(f"\nSaved: {RESULTS}/type_breakdown.json")
print(f"Saved: {RESULTS}/domain_breakdown.json")

# ── Task 9: Statistical Significance ──────────────────────────────────────

print("\n" + "=" * 70)
print("TASK 9: Statistical Significance")
print("=" * 70)

from scipy.stats import chi2_contingency

COMPARISONS = [
    ("O1", "O2", "ZS vs FS (GPT-4o)"),
    ("O3", "O4", "ZS vs FS (mini)"),
    ("O1", "O5", "GPT-4o vs Llama 8B"),
    ("O1", "C2", "Open vs Closed world"),
    ("O5", "O7", "8B vs 1B"),
]

def mcnemar_test(hits_a, hits_b):
    """McNemar's test using chi2_contingency on the 2x2 contingency table."""
    # b = A correct, B wrong; c = A wrong, B correct
    b = sum(1 for a, bb in zip(hits_a, hits_b) if a == 1 and bb == 0)
    c = sum(1 for a, bb in zip(hits_a, hits_b) if a == 0 and bb == 1)
    # Also compute: both correct, both wrong
    both_correct = sum(1 for a, bb in zip(hits_a, hits_b) if a == 1 and bb == 1)
    both_wrong = sum(1 for a, bb in zip(hits_a, hits_b) if a == 0 and bb == 0)
    table = np.array([[both_correct, b], [c, both_wrong]])
    # McNemar focuses on discordant pairs; use chi2_contingency on 2x2
    # Standard McNemar: chi2 = (b - c)^2 / (b + c)
    if b + c == 0:
        return {"chi2": 0.0, "p_value": 1.0, "b": b, "c": c}
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)  # with continuity correction
    from scipy.stats import chi2 as chi2_dist
    p_value = 1 - chi2_dist.cdf(chi2, df=1)
    return {"chi2": round(float(chi2), 4), "p_value": round(float(p_value), 6), "b_discordant": b, "c_discordant": c}

def bootstrap_ci(hits, n_boot=1000, ci=0.95, seed=42):
    rng = np.random.RandomState(seed)
    hits = np.array(hits)
    n = len(hits)
    means = []
    for _ in range(n_boot):
        sample = rng.choice(hits, size=n, replace=True)
        means.append(sample.mean())
    means = sorted(means)
    lo = (1 - ci) / 2
    hi = 1 - lo
    return {
        "mean": round(float(np.mean(means)), 4),
        "ci_lower": round(float(means[int(lo * n_boot)]), 4),
        "ci_upper": round(float(means[int(hi * n_boot)]), 4),
    }

significance_results = []

for exp_a, exp_b, label in COMPARISONS:
    if exp_a not in all_per_item or exp_b not in all_per_item:
        print(f"  SKIP {label}: missing data")
        continue

    items_a = all_per_item[exp_a]
    items_b = all_per_item[exp_b]

    # Find common UIDs
    common_uids = sorted(set(items_a.keys()) & set(items_b.keys()))
    n = len(common_uids)

    hits_a = [items_a[u]["hit1"] for u in common_uids]
    hits_b = [items_b[u]["hit1"] for u in common_uids]

    acc_a = sum(hits_a) / n if n else 0
    acc_b = sum(hits_b) / n if n else 0

    mcn = mcnemar_test(hits_a, hits_b)
    boot_a = bootstrap_ci(hits_a)
    boot_b = bootstrap_ci(hits_b)

    result = {
        "comparison": label,
        "exp_a": exp_a,
        "exp_b": exp_b,
        "n_common": n,
        "acc_a": round(acc_a, 4),
        "acc_b": round(acc_b, 4),
        "mcnemar": mcn,
        "bootstrap_a": boot_a,
        "bootstrap_b": boot_b,
    }
    significance_results.append(result)

    sig_marker = "*" if mcn["p_value"] < 0.05 else ""
    print(f"  {label}")
    print(f"    {exp_a}: {acc_a*100:.2f}% [{boot_a['ci_lower']*100:.2f}, {boot_a['ci_upper']*100:.2f}]")
    print(f"    {exp_b}: {acc_b*100:.2f}% [{boot_b['ci_lower']*100:.2f}, {boot_b['ci_upper']*100:.2f}]")
    print(f"    McNemar chi2={mcn['chi2']:.4f}, p={mcn['p_value']:.6f} {sig_marker}")
    print(f"    Discordant: {exp_a}-only={mcn['b_discordant']}, {exp_b}-only={mcn['c_discordant']}")
    print()

save_json(f"{RESULTS}/significance_tests.json", significance_results)
print(f"Saved: {RESULTS}/significance_tests.json")
print("\nDone.")
