"""
Retrieve results from 4 completed OpenAI batch jobs for IRC-Bench v5 Phase B:
  O11: GPT-4.1-mini CoT
  O12: GPT-4o CoT
  RAG1: BGE top-5 + GPT-4.1-mini
  EA: Error Analysis (O1..O6)
"""
import json
import re
import sys
from pathlib import Path
from collections import Counter, defaultdict

# Setup paths
V5_DIR = Path(__file__).resolve().parent
RESULTS_DIR = V5_DIR / "results"
DATA_DIR = V5_DIR.parent / "data" / "benchmark"

from openai_batch_client import OpenAIBatchClient

# Load batch IDs
phase_b = json.loads((RESULTS_DIR / "phase_b_batches.json").read_text(encoding="utf-8"))
BATCH_IDS = phase_b["batches"]

# Load test data for gold labels
print("Loading test data...")
test_data = json.loads((DATA_DIR / "irc_bench_v5_test.json").read_text(encoding="utf-8"))
test_by_uid = {s["uid"]: s for s in test_data}
print(f"  Loaded {len(test_data)} test samples")

# Load file maps
o11_map = json.loads((V5_DIR / "batches" / "v5_O11_map_20260418_081524.json").read_text(encoding="utf-8"))
o12_map = json.loads((V5_DIR / "batches" / "v5_O12_map_20260418_081524.json").read_text(encoding="utf-8"))
rag1_map = json.loads((V5_DIR / "batches" / "v5_RAG1_map_20260418_081532.json").read_text(encoding="utf-8"))
ea_map = json.loads((V5_DIR / "batches" / "v5_EA_map_20260418_081535.json").read_text(encoding="utf-8"))

# Invert uid->index maps to index->uid
o11_idx_to_uid = {v: k for k, v in o11_map.items()}
o12_idx_to_uid = {v: k for k, v in o12_map.items()}
rag1_idx_to_uid = {v: k for k, v in rag1_map.items()}

client = OpenAIBatchClient()


def download_batch_results(batch_id: str, label: str) -> list[str | None]:
    """Download and parse results from a completed batch."""
    print(f"\n{'=' * 60}")
    print(f"Retrieving {label} (batch: {batch_id})")
    print("=" * 60)

    status = client.check_status(batch_id)
    print(f"  Status: {status['status']}")
    counts = status["request_counts"]
    print(f"  Completed: {counts['completed']}/{counts['total']}, Failed: {counts['failed']}")

    if status["status"] != "completed":
        print(f"  ERROR: Batch not completed (status={status['status']})")
        return []

    # Download output file
    output_file_id = status["output_file_id"]
    output_path = V5_DIR / "batches" / f"output_{label}_{batch_id}.jsonl"

    if output_path.exists():
        print(f"  Using cached output: {output_path}")
    else:
        print(f"  Downloading output file {output_file_id}...")
        content = client.client.files.content(output_file_id).content
        with open(output_path, "wb") as f:
            f.write(content)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"  Downloaded {size_mb:.2f} MB")

    # Parse JSONL
    return client._parse_output(output_path)


def extract_entity_from_cot(response: str | None) -> str:
    """Extract entity name from CoT response (last line after 'Entity:')."""
    if not response:
        return ""
    # Look for "Entity:" line
    for line in reversed(response.strip().split("\n")):
        line = line.strip()
        if line.lower().startswith("entity:"):
            entity = line.split(":", 1)[1].strip()
            # Remove surrounding quotes or brackets
            entity = entity.strip("\"'[]")
            return entity
    # Fallback: last non-empty line
    lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
    if lines:
        last = lines[-1]
        last = last.strip("\"'[]")
        return last
    return ""


def extract_entity_from_rag(response: str | None) -> str:
    """Extract entity name from RAG response (format: 'Answer: N. Entity Name')."""
    if not response:
        return ""
    text = response.strip()
    # Strip leading "Answer:" prefix if present
    if text.lower().startswith("answer:"):
        text = text[len("answer:"):].strip()
    # Try matching "N. Entity Name" pattern
    m = re.match(r'^\d+\.\s*(.+)', text)
    if m:
        entity = m.group(1).strip()
        # Remove trailing description after " - "
        if " - " in entity:
            entity = entity.split(" - ")[0].strip()
        # Remove trailing parenthetical like "(Organization)"
        entity = re.sub(r'\s*\([^)]*\)\s*$', '', entity)
        # Remove trailing punctuation
        entity = entity.rstrip(".")
        return entity
    # Fallback: entire response cleaned up
    return text.strip("\"'[]")


def exact_match(gold: str, pred: str) -> bool:
    """Check exact match (case-insensitive)."""
    return gold.strip().lower() == pred.strip().lower()


def alias_match(gold_entity: str, pred: str, sample: dict) -> bool:
    """Check if prediction matches gold entity or any alias."""
    pred_lower = pred.strip().lower()
    if not pred_lower:
        return False
    if gold_entity.strip().lower() == pred_lower:
        return True
    aliases_str = sample.get("entity_aliases", "")
    if aliases_str:
        for alias in aliases_str.split("|"):
            if alias.strip().lower() == pred_lower:
                return True
    return False


def compute_accuracy(predictions: list[dict], label: str) -> dict:
    """Compute exact match and alias match accuracy."""
    n = len(predictions)
    exact = sum(1 for p in predictions if exact_match(p["gold_entity"], p["prediction"]))
    alias = sum(1 for p in predictions if alias_match(
        p["gold_entity"], p["prediction"], test_by_uid.get(p["uid"], {})
    ))
    metrics = {
        "exp_id": label,
        "n_test": n,
        "exact_match": round(exact / n, 4) if n else 0,
        "alias_match": round(alias / n, 4) if n else 0,
        "exact_correct": exact,
        "alias_correct": alias,
    }
    print(f"  {label}: exact_match={metrics['exact_match']:.4f} ({exact}/{n}), "
          f"alias_match={metrics['alias_match']:.4f} ({alias}/{n})")
    return metrics


# ============================================================
# O11: GPT-4.1-mini CoT
# ============================================================

o11_results = download_batch_results(BATCH_IDS["O11"], "O11")

o11_predictions = []
for idx, response in enumerate(o11_results):
    uid = o11_idx_to_uid.get(idx)
    if uid is None:
        continue
    sample = test_by_uid.get(uid, {})
    entity = extract_entity_from_cot(response)
    o11_predictions.append({
        "uid": uid,
        "gold_entity": sample.get("entity", ""),
        "gold_type": sample.get("entity_type", ""),
        "gold_qid": sample.get("entity_qid", ""),
        "prediction": entity,
        "raw_response": response or "",
        "implicit_text": sample.get("implicit_text", ""),
        "explicit_text": sample.get("explicit_text", ""),
    })

(RESULTS_DIR / "O11_predictions.json").write_text(
    json.dumps(o11_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
)
o11_metrics = compute_accuracy(o11_predictions, "O11")
(RESULTS_DIR / "O11_metrics.json").write_text(
    json.dumps(o11_metrics, indent=2), encoding="utf-8"
)


# ============================================================
# O12: GPT-4o CoT
# ============================================================

o12_results = download_batch_results(BATCH_IDS["O12"], "O12")

o12_predictions = []
for idx, response in enumerate(o12_results):
    uid = o12_idx_to_uid.get(idx)
    if uid is None:
        continue
    sample = test_by_uid.get(uid, {})
    entity = extract_entity_from_cot(response)
    o12_predictions.append({
        "uid": uid,
        "gold_entity": sample.get("entity", ""),
        "gold_type": sample.get("entity_type", ""),
        "gold_qid": sample.get("entity_qid", ""),
        "prediction": entity,
        "raw_response": response or "",
        "implicit_text": sample.get("implicit_text", ""),
        "explicit_text": sample.get("explicit_text", ""),
    })

(RESULTS_DIR / "O12_predictions.json").write_text(
    json.dumps(o12_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
)
o12_metrics = compute_accuracy(o12_predictions, "O12")
(RESULTS_DIR / "O12_metrics.json").write_text(
    json.dumps(o12_metrics, indent=2), encoding="utf-8"
)


# ============================================================
# RAG1: BGE top-5 + GPT-4.1-mini
# ============================================================

rag1_results = download_batch_results(BATCH_IDS["RAG1"], "RAG1")

rag1_predictions = []
for idx, response in enumerate(rag1_results):
    uid = rag1_idx_to_uid.get(idx)
    if uid is None:
        continue
    sample = test_by_uid.get(uid, {})
    entity = extract_entity_from_rag(response)
    rag1_predictions.append({
        "uid": uid,
        "gold_entity": sample.get("entity", ""),
        "gold_type": sample.get("entity_type", ""),
        "gold_qid": sample.get("entity_qid", ""),
        "prediction": entity,
        "raw_response": response or "",
        "implicit_text": sample.get("implicit_text", ""),
        "explicit_text": sample.get("explicit_text", ""),
    })

(RESULTS_DIR / "RAG1_predictions.json").write_text(
    json.dumps(rag1_predictions, indent=2, ensure_ascii=False), encoding="utf-8"
)
rag1_metrics = compute_accuracy(rag1_predictions, "RAG1")
(RESULTS_DIR / "RAG1_metrics.json").write_text(
    json.dumps(rag1_metrics, indent=2), encoding="utf-8"
)


# ============================================================
# EA: Error Analysis
# ============================================================

ea_results = download_batch_results(BATCH_IDS["EA"], "EA")

VALID_ERROR_TYPES = {
    "SAME_TYPE_RELATED", "SAME_TYPE_UNRELATED", "WRONG_TYPE",
    "PARTIAL", "HALLUCINATION", "EMPTY", "AMBIGUOUS",
}

# Aggregate by experiment and error type
ea_by_exp = defaultdict(lambda: Counter())
ea_records = []

for idx, response in enumerate(ea_results):
    if idx >= len(ea_map):
        break
    meta = ea_map[idx]
    exp_id = meta["exp_id"]

    # Parse error type from response
    error_type = (response or "").strip().upper().replace(" ", "_")
    # Normalize: try to match to valid types
    if error_type not in VALID_ERROR_TYPES:
        # Try partial match
        matched = False
        for valid in VALID_ERROR_TYPES:
            if valid in error_type or error_type in valid:
                error_type = valid
                matched = True
                break
        if not matched:
            error_type = "UNKNOWN"

    ea_by_exp[exp_id][error_type] += 1
    ea_records.append({
        "exp_id": exp_id,
        "uid": meta["uid"],
        "gold_entity": meta["gold_entity"],
        "gold_type": meta["gold_type"],
        "prediction": meta["prediction"],
        "error_type": error_type,
        "raw_response": response or "",
    })

# Build summary
ea_summary = {}
for exp_id in sorted(ea_by_exp.keys()):
    counts = ea_by_exp[exp_id]
    total = sum(counts.values())
    ea_summary[exp_id] = {
        "total_analyzed": total,
        "counts": dict(sorted(counts.items())),
        "percentages": {k: round(v / total * 100, 1) for k, v in sorted(counts.items())},
    }

ea_output = {
    "summary_by_experiment": ea_summary,
    "records": ea_records,
}

(RESULTS_DIR / "error_analysis.json").write_text(
    json.dumps(ea_output, indent=2, ensure_ascii=False), encoding="utf-8"
)


# ============================================================
# Print summary
# ============================================================

print("\n" + "=" * 60)
print("SUMMARY: Exact Match Accuracies")
print("=" * 60)
print(f"  O11 (GPT-4.1-mini CoT):    {o11_metrics['exact_match']:.4f}  (alias: {o11_metrics['alias_match']:.4f})")
print(f"  O12 (GPT-4o CoT):          {o12_metrics['exact_match']:.4f}  (alias: {o12_metrics['alias_match']:.4f})")
print(f"  RAG1 (BGE+GPT-4.1-mini):   {rag1_metrics['exact_match']:.4f}  (alias: {rag1_metrics['alias_match']:.4f})")

print("\n" + "=" * 60)
print("ERROR ANALYSIS SUMMARY")
print("=" * 60)
# Header
all_types = sorted(set(t for exp_counts in ea_by_exp.values() for t in exp_counts))
header = f"{'Exp':<6}" + "".join(f"{t:<22}" for t in all_types) + "Total"
print(header)
print("-" * len(header))
for exp_id in sorted(ea_by_exp.keys()):
    counts = ea_by_exp[exp_id]
    total = sum(counts.values())
    row = f"{exp_id:<6}"
    for t in all_types:
        c = counts.get(t, 0)
        pct = c / total * 100 if total else 0
        row += f"{c:>3} ({pct:4.1f}%)          "
    row += f"{total:>4}"
    print(row)

print("\nDone! All results saved to experiments/v5/results/")
