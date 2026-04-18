"""
Phase B batch submissions for IRC-Bench v5:
  Experiment 4: Chain-of-Thought (O11, O12)
  Experiment 8: RAG Baseline (RAG1)
  Experiment 3: Error Analysis (EA_O1..EA_O6)
"""
import json
import random
import sys
from pathlib import Path
from datetime import datetime

# Setup paths
V5_DIR = Path(__file__).resolve().parent
RESULTS_DIR = V5_DIR / "results"
DATA_DIR = V5_DIR.parent / "data" / "benchmark"

from openai_batch_client import OpenAIBatchClient

# Load test data
print("Loading test data...")
test_data = json.loads((DATA_DIR / "irc_bench_v5_test.json").read_text(encoding="utf-8"))
print(f"  Loaded {len(test_data)} test samples")

client = OpenAIBatchClient()
all_batch_ids = {}


# ============================================================
# Experiment 4: Chain-of-Thought Baseline (O11, O12)
# ============================================================

COT_SYSTEM = "You are an entity recognition expert. Think step by step."

def make_cot_prompt(implicit_text: str) -> list[dict]:
    return [
        {"role": "system", "content": COT_SYSTEM},
        {"role": "user", "content": (
            'What named entity is implicitly referenced in this text? '
            'The entity is never mentioned by name.\n\n'
            f'Text: "{implicit_text}"\n\n'
            'Think step by step:\n'
            '1. What contextual cues are present? (dates, places, events, people, roles)\n'
            '2. What type of entity do these cues suggest? (Person, Place, Organization, Event)\n'
            '3. What specific named entity matches ALL these cues?\n\n'
            'Reasoning: [your step-by-step analysis]\n'
            'Entity: [canonical Wikipedia name]'
        )}
    ]


print("\n" + "=" * 60)
print("EXPERIMENT 4: Chain-of-Thought Baseline")
print("=" * 60)

cot_prompts = [make_cot_prompt(s["implicit_text"]) for s in test_data]

# O11: GPT-4.1-mini CoT
print("\n--- O11: GPT-4.1-mini CoT ---")
o11_batch_id = client.submit_batch(
    prompts=cot_prompts,
    model="gpt-4.1-mini",
    temperature=0.7,
    max_tokens=300,
    job_name="v5_O11",
)
all_batch_ids["O11"] = o11_batch_id

# Save O11 file map (uid -> index)
o11_map = {s["uid"]: i for i, s in enumerate(test_data)}
map_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
o11_map_path = V5_DIR / "batches" / f"v5_O11_map_{map_ts}.json"
o11_map_path.write_text(json.dumps(o11_map, indent=2), encoding="utf-8")

# Save batch info
o11_info = {
    "exp_id": "O11",
    "batch_id": o11_batch_id,
    "model": "gpt-4.1-mini",
    "mode": "cot",
    "n_samples": len(test_data),
    "map_path": str(o11_map_path),
    "timestamp": map_ts,
}
(RESULTS_DIR / "O11_batch_info.json").write_text(json.dumps(o11_info, indent=2), encoding="utf-8")
print(f"  O11 batch ID: {o11_batch_id}")

# O12: GPT-4o CoT
print("\n--- O12: GPT-4o CoT ---")
o12_batch_id = client.submit_batch(
    prompts=cot_prompts,
    model="gpt-4o",
    temperature=0.7,
    max_tokens=300,
    job_name="v5_O12",
)
all_batch_ids["O12"] = o12_batch_id

o12_map_path = V5_DIR / "batches" / f"v5_O12_map_{map_ts}.json"
o12_map_path.write_text(json.dumps(o11_map, indent=2), encoding="utf-8")

o12_info = {
    "exp_id": "O12",
    "batch_id": o12_batch_id,
    "model": "gpt-4o",
    "mode": "cot",
    "n_samples": len(test_data),
    "map_path": str(o12_map_path),
    "timestamp": map_ts,
}
(RESULTS_DIR / "O12_batch_info.json").write_text(json.dumps(o12_info, indent=2), encoding="utf-8")
print(f"  O12 batch ID: {o12_batch_id}")


# ============================================================
# Experiment 8: RAG Baseline (RAG1)
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 8: RAG Baseline (BGE top-5 + GPT-4.1-mini)")
print("=" * 60)

# Load C2 predictions for top-5 candidates
print("Loading C2 predictions...")
c2_preds = json.loads((RESULTS_DIR / "C2_predictions.json").read_text(encoding="utf-8"))
c2_by_uid = {p["uid"]: p for p in c2_preds}
print(f"  Loaded {len(c2_preds)} C2 predictions")

# Load entity KB for descriptions
print("Loading entity KB...")
entity_kb = json.loads((DATA_DIR / "entity_kb.json").read_text(encoding="utf-8"))
print(f"  Loaded {len(entity_kb)} entities in KB")

def get_entity_description(name: str) -> str:
    """Look up entity description from KB, fallback to empty string."""
    if name in entity_kb:
        return entity_kb[name].get("description", "")
    # Try case-insensitive match
    for k, v in entity_kb.items():
        if k.lower() == name.lower():
            return v.get("description", "")
    return ""

def make_rag_prompt(sample: dict, c2_pred: dict) -> list[dict]:
    implicit_text = sample["implicit_text"]
    top5 = c2_pred.get("top_10", [])[:5]

    # Build candidate list with descriptions
    candidates_lines = []
    for i, name in enumerate(top5, 1):
        desc = get_entity_description(name)
        if desc:
            candidates_lines.append(f"{i}. {name} - {desc}")
        else:
            candidates_lines.append(f"{i}. {name}")

    # Pad if fewer than 5 candidates
    while len(candidates_lines) < 5:
        candidates_lines.append(f"{len(candidates_lines)+1}. [no candidate]")

    candidates_str = "\n".join(candidates_lines)

    return [
        {"role": "user", "content": (
            'This text implicitly references a named entity without naming it. '
            'Based on the contextual cues, which candidate is most likely?\n\n'
            f'Text: "{implicit_text}"\n\n'
            f'Candidates:\n{candidates_str}\n\n'
            'If none match well, suggest a better entity.\n'
            'Answer: [number]. [entity name]'
        )}
    ]


rag_prompts = []
rag_uids = []
skipped = 0
for sample in test_data:
    uid = sample["uid"]
    if uid in c2_by_uid:
        rag_prompts.append(make_rag_prompt(sample, c2_by_uid[uid]))
        rag_uids.append(uid)
    else:
        skipped += 1

if skipped > 0:
    print(f"  WARNING: Skipped {skipped} samples without C2 predictions")

print(f"  Prepared {len(rag_prompts)} RAG prompts")

rag_batch_id = client.submit_batch(
    prompts=rag_prompts,
    model="gpt-4.1-mini",
    temperature=0.7,
    max_tokens=50,
    job_name="v5_RAG1",
)
all_batch_ids["RAG1"] = rag_batch_id

rag_map = {uid: i for i, uid in enumerate(rag_uids)}
rag_map_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
rag_map_path = V5_DIR / "batches" / f"v5_RAG1_map_{rag_map_ts}.json"
rag_map_path.write_text(json.dumps(rag_map, indent=2), encoding="utf-8")

rag_info = {
    "exp_id": "RAG1",
    "batch_id": rag_batch_id,
    "model": "gpt-4.1-mini",
    "mode": "rag_bge_top5",
    "n_samples": len(rag_prompts),
    "map_path": str(rag_map_path),
    "timestamp": rag_map_ts,
}
(RESULTS_DIR / "RAG1_batch_info.json").write_text(json.dumps(rag_info, indent=2), encoding="utf-8")
print(f"  RAG1 batch ID: {rag_batch_id}")


# ============================================================
# Experiment 3: Error Analysis (EA_O1..EA_O6)
# ============================================================

print("\n" + "=" * 60)
print("EXPERIMENT 3: Error Analysis")
print("=" * 60)

random.seed(42)

def make_error_analysis_prompt(gold_entity: str, gold_type: str,
                                prediction: str, implicit_text: str) -> list[dict]:
    return [
        {"role": "user", "content": (
            'Classify this entity recognition error.\n\n'
            f'Gold entity: "{gold_entity}" ({gold_type})\n'
            f'Predicted entity: "{prediction}"\n'
            f'Text: "{implicit_text}"\n\n'
            'Error type (pick ONE):\n'
            '- SAME_TYPE_RELATED: correct type, related but wrong entity (e.g., Okinawa vs Iwo Jima)\n'
            '- SAME_TYPE_UNRELATED: correct type, completely different entity\n'
            '- WRONG_TYPE: predicted a different entity type\n'
            '- PARTIAL: prediction contains or is contained in the gold (partial name match)\n'
            '- HALLUCINATION: predicted entity doesn\'t exist or is nonsensical\n'
            '- EMPTY: no meaningful prediction\n'
            '- AMBIGUOUS: prediction could be considered correct with loose matching\n\n'
            'Output ONLY the error type, nothing else.'
        )}
    ]


def is_correct(pred: dict) -> bool:
    """Check if prediction matches gold (exact or alias match)."""
    gold = pred.get("gold_entity", "").strip().lower()
    prediction = pred.get("prediction", "").strip().lower()
    if not prediction or not gold:
        return False
    if prediction == gold:
        return True
    # Check if prediction is in gold or gold is in prediction
    if gold in prediction or prediction in gold:
        return True
    return False


exp_ids = ["O1", "O2", "O3", "O4", "O5", "O6"]
ea_prompts_all = []
ea_metadata_all = []

for exp_id in exp_ids:
    pred_path = RESULTS_DIR / f"{exp_id}_predictions.json"
    if not pred_path.exists():
        print(f"  WARNING: {pred_path} not found, skipping")
        continue

    preds = json.loads(pred_path.read_text(encoding="utf-8"))
    print(f"\n  {exp_id}: {len(preds)} predictions total")

    # Filter incorrect predictions
    incorrect = [p for p in preds if not is_correct(p)]
    print(f"  {exp_id}: {len(incorrect)} incorrect predictions")

    # Sample 200 (or fewer if not enough)
    n_sample = min(200, len(incorrect))
    sampled = random.sample(incorrect, n_sample)
    print(f"  {exp_id}: sampled {n_sample} for error analysis")

    ea_prompts = []
    ea_meta = []
    for p in sampled:
        prompt = make_error_analysis_prompt(
            p["gold_entity"], p["gold_type"],
            p.get("prediction", ""), p["implicit_text"]
        )
        ea_prompts.append(prompt)
        ea_meta.append({
            "exp_id": exp_id,
            "uid": p["uid"],
            "gold_entity": p["gold_entity"],
            "gold_type": p["gold_type"],
            "prediction": p.get("prediction", ""),
        })

    ea_prompts_all.extend(ea_prompts)
    ea_metadata_all.extend(ea_meta)

print(f"\n  Total error analysis prompts: {len(ea_prompts_all)}")

# Submit as one batch (all experiments combined)
ea_batch_id = client.submit_batch(
    prompts=ea_prompts_all,
    model="gpt-4.1-mini",
    temperature=0.0,
    max_tokens=20,
    job_name="v5_EA",
)
all_batch_ids["EA"] = ea_batch_id

# Save metadata (maps prompt index to experiment + uid)
ea_map_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
ea_map_path = V5_DIR / "batches" / f"v5_EA_map_{ea_map_ts}.json"
ea_map_path.write_text(json.dumps(ea_metadata_all, indent=2, ensure_ascii=False), encoding="utf-8")

ea_info = {
    "exp_id": "EA",
    "batch_id": ea_batch_id,
    "model": "gpt-4.1-mini",
    "mode": "error_analysis",
    "n_samples": len(ea_prompts_all),
    "experiments_analyzed": exp_ids,
    "samples_per_experiment": 200,
    "map_path": str(ea_map_path),
    "timestamp": ea_map_ts,
}
(RESULTS_DIR / "EA_batch_info.json").write_text(json.dumps(ea_info, indent=2), encoding="utf-8")
print(f"  EA batch ID: {ea_batch_id}")


# ============================================================
# Save all batch IDs to phase_b_batches.json
# ============================================================

print("\n" + "=" * 60)
print("ALL BATCH IDS")
print("=" * 60)

phase_b = {
    "submitted_at": datetime.now().isoformat(),
    "batches": all_batch_ids,
    "details": {
        "O11": {"model": "gpt-4.1-mini", "mode": "cot", "n_samples": len(test_data)},
        "O12": {"model": "gpt-4o", "mode": "cot", "n_samples": len(test_data)},
        "RAG1": {"model": "gpt-4.1-mini", "mode": "rag_bge_top5", "n_samples": len(rag_prompts)},
        "EA": {"model": "gpt-4.1-mini", "mode": "error_analysis", "n_samples": len(ea_prompts_all)},
    }
}

(RESULTS_DIR / "phase_b_batches.json").write_text(
    json.dumps(phase_b, indent=2), encoding="utf-8"
)

for name, bid in all_batch_ids.items():
    print(f"  {name}: {bid}")

print(f"\nSaved to {RESULTS_DIR / 'phase_b_batches.json'}")
print("Done!")
