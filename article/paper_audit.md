# Paper Audit Report: article_v4.html

Audited: 2026-04-18
Source: `article/article_v4.html`

---

## 1. Dataset Claims

| Claim | Source File | Paper Value | Actual Value | Status |
|-------|-----------|-------------|--------------|--------|
| Total samples | `split_metadata.json` | 25,136 | 25,136 | MATCH |
| Total entities | `split_metadata.json` | 12,337 | 12,337 | MATCH |
| Train samples | `split_metadata.json` | 17,971 | 17,971 | MATCH |
| Dev samples | `split_metadata.json` | 2,532 | 2,532 | MATCH |
| Test samples | `split_metadata.json` | 4,633 | 4,633 | MATCH |
| Train entities (Table 2) | `split_metadata.json` | 8,635 | 8,635 | MATCH |
| Dev entities (Table 2) | `split_metadata.json` | 1,234 | 1,234 | MATCH |
| Test entities (Table 2) | `split_metadata.json` | 2,468 | 2,468 | MATCH |
| Train+Dev+Test = Total | computed | 25,136 | 25,136 | MATCH |
| Entity overlap | `split_metadata.json` | 0 | 0 | MATCH |
| Split ratio | `split_metadata.json` | 70/10/20 | 70/10/20 | MATCH |
| Seed | `split_metadata.json` | 42 | 42 | MATCH |
| entity_kb.json count | `entity_kb.json` | 12,337 | 12,337 | MATCH |
| Transcript count | file count | 1,994 | 1,994 | MATCH |
| Entities extracted files | file count | 1,752 | 1,752 | MATCH |
| Summary files | file count | 1,601 | 1,601 | MATCH |
| Implicit rewrite files | file count | 1,600 | 1,600 | MATCH |
| Entities coverage (87.9%) | computed | 87.9% | 87.9% | MATCH |
| Summaries coverage (80.3%) | computed | 80.3% | 80.3% | MATCH |
| Implicit coverage (80.2%) | computed | 80.2% | 80.2% | MATCH |

### Entity Type Distribution (Table 3)

| Type | Paper Samples | Actual | Paper % | Actual % | Status |
|------|-------------|--------|---------|----------|--------|
| Place | 11,893 | 11,893 | 47.3% | 47.3% | MATCH |
| Organization | 5,366 | 5,366 | 21.3% | 21.3% | MATCH |
| Person | 3,450 | 3,450 | 13.7% | 13.7% | MATCH |
| Event | 2,162 | 2,162 | 8.6% | 8.6% | MATCH |
| Work | 1,195 | 1,195 | 4.8% | 4.8% | MATCH |
| Military Unit | 537 | 537 | 2.1% | 2.1% | MATCH |
| Other | 533 | 533 | 2.1% | 2.1% | MATCH |
| **Sum** | 25,136 | 25,136 | ~100% | 99.9% | MATCH (rounding) |

---

## 2. Open-World Results (Table 4)

### O1 through O6, O10, RAG1

| ID | Metric | Paper | Actual (JSON) | Status |
|----|--------|-------|---------------|--------|
| O1 | Exact | 27.02% | 27.02% | MATCH |
| O1 | Alias | 33.30% | 33.30% | MATCH |
| O1 | Contain | 33.30% | 33.30% | MATCH |
| O1 | Jaccard | 35.05% | 35.05% | MATCH |
| O2 | Exact | 31.62% | 31.62% | MATCH |
| O2 | Alias | 38.94% | 38.94% | MATCH |
| O2 | Contain | 38.94% | 38.94% | MATCH |
| O2 | Jaccard | 41.10% | 41.10% | MATCH |
| O3 | Exact | 25.71% | 25.71% | MATCH |
| O3 | Alias | 27.09% | 27.09% | MATCH |
| O3 | Contain | 33.50% | 33.50% | MATCH |
| O3 | Jaccard | 35.94% | 35.94% | MATCH |
| O4 | Exact | 28.66% | 28.66% | MATCH |
| O4 | Alias | 36.89% | 36.89% | MATCH |
| O4 | Contain | 36.89% | 36.89% | MATCH |
| O4 | Jaccard | 39.48% | 39.48% | MATCH |
| O5 | Exact | 13.92% | 13.92% | MATCH |
| O5 | Alias | 14.81% | 14.81% | MATCH |
| O5 | Contain | 19.47% | 19.47% | MATCH |
| O5 | Jaccard | 20.18% | 20.18% | MATCH |
| O6 | Exact | 17.83% | 17.83% | MATCH |
| O6 | Alias | 18.80% | 18.80% | MATCH |
| O6 | Contain | 24.61% | 24.61% | MATCH |
| O6 | Jaccard | 25.66% | 25.66% | MATCH |
| O7 | Jaccard* | 17.27% | 17.27% | MATCH |
| O8 | Jaccard* | 15.22% | 15.22% | MATCH |
| O10 | Exact | 38.94% | 38.94% | MATCH |
| O10 | Alias | 41.42% | 41.42% | MATCH |
| O10 | Contain | 47.90% | 47.90% | MATCH |
| O10 | Jaccard | 51.59% | 51.59% | MATCH |
| RAG1 | Exact | 19.71% | 19.71% | MATCH |
| RAG1 | Alias | 20.53% | 20.53% | MATCH |
| RAG1 | Contain | 28.75% | 28.75% | MATCH |
| RAG1 | Jaccard | 29.55% | 29.55% | MATCH |

### Chain-of-Thought Experiments (O11, O12, O13) - ALL MISMATCHED

| ID | Metric | Paper | Actual (JSON) | Status |
|----|--------|-------|---------------|--------|
| O11 | Exact | 18.95% | 18.93% | **MISMATCH** |
| O11 | Alias | 20.31% | 20.27% | **MISMATCH** |
| O11 | Contain | 27.69% | 26.48% | **MISMATCH** |
| O11 | Jaccard | 28.97% | 27.69% | **MISMATCH** |
| O12 | Exact | 22.36% | 22.51% | **MISMATCH** |
| O12 | Alias | 23.72% | 23.89% | **MISMATCH** |
| O12 | Contain | 30.82% | 30.91% | **MISMATCH** |
| O12 | Jaccard | 32.20% | 32.33% | **MISMATCH** |
| O13 | Exact | 9.52% | 6.22% | **MAJOR MISMATCH** |
| O13 | Alias | 10.27% | 6.69% | **MAJOR MISMATCH** |
| O13 | Contain | 15.50% | 11.72% | **MAJOR MISMATCH** |
| O13 | Jaccard | 16.19% | 12.24% | **MAJOR MISMATCH** |

**Note:** O13 values in the paper (9.52%, 10.27%, etc.) do not match the metrics JSON at all. The paper's O13 exact=9.52% is curiously close to what one might expect from a different model or dataset version. The actual O13 exact match is 6.22%.

---

## 3. Closed-World Results (Table 5)

| ID | Metric | Paper | Actual (JSON) | Status |
|----|--------|-------|---------------|--------|
| C1 | Hit@1 | 16.51% | 16.51% | MATCH |
| C1 | Hit@3 | 26.38% | 26.38% | MATCH |
| C1 | Hit@5 | 30.97% | 30.97% | MATCH |
| C1 | Hit@10 | 36.76% | 36.76% | MATCH |
| C1 | MRR | 0.2362 | 0.2362 | MATCH |
| C1 | Alias H@1 | 22.08% | 22.08% | MATCH |
| C2 | Hit@1 | 16.64% | 16.64% | MATCH |
| C2 | Hit@3 | 27.78% | 27.78% | MATCH |
| C2 | Hit@5 | 33.41% | 33.41% | MATCH |
| C2 | Hit@10 | 40.60% | 40.60% | MATCH |
| C2 | MRR | 0.2480 | 0.2480 | MATCH |
| C2 | Alias H@1 | 21.78% | 21.78% | MATCH |
| C3 | Hit@1 | 14.38% | 14.38% | MATCH |
| C3 | Hit@3 | 25.10% | 25.10% | MATCH |
| C3 | Hit@5 | 29.92% | 29.92% | MATCH |
| C3 | Hit@10 | 37.32% | 37.32% | MATCH |
| C3 | MRR | 0.2211 | 0.2211 | MATCH |
| C3 | Alias H@1 | 19.32% | 19.32% | MATCH |
| C4 | Hit@1 | 30.00% | 30.00% | MATCH |
| C4 | Hit@3 | 46.36% | 46.36% | MATCH |
| C4 | Hit@5 | 53.66% | 53.66% | MATCH |
| C4 | Hit@10 | 63.31% | 63.31% | MATCH |
| C4 | MRR | 0.4131 | 0.4131 | MATCH |
| C4 | Alias H@1 | 37.10% | 37.10% | MATCH |
| C5 | Hit@1 | 35.38% | 35.38% | MATCH |
| C5 | Hit@3 | 53.51% | 53.51% | MATCH |
| C5 | Hit@5 | 61.82% | 61.82% | MATCH |
| C5 | Hit@10 | 71.49% | 71.49% | MATCH |
| C5 | MRR | 0.4751 | 0.4751 | MATCH |
| C5 | Alias H@1 | 42.80% | 42.80% | MATCH |
| C6 | Hit@1 | 27.95% | 27.95% | MATCH |
| C6 | Hit@3 | 44.98% | 44.98% | MATCH |
| C6 | Hit@5 | 51.82% | 51.82% | MATCH |
| C6 | Hit@10 | 59.55% | 59.55% | MATCH |
| C6 | MRR | 0.3851 | 0.3851 | MATCH |
| C6 | Alias H@1 | 34.38% | 34.38% | MATCH |

---

## 4. Per-Entity-Type Analysis (Table 7)

All values in Table 7 match `type_breakdown.json` exactly.

| Type | n (paper) | n (actual) | Status |
|------|-----------|------------|--------|
| Place | 2,076 | 2,076 | MATCH |
| Organization | 1,152 | 1,152 | MATCH |
| Person | 698 | 697 (O1) / 698 (O2) | MATCH (O2 used) |
| Event | 273 | 273 | MATCH |
| Work | 215 | 215 | MATCH |
| Military Unit | 121 | 121 | MATCH |
| Other | 98 | 98 | MATCH |

All per-type percentages (O1, O2, O5, C1, C2) match the JSON data exactly.

---

## 5. Statistical Significance (Table 9)

| Comparison | Metric | Paper | Actual (JSON) | Status |
|-----------|--------|-------|---------------|--------|
| O1 vs O2 | Acc A | 35.06% | 35.06% | MATCH |
| O1 vs O2 | Acc B | 41.11% | 41.11% | MATCH |
| O1 vs O2 | chi2 | 149.69 | 149.69 | MATCH |
| O1 vs O2 | A-only | 120 | 120 | MATCH |
| O1 vs O2 | B-only | 400 | 400 | MATCH |
| O3 vs O4 | Acc A | 35.16% | 35.16% | MATCH |
| O3 vs O4 | Acc B | 38.72% | 38.72% | MATCH |
| O3 vs O4 | chi2 | 36.18 | 36.18 | MATCH |
| O3 vs O4 | A-only | 150 | 150 | MATCH |
| O3 vs O4 | B-only | 275 | 275 | MATCH |
| O1 vs O5 | Acc A | 35.06% | 35.06% | MATCH |
| O1 vs O5 | Acc B | 20.19% | 20.19% | MATCH |
| O1 vs O5 | chi2 | 432.28 | 432.28 | MATCH |
| O1 vs O5 | A-only | 892 | 892 | MATCH |
| O1 vs O5 | B-only | 203 | 203 | MATCH |
| O1 vs C2 | Acc A | 35.06% | 35.06% | MATCH |
| O1 vs C2 | Acc B | 22.58% | 22.58% | MATCH |
| O1 vs C2 | chi2 | 181.93 | 181.93 | MATCH |
| O1 vs C2 | A-only | 1,204 | 1,204 | MATCH |
| O1 vs C2 | B-only | 626 | 626 | MATCH |
| O5 vs O7 | Acc A | 20.18% | 20.18% | MATCH |
| O5 vs O7 | Acc B | 17.27% | 17.27% | MATCH |
| O5 vs O7 | chi2 | 12.64 | 12.64 | MATCH |
| O5 vs O7 | A-only | 778 | 778 | MATCH |
| O5 vs O7 | B-only | 643 | 643 | MATCH |

---

## 6. Error Analysis (Table 8)

| Exp | Category | Paper | Actual | Status |
|-----|----------|-------|--------|--------|
| O1 | Same-type, unrelated | 43.0 | 43.0 | MATCH |
| O1 | Wrong type | 28.5 | 28.5 | MATCH |
| O1 | Same-type, related | 24.5 | 24.5 | MATCH |
| O1 | Partial match | 3.5 | 3.5 | MATCH |
| O1 | Empty/hallucination | 0.5 | 0.5 (EMPTY) | MATCH |
| O2 | Same-type, unrelated | 42.0 | 42.0 | MATCH |
| O2 | Wrong type | 27.5 | 27.5 | MATCH |
| O2 | Same-type, related | 25.5 | 25.5 | MATCH |
| O2 | Partial match | 4.0 | 4.0 | MATCH |
| O2 | Empty/hallucination | 1.0 | 0.5+0.5 (EMPTY+AMBIG) | MATCH (combined) |
| O3 | Same-type, unrelated | 43.5 | 43.5 | MATCH |
| O3 | Wrong type | 29.5 | 29.5 | MATCH |
| O3 | Same-type, related | 22.5 | 22.5 | MATCH |
| O3 | Partial match | 3.0 | 3.0 | MATCH |
| O3 | Empty/hallucination | 1.5 | 0.5+1.0 (EMPTY+HALLUC) | MATCH (combined) |
| O4 | Same-type, unrelated | 45.0 | 45.0 | MATCH |
| O4 | Wrong type | 22.5 | 22.5 | MATCH |
| O4 | Same-type, related | 24.0 | 24.0 | MATCH |
| O4 | Partial match | 6.0 | 6.0 | MATCH |
| O4 | Empty/hallucination | 2.5 | 2.5 (EMPTY) | MATCH |
| O5 | Same-type, unrelated | 52.0 | 52.0 | MATCH |
| O5 | Wrong type | 31.0 | 31.0 | MATCH |
| O5 | Same-type, related | 13.5 | 13.5 | MATCH |
| O5 | Partial match | 2.5 | 2.5 | MATCH |
| O5 | Empty/hallucination | 1.0 | 1.0 (EMPTY) | MATCH |
| O6 | Same-type, unrelated | 46.0 | 46.0 | MATCH |
| O6 | Wrong type | 35.0 | 35.0 | MATCH |
| O6 | Same-type, related | 17.0 | 17.0 | MATCH |
| O6 | Partial match | 1.5 | 1.5 | MATCH |
| O6 | Empty/hallucination | 0.0 | 0.0 | **MISMATCH** |

**Note on O6:** The actual data has AMBIGUOUS=0.5% which the paper drops, making the paper total 99.5% instead of 100%. The paper shows 0.0% for Empty/hallucination, but the AMBIGUOUS category (0.5%) is unaccounted for.

---

## 7. Training Hyperparameters

### DPR (Appendix B.1)

| Parameter | Paper | Code (`train_dpr.py`) | Status |
|-----------|-------|-----------------------|--------|
| Base model | BAAI/bge-base-en-v1.5 | BAAI/bge-base-en-v1.5 | MATCH |
| Model params | ~110M | ~110M | MATCH |
| Embedding dim | 768 | 768 | MATCH |
| Training examples | 17,971 | 17,971 | MATCH |
| Epochs | 3 | 3 | MATCH |
| Batch size | 48 | 48 | MATCH |
| Learning rate | 2e-5 | 2e-5 | MATCH |
| Warmup steps | 100 | 100 | MATCH |
| Loss function | MNRL | MultipleNegativesRankingLoss | MATCH |
| Optimizer | AdamW | AdamW (default) | MATCH |
| Mixed precision | FP16 (AMP) | use_amp=True | MATCH |
| Negatives | In-batch (47) | In-batch (batch-1=47) | MATCH |
| Seed | 42 | 42 | MATCH |

### QLoRA O10 (Appendix B.2)

| Parameter | Paper (Appendix B.2) | Code (`o10_train_eval_llama8b.py`) | Status |
|-----------|----------------------|------------------------------------|--------|
| Base model | meta-llama/Llama-3.1-8B-Instruct | meta-llama/Llama-3.1-8B-Instruct | MATCH |
| Quantization | 4-bit NF4 | nf4 | MATCH |
| Compute dtype | bfloat16 | bfloat16 | MATCH |
| LoRA rank | 16 | 16 | MATCH |
| LoRA alpha | 32 | 32 | MATCH |
| LoRA dropout | 0.05 | 0.05 | MATCH |
| Target modules | q,v,k,o_proj | q,v,k,o_proj | MATCH |
| Training examples | 17,971 | 17,971 | MATCH |
| Epochs | 2 | 2 | MATCH |
| Per-device batch size | **16** | **48** | **MISMATCH** |
| Gradient accumulation | **2 (effective 32)** | **1 (effective 48)** | **MISMATCH** |
| Learning rate | 2e-4 | 2e-4 | MATCH |
| Max seq length | 192 | 192 | MATCH |
| Warmup steps | 50 | 50 | MATCH |
| Validation samples | 500 | 500 | MATCH |
| Framework | TRL SFTTrainer + PEFT | TRL SFTTrainer + PEFT | MATCH |

**Critical:** The paper's Appendix B.2 says per-device batch size is 16 with gradient accumulation 2 (effective batch 32). The actual code uses TRAIN_BATCH=48 with GRAD_ACCUM=1 (effective batch 48). The O10 metrics JSON also records `train_batch: 48`.

Additionally, the paper body (Section 4.3.3) says "batch=48" which matches the code but contradicts Appendix B.2.

---

## 8. Experiment Count

| Claim Location | Paper Value | Actual Count | Status |
|----------------|-------------|--------------|--------|
| Abstract | "17 experimental configurations" | 19 | **MISMATCH** |
| Contribution #4 | "20 experimental configurations" | 19 | **MISMATCH** |
| Conclusion | "20 experimental configurations" | 19 | **MISMATCH** |

**Actual experiment IDs (19):** O1, O2, O3, O4, O5, O6, O7, O8, O10, O11, O12, O13, RAG1, C1, C2, C3, C4, C5, C6

The paper is internally inconsistent: the abstract says "17" while the contributions and conclusion say "20". Neither matches the actual count of 19.

---

## 9. Prose Claims in Discussion/Findings

| Claim | Location | Paper Value | Actual Value | Status |
|-------|----------|-------------|--------------|--------|
| "Llama 3.2 1B (9.52%)" model scale | Section 6.1, para 2 | 9.52% | O7=17.27% (4-tier), no separate exact metric | **MISMATCH** |
| O10 exact match = 38.94% | Finding 2 | 38.94% | 38.94% | MATCH |
| O10 Jaccard = 51.59% | Finding 2 | 51.59% | 51.59% | MATCH |
| O2 exact = 31.62% | Finding 2 | 31.62% | 31.62% | MATCH |
| CoT reduces GPT-4o from 33.30% to 23.72% | Finding 3 | 23.72% | 23.89% | **MISMATCH** |
| CoT reduces GPT-4.1-mini from 25.71% to 18.95% | Finding 3 | 18.95% | 18.93% | **MISMATCH** |
| CoT reduces Llama 8B from 13.92% to 9.52% | Finding 3 | 9.52% | 6.22% | **MAJOR MISMATCH** |
| CoT relative drop GPT-4o = 28.7% | Finding 3 | 28.7% | 28.3% | **MISMATCH** |
| CoT relative drop mini = 26.3% | Finding 3 | 26.3% | 26.4% | **MISMATCH** |
| CoT relative drop Llama = 31.6% | Finding 3 | 31.6% | 55.3% | **MAJOR MISMATCH** |
| CoT range "4.40 to 6.76 pp" | Conclusion | 4.40-6.76 | 4.51-7.70 | **MISMATCH** |
| Few-shot: GPT-4o +4.60 pp | Finding 4 | +4.60 | +4.60 | MATCH |
| Few-shot: GPT-4.1-mini +2.95 pp | Finding 4 | +2.95 | +2.95 | MATCH |
| Few-shot: Llama 8B +3.91 pp | Finding 4 | +3.91 | +3.91 | MATCH |
| DPR+Desc vs DPR+Name = +5.38 pp | Finding 5 | +5.38 | +5.38 | MATCH |
| DPR+Desc vs DPR+Wiki = +7.43 pp | Finding 5 | +7.43 | +7.43 | MATCH |
| RAG1 vs mini ZS = 5.99 pp below | Finding 6 | 5.99 | 6.00 | MATCH (rounding) |
| GPT-4o ZS vs Llama 8B ZS = +13.10 pp | Finding 7 | 13.10 | 13.10 | MATCH |

---

## 10. Prompt Templates (Appendix A vs Code)

| Prompt | Paper (Appendix A) | Code | Status |
|--------|-------------------|------|--------|
| ZS system | Matches | `run_open_world.py` ZS_SYSTEM | MATCH |
| ZS user | Matches | `run_open_world.py` ZS_PROMPT | MATCH |
| FS system | Matches | Same as ZS | MATCH |
| FS user + examples | Matches | `run_open_world.py` FS_PROMPT + FS_EXAMPLES | MATCH |
| FS example count | 5 | 5 | MATCH |
| CoT system | Matches | `submit_phase_b.py` COT_SYSTEM | MATCH |
| CoT user | Matches | `submit_phase_b.py` make_cot_prompt | MATCH |
| RAG prompt | Matches | `submit_phase_b.py` make_rag_prompt | MATCH |
| QLoRA prompt | Matches | `o10_train_eval_llama8b.py` build_prompt | MATCH |
| ZS/FS: temp=0.0, max=100 | Paper A.1/A.2 | Code | MATCH |
| CoT: temp=0.7, max=300 | Paper A.3 | Code | MATCH |
| RAG: temp=0.7, max=50 | Paper A.4 | Code | MATCH |
| QLoRA: greedy, max_new=30 | Paper A.5 | Code | MATCH |

---

## 11. Cross-Paradigm Ranking (Table 6)

Table 6 uses O10 Jaccard (51.59%) as the "Alias H@1" for the open-world entry. This is technically a Jaccard metric, not an alias metric. The table header says "Alias H@1 (%)" but the O10 entry uses the Jaccard value. This conflation is misleading but consistent with how the paper defines its cross-paradigm comparison.

| Rank | System | Paper Value | Actual Value | Status |
|------|--------|-------------|--------------|--------|
| 1 | O10 | 51.59% | 51.59% (Jaccard) | MATCH |
| 2 | C5 | 42.80% | 42.80% | MATCH |
| 3 | O2 | 41.10% | 41.10% (Jaccard) | MATCH |
| 4 | O4 | 39.48% | 39.48% (Jaccard) | MATCH |
| 5 | C4 | 37.10% | 37.10% | MATCH |
| 6 | O3 | 35.94% | 35.94% (Jaccard) | MATCH |
| 7 | O1 | 35.05% | 35.05% (Jaccard) | MATCH |
| 8 | C6 | 34.38% | 34.38% | MATCH |

---

## Summary of All Mismatches Requiring Correction

### Critical (numbers wrong)

1. **Table 4, O13 (Llama 3.1 8B CoT):** All four metrics are wrong.
   - Paper: 9.52 / 10.27 / 15.50 / 16.19
   - Actual: 6.22 / 6.69 / 11.72 / 12.24
   - Fix: Update Table 4 O13 row to actual values.

2. **Table 4, O11 (GPT-4.1-mini CoT):** All four metrics are slightly wrong.
   - Paper: 18.95 / 20.31 / 27.69 / 28.97
   - Actual: 18.93 / 20.27 / 26.48 / 27.69
   - Fix: Update Table 4 O11 row to actual values.

3. **Table 4, O12 (GPT-4o CoT):** All four metrics are slightly wrong.
   - Paper: 22.36 / 23.72 / 30.82 / 32.20
   - Actual: 22.51 / 23.89 / 30.91 / 32.33
   - Fix: Update Table 4 O12 row to actual values.

4. **Section 6.1, paragraph 2:** "Llama 3.2 1B (9.52%)" is wrong. 9.52% is not the O7 value. O7's 4-tier alias eval is 17.27%. There is no separate exact match for O7. If referring to O7 exact match from the alias eval tier breakdown, exact (tier 1) is 59/4633 = 1.27%. The value 9.52% does not correspond to any known metric.
   - Fix: Correct the value to match O7's actual metric, or clarify which metric is being referenced.

5. **Appendix B.2, QLoRA batch size:** Paper says per-device=16, gradient accumulation=2 (effective 32). Code uses per-device=48, gradient accumulation=1 (effective 48). O10 metrics JSON records train_batch=48.
   - Fix: Change Appendix B.2 to per-device=48, gradient accumulation=1, effective batch=48. Or if the actual training was done with different params from the committed code, clarify.

### Moderate (inconsistencies in prose)

6. **Experiment count inconsistency:** Abstract says "17", Contribution #4 and Conclusion say "20", actual count is 19.
   - Fix: Use consistent number (19) throughout.

7. **Finding 3 CoT values:** All three CoT-related claims use the old/wrong O11/O12/O13 numbers and need updating.
   - GPT-4o CoT alias: paper 23.72%, actual 23.89%
   - GPT-4.1-mini CoT exact: paper 18.95%, actual 18.93%
   - Llama 8B CoT exact: paper 9.52%, actual 6.22%
   - All relative drop percentages need recalculation.

8. **Conclusion CoT drop range:** Paper says "4.40 to 6.76 pp". Actual range is 4.51 to 7.70 pp (using corrected O13).
   - Fix: Recalculate based on corrected metrics.

### Minor

9. **Table 8, O6 Empty/hallucination:** Paper shows 0.0%, but actual data has AMBIGUOUS=0.5% that is unaccounted for, making the paper row sum 99.5%.
   - Fix: Either include AMBIGUOUS in Empty/hallucination (making it 0.5%) or note the discrepancy.

10. **data/README.md entity counts:** README says Train=8,631/Dev=1,233/Test=2,473 but split_metadata.json says Train=8,635/Dev=1,234/Test=2,468. The paper matches the metadata, so this is a README staleness issue, not a paper issue.

---

## Claims That Could Not Be Verified

1. **O13 source code:** No Python script was found that generates O13 (Llama 3.1 8B CoT). The `submit_phase_b.py` script only handles O11 and O12 (GPT models via OpenAI Batch API). O13 was presumably run separately via OpenRouter, but the script is not in the repository.

2. **O14:** A predictions file exists (`O14_predictions.json`) but no metrics file. This experiment is not mentioned in the paper.

3. **Wikipedia coverage (84.6%), description coverage (70.9%), alias coverage (51.2%):** Reported in the paper and README for entity_kb.json. Not independently verified from the KB file contents.

4. **Stage 2 total entity mentions (31,284):** Reported in README and paper. Not verified by summing entity extraction files.

5. **Stage 3 total summaries (25,161):** Reported in README and paper. Not verified by counting individual samples in summary files.

6. **"25+ institutional archives" claim (Table 1):** The number of distinct source institutions was not independently counted.
