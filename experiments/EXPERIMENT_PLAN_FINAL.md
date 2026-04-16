# Final Experiment Plan

## Benchmark
- **File:** `data/benchmark_v2/IRC_Bench_v3.csv`
- **Samples:** 3,325 (partition column: train=2,632 / test=693, 0 entity overlap)
- **Variants:** veterans_t2e (530), veterans_e2t (1,246), twitter_t2e (807), twitter_e2t (742)
- **Entities:** 803 unique, 100% Wikidata linked, 91% with descriptions

## Evaluation
- **Matching:** 6-tier (exact, Wikidata alias, containment, Jaccard, WordNet synonym, LLM alias)
- **Metrics:** Hit@K, P@K, R@K, nDCG@K (K=1,3,5,10), Global MRR, Filtered MRR
- **Levels:** Strict (exact only), Normalized (all 6 tiers), Graded (semantic similarity)

## Methods (16 configurations)

### A. LLM Generative (9)

| ID | Model | Mode | Entity Target | Backend | Est. Cost |
|---|---|---|---|---|---|
| A1 | GPT-4o-mini | Zero-shot | Free generation | OpenAI Batch 50% | $0.12 |
| A2 | GPT-4o-mini | Few-shot | Free generation | OpenAI Batch 50% | $0.12 |
| A3 | GPT-4o-mini | Closed-set | Entity list (803) | OpenAI Batch 50% | $0.12 |
| A4 | Llama 3.1 8B | Zero-shot | Free generation | OpenRouter :floor | $0.05 |
| A5 | Llama 3.1 8B | Few-shot | Free generation | OpenRouter :floor | $0.05 |
| A6 | Llama 3.1 8B | Closed-set | Entity list (803) | OpenRouter :floor | $0.10 |
| A7 | GPT-4o | Zero-shot | Free generation | OpenAI Batch 50% | $2.00 |
| A8 | GPT-4o | Few-shot | Free generation | OpenAI Batch 50% | $2.00 |
| A9 | GPT-4o | Closed-set | Entity list (803) | OpenAI Batch 50% | $2.00 |

### B. Embedding Retrieval (6)

| ID | Model | Fine-tuned | Entity Target | Backend | Est. Cost |
|---|---|---|---|---|---|
| B1 | BGE-base (109M) | No | Entity name | Local GPU | $0 |
| B2 | BGE-base (109M) | No | Type + name | Local GPU | $0 |
| B3 | BGE-base (109M) | No | Wikipedia description | Local GPU | $0 |
| B4 | BGE-base (109M) | DPR (MNRL) | Entity name | Local GPU | $0 |
| B5 | BGE-base (109M) | DPR (MNRL) | Type + name | Local GPU | $0 |
| B6 | BGE-base (109M) | DPR (MNRL) | Wikipedia description | Local GPU | $0 |

### C. Fine-tuned LLM (1)

| ID | Model | Fine-tuned | Entity Target | Backend | Est. Cost |
|---|---|---|---|---|---|
| C1 | Llama 3.2 1B (4bit) | LoRA r=16 | Free generation | Local GPU | $0 |

## Cost Summary

| Group | Experiments | Cost |
|---|---|---|
| A1-A3 (GPT-4o-mini) | 3 x 4 variants = 12 | $0.36 |
| A4-A6 (Llama 8B) | 3 x 4 variants = 12 | $0.20 |
| A7-A9 (GPT-4o) | 3 x 4 variants = 12 | $6.00 |
| B1-B6 (Embedding + DPR) | 6 x 4 variants = 24 | $0 |
| C1 (LoRA) | 1 x 4 variants = 4 | $0 |
| **Total** | **64 runs** | **~$6.56** |

## Execution Order

1. **B1-B3** (zero-shot embedding, local GPU, free, fast)
2. **A1-A3** (GPT-4o-mini, submit OpenAI Batch, async)
3. **A4-A6** (Llama 8B, OpenRouter, concurrent)
4. **B4-B6** (DPR fine-tune BGE-base x 3 targets, local GPU, sequential)
5. **A7-A9** (GPT-4o, submit OpenAI Batch, async)
6. **C1** (LoRA Llama, local GPU)
7. **Re-evaluate all** on test partition with 6-tier matching
8. **Generate final comparison table**

## Prerequisites
- [ ] Fetch Wikipedia first sentences for B3/B5/B6 entity descriptions
- [ ] Update load_dataset() to use IRC_Bench_v3.csv with partition filter
- [ ] Update DPR training to use IRC_Bench_v3.csv train partition
