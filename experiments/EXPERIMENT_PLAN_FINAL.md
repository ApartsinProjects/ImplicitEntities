# Final Experiment Plan v2

## Benchmark
- **File:** `data/benchmark_v2/IRC_Bench_v3.csv`
- **Samples:** 3,325 (partition: train=2,632 / test=693, 0 entity overlap)
- **Variants:** veterans_t2e (530), veterans_e2t (1,246), twitter_t2e (807), twitter_e2t (742)
- **Entities:** 803 unique, 100% Wikidata linked, 91% with descriptions, 73% with aliases

## Evaluation
- **Matching:** 6-tier (exact, Wikidata alias, containment, Jaccard, WordNet synonym, LLM alias)
- **Metrics:** Hit@K, P@K, R@K, nDCG@K (K=1,3,5,10), Global MRR, Filtered MRR
- **Reporting:** Strict (exact only), Normalized (all 6 tiers), Graded (semantic similarity)
- **Evaluation set:** test partition only (693 samples, 158 unseen entities)

## Methods (13 configurations)

### A. LLM Generative (6)

| ID | Model | Mode | Prompt | Backend | Est. Cost |
|---|---|---|---|---|---|
| A1 | GPT-4o-mini | Zero-shot | v3 (typed defs + Wikipedia criterion) | OpenAI Batch 50% | $0.12 |
| A2 | GPT-4o-mini | Few-shot | 5 worked examples | OpenAI Batch 50% | $0.12 |
| A3 | Llama 3.1 8B | Zero-shot | v3 (typed defs + Wikipedia criterion) | OpenRouter :floor | $0.05 |
| A4 | Llama 3.1 8B | Few-shot | 5 worked examples | OpenRouter :floor | $0.05 |
| A5 | GPT-4o | Zero-shot | v3 (typed defs + Wikipedia criterion) | OpenAI Batch 50% | $2.00 |
| A6 | GPT-4o | Few-shot | 5 worked examples | OpenAI Batch 50% | $2.00 |

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

| ID | Model | Fine-tuned | Prompt | Backend | Est. Cost |
|---|---|---|---|---|---|
| C1 | Llama 3.2 1B (4bit) | LoRA r=16 | Instruction | Local GPU | $0 |

## Cost Summary

| Group | Runs | Cost |
|---|---|---|
| A1-A2 (GPT-4o-mini ZS+FS) | 2 x 4 variants = 8 | $0.24 |
| A3-A4 (Llama ZS+FS) | 2 x 4 variants = 8 | $0.10 |
| A5-A6 (GPT-4o ZS+FS) | 2 x 4 variants = 8 | $4.00 |
| B1-B6 (Embedding + DPR) | 6 x 4 variants = 24 | $0 |
| C1 (LoRA) | 1 x 4 variants = 4 | $0 |
| **Total** | **52 runs** | **~$4.34** |

## Execution Order

1. Fix all blockers (IRC_Bench_v3 loader, Wikipedia descriptions, DPR BGE support)
2. B1-B3 (zero-shot embedding, local GPU, free, fast)
3. A1-A2 (GPT-4o-mini, submit OpenAI Batch, async)
4. A3-A4 (Llama 8B, OpenRouter, concurrent)
5. B4-B6 (DPR fine-tune BGE-base x 3 targets, local GPU, sequential)
6. A5-A6 (GPT-4o, submit OpenAI Batch, async)
7. C1 (LoRA Llama, local GPU)
8. Re-evaluate all on test partition with 6-tier matching + semantic similarity
9. Generate final comparison tables

## Blockers (must fix before running)

- [ ] Add IRC_Bench_v3.csv loader with partition filtering to run_experiments.py
- [ ] Fetch Wikipedia first sentences for 803 entities (B3/B6 target)
- [ ] Add BGE-base to DPR training script (train_dpr.py BASE_MODEL_MAP)
- [ ] Update train_crossencoder.py and train_lora_llm.py to load IRC_Bench_v3.csv
- [ ] Integrate embedding_cache.py into retrieval scripts
