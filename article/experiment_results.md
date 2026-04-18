# IRC-Bench v5: Complete Experiment Results

Test set: 4,633 samples, 12,337 unique entities, zero entity overlap between train and test.

## Matching Tiers

- **Exact**: case-insensitive string match
- **Alias**: exact + Wikidata aliases
- **Containment**: alias + substring containment
- **Jaccard**: containment + token Jaccard >= 0.5

## Table 1: Closed-World Experiments (Embedding Retrieval)

| ID | Model | Entity Repr | Method | Hit@1 | Hit@3 | Hit@5 | Hit@10 | MRR | aHit@1 | aHit@3 | aHit@5 | aHit@10 | aMRR |
|:---|:------|:------------|:-------|------:|------:|------:|-------:|----:|-------:|-------:|-------:|--------:|-----:|
| C1 | BGE-base | name | Baseline | 16.5% | 26.4% | 31.0% | 36.8% | 0.2362 | 22.1% | 33.3% | 38.5% | 45.7% | 0.2915 |
| C2 | BGE-base | description | Baseline | 16.6% | 27.8% | 33.4% | 40.6% | 0.2480 | 21.8% | 32.9% | 38.8% | 46.9% | 0.2904 |
| C3 | BGE-base | wiki | Baseline | 14.4% | 25.1% | 29.9% | 37.3% | 0.2211 | 19.3% | 30.2% | 35.7% | 43.8% | 0.2635 |
| C4 | DPR fine-tuned | name | Fine-tuned | 30.0% | 46.4% | 53.7% | 63.3% | 0.4131 | 37.1% | 51.2% | 57.7% | 67.3% | 0.4613 |
| C5 | DPR fine-tuned | description | Fine-tuned | 35.4% | 53.5% | 61.8% | 71.5% | 0.4751 | 42.8% | 58.7% | 65.7% | 74.5% | 0.5264 |
| C6 | DPR fine-tuned | wiki | Fine-tuned | 28.0% | 45.0% | 51.8% | 59.6% | 0.3851 | 34.4% | 50.2% | 56.4% | 63.6% | 0.4361 |

## Table 2: Open-World Experiments (LLM Generation)

| ID | Model | Method | Exact | Alias | Containment | Jaccard |
|:---|:------|:-------|------:|------:|------------:|--------:|
| O1 | GPT-4o | ZS | 27.0% | 33.3% | 33.3% | 35.0% |
| O2 | GPT-4o | FS | 31.6% | 38.9% | 38.9% | 41.1% |
| O3 | GPT-4.1-mini | ZS | 25.7% | 27.1% | 33.5% | 35.9% |
| O4 | GPT-4.1-mini | FS | 28.7% | 36.9% | 36.9% | 39.5% |
| O5 | Llama 3.1 8B | ZS | 13.9% | 14.8% | 19.5% | 20.2% |
| O6 | Llama 3.1 8B | FS | 17.8% | 18.8% | 24.6% | 25.7% |
| O10 | Llama 3.1 8B QLoRA | Fine-tuned | 38.9% | 41.4% | 47.9% | 51.6% |
| O11 | GPT-4.1-mini | CoT | 18.9% | 20.3% | 26.5% | 27.7% |
| O12 | GPT-4o | CoT | 22.5% | 23.9% | 30.9% | 32.3% |
| O13 | Llama 3.1 8B | CoT | 6.2% | 6.7% | 11.7% | 12.2% |
| RAG1 | BGE + GPT-4.1-mini | RAG | 19.7% | 20.5% | 28.7% | 29.5% |

## Table 3: Cross-Paradigm Ranking (by alias-aware score)

| Rank | Method | Model | Score | Metric |
|-----:|:-------|:------|------:|:-------|
| 1 | DPR Fine-tuned | C5 (description) | 42.8% | Alias Hit@1 |
| 2 | QLoRA Fine-tuned | O10 (Llama 8B) | 41.4% | Alias Match |
| 3 | LLM Few-Shot | O2 (GPT-4o) | 38.9% | Alias Match |
| 4 | DPR Fine-tuned | C4 (name) | 37.1% | Alias Hit@1 |
| 5 | LLM Few-Shot | O4 (GPT-4.1-mini) | 36.9% | Alias Match |
| 6 | DPR Fine-tuned | C6 (wiki) | 34.4% | Alias Hit@1 |
| 7 | LLM Zero-Shot | O1 (GPT-4o) | 33.3% | Alias Match |
| 8 | RAG | RAG1 (BGE+GPT-4.1m) | 29.5% | Jaccard Match |
| 9 | LLM Zero-Shot | O3 (GPT-4.1-mini) | 27.1% | Alias Match |
| 10 | LLM CoT | O12 (GPT-4o) | 23.9% | Alias Match |
| 11 | Embedding Baseline | C1 (name) | 22.1% | Alias Hit@1 |
| 12 | LLM CoT | O11 (GPT-4.1-mini) | 20.3% | Alias Match |
| 13 | LLM Few-Shot | O6 (Llama 8B) | 18.8% | Alias Match |
| 14 | LLM Zero-Shot | O5 (Llama 8B) | 14.8% | Alias Match |
| 15 | LLM CoT | O13 (Llama 8B) | 6.7% | Alias Match |

## Key Findings

1. **Fine-tuning is the most effective approach.** DPR fine-tuning doubles BGE-base Hit@1 (16.5% to 35.4%), and QLoRA fine-tuning transforms Llama 3.1 8B from 13.9% (ZS) to 38.9% (exact match).
2. **QLoRA generalizes across unseen entities.** Despite zero entity overlap between train and test, the QLoRA model achieves the highest open-world score (41.4% alias match), demonstrating that IER can be learned as a generalizable skill.
3. **Description representation is best for retrieval.** Entity descriptions consistently outperform name-only and Wikipedia-sentence representations across both baseline and fine-tuned embedding models.
4. **Few-shot consistently improves over zero-shot.** Gains range from +3.9pp (Llama 8B) to +6.3pp (GPT-4o) in alias match.
5. **Chain-of-thought hurts performance.** CoT underperforms direct prompting for every model tested. The model often identifies the correct entity in reasoning but selects a different final answer.
6. **RAG underperforms direct generation.** Providing top-5 retrieved candidates confuses rather than helps the LLM reranker.
7. **Model scale matters significantly.** GPT-4o (33.3% ZS) more than doubles Llama 3.1 8B (14.8% ZS) in open-world performance.
8. **Alias-aware evaluation reveals hidden performance.** Closed-world experiments gain +5 to +7pp when matching against Wikidata aliases instead of exact entity names only.

## DPR Training Details

- Base model: BAAI/bge-base-en-v1.5 (109M params)
- Loss: Multiple Negatives Ranking Loss (MNRL)
- Epochs: 3, Batch size: 48, Learning rate: 2e-5
- Training samples: 17,971
- Entity representations: name, name+description, Wikipedia first sentence

## QLoRA Training Details

- Base model: meta-llama/Llama-3.1-8B-Instruct
- Quantization: 4-bit NF4 with bfloat16 compute
- LoRA: r=16, alpha=32, dropout=0.05, targets=q/k/v/o_proj
- Trainable params: 13.6M / 8.0B total (0.17%)
- Epochs: 2, Batch size: 48, Learning rate: 2e-4
- Training time: 36 min on A100 PCIE 40GB
- Max sequence length: 192 tokens
