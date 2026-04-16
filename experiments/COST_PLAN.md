# Experiment Cost Plan

## OpenRouter Cost Optimization

OpenRouter has **no batch API**. Cost reduction strategies:

| Strategy | Savings | How to Use |
|---|---|---|
| `:floor` suffix | ~25-50% | Append to model slug: `google/gemini-2.0-flash-001:floor` |
| `:free` suffix | 100% | Free tier, rate limited: 20 req/min, 200 req/day |
| Prompt caching | 50-75% | Automatic on repeated system prompts (sticky routing) |
| `service_tier: "flex"` | ~50% | OpenAI reasoning models only |
| Free models | 100% | DeepSeek V3, Llama 3.3 70B, Gemma 3 27B, etc. |

## Experiment Cost Estimates

### Already Completed
- E2T generation (2,415 prompts, Gemini Flash): ~$0.01
- Baseline experiments (12 runs, Gemini Flash): ~$0.05

### Tier 1: Post-hoc Analysis (analyze_results.py)
- **Cost: $0** (all local computation on prediction CSVs)
- Per-entity-type breakdown, matching layers, kappa, ensembles, bootstrap, nDCG

### Tier 2a: Multi-Model Comparison (run_multimodel.py)

| Model | Est. Prompts | Est. Cost | Notes |
|---|---|---|---|
| google/gemini-2.0-flash-001:floor | 0 | $0 | Already done |
| meta-llama/llama-3.1-8b-instruct:floor | ~3,400 | ~$0.01 | Very cheap |
| mistralai/mistral-7b-instruct:floor | ~3,400 | ~$0.01 | Very cheap |
| deepseek/deepseek-chat-v3-0324:free | ~3,400 | $0 | Free but slow (5 concurrency) |
| anthropic/claude-3.5-haiku:floor | ~3,400 | ~$0.15 | Moderate |
| openai/gpt-4o-mini:floor | ~3,400 | ~$0.05 | Moderate |
| **Total multi-model** | | **~$0.22** | |

### Tier 2b: Ablation Experiments (run_ablations.py)

6 ablation variants x 4 datasets x ~850 avg samples = ~20,400 prompts
Using Gemini Flash :floor: **~$0.03**

### Tier 2c: Embedding Comparison (run_embedding_compare.py)
- **Cost: $0** (all local with sentence-transformers models)

### Budget Summary

| Tier | Cost | Experiments |
|---|---|---|
| Tier 1 (analysis) | $0 | 10 analyses |
| Tier 2a (multi-model) | ~$0.22 | 6 models x 4 datasets |
| Tier 2b (ablations) | ~$0.03 | 6 ablations x 4 datasets |
| Tier 2c (embeddings) | $0 | 4 embedding models |
| **Total** | **~$0.25** | |

## Recommended Execution Order

1. `python analyze_results.py` (free, immediate)
2. `python run_embedding_compare.py --dataset all` (free, local)
3. `python run_ablations.py --dataset veterans_t2e --ablation all` (cheapest, validates pipeline)
4. `python run_multimodel.py --dataset veterans_t2e --models cheap` (test cheap models first)
5. `python run_multimodel.py --dataset all --models all` (full sweep)
6. `python run_ablations.py --dataset all --ablation all` (full ablation sweep)

## Free Model Strategy

For maximum data at zero cost, run experiments sequentially with free models:
```bash
# Each free model: ~200 req/day limit, so one dataset per day per model
python run_multimodel.py --dataset veterans_t2e --models "deepseek/deepseek-chat-v3-0324:free" --concurrency 5
python run_multimodel.py --dataset twitter --models "meta-llama/llama-3.3-70b-instruct:free" --concurrency 5
```
