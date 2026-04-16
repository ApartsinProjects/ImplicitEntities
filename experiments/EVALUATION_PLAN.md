# Evaluation Plan

## Benchmark
- **File:** `data/benchmark_v2/IRC_Bench_v3.csv`
- **15 columns:** uid, partition, variant, domain, generation, text, entity, entity_type, alternative_entities, entity_qid, entity_canonical, entity_aliases, entity_description (LLM), source, entity_description_wiki
- **Evaluation set:** partition=test (693 samples, 158 unseen entities)

## Matching Hierarchy (6 tiers, applied in order)
1. **Exact** - normalized string equality (free)
2. **Wikidata alias** - precomputed alias lookup from wikidata_aliases.json (free)
3. **Containment** - substring in either direction (free)
4. **Jaccard** - token overlap >= 0.50 (free)
5. **WordNet synonym** - Wu-Palmer >= 0.90 or shared lemma (free)
6. **LLM alias** - ask LLM "same entity?" (API cost, top-3 only)

For embedding methods: skip tier 6 (no LLM calls needed).

## Metrics (per experiment)

### Retrieval (at K = 1, 3, 5, 10)
- Hit@K
- Precision@K
- Recall@K (with multi-reference: found / total valid golds)
- nDCG@K (binary relevance: gold found = 1, else = 0)

### Ranking
- Global MRR (0 for misses)
- Filtered MRR (hits only)

### Match Tier Distribution
- Count per tier: exact, wikidata_alias, containment, jaccard, synonym, llm_alias, none

### Graded (semantic similarity)
- Graded nDCG@K (relevance 0-3: exact=3, relevant=2, partial=1, irrelevant=0)
- Mean top-1 embedding cosine similarity
- Mean top-1 combined similarity score
- Partial match rate (combined >= 0.25)
- Strong match rate (combined >= 0.50)
- Wikidata QID match score (1.0/0.9/0.5/0.0)

## Reporting

### Table 2 (main results)
All 13 methods x 4 variants. Hit@1, Hit@3, nDCG@5, MRR.
Normalized matching (all 6 tiers).

### Table 3 (matching analysis)
Top methods only. Tier distribution showing contribution of each tier.

### Table 4 (graded evaluation)
Top methods only. Graded nDCG, semantic similarity, QID match.

### Table 5 (fine-tuned vs zero-shot)
Llama 8B: A3 (ZS) vs A4 (FS) vs C1 (QLoRA). Direct comparison.
BGE-base: B1-B3 (raw) vs B4-B6 (DPR). Effect of fine-tuning.
