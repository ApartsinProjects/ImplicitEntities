# Review Section 2: Additional Results Obtainable from Existing Data

**Reviewer Role:** Top-tier AI/NLP journal reviewer (ACL/EMNLP/NAACL caliber)

These are experiments and analyses that can be conducted using the existing datasets and infrastructure, without collecting new data.

---

## Top 20 Recommendations

### 1. Compare multiple LLM backbones
The paper uses a single unnamed LLM. Run the same pipeline with GPT-4o, GPT-3.5-turbo, Claude 3.5 Sonnet, Llama 3 70B, Mistral Large, and Gemini Pro. This would transform a single-model study into a comprehensive benchmark and is the single most impactful addition.

### 2. Compare multiple embedding models
Test at least 3-4 embedding models: OpenAI text-embedding-3-large, sentence-transformers/all-MiniLM-L6-v2, E5-large-v2, and BGE-large-en. Embedding choice can shift retrieval performance by 15+ points.

### 3. Report per-entity-type breakdown for all metrics
The paper mentions entity type effects qualitatively but Table 2 only shows aggregate results. Provide Hit@1 and MRR separately for Person, Place, and Event entities within each dataset. This is critical since entity type distributions are extremely skewed (86% Place in veterans).

### 4. Compute statistical significance
With N=126 to N=1110 samples, compute paired bootstrap confidence intervals (95%) and McNemar's test for pairwise method comparisons. Report p-values. Several "winning" margins may not be statistically significant.

### 5. Break down matching layer contribution
For every dataset-method pair, report what percentage of successful matches came from: (a) exact match, (b) LLM alias match, (c) Jaccard fuzzy match. This reveals whether high scores reflect genuine recognition or lenient matching.

### 6. Ablate entity type conditioning
Run the LLM and RAG methods without providing entity type as input. This tests whether the methods can simultaneously identify entity type and resolve the entity, which is the realistic deployment scenario.

### 7. Ablate the contextualization step
The LLM method uses a two-step prompt (contextualization then inference). Run with inference-only (skip step 1) to quantify the value of explicit contextualization.

### 8. Test different prompt strategies
The paper fixes "historical background" as the prompt strategy. Test alternatives: (a) direct inference without context, (b) chain-of-thought reasoning, (c) few-shot with examples, (d) structured output format (JSON). Report which yields best results per dataset.

### 9. Vary K for retrieval (RAG and Embedding)
Currently fixed at K=5. Sweep K from 1 to 20 and plot Hit@1 and MRR as a function of K. This reveals the retrieval saturation point and optimal configuration.

### 10. Compute nDCG@K (already mentioned in methodology but not reported)
The methodology section mentions nDCG@K with Jaccard-based graded relevance, but no nDCG results appear in the paper. Compute and report nDCG@1, nDCG@3, nDCG@5, nDCG@10 to capture partial-credit ranking quality.

### 11. Add a fine-tuned NER baseline
Fine-tune a BERT-base or RoBERTa model on the training split for entity classification. Even if it performs poorly (expected, since NER targets explicit mentions), it establishes a supervised baseline that contextualizes the zero-shot LLM results.

### 12. Cross-dataset transfer evaluation
Train/calibrate on one dataset and test on another (e.g., train prompt on veterans, test on tweets). This measures generalization and is critical for practical deployment claims.

### 13. Analyze failure cases systematically
Categorize the failures of each method into taxonomy: (a) entity not in LLM knowledge, (b) description too vague, (c) multiple valid entities, (d) entity type mismatch, (e) cultural specificity. Report percentages. This is more valuable than aggregate metrics for understanding limitations.

### 14. Measure inter-method agreement
Compute Cohen's kappa or Fleiss' kappa between the three methods' predictions. High agreement on failures suggests fundamental task difficulty; low agreement suggests complementary strengths and motivates ensemble approaches.

### 15. Test ensemble methods
Simple ensembles: (a) majority vote across 3 methods, (b) rank fusion (reciprocal rank fusion, CombMNZ), (c) cascading (embedding first, then LLM on failures). These can be computed from existing prediction files without re-running models.

### 16. Analyze text length effects
Bin samples by token count (short/medium/long) and report metrics per bin. The hypothesis that longer texts are harder (veterans vs. tweets) can be tested within-dataset, controlling for domain.

### 17. Report entity frequency/popularity effects
Some entities are well-known (Statue of Liberty, Pearl Harbor) while others are obscure (a specific local school). Correlate entity Wikipedia page views or frequency in Common Crawl with recognition accuracy. This tests whether LLMs succeed because of memorization vs. reasoning.

### 18. Evaluate at K=10 (reported in methodology but missing from results)
The methodology states "cutoff values of K = 1, 3, 5, and 10" but Table 2 only shows K=1, 3, 5. Report K=10 results to show the ceiling effect.

### 19. Compute dataset difficulty metrics
For each sample, compute: (a) text perplexity under a language model, (b) entity ambiguity (number of candidate entities with Jaccard > 0.3), (c) implicit reference "opacity" (cosine distance between implicit text embedding and entity name embedding). Correlate with recognition success.

### 20. Provide per-recollection (per-interview) variance analysis
For Vet-T2E (25 interviews, 1110 segments), report variance in performance across interviews. Some narrators may use more recognizable implicit references. This reveals whether performance is driven by a few easy interviews or is consistent.
