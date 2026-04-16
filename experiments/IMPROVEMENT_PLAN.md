# Improvement Plan: Data, Process, and Methods

Based on analysis of all experiment results across 12 baseline runs, 3 ablations,
2 multi-model comparisons, and 4 embedding model comparisons.

## Current Best Results

| Dataset | Best Method | Hit@1 | Gap to Twitter Best |
|---|---|---|---|
| twitter (LLM) | 0.787 | - |
| e2t_twitter (LLM) | 0.606 | -0.181 |
| veterans_t2e (LLM) | 0.356 | -0.431 |
| e2t_veterans (LLM) | 0.312 | -0.475 |

Veterans performance is 2x worse than Twitter. The gap is the target.

---

## A. DATA IMPROVEMENTS (Expected Impact: +10-20% Hit@1)

### A1. Use cleaned v2 dataset [Ready, just re-run]
- veterans_t2e_v2.csv: 640 samples (removed 206 = 24.3%)
- Removed: generic entities (war, military, I), duplicate texts, leaks
- Expected impact: +5-10% Hit@1 (removes guaranteed failures)
- Command: `python run_experiments.py --dataset veterans_t2e_v2 --method llm`

### A2. Filter to core types only (Person/Place/Event)
- Currently: 5 types including Profession (14.7%) and Organization (16.1%)
- Professions have 39% success vs Places 64%
- Filter to 443 core-type samples for cleaner comparison to paper
- Expected impact: +3-5% Hit@1

### A3. Single entity per text (strict dedup)
- 75 texts have 2-4 gold entities; model can only match 1
- v2 already fixes this, but verify zero duplicates remain
- Expected impact: Already in v2

### A4. Better entity annotations
- "statue of liberty" fails 4x despite being a classic test case
- Likely issue: the implicit rewrite still contains strong cues but matching fails
- Action: review all failed "obvious" entities, check if prediction is actually correct but doesn't match gold label
- Expected impact: +2-3% (fix false negatives in evaluation)

---

## B. PROMPT IMPROVEMENTS (Expected Impact: +5-15% Hit@1)

### B1. Drop the contextualization step [Validated]
- Ablation showed: no_context IMPROVES Hit@1 by +1% and is 37% faster
- The background generation adds noise
- Action: make direct inference the default
- Expected impact: +1% Hit@1, -37% latency, -50% API cost

### B2. Better inference prompt with structured output
- Current prompt: "What [type] is implicitly described? Give top 3 guesses."
- Problem: LLM returns verbose descriptions instead of entity names
- Fix: "List exactly 3 entity names, one per line. No descriptions."
- Also try: JSON output format `{"guesses": ["entity1", "entity2", "entity3"]}`
- Expected impact: +3-5% (fewer parsing failures)

### B3. Few-shot examples in prompt
- Include 3 examples of (implicit text -> entity name) in the prompt
- Choose examples that cover Person, Place, Event
- Teaches the model the expected output format AND the task
- Expected impact: +3-5% Hit@1

### B4. Entity candidate list in prompt (closed-set recognition)
- Instead of open generation, provide the full entity list (532 entities)
- Ask: "Which of these entities is described? Pick top 3."
- Converts open-ended generation to closed-set ranking
- Dramatically reduces hallucination
- Expected impact: +10-15% Hit@1 (biggest single improvement)
- Downside: prompt becomes very long; may need chunking

### B5. Chain-of-thought for hard cases
- CoT ablation not yet run; expected to help on complex multi-hop references
- "Let's think step by step: What clues are in the text? What entity matches?"
- Expected impact: +2-5% on hard cases

---

## C. METHOD IMPROVEMENTS (Expected Impact: +10-25% Hit@1)

### C1. Entity descriptions for embedding (RC4) [Ready, descriptions generated]
- Current: compare text embedding to entity NAME embedding (terrible: 0.027 Hit@1)
- Fix: compare text embedding to entity DESCRIPTION embedding
- 431 descriptions already generated via LLM
- Expected impact: 3-5x improvement for embedding method (0.027 -> 0.10+)

### C2. Two-stage RAG with entity descriptions
- Stage 1: embed text, find top-20 nearest entity DESCRIPTIONS
- Stage 2: LLM re-ranks from those 20 candidates
- Combines C1 improvement with LLM reasoning
- Expected impact: +5-10% over current hybrid

### C3. Ensemble: LLM + Hybrid vote
- LLM and Hybrid disagree on 106 samples where Hybrid wins
- Simple majority vote or union of top-3 from both methods
- Can compute from existing predictions (no new API calls)
- Expected impact: +3-5% Hit@1

### C4. Iterative refinement
- If LLM returns low-confidence guess, retry with different prompt
- Or: ask LLM "Is [prediction] a good match for this text? If not, try again."
- Expected impact: +2-3% Hit@1

### C5. Use stronger models
- Llama 3.1 8B already beats Gemini Flash (0.382 vs 0.356)
- GPT-4o-mini via OpenAI Batch (50% off) may do better
- Claude 3.5 Haiku likely best quality
- Expected impact: +5-10% from model upgrade alone

---

## D. EVALUATION IMPROVEMENTS (Expected Impact: +5-10% apparent Hit@1)

### D1. Fix false negatives in matching
- "Statue of Liberty" gold vs "The Statue of Liberty" prediction = fail under current matching
- Add article-stripping normalization (remove the/a/an)
- Add common abbreviation handling
- Expected impact: +2-3% Hit@1

### D2. Multi-reference evaluation
- Some samples have multiple valid answers (Pearl Harbor text could be "Pearl Harbor" or "WWII")
- Allow 2-3 gold references per sample
- Expected impact: +3-5% Hit@1

### D3. Semantic matching via LLM
- Current alias matching asks "Do X and Y refer to the same entity?"
- But it runs ONLY on top-3 predictions
- Extend to top-10 predictions
- Expected impact: +1-2% Hit@1

---

## PRIORITY ORDER (bang for buck)

| Priority | Improvement | Effort | Expected Impact | Cost |
|---|---|---|---|---|
| 1 | A1: Use v2 cleaned data | 5 min | +5-10% | $0.05 |
| 2 | B4: Closed-set entity list | 30 min | +10-15% | $0.10 |
| 3 | C3: Ensemble from existing | 15 min | +3-5% | $0 |
| 4 | B1: Drop context step | 0 min | +1% | (already shown) |
| 5 | B2: Structured output prompt | 15 min | +3-5% | $0.05 |
| 6 | C5: Stronger model (GPT-4o-mini) | 20 min | +5-10% | $0.15 |
| 7 | C1+C2: Entity description embeddings | 30 min | +5-10% | $0 |
| 8 | D1: Better matching normalization | 15 min | +2-3% | $0 |
| 9 | B3: Few-shot examples | 15 min | +3-5% | $0.05 |
| 10 | A2: Core types only | 5 min | +3-5% | $0.05 |

**Total estimated cost for top 10 improvements: ~$0.45**
**Total estimated impact: +25-40% Hit@1 on veterans data**
