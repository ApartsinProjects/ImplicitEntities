# Review Section 3: Dataset Generation and Synthesis Opportunities

**Reviewer Role:** Top-tier AI/NLP journal reviewer (ACL/EMNLP/NAACL caliber)

These recommendations address how to generate, synthesize, or augment datasets to strengthen the empirical foundation.

---

## Top 20 Recommendations

### 1. Generate adversarial implicit references
Use an LLM to create maximally difficult implicit descriptions: ones that could plausibly refer to multiple entities. For example, "the island nation that was devastated by a tsunami" could be Japan, Indonesia, or Sri Lanka. Adversarial samples test method robustness and reveal disambiguation weaknesses.

### 2. Create a multi-reference gold standard
Current datasets have exactly one gold entity per sample. Generate datasets where 2-3 valid entities are annotated per sample (primary and acceptable alternatives), enabling evaluation with relaxed matching. Use multiple human annotators to establish this.

### 3. Synthesize datasets in other languages
Use the same entity lists to generate implicit narratives in Spanish, Hebrew, Arabic, and Mandarin. This tests whether the methodology transfers across languages and cultural contexts, crucial for a paper targeting elderly populations worldwide.

### 4. Generate graded difficulty levels
For each entity, create 3 implicit descriptions at varying difficulty: (a) easy (one distinctive cue, e.g., "the tower in Paris"), (b) medium (cultural reference, e.g., "the iron lattice glowing at dusk"), (c) hard (emotional/sensory only, e.g., "we sat on the grass, splitting a baguette, watching it glow"). This enables difficulty-stratified evaluation.

### 5. Create entity-pair datasets for shared context detection
The downstream goal is connecting people with shared memories. Generate pairs of implicit narratives that reference the same entity from different perspectives. Evaluate whether the system can detect that two texts reference the same implicit entity, which is the actual deployment task.

### 6. Augment with real elderly narratives from oral history archives
StoryCorps, the Veterans History Project (Library of Congress), and the USC Shoah Foundation have thousands of transcribed interviews. Extract segments with implicit references and annotate them. This addresses the single-source limitation of 25 interviews.

### 7. Generate temporal-context datasets
Create implicit references that require temporal reasoning: "the year everything changed in the Pacific" (1941) vs. "when the wall came down" (1989). Add temporal metadata to entities and evaluate whether methods can disambiguate time-dependent references.

### 8. Synthesize cross-cultural entity equivalents
Some entities have cultural equivalents: "Remembrance Day" (UK) vs. "Veterans Day" (US) vs. "ANZAC Day" (Australia). Generate datasets where implicit references could map to culture-specific variants, testing cultural awareness.

### 9. Generate negative samples (no implicit entity)
Current datasets assume every text contains an implicit entity. Create samples that describe generic experiences without referencing any specific entity (e.g., "I remember walking to school in the rain"). Methods should return an empty list or low-confidence scores. This tests precision and false-positive rates.

### 10. Create a benchmark with entity type ambiguity
Generate texts where the entity type is ambiguous: "the gathering that drew millions" could be an Event (Woodstock) or a Place (Washington Mall during the March). Remove entity type conditioning to test type-agnostic recognition.

### 11. Synthesize datasets from different narrative styles
Generate implicit narratives in: (a) formal memoir style, (b) conversational dialect, (c) fragmented notes, (d) poetic/literary style. This tests robustness to register variation, critical since real elderly narratives span all these styles.

### 12. Generate datasets with deliberate name leakage at varying levels
Create a continuum from fully implicit to partially explicit: (a) "the iron tower in the French capital" (fully implicit), (b) "that tower on the Champ de Mars" (partial cue), (c) "the Eiffel structure" (near-explicit). This calibrates method sensitivity to information density.

### 13. Augment Twitter dataset with more entity types
The current Twitter dataset is 100% "Work" type in the implicit set. Generate balanced tweet datasets with equal representation of Person, Place, Event, Organization, and Work entities to enable fair cross-type comparison.

### 14. Create synthetic interview transcripts using persona-based generation
Use LLMs with detailed persona prompts (e.g., "You are a 85-year-old Korean War veteran from rural Ohio") to generate full interview transcripts. Extract implicit references from these. This scales data generation while controlling demographic variables.

### 15. Build a knowledge-grounded evaluation set
For each entity, compile a knowledge card (Wikipedia summary, key facts, dates). Generate implicit references that require specific knowledge facts. Evaluate whether RAG methods with access to knowledge cards outperform methods without, quantifying the value of grounding.

### 16. Synthesize datasets with entity evolution over time
Some entities change: "Bombay" became "Mumbai," "Peking" became "Beijing," "the twin towers" shifted meaning after 2001. Generate references using outdated terminology and test whether methods can resolve to modern canonical forms.

### 17. Create a human evaluation benchmark
Recruit 20+ annotators (ideally including elderly participants) to: (a) write their own implicit references to given entities, (b) guess entities from synthetic implicit references, (c) rate difficulty. This provides human performance baselines and inter-annotator agreement.

### 18. Generate multi-hop implicit references
Create references requiring 2+ reasoning steps: "the school my mother attended in the city where they make that famous cheese" requires resolving mother's school + cheese city (e.g., Cheddar, England). Multi-hop references test deeper reasoning capabilities.

### 19. Augment with social media data beyond Twitter
Scrape Reddit r/AskOldPeople, r/nostalgia, or Facebook reminiscence groups (with ethical approval) for naturally occurring implicit references. These provide diverse, authentic implicit entity mentions at scale without generation artifacts.

### 20. Create a longitudinal dataset simulating daily narrative accumulation
The paper's motivation mentions "individuals potentially adding short narratives daily." Generate a simulated 30-day narrative stream for 10 personas, where implicit references accumulate context over time. Test whether methods improve when given access to a user's full narrative history vs. individual segments in isolation. This directly tests the motivating use case.
