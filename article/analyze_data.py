"""
Comprehensive analysis of the Twitter Explicit and Implicit Entity datasets.
Computes distributions, overlaps, and explores synthetic data generation potential.
"""

import csv
import os
from collections import Counter, defaultdict

DATA_DIR = r"E:\Projects\ImplicitEntities\data"
EXPLICIT_PATH = os.path.join(DATA_DIR, "twitter_explicit_dataset (1).csv")
IMPLICIT_PATH = os.path.join(DATA_DIR, "twitter_implicit_dataset (1).csv")


def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def simple_tokenize(text):
    return text.split()


def pct(count, total):
    return f"{count / total * 100:.1f}%" if total > 0 else "0.0%"


def print_distribution(counter, label, top_n=None):
    total = sum(counter.values())
    items = counter.most_common(top_n)
    print(f"\n  {label} (total={total}):")
    max_label_len = max((len(str(k)) for k, _ in items), default=0)
    for k, v in items:
        bar = "#" * max(1, int(v / total * 50))
        print(f"    {str(k):<{max_label_len}}  {v:>5}  ({pct(v, total):>6})  {bar}")
    if top_n and len(counter) > top_n:
        print(f"    ... and {len(counter) - top_n} more categories")


def analyze_dataset(rows, name):
    print("=" * 80)
    print(f"  DATASET: {name}")
    print(f"  Total rows: {len(rows)}")
    print("=" * 80)

    # --- Basic counts ---
    texts = [r["text"] for r in rows]
    entities = [r["entity"] for r in rows]
    unique_entities = set(entities)
    unique_texts = set(texts)

    print(f"\n  Unique tweets: {len(unique_texts)}")
    print(f"  Unique entities: {len(unique_entities)}")

    # --- Text length stats ---
    char_lengths = [len(t) for t in texts]
    token_lengths = [len(simple_tokenize(t)) for t in texts]
    avg_chars = sum(char_lengths) / len(char_lengths) if char_lengths else 0
    avg_tokens = sum(token_lengths) / len(token_lengths) if token_lengths else 0
    min_chars = min(char_lengths) if char_lengths else 0
    max_chars = max(char_lengths) if char_lengths else 0
    min_tokens = min(token_lengths) if token_lengths else 0
    max_tokens = max(token_lengths) if token_lengths else 0

    print(f"\n  Text length (chars): avg={avg_chars:.1f}, min={min_chars}, max={max_chars}")
    print(f"  Text length (tokens): avg={avg_tokens:.1f}, min={min_tokens}, max={max_tokens}")

    # --- Entity class distribution ---
    class_counter = Counter(r["class"] for r in rows)
    print_distribution(class_counter, "Entity CLASS distribution")

    # --- Entity subclass distribution ---
    subclass_counter = Counter(r["subclass"] for r in rows)
    print_distribution(subclass_counter, "Entity SUBCLASS distribution (top 20)", top_n=20)

    # --- NER type distribution ---
    ner_counter = Counter(r["ner_type"] for r in rows)
    print_distribution(ner_counter, "NER TYPE distribution")

    # --- Entities per class (samples) ---
    class_entities = defaultdict(set)
    for r in rows:
        class_entities[r["class"]].add(r["entity"])
    print(f"\n  Sample entities per class:")
    for cls in sorted(class_entities.keys()):
        ents = sorted(class_entities[cls])
        sample = ents[:8]
        print(f"    {cls} ({len(ents)} unique): {', '.join(sample)}")
        if len(ents) > 8:
            print(f"      ... and {len(ents) - 8} more")

    # --- Entities per subclass (samples) ---
    subclass_entities = defaultdict(set)
    for r in rows:
        subclass_entities[r["subclass"]].add(r["entity"])
    print(f"\n  Sample entities per subclass (top 15):")
    for sc, _ in subclass_counter.most_common(15):
        ents = sorted(subclass_entities[sc])
        sample = ents[:6]
        print(f"    {sc} ({len(ents)} unique): {', '.join(sample)}")

    # --- Source distribution ---
    source_counter = Counter(r["source"] for r in rows)
    print_distribution(source_counter, "SOURCE distribution")

    # --- Origin field ---
    origin_counter = Counter(r.get("origin", "") for r in rows)
    non_empty_origins = {k: v for k, v in origin_counter.items() if k.strip()}
    if non_empty_origins:
        print(f"\n  Non-empty ORIGIN values: {len(non_empty_origins)}")
        for k, v in sorted(non_empty_origins.items(), key=lambda x: -x[1])[:10]:
            print(f"    '{k}': {v}")
    else:
        print(f"\n  All ORIGIN values are empty.")

    # --- start_char / end_char analysis ---
    start_chars = [int(r["start_char"]) for r in rows if r["start_char"].lstrip("-").isdigit()]
    end_chars = [int(r["end_char"]) for r in rows if r["end_char"].lstrip("-").isdigit()]
    neg_start = sum(1 for s in start_chars if s < 0)
    neg_end = sum(1 for e in end_chars if e < 0)
    print(f"\n  Character span analysis:")
    print(f"    Rows with start_char < 0 (implicit/missing span): {neg_start} ({pct(neg_start, len(rows))})")
    print(f"    Rows with end_char < 0: {neg_end} ({pct(neg_end, len(rows))})")
    if start_chars:
        valid_spans = [(s, e) for s, e in zip(start_chars, end_chars) if s >= 0 and e >= 0]
        if valid_spans:
            span_lengths = [e - s for s, e in valid_spans]
            avg_span = sum(span_lengths) / len(span_lengths)
            print(f"    Valid spans: {len(valid_spans)}, avg span length: {avg_span:.1f} chars")

    # --- Tweets with multiple entities ---
    tweets_entities = defaultdict(list)
    for r in rows:
        tweets_entities[r["text"]].append(r["entity"])
    multi_entity_tweets = {t: ents for t, ents in tweets_entities.items() if len(ents) > 1}
    print(f"\n  Tweets with multiple entity annotations: {len(multi_entity_tweets)}")
    if multi_entity_tweets:
        for t, ents in list(multi_entity_tweets.items())[:3]:
            print(f"    Entities: {ents}")
            print(f"    Tweet: {t[:100]}...")

    return {
        "rows": rows,
        "unique_entities": unique_entities,
        "class_counter": class_counter,
        "subclass_counter": subclass_counter,
        "ner_counter": ner_counter,
        "class_entities": class_entities,
        "subclass_entities": subclass_entities,
    }


def cross_dataset_analysis(exp_data, imp_data):
    print("\n" + "=" * 80)
    print("  CROSS-DATASET ANALYSIS")
    print("=" * 80)

    exp_ents = exp_data["unique_entities"]
    imp_ents = imp_data["unique_entities"]

    overlap = exp_ents & imp_ents
    only_explicit = exp_ents - imp_ents
    only_implicit = imp_ents - exp_ents

    print(f"\n  Unique entities in EXPLICIT: {len(exp_ents)}")
    print(f"  Unique entities in IMPLICIT: {len(imp_ents)}")
    print(f"  Entities in BOTH datasets:   {len(overlap)}")
    print(f"  Only in EXPLICIT:            {len(only_explicit)}")
    print(f"  Only in IMPLICIT:            {len(only_implicit)}")

    if overlap:
        print(f"\n  Overlapping entities ({len(overlap)}):")
        for e in sorted(overlap):
            print(f"    - {e}")

    if only_implicit:
        print(f"\n  Entities ONLY in implicit dataset ({len(only_implicit)}):")
        for e in sorted(only_implicit):
            print(f"    - {e}")

    # --- Class distribution comparison ---
    print(f"\n  Class distribution comparison:")
    all_classes = sorted(set(exp_data["class_counter"].keys()) | set(imp_data["class_counter"].keys()))
    exp_total = sum(exp_data["class_counter"].values())
    imp_total = sum(imp_data["class_counter"].values())
    print(f"    {'Class':<20} {'Explicit':>10} {'Exp%':>8} {'Implicit':>10} {'Imp%':>8}")
    print(f"    {'-'*56}")
    for cls in all_classes:
        ec = exp_data["class_counter"].get(cls, 0)
        ic = imp_data["class_counter"].get(cls, 0)
        print(f"    {cls:<20} {ec:>10} {pct(ec, exp_total):>8} {ic:>10} {pct(ic, imp_total):>8}")

    # --- Subclass distribution comparison ---
    print(f"\n  Subclass distribution comparison (subclasses in implicit dataset):")
    imp_subclasses = sorted(imp_data["subclass_counter"].keys())
    print(f"    {'Subclass':<25} {'Explicit':>10} {'Exp%':>8} {'Implicit':>10} {'Imp%':>8}")
    print(f"    {'-'*61}")
    for sc in imp_subclasses:
        ec = exp_data["subclass_counter"].get(sc, 0)
        ic = imp_data["subclass_counter"].get(sc, 0)
        print(f"    {sc:<25} {ec:>10} {pct(ec, exp_total):>8} {ic:>10} {pct(ic, imp_total):>8}")

    # --- NER type comparison ---
    print(f"\n  NER type comparison:")
    all_ner = sorted(set(exp_data["ner_counter"].keys()) | set(imp_data["ner_counter"].keys()))
    print(f"    {'NER Type':<20} {'Explicit':>10} {'Exp%':>8} {'Implicit':>10} {'Imp%':>8}")
    print(f"    {'-'*56}")
    for ner in all_ner:
        ec = exp_data["ner_counter"].get(ner, 0)
        ic = imp_data["ner_counter"].get(ner, 0)
        print(f"    {ner:<20} {ec:>10} {pct(ec, exp_total):>8} {ic:>10} {pct(ic, imp_total):>8}")

    # --- For overlapping entities, compare explicit vs implicit tweets ---
    if overlap:
        print(f"\n  For overlapping entities, example explicit vs implicit tweets:")
        exp_by_entity = defaultdict(list)
        imp_by_entity = defaultdict(list)
        for r in exp_data["rows"]:
            if r["entity"] in overlap:
                exp_by_entity[r["entity"]].append(r["text"])
        for r in imp_data["rows"]:
            if r["entity"] in overlap:
                imp_by_entity[r["entity"]].append(r["text"])

        for entity in sorted(overlap)[:5]:
            print(f"\n    Entity: {entity}")
            exp_tweets = exp_by_entity.get(entity, [])
            imp_tweets = imp_by_entity.get(entity, [])
            if exp_tweets:
                print(f"      Explicit ({len(exp_tweets)} tweets):")
                for t in exp_tweets[:2]:
                    print(f"        - {t[:120]}")
            if imp_tweets:
                print(f"      Implicit ({len(imp_tweets)} tweets):")
                for t in imp_tweets[:2]:
                    print(f"        - {t[:120]}")


def synthetic_data_analysis(exp_data, imp_data):
    print("\n" + "=" * 80)
    print("  SYNTHETIC DATA GENERATION POTENTIAL")
    print("=" * 80)

    exp_rows = exp_data["rows"]
    imp_rows = imp_data["rows"]

    # --- 1. Can explicit dataset train implicit entity recognition? ---
    print("""
  1. CAN THE EXPLICIT DATASET TRAIN IMPLICIT ENTITY RECOGNITION?
  ---------------------------------------------------------------
  The explicit dataset has 1721 annotated tweets where entities are directly
  mentioned by name (with character spans). The implicit dataset has 115 tweets
  where entities are referenced indirectly (start_char/end_char = -1).
""")

    # Check how many explicit rows have valid spans
    valid_span_rows = [r for r in exp_rows if int(r["start_char"]) >= 0 and int(r["end_char"]) >= 0]
    print(f"  Explicit rows with valid character spans: {len(valid_span_rows)} / {len(exp_rows)}")

    # Verify span accuracy on a sample
    print(f"\n  Span accuracy check (first 10 with valid spans):")
    correct = 0
    checked = 0
    for r in valid_span_rows[:10]:
        s, e = int(r["start_char"]), int(r["end_char"])
        text = r["text"]
        extracted = text[s:e] if s < len(text) and e <= len(text) else "[OUT OF BOUNDS]"
        entity = r["entity"]
        match = extracted.lower().strip() in entity.lower() or entity.lower() in extracted.lower().strip()
        if match:
            correct += 1
        checked += 1
        status = "OK" if match else "MISMATCH"
        print(f"    [{status}] entity='{entity}', span='{extracted}' (chars {s}:{e})")

    print(f"  Span match rate (sample): {correct}/{checked}")

    # --- 2. Rewriting explicit to implicit ---
    print("""
  2. CAN WE CREATE IMPLICIT REFERENCES BY REWRITING EXPLICIT ONES?
  -----------------------------------------------------------------
  Strategy: For each explicit mention, replace the entity name with a descriptive
  phrase, pronoun, or contextual reference. This requires knowing the entity's
  attributes (class, subclass) and context.
""")

    # Analyze which classes have enough data
    print("  Data volume per class for rewriting:")
    for cls, count in exp_data["class_counter"].most_common():
        imp_count = imp_data["class_counter"].get(cls, 0)
        ent_count = len(exp_data["class_entities"][cls])
        print(f"    {cls}: {count} explicit tweets, {imp_count} implicit tweets, {ent_count} unique entities")

    # Show rewriting examples for overlapping entities
    overlap = exp_data["unique_entities"] & imp_data["unique_entities"]
    if overlap:
        print(f"\n  Rewriting examples from overlapping entities:")
        exp_by_entity = defaultdict(list)
        imp_by_entity = defaultdict(list)
        for r in exp_rows:
            if r["entity"] in overlap:
                exp_by_entity[r["entity"]].append(r)
        for r in imp_rows:
            if r["entity"] in overlap:
                imp_by_entity[r["entity"]].append(r)

        for entity in sorted(overlap)[:3]:
            print(f"\n    Entity: {entity}")
            exp_r = exp_by_entity[entity][0]
            imp_r = imp_by_entity[entity][0]
            print(f"      Class: {exp_r['class']}, Subclass: {exp_r['subclass']}")
            print(f"      EXPLICIT: {exp_r['text'][:120]}")
            print(f"      IMPLICIT: {imp_r['text'][:120]}")
            s, e = int(exp_r["start_char"]), int(exp_r["end_char"])
            if s >= 0 and e > s:
                before = exp_r["text"][:s]
                after = exp_r["text"][e:]
                print(f"      REWRITE TEMPLATE: {before}[IMPLICIT_REF]{after[:60]}")

    # --- 3. Synthetic data generation pipelines ---
    print("""
  3. SYNTHETIC DATA GENERATION PIPELINES
  ----------------------------------------

  Pipeline A: Entity Substitution
    Input:  Explicit tweet + entity metadata
    Method: Replace entity name with a descriptive phrase based on subclass
    Output: Implicit reference tweet
    Example transforms by subclass:
""")

    subclass_transforms = {
        "Actor": "the [nationality] actor from [movie]",
        "Musician": "the [genre] singer/band",
        "Politician": "the [country] [title]",
        "Athlete": "the [sport] [position] for [team]",
        "Movie": "that [director]'s [genre] film",
        "TV Show": "the [network] show about [topic]",
        "Company": "the [industry] giant / tech company",
        "Sports Team": "the [city] [sport] team",
    }
    for sc, template in subclass_transforms.items():
        count = exp_data["subclass_counter"].get(sc, 0)
        print(f"      {sc} ({count} examples): entity -> '{template}'")

    print("""
  Pipeline B: LLM-based Paraphrasing
    Input:  Explicit tweet + entity name + class/subclass
    Method: Prompt an LLM to rewrite the tweet replacing the entity name
            with an indirect reference (description, metonymy, pronoun, etc.)
    Output: Parallel corpus of (explicit, implicit) tweet pairs
    Scale:  Can generate multiple implicit variants per explicit tweet

  Pipeline C: Template-based Generation
    Input:  Entity knowledge base (name, class, subclass, attributes)
    Method: Create tweet templates with slots for implicit references
            Fill slots with entity descriptions from knowledge base
    Output: New synthetic tweets with implicit entity references

  Pipeline D: Back-translation + Entity Masking
    Input:  Explicit tweets
    Method: 1. Mask entity name with [ENTITY] token
            2. Back-translate through another language
            3. Replace [ENTITY] with descriptive phrase
    Output: Diverse implicit reference tweets

  Pipeline E: Contrastive Pair Generation
    Input:  Each explicit tweet
    Method: Generate (explicit, implicit) pairs for contrastive learning
            The explicit version has the entity name; the implicit version
            replaces it with a reference that requires world knowledge
    Output: Training pairs for contrastive NER models
""")

    # --- 4. Feasibility assessment ---
    print("""  4. FEASIBILITY ASSESSMENT
  --------------------------""")

    # How many explicit tweets could be converted?
    convertible = [r for r in exp_rows if int(r["start_char"]) >= 0 and int(r["end_char"]) > 0]
    print(f"  Explicit tweets with valid spans (convertible): {len(convertible)} / {len(exp_rows)}")

    # Coverage by class
    print(f"\n  Convertible tweets by class:")
    conv_by_class = Counter(r["class"] for r in convertible)
    for cls, count in conv_by_class.most_common():
        print(f"    {cls}: {count} tweets")

    # Estimate potential dataset size
    variants_per_tweet = 3  # conservative estimate
    potential_size = len(convertible) * variants_per_tweet
    print(f"\n  If generating {variants_per_tweet} implicit variants per explicit tweet:")
    print(f"    Potential synthetic implicit dataset size: ~{potential_size} tweets")
    print(f"    Current implicit dataset size: {len(imp_rows)} tweets")
    print(f"    Expansion factor: ~{potential_size / len(imp_rows):.0f}x")

    # Quality considerations
    print("""
  Quality Considerations:
    - Overlapping entities provide ground truth for validation
    - Class/subclass metadata enables type-aware rewriting
    - Character spans allow precise entity localization for substitution
    - Twitter text is informal, so synthetic data should preserve that style
    - Need human evaluation to verify implicit references are natural
    - Risk of generating trivially easy implicit references (e.g., just pronouns)
    - Should aim for diverse reference strategies: descriptions, metonymy,
      titles, roles, relationships, achievements, etc.
""")

    # --- 5. Full implicit entity listing ---
    print("  5. ALL UNIQUE ENTITIES IN IMPLICIT DATASET:")
    print("  " + "-" * 50)
    imp_entity_info = defaultdict(lambda: {"class": "", "subclass": "", "count": 0})
    for r in imp_rows:
        info = imp_entity_info[r["entity"]]
        info["class"] = r["class"]
        info["subclass"] = r["subclass"]
        info["count"] += 1

    for entity in sorted(imp_entity_info.keys()):
        info = imp_entity_info[entity]
        print(f"    {entity:<40} {info['class']:<12} {info['subclass']:<20} ({info['count']} tweets)")


def main():
    print("Loading datasets...")
    explicit_rows = load_csv(EXPLICIT_PATH)
    implicit_rows = load_csv(IMPLICIT_PATH)
    print(f"  Explicit: {len(explicit_rows)} rows")
    print(f"  Implicit: {len(implicit_rows)} rows")

    exp_data = analyze_dataset(explicit_rows, "EXPLICIT (twitter_explicit_dataset)")
    print()
    imp_data = analyze_dataset(implicit_rows, "IMPLICIT (twitter_implicit_dataset)")

    cross_dataset_analysis(exp_data, imp_data)
    synthetic_data_analysis(exp_data, imp_data)

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
