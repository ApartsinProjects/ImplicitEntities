# IRC-Bench v5: Implicit Entity Recognition Dataset

A benchmark for Implicit Entity Recognition (IER) from oral history transcripts.
Given a first-person narrative that implicitly references a named entity (without
naming it), the task is to identify which entity is being referenced.

## Directory Structure

```
data/
  benchmark/            Experiment-ready splits (the dataset)
  pipeline/             Source data and intermediate processing stages
    transcripts/        Stage 1: cleaned oral history transcripts
    entities_extracted/ Stage 2: named entities per transcript
    summaries/          Stage 3: explicit entity-focused summaries
    implicit/           Stage 4: implicit rewrites (entity removed)
  provenance/           Data source metadata
```

## Benchmark (`benchmark/`)

| File | Description |
|------|-------------|
| `irc_bench_v5.csv` | Full dataset (all partitions) |
| `irc_bench_v5_train.csv/json` | Training split |
| `irc_bench_v5_dev.csv/json` | Development split |
| `irc_bench_v5_test.csv/json` | Test split |
| `entity_kb.json` | Entity knowledge base |
| `entity_list_{train,dev,test}.txt` | Entity names per partition |
| `split_metadata.json` | Split statistics and configuration |

### Benchmark Statistics

| | Samples | Entities |
|----------|--------:|--------:|
| **Train** | 17,971 | 8,631 |
| **Dev** | 2,532 | 1,233 |
| **Test** | 4,633 | 2,473 |
| **Total** | 25,136 | 12,337 |

Entity-level split: zero entity overlap between train, dev, and test.

### Entity Type Distribution

| Type | Samples | Percentage |
|------|--------:|-----------:|
| Place | 11,893 | 47.3% |
| Organization | 5,366 | 21.3% |
| Person | 3,450 | 13.7% |
| Event | 2,162 | 8.6% |
| Work | 1,195 | 4.8% |
| Military Unit | 537 | 2.1% |
| Other | 533 | 2.1% |

### Entity Knowledge Base

The `entity_kb.json` file contains 12,337 entities with:
- 84.6% have Wikipedia URLs
- 70.9% have LLM-generated descriptions
- 51.2% have aliases

### CSV Columns

| Column | Description |
|--------|-------------|
| `uid` | Stable unique ID (MD5-based) |
| `partition` | train, dev, or test |
| `entity` | Gold entity name |
| `entity_type` | Place, Organization, Person, Event, Work, Military_Unit |
| `entity_qid` | Wikidata QID |
| `entity_aliases` | Pipe-separated aliases |
| `entity_description` | LLM-generated entity description |
| `entity_description_wiki` | Wikipedia first sentence |
| `entity_wikipedia_url` | Wikipedia URL |
| `explicit_text` | First-person narrative mentioning the entity by name |
| `implicit_text` | Same narrative with the entity name removed |
| `cues` | Pipe-separated contextual cues |
| `transcript_ref` | Source transcript path |
| `collection` | Oral history collection name |

## Pipeline (`pipeline/`)

The benchmark was constructed through a four-stage pipeline:

### Stage 1: Transcripts (1,994 YAML files)

Cleaned oral history transcripts from 11 thematic collections:

| Collection | Transcripts |
|------------|------------:|
| veterans | 517 |
| immigration | 402 |
| regional | 314 |
| depression_era | 213 |
| japanese_american | 156 |
| academic | 153 |
| september_11 | 72 |
| civil_rights | 68 |
| covid_19 | 42 |
| labor | 30 |
| refugee | 27 |

### Stage 2: Entities Extracted (1,752 JSON files)

NER via GPT-4.1-mini on each transcript. Each file lists named entities with
Wikidata linking, surface forms, and Wikipedia URLs.

Total entity mentions across all transcripts: 31,284.

### Stage 3: Explicit Summaries (1,601 JSON files)

For each (transcript, entity) pair, GPT generates a first-person narrative
focused on that entity, preserving contextual cues.

Total explicit summaries: 25,161.

### Stage 4: Implicit Rewrites (1,600 JSON files)

Each explicit summary is rewritten to remove the entity name while preserving
all contextual cues. These become the benchmark's `implicit_text` field.

Total implicit rewrites: 25,136.

### Pipeline Coverage

- 87.9% of transcripts have extracted entities
- 80.3% have explicit summaries
- 80.2% have implicit rewrites
- Some transcripts lack entities due to insufficient content or processing errors

## Regeneration

To rebuild the benchmark from pipeline data:

```bash
cd experiments
python build_splits.py
```

This reads `pipeline/transcripts/**/*.implicit.json`, collects all samples,
performs entity-level splitting (seed=42), and writes all files to `benchmark/`.
