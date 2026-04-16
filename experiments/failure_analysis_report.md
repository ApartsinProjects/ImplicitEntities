# Failure Analysis Report

Comprehensive analysis of all 22 prediction files across datasets, methods, and entity types.

## 1. Cross-Method Failure Analysis

### Dataset: e2t_twitter

- Total samples: 769
- All methods fail: 149 (19.4%)
- At least one succeeds: 620 (80.6%)
- **Union ceiling (best ensemble): 80.6%**
- Method accuracy:
  - embedding: 45.1%
  - hybrid: 47.5%
  - llm: 71.8%
- Disagreements between methods: 341

### Dataset: e2t_veterans

- Total samples: 1560
- All methods fail: 644 (41.3%)
- At least one succeeds: 916 (58.7%)
- **Union ceiling (best ensemble): 58.7%**
- Method accuracy:
  - embedding: 29.4%
  - hybrid: 29.7%
  - llm: 45.1%
- Disagreements between methods: 669

### Dataset: twitter

- Total samples: 850
- All methods fail: 42 (4.9%)
- At least one succeeds: 808 (95.1%)
- **Union ceiling (best ensemble): 95.1%**
- Method accuracy:
  - embedding: 62.1%
  - hybrid: 65.2%
  - llm: 91.1%
- Disagreements between methods: 311

### Dataset: veterans_t2e

- Total samples: 846
- All methods fail: 310 (36.6%)
- At least one succeeds: 536 (63.4%)
- **Union ceiling (best ensemble): 63.4%**
- Method accuracy:
  - embedding: 26.7%
  - hybrid: 26.8%
  - llm: 50.8%
- Disagreements between methods: 417

## 2. Prediction Quality Analysis

Categories of failed predictions (pred_1 vs gold):

| Experiment | Failures | Near Match | More Specific | Related | Completely Wrong | Empty |
|---|---|---|---|---|---|---|
| e2t_twitter/embedding/default | 422 | 1 | 0 | 8 | 413 | 0 |
| e2t_twitter/hybrid/default | 404 | 6 | 3 | 13 | 382 | 0 |
| e2t_twitter/llm/default | 217 | 6 | 0 | 7 | 203 | 0 |
| e2t_veterans/embedding/default | 1102 | 7 | 1 | 7 | 1086 | 0 |
| e2t_veterans/hybrid/default | 1097 | 11 | 3 | 21 | 1042 | 0 |
| e2t_veterans/llm/default | 857 | 66 | 44 | 64 | 683 | 0 |
| twitter/embedding/default | 324 | 1 | 0 | 4 | 319 | 0 |
| twitter/hybrid/default | 298 | 2 | 3 | 18 | 275 | 0 |
| twitter/llm/default | 76 | 0 | 1 | 4 | 66 | 0 |
| veterans_t2e/ablation/no_context | 403 | 27 | 20 | 14 | 341 | 0 |
| veterans_t2e/ablation/no_type | 463 | 22 | 33 | 19 | 385 | 0 |
| veterans_t2e/ablation/no_type_no_ctx | 469 | 30 | 31 | 15 | 386 | 0 |
| veterans_t2e/embedding/all-MiniLM-L6-v2 | 620 | 1 | 6 | 4 | 609 | 0 |
| veterans_t2e/embedding/all-mpnet-base-v2 | 603 | 3 | 10 | 7 | 583 | 0 |
| veterans_t2e/embedding/bge-small-en-v1.5 | 627 | 5 | 6 | 4 | 611 | 0 |
| veterans_t2e/embedding/default | 620 | 1 | 6 | 4 | 609 | 0 |
| veterans_t2e/embedding/multi-qa-MiniLM-L6 | 646 | 1 | 5 | 4 | 636 | 0 |
| veterans_t2e/hybrid/default | 619 | 11 | 19 | 4 | 581 | 0 |
| veterans_t2e/llm/default | 416 | 34 | 28 | 22 | 331 | 0 |
| veterans_t2e/llm/llama-3.1-8b | 384 | 9 | 9 | 4 | 361 | 0 |
| veterans_t2e/llm/mistral-7b | 846 | 0 | 0 | 0 | 0 | 846 |

## 3. False Negative Detection

Predictions that SHOULD have matched gold with better normalization:

| Experiment | Failures | False Negatives | Current Acc | Improved Acc | Gain |
|---|---|---|---|---|---|
| e2t_twitter/embedding/default | 422 | 1 | 45.1% | 45.2% | +0.1pp |
| e2t_twitter/hybrid/default | 404 | 3 | 47.5% | 47.9% | +0.4pp |
| e2t_twitter/llm/default | 217 | 3 | 71.8% | 72.2% | +0.4pp |
| e2t_veterans/embedding/default | 1102 | 37 | 29.4% | 31.7% | +2.4pp |
| e2t_veterans/hybrid/default | 1097 | 35 | 29.7% | 31.9% | +2.2pp |
| e2t_veterans/llm/default | 857 | 103 | 45.1% | 51.7% | +6.6pp |
| twitter/embedding/default | 324 | 4 | 62.1% | 62.6% | +0.5pp |
| twitter/hybrid/default | 298 | 4 | 65.2% | 65.7% | +0.5pp |
| twitter/llm/default | 76 | 6 | 91.1% | 91.8% | +0.7pp |
| veterans_t2e/ablation/no_context | 403 | 40 | 52.4% | 57.1% | +4.7pp |
| veterans_t2e/ablation/no_type | 463 | 53 | 45.3% | 51.5% | +6.3pp |
| veterans_t2e/ablation/no_type_no_ctx | 469 | 61 | 44.6% | 51.8% | +7.2pp |
| veterans_t2e/embedding/all-MiniLM-L6-v2 | 620 | 50 | 26.7% | 32.6% | +5.9pp |
| veterans_t2e/embedding/all-mpnet-base-v2 | 603 | 41 | 28.7% | 33.6% | +4.8pp |
| veterans_t2e/embedding/bge-small-en-v1.5 | 627 | 41 | 25.9% | 30.7% | +4.8pp |
| veterans_t2e/embedding/default | 620 | 50 | 26.7% | 32.6% | +5.9pp |
| veterans_t2e/embedding/multi-qa-MiniLM-L6 | 646 | 44 | 23.6% | 28.8% | +5.2pp |
| veterans_t2e/hybrid/default | 619 | 50 | 26.8% | 32.7% | +5.9pp |
| veterans_t2e/llm/default | 416 | 57 | 50.8% | 57.6% | +6.7pp |
| veterans_t2e/llm/llama-3.1-8b | 384 | 20 | 54.6% | 57.0% | +2.4pp |
| veterans_t2e/llm/mistral-7b | 846 | 0 | 0.0% | 0.0% | +0.0pp |

## 4. Entity Difficulty Ranking

- Total unique entities: 797
- Always easy (>80% success): 89
- Sometimes hard (30-80%): 419
- Always hard (<30%): 289

### Hardest Entities (min 3 attempts)

| Entity | Type | Success Rate | Attempts |
|---|---|---|---|
| birth certificate | events | 0% | 33 |
| infiltration courses | professions | 0% | 33 |
| military training | professions | 0% | 22 |
| 16,000 men | people | 0% | 21 |
| 331st infantry | organizations | 0% | 21 |
| machine gun fire | events | 0% | 21 |
| central europe | places | 0% | 21 |
| women | people | 0% | 21 |
| surgery | professions | 0% | 21 |
| spec 5 | professions | 0% | 21 |
| the other boy | people | 0% | 21 |
| his mother | people | 0% | 21 |
| ba airman | people | 0% | 21 |
| reservoirs | places | 0% | 21 |
| 147 | people | 0% | 21 |
| people | people | 0% | 21 |
| Horns 28novel29 | Book | 0% | 15 |
| Stone Mattress | Book | 0% | 15 |
| Wish I Was Here | Movie | 0% | 12 |
| i | people | 2% | 97 |

### Success Rate by Entity Type

| Type | Matched | Total | Rate |
|---|---|---|---|
| Founder | 15 | 60 | 25.0% |
| Leader | 6 | 24 | 25.0% |
| Model | 3 | 12 | 25.0% |
| Comedian | 3 | 12 | 25.0% |
| Co-founder | 5 | 18 | 27.8% |
| CEO | 16 | 54 | 29.6% |
| Awards | 9 | 27 | 33.3% |
| Entrepreneur | 8 | 24 | 33.3% |
| ReligiousEvent | 6 | 18 | 33.3% |
| people | 808 | 2360 | 34.2% |
| events | 931 | 2710 | 34.4% |
| professions | 1041 | 3007 | 34.6% |
| places | 1703 | 4886 | 34.9% |
| organizations | 716 | 2050 | 34.9% |
| Executive | 11 | 27 | 40.7% |
| Writer | 89 | 201 | 44.3% |
| MusiciansFromHawaii | 7 | 15 | 46.7% |
| Book | 308 | 645 | 47.8% |
| Rapper | 6 | 12 | 50.0% |
| Lufkin_Industries | 6 | 12 | 50.0% |
| Band | 25 | 48 | 52.1% |
| Movie | 367 | 696 | 52.7% |
| MusicalPerformer | 10 | 18 | 55.6% |
| Author | 27 | 48 | 56.2% |
| Actor | 84 | 144 | 58.3% |
| Athlete | 29 | 48 | 60.4% |
| SocietalEvent | 19 | 30 | 63.3% |
| BasketballTeam | 60 | 93 | 64.5% |
| MusicGroup | 37 | 57 | 64.9% |
| Software | 12 | 18 | 66.7% |
| Charity | 12 | 18 | 66.7% |
| ArchitecturalStructure | 59 | 87 | 67.8% |
| Country | 225 | 327 | 68.8% |
| Royalty | 63 | 90 | 70.0% |
| VicePresident | 15 | 21 | 71.4% |
| Company | 73 | 102 | 71.6% |
| OfficeHolder | 55 | 75 | 73.3% |
| ReligiousBuilding | 55 | 75 | 73.3% |
| Businessperson | 58 | 78 | 74.4% |
| Event | 32 | 42 | 76.2% |
| Building | 39 | 51 | 76.5% |
| Organisation | 131 | 171 | 76.6% |
| MusicalArtist | 92 | 120 | 76.7% |
| PrimeMinister | 28 | 36 | 77.8% |
| SportsEvent | 21 | 27 | 77.8% |
| HistoricalPlace | 14 | 18 | 77.8% |
| SoccerClub | 47 | 60 | 78.3% |
| City | 137 | 171 | 80.1% |
| Public_company | 34 | 42 | 81.0% |
| HockeyTeam | 74 | 90 | 82.2% |
| PoliticalParty | 148 | 177 | 83.6% |
| SoccerTournament | 26 | 30 | 86.7% |
| Director | 42 | 48 | 87.5% |
| President | 66 | 75 | 88.0% |
| CricketTeam | 61 | 69 | 88.4% |
| CellularTelephone | 83 | 93 | 89.2% |
| Scientist | 46 | 51 | 90.2% |
| Instrument | 22 | 24 | 91.7% |
| Cricketer | 21 | 21 | 100.0% |
| Skyscraper | 18 | 18 | 100.0% |
| MusicalInstrument | 12 | 12 | 100.0% |
| BaseballTeam | 24 | 24 | 100.0% |

## 5. Recommendations

1. NORMALIZATION IMPROVEMENT (avg +3.5pp): Improve entity matching with better normalization. Key issues: article stripping (the/a/an), substring containment matching, spelling variant handling. This is the lowest-hanging fruit.

2. ENSEMBLE FOR e2t_twitter (+8.8pp): Union ceiling is 80.6% vs best single method at 71.8%. An ensemble or voting strategy combining methods could capture this gap.

3. ENSEMBLE FOR e2t_veterans (+13.7pp): Union ceiling is 58.7% vs best single method at 45.1%. An ensemble or voting strategy combining methods could capture this gap.

4. ENSEMBLE FOR twitter (+3.9pp): Union ceiling is 95.1% vs best single method at 91.1%. An ensemble or voting strategy combining methods could capture this gap.

5. ENSEMBLE FOR veterans_t2e (+12.5pp): Union ceiling is 63.4% vs best single method at 50.8%. An ensemble or voting strategy combining methods could capture this gap.

6. FAILURE BREAKDOWN: Of 11513 failures, 2.1% are near-matches, 2.0% are more-specific, 2.1% are related, and 86.0% are completely wrong. Focus on the 14.0% that show partial understanding.

7. HARDEST ENTITY TYPE: 'Founder' at 25.0% success. Consider type-specific prompting or specialized entity databases for this category.

8. HARD FLOOR: 1145/4025 samples (28.4%) fail ALL methods. These represent the genuine difficulty ceiling of the task. Many are ambiguous texts or extremely obscure entities.

9. REALISTIC CEILING ESTIMATE: Current best single method ~91.1%. With normalization fixes (+~3.5pp) and ensemble methods (+~5pp), realistic ceiling is approximately 100%. The remaining gap requires fundamentally different approaches (knowledge graphs, retrieval augmentation).
