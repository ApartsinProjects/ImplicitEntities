# Reference Validation Report: article_v4.html

**Date:** 2026-04-18
**Validated by:** Claude Opus 4.6 (automated web search verification)
**Total references:** 43 ([1] through [43])

## Summary

- **43 references** in the bibliography, numbered [1] through [43], sequential with no gaps
- **All 43** are now cited at least once in the body text (after fix)
- **No duplicate references** found
- **2 references had fabricated/incorrect author lists** (corrected)
- **1 reference was an orphan** (never cited in body; citation added)

## Issues Found and Corrected

### CRITICAL: Reference [11] had wrong authors and wrong title

**Before (INCORRECT):**
> Xie, T., Li, J., Sun, Z., Li, N., Rich, K., and Glass, J. (2023). An empirical study on ChatGPT for zero-shot NER. *arXiv preprint arXiv:2310.10035*.

**After (CORRECTED):**
> Xie, T., Li, Q., Zhang, J., Zhang, Y., Liu, Z., and Wang, H. (2023). Empirical study of zero-shot NER with ChatGPT. In *Proc. EMNLP*, pages 7935-7956.

**What was wrong:**
- Authors were completely fabricated. The real authors are Tingyu Xie, Qi Li, Jian Zhang, Yan Zhang, Zuozhu Liu, and Hongwei Wang (not "Li, J., Sun, Z., Li, N., Rich, K., and Glass, J.").
- Title was slightly wrong: "An empirical study on ChatGPT for zero-shot NER" vs the correct "Empirical study of zero-shot NER with ChatGPT".
- The paper was also published at EMNLP 2023, not just an arXiv preprint; venue updated.

### CRITICAL: Reference [19] had wrong authors

**Before (INCORRECT):**
> Ayoola, T., Shivade, C., Mukherjee, T., Zhu, Y., and Korycinski, D. (2022). ReFinED: An efficient zero-shot-capable approach to end-to-end entity linking. In *Proc. NAACL*.

**After (CORRECTED):**
> Ayoola, T., Tyagi, S., Fisher, J., Christodoulopoulos, C., and Pierleoni, A. (2022). ReFinED: An efficient zero-shot-capable approach to end-to-end entity linking. In *Proc. NAACL (Industry Track)*.

**What was wrong:**
- Four out of five co-authors were fabricated. The real authors are Tom Ayoola, Shubhi Tyagi, Joseph Fisher, Christos Christodoulopoulos, and Andrea Pierleoni (not "Shivade, C., Mukherjee, T., Zhu, Y., and Korycinski, D.").
- Venue clarified as Industry Track.

### MINOR: Reference [35] was never cited in body text

**Before:** Reference [35] (Touvron et al., 2023, Llama 2) existed in the bibliography but was never cited in the body.

**Fix:** Added citation [35] alongside [36] where Llama 3.1 8B Instruct is first introduced in Section 4.3.2 (Models), since Llama 3.1 builds on the Llama 2 architecture. The text now reads "Llama 3.1 8B Instruct [35, 36]".

## Verified References (No Issues)

The following 40 references were verified via web search and found to be correct in all fields (authors, title, year, venue, pages where listed):

| Ref | Status | Notes |
|-----|--------|-------|
| [1] Nadeau & Sekine (2007) | OK | Lingvisticae Investigationes 30(1):3-26 |
| [2] Li, Sun, Han, Li (2022) | OK | IEEE TKDE 34(1):50-70 |
| [3] Ganea & Hofmann (2017) | OK | EMNLP, pages 2619-2629 |
| [4] Kolitsas, Ganea, Hofmann (2018) | OK | CoNLL, pages 519-529 |
| [5] Lee, He, Lewis, Zettlemoyer (2017) | OK | EMNLP, pages 188-197 |
| [6] Boyd (2012) | OK | Oxford Handbook of Oral History (print 2010, online 2012) |
| [7] Lazar, Demiris, Thompson (2016) | OK | Informatics for Health and Social Care 41(4):373-389 |
| [8] Subramaniam & Woods (2012) | OK | Expert Review of Neurotherapeutics 12(5):545-555 |
| [9] Lample et al. (2016) | OK | NAACL, pages 260-270 |
| [10] Devlin et al. (2019) | OK | NAACL, pages 4171-4186 |
| [12] Ashok & Lipton (2023) | OK | arXiv:2305.15444 |
| [13] Sang & De Meulder (2003) | OK | CoNLL, pages 142-147 |
| [14] Malmasi et al. (2022) | OK | COLING |
| [15] Li, Fei, Liu et al. (2022) | OK | AAAI |
| [16] Zhou et al. (2024) | OK | ICLR |
| [17] Wu et al. (2020) | OK | EMNLP, pages 6397-6407 |
| [18] De Cao et al. (2021) | OK | ICLR |
| [20] Botha, Shan, Gillick (2020) | OK | EMNLP, pages 7833-7845 |
| [21] Hosseini (2022) | OK | PhD thesis, Toronto Metropolitan University |
| [22] Hosseini & Bagheri (2021) | OK | Information Processing & Management 58(3):102503 |
| [23] Perera, Dehmer, Emmert-Streib (2020) | OK | Frontiers in Cell and Developmental Biology 8:673 |
| [24] Treder, Lee, Tsvetanov (2024) | OK | Frontiers in Dementia 3:1385303 |
| [25] Broadbent, Stafford, MacDonald (2009) | OK | Int. Journal of Social Robotics 1(4):319-330 |
| [26] de Jager et al. (2017) | OK | The Qualitative Report 22(10):2548-2582 |
| [27] Yang et al. (2018) | OK | EMNLP |
| [28] Petroni et al. (2019) | OK | EMNLP |
| [29] Lewis et al. (2020) | OK | NeurIPS |
| [30] Karpukhin et al. (2020) | OK | EMNLP, pages 6769-6781 |
| [31] Xiao et al. (2023) | OK | arXiv:2309.07597 |
| [32] Wei et al. (2022) | OK | NeurIPS |
| [33] Hu et al. (2022) | OK | ICLR |
| [34] Dettmers et al. (2023) | OK | NeurIPS |
| [35] Touvron et al. (2023) | OK | arXiv:2307.09288 |
| [36] Dubey et al. (2024) | OK | arXiv:2407.21783 |
| [37] OpenAI (2024) | OK | GPT-4o System Card (also arXiv:2410.21276) |
| [38] Butler (1963) | OK | Psychiatry 26(1):65-76 |
| [39] Webster (1993) | OK | Journal of Gerontology 48(5):P256-P262 |
| [40] Nikitina, Callaioli, Baez (2018) | OK | SE4COG Workshop, pages 52-57 |
| [41] Pessanha & Akdag Salah (2022) | OK | ACM JOCCH 15(1):6:1-6:16 |
| [42] Hou (2020) | OK | ACL, pages 1428-1438 |
| [43] Poesio, Stuckardt, Versley (2016) | OK | Springer book |

## Structural Checks

- [x] References numbered [1] through [43] sequentially, no gaps
- [x] Every bibliography entry is cited at least once in the body text
- [x] Every in-text citation has a corresponding bibliography entry
- [x] No duplicate references
- [x] No fabricated references (after corrections)
