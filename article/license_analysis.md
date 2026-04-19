# IRC-Bench License and Copyright Analysis

**Prepared:** 2026-04-18
**Purpose:** Assess whether IRC-Bench can be released publicly for a NeurIPS 2026 Evaluations & Datasets Track submission.

---

## 1. Per-Source License Analysis

### 1.1 Library of Congress: Veterans History Project (LOC VHP)

| Field | Detail |
|---|---|
| **Institution** | Library of Congress, American Folklife Center |
| **URL** | https://www.loc.gov/collections/veterans-history-project-collection/ |
| **Collections used** | `loc_vhp`, `loc_vhp_xml` (88 transcripts, 561 samples) |
| **Copyright holder** | Veterans and interviewers retain copyright; materials contributed for "scholarly and educational purposes" |
| **License type** | No standard license; permission-based |
| **Redistribution** | Restricted. "Permission must be obtained before using the interview or other materials in exhibition or publication." Written permission from the interviewee (or next-of-kin if deceased) is required for duplication beyond personal use or fair use. |
| **Derived works** | Not explicitly addressed. Fair use likely applies to annotations and entity labels derived from publicly accessible content. |
| **Attribution** | Required: cite Library of Congress Veterans History Project |
| **Risk level** | **MEDIUM-HIGH.** Full transcript redistribution requires individual permissions. Releasing only annotations (entity spans, labels) without full text is likely defensible under fair use, but full-text redistribution is not permitted without consent. |

### 1.2 Library of Congress: Federal Writers' Project (FWP)

| Field | Detail |
|---|---|
| **Institution** | Library of Congress |
| **URL** | https://www.loc.gov/collections/federal-writers-project/ |
| **Collections used** | `federal_writers_project` (96 transcripts, 284 samples) |
| **Copyright holder** | U.S. Government (Works Progress Administration / WPA) |
| **License type** | **Public domain** (U.S. Government work) |
| **Redistribution** | **Unrestricted.** LOC states: "The Library of Congress is not aware of any copyright in the documents in this collection." Created by federal employees as part of the WPA Federal Writers' Project. |
| **Derived works** | Permitted without restriction |
| **Attribution** | Courtesy citation to LOC recommended |
| **Caveats** | Privacy/publicity rights of interviewees may apply, though subjects are from the 1930s (all deceased). |
| **Risk level** | **LOW.** Clear public domain status. |

### 1.3 Library of Congress: Civil Rights History Project

| Field | Detail |
|---|---|
| **Institution** | Library of Congress, American Folklife Center; Smithsonian NMAAHC |
| **URL** | https://www.loc.gov/collections/civil-rights-history-project/ |
| **Collections used** | `civil_rights` (62 transcripts, 324 samples) |
| **Copyright holder** | Mixed. LOC/Smithsonian hold copyright for many interviews via release forms. Some materials from the Civil Rights Movement Archive (crmvet.org) are jointly owned by CRMA Inc. and contributors. |
| **License type** | LOC portion: no standard license, permission-based. CRMA portion: all rights reserved, non-commercial use only with attribution. |
| **Redistribution** | LOC portion: "Duplication of collection materials may be governed by copyright and other restrictions." CRMA portion: explicitly prohibits commercial redistribution; non-commercial use requires attribution. |
| **Derived works** | Not explicitly permitted for LOC portion. CRMA prohibits derivative works for commercial purposes without written authorization. |
| **Attribution** | Required for both LOC and CRMA sources |
| **Risk level** | **MEDIUM-HIGH.** Mixed provenance. LOC interviews conducted by government employees may be public domain as government works, but interviewees retain some rights. CRMA materials have explicit restrictions. |

### 1.4 Library of Congress: Influenza/Pandemic Oral Histories

| Field | Detail |
|---|---|
| **Institution** | Library of Congress (via Internet Archive) |
| **URL** | Various Internet Archive items |
| **Collections used** | `loc_influenza` (29 transcripts, 94 samples) |
| **Copyright holder** | Mixed; some are University of Washington Labor Archives COVID-era oral histories uploaded to Internet Archive |
| **License type** | Varies per item; no uniform license |
| **Redistribution** | Individual assessment needed per item |
| **Risk level** | **MEDIUM.** Items on Internet Archive vary in licensing. Many are educational/research materials uploaded by institutions. |

### 1.5 Densho Digital Archive (Japanese American Oral Histories)

| Field | Detail |
|---|---|
| **Institution** | Densho: The Japanese American Legacy Project |
| **URL** | https://ddr.densho.org/ |
| **Collections used** | `densho` (102 transcripts, 490 samples), sourced via Internet Archive |
| **Copyright holder** | Densho and partner institutions |
| **License type** | **CC BY-NC-SA 4.0** for most materials in the Densho Digital Repository |
| **Redistribution** | Permitted for non-commercial purposes with attribution and share-alike. |
| **Derived works** | Permitted under CC BY-NC-SA 4.0 (must share under same license, non-commercial only) |
| **Attribution** | Required: "Courtesy of Densho" plus specific collection credit |
| **Caveats** | Materials accessed via Internet Archive cached copies, not directly from DDR. Some items may have different rights. The SA (share-alike) clause means derived datasets must also be CC BY-NC-SA. |
| **Risk level** | **LOW-MEDIUM.** CC BY-NC-SA 4.0 is compatible with academic/research release. The non-commercial and share-alike clauses will constrain the overall benchmark license. |

### 1.6 University of Nevada, Reno (UNR) Oral History Program

| Field | Detail |
|---|---|
| **Institution** | University of Nevada, Reno Libraries, Special Collections |
| **URL** | https://library.unr.edu/places/knowledge-center/special-collections/collections/audiovisual/oral-histories |
| **Collections used** | `unr_oral_history` (22 transcripts, 92 samples) |
| **Copyright holder** | University of Nevada, Reno Libraries |
| **License type** | Restricted; transcripts "may be downloaded and/or printed for personal reference and educational use, but not sold or used in commercial products without permission." Archival metadata is CC BY 4.0. |
| **Redistribution** | Not permitted without permission for full transcripts |
| **Derived works** | Not explicitly addressed |
| **Risk level** | **MEDIUM.** Educational use permitted; full redistribution restricted. Annotations-only release may qualify under educational/fair use. |

### 1.7 University of Nevada, Reno: WWII Veterans Project

| Field | Detail |
|---|---|
| **Institution** | University of Nevada, Reno (via Internet Archive) |
| **URL** | https://archive.org/details/WWIIVeteransProject |
| **Collections used** | `nevada_wwii` (328 transcripts, 1411 samples; the largest single collection) |
| **Copyright holder** | University of Nevada Oral History Program |
| **License type** | Same as UNR above: personal/educational use permitted |
| **Redistribution** | Restricted for full transcripts |
| **Derived works** | Not explicitly addressed |
| **Risk level** | **MEDIUM.** Large collection. Same restrictions as UNR above. |

### 1.8 Louie B. Nunn Center for Oral History (Kentucky)

| Field | Detail |
|---|---|
| **Institution** | University of Kentucky Libraries, Louie B. Nunn Center |
| **URL** | https://kentuckyoralhistory.org/ |
| **Collections used** | `kentucky_nunn` (68 transcripts, 269 samples) |
| **Copyright holder** | University of Kentucky Libraries. "All rights to the interviews, including but not restricted to legal title, copyrights and literary property rights, have been transferred to the University of Kentucky Libraries." |
| **License type** | Proprietary; permission-based, no standard open license |
| **Redistribution** | "Interviews may only be reproduced with permission from Louie B. Nunn Center for Oral History." |
| **Derived works** | Not explicitly addressed |
| **Risk level** | **HIGH.** Explicit reproduction restriction. Permission required even for partial reproduction. |

### 1.9 UCLA Center for Oral History Research

| Field | Detail |
|---|---|
| **Institution** | UCLA Library, University of California |
| **URL** | https://oac.cdlib.org/findaid/ark:/13030/kt129033hb/ |
| **Collections used** | `ucla_oral_history` (104 transcripts, 532 samples) |
| **Copyright holder** | The Regents of the University of California |
| **License type** | All rights reserved. Transcripts carry explicit restrictions: "All literary rights in the manuscript, including the right to publication, are reserved to the University Library of the University of California, Los Angeles. No part of the manuscript may be quoted for publication without the written permission of the University Librarian." |
| **Redistribution** | **Prohibited without written permission.** Materials on Internet Archive are "digitized with permission of the Regents of the University of California" (for IA access only). |
| **Derived works** | Requires written permission from University Librarian |
| **Risk level** | **HIGH.** Explicit, strong copyright reservation. Cannot redistribute without formal permission from UCLA. |

### 1.10 Columbia University Oral History Archives

| Field | Detail |
|---|---|
| **Institution** | Columbia University Libraries, Rare Book & Manuscript Library |
| **URL** | https://library.columbia.edu/libraries/ccoh.html |
| **Collections used** | `columbia_oral_history` (16 transcripts, 88 samples) |
| **Copyright holder** | Columbia University; some interviews have rights retained by narrators |
| **License type** | No standard open license. "Columbia no longer requires permission to cite or quote from its collections," but reproduction rights depend on "contractual obligations to narrators and copyright law." |
| **Redistribution** | Restricted. "The archives do not reproduce entire collections for remote research use." |
| **Derived works** | Quoting/citing is permitted; full reproduction is restricted |
| **Risk level** | **MEDIUM-HIGH.** Quoting permitted but full transcript redistribution restricted. |

### 1.11 Smithsonian Archives of American Art

| Field | Detail |
|---|---|
| **Institution** | Smithsonian Institution, Archives of American Art |
| **URL** | https://www.aaa.si.edu/ |
| **Collections used** | `smithsonian_art` (22 transcripts, 103 samples) |
| **Copyright holder** | Mixed. Smithsonian Open Access items are CC0 (public domain). Others may be under third-party copyright. |
| **License type** | Smithsonian Open Access (CC0) for some items; others require individual assessment |
| **Redistribution** | CC0 items: unrestricted. Others: "It is the sole responsibility of the applicant to determine whether any such rights exist." |
| **Derived works** | Permitted for CC0 items |
| **Risk level** | **MEDIUM.** Need to verify which items are in Open Access (CC0) vs. restricted. |

### 1.12 National Park Service: September 11 Oral History Project

| Field | Detail |
|---|---|
| **Institution** | National Park Service (U.S. Department of Interior) |
| **URL** | https://home.nps.gov/articles/000/september-11-2001-nps-oral-history-project.htm |
| **Collections used** | `nps_911` (70 transcripts, 344 samples) |
| **Copyright holder** | National Park Service (U.S. Government) |
| **License type** | **Public domain** (U.S. Government work). Metadata explicitly states: "Public domain (US Government work)" |
| **Redistribution** | **Unrestricted.** NPS interviews are conducted by government employees; copyright has been transferred to NPS via signed release forms. |
| **Derived works** | Permitted without restriction |
| **Attribution** | Courtesy citation to NPS recommended |
| **Risk level** | **LOW.** Clear public domain status as U.S. Government work. |

### 1.13 National Park Service: Ellis Island Immigration Oral Histories

| Field | Detail |
|---|---|
| **Institution** | National Park Service, Ellis Island |
| **URL** | https://www.nps.gov/elis/learn/historyculture/oral-histories.htm |
| **Collections used** | Part of `immigration` collection (short excerpts) |
| **Copyright holder** | National Park Service (U.S. Government) |
| **License type** | **Public domain** (U.S. Government work) |
| **Redistribution** | **Unrestricted** |
| **Risk level** | **LOW.** |

### 1.14 UNC Southern Oral History Program (DocSouth)

| Field | Detail |
|---|---|
| **Institution** | University of North Carolina at Chapel Hill |
| **URL** | https://docsouth.unc.edu/sohp/ |
| **Collections used** | `unc_sohp` (97 transcripts, 476 samples) |
| **Copyright holder** | University of North Carolina (for most unrestricted interviews); some retained by interviewers/interviewees |
| **License type** | Educational fair use encouraged. "Material on this site may be quoted or reproduced for private purposes or student or instructor use in a classroom or classroom assignment without prior permission as long as a proper citation is given." Commercial use prohibited without permission. |
| **Redistribution** | Classroom/educational use permitted without prior permission. Commercial use requires permission. |
| **Derived works** | Quoting/reproducing for educational purposes is permitted with citation |
| **Risk level** | **MEDIUM.** Educational use explicitly permitted. Dataset release for research purposes may qualify, but full-text redistribution for commercial use is prohibited. |

### 1.15 Voices of Oklahoma

| Field | Detail |
|---|---|
| **Institution** | Voices of Oklahoma (501(c)(3) non-profit); now archived at Oklahoma Historical Society |
| **URL** | https://voicesofoklahoma.com/ |
| **Collections used** | `voices_of_oklahoma` (69 transcripts, 306 samples) |
| **Copyright holder** | Voices of Oklahoma Inc. |
| **License type** | Standard copyright (All Rights Reserved). No Creative Commons or open license found. |
| **Redistribution** | No explicit terms found. General copyright applies. |
| **Derived works** | Not explicitly addressed |
| **Risk level** | **HIGH.** No open license; standard copyright with no explicit permission for redistribution or derived works. |

### 1.16 COVID-19 Oral History Project (National Humanities Center)

| Field | Detail |
|---|---|
| **Institution** | National Humanities Center, Research Triangle, NC |
| **URL** | https://covidoralhistory.org/ |
| **Collections used** | `covid_oral_history` (42 transcripts, 142 samples) |
| **Copyright holder** | National Humanities Center |
| **License type** | **CC BY 4.0** (Creative Commons Attribution 4.0 International) |
| **Redistribution** | **Permitted** with attribution |
| **Derived works** | **Permitted** with attribution |
| **Attribution** | Required: credit National Humanities Center COVID-19 Oral History Project |
| **Risk level** | **LOW.** Clear, permissive open license. |

### 1.17 Princeton RFMI (Refugee Oral Histories)

| Field | Detail |
|---|---|
| **Institution** | Princeton University, Religion and Forced Migration Initiative |
| **URL** | https://www.rfmi.princeton.edu/archive |
| **Collections used** | `refugee_archive` (22 transcripts, 108 samples) |
| **Copyright holder** | Princeton University |
| **License type** | Not explicitly stated. Transcripts shared via public Google Docs links. |
| **Redistribution** | Not explicitly addressed. Public availability suggests educational/research use intended. |
| **Derived works** | Not explicitly addressed |
| **Risk level** | **MEDIUM.** No explicit license. Public availability is encouraging, but formal permission may be needed. |

### 1.18 University of Minnesota Immigrant Stories

| Field | Detail |
|---|---|
| **Institution** | University of Minnesota Libraries |
| **URL** | https://umedia.lib.umn.edu/ |
| **Collections used** | Part of `immigration` collection |
| **Copyright holder** | University of Minnesota |
| **License type** | Not explicitly determined from available metadata |
| **Risk level** | **MEDIUM.** |

### 1.19 Niles-Maine District Library (Veterans)

| Field | Detail |
|---|---|
| **Institution** | Niles-Maine District Library, Niles, IL |
| **URL** | https://www.nileslibrary.org/research-learn/more-resources/veterans-history-project |
| **Collections used** | `niles_library` (52 transcripts, 301 samples) |
| **Copyright holder** | Veterans and interviewers (contributed to Library of Congress VHP and local library) |
| **License type** | Same as LOC VHP: permission-based |
| **Redistribution** | Restricted. Same terms as LOC VHP. |
| **Risk level** | **MEDIUM-HIGH.** |

### 1.20 Indian Prairie Public Library (IPPL) Veterans

| Field | Detail |
|---|---|
| **Institution** | Indian Prairie Public Library, Darien, IL |
| **URL** | https://ippl.info/learn-research/veterans-history |
| **Collections used** | `ippl_library` (12 transcripts, 56 samples) |
| **Copyright holder** | Veterans and interviewers |
| **License type** | Same as LOC VHP: permission-based |
| **Risk level** | **MEDIUM-HIGH.** |

### 1.21 Wisconsin Veterans Museum

| Field | Detail |
|---|---|
| **Institution** | Wisconsin Veterans Museum |
| **URL** | https://wisvetsmuseum.com/learning-research/oral-history/ |
| **Collections used** | `wisconsin_veterans_museum` (17 transcripts, 106 samples) |
| **Copyright holder** | Wisconsin Veterans Museum (for most interviews) |
| **License type** | "Education and research use is permitted." Copying prohibited without WVM permission. |
| **Redistribution** | Restricted. "Patrons may NOT make copies of the interviews without permission from WVM." |
| **Risk level** | **MEDIUM-HIGH.** |

### 1.22 University of Washington Labor Archives

| Field | Detail |
|---|---|
| **Institution** | University of Washington Libraries, Special Collections |
| **URL** | https://lib.uw.edu/specialcollections/laws/ |
| **Collections used** | `labor_history` (20 transcripts, 88 samples) |
| **Copyright holder** | University of Washington |
| **License type** | Permission-based; contact required for reproduction |
| **Redistribution** | Restricted. Permission required from UW Libraries Special Collections. |
| **Risk level** | **MEDIUM-HIGH.** |

### 1.23 Disasters Collection (Mixed Internet Archive sources)

| Field | Detail |
|---|---|
| **Institution** | Various (UC Berkeley Bancroft Library, CSUSB, others via Internet Archive) |
| **URL** | Various Internet Archive URLs |
| **Collections used** | `disasters` (26 transcripts, 103 samples) |
| **Copyright holder** | Various institutions |
| **License type** | Varies by item; many are university oral history programs with similar restrictions as UCLA |
| **Risk level** | **MEDIUM-HIGH.** Mixed provenance; individual assessment needed. |

### 1.24 Internet Archive (Veterans, Misc.)

| Field | Detail |
|---|---|
| **Institution** | Various (via Internet Archive) |
| **Collections used** | `internet_archive` (1 transcript, 8 samples) |
| **Risk level** | **LOW** (tiny collection). |

### 1.25 Civil Rights Movement Archive (crmvet.org)

| Field | Detail |
|---|---|
| **Institution** | Civil Rights Movement Archive Inc. (CRMA) |
| **URL** | https://www.crmvet.org/ |
| **Collections used** | Part of `civil_rights` collection |
| **Copyright holder** | CRMA Inc., Bruce Hartford, and individual contributors |
| **License type** | All rights reserved; non-commercial use with attribution permitted |
| **Redistribution** | Commercial redistribution prohibited without written authorization |
| **Risk level** | **MEDIUM.** Non-commercial academic use with attribution may be acceptable, but formal permission recommended. |

---

## 2. Overall Assessment: Can We Release IRC-Bench?

### Summary Table

| Risk Level | Collections | Sample Count (approx.) |
|---|---|---|
| **LOW** (public domain or CC-licensed) | FWP, NPS 9/11, Ellis Island NPS, COVID-19 NHC (CC BY 4.0) | ~632 |
| **LOW-MEDIUM** (CC BY-NC-SA) | Densho | ~490 |
| **MEDIUM** (educational use permitted, gray area) | UNR, Nevada WWII, UNC SOHP, Smithsonian, RFMI, UMN, LOC Influenza | ~2,277 |
| **MEDIUM-HIGH** (permission required) | LOC VHP, Civil Rights, Columbia, Niles, IPPL, WVM, UW Labor, Disasters | ~1,686 |
| **HIGH** (explicit restrictions) | UCLA, Kentucky Nunn, Voices of Oklahoma | ~1,107 |

### Key Findings

1. **Full-text redistribution is NOT feasible** for the majority of sources without obtaining permissions. Only ~1,122 samples (FWP, NPS 9/11, Ellis Island, COVID-19 NHC, Densho) come from sources with clear open licenses or public domain status.

2. **An annotations-only release IS feasible** under a fair use theory. The benchmark consists of entity annotations (entity mention spans, entity types, Wikidata links) derived from the transcripts. Releasing only the annotations, metadata, and download scripts (not the full transcript text) is a well-established pattern in NLP research for copyrighted data.

3. **The Densho CC BY-NC-SA 4.0 license constrains the overall dataset license** if Densho materials are included as full text. Any derivative work must also be CC BY-NC-SA (non-commercial, share-alike).

### Release Strategy Options

**Option A: Annotations-Only Release (RECOMMENDED)**

Release IRC-Bench as:
- Entity annotations (character offsets, entity types, Wikidata IDs)
- Metadata (source collection, interview ID, themes)
- Download scripts that users run to fetch original transcripts from their public sources
- Pre-computed evaluation metrics and baselines

This is the standard approach for NLP benchmarks built on copyrighted text (similar to how SQuAD distributes question-answer pairs with Wikipedia paragraph IDs, or how many dialogue benchmarks provide annotation layers separate from underlying text).

**Option B: Mixed Release (PARTIAL)**

Release full text only for clearly open-licensed sources:
- Federal Writers' Project (public domain): 284 samples
- NPS September 11 (public domain): 344 samples
- NPS Ellis Island (public domain): portion of immigration samples
- National Humanities Center COVID-19 (CC BY 4.0): 142 samples
- Densho (CC BY-NC-SA 4.0): 490 samples

For all other sources, release annotations-only with download scripts.

**Option C: Seek Permissions (IDEAL but time-consuming)**

Contact each institution to request formal permission for dataset redistribution for non-commercial research purposes. Many oral history programs have granted such permissions in the past. Priority targets:
1. UCLA (532 samples, explicit restrictions)
2. Kentucky Nunn Center (269 samples, explicit restrictions)
3. University of Nevada, Reno (1,503 samples combined, large portion of dataset)
4. Voices of Oklahoma (306 samples, no explicit license)

---

## 3. Recommended License for IRC-Bench

### If annotations-only (Option A):
**CC BY-NC-SA 4.0** (to honor the Densho share-alike requirement)

Alternatively, if Densho annotations can be separated:
**CC BY 4.0** for the annotation layer, with Densho-derived annotations separately under CC BY-NC-SA 4.0.

### If mixed release (Option B):
**CC BY-NC-SA 4.0** for the entire package (driven by Densho's license)

### Rationale:
- CC BY-NC-SA 4.0 is compatible with all open-licensed sources in the dataset
- The non-commercial clause respects the educational/research intent of most source institutions
- NeurIPS accepts CC BY-NC-SA datasets (it is among the most common licenses in accepted papers)
- The share-alike clause is required by Densho's license terms

---

## 4. Required Attributions

The following attributions MUST be included in any release:

1. **Library of Congress, Veterans History Project** (American Folklife Center)
2. **Library of Congress, Federal Writers' Project** (WPA Life Histories)
3. **Library of Congress, Civil Rights History Project** (American Folklife Center & Smithsonian NMAAHC)
4. **Densho: The Japanese American Legacy Project** (CC BY-NC-SA 4.0)
5. **National Park Service, September 11, 2001 Oral History Project**
6. **National Park Service, Ellis Island Immigration Museum**
7. **National Humanities Center, COVID-19 Oral History Project** (CC BY 4.0)
8. **University of Nevada, Reno, Oral History Program**
9. **Louie B. Nunn Center for Oral History, University of Kentucky Libraries**
10. **UCLA Library Center for Oral History Research, University of California**
11. **Columbia University, Oral History Archives**
12. **Smithsonian Institution, Archives of American Art**
13. **University of North Carolina at Chapel Hill, Southern Oral History Program (DocSouth)**
14. **Voices of Oklahoma**
15. **Princeton University, Religion and Forced Migration Initiative**
16. **University of Minnesota Libraries, Immigrant Stories**
17. **Niles-Maine District Library, Veterans History Project**
18. **Indian Prairie Public Library, Veterans History Project**
19. **Wisconsin Veterans Museum**
20. **University of Washington, Labor Archives of Washington**
21. **Civil Rights Movement Archive (crmvet.org)**
22. **Internet Archive** (hosting platform for many sources)

---

## 5. Sources That Must Be Excluded or Handled Carefully

### Must exclude from full-text redistribution (unless permission obtained):
1. **UCLA Oral History** (532 samples): Explicit "all literary rights reserved" with publication prohibition
2. **Kentucky Nunn Center** (269 samples): Explicit reproduction restriction
3. **Voices of Oklahoma** (306 samples): Standard copyright, no open license

### Should exclude or seek permission:
4. **Columbia University** (88 samples): Does not reproduce entire collections remotely
5. **Wisconsin Veterans Museum** (106 samples): "Patrons may NOT make copies without permission"

### Can include with annotations-only approach:
All of the above can be included in an annotations-only release with download scripts, as the annotations themselves are original scholarly work.

---

## 6. Risk Assessment

### Legal Risks

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| Copyright infringement claim (full text) | Medium-High (for UCLA, Kentucky, VoOK) | High | Use annotations-only release |
| Fair use challenge (annotations) | Very Low | Low | Annotations are transformative scholarly work |
| Privacy concerns | Low | Medium | All interviews are public records with consenting participants; subjects consented to being recorded |
| Institutional objection to inclusion | Low-Medium | Medium | Include opt-out mechanism; contact institutions pre-publication |
| NeurIPS reviewer concern about licensing | Medium | Medium | Clearly document all licenses; use standard annotations-only pattern |

### NeurIPS-Specific Considerations

1. **Accessibility:** Reviewers must be able to access the dataset. An annotations-only release with download scripts means reviewers would need to run the scripts to reconstruct the full dataset. This adds friction but is an accepted practice.

2. **Hosting:** The annotations and metadata can be hosted on Hugging Face or Dataverse. The download scripts can be included in the repository.

3. **Croissant metadata:** Can be generated for the annotations layer. The underlying transcript sources are already documented with provenance metadata.

4. **Reproducibility:** Download scripts must be robust. Some Internet Archive items may become unavailable. Consider providing SHA-256 checksums so users can verify they have the correct source text.

### Recommended Actions Before Submission

1. **Implement annotations-only release format** with download scripts and reconstruction code
2. **Contact top-priority institutions** (UCLA, Kentucky, UNR) for formal permission letters; even a brief email exchange showing awareness and good faith significantly reduces risk
3. **Add a DATA_LICENSE.md** to the repository documenting per-source licenses
4. **Include an opt-out mechanism** for any source institution that objects
5. **Verify all Internet Archive links** are still active
6. **For the paper:** Include a "Licensing and Ethical Considerations" section describing the multi-source provenance, the annotations-only distribution approach, and the fair use rationale

### Bottom Line

**IRC-Bench CAN be released for NeurIPS 2026**, provided we use the annotations-only distribution model with download scripts. This is the standard, legally sound approach for NLP benchmarks built on third-party text. The recommended license is **CC BY-NC-SA 4.0** (driven by Densho's share-alike requirement). Full-text redistribution is not feasible without obtaining individual permissions from 10+ institutions.
