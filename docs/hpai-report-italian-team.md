# HPAI in LiteRev-Evidence: findings for the Italian team

Prepared for the Italian partner team. State of production on 2026-10-12, scenario HPAI_last (`usr-b1e16a5d9c38`), live at https://literev-scenario.com/?scenario=usr-b1e16a5d9c38. The team receives the same material as a shared document with a user guide; this file is the repository's record of it. It replaces the version of 8 October, which described the earlier scenario `usr-4757684f5462` (640 articles, threshold 0.30). Updated on 12 October after an audit of every cap and fixed value in LiteRev: the search was run again, the relevance score now reads the title and abstract, and a work found under several DOIs counts once.

## 1. In one page

- **Search** (12 October 2026, exhaustive mode, 11 databases): 34,626 records identified (28,123 from the databases, 6,503 from LiteRev's library), 15,447 duplicates removed (the same article, or the same work under another DOI), 1,669 removed for not matching the full query, 17,510 screened, 1,820 of them without an abstract (kept, screened by title).
- **Relevance**: cosine similarity of the title and abstract to the query, threshold 0.5201 (the same number of articles as the 10 October selection, among those both corpora share): 1,075 relevant articles, 426 with a full text. No human screening yet.
- **Extraction**: 1,072 of 1,075 relevant articles read (426 from the full text, 646 from the abstract), 14,908 values, 93% of them with their quote found in the text. 3 articles are not extracted. 0 values reviewed by a person.
- **Pooled estimates**: 45 labels reach the 3 studies needed; nearly all are very heterogeneous.
- **Geography**: 562 of 1,072 extracted articles resolve to a country, 41 countries in all; Italy has 19.
- **Evidence brief**: low certainty, grade C; 6 claims, all rated low; 23 references, the 2025 Microorganisms review by De Marco, Cotti and colleagues being reference 7.
- **Rerank** (Cohere): 97 of 17,510 articles scored. The Cohere trial key has used its 1,000 calls for the month; the scoring panel now says so (PR #337).

## 2. The search, database by database

| Database | Records kept | Database's own total | How it was searched |
| --- | --- | --- | --- |
| Europe PMC | 13,200 | 13,604 | Full query, all matches |
| OpenAlex | 4,565 | 3,738 in titles and abstracts | All title and abstract matches, plus its 2,000 best full-text matches |
| Semantic Scholar | 3,280 | 3,343 | Full query, all matches (returned unranked) |
| CORE | 2,022 | 2,032 | Full query, all matches |
| Crossref | 1,967 | 2,635,306 | Keywords only: its 2,000 most relevant, the only capped source |
| PubMed | 1,900 | 1,914 | Full query, MeSH terms or the same words in free text |
| DOAJ | 514 | not reported | Full query split into parts |
| Europe PMC preprints | 459 | 459 | Full query on the preprint collection |
| OpenAIRE | 165 | not reported | Shortened query (4 operators at most), records re-checked against the full query |
| ClinicalTrials.gov | 49 | 49 | Full query |
| arXiv | 2 | 2 | Full query |

Small gaps between kept and total are records without a usable title. About 30 of the relevant articles are preprints (about 430 in the whole corpus). A preprint, its versions and the figures deposited on figshare count as the published article (PR #350).

## 3. What the extracted articles report

Counts are out of the 1,072 extracted articles.

| Topic | Articles | Share | From the full text |
| --- | --- | --- | --- |
| Animal hosts | 685 | 64% | 313 |
| Environment | 599 | 56% | 302 |
| Occupation | 622 | 58% | 269 |
| Human testing | 476 | 44% | 221 |
| Age | 339 | 32% | 204 |
| Vaccination | 309 | 29% | 164 |
| Protective equipment | 275 | 26% | 157 |
| Sex or gender | 229 | 21% | 153 |
| Knowledge, attitudes, practices, risk perception | 310 | 29% | 116 |
| Vectors | 30 | 3% | 21 |

| Sheet | Values | Articles | Quote found |
| --- | --- | --- | --- |
| Human exposure | 5,690 | 772 | 93% |
| Animals and reservoirs | 5,080 | 673 | 93% |
| Human susceptibility | 2,162 | 481 | 93% |
| Environment | 1,955 | 501 | 94% |
| Vectors | 21 | 17 | 76% |

## 4. Pooled estimates: highlights and cautions

| Measure (codebook label) | Disease as written | Studies | Pooled share (95% CI) | I2 |
| --- | --- | --- | --- | --- |
| Seropositive people (immune status) | hpai h5n1 | 6 | 0.4% (0.1% to 1.7%) | 83% |
| Knowledge of avian influenza (KAP) | avian influenza | 6 | 47% (23% to 72%) | 98% |
| Direct contact (contact type) | hpai h5n1 | 4 | 44% (1.1% to 98%) | 99% |
| Partial PPE use (PPE) | hpai h5n1 | 3 | 30% (0.1% to 99.2%) | 99% |

- The disease is free text, so one infection is pooled separately under each spelling: human H5N1 seropositivity sits in 14 pools, 28 studies in all. A read-time disease map is the fix (PR #341, open).
- "Comparisons between groups" often set two shares of one sample side by side (for example 119 men and 81 women of the same 200 farmers): that is the sample's composition, not a risk (PR #339, open).
- A study still counts twice when a database misprints its DOI by one character: Yu 2013 (poultry workers, China) sits twice in the H9N2 male and female comparison. A reviewer excludes the second record.
- The panel lists up to 120 labels and says when there are more; its CSV holds all 2,580.

## 5. The Dressler paper, field by field

Dressler et al., Euro Surveill 2026;31(17):2600293 (doi 10.2807/1560-7917.ES.2026.31.17.2600293), article 475185, ranks 327th of the 1,075 relevant articles (similarity 0.551, read from its title and abstract). LiteRev read its full text (24,319 characters) on 8 October and recorded 22 values, all with their quote found; 9 map to the default codebook. Compared with the manual T4.5 extraction shared on 8 October:

| T4.5 slot | Manual reading | LiteRev's extraction | Verdict |
| --- | --- | --- | --- |
| Pathogen hazard | Low biosecurity raises the hazard; shelter staff wore surgical masks instead of FFP2 or FFP3 respirators | No biosecurity at the holding; PPE per group (none for the holder and police, none or gloves for veterinary staff early, nobody before 18 February, one veterinarian without PPE); shelter staff "full PPE" (Table 4) | Partial: the respirator class is missing and "full PPE" overstates the shelter's protection |
| Host and reservoir | 21 chickens, 9 free-roaming cats; serology available | Poultry 4 dead PCR positive, 17 culled negative; cats 3 dead or euthanised PCR positive, 6 alive with 3 PCR positive, all 6 H5-seropositive | Complete and richer; misses that the cat test was validated on ferret sera only |
| Environment | Remote rural area, near habitats of wild reservoirs | Remote rural holding; outdoor access and wild-bird contact; outdoor feeding; house sealed for 4 months | Complete and richer; the habitat point is not in the paper |
| Population | 17 exposed, by profession, PPE, influenza vaccination and seroprevalence | 5 groups summing to 17, each with exposure, PPE and vaccination | Substantial; misses the human serology (0 of 11 with H5 antibodies) and two negative tests, which the PICO card states |
| Mobility and contact | Direct contact with infected birds, secretions, contaminated materials or environments | Contact type per group; the holder's row maps to "direct contact" | Substantial; the definition of a contact and the day-by-day timeline are not rows |
| Exposure pathways | Poultry house (backyard?) with outdoor access and wild-bird contact; spillover to cats | Wild-bird contact, outdoor feeding (mapped to "backyard"), cat infections, occupational exposure | Partial: no chain from wild birds to poultry, cats and people; the predation route is missed |

## 6. Where the evidence comes from

562 of 1,072 extracted articles resolve to a country: China (125), United States (68), Bangladesh (36), United Kingdom (35), Indonesia (32). Italy has 19 articles (217 values); regions: Campania ITF3 (1), Emilia-Romagna ITH5 (1). Of the other 510, 202 name no place; the rest name a continent, several countries, or a place the resolver does not know yet (Hong Kong, Taiwan, Chinese cities). Only the built-in NUTS list is loaded (Germany NUTS 1, Italy NUTS 1 and 2).

## 7. What is left

- **Blocked on files from the team**: the Annex 2 label hierarchy (codebook import relabels every row with no new AI call), the Eurostat GISCO NUTS file, and the extraction template itself, to check the Excel export against it.
- **Development**: typed transmission relations for T4.6; health system capacity (no sheet yet); a way for a reviewer to add a row the model missed; a disease map for pooling; a guard against composition comparisons; more places in the resolver; accuracy of the extraction never measured against a gold standard.
- **Operations**: a Cohere production key, so the rerank can score the whole corpus.
- **Robustness**: a source that fails during a new search should keep its previous records instead of shrinking the corpus (a Europe PMC outage on 12 October did so for a few minutes; queued as a task).

## 8. Validation on the team's side

- [ ] Query: an information specialist reviews the terms (H7N7, H7N3, H5N2 and H10N3 are not named).
- [ ] Known papers: 10 to 20 papers that must be in, looked up by DOI or PMID in the corpus search.
- [ ] Threshold: about 100 articles just below the cut screened by hand.
- [ ] Screening: two reviewers in Double-Blind on the relevant set, kappa of 0.6 or more on a first batch.
- [ ] Full texts: the 649 relevant articles without one, retrieved through the library.
- [ ] Extracted data: two reviewers, one article in ten at first, starting with the values whose quote was not found; the Dressler shelter PPE row corrected first.
- [ ] Pooled figures: none quoted before review; diseases kept apart; each forest plot opened.
- [ ] Brief: each claim checked against its cited articles; the R0 values (7 studies, 0.27 to 4.6) mix animal and human transmission.
- [ ] Preprints: decide whether the 30 or so in the relevant set stay.
- [ ] Duplicates: exclude a second record of one study when a database misprinted its DOI.
- [ ] Reporting: flow numbers from the PRISMA card (15,447 duplicates removed) and the search date, 12 October 2026.
