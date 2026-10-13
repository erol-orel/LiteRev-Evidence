# HPAI in LiteRev-Evidence: findings for the Italian team

Prepared for the Italian partner team. State of production on 2026-10-13 (figures of 10:08 UTC), scenario HPAI_last (`usr-b1e16a5d9c38`), live at https://literev-scenario.com/?scenario=usr-b1e16a5d9c38. The team receives the same material as a shared document with a user guide; this file is the repository's record of it. It replaces the version of 12 October: the refresh of 12 and 13 October read the full texts of the relevant articles, pooled each infection once, scored every article with Cohere's rerank and placed the studies in Europe's NUTS regions. The search itself is the one of 12 October. HPAI_last sits in the GEOAI4EI folder with the two T2.4 scenarios, susceptibility (`usr-46cfc5a5efa5`) and exposure (`usr-be61e0ad490a`).

## 1. In one page

- **Search** (12 October 2026, exhaustive, 11 databases): 34,626 records identified (28,123 from the databases, 6,503 from LiteRev's library), 15,447 duplicates removed (the same article, or the same work under another DOI), 1,669 removed for not matching the full query, 17,510 screened. 1,755 have no abstract (1,820 at the search; 65 gained one since): they stay in the corpus, screened by title.
- **Relevance**: cosine similarity of the title and abstract to the query, threshold 0.5201: 1,079 relevant articles, 595 with a full text. Cohere's rerank (production key) scored 15,720 of the 15,755 articles with an abstract and orders the list; its threshold is 0, so it removes none. No human screening yet.
- **Extraction**: 1,076 of 1,079 relevant articles read (593 from the full text, 483 from the abstract), 17,489 values, 91.7% of them with their quote found in the text. The other 3 have no text to read (a PLOS table record and two thesis notices). 0 values reviewed by a person.
- **Pooled estimates**: 91 labels reach the 3 studies needed; 73 of them are very heterogeneous (I2 of 90% or more).
- **Geography**: 732 of the 1,076 extracted articles resolve to one country, 67 countries in all; Italy has 20, five of them with a NUTS region.
- **Evidence brief**: low certainty, grade C; 7 claims (3 low, 3 very low, 1 unrated); 18 references.
- **The team's own search** (Rayyan export of 13 October, 1,524 records from PubMed and Embase, 73 included): 68 of the 73 are in LiteRev's corpora, 32 in a relevant set (section 7).

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

Small gaps between kept and total are records without a usable title. About 30 of the relevant articles are preprints (about 430 in the whole corpus). A preprint, its versions and the figures deposited on figshare count as the published article (PR #350). Every LiteRev search is now exhaustive: the standard search, which kept each database's most relevant records, was removed on 13 October.

## 3. What the extracted articles report

Counts are out of the 1,076 extracted articles.

| Topic | Articles | Share | From the full text |
| --- | --- | --- | --- |
| Animal hosts | 697 | 65% | 432 |
| Occupation | 646 | 60% | 363 |
| Environment | 641 | 60% | 423 |
| Human testing | 479 | 45% | 291 |
| Age | 395 | 37% | 294 |
| Vaccination | 340 | 32% | 229 |
| Knowledge, attitudes, practices, risk perception | 316 | 29% | 166 |
| Protective equipment | 307 | 29% | 214 |
| Sex or gender | 272 | 25% | 211 |
| Vectors | 43 | 4% | 36 |

| Sheet | Values | Articles | Quote found |
| --- | --- | --- | --- |
| Human exposure | 6,752 | 781 | 91% |
| Animals and reservoirs | 5,774 | 687 | 92% |
| Human susceptibility | 2,606 | 515 | 92% |
| Environment | 2,321 | 544 | 93% |
| Vectors | 36 | 25 | 81% |

37% of the values carry a label of the default codebook; most of the rest will map once the team's Annex 2 hierarchy is loaded.

## 4. Pooled estimates: highlights and cautions

| Measure (codebook label) | Disease | Studies | Pooled share (95% CI) | I2 |
| --- | --- | --- | --- | --- |
| Seropositive people (immune status) | A(H5N1) | 30 | 2.0% (0.7% to 5.3%) | 95% |
| Seropositive people (immune status) | A(H9N2) | 10 | 3.5% (1.6% to 7.5%) | 99% |
| Knowledge of avian influenza (KAP) | avian influenza | 8 | 59% (38% to 77%) | 98% |
| Direct contact (contact type) | A(H5N1) | 12 | 38% (13% to 73%) | 98% |
| Partial PPE use (PPE) | A(H5N1) | 6 | 22% (3.7% to 66%) | 98% |

- One infection is pooled once however the papers spell it (A(H5N1), HPAI H5N1, H5N1 HPAIV...): human H5N1 seropositivity is one pool of 30 studies, where it was 14 pools under 14 spellings (PR #341).
- A male against female comparison often sets the two shares of one sample side by side (119 men and 81 women of the same 200 farmers): the sample's composition, not a risk. Such pairs are left out of the odds ratio (PR #356): for A(H5N1), 4 are set aside and the 7 studies that remain show no difference, 0.77 (0.41 to 1.44).
- A study still counts twice when a database misprints its DOI by one character: Yu 2013 (poultry workers, China) has two records. A reviewer excludes the second one.
- The panel lists up to 120 labels and says when there are more; its CSV holds all 2,964.

## 5. The Dressler paper, field by field

Dressler et al., Euro Surveill 2026;31(17):2600293 (doi 10.2807/1560-7917.ES.2026.31.17.2600293) ranks 73rd of the 1,079 relevant articles now that Cohere's rerank orders the list (similarity 0.551; it was 327th on similarity alone). LiteRev read its full text (24,319 characters) on 8 October and recorded 22 values, all with their quote found; 9 map to the default codebook. Since the NUTS 2024 list was loaded, its place resolves to Sigmaringen (DE149, Baden-Württemberg). Compared with the manual T4.5 extraction shared on 8 October:

| T4.5 slot | Manual reading | LiteRev's extraction | Verdict |
| --- | --- | --- | --- |
| Pathogen hazard | Low biosecurity raises the hazard; shelter staff wore surgical masks instead of FFP2 or FFP3 respirators | No biosecurity at the holding; PPE per group (none for the holder and police, none or gloves for veterinary staff early, nobody before 18 February, one veterinarian without PPE); shelter staff "full PPE" (Table 4) | Partial: the respirator class is missing and "full PPE" overstates the shelter's protection |
| Host and reservoir | 21 chickens, 9 free-roaming cats; serology available | Poultry 4 dead PCR positive, 17 culled negative; cats 3 dead or euthanised PCR positive, 6 alive with 3 PCR positive, all 6 H5-seropositive | Complete and richer; misses that the cat test was validated on ferret sera only |
| Environment | Remote rural area, near habitats of wild reservoirs | Remote rural holding; outdoor access and wild-bird contact; outdoor feeding; house sealed for 4 months | Complete and richer; the habitat point is not in the paper |
| Population | 17 exposed, by profession, PPE, influenza vaccination and seroprevalence | 5 groups summing to 17, each with exposure, PPE and vaccination | Substantial; misses the human serology (0 of 11 with H5 antibodies) and two negative tests, which the PICO card states |
| Mobility and contact | Direct contact with infected birds, secretions, contaminated materials or environments | Contact type per group; the holder's row maps to "direct contact" | Substantial; the definition of a contact and the day-by-day timeline are not rows |
| Exposure pathways | Poultry house (backyard?) with outdoor access and wild-bird contact; spillover to cats | Wild-bird contact, outdoor feeding (mapped to "backyard"), cat infections, occupational exposure | Partial: no chain from wild birds to poultry, cats and people; the predation route is missed |

## 6. Where the evidence comes from

732 of the 1,076 extracted articles resolve to one country, 67 in all: China (169), United States (113), Bangladesh (36), United Kingdom (36), Indonesia (32), Netherlands (30). Italy has 20 articles (316 values); five name a region: Emilia-Romagna ITH5 (2), Abruzzo ITF1 (1), Campania ITF3 (1), Veneto ITH3 (1). Eurostat's NUTS 2024 list is loaded since 13 October, so regions are named across Europe down to NUTS 3; a city that is not a NUTS name gives only its country, and the United Kingdom is outside NUTS 2024. Of the other 344 articles, 196 name no place, 64 a multi-country area such as Europe or Asia, 44 several countries and 40 a place the resolver does not know.

## 7. The team's own search, compared

The team's Rayyan export (13 October) holds 1,524 records from Embase (968) and PubMed (556), for 2000 to 2026, screened by one reviewer from 15 July 2026: 73 included, 6 maybe, 1,445 excluded. They keep European studies on risk factors and on the virus's persistence.

| Team's papers | Number | In a LiteRev corpus | In a LiteRev relevant set |
| --- | --- | --- | --- |
| Included | 73 | 68 (93%) | 32 (44%) |
| Maybe | 6 | 5 | 2 |
| Excluded | 1,445 | 1,052 (73%) | 347 (24%) |
| All | 1,524 | 1,125 (74%) | 381 (25%) |

- All but 2 of their 556 PubMed records are in the exposure scenario's corpus: they ran the T2.4 exposure strategy. Embase is the gap: 570 of their 968 Embase records are in a LiteRev corpus.
- The 5 included papers LiteRev never found came only through Embase, whose index terms matched the strategy where the papers' titles and abstracts do not.
- 36 of the 68 found sit below LiteRev's relevance thresholds, which were set for about 1,000 articles per scenario, not for the team's criteria. Occupational exposure papers are well caught (25 of 33 relevant); laboratory studies of the virus's persistence are not (3 of 20).
- LiteRev's relevant sets hold 2,141 articles their search never retrieved, 199 of them European Union studies by the place LiteRev extracted.
- LiteRev already holds extracted values for 30 of their 73 included papers (689 values).

## 8. What is left

- **Blocked on files from the team**: the Annex 2 label hierarchy (codebook import relabels every row with no new AI call), and the extraction template itself, to check the Excel export against it.
- **Erol's call, not started**: marking the team's 68 included papers as included in LiteRev; Semantic Scholar's full endpoint for a strategy with a NOT (the exposure scenario gets only its 1,000 most relevant of 10,257 there); places below NUTS 3 (municipalities); typed transmission relations for T4.6; health system capacity (no sheet yet); a way for a reviewer to add a row the model missed; accuracy of the extraction against a gold standard.
- **Done since 12 October**: the Semantic Scholar, Cohere and OpenAlex production keys are installed by the deploy; the NUTS 2024 file is loaded; a database that fails during a new search keeps its previous records (PR #349).

## 9. Validation on the team's side

- [ ] Query: an information specialist reviews the terms (H7N7, H7N3, H5N2 and H10N3 are not named).
- [ ] Known papers: the team's own 73 included papers serve; 68 are in LiteRev's corpora (section 7).
- [ ] Threshold: about 100 articles just below the cut screened by hand.
- [ ] Screening: two reviewers in Double-Blind on the relevant set, kappa of 0.6 or more on a first batch.
- [ ] Full texts: the 484 relevant articles without one, retrieved through the library.
- [ ] Extracted data: two reviewers, one article in ten at first, starting with the 1,454 values whose quote was not found; the Dressler shelter PPE row corrected first.
- [ ] Pooled figures: none quoted before review; diseases kept apart; each forest plot opened.
- [ ] Brief: each claim checked against its cited articles; the R0 values (6 studies) mix animal and human transmission.
- [ ] Preprints: decide whether the 30 or so in the relevant set stay.
- [ ] Duplicates: exclude a second record of one study when a database misprinted its DOI.
- [ ] Reporting: flow numbers from the PRISMA card (15,447 duplicates removed) and the search date, 12 October 2026.
