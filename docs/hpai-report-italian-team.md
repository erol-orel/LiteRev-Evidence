# HPAI in LiteRev-Evidence: what was done, what it found, what is left

Prepared for the Italian partner team. State of production on 2026-10-08. Every figure in this
document was read from the live system, and each was re-checked independently before it was written
down. Where a number could not be reproduced exactly, it is given as a range and labelled as such.

Scenario: `usr-4757684f5462`, named HPAI, created 2026-09-29.
Live at https://literev-scenario.com

---

## 1. In one page

A boolean PubMed search crossing environmental and occupational exposure with avian influenza
pathogen terms brought in **640 articles**, of which **602 are relevant** at the scenario's
similarity threshold of 0.30.

On 2026-10-08, between roughly 10:49 and 12:11 UTC, LiteRev read **601 of those 602 papers** and
extracted structured data from each one: **573 from the full text** and **28 from the abstract
alone**. One paper was abandoned after three failed attempts. The run produced **10 893
observations**.

This was not a sample. The tool's standing rule is that an extraction reads every relevant paper,
and the run honoured it: the article cap is zero, meaning no limit, and that default is pinned by a
test.

Of the 10 893 observations, **10 291 (94.5%) carry a quotation that was located verbatim in the
source text**. The remaining 602 rows carry a quotation the system could not find again, and they
are excluded from every pooled estimate for that reason.

Two things must be said plainly before anything else, because they bound what this corpus can be
used for today:

1. **About half the extracted rows are not about avian influenza.** COVID-19 alone accounts for
   **2 091 rows (19.2%)** across 136 papers. Roughly 4 300 rows (about 39%) carry an avian or H5/H7/H9
   label. The search's exposure facet is broad enough that papers about other respiratory pathogens
   cleared the 0.30 similarity threshold, and nobody has screened them out by hand.
2. **Not one observation has been reviewed by a human.** All 10 893 are the model's unchecked
   reading. The machinery for reviewing them exists and works; it has simply never been used on this
   scenario.

---

## 2. What is extracted from each paper

One model call per paper returns three blocks.

**A study-level record (11 fields):** description, article type, study start, study end, location,
geographic notes, population at risk, number positive, percent positive, whether a mathematical
model is used, and the model type. Article type is constrained to research, short communication,
outbreak report, review or other.

**Ten topic flags**, each a strict yes or no, saying whether the paper carries anything on that
topic at all. Across the 602 relevant papers:

| Topic | Papers | Share |
|---|---|---|
| Environment | 466 | 77.4% |
| Animal host | 450 | 74.8% |
| Age | 355 | 59.0% |
| Vaccination | 318 | 52.8% |
| Human testing | 314 | 52.2% |
| Occupation | 277 | 46.0% |
| Sex and gender | 209 | 34.7% |
| PPE | 192 | 31.9% |
| Vector | 172 | 28.6% |
| Knowledge, attitudes, practices | 136 | 22.6% |

Between 95.6% and 100% of each flag was set from a full text rather than from an abstract, so these
are strong figures rather than artefacts of thin input.

**Observations**, filed on five sheets matching the extraction template. Each row carries the sheet,
transmission mode, disease, group, covariate, value, a description, notes, number of cases,
population at risk, the label the paper itself used, the page or section, whether the source was a
table, a figure or running text, the quotation, and whether that quotation was found again in the
source.

| Sheet | Rows | Papers | Quote found |
|---|---|---|---|
| Animal | 3 336 | 445 | 3 124 |
| Human exposure | 3 172 | 441 | 3 037 |
| Human susceptibility | 2 449 | 360 | 2 308 |
| Environment | 1 619 | 364 | 1 530 |
| Vector | 317 | 114 | 292 |

A full text yields far more than an abstract: **18.8 rows per full-text paper** (median 18, range 0
to 68) against **4.6 rows per abstract-only paper** (median 4.5, range 1 to 11). No abstract
produced a single table or figure row.

---

## 3. The field-by-field comparison

This is the part to read first, because it is the only place where LiteRev's output can be checked
against a human reading of the same paper.

**The paper.** Dressler A, Wagner-Wiening C, Tegtmeyer B, Haag-Milz S, Demattio B, Dürrwald R.
*Highly pathogenic avian influenza A(H5N1) in poultry and domestic cats and occupational exposure
among veterinary and other first responders, Germany, February 2026.* Euro Surveillance
2026;31(17):2600293. doi:10.2807/1560-7917.ES.2026.31.17.2600293. PMID 42141860. Open access.

It is in the corpus as article **475185**, relevant at similarity 0.421 and rerank 0.668. LiteRev
read it **from the full text, 24 319 characters, not truncated**, and produced **22 observations, all
22 of them quote-verified**.

The study-level record LiteRev built for it:

- location: Sigmaringen, Baden-Wuerttemberg, Germany
- geographic notes: small, remote poultry holding in a rural area; the investigation also involved an
  animal shelter and a veterinary practice
- population at risk: 17
- study window: 16 February 2026 to 9 March 2026
- article type: outbreak report
- mathematical model: no

### Slot by slot against the manual reading

**1. Pathogen hazard.** Manual reading: low biosecurity raises the hazard; shelter staff handled the
animals with inadequate PPE, surgical masks rather than FFP2 or FFP3 respirators. Marked as lacking
structured data.

LiteRev extracted the hazard as a value: an environment row `biosecurity measures = 0`, quoting *"The
holding, which had no biosecurity measures, comprised ca 21 chickens and nine free-roaming cats"*. It
then gave PPE per occupational group across six rows: poultry holder none, veterinary authority staff
none or partial gloves during early visits, police officers none, animal shelter staff full PPE,
veterinarians with one who euthanised the first cat without PPE, and one row covering all 17 contacts
quoting *"None of the contact persons had any PPE until 18 February while on site."*

Verdict: **substantially covered, and structured where the manual sheet was narrative.** One gap: the
respirator class is not captured. LiteRev has "none", "partial (gloves)" and "full PPE" from Table 4,
but not the distinction between a surgical mask and an FFP2 or FFP3 respirator. Note also a tension
worth checking against the paper: Table 4 records *full PPE* for animal shelter staff, whereas the
manual reading records inadequate PPE for them. The mask-class detail most likely sits in the running
text, and that is precisely what LiteRev did not pick up.

**2. Host and reservoir.** Manual reading: 21 chickens and 9 free-roaming cats, with seroprevalence
data available.

LiteRev: poultry 4 PCR-positive plus 17 PCR-negative, which is 21. Cats, three found or euthanised
individually and PCR-positive, plus six alive and transferred to the shelter of which three were
PCR-positive, which is 9. And the serology the manual sheet flagged as merely available was actually
extracted: **all six surviving cats seropositive for H5-specific antibodies by commercial ELISA**.

Verdict: **complete, and it adds the serology.** One caveat on form: the censuses arrive as
test-result rows, so "21 chickens" is only recoverable by adding 4 and 17. There is no single census
row.

**3. Environment.** Manual reading: a remote rural area near suitable habitat for the natural
reservoir. Marked as lacking structured data.

LiteRev: the geographic note above, plus three environment rows. Contact with wild birds (*"The birds
were in a poultry house but had access to the outside and contact with wild birds."*), outdoor
feeding (*"Poultry were fed outdoors, which may have attracted wild birds to the property."*), and the
control measure (*"the poultry house was sealed for 4 months in accordance with veterinary
regulations."*). Two of these mapped to the codebook as `setting > wild_bird_contact` and `setting >
backyard`, the latter confirming the manual sheet's tentative "maybe backyard poultry farm?".

Verdict: **complete and richer than the manual reading**, adding the outdoor feeding and the control
measure.

**4. Population.** Manual reading: 17 exposed humans, disaggregated by profession, with PPE uptake,
influenza vaccination status and seroprevalence available.

LiteRev: population at risk 17, and the full disaggregation, poultry holder 1, veterinary authority
staff 7, police officers 2, animal shelter staff 3, veterinarians 4, summing to exactly 17. Then five
further rows giving influenza vaccination per group: holder not vaccinated, veterinary authority staff
none vaccinated, police vaccinated, shelter staff vaccinated post-exposure, one veterinarian of four
unvaccinated.

Verdict: **complete on the count, the professions, PPE and vaccination.** One real miss: **no human
test result**. The human-testing flag is set for this paper, but not one of the 22 observations
carries the human PCR or serology outcome, which the manual sheet flagged as available data.

**5. Mobility and contact.** Manual reading: direct contact by handling infected birds, their
secretions, or contaminated materials and environments, with data on the type of contact.

LiteRev: the type of contact is the covariate on each exposure row, contact with animals, poultry
holding inspection and cats, site presence, contact with cats, handling cats, with transmission mode
"occupational exposure" throughout. One row mapped to `contact_type > direct_contact`.

Verdict: **covered per group.** The generic pathway sentence about secretions and contaminated
materials is not represented as a row.

**6. Exposure pathways.** Manual reading: the poultry house had outdoor access and contact with wild
birds as natural reservoirs, with cross-species transmission since cats were involved. Marked as
lacking structured data.

LiteRev holds every one of those facts, but as separate rows on separate sheets: wild bird contact and
outdoor feeding on the environment sheet, cat infection and seroconversion on the animal sheet.
**Nothing asserts the chain.** There is no typed relation saying wild bird to poultry to cat to human.

Verdict: **the facts are present, the pathway as a relation is not.** This is the clearest structural
gap in the tool for T4.5 and T4.6, and it is a known one.

### Scorecard

Of the six slots, LiteRev covers two completely and adds material beyond the manual reading (host and
reservoir, environment), and covers four substantially with one identified gap each (respirator class;
human test results; the generic contact pathway; typed relations).

It also supplied, unprompted, material the manual sheet does not hold: the cat serology, the outdoor
feeding, the four-month sealing, per-group vaccination detail, the exact study window, and the place
resolved to Sigmaringen in Baden-Württemberg at NUTS level 1.

Sex, gender and age are both flagged absent for this paper, which agrees with the manual reading.

One honest caveat about form: **9 of the 22 rows matched the default codebook and 13 did not.** Every
unmatched row is an occupational group, animal shelter staff, police officers, veterinarians,
veterinary authority staff. The default codebook has no node for them. This is the single change that
would most improve the output and it needs no new model call (see section 6).

---

## 4. Quantitative results, and why there are so few

LiteRev computes random-effects pooled proportions: a logit transform per study, DerSimonian and
Laird between-study variance, a Hartung-Knapp-Sidik-Jonkman 95% interval never allowed to be narrower
than the DerSimonian-Laird one, a 95% prediction interval from three studies up, and a hard minimum
of three studies.

Out of 10 893 observations, **120 label groups form and only 4 reach the three-study minimum.** One
hundred and one groups hold a single study and fifteen hold two.

| Pool | Studies | Events | Estimate | 95% CI | Prediction interval | I² |
|---|---|---|---|---|---|---|
| Animal, PCR-positive, HPAI H5N1 | 5 | 13/16 | 73.7% | 27.6 to 95.4% | 9.9 to 98.6% | 20% |
| Human, female, COVID-19 | 4 | 962/2 029 | 50.7% | 34.7 to 66.6% | 13.1 to 87.5% | 94% |
| Human, male, COVID-19 | 3 | 430/933 | 46.0% | 23.0 to 70.9% | 0.2 to 99.7% | 91% |
| Human, diabetes, COVID-19 | 3 | 125/1 616 | 7.8% | 5.3 to 11.2% | 2.5 to 21.5% | 0% |

**Three of the four are COVID-19**, from four SARS-CoV-2 clinical papers that entered the relevant
subset on semantic similarity alone (0.33 to 0.46 against a threshold of 0.30) and that nobody has
excluded by hand.

**The single avian influenza pool should not be read as an epidemiological result.** It pools five
studies totalling sixteen animals, and four of the five reported 100% on between one and five animals
each. The 73.7% is largely an artefact of the continuity correction applied to those saturated counts.

There is also one pooled odds ratio, male against female for COVID-19, at 0.73 with a 95% interval of
0.09 to 5.88 and I² of 95.5%. Its prediction interval spans five orders of magnitude. It is not a
usable quantity, and structurally it cannot be: the two labels partition the same denominator in all
three papers, so what is being pooled is not an odds ratio in the epidemiological sense.

**Why so little pools.** Three preconditions do the filtering, and the run is not the problem:

- **8 854 rows carry no case-and-population pair.** Only 19.3% of rows have a case count and 22.2% a
  population at risk, so most rows are descriptive rather than quantitative. This is the dominant
  cause.
- **602 rows had a quotation that could not be found again** in the source text and are excluded on
  integrity grounds.
- **8 997 rows carry a label absent from the codebook**, so they pool only with an identically spelled
  label, which in free text almost never happens.

After all exclusions, 1 434 rows are usable and 122 duplicates are dropped.

A technical note on reading the panel: it lists at most 120 groups and does not say when it has
truncated. Treat the list as the top of a longer one.

---

## 5. Geography

Place is resolved from the location string the extraction stored, with the article's own country as a
fallback hint, over every extracted paper with no sampling.

Of 601 extracted papers, **270 resolve to a country across 36 countries** and **331 do not**: 199
papers state no place at all and 132 name a place the resolver cannot use. Eighteen papers name several
countries and are counted separately.

Leading countries by paper count: China 81 (1 639 rows), United States 55 (1 135), Netherlands 13
(223), India 12 (190), South Korea 9 (135), Ghana 8 (194), United Kingdom 8 (130), then a four-way tie
at 6 papers: Germany (111 rows), Italy (95), Canada (101), Australia (73). France follows at 5 (74).

**For Italy specifically: 6 papers and 95 observation rows.** One reaches NUTS level 2, Emilia-Romagna
(ITH5); the other five stay at country level. Three further papers that name Italy among their study
area are absent from this distribution because they name several countries, so **nine papers mention
Italy and only six are counted**.

Two limits matter for regional work:

- **The NUTS layer is essentially empty.** The resolver knows 42 regions built into the code: Germany's
  sixteen Länder at NUTS 1, and Italy's five macro-areas at NUTS 1 plus twenty-one regions at NUTS 2.
  There is **no NUTS 3 and no city**, and no region of any other country, until Eurostat's GISCO NUTS
  file is loaded. In production it has never been loaded. Across all 601 papers, exactly **two carry
  any NUTS region**: one German and one Italian.
- **Two resolver behaviours to know about.** The longest region *name* wins rather than the deepest
  level, so "Umbria, Central Italy" resolves to NUTS 1 while "Umbria" alone resolves to NUTS 2. And a
  paper naming two regions silently keeps one: "Lombardia, Veneto, Italy" resolves to Lombardia only.

The report's country table stops at 25 of the 36 countries. The CSV export carries no geography at
all; the Excel workbook does.

---

## 6. What needs to be done

**Three things only the Italian team can supply.** Each unblocks work that is already built and
waiting.

1. **The Annex 2 label hierarchy.** This is the highest-value item by a wide margin. Today 8 997 of
   10 893 rows carry a label the default codebook cannot map, which is why only 17% of rows are
   normalised and why 13 of the 22 rows on the comparison paper went unmatched. Loading the hierarchy
   **relabels every stored row retroactively with no new model call and no cost**, because
   normalisation happens at read time and never rewrites what is stored. The default codebook also has
   seven groups with no second level at all (pregnancy, ethnicity, socioeconomic, climate, persistence,
   vector species, vector density), and rows in those groups can never map until the hierarchy fills
   them.
2. **The Eurostat GISCO NUTS file.** Loading it is what turns the geography from 42 hard-coded regions
   into real regional resolution, including NUTS 3.
3. **The extraction template itself.** `DATA_Extraction_Template_scenario_xxxx.xlsx` is not in the
   repository. The column titles in the code were transcribed by hand from the partner file, so nobody
   on this side can check the export against the real template column by column. Sending the file
   makes that check possible.

**Three things that need development.**

4. **Typed transmission relations** for T4.6. Today transmission is an unnormalised free-text column
   and the concept graph holds untyped co-occurrence triples with weights, not asserted predicates.
   This is the gap the comparison paper exposed most clearly.
5. **Health system capacity is not covered at all.** There is no sheet, no codebook node and no
   study-level column for it. Two topic flags (vaccination 318 papers, human testing 314) mark that a
   paper touches the subject, but they carry no capacity value. Of the six areas the template covers,
   this is the one that is genuinely absent rather than partial.
6. **Accuracy has never been measured for this extraction.** A validation harness exists in the
   repository, with a stratified sample that includes articles the screen rejected, blind annotation
   workbooks, inter-annotator kappa, and per-field precision, recall and F1 with a 10% value
   tolerance. It was built for a different extraction and does not cover this one. Nothing in the
   application computes precision or recall for the structured extraction today. The comparison in
   section 3 is one paper, which is an illustration, not a measurement.

**Two defects to be aware of.**

7. **The 601 stored extractions carry no model identifier and no prompt fingerprint.** The code records
   both, but the run finished about twenty-five minutes before the commit that writes them. The
   consequence is visible: the generated report's method line reads "Made with: unknown, prompt unknown
   (601)". Because the extraction version did not change, a re-run will not repair this without a code
   change, and a re-run costs model credits.
8. **192 papers, a third of the full-text set, had their text cut** at the 60 000 character prompt
   limit. This appears in the interface and in the API but in none of the deliverables: not in the
   report, not in the workbook, not in the CSV.

Two smaller ones: the study-level first-author name is filled on only 115 of 601 rows (19.1%), which
is why the report's appendix cites some studies as "? 2013"; and one paper (article 163468) is
permanently stuck at the three-attempt ceiling and will never be retried.

---

## 7. What the Italian team should test

Nothing in this list needs a key or an account. **Every read and every download works unauthenticated:**
CSV, Excel, Markdown, Word, PDF, pooled estimates, geography and codebook all return normally with no
credentials. Only writing needs a key, and writing is not part of this list.

Open https://literev-scenario.com, open the HPAI scenario, and go to the Evidence tab, Extraction
sub-tab.

**Check the extraction against your own reading.**

1. Find article 475185, the Dressler paper. Compare its 22 observations against section 3 above and
   against your own sheet. Confirm or correct the four gaps named there, especially the respirator
   class and the missing human test result.
2. For any row, use "View in the paper". It opens the source text at the quotation. This is the fastest
   way to judge whether a row is trustworthy, and it is the feature to lean on hardest.
3. Pick five papers you know well and check them the same way. Five is enough to form a view on
   whether a measured validation is worth the effort.

**Check the Italian material specifically.**

4. In the Geography panel, find Italy: 6 papers and 95 rows, one at Emilia-Romagna. Confirm those six
   are the papers you would expect, and tell us which of the three multi-country papers should also
   count as Italian.
5. Look at article 326185, *H5N1 Clade 2.3.4.4b Infections in Domestic Cats During an Avian Influenza
   Outbreak in Italy*. It is the Italian counterpart to the German comparison paper and the obvious
   second test.

**Check the outputs you would actually use.**

6. Download the Excel workbook (seven sheets, about 1.6 MB). The study-level sheet reproduces all 19
   template columns in order and the five observation sheets carry the template columns with review
   columns appended on the right, so it should paste into your file column by column. **Tell us where
   it does not line up.** We cannot check this ourselves without your template.
7. Download the flat CSV (21 columns, 10 893 rows, about 4.4 MB). Note that it is observation-level
   only: no author, year, DOI, location, region, study dates or model type, so it cannot fill the
   study-level sheet. Use the workbook for that.
8. Generate the report as Markdown, Word and PDF. Expect the method line to read "Made with: unknown"
   and some appendix citations to read "? 2013". Both are known and explained above.

**Look at the pooled estimates critically.**

9. Open the Pooled panel and read section 4 first. Three of the four pools are COVID-19 and the avian
   one rests on sixteen animals. The useful question is not whether the numbers are right but whether
   the method is the one you want once the labels are normalised.
10. Do not switch on "reviewed only": it empties the panel, correctly, because nothing has been
    reviewed. The labelled dataset download returns an empty file for the same reason.

**Judge the corpus boundary.** This is the most consequential thing you can do.

11. About half the rows are not avian influenza, and COVID-19 alone is 19.2% of them. Decide whether
    that is acceptable contamination or whether the threshold should rise above 0.30 or the papers
    should be screened by hand. Raising the threshold is one slider and costs nothing; hand screening
    is work but it is the defensible route, and the two-reviewer screen with Cohen's kappa is built and
    waiting.
12. Try the review workflow on twenty rows: accept, edit, reject. A rejected row leaves the pooled
    estimates and the exports immediately. Twenty rows is enough to see how it behaves, and this is
    where a second reader turns the model's reading into evidence.

**If a second scenario is wanted.** `usr-54fc5e52fea5` is the better choice. Its query is the HPAI
pathogen clause on its own, without the exposure, transmission and KAP facet, so it is a broader view
of the same question rather than a different one. Because extracted data is cached per paper rather
than per scenario, it **already inherits 312 extracted papers and 5 900 observations from the HPAI run
at no extra cost**, and its extraction, coverage, pooled and geography panels are already populated.
Its weakness: only 16.8% of its 6 523 relevant papers have full text, against 95.3% for HPAI, so a
full run there would read four papers in five from abstracts alone, at roughly a quarter of the yield
per paper.

The other influenza scenario, `usr-5ee70446a248`, is not ready to show: its threshold of 0.60 keeps
only 32 of 21 804 papers, and its brief grades the evidence highly on those 32 with no resolved
references. Lowering the threshold to get a credible corpus would put enrichment and extraction back
in the queue and invalidate the brief.

---

## 8. Two cautions about timing

Merging a change to the production branch restarts the API and cuts any search, pipeline or extraction
run in flight. **Do not deploy during a presentation.**

And the figures in this document are a snapshot. The corpus grows, the threshold is adjustable, and a
re-run would change the counts. Re-read the coverage panel on the day rather than quoting these
numbers from memory.
