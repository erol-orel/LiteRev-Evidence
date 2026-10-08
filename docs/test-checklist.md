# What to test, 2026-10-08

Everything below is merged to `main` and deployed. Two bodies of work landed the same afternoon from
two separate sessions, so this list covers both, plus the HPAI scenario specifically.

Figures marked **live** were read from production on 2026-10-08 and verified independently. They are a
snapshot: corpora grow and thresholds move.

No read and no download needs a key. Only writing does.

---

## A. The review and predictive split

The idea: not every saved search is a prediction problem. A scenario is now either a literature review
or a predictive scenario, and a review withholds exactly one thing, the model specification. Nothing
else is hidden.

1. **The switcher.** Open any scenario. In the tab row there are two buttons, "Literature review" and
   "Predictive scenario". Live check: `usr-69cc64786731` reads `kind = predictive`, `capabilities =
   ["model_spec"]`.
2. **Switch to Literature review.** The Variables tab should leave the tab row. If you were on it, you
   should land back on the review section rather than on a blank panel.
3. **Switch back to Predictive scenario.** The Variables tab should return.
4. **The scenario list.** A review scenario's card should lose its predictive-model card and gain a
   kind chip. The filter chips at the top should narrow the list to one kind or the other.
5. **A review's Evidence tab** should show the pooled epidemic parameters panel. The point is that a
   review gets its parameters pooled directly rather than being told to build a model specification
   first.
6. **Run the pipeline on a review.** The variables step should report itself *skipped*, with a reason,
   not fail and not hang.
7. **The failure mode to probe.** An unknown or missing kind must behave as predictive, not as a
   review. A scenario created before the column existed should keep every capability it had. Both
   `usr-69cc64786731` and `usr-4757684f5462` predate the column and both read `predictive`, so this
   already holds in production, but it is worth confirming you cannot lose a tab by accident.

## B. Enrichment scope

The idea: enrichment used to run over the whole corpus only. It can now run per scenario, and within a
scenario over all references or over just the relevant ones.

8. **The scope control** in a scenario's Enrichment section: "Relevant articles" against "Whole
   scenario". **Live** for `usr-69cc64786731`:

   | | Whole scenario | Relevant articles |
   |---|---|---|
   | Articles | 6 564 | 467 |
   | PICO | 6 520 done, 33 to process | 466 done (99.8%), 1 to process |
   | Metadata | 850 done, 5 714 to process | 135 done (28.9%), 332 to process |
   | Full text | 4 275 done, 2 033 to process | 420 done (89.9%), 46 to process |

   The metadata row is the whole point: 332 model calls instead of 5 714.
9. **The cap note.** Choose Whole scenario, Metadata, batch size 1 000. The card should read "1 000 to
   process, of 5 714 pending, the rest on the next run". The number shown must be the number the button
   will actually process.
10. **French.** Toggle to French and the same card should read "sur 5 714 en attente, le reste au
    prochain lancement". Check "Tout le scénario" and "Articles pertinents" too. Both locale files must
    carry the same keys and the same placeholders.
11. **A bad scope must be refused.** `GET /api/enrichment/status?scenario_id=...&scope=bogus` returns
    **400** with "Portée inconnue : 'bogus' (attendu : all, relevant)". Confirmed live.

## C. Date, time and IP of each search

12. **This needs a new search to test.** Existing scenarios record no IP, correctly: the column did not
    exist when they were created. `usr-69cc64786731` and `usr-4757684f5462` both read `created_ip =
    null`. Create a search now, expand its card in the list, and the provenance line should carry the
    creation time and "from <ip>".
13. Check the IP is the real client address and not the proxy's. The code reads the end of the
    forwarded-for chain according to the configured number of trusted hops.

## D. Two figures in the synthesis payload

14. **PICO coverage** is now counted over the relevant subset, as every number beside it already was.
    **Live** for `usr-69cc64786731`: 467 of 467 relevant, so **100%**. Under the old code the same
    screen computed 466 of 6 564, about 7%.
15. **The double-blind block** counts what the two reviewers actually did. For `usr-69cc64786731` it
    reads all zeros, and that is now a *true* zero: nobody has screened it twice. The old code returned
    zeros whatever you did, which is the bug. To see it work, screen a handful of articles as both
    reviewers in the double-blind screen, then reload the brief and expect real counts with agreements
    and conflicts split out.
16. **Known mismatch, explainable, not a regression.** The Evidence tab says 467 of 467 relevant papers
    have a PICO while the Enrichment panel says 1 still to process. Both are right under their own
    definition: the brief counts any PICO, the panel counts only a PICO with confidence at or above 0.5,
    and one paper has a weak one. Worth reconciling, but it is not new breakage.

## E. The alert digest

17. Not clickable. A review's digest points at its pooled parameters rather than at a model
    specification it does not have, and drops the SEIR projection from the caveat. Visible only in a
    sent email. Covered by tests.

---

## F. Structured extraction (the parallel session's work)

Live in the Evidence tab under an Extraction sub-tab. See `docs/hpai-report-italian-team.md` for the
figures and the field inventory.

18. **The run is finished.** `GET /api/user-scenarios/usr-4757684f5462/extraction/status` returns 602
    relevant, 601 extracted, 573 from full text, 28 from abstract, 10 893 observations, 1 given up, not
    running. The handoff note said 558 relevant and 5 extracted; that note was stale and has been
    rewritten.
19. **Per-article view.** Open article 475185 and confirm 22 observations, all quote-verified, read from
    full text at 24 319 characters, not truncated.
20. **"View in the paper"** on any row opens the source text at the quotation. This is the feature that
    makes a row checkable, so exercise it hard. 94.5% of rows corpus-wide have a locatable quotation;
    the other 602 rows are excluded from pooling for exactly that reason.
21. **The codebook panel.** 1 896 of 10 893 rows (17.4%) carry a normalised label today, because the
    partner hierarchy has never been loaded. On article 475185, 9 of 22 rows matched and the 13 that did
    not are all occupational groups. Confirm that loading a hierarchy relabels stored rows without a new
    model call: the only write is to the scenario's settings, and reads annotate a fresh copy.
22. **The review workflow.** Accept, edit, reject, clear, plus bulk accept and reject. Each writes one
    row keyed by document, observation and reviewer. Confirm that nothing a reviewer does alters the
    model's stored output, and that a rejected row leaves the pooled estimates and the exports.
23. **Cohen's kappa** needs two named reviewers who both decided the same observations. Today every one
    of the 10 893 observations is unreviewed, so kappa is not computable. Review twenty rows under two
    names to see it appear. Below twenty rows in common the interface should warn.
24. **Pooled estimates.** 120 groups form, only 4 reach the three-study minimum, and three of those four
    are COVID-19. Read section 4 of the report before judging. Do not switch on "reviewed only": it
    empties the panel, correctly.
25. **The exports.** Excel workbook, seven sheets, about 1.6 MB. Flat CSV, 21 columns, 10 893 rows,
    about 4.4 MB. Report as Markdown, Word and PDF. Do not quote a byte-exact size for the workbook: it
    is rebuilt per request and its size moves by a byte or two. The CSV is byte-stable.
26. **The empty download.** The labelled dataset button yields a zero-byte file today because nothing is
    reviewed and the interface does not ask for unreviewed rows. Either review something or expect the
    empty file.
27. **Geography.** 270 of 601 papers resolve to a country across 36 countries; 331 do not, of which 199
    state no place at all. Only 2 papers carry any NUTS region. The regional layer is 42 hard-coded
    regions, Germany at NUTS 1 and Italy at NUTS 1 and 2, until the Eurostat file is loaded. There is no
    NUTS 3 and no city.
28. **Two provenance defects to see for yourself.** The report's method line reads "Made with: unknown,
    prompt unknown (601)", because the run finished about twenty-five minutes before the commit that
    records the model and prompt. And 192 papers, a third of the full-text set, had their text cut at
    60 000 characters, a fact that appears in the interface and the API but in none of the deliverables.

## G. Things a partner team would hit

29. **Everything readable works with no key.** CSV, Excel, Markdown, Word, PDF, pooled, geography and
    codebook all return 200 unauthenticated. Confirm this from a private window, because it is what you
    are relying on when you send a link.
30. **Five actions need the write key** and each should now show an explanatory line rather than a dead
    button. Click each one without a key and confirm you get the explanation, not silence.

---

## H. Pre-existing issues, not from this week

31. **`/ask/stream` and `/ask/stream/filtered` take no API key**, so the only guard on those paid model
    calls is the rate limiter at 30 expensive calls a minute. A June commit on an abandoned branch would
    have closed this, and its own message explains why it was never merged: the browser has no way to
    send the key, so guarding them would break the app for everyone. A real decision, not a quick patch.
32. **One paper is permanently stuck** at the three-attempt ceiling (article 163468) and will never be
    retried. Seven further papers were read successfully but produced no observations.
33. **The pooled panel lists at most 120 groups** and does not say when it has truncated.
34. **The report's country table stops at 25** of the 36 countries.
35. **Study-level first-author name is filled on 115 of 601 rows** (19.1%), which is why some appendix
    citations read "? 2013".

---

## I. Housekeeping

36. **`claude/write-probe-0u6dce`** is a throwaway branch from a push outage that will not delete over
    git. One click on the branches page.
37. **Stale branches.** Every remote branch is accounted for: each came from a merged pull request
    except that probe and `claude/security-auth-fixes`, which holds the project's original history from
    before the history was restarted. No code is missing from `main`, but that branch is the only place
    the pre-restart commits survive, so do not delete it if you want that history.
