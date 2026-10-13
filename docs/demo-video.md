# The demo video

A guided tour of LiteRev, about five minutes long, in English and in French, filmed
read-only on the live site (https://literev-scenario.com) on a real scenario, HPAI_last:
avian influenza in people exposed at work.

| | Video | Subtitles | Still |
|---|---|---|---|
| English | [literev-demo-en.mp4](demo/literev-demo-en.mp4) | [.vtt](demo/literev-demo-en.vtt) | [.jpg](demo/literev-demo-en.jpg) |
| Français | [literev-demo-fr.mp4](demo/literev-demo-fr.mp4) | [.vtt](demo/literev-demo-fr.vtt) | [.jpg](demo/literev-demo-fr.jpg) |

On the site, the **Demo video** button in the header plays the one in the interface's
language, under a title that gives its length, read from the file itself: the frontend
build copies `docs/demo/` into its output under `/demo/` (`frontend/vite.config.ts`), so
the files are not committed twice.

The video is code. A script drives the real interface in a browser, takes the
screenshots, frames and captions them, and encodes the result, with a voice reading each
scene when a text-to-speech key is given. When the interface changes, run it again and
the video follows.

## What it shows

A title card (a real scenario, filmed read-only, and what the AI writes is a draft a
reviewer checks), then the scenes below, each with a caption and a narration, a card of
everything that leaves the application, and an end card that says where and when it was
recorded. The scene ids are those of `frontend/demo/storyboard.json`.

1. **Ask** (`ask`): the question typed in the search box, live sources ticked, never sent.
2. **Every search becomes a scenario** (`scenarios`): the scenario in the list.
3. **Score and select** (`corpus`): the relevance scores and the threshold; **The relevant
   articles, ranked** (`sources`): the export menu opened on its six formats, the ranked
   list, the databases and the years; **Read and decide** (`article`): an article opened
   on its screening decision, its PICO and its abstract.
4. **Screen the PRISMA way** (`prisma`): the four stages of the PRISMA 2020 flow;
   **Database by database** (`prismaSources`): each database's count, the total it
   announced, the caps and the reduced queries; **The search strategy, kept**
   (`strategy`): the Boolean strategy, its PubMed version with MeSH terms, its explanation.
5. **Synthesise the whole corpus** (`evidence`): designs, sources and levels of evidence
   counted over every relevant article; **Epidemiological parameters** (`epiParams`): R0,
   case fatality, incubation and the others, with their intervals and numbers of studies;
   **An evidence brief, as a draft** (`brief`): its level of evidence, its GRADE rating,
   its summary, its key findings and its exports; **Claims, actions and gaps** (`claims`):
   the claims with their computed strength and cited articles, the recommended actions,
   the research gaps and the gap matrix; **One extraction per article** (`pico`): the PICO
   table.
6. **Structured extraction** (`extraction`): the counts and how each article was read;
   **What the papers report** (`coverage`); **Five sheets and a codebook** (`sheets`);
   **Every value has its quote** (`quotes`): one article's rows, the exact quote, *View in
   the paper* with the sentence highlighted in the text, and the review buttons, never
   clicked; **Take the extraction away** (`downloads`): Excel, CSV, PDF and Word, the
   review and the annotated dataset; **Where the evidence comes from** (`geography`): the
   countries, and Italy's NUTS regions.
7. **Pooled estimates and comparisons** (`pooled`): the table, its download, the forest
   plot of a label with 5 to 12 studies, and the odds ratios between groups.
8. **Ask the corpus** (`assistant`): a question typed, never sent; **A cited answer,
   kept** (`assistantAnswer`): a stored answer opened, with its exports, when the scenario
   has one.
9. **Explore the literature** (`clusters`): the clusters with their AI summary, key words,
   representative article and export; **Map concepts and gaps** (`concepts`): the concept
   map, a population clicked to show its links and articles, the evidence gaps.
10. **From evidence to a model** (`variables`): the outcome, the predictors with their
    sources, the algorithm; **Train, explain, monitor** (`model`): the trained model, when
    there is one.
11. **Keep it living** (`alerts`): the alerts and the living review.
12. **Everything you can take away** (`takeaway`): every export, in one card.

A scene whose content is not there is skipped and listed in the run's output, never
computed for the occasion. On HPAI_last today: no answer is kept yet (`assistantAnswer`)
and no model is trained (`model`); in French, the brief, its claims, the cluster
summaries and the model variables have not been generated yet, so the French video skips
`brief`, `claims`, `clusters` and `variables` and is shorter. Generating them is a write,
for someone holding the key: in the app, or with
`WRITE_API_KEY=... python3 scripts/preflight_demo.py --base https://literev-scenario.com/api --scenario usr-b1e16a5d9c38 --lang fr`;
the next recording films them.

## Filmed read-only

The tour never changes the site it films:

- **No key in the browser.** With `--base-url`, the script gives the browser no key, so
  the app is in its read-only mode, as for any visitor.
- **A guard on every request.** In the browser, every request to the API that is not a
  GET is aborted, and so is a GET that forces a recomputation. The few GETs that compute
  and store their result when their cache is cold are aborted too (recommended actions,
  the article similarity graph, a SEIR projection not stored), or read from the stored
  copy (model variables, cluster summaries in a language not stored); the scenes that
  would show them are skipped. The run prints what the guard did, and `manifest.json`
  keeps it.
- **Typed, never sent.** The search and the assistant's question are typed and cleared:
  asking the assistant calls the model and stores the answer and its cost, so a stored
  answer is only filmed when there is one.
- **The figures are the scenario's.** The captions' `{records}`, `{relevant}`,
  `{extracted}`... are read from the scenario when the tour starts, so they stay true when
  the corpus grows.

Every screen the tour opens is shown as it is, so choose a scenario whose content may be
published.

## Make it again

The site's video, from the **Demo video** workflow (`.github/workflows/demo-video.yml`):
run it from the Actions tab with

- `base_url`: `https://literev-scenario.com` (empty: the local demo on its synthetic
  corpus, below);
- `scenario`: `usr-b1e16a5d9c38`;
- `commit`: ticked to put the six files (`literev-demo-{en,fr}.{mp4,vtt,jpg}`) into
  `docs/demo/` on the branch the workflow was started on.

The render job records and encodes both languages (about twenty minutes, an hour at
most); the narration key reaches its render step only. The commit is a job of its own,
with the only write token: it takes the files from the render job's artifact, never sees
the narration key, and pushes as `github-actions[bot]`. A push made with the workflow's
token starts no other workflow, so the site plays the new video after the next deploy.
Without `commit`, download the videos from the run's artifacts.

By hand, with Node and `npm ci` done in `frontend/`, a Playwright Chromium
(`npx playwright install chromium` in `frontend/`), `ffmpeg` and `Pillow`:

```bash
python3 scripts/demo_video.py --base-url https://literev-scenario.com --scenario usr-b1e16a5d9c38 \
    --lang en --lang fr
```

writes `docs/demo/literev-demo-<lang>.mp4` with its `.vtt` subtitles and its `.jpg`
still. Useful options: `--out-dir` writes elsewhere; `--keep-work` keeps the
screenshots, to render again without recording
(`python3 scripts/demo_render.py --frames <work>/<lang> -o out.mp4`); `--no-narration`
keeps it silent even with a key set; `--crf` trades size for sharpness (28 by default,
about 20 MB for five minutes).

## Change it

- **The words**: `frontend/demo/storyboard.json`. Each scene has, in each language, a
  caption (`step`, `title`, and a `body` of two lines on the caption card, at most 230
  characters) and a `narration`, the same thing written for the ear (abbreviations
  spelled out, numbers in digits). The title card, the takeaway card, the end card and
  the questions typed are there too. `tests/test_demo_video.py` checks that both
  languages caption and narrate the same scenes, name the same `{figures}`, and fit.
- **The scenes**: `frontend/demo/tour.spec.ts`, one function per scene in `FILMS`. A
  scene opens its screen, waits for its content, skips itself (`Skip`) when the content
  is not there, and returns its beats: a screenshot, the box to frame in it, whether to
  spotlight it, and how long to hold. Labels are read from the locale files, so a renamed
  label breaks the recording at once rather than filming the wrong screen.
- **The order**: `order` in the storyboard. To record only some scenes while editing:
  `DEMO_SCENES=prisma,evidence` with the variables listed at the top of `tour.spec.ts`.
- **The pace**: a scene is held as long as its caption takes to read at 15 characters a
  second, transitions included; narrated, as long as its voice and a second more. Keep
  the narration of a scene near 150 characters: the whole tour then lasts five to six
  minutes.
- **The look**: captions and cards are HTML in `tour.spec.ts`; the camera, spotlight,
  transitions and timing are in `scripts/demo_render.py`.

The double-blind scene exists but is not in the default order: the kappa
interpretation it shows is served in French whatever the interface language
(`api/double_blind.py`). Add `doubleBlind` back to `order` once it is translated.

## Narration

Give a text-to-speech key and each scene's narration is read aloud:

```bash
DEMO_TTS_API_KEY=sk-... python3 scripts/demo_video.py --base-url ... --scenario ... --lang en --lang fr
```

The key is an OpenAI key by default (`gpt-4o-mini-tts`, voice `coral`); any API that
speaks `/v1/audio/speech` works with `DEMO_TTS_BASE_URL`. `DEMO_TTS_MODEL` and
`DEMO_TTS_VOICE` change the voice. `--narrate` uses `OPENAI_API_KEY` when
`DEMO_TTS_API_KEY` is not set; `--no-narration` keeps the video silent. Each scene
then lasts at least as long as its voice. The clips are cached in the work directory,
so rendering again does not pay for them twice. For the workflow, store the key as the
secret `DEMO_TTS_API_KEY` of the GitHub environment "production", beside the other keys
(a repository secret of that name works too): the job reads that environment without
recording a deployment.

## The local demo, on a synthetic corpus

Without `--base-url`, the script films the built-in demo scenario on a throwaway
database instead, the way the browser smoke test runs the app: no network, no key, and
nothing real to publish. It takes its own question and title card from the `synthetic`
block of the storyboard, and skips what its corpus does not have.

```bash
DB_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres \
    python3 scripts/demo_video.py --lang en --lang fr
```

It creates a throwaway database next to `DB_URL` (never touching the one `DB_URL`
names), boots the API on it without any LLM key, seeds the corpus, builds the frontend,
records the tour once per language, renders into `demo-video/` (never into `docs/demo/`,
so a synthetic recording does not replace the site's video by accident), then stops the
API and drops the database (`--keep-db` keeps it). It needs, besides the above, a
Postgres server, the backend dependencies with `uvicorn`, and `umap-learn` with `hdbscan`
for the clustering scene.

What is real there and what is not:

- **Synthetic: the 200 articles.** `scripts/demo_corpus.py` writes fictional ones on the
  demo scenario's subject (influenza, wastewater and weather signals): titles, abstracts,
  years, sources, relevance scores, designs, PICO, main findings, typed concepts and
  full-text sections. The journal of every record reads "Synthetic demo record", none has
  authors, a DOI or a PMID, and every frame carries a "Demo data · synthetic corpus"
  badge. The screening decisions are made through the API by two fictional reviewer
  codes.
- **Computed by the app: everything shown from them.** The counts, the PRISMA flow, the
  evidence profile, the PICO table, the clusters and the concept map are the app's own
  computations over those rows.
- **Real data: the model.** At startup the API trains the demo scenario's model on the
  committed EAWAG wastewater series (`scripts/trial_output/ch-influenza_dataset.csv`).
- **Not shown: what needs an LLM.** Without an OpenAI key, the narrative brief, the
  assistant's answers, the cluster summaries and the structured extraction are not
  generated, and the tour skips the scenes that would show them.
