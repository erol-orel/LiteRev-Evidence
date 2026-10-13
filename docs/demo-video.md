# The demo video

A short video of the main pipeline, from a research question to a trained model, in
English and in French:

| | Video | Subtitles | Still |
|---|---|---|---|
| English | [literev-demo-en.mp4](demo/literev-demo-en.mp4) | [.vtt](demo/literev-demo-en.vtt) | [.jpg](demo/literev-demo-en.jpg) |
| Français | [literev-demo-fr.mp4](demo/literev-demo-fr.mp4) | [.vtt](demo/literev-demo-fr.vtt) | [.jpg](demo/literev-demo-fr.jpg) |

The video is code. A script drives the real interface in a browser, takes the
screenshots, frames and captions them, and encodes the result. When the interface
changes, run the script again and the video follows.

## What it shows

Twelve scenes between a title card and an end card, each with a caption (and, when a
text-to-speech key is given, a narration):

1. **Ask**: the question typed in the search box, live sources ticked.
2. **Every search becomes a scenario**: the scenario in the list.
3. **Score and select**: the relevance threshold, the corpus, its sources and years;
   then **Read and decide**: an article opened on its abstract, its PICO and the
   screening decision.
4. **Screen the PRISMA way**: the PRISMA 2020 flow, identification to evidence set.
5. **Synthesise the whole corpus**: the evidence profile (designs, sources, GRADE
   ceilings) and the brief's export buttons; then **One extraction per article**: the
   PICO table.
6. **Explore the literature**: the UMAP and HDBSCAN clusters; then **Map concepts and
   gaps**: the concept map, its strongest links and its evidence gaps.
7. **From evidence to a model**: the model variables; then **Train, explain,
   monitor**: the trained model's dashboard and guardrails.
8. **Keep it living**: the alerts and the living review.

## What is real and what is not

Every frame is the real interface, served by the real API, computing on what is in its
database. What is in that database is the part to know about:

- **Synthetic: the 200 articles.** Real articles can neither ship with the repository
  nor be fetched offline, so `scripts/demo_corpus.py` writes fictional ones on the
  demo scenario's subject (influenza, wastewater and weather signals): titles,
  abstracts, years, sources, relevance scores, designs, PICO, main findings, typed
  concepts and full-text sections, the facts a finished search and map step leave on
  an article. The journal of every record reads "Synthetic demo record", none has
  authors, a DOI or a PMID, and every frame of the video carries a "Demo data ·
  synthetic corpus" badge. The screening decisions are made through the API by two
  fictional reviewer codes.
- **Computed by the app: everything shown from them.** The counts, the PRISMA flow, the
  evidence profile and its GRADE ceilings, the PICO table, the clusters (TF-IDF, UMAP
  and HDBSCAN, computed by the API during the recording) and the concept map are the
  app's own computations over those rows.
- **Real data: the model.** The demo scenario is the built-in one (`demo_seed.py`): at
  startup the API trains its model on the committed EAWAG wastewater series
  (`scripts/trial_output/ch-influenza_dataset.csv`). The dashboard, the drivers and the
  guardrails come from that training.
- **Not shown: what needs an LLM.** The API runs without an OpenAI key, so the
  narrative brief, the assistant's answers, the cluster summaries and the Boolean
  translation of a question are not generated. The video shows where they are, and
  never a text the app did not write.

## Make it again

Requirements: those of the browser smoke test (`docs/ops-runbook.md` §8): a Postgres
server, the backend dependencies with `uvicorn`, Node with `npm ci` done in `frontend/`
and a Playwright Chromium (`npx playwright install chromium` in `frontend/`); plus
`ffmpeg`, `Pillow`, and `umap-learn` with `hdbscan` for the clustering scene.

```bash
DB_URL=postgresql+psycopg://postgres:postgres@127.0.0.1:5432/postgres \
    python3 scripts/demo_video.py --lang en --lang fr
```

About five minutes for one language, three more for the other: the script creates a throwaway database next to `DB_URL` (never
touching the one `DB_URL` names), boots the API on it without any LLM key, seeds the
corpus, builds the frontend, records the tour once per language, renders
`docs/demo/literev-demo-<lang>.mp4` with its `.vtt` subtitles and its `.jpg` still,
then stops the API and drops the database.

Useful options: `--keep-work` keeps the screenshots, to render again without recording
(`python3 scripts/demo_render.py --frames <work>/<lang> -o out.mp4`); `--crf` trades
size for sharpness (28 by default, about 5 MB for 95 seconds); `--keep-db` keeps the
database to look at it in the app.

The **Demo video** workflow (`.github/workflows/demo-video.yml`) does the same on
GitHub: run it from the Actions tab and download the videos from the run's artifacts.
It also runs on a pull request that changes the demo's own files.

## Change it

- **The words**: `frontend/demo/storyboard.json`, one caption per scene in each
  language, the question typed in the first scene and the two cards. A caption body is
  also the narration, so it is written as full sentences. `tests/test_demo_video.py`
  checks that both languages caption the same scenes and that each body fits on two
  lines.
- **The scenes**: `frontend/demo/tour.spec.ts`, one function per scene in `FILMS`. A
  scene opens its screen, waits for its content, and returns its beats: a screenshot,
  the box to frame in it, whether to spotlight it, and how long to hold. Labels are
  read from the locale files, so a renamed label breaks the recording at once rather
  than filming the wrong screen.
- **The order**: `order` in the storyboard. To record only some scenes while editing:
  `DEMO_SCENES=prisma,evidence` with the variables listed at the top of `tour.spec.ts`.
- **The look**: captions and cards are HTML in `tour.spec.ts`; the camera, spotlight,
  transitions and timing are in `scripts/demo_render.py`.

The double-blind scene exists but is not in the default order: the kappa
interpretation it shows is served in French whatever the interface language
(`api/double_blind.py`). Add `doubleBlind` back to `order` once it is translated.

## Narration

Give a text-to-speech key and each caption is also read aloud:

```bash
DEMO_TTS_API_KEY=sk-... python3 scripts/demo_video.py --lang en --lang fr
```

The key is an OpenAI key by default (`gpt-4o-mini-tts`, voice `coral`); any API that
speaks `/v1/audio/speech` works with `DEMO_TTS_BASE_URL`. `DEMO_TTS_MODEL` and
`DEMO_TTS_VOICE` change the voice. `--narrate` uses `OPENAI_API_KEY` when
`DEMO_TTS_API_KEY` is not set; `--no-narration` keeps the video silent. Each scene
then lasts at least as long as its voice. The clips are cached in the work directory,
so rendering again does not pay for them twice. For the workflow, store the key as the
repository secret `DEMO_TTS_API_KEY`.

## Film a real deployment

To show real articles and real LLM outputs, film a running deployment instead:

```bash
python3 scripts/demo_video.py --base-url https://literev-scenario.com --scenario usr-xxxxxxxxxxxx
```

The tour only reads: it types a question without launching the search, opens tabs and
expands an article (a tab whose cache is cold starts the computation it shows, as it
does for any visitor; run `scripts/preflight_demo.py` first to warm them). It filters the scenario list down to the filmed scenario, but every
screen it opens is shown as it is, so choose a scenario whose content may be published.
A scene whose screen is missing there (a literature review has no model) is skipped
and listed in the run's output.
