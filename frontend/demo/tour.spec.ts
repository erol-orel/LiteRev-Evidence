// The demo video's tour: drives the real interface through the main pipeline and writes
// what the renderer (scripts/demo_video.py) needs: screenshots, the boxes to frame in
// them, a caption overlay per scene, and manifest.json describing the timeline.
//
// Run by scripts/demo_video.py, which boots the API on a throwaway database with the
// synthetic demo corpus (scripts/demo_corpus.py), or points DEMO_BASE_URL at a running
// deployment. Nothing here writes: the tour types a question without launching the search,
// opens tabs and expands an article. Labels come from the locale files, so a renamed
// label fails here at once instead of silently filming the wrong screen.
//
//   DEMO_OUT          directory for the frames and manifest.json (required)
//   DEMO_LANG         en | fr (default en)
//   DEMO_SCENARIO_ID  scenario to film (default: the built-in demo scenario)
//   DEMO_API_KEY      admin key put in the browser, so write controls are shown
//   DEMO_SYNTHETIC    1 when the corpus is the synthetic demo corpus (badge on every frame)
//   DEMO_STRICT       1 to fail on a scene that breaks instead of skipping it
//   DEMO_SCENES       comma-separated scene ids to record only those (while editing)
import { expect, test, type Browser, type Locator, type Page } from "@playwright/test";
import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";
import { en } from "../src/i18n/locales/en";
import { fr } from "../src/i18n/locales/fr";

// ── The words: demo/storyboard.json ──────────────────────────────────────────
type Lang = "en" | "fr";
type SceneId =
  | "ask" | "scenarios" | "corpus" | "article" | "prisma" | "doubleBlind"
  | "evidence" | "pico" | "clusters" | "concepts" | "variables" | "model" | "alerts";
type Caption = { step: string; title: string; body: string };
type Card = { title: string; subtitle: string; lines: string[]; narration: string };
type Words = {
  question: string; demoBadge: string; demoNote: string;
  intro: Card; outro: Card; scenes: Record<SceneId, Caption>;
};
type Storyboard = { order: SceneId[]; en: Words; fr: Words };

const BOARD = JSON.parse(fs.readFileSync(new URL("./storyboard.json", import.meta.url), "utf8")) as Storyboard;
const LANG: Lang = process.env.DEMO_LANG === "fr" ? "fr" : "en";
const T = LANG === "fr" ? fr : en;
const S = BOARD[LANG];
const SCENE_ORDER = BOARD.order;
const OUT = path.resolve(process.env.DEMO_OUT ?? "demo-out");
const SCENARIO_ID = process.env.DEMO_SCENARIO_ID ?? "usr-deadbeef0001";
const API_KEY = process.env.DEMO_API_KEY ?? "";
const SYNTHETIC = process.env.DEMO_SYNTHETIC === "1";
const STRICT = process.env.DEMO_STRICT === "1";
const ONLY = (process.env.DEMO_SCENES ?? "").split(",").map((s) => s.trim()).filter(Boolean);

const VIEWPORT = { width: 1440, height: 810 };
const SCALE = 2;
const FRAME = { width: 1920, height: 1080 };
/** The tallest region kept from a page, in CSS pixels: memory for the renderer. */
const MAX_CLIP = 3000;
const LOGO = fileURLToPath(new URL("../public/literev-logo.png", import.meta.url));

type Box = { x: number; y: number; w: number; h: number };
/** One held framing of a scene: an image, the box to frame in it (CSS px of that image),
 *  whether to spotlight that box, and how long to hold it once reached. */
type Beat = { image: string; size: [number, number]; focus: Box | null; spot: boolean; hold: number; cut?: boolean };
type SceneOut = {
  id: string;
  kind: "screen" | "card";
  caption: Caption | null;
  narration: string;
  overlay: string | null;
  /** Bottom share of the frame the caption covers: the camera keeps the focus above it. */
  band: number;
  beats: Beat[];
};

class Skip extends Error {}

// ── Geometry, read in the page ───────────────────────────────────────────────
async function boxOf(loc: Locator): Promise<Box> {
  return loc.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  });
}

/** The panel around an element, as a viewer sees it: its closest rounded card, above
 *  the element itself (an input can be rounded too). `outer` asks for the larger panels. */
async function cardOf(loc: Locator, outer = false): Promise<Box> {
  return loc.evaluate((el, big) => {
    const sel = big ? '[class*="rounded-3xl"]' : '[class*="rounded-2xl"], [class*="rounded-3xl"]';
    const card = el.parentElement?.closest(sel) ?? el;
    const r = card.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  }, outer);
}

/** The grid an element sits in: a row of stat cards. */
async function gridOf(loc: Locator): Promise<Box> {
  return loc.evaluate((el) => {
    const grid = el.closest('[class*="grid"]') ?? el;
    const r = grid.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  });
}

function union(...boxes: Box[]): Box {
  const x0 = Math.min(...boxes.map((b) => b.x));
  const y0 = Math.min(...boxes.map((b) => b.y));
  const x1 = Math.max(...boxes.map((b) => b.x + b.w));
  const y1 = Math.max(...boxes.map((b) => b.y + b.h));
  return { x: x0, y: y0, w: x1 - x0, h: y1 - y0 };
}

function pad(b: Box, by: number): Box {
  return { x: b.x - by, y: b.y - by, w: b.w + 2 * by, h: b.h + 2 * by };
}

function capHeight(b: Box, h: number): Box {
  return { ...b, h: Math.min(b.h, h) };
}

// ── Capture ──────────────────────────────────────────────────────────────────
class Recorder {
  scenes: SceneOut[] = [];
  skipped: { id: string; reason: string }[] = [];
  private n = 0;
  private page: Page;
  private browser: Browser;

  constructor(page: Page, browser: Browser) {
    this.page = page;
    this.browser = browser;
  }

  private file(name: string): string {
    this.n += 1;
    return path.join("frames", `${String(this.n).padStart(3, "0")}-${name}.png`);
  }

  /** The page region holding `focuses`, as one image; the boxes are returned in its
   *  coordinates. Full width, at least one viewport tall, at most MAX_CLIP. */
  async region(name: string, focuses: Box[]): Promise<{ image: string; size: [number, number]; focus: Box[] }> {
    const page = this.page;
    const docH = await page.evaluate(() => document.documentElement.scrollHeight);
    const all = union(...focuses);
    let top = Math.max(0, Math.floor(all.y - 140));
    let bottom = Math.min(docH, Math.ceil(Math.max(all.y + all.h + 200, top + VIEWPORT.height)));
    if (bottom - top > MAX_CLIP) bottom = top + MAX_CLIP;
    if (bottom - top < VIEWPORT.height) top = Math.max(0, bottom - VIEWPORT.height);
    const image = this.file(name);
    await page.screenshot({
      path: path.join(OUT, image), fullPage: true,
      clip: { x: 0, y: top, width: VIEWPORT.width, height: bottom - top },
    });
    return {
      image, size: [VIEWPORT.width, bottom - top],
      focus: focuses.map((f) => ({ x: f.x, y: f.y - top, w: f.w, h: f.h })),
    };
  }

  /** The viewport as the user sees it (scrolled to the top). */
  async viewport(name: string): Promise<{ image: string; size: [number, number] }> {
    const image = this.file(name);
    await this.page.screenshot({ path: path.join(OUT, image) });
    return { image, size: [VIEWPORT.width, VIEWPORT.height] };
  }

  /** Renders HTML at the output size in a page of its own: caption overlays (transparent)
   *  and title cards. Returns the image path and the HTML's measured caption top. */
  async render(name: string, html: string, transparent: boolean): Promise<{ image: string; top: number }> {
    const ctx = await this.browser.newContext({ viewport: FRAME, deviceScaleFactor: 1 });
    const p = await ctx.newPage();
    await p.setContent(html);
    await p.evaluate(() => document.fonts.ready);
    const top = await p.evaluate(() => {
      const el = document.querySelector(".cap");
      return el ? el.getBoundingClientRect().top : 0;
    });
    const image = path.join("overlays", `${name}.png`);
    await p.screenshot({ path: path.join(OUT, image), omitBackground: transparent });
    await ctx.close();
    return { image, top };
  }
}

// ── Captions and cards, laid out as HTML ─────────────────────────────────────
const FONT = `Inter, "Inter Display", ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
const MONO = `ui-monospace, "SFMono-Regular", "DejaVu Sans Mono", Menlo, Consolas, monospace`;

function esc(s: string): string {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function badgeHtml(): string {
  return SYNTHETIC
    ? `<div class="badge"><span class="dot"></span>${esc(S.demoBadge)}</div>`
    : "";
}

const BASE_CSS = `
  html, body { margin: 0; width: ${FRAME.width}px; height: ${FRAME.height}px; font-family: ${FONT}; }
  .badge { position: absolute; top: 26px; right: 30px; display: flex; align-items: center; gap: 10px;
           padding: 9px 18px; border-radius: 999px; font-size: 21px; font-weight: 600; color: #f6e7c1;
           background: rgba(11, 20, 17, 0.82); border: 1px solid rgba(217, 179, 92, 0.55); }
  .badge .dot { width: 10px; height: 10px; border-radius: 50%; background: #d9b35c; }
`;

function captionHtml(c: Caption): string {
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    ${BASE_CSS}
    html, body { background: transparent; }
    .cap { position: absolute; left: 56px; right: 56px; bottom: 42px; display: flex; gap: 28px;
           align-items: flex-start; padding: 26px 36px 28px; border-radius: 28px;
           background: rgba(9, 19, 15, 0.9); border: 1px solid rgba(217, 179, 92, 0.38);
           box-shadow: 0 24px 70px rgba(0, 0, 0, 0.5); }
    .step { flex: none; width: 66px; height: 66px; border-radius: 50%; background: #d9b35c; color: #0b1411;
            font-weight: 800; font-size: 34px; display: flex; align-items: center; justify-content: center; }
    .title { color: #ffffff; font-weight: 700; font-size: 38px; letter-spacing: -0.01em; }
    .body { color: rgba(242, 246, 244, 0.86); font-size: 29px; line-height: 1.38; margin-top: 8px; }
    /* On the caption's top edge, where it never hides the interface. */
    .cap .badge { top: -22px; right: 34px; font-size: 19px; padding: 7px 16px; background: #0b1411; }
  </style></head><body>
    <div class="cap">${badgeHtml()}<div class="step">${esc(c.step)}</div>
      <div><div class="title">${esc(c.title)}</div><div class="body">${esc(c.body)}</div></div></div>
  </body></html>`;
}

function cardHtml(card: Card, extra: string[]): string {
  const logo = fs.existsSync(LOGO) ? `data:image/png;base64,${fs.readFileSync(LOGO).toString("base64")}` : "";
  const lines = [...card.lines, ...extra].map((l) =>
    `<div class="line${/^python3 /.test(l) ? " code" : ""}">${esc(l)}</div>`).join("");
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    ${BASE_CSS}
    body { background: radial-gradient(1200px 700px at 30% 20%, #17402c 0%, #0b1411 62%) #0b1411;
           color: #f2f6f4; display: flex; align-items: center; }
    .wrap { margin-left: 150px; margin-right: 150px; }
    img { width: 150px; height: 150px; border-radius: 30px; background: #f2f6f4; padding: 10px; box-sizing: border-box; }
    h1 { font-size: 82px; margin: 46px 0 0; letter-spacing: -0.02em; }
    h2 { font-size: 40px; font-weight: 500; color: #3fb97e; margin: 18px 0 44px; }
    .rule { width: 120px; height: 5px; border-radius: 3px; background: #d9b35c; margin-bottom: 40px; }
    .line { font-size: 30px; color: rgba(242, 246, 244, 0.78); margin-top: 14px; max-width: 1500px; line-height: 1.35; }
    .code { font-family: ${MONO}; color: #f6e7c1; font-size: 30px; }
  </style></head><body>
    ${badgeHtml()}
    <div class="wrap">${logo ? `<img src="${logo}" alt="">` : ""}
      <h1>${esc(card.title)}</h1><h2>${esc(card.subtitle)}</h2><div class="rule"></div>${lines}</div>
  </body></html>`;
}

// ── The page ─────────────────────────────────────────────────────────────────
async function settle(page: Page, ms = 700): Promise<void> {
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => undefined);
  await page.waitForTimeout(ms);
}

/** A tab button, looked up where it lives: the app's navigation is in the header, the
 *  scenario's sections and sub-tabs in <main>. The scope matters: "Corpus" names both the
 *  corpus statistics tab of the header and the corpus sub-tab of a scenario. */
const tabButton = (page: Page, label: string, where: "header" | "main" = "main") =>
  page.locator(where).getByRole("button", { name: label, exact: true }).first();

async function openTab(page: Page, label: string, where: "header" | "main" = "main"): Promise<void> {
  const b = tabButton(page, label, where);
  if (!(await b.count())) throw new Skip(`no "${label}" tab here`);
  await b.click();
  await settle(page);
}

/** The scenario page, from wherever the tour is. The app keeps the open scenario in the
 *  address (?scenario=), which is also how a shared link opens it. */
async function openScenario(page: Page): Promise<void> {
  if (new URL(page.url()).searchParams.get("scenario") === SCENARIO_ID) return;
  await page.goto(`/?scenario=${encodeURIComponent(SCENARIO_ID)}`);
  await expect(page.getByRole("heading", { level: 2 }).first()).toBeVisible({ timeout: 30_000 });
  await settle(page);
}

// ── The scenes ───────────────────────────────────────────────────────────────
type Ctx = { page: Page; rec: Recorder };
type Film = (ctx: Ctx) => Promise<Beat[]>;

const beat = (img: { image: string; size: [number, number] }, focus: Box | null, hold: number,
              spot = true, cut = false): Beat => ({ image: img.image, size: img.size, focus, spot, hold, cut });

const FILMS: Record<SceneId, Film> = {
  async ask({ page, rec }) {
    await page.goto("/");
    const input = page.getByPlaceholder(T.search.queryPlaceholder);
    await expect(input).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const panel = await cardOf(input, true);
    const beats: Beat[] = [];
    const q = S.question;
    const steps = 10;
    for (let i = 1; i <= steps; i++) {             // typing, one frame per few characters
      await input.fill(q.slice(0, Math.round((q.length * i) / steps)));
      await page.waitForTimeout(60);
      beats.push(beat(await rec.viewport("ask-typing"), panel, 0.12, false, true));
    }
    await page.getByRole("checkbox", { name: T.search.includeLive }).check();
    await settle(page, 400);
    const last = await rec.viewport("ask-live");
    beats.push(beat(last, await cardOf(input, true), 4.2, true, true));
    return beats;
  },

  async scenarios({ page, rec }) {
    await openTab(page, T.nav.scenarios, "header");
    const card = page.locator(`[data-testid="scenario-card"][data-scenario-id="${SCENARIO_ID}"]`);
    await expect(card).toBeVisible({ timeout: 30_000 });
    // Filter the list down to the filmed scenario: a deployment lists everyone's.
    const name = (await card.locator("h3").first().innerText()).trim();
    await page.getByPlaceholder(T.scenarios.searchPlaceholder).fill(name);
    await settle(page, 500);
    const img = await rec.viewport("scenarios");
    return [beat(img, null, 1.0, false), beat(img, pad(await boxOf(card), 8), 3.6)];
  },

  async corpus({ page, rec }) {
    // From the scenario list when the previous scene left it on screen, as a user would.
    const open = page.locator(`[data-testid="scenario-card"][data-scenario-id="${SCENARIO_ID}"]`)
      .getByRole("button", { name: T.scenarios.detailPage, exact: true });
    if (await open.count()) await open.click();
    else await openScenario(page);
    await expect(page.getByRole("heading", { level: 2 }).first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByText(T.scenarioDetail.seuil.thresholdLabel).first()).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("h4").first()).toBeVisible({ timeout: 60_000 });
    await settle(page);
    const header = union(await boxOf(page.getByRole("heading", { level: 2 }).first()),
                         await boxOf(tabButton(page, T.scenarioDetail.page.sections.review)));
    const threshold = await cardOf(page.getByText(T.scenarioDetail.seuil.thresholdLabel).first());
    const list = capHeight(union(await cardOf(page.locator("h4").nth(0)), await cardOf(page.locator("h4").nth(2)),
                                 await cardOf(page.getByText(T.scenarioDetail.corpus.literatureSources).first())), 760);
    const r = await rec.region("corpus", [header, threshold, list]);
    return [beat(r, pad(r.focus[0], 16), 2.0, false), beat(r, r.focus[1], 3.0), beat(r, pad(r.focus[2], 10), 3.4)];
  },

  async article({ page, rec }) {
    const titles = page.locator("h4");
    const preferred = titles.filter({ hasText: /wastewater|eaux us/i }).first();
    const title = (await preferred.count()) ? preferred : titles.first();
    await title.click();
    await expect(page.getByText(T.scenarioDetail.articleRow.screeningDecision).first()).toBeVisible({ timeout: 15_000 });
    await settle(page, 500);
    const card = capHeight(await cardOf(title), 900);
    const r = await rec.region("article", [card]);
    return [beat(r, pad(r.focus[0], 8), 4.6)];
  },

  async prisma({ page, rec }) {
    await openTab(page, T.scenarioDetail.review.subPrisma);
    await expect(page.getByText(T.scenarioDetail.prisma.stage1).first()).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const s1 = await cardOf(page.getByText(T.scenarioDetail.prisma.stage1).first());
    const s4 = await cardOf(page.getByText(T.scenarioDetail.prisma.stage4).first());
    const middle = { x: s1.x, y: s1.y + s1.h, w: s1.w, h: s4.y - (s1.y + s1.h) };
    const r = await rec.region("prisma", [s1, middle, s4]);
    return [beat(r, pad(r.focus[0], 10), 2.8), beat(r, pad(r.focus[1], 10), 2.8), beat(r, pad(r.focus[2], 10), 2.2)];
  },

  async doubleBlind({ page, rec }) {
    await openTab(page, T.scenarioDetail.review.subScreening);
    const kappa = page.getByText(T.scenarioDetail.doubleBlind.statKappa, { exact: true }).first();
    await expect(kappa).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const row = await gridOf(kappa);
    const title = await boxOf(page.getByText(T.scenarioDetail.doubleBlind.title, { exact: true }).first());
    const r = await rec.region("double-blind", [union(title, row)]);
    return [beat(r, pad(r.focus[0], 14), 4.2)];
  },

  async evidence({ page, rec }) {
    await openTab(page, T.scenarioDetail.page.sections.evidence);
    const corpusStat = page.getByText(T.scenarioDetail.evidences.statCorpusArticles, { exact: true }).first();
    await expect(corpusStat).toBeVisible({ timeout: 60_000 });
    await settle(page);
    // The section header, found by its subtitle: "Evidence" alone also names the sub-tab.
    const header = await boxOf(page.getByText(T.scenarioDetail.evidences.subtitle, { exact: true }).first());
    const stats = await gridOf(corpusStat);
    const buttons = await boxOf(page.getByRole("button", { name: T.scenarioDetail.evidences.exportPdf }).first())
      .catch(() => header);
    const charts = union(await cardOf(page.getByText(T.scenarioDetail.evidences.studyTypes, { exact: true }).first()),
                         await cardOf(page.getByText(T.scenarioDetail.evidences.evidenceLevels, { exact: true }).first()));
    const r = await rec.region("evidence", [union(header, buttons, stats), charts]);
    return [beat(r, pad(r.focus[0], 14), 3.4), beat(r, pad(r.focus[1], 10), 3.6)];
  },

  async pico({ page, rec }) {
    await openTab(page, T.scenarioDetail.evidenceTab.subPicoTable);
    await expect(page.getByText(T.scenarioDetail.pico.title).first()).toBeVisible({ timeout: 30_000 });
    const table = page.locator("table").first();
    await expect(table).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const top = union(await boxOf(page.getByText(T.scenarioDetail.pico.title).first()), capHeight(await boxOf(table), 620));
    const r = await rec.region("pico", [top]);
    return [beat(r, pad(r.focus[0], 12), 4.6)];
  },

  async clusters({ page, rec }) {
    await openTab(page, T.scenarioDetail.page.sections.viz);
    const umap = page.getByText(T.scenarioDetail.clustering.umapProjection).first();
    await expect(umap).toBeVisible({ timeout: 180_000 });   // computed in the background on a cold cache
    await settle(page, 1500);
    const map = await cardOf(umap);
    const detail = await cardOf(page.getByText(T.scenarioDetail.clustering.topKeywords).first()).catch(() => map);
    const r = await rec.region("clusters", [map, union(map, detail)]);
    return [beat(r, pad(r.focus[0], 10), 2.4), beat(r, pad(r.focus[1], 10), 3.2)];
  },

  async concepts({ page, rec }) {
    await openTab(page, T.scenarioDetail.vizTab.subKnowledgeGraph);
    const title = page.getByText(T.scenarioDetail.knowledgeGraph.conceptTitle).first();
    await expect(title).toBeVisible({ timeout: 120_000 });
    await settle(page, 1500);
    // The concept map is the largest drawing on the page.
    const graph = await page.evaluate(() => {
      let best: Element | null = null;
      let area = 0;
      for (const svg of Array.from(document.querySelectorAll("svg"))) {
        const r = svg.getBoundingClientRect();
        if (r.width * r.height > area) { area = r.width * r.height; best = svg; }
      }
      const r = (best ?? document.body).getBoundingClientRect();
      return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
    });
    const gaps = await cardOf(page.getByText(T.scenarioDetail.knowledgeGraph.evidenceGaps).first()).catch(() => graph);
    const r = await rec.region("concepts", [union(await boxOf(title), graph), union(graph, gaps)]);
    return [beat(r, pad(r.focus[0], 10), 3.0), beat(r, pad(r.focus[1], 10), 3.2)];
  },

  async variables({ page, rec }) {
    await openTab(page, T.scenarioDetail.page.sections.variables);
    const title = page.getByText(T.scenarioDetail.variables.modelVariablesTitle).first();
    await expect(title).toBeVisible({ timeout: 60_000 });
    await settle(page);
    // The variables, down to the outcome templates (their names are not translated).
    const card = await cardOf(title);
    let focus = capHeight(card, 640);
    const templates = page.getByText(T.scenarioDetail.outcomeTemplates.title).first();
    if (await templates.count()) {
      const below = await boxOf(templates);
      if (below.y > card.y + 100) focus = capHeight(card, below.y - 28 - card.y);
    }
    const r = await rec.region("variables", [focus]);
    return [beat(r, pad(r.focus[0], 10), 4.2)];
  },

  async model({ page, rec }) {
    await openTab(page, T.scenarioDetail.variablesModelTab.subModel);
    const dash = page.getByText(T.scenarioDetail.model.dashboard.title).first();
    await expect(dash).toBeVisible({ timeout: 60_000 });
    await settle(page, 1200);
    const dashboard = await cardOf(dash);
    const trained = await cardOf(page.getByText(T.scenarioDetail.model.guardrails).first()).catch(() => dashboard);
    const r = await rec.region("model", [dashboard, trained]);
    return [beat(r, pad(r.focus[0], 10), 3.2), beat(r, pad(r.focus[1], 10), 3.2)];
  },

  async alerts({ page, rec }) {
    await openTab(page, T.scenarioDetail.page.sections.alerts);
    const email = page.getByText(T.scenarioDetail.alerts.emailAlerts).first();
    await expect(email).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const both = union(await cardOf(email),
                       await cardOf(page.getByText(T.scenarioDetail.alerts.livingReviewPipeline).first()));
    const r = await rec.region("alerts", [both]);
    return [beat(r, pad(r.focus[0], 10), 4.2)];
  },
};

/** Scenes that need the scenario page open before they start. */
const ON_SCENARIO = new Set<SceneId>(["article", "prisma", "doubleBlind", "evidence", "pico", "clusters",
                                      "concepts", "variables", "model", "alerts"]);

test("records the main pipeline", async ({ page, browser }) => {
  test.setTimeout(20 * 60_000);
  fs.mkdirSync(path.join(OUT, "frames"), { recursive: true });
  fs.mkdirSync(path.join(OUT, "overlays"), { recursive: true });
  await page.setViewportSize(VIEWPORT);
  await page.addInitScript(({ lang, key }) => {
    localStorage.setItem("literev-lang", lang);
    if (key) localStorage.setItem("api_key", key);
  }, { lang: LANG, key: API_KEY });
  const rec = new Recorder(page, browser);

  const card = async (id: "intro" | "outro", c: Card, extra: string[], hold: number) => {
    const img = await rec.render(id, cardHtml(c, extra), false);
    rec.scenes.push({ id, kind: "card", caption: null, narration: c.narration, overlay: null, band: 0,
                      beats: [{ image: img.image, size: [FRAME.width, FRAME.height], focus: null, spot: false, hold }] });
  };

  if (!ONLY.length) await card("intro", S.intro, [], 3.6);
  // DEMO_SCENES picks scenes by id, in the order of FILMS, including those the default
  // order leaves out.
  const order = ONLY.length ? (Object.keys(FILMS) as SceneId[]).filter((id) => ONLY.includes(id)) : SCENE_ORDER;
  for (const id of order) {
    const caption = S.scenes[id];
    const started = Date.now();
    try {
      if (ON_SCENARIO.has(id)) await openScenario(page);
      if (id === "article") {
        await openTab(page, T.scenarioDetail.page.sections.review);
        await openTab(page, T.scenarioDetail.review.subCorpus);
      }
      const beats = await FILMS[id]({ page, rec });
      const overlay = await rec.render(id, captionHtml(caption), true);
      const band = Math.min(0.45, (FRAME.height - overlay.top + 18) / FRAME.height);
      rec.scenes.push({ id, kind: "screen", caption, narration: caption.body, overlay: overlay.image, band, beats });
      console.log(`scene ${id}: ${beats.length} beat(s) in ${((Date.now() - started) / 1000).toFixed(1)} s`);
    } catch (e) {
      const reason = e instanceof Error ? e.message.split("\n")[0] : String(e);
      if (STRICT && !(e instanceof Skip)) throw e;
      rec.skipped.push({ id, reason });
      console.log(`skipped scene ${id}: ${reason}`);
    }
  }
  if (!ONLY.length) await card("outro", S.outro, SYNTHETIC ? [S.demoNote] : [], 5.5);

  const manifest = {
    version: 1, lang: LANG, scenario: SCENARIO_ID, synthetic: SYNTHETIC,
    viewport: VIEWPORT, scale: SCALE, frame: FRAME,
    recorded_at: new Date().toISOString(),
    scenes: rec.scenes, skipped: rec.skipped,
  };
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
  expect(rec.scenes.length, "scenes recorded").toBeGreaterThan(0);
});
