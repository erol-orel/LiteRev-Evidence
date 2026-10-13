// The demo video's tour: drives the real interface through the pipeline and writes what the
// renderer (scripts/demo_video.py, scripts/demo_render.py) needs: screenshots, the boxes to
// frame in them, a caption overlay per scene, and manifest.json describing the timeline.
//
// Two ways to run it, both through scripts/demo_video.py:
//   - the local demo: a throwaway database with the synthetic demo corpus
//     (scripts/demo_corpus.py), served by the local preview;
//   - a running deployment (DEMO_BASE_URL), read-only. The tour types without sending,
//     opens tabs, articles and plots, and never sends a write: a guard aborts every request
//     to the API that is not a read, and a screen whose content is not computed yet is
//     skipped rather than computed (see "Read-only on a deployment" below).
// Labels are read from the locale files, so a renamed label fails here at once instead of
// silently filming the wrong screen.
//
//   DEMO_OUT          directory for the frames and manifest.json (required)
//   DEMO_LANG         en | fr (default en)
//   DEMO_SCENARIO_ID  scenario to film (default: the built-in demo scenario)
//   DEMO_BASE_URL     a deployment to film, read-only (default: the local preview)
//   DEMO_API_KEY      admin key put in the browser, so write controls are shown (local demo
//                     only: never on a deployment)
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
  | "ask" | "scenarios" | "corpus" | "sources" | "article" | "prisma" | "prismaSources" | "strategy"
  | "doubleBlind" | "evidence" | "epiParams" | "brief" | "claims" | "pico" | "extraction" | "coverage"
  | "sheets" | "quotes" | "downloads" | "geography" | "pooled" | "assistant" | "assistantAnswer"
  | "clusters" | "concepts" | "variables" | "model" | "alerts" | "takeaway";
/** On screen: step, title and body. Spoken: the narration, written for the ear. */
type Caption = { step: string; title: string; body: string; narration: string };
type Card = { title: string; subtitle: string; lines: string[]; narration: string };
type TakeawayCard = { title: string; subtitle: string; groups: { title: string; items: string[] }[] };
type Words = {
  question: string; assistantQuestion: string; demoBadge: string; demoNote: string; filmedOn: string;
  intro: Card; outro: Card; takeawayCard: TakeawayCard;
  /** The local demo on its synthetic corpus: its own questions and title card. */
  synthetic: { question: string; assistantQuestion: string; intro: Card };
  scenes: Record<SceneId, Caption>;
};
type Storyboard = { order: SceneId[]; en: Words; fr: Words };

const BOARD = JSON.parse(fs.readFileSync(new URL("./storyboard.json", import.meta.url), "utf8")) as Storyboard;
const LANG: Lang = process.env.DEMO_LANG === "fr" ? "fr" : "en";
const LOCALE = LANG === "fr" ? "fr-FR" : "en-US";
const T = LANG === "fr" ? fr : en;
const S = BOARD[LANG];
const SCENE_ORDER = BOARD.order;
const OUT = path.resolve(process.env.DEMO_OUT ?? "demo-out");
const SCENARIO_ID = process.env.DEMO_SCENARIO_ID ?? "usr-deadbeef0001";
const REMOTE = Boolean(process.env.DEMO_BASE_URL);
/** Never a key in the browser of a deployment: the tour only reads there. */
const API_KEY = REMOTE ? "" : process.env.DEMO_API_KEY ?? "";
const SYNTHETIC = process.env.DEMO_SYNTHETIC === "1";
const STRICT = process.env.DEMO_STRICT === "1";
const ONLY = (process.env.DEMO_SCENES ?? "").split(",").map((s) => s.trim()).filter(Boolean);
const QUESTION = SYNTHETIC ? S.synthetic.question : S.question;
const ASSISTANT_QUESTION = SYNTHETIC ? S.synthetic.assistantQuestion : S.assistantQuestion;
const INTRO = SYNTHETIC ? S.synthetic.intro : S.intro;

const VIEWPORT = { width: 1440, height: 810 };
const SCALE = 2;
const FRAME = { width: 1920, height: 1080 };
/** The tallest region kept from a page, in CSS pixels: memory for the renderer. */
const MAX_CLIP = 3000;
/** Empty space added under the page, in CSS pixels: the camera needs it to lift a box from
 *  the very bottom of a page above the caption. */
const ROOM = 360;
/** Below this size, in bytes, a screenshot of the interface is a blank frame: a real one
 *  weighs hundreds of kilobytes. */
const BLANK_PNG = 60_000;
/** Characters read per second: a scene is held at least as long as its caption takes to read
 *  and its narration to say, so the silent video can be read and the voice is not rushed. */
const READ_CPS = 15;
/** The renderer's transitions, in seconds (scripts/demo_render.py: FADE_SCENE, MOVE, FADE_BEAT):
 *  the dip into a scene, a camera move within one screenshot, a crossfade between two. The
 *  caption is on screen through them, so they count as time to read it. */
const DIP = 0.7;
const MOVE = 0.9;
const FADE = 0.45;
const LOGO = fileURLToPath(new URL("../public/literev-logo.png", import.meta.url));

type Box = { x: number; y: number; w: number; h: number };
/** One held framing of a scene: an image, the box to frame in it (CSS px of that image),
 *  whether to spotlight that box, and how long to hold it once reached. */
type Beat = { image: string; size: [number, number]; focus: Box | null; spot: boolean; hold: number; cut?: boolean };
type SceneOut = {
  id: string;
  kind: "screen" | "card";
  caption: { step: string; title: string; body: string } | null;
  narration: string;
  overlay: string | null;
  /** Bottom share of the frame the caption covers: the camera keeps the focus above it. */
  band: number;
  beats: Beat[];
};

class Skip extends Error {}

type Json = Record<string, unknown>;
const obj = (v: unknown): Json => (v !== null && typeof v === "object" && !Array.isArray(v) ? (v as Json) : {});
const positive = (v: unknown): number | null => (typeof v === "number" && Number.isFinite(v) && v > 0 ? v : null);
const escapeRe = (s: string): string => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
/** The text of a locale string before its first {placeholder}: what stays the same on screen. */
const lead = (s: string): string => s.split(/\{\w+\}/)[0].trim();

// ── The scenario's own figures, read once ────────────────────────────────────
/** The figures a caption or a narration may name: {records}, {sources}... They are read from
 *  the scenario when the tour starts, so a caption stays true when the corpus grows. */
const FACT_NAMES = ["records", "sources", "total", "relevant", "extracted", "values"] as const;
type FactName = (typeof FACT_NAMES)[number];
type Facts = Record<FactName, number | null> & { brief: boolean; answers: number };
let FACTS: Facts = { records: null, sources: null, total: null, relevant: null, extracted: null, values: null,
                     brief: false, answers: 0 };

/** A GET of the API, as the page makes it (same origin, /api). Null when it fails. */
async function getJson(page: Page, p: string): Promise<unknown> {
  try {
    const r = await page.request.get(`/api${p}`, { timeout: 90_000 });
    return r.ok() ? await r.json() : null;
  } catch {
    return null;
  }
}

async function readFacts(page: Page): Promise<Facts> {
  const sid = encodeURIComponent(SCENARIO_ID);
  const counts = obj(await getJson(page, `/user-scenarios/${sid}/counts`));
  const prisma = obj(await getJson(page, `/user-scenarios/${sid}/prisma`));
  const extraction = obj(await getJson(page, `/user-scenarios/${sid}/extraction/status`));
  // The narrative brief in this language: read only, a GET does not generate it.
  const brief = obj(await getJson(page, `/scenarios/${sid}/evidence-brief/llm?lang=${LANG}`));
  const questions = obj(await getJson(page, `/user-scenarios/${sid}/questions?limit=1`));
  const ident = obj(prisma.identification);
  const answered = Object.values(obj(ident.by_source)).filter((n) => positive(n) !== null).length;
  return {
    records: positive(ident.total_records ?? ident.total_records_identified),
    sources: positive(ident.sources_searched ?? answered),
    total: positive(obj(counts.counts).total ?? counts.article_count),
    relevant: positive(obj(counts.counts).relevant),
    extracted: positive(extraction.n_extracted),
    values: positive(extraction.n_observations),
    brief: Boolean(brief.executive_summary) || (Array.isArray(brief.key_findings) && brief.key_findings.length > 0),
    answers: positive(questions.total) ?? 0,
  };
}

/** A caption's {figures}, from the scenario. `speech` writes them for the voice: digits with no
 *  separator, read as one number in both languages. A scene whose figure is unknown is skipped. */
function withFacts(text: string, speech = false): string {
  return text.replace(/\{(\w+)\}/g, (_m, name: string) => {
    const v = (FACTS as unknown as Json)[name];
    if (typeof v !== "number" || !(v > 0)) throw new Skip(`no figure for {${name}}`);
    return speech ? String(Math.round(v)) : v.toLocaleString(LOCALE);
  });
}

// ── Read-only on a deployment ────────────────────────────────────────────────
// Opening a tab is a read for a visitor, but on a cold cache a few GETs start the very
// computation they show and store its result: the model variables translated by an LLM into
// another language, the cluster summaries written in another language, the SEIR projection,
// the recommended actions, the article similarity graph. The tour films what is already there:
// it asks the server with pure reads what is computed (`readCaches`), redirects the reads that
// would translate or summarise to the stored copy, skips the scenes that would show it, and
// aborts every request that is not a read. The manifest lists what the guard did.
type Cached = "ready" | "partial" | "none";
type Caches = { variables: Cached; clusters: Cached; seir: boolean };
let CACHES: Caches | null = null;
const GUARD = { blocked: [] as string[], redirected: [] as string[] };

async function readCaches(page: Page): Promise<Caches> {
  const sid = encodeURIComponent(SCENARIO_ID);
  const settings = obj(await getJson(page, `/scenarios/${sid}/settings`));
  const cached = obj(settings.cached);
  const base = typeof settings.variables_lang === "string" ? settings.variables_lang.slice(0, 2).toLowerCase() : null;
  const variables: Cached = !cached.variables ? "none"
    : base === LANG || cached.variables_i18n ? "ready" : "partial";
  const status = obj(await getJson(page, `/user-scenarios/${sid}/clustering/status?lang=${LANG}`));
  const dense = (Array.isArray(status.clusters) ? status.clusters : []).map(obj).filter((c) => !c.is_noise);
  const clusters: Cached = status.status !== "done" ? "none"
    : dense.every((c) => LANG in obj(c.summaries)) ? "ready" : "partial";
  return { variables, clusters, seir: Boolean(cached.seir_projection) };
}

async function installReadOnlyGuard(page: Page, caches: Caches): Promise<void> {
  await page.route(/\/api\//, async (route) => {
    const request = route.request();
    const url = new URL(request.url());
    const where = url.pathname.replace(/^\/api/, "");
    const method = request.method();
    const block = async (why: string) => {
      GUARD.blocked.push(`${method} ${where} (${why})`);
      await route.abort("blockedbyclient");
    };
    const redirect = async (to: URL, why: string) => {
      GUARD.redirected.push(`${where}${url.search} to ${to.pathname.replace(/^\/api/, "")}${to.search} (${why})`);
      await route.continue({ url: to.toString() });
    };
    if (method !== "GET" && method !== "HEAD") return block("not a read");
    if (["force_refresh", "refresh", "force"].some((k) => url.searchParams.get(k) === "true")) {
      return block("a forced recomputation");
    }
    // Never filmed, and computed or generated when their cache is cold.
    if (/\/(recommended-actions|knowledge-graph)$/.test(where)) return block("computed when its cache is cold");
    if (/\/seir\/projection$/.test(where) && !caches.seir) return block("computed when its cache is cold");
    if (/\/(variables|model\/spec)$/.test(where)) {
      if (caches.variables === "none") return block("generated when its cache is cold");
      if (caches.variables === "partial" && url.searchParams.has("lang")) {
        const to = new URL(url);
        to.searchParams.delete("lang");
        return redirect(to, "translated by an LLM when this language is not stored");
      }
    }
    if (/^\/user-scenarios\/[^/]+\/clustering$/.test(where) && caches.clusters !== "ready") {
      const to = new URL(url);
      to.pathname = `${to.pathname}/status`;
      return redirect(to, "summaries written by an LLM when this language is not stored");
    }
    return route.continue();
  });
}

// ── Geometry, read in the page ───────────────────────────────────────────────
async function boxOf(loc: Locator): Promise<Box> {
  return loc.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  });
}

/** The box in the viewport, for a viewport screenshot (a fixed drawer), not in the page. */
async function viewBoxOf(loc: Locator): Promise<Box> {
  return loc.evaluate((el) => {
    const r = el.getBoundingClientRect();
    return { x: r.left, y: r.top, w: r.width, h: r.height };
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

/** The closest ancestor matching `selector`, above the element (the element itself if none). */
async function boxUp(loc: Locator, selector: string): Promise<Box> {
  return loc.evaluate((el, sel) => {
    const up = el.parentElement?.closest(sel) ?? el;
    const r = up.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  }, selector);
}

/** The ancestor `levels` up: a title's block when it has no card of its own. */
async function ancestorBox(loc: Locator, levels: number): Promise<Box> {
  return loc.evaluate((el, n) => {
    let up: Element = el;
    for (let i = 0; i < n && up.parentElement; i++) up = up.parentElement;
    const r = up.getBoundingClientRect();
    return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
  }, levels);
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

  /** A screenshot written to `image`, taken again when it comes back blank: a shot right
   *  after another one can catch the page before it is painted again. */
  private async shoot(image: string, options: Parameters<Page["screenshot"]>[0]): Promise<void> {
    for (let attempt = 1; ; attempt++) {
      await this.page.evaluate(() => new Promise((done) => requestAnimationFrame(() => requestAnimationFrame(done))));
      const png = await this.page.screenshot(options);
      if (png.length >= BLANK_PNG) {
        fs.writeFileSync(path.join(OUT, image), png);
        return;
      }
      if (attempt === 3) throw new Error(`the screenshot ${image} stays blank`);
      await this.page.waitForTimeout(1000);
    }
  }

  /** The page region holding `focuses`, as one image; the boxes are returned in its
   *  coordinates. Full width, at least one viewport tall, at most MAX_CLIP. */
  async region(name: string, focuses: Box[]): Promise<{ image: string; size: [number, number]; focus: Box[] }> {
    const page = this.page;
    // Room under the end of the page, so that a box at the very bottom can be framed above
    // the caption: the shot then shows the page's background below it.
    const docH = await page.evaluate((room) => {
      if (!document.getElementById("demo-room")) {
        const spacer = document.createElement("div");
        spacer.id = "demo-room";
        spacer.style.height = `${room}px`;
        document.body.appendChild(spacer);
      }
      return document.documentElement.scrollHeight;
    }, ROOM);
    const all = union(...focuses);
    let top = Math.max(0, Math.floor(all.y - 140));
    let bottom = Math.min(docH, Math.ceil(Math.max(all.y + all.h + 200, top + VIEWPORT.height)));
    if (bottom - top > MAX_CLIP) bottom = top + MAX_CLIP;
    if (bottom - top < VIEWPORT.height) top = Math.max(0, bottom - VIEWPORT.height);
    const image = this.file(name);
    await this.shoot(image, { fullPage: true, clip: { x: 0, y: top, width: VIEWPORT.width, height: bottom - top } });
    return {
      image, size: [VIEWPORT.width, bottom - top],
      focus: focuses.map((f) => ({ x: f.x, y: f.y - top, w: f.w, h: f.h })),
    };
  }

  /** The viewport as the user sees it, wherever it is scrolled. */
  async viewport(name: string): Promise<{ image: string; size: [number, number] }> {
    const image = this.file(name);
    await this.shoot(image, {});
    return { image, size: [VIEWPORT.width, VIEWPORT.height] };
  }

  /** Renders HTML at the output size in a page of its own: caption overlays (transparent)
   *  and cards. Returns the image path and the HTML's measured caption top. */
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
const CARD_BG = `background: radial-gradient(1200px 700px at 30% 20%, #17402c 0%, #0b1411 62%) #0b1411; color: #f2f6f4;`;

function captionHtml(c: { step: string; title: string; body: string }): string {
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    ${BASE_CSS}
    html, body { background: transparent; }
    .cap { position: absolute; left: 56px; right: 56px; bottom: 42px; display: flex; gap: 28px;
           align-items: flex-start; padding: 26px 36px 28px; border-radius: 28px;
           background: rgba(9, 19, 15, 0.9); border: 1px solid rgba(217, 179, 92, 0.38);
           box-shadow: 0 24px 70px rgba(0, 0, 0, 0.5); }
    .step { flex: none; width: 66px; height: 66px; border-radius: 50%; background: #d9b35c; color: #0b1411;
            font-weight: 800; font-size: ${c.step.length > 1 ? 28 : 34}px; display: flex; align-items: center;
            justify-content: center; }
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
    body { ${CARD_BG} display: flex; align-items: center; }
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

/** Everything a review can take away, as a card: kept above the caption band. */
function takeawayHtml(card: TakeawayCard): string {
  const groups = card.groups.map((g) => `<div class="group"><div class="gt">${esc(g.title)}</div>${
    g.items.map((i) => `<div class="it">${esc(i)}</div>`).join("")}</div>`).join("");
  return `<!doctype html><html><head><meta charset="utf-8"><style>
    ${BASE_CSS}
    body { ${CARD_BG} }
    .wrap { position: absolute; left: 140px; right: 140px; top: 92px; }
    h1 { font-size: 60px; margin: 0; letter-spacing: -0.02em; }
    h2 { font-size: 30px; font-weight: 500; color: #3fb97e; margin: 14px 0 0; }
    .rule { width: 120px; height: 5px; border-radius: 3px; background: #d9b35c; margin: 28px 0 34px; }
    .grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 34px 56px; }
    .gt { font-size: 22px; font-weight: 700; letter-spacing: 0.1em; text-transform: uppercase; color: #d9b35c; margin-bottom: 8px; }
    .it { font-size: 25px; line-height: 1.3; color: rgba(242, 246, 244, 0.84); margin-top: 7px; }
  </style></head><body>
    ${badgeHtml()}
    <div class="wrap"><h1>${esc(card.title)}</h1><h2>${esc(card.subtitle)}</h2><div class="rule"></div>
      <div class="grid">${groups}</div></div>
  </body></html>`;
}

// ── The page ─────────────────────────────────────────────────────────────────
async function settle(page: Page, ms = 700): Promise<void> {
  await page.waitForLoadState("networkidle", { timeout: 15_000 }).catch(() => undefined);
  await page.waitForTimeout(ms);
}

/** True once `loc` is visible, false after `ms`: a scene checks what it films before
 *  waiting on it, and skips when it is not there. */
async function shows(loc: Locator, ms = 15_000): Promise<boolean> {
  return loc.first().waitFor({ state: "visible", timeout: ms }).then(() => true, () => false);
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

const SECTION = T.scenarioDetail.page.sections;

/** A scenario section, then one of its sub-tabs. Clicking the open section keeps its sub-tab. */
async function openSub(page: Page, section: string, sub: string): Promise<void> {
  await openTab(page, section);
  await openTab(page, sub);
}

const toCorpus = (page: Page) => openSub(page, SECTION.review, T.scenarioDetail.review.subCorpus);
const toPrisma = (page: Page) => openSub(page, SECTION.review, T.scenarioDetail.review.subPrisma);
const toEvidence = (page: Page) => openSub(page, SECTION.evidence, T.scenarioDetail.evidenceTab.subEvidences);
const toExtraction = (page: Page) => openSub(page, SECTION.evidence, T.scenarioDetail.evidenceTab.subExtraction);

/** The scenario page, from wherever the tour is. The app keeps the open scenario in the
 *  address (?scenario=), which is also how a shared link opens it. */
async function openScenario(page: Page): Promise<void> {
  if (new URL(page.url()).searchParams.get("scenario") === SCENARIO_ID) return;
  await page.goto(`/?scenario=${encodeURIComponent(SCENARIO_ID)}`);
  await expect(page.getByRole("heading", { level: 2 }).first()).toBeVisible({ timeout: 30_000 });
  await settle(page);
}

/** A native select's list is drawn by the system, outside the page: no screenshot shows it.
 *  Its real options are shown in the page instead, as a list box, until `close` is called. */
async function openSelect(sel: Locator): Promise<() => Promise<void>> {
  await sel.evaluate((el) => {
    const s = el as HTMLSelectElement;
    s.setAttribute("size", String(Math.min(s.options.length, 10)));
  });
  return async () => {
    await sel.evaluate((el) => el.removeAttribute("size")).catch(() => undefined);
  };
}

// ── The scenes ───────────────────────────────────────────────────────────────
type Ctx = { page: Page; rec: Recorder };
type Film = (ctx: Ctx) => Promise<Beat[]>;

const beat = (img: { image: string; size: [number, number] }, focus: Box | null, hold: number,
              spot = true, cut = false): Beat => ({ image: img.image, size: img.size, focus, spot, hold, cut });

/** A field being typed, a frame every few characters, then held with its spotlight. Never sent. */
async function typed(rec: Recorder, input: Locator, text: string, focus: Box, name: string, hold: number): Promise<Beat[]> {
  const beats: Beat[] = [];
  const steps = 8;
  for (let i = 1; i <= steps; i++) {
    await input.fill(text.slice(0, Math.round((text.length * i) / steps)));
    await input.page().waitForTimeout(60);
    const r = await rec.region(name, [focus]);
    beats.push(beat(r, pad(r.focus[0], 10), i === steps ? hold : 0.12, i === steps, true));
  }
  return beats;
}

/** The seconds a scene spends between its holds: the dip that opens it, then a move, a
 *  crossfade or a cut before each further beat. */
function transitions(beats: Beat[]): number {
  return beats.reduce((s, b, i) =>
    s + (i === 0 ? DIP : b.cut ? 0 : b.image === beats[i - 1].image ? MOVE : FADE), 0);
}

/** Holds the beats long enough for `chars` characters to be read (or said), transitions
 *  included. Typing frames keep their pace; the time is shared between the held beats. */
function fitToCaption(beats: Beat[], chars: number): Beat[] {
  const need = chars / READ_CPS + 1.0;
  const have = beats.reduce((s, b) => s + b.hold, 0) + transitions(beats);
  const held = beats.filter((b) => b.hold >= 0.5);
  if (have >= need || !held.length) return beats;
  const extra = (need - have) / held.length;
  return beats.map((b) => (b.hold >= 0.5 ? { ...b, hold: Math.round((b.hold + extra) * 100) / 100 } : b));
}

const FILMS: Record<SceneId, Film> = {
  async ask({ page, rec }) {
    await page.goto("/");
    const input = page.getByPlaceholder(T.search.queryPlaceholder);
    await expect(input).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const panel = await cardOf(input, true);
    const beats: Beat[] = [];
    const steps = 10;
    for (let i = 1; i <= steps; i++) {             // typing, one frame per few characters
      await input.fill(QUESTION.slice(0, Math.round((QUESTION.length * i) / steps)));
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
    await toCorpus(page);
    const thresholdLabel = page.getByText(T.scenarioDetail.seuil.thresholdLabel).first();
    await expect(thresholdLabel).toBeVisible({ timeout: 30_000 });
    await expect(page.locator("h4").first()).toBeVisible({ timeout: 60_000 });
    await settle(page);
    const header = union(await boxOf(page.getByRole("heading", { level: 2 }).first()),
                         await boxOf(tabButton(page, SECTION.review)));
    const threshold = await cardOf(thresholdLabel);
    // The two scores of every record, and how far each has got.
    const scoresLabel = page.getByText(T.scenarioDetail.corpus.relevanceScores, { exact: true }).first();
    const scores = (await shows(scoresLabel, 10_000)) ? await cardOf(scoresLabel) : threshold;
    const r = await rec.region("corpus", [header, threshold, scores]);
    return [beat(r, pad(r.focus[0], 16), 2.0, false), beat(r, pad(r.focus[1], 8), 3.0), beat(r, pad(r.focus[2], 8), 3.0)];
  },

  async sources({ page, rec }) {
    await toCorpus(page);
    const exportMenu = page.getByLabel(T.scenarioDetail.corpus.exportRelevant, { exact: true }).first();
    await expect(exportMenu).toBeVisible({ timeout: 60_000 });
    await settle(page);
    // The counts and the export menu of the relevant articles, opened on its formats.
    const close = await openSelect(exportMenu);
    await page.waitForTimeout(250);
    const menu = union(await boxUp(exportMenu, "div.flex-wrap"), await boxOf(exportMenu));
    const r1 = await rec.region("sources-export", [menu]);
    await close();
    // The ranked list beside the sources and the years, the histogram at its latest years.
    const yearsTitle = page.getByText(T.scenarioDetail.corpus.distributionByYear, { exact: true }).first();
    await yearsTitle.evaluate((el) => {
      const scroller = el.closest('[class*="rounded-3xl"]')?.querySelector(".overflow-x-auto");
      if (scroller) scroller.scrollLeft = scroller.scrollWidth;
    });
    await page.waitForTimeout(200);
    const sources = await cardOf(page.getByText(T.scenarioDetail.corpus.literatureSources, { exact: true }).first());
    const list = capHeight(union(await cardOf(page.locator("h4").nth(0)), await cardOf(page.locator("h4").nth(1))), 420);
    const years = await cardOf(yearsTitle);
    const r2 = await rec.region("sources", [union(list, sources), years]);
    return [beat(r1, pad(r1.focus[0], 10), 3.2), beat(r2, pad(r2.focus[0], 10), 3.4), beat(r2, pad(r2.focus[1], 10), 2.6)];
  },

  async article({ page, rec }) {
    await toCorpus(page);
    const titles = page.locator("h4");
    await expect(titles.first()).toBeVisible({ timeout: 60_000 });
    const preferred = titles.filter({ hasText: /wastewater|eaux us/i }).first();
    const title = (await preferred.count()) ? preferred : titles.first();
    await title.click();
    await expect(page.getByText(T.scenarioDetail.articleRow.screeningDecision).first()).toBeVisible({ timeout: 15_000 });
    // Its PICO arrives after the click: wait until it has loaded, found or not.
    await expect(page.getByText(T.scenarioDetail.articleRow.loadingPico)).toHaveCount(0, { timeout: 20_000 })
      .catch(() => undefined);
    await settle(page, 500);
    const card = capHeight(await cardOf(title), 900);
    const picoLabel = page.getByText(`P : ${T.scenarioDetail.articleRow.populationLabel}`, { exact: true }).first();
    const pico = (await picoLabel.count()) ? await boxUp(picoLabel, "div.rounded-xl.space-y-2") : card;
    const abstractLabel = page.getByText(T.scenarioDetail.articleRow.abstract, { exact: true }).first();
    const abstract = (await abstractLabel.count()) ? capHeight(await ancestorBox(abstractLabel, 1), 420) : card;
    const r = await rec.region("article", [card, pico, abstract]);
    return [beat(r, pad(r.focus[0], 8), 3.0), beat(r, pad(r.focus[1], 8), 3.2), beat(r, pad(r.focus[2], 8), 3.0)];
  },

  async prisma({ page, rec }) {
    await toPrisma(page);
    const stage1 = page.getByText(T.scenarioDetail.prisma.stage1).first();
    await expect(stage1).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const s1 = await cardOf(stage1);
    // Stage 1 down to its counts: the databases have a scene of their own.
    const lastRow = await boxOf(page.getByText(T.scenarioDetail.prisma.embeddedSearchable, { exact: true })
      .or(page.getByText(T.scenarioDetail.prisma.identifiedLibrary, { exact: true })).last());
    const counts = { x: s1.x, y: s1.y, w: s1.w, h: lastRow.y + lastRow.h + 14 - s1.y };
    const s2 = await cardOf(page.getByText(T.scenarioDetail.prisma.stage2).first());
    const s3 = await cardOf(page.getByText(T.scenarioDetail.prisma.stage3).first());
    const s4 = await cardOf(page.getByText(T.scenarioDetail.prisma.stage4).first());
    const r = await rec.region("prisma", [counts, s2, union(s3, s4)]);
    return [beat(r, pad(r.focus[0], 10), 3.2), beat(r, pad(r.focus[1], 10), 3.0), beat(r, pad(r.focus[2], 10), 3.0)];
  },

  async prismaSources({ page, rec }) {
    await toPrisma(page);
    const stage1 = page.getByText(T.scenarioDetail.prisma.stage1).first();
    await expect(stage1).toBeVisible({ timeout: 30_000 });
    await settle(page);
    // Each database with its count, then the notes on what each one received.
    const parts = await stage1.evaluate((el) => {
      const card = el.closest('[class*="rounded-2xl"]');
      const chips = card?.querySelector("div.flex-wrap.border-t");
      if (!card || !chips) return null;
      const c = card.getBoundingClientRect();
      const k = chips.getBoundingClientRect();
      const sx = window.scrollX, sy = window.scrollY;
      return {
        chips: { x: k.left + sx, y: k.top + sy, w: k.width, h: k.height },
        notes: { x: c.left + sx, y: k.bottom + sy, w: c.width, h: c.bottom - k.bottom },
      };
    });
    if (!parts) throw new Skip("no database counts on the PRISMA card");
    const notes = capHeight(parts.notes, 560);
    const r = await rec.region("prisma-sources", [parts.chips, notes]);
    const beats = [beat(r, pad(r.focus[0], 10), 3.6)];
    if (notes.h > 40) beats.push(beat(r, pad(r.focus[1], 6), 4.2));
    return beats;
  },

  async strategy({ page, rec }) {
    await openTab(page, SECTION.queries);
    const q = T.scenarioDetail.queries;
    const title = page.getByText(q.title, { exact: true }).first();
    await expect(title).toBeVisible({ timeout: 30_000 });
    // The stored strategy is read when the tab opens: wait for its general query.
    const general = page.getByText(q.generalQuery, { exact: true }).first();
    if (!(await shows(general, 30_000))) throw new Skip("no search strategy stored for this scenario");
    await settle(page);
    const queries = capHeight(await ancestorBox(page.getByText(`${q.booleanQueries} (`).first(), 2), 360);
    // The strategy as it was sent: the general query and its PubMed version, then the
    // explanation stored with them.
    const card = (label: string) => boxUp(page.getByText(label, { exact: true }).first(), '[class*="rounded-xl"]');
    const sent = capHeight(union(await card(q.generalQuery), await card(q.pubmedQueryMesh)), 640);
    const explained = (await page.getByText(q.explanation, { exact: true }).count()) ? await card(q.explanation) : null;
    const r = await rec.region("strategy", [union(await boxOf(title), queries), sent, explained ?? sent]);
    const beats = [beat(r, pad(r.focus[0], 10), 3.0), beat(r, pad(r.focus[1], 8), 3.6)];
    if (explained) beats.push(beat(r, pad(r.focus[2], 8), 3.0));
    return beats;
  },

  async doubleBlind({ page, rec }) {
    await openSub(page, SECTION.review, T.scenarioDetail.review.subScreening);
    const kappa = page.getByText(T.scenarioDetail.doubleBlind.statKappa, { exact: true }).first();
    await expect(kappa).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const row = await gridOf(kappa);
    const title = await boxOf(page.getByText(T.scenarioDetail.doubleBlind.title, { exact: true }).first());
    const r = await rec.region("double-blind", [union(title, row)]);
    return [beat(r, pad(r.focus[0], 14), 4.2)];
  },

  async evidence({ page, rec }) {
    await toEvidence(page);
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

  async epiParams({ page, rec }) {
    await toEvidence(page);
    const title = page.getByText(T.scenarioDetail.epiParams.title, { exact: true }).first();
    if (!(await shows(title, 30_000))) throw new Skip("no article of this corpus reports an epidemiological parameter");
    const panel = title.locator("xpath=ancestor::div[contains(@class, 'rounded-2xl')][1]");
    // The pooled values come with the model variables: a row each, or none yet.
    if (!(await shows(panel.locator("code"), 30_000))) throw new Skip("no epidemiological parameter pooled yet");
    await settle(page);
    const r = await rec.region("epi-params", [await boxOf(panel)]);
    return [beat(r, pad(r.focus[0], 10), 5.0)];
  },

  async brief({ page, rec }) {
    if (!FACTS.brief) throw new Skip(`no narrative brief in ${LANG} yet`);
    await toEvidence(page);
    const e = T.scenarioDetail.evidences;
    const summary = page.getByText(e.executiveSummary, { exact: true }).first();
    if (!(await shows(summary, 30_000))) throw new Skip("no narrative brief on screen");
    await settle(page);
    // Its level of evidence and GRADE, its summary, its key findings, and how it leaves.
    const level = page.getByText(new RegExp(`^${escapeRe(e.levelPrefix)} `)).first();
    const top = (await level.count()) ? union(await boxOf(level), await cardOf(summary)) : await cardOf(summary);
    const findingsLabel = page.getByText(e.keyFindings, { exact: true }).first();
    const findings = (await findingsLabel.count()) ? capHeight(await ancestorBox(findingsLabel, 1), 520) : top;
    const exportsBox = union(await boxOf(page.getByRole("button", { name: e.exportPdf }).first()),
                             await boxOf(page.getByRole("button", { name: e.exportReport }).first()));
    const r1 = await rec.region("brief", [top, findings]);
    const r2 = await rec.region("brief-exports", [exportsBox]);
    return [beat(r1, pad(r1.focus[0], 10), 4.0), beat(r1, pad(r1.focus[1], 10), 3.0), beat(r2, pad(r2.focus[0], 12), 2.6)];
  },

  async claims({ page, rec }) {
    if (!FACTS.brief) throw new Skip(`no narrative brief in ${LANG} yet`);
    await toEvidence(page);
    const e = T.scenarioDetail.evidences;
    const title = page.getByText(e.claims.title, { exact: true }).first();
    if (!(await shows(title, 30_000))) throw new Skip("no graded claim in this brief");
    await settle(page);
    // The claims: how a strength is computed, then the claims with their articles.
    const blocks = await title.evaluate((el) => {
      const box = (x: Element) => {
        const r = x.getBoundingClientRect();
        return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
      };
      const block = el.parentElement;
      const list = block?.querySelector(":scope > div.space-y-2");
      return { head: block ? box(block) : box(el), cards: list ? Array.from(list.children).slice(0, 4).map(box) : [] };
    });
    const [c1, c2, c3, c4] = blocks.cards;
    const first = c2 ? union(blocks.head, c1, c2) : blocks.head;
    const r1 = await rec.region("claims", [{ ...first, h: Math.min(first.h, c2 ? c2.y + c2.h - first.y : first.h) },
                                           c3 ? union(c3, c4 ?? c3) : first]);
    const beats = [beat(r1, pad(capHeight(r1.focus[0], 640), 8), 4.0)];
    if (c3) beats.push(beat(r1, pad(capHeight(r1.focus[1], 520), 8), 3.2));
    // The recommended actions, then the research gaps beside the gap matrix counted in SQL.
    const actionsLabel = page.getByText(e.recommendedActions, { exact: true }).first();
    const gapsLabel = page.getByText(e.researchGaps, { exact: true }).first();
    const matrixLabel = page.getByText(e.gaps.title, { exact: true }).first();
    const focuses: Box[] = [];
    if (await actionsLabel.count()) focuses.push(capHeight(await cardOf(actionsLabel), 520));
    if (await gapsLabel.count()) {
      const gaps = await ancestorBox(gapsLabel, 1);
      focuses.push(capHeight((await matrixLabel.count()) ? union(gaps, await ancestorBox(matrixLabel, 1)) : gaps, 760));
    }
    if (focuses.length) {
      const r2 = await rec.region("claims-gaps", focuses);
      r2.focus.forEach((f, i) => beats.push(beat(r2, pad(f, 8), i === 0 ? 3.4 : 4.0)));
    }
    return beats;
  },

  async pico({ page, rec }) {
    await openSub(page, SECTION.evidence, T.scenarioDetail.evidenceTab.subPicoTable);
    await expect(page.getByText(T.scenarioDetail.pico.title).first()).toBeVisible({ timeout: 30_000 });
    const table = page.locator("table").first();
    await expect(table).toBeVisible({ timeout: 60_000 });
    await settle(page);
    const top = union(await boxOf(page.getByText(T.scenarioDetail.pico.title).first()), capHeight(await boxOf(table), 620));
    const r = await rec.region("pico", [top]);
    return [beat(r, pad(r.focus[0], 12), 4.6)];
  },

  async extraction({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const x = T.scenarioDetail.extraction;
    const title = page.getByText(x.title, { exact: true }).first();
    await expect(title).toBeVisible({ timeout: 90_000 });
    if (await page.getByText(x.nothingYet).count()) throw new Skip("nothing extracted in this scenario yet");
    await settle(page, 1200);
    const head = union(await ancestorBox(title, 2), await gridOf(page.getByText(x.statRelevant, { exact: true }).first()));
    // The run: all articles read; and, said plainly, how many were read from the abstract only.
    let run = await boxUp(page.getByText(x.progressLabel, { exact: true }).first(), '[class*="rounded-xl"]');
    const warningText = (x.abstractWarning.split("{n}")[1] ?? x.abstractWarning).trim().slice(0, 40);
    const warning = page.getByText(new RegExp(escapeRe(warningText))).first();
    if (await warning.count()) run = union(run, await boxUp(warning, '[class*="rounded-xl"]'));
    const r = await rec.region("extraction", [head, run]);
    return [beat(r, pad(r.focus[0], 10), 4.0), beat(r, pad(r.focus[1], 8), 3.6)];
  },

  async coverage({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const title = page.getByText(T.scenarioDetail.extraction.coverageTitle, { exact: true }).first();
    if (!(await shows(title, 60_000))) throw new Skip("no coverage of the extraction yet");
    await settle(page);
    const r = await rec.region("coverage", [await ancestorBox(title, 2)]);
    return [beat(r, pad(r.focus[0], 10), 5.0)];
  },

  async sheets({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const x = T.scenarioDetail.extraction;
    const title = page.getByText(x.sheetsTitle, { exact: true }).first();
    if (!(await shows(title, 60_000))) throw new Skip("no extracted row yet");
    await settle(page);
    // The five sheets first: opening the codebook moves them down.
    const r1 = await rec.region("sheets", [await ancestorBox(title, 1)]);
    const beats = [beat(r1, pad(r1.focus[0], 10), 4.0)];
    // The codebook, opened: which one is in use, its download, the labels still to map.
    const cbTitle = page.getByText(x.codebook.title, { exact: true }).first();
    if (await cbTitle.count()) {
      const toggle = cbTitle.locator("xpath=ancestor::button[1]");
      if ((await toggle.getAttribute("aria-expanded")) !== "true") await toggle.click();
      await shows(page.getByText(x.codebook.sourceDefault).or(page.getByText(lead(x.codebook.sourceCustom))), 15_000);
      await settle(page, 600);
      const r2 = await rec.region("codebook", [capHeight(await cardOf(cbTitle), 640)]);
      beats.push(beat(r2, pad(r2.focus[0], 8), 4.0));
    }
    return beats;
  },

  async quotes({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const x = T.scenarioDetail.extraction;
    await expect(page.getByText(x.title, { exact: true }).first()).toBeVisible({ timeout: 90_000 });
    await settle(page);
    // An article of the table read in full text, all its quotes found, a handful of rows.
    const index = await page.evaluate((labels) => {
      const table = Array.from(document.querySelectorAll("table")).find((t) => {
        const heads = Array.from(t.querySelectorAll("thead th")).map((th) => th.textContent?.trim());
        return heads.includes(labels.reports) && heads.includes(labels.quotes);
      });
      if (!table) return -1;
      table.setAttribute("data-demo", "extraction-articles");
      const rows = Array.from(table.querySelectorAll(":scope > tbody > tr"));
      let fallback = -1;
      for (let i = 0; i < rows.length; i++) {
        const cells = rows[i].querySelectorAll(":scope > td");
        if (cells.length < 7) continue;
        const n = parseInt(cells[4].textContent ?? "", 10);
        if (cells[5].textContent?.trim() !== labels.ok || !(n >= 3)) continue;
        if (fallback < 0) fallback = i;
        if (n <= 10 && cells[2].textContent?.includes(labels.fulltext)) return i;
      }
      return fallback;
    }, { reports: x.colReports, quotes: x.colQuotes, ok: x.quotesOk, fulltext: x.sourceFulltext });
    if (index < 0) throw new Skip("no extracted article with its quotes found");
    const row = page.locator('table[data-demo="extraction-articles"] > tbody > tr').nth(index);
    await row.scrollIntoViewIfNeeded();
    await row.click();
    const panel = row.locator("xpath=following-sibling::tr[1]");
    const rows = panel.locator("table").first();
    await expect(rows).toBeVisible({ timeout: 30_000 });
    await settle(page, 600);
    // Its rows: the sheet, the group, the value, the cases and the group size, the quote.
    const panelBox = capHeight(await boxOf(panel), 760);
    const quoteColumn = await rows.evaluate((t, label) => {
      const heads = Array.from(t.querySelectorAll("thead th"));
      const qi = heads.findIndex((th) => th.textContent?.trim() === label);
      const cells = Array.from(t.querySelectorAll("tbody tr")).slice(0, 3)
        .map((r) => r.children[qi]).filter((c): c is Element => Boolean(c));
      if (qi < 0 || !cells.length) return null;
      const rs = cells.map((c) => c.getBoundingClientRect());
      const x0 = Math.min(...rs.map((b) => b.left)), y0 = Math.min(...rs.map((b) => b.top));
      const x1 = Math.max(...rs.map((b) => b.right)), y1 = Math.max(...rs.map((b) => b.bottom));
      return { x: x0 + window.scrollX, y: y0 + window.scrollY, w: x1 - x0, h: y1 - y0 };
    }, x.obsQuote);
    const r1 = await rec.region("quotes", [panelBox, quoteColumn ?? panelBox]);
    const beats = [beat(r1, pad(r1.focus[0], 8), 3.4), beat(r1, pad(r1.focus[1], 10), 3.0)];
    // View in the paper: the article's own text, the quote highlighted where it was found.
    const view = panel.getByRole("button", { name: x.source.view }).first();
    if (await view.count()) {
      await view.click();
      const drawer = page.getByRole("dialog", { name: x.source.title });
      await expect(drawer).toBeVisible({ timeout: 15_000 });
      const mark = drawer.locator("mark").first();
      await shows(mark, 15_000);
      await settle(page, 600);
      const shot = await rec.viewport("quotes-paper");
      const d = await viewBoxOf(drawer);
      const top = await viewBoxOf(drawer.locator("p").first());
      const focus = (await mark.count())
        ? union({ x: d.x, y: d.y, w: d.w, h: 10 }, top, await viewBoxOf(mark)) : d;
      beats.push(beat(shot, capHeight(focus, d.h), 4.2));
      await page.keyboard.press("Escape");
      await expect(drawer).toBeHidden({ timeout: 10_000 });
      // The focus comes back to the link, and the pointer still hovers it: neither belongs
      // in the next shot.
      await page.evaluate(() => (document.activeElement as HTMLElement | null)?.blur());
      await page.mouse.move(0, 0);
    }
    // The review cell of the first rows, their status and the buttons to accept, edit or
    // reject, framed close and never clicked. Measured now: panels still loading above the
    // table while the paper was open can have moved it down.
    const accept = panel.getByRole("button", { name: x.review.accept, exact: true });
    const reviewCells: Box[] = [];
    for (let i = 0; i < Math.min(3, await accept.count()); i++) reviewCells.push(await boxUp(accept.nth(i), "td"));
    if (reviewCells.length) {
      const r2 = await rec.region("quotes-review", [union(...reviewCells)]);
      beats.push(beat(r2, pad(r2.focus[0], 12), 3.0));
    }
    return beats;
  },

  async downloads({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const x = T.scenarioDetail.extraction;
    const excel = page.getByRole("link", { name: x.downloadExcel }).first();
    if (!(await shows(excel, 60_000))) throw new Skip("nothing to download yet");
    await settle(page);
    // Excel in the review's template, CSV, and the report in PDF or Word.
    const r1 = await rec.region("downloads", [await boxUp(excel, "div.flex-wrap")]);
    const beats = [beat(r1, pad(r1.focus[0], 10), 4.0)];
    // The review of the rows, and the reviewed rows as an annotated dataset.
    const dataset = page.getByRole("link", { name: x.review.dataset }).first();
    if (await dataset.count()) {
      const r2 = await rec.region("review", [capHeight(await cardOf(dataset), 620)]);
      beats.push(beat(r2, pad(r2.focus[0], 8), 3.6));
    }
    return beats;
  },

  async geography({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const title = page.getByText(T.scenarioDetail.extraction.geo.title, { exact: true }).first();
    if (!(await shows(title, 60_000))) throw new Skip("no study place resolved yet");
    await settle(page);
    const boxes = await title.evaluate((el) => {
      const box = (x: Element) => {
        const r = x.getBoundingClientRect();
        return { x: r.left + window.scrollX, y: r.top + window.scrollY, w: r.width, h: r.height };
      };
      const panelEl = el.closest("div.space-y-3");
      const bars = panelEl?.querySelector('[class*="rounded-2xl"]');
      // A country with its NUTS regions: Italy when it has some, else the first that has.
      // Its name and its region chips only, so the camera comes close enough to read them.
      const chips = Array.from(panelEl?.querySelectorAll("span[title]") ?? [])
        .filter((s) => /^[A-Z]{2}[A-Z0-9]{1,3}$/.test(s.getAttribute("title") ?? ""));
      const chip = chips.find((s) => s.getAttribute("title")?.startsWith("IT")) ?? chips[0];
      const name = chip?.parentElement?.parentElement?.querySelector("span[title]");
      const parts = [name, ...Array.from(chip?.parentElement?.children ?? [])]
        .filter((e): e is Element => Boolean(e)).map((e) => e.getBoundingClientRect());
      const x0 = Math.min(...parts.map((r) => r.left)), y0 = Math.min(...parts.map((r) => r.top));
      const x1 = Math.max(...parts.map((r) => r.right)), y1 = Math.max(...parts.map((r) => r.bottom));
      const country = chip ? { x: x0 + window.scrollX, y: y0 + window.scrollY, w: x1 - x0, h: y1 - y0 } : null;
      return { head: box(el.parentElement ?? el), bars: bars ? box(bars) : null, country };
    });
    const top = union(boxes.head, capHeight(boxes.bars ?? boxes.head, 620));
    const r = await rec.region("geography", [top, boxes.country ?? top]);
    const beats = [beat(r, pad(r.focus[0], 10), 4.0)];
    if (boxes.country) beats.push(beat(r, pad(r.focus[1], 8), 3.4));
    return beats;
  },

  async pooled({ page, rec }) {
    if (!FACTS.extracted) throw new Skip("nothing extracted in this scenario yet");
    await toExtraction(page);
    const p = T.scenarioDetail.extraction.pooled;
    const title = page.getByText(p.title, { exact: true }).first();
    if (!(await shows(title, 60_000))) throw new Skip("no pooled estimate yet");
    const panel = title.locator("xpath=ancestor::div[contains(@class, 'space-y-3')][1]");
    const firstRow = panel.locator("table tbody tr").first();
    if (!(await shows(firstRow, 30_000))) throw new Skip("no label has enough studies to pool");
    await settle(page);
    // The table: pooled value, interval, prediction interval, disagreement. And its download.
    const tableTop = union(await ancestorBox(title, 1), capHeight(await boxOf(panel.locator("table").first()), 430));
    const downloadButton = panel.getByRole("button", { name: p.download }).first();
    const download = (await downloadButton.count()) ? await boxOf(downloadButton) : tableTop;
    const r1 = await rec.region("pooled", [tableTop, download]);
    const beats = [beat(r1, pad(r1.focus[0], 8), 4.0), beat(r1, pad(r1.focus[1], 12), 2.4)];
    // The forest plot of a label with 5 to 12 studies, small enough to read whole, and none
    // of them giving the composition of its sample; else the first label.
    await panel.locator("table").first().evaluate((t) => {
      const rows = Array.from(t.querySelectorAll(":scope > tbody > tr")).filter((r) => r.children.length >= 7);
      const fits = rows.find((r) => {
        const k = parseInt(r.children[2].textContent ?? "", 10);
        return k >= 5 && k <= 12 && !r.children[0].querySelector('[class*="text-gold"]');
      });
      (fits ?? rows[0])?.setAttribute("data-demo", "pooled-row");
    });
    await panel.locator('tr[data-demo="pooled-row"]').click();
    const forest = panel.getByRole("img", { name: p.forest }).first();
    if (await shows(forest, 10_000)) {
      await settle(page, 400);
      const r2 = await rec.region("pooled-forest", [await boxUp(forest, "td")]);
      beats.push(beat(r2, pad(r2.focus[0], 8), 4.0));
    }
    // Comparisons between groups: an odds ratio, and its own forest plot.
    const cmpTitle = panel.getByText(p.comparisonsTitle, { exact: true }).first();
    if (await cmpTitle.count()) {
      const block = cmpTitle.locator("xpath=..");
      await block.locator("button").first().click();
      await shows(block.getByRole("img", { name: p.forest }), 10_000);
      await settle(page, 400);
      const r3 = await rec.region("comparisons", [capHeight(await boxOf(block), 700)]);
      beats.push(beat(r3, pad(r3.focus[0], 8), 4.2));
    }
    return beats;
  },

  async assistant({ page, rec }) {
    await openTab(page, SECTION.assistant);
    const g = T.scenarioDetail.rag;
    const title = page.getByText(g.title, { exact: true }).first();
    await expect(title).toBeVisible({ timeout: 30_000 });
    // What it can read: the relevant articles indexed for it. Its card arrives after the
    // title and pushes the question box down, so every box is measured once it is there.
    const indexing = page.getByText(g.indexingReady).or(page.getByText(g.indexingInProgress)).first();
    await shows(indexing, 20_000);
    await settle(page);
    const head = (await indexing.count()) ? union(await boxOf(title), await cardOf(indexing)) : await boxOf(title);
    const r = await rec.region("assistant", [head]);
    const beats = [beat(r, pad(r.focus[0], 12), 3.0)];
    // A question typed, never sent: asking calls the model, and the answer is kept.
    const input = page.getByPlaceholder(g.questionPlaceholder);
    beats.push(...await typed(rec, input, ASSISTANT_QUESTION, await boxUp(input, "div.flex"), "assistant-typing", 3.4));
    await input.fill("");
    return beats;
  },

  async assistantAnswer({ page, rec }) {
    if (!FACTS.answers) throw new Skip("no answer kept for this scenario yet");
    await openTab(page, SECTION.assistant);
    const g = T.scenarioDetail.rag;
    const history = page.getByText(new RegExp(`^${escapeRe(g.historyTitle)} \\(`)).first();
    if (!(await shows(history, 20_000))) throw new Skip("the kept answers did not load");
    const item = history.locator("xpath=ancestor::div[contains(@class, 'rounded-3xl')][1]").locator("div.rounded-2xl").first();
    // Opening it is a read: never "Ask again", never "Delete".
    await item.locator("button").first().click();
    await shows(item.locator("p.whitespace-pre-wrap"), 10_000);
    await settle(page, 500);
    const itemBox = capHeight(await boxOf(item), 760);
    const exportsLabel = item.getByText(g.historyExport, { exact: true }).first();
    const exportsRow = (await exportsLabel.count()) ? await boxUp(exportsLabel, "div.flex-wrap") : itemBox;
    const r = await rec.region("assistant-answer", [itemBox, exportsRow]);
    return [beat(r, pad(r.focus[0], 8), 5.0), beat(r, pad(r.focus[1], 10), 2.6)];
  },

  async clusters({ page, rec }) {
    if (CACHES && CACHES.clusters !== "ready") throw new Skip(`the cluster summaries are not written in ${LANG} yet`);
    await openTab(page, SECTION.viz);
    const umap = page.getByText(T.scenarioDetail.clustering.umapProjection).first();
    await expect(umap).toBeVisible({ timeout: 180_000 });   // computed in the background on a cold cache
    await settle(page, 1500);
    const map = await cardOf(umap);
    const detail = await cardOf(page.getByText(T.scenarioDetail.clustering.topKeywords).first()).catch(() => map);
    const r = await rec.region("clusters", [map, union(map, detail)]);
    const beats = [beat(r, pad(r.focus[0], 10), 2.4), beat(r, pad(r.focus[1], 10), 3.4)];
    // A group's articles leave in the same six formats.
    const exportMenu = page.getByLabel(T.scenarioDetail.clustering.exportCluster, { exact: true }).first();
    if (await exportMenu.count()) {
      const close = await openSelect(exportMenu);
      await page.waitForTimeout(250);
      const r2 = await rec.region("clusters-export", [union(await boxOf(exportMenu), await boxUp(exportMenu, "div.flex-wrap"))]);
      await close();
      beats.push(beat(r2, pad(r2.focus[0], 10), 2.4));
    }
    return beats;
  },

  async concepts({ page, rec }) {
    await openSub(page, SECTION.viz, T.scenarioDetail.vizTab.subKnowledgeGraph);
    const k = T.scenarioDetail.knowledgeGraph;
    const title = page.getByText(k.conceptTitle).first();
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
    const r1 = await rec.region("concepts", [union(await boxOf(title), graph)]);
    const beats = [beat(r1, pad(r1.focus[0], 10), 3.0)];
    // A concept clicked: its links and its articles open beside the map. The most cited
    // population (who was studied), else exposure, else host, else the most cited of all;
    // a node's circle grows with its articles, and its colour is its column's.
    const nodes = page.locator("svg g.cursor-pointer");
    const kinds = [k.conceptTypes.population, k.conceptTypes.exposure, k.conceptTypes.host];
    const chosen = await nodes.evaluateAll((gs, names) => {
      const svg = gs[0]?.closest("svg");
      const colour = new Map<string, string>();
      for (const head of Array.from(svg?.querySelectorAll(":scope > g:not(.cursor-pointer)") ?? [])) {
        const name = head.querySelector("text")?.textContent?.trim();
        const fill = head.querySelector("circle")?.getAttribute("fill");
        if (name && fill) colour.set(name, fill);
      }
      const dot = (g: Element) => {
        const c = Array.from(g.querySelectorAll("circle")).filter((x) => x.getAttribute("fill") !== "none").pop();
        return { fill: c?.getAttribute("fill") ?? "", r: parseFloat(c?.getAttribute("r") ?? "0") };
      };
      for (const fill of [...names.map((n) => colour.get(n)).filter(Boolean), null]) {
        let best = -1;
        let size = 0;
        gs.forEach((g, i) => {
          const d = dot(g);
          if ((fill === null || d.fill === fill) && d.r > size) { size = d.r; best = i; }
        });
        if (best >= 0) return best;
      }
      return -1;
    }, kinds);
    if (chosen >= 0) {
      // Dispatched on the node itself: a click at its centre can land on a neighbour's label.
      await nodes.nth(chosen).dispatchEvent("click");
      const side = page.getByText(k.connectedConcepts, { exact: true }).first();
      if (await shows(side, 5_000)) {
        await settle(page, 500);
        const r2 = await rec.region("concepts-selected", [union(graph, await cardOf(side))]);
        beats.push(beat(r2, pad(r2.focus[0], 10), 3.8));
      }
    }
    // The gaps: frequent concepts that no study has examined together.
    const gaps = page.getByText(k.evidenceGaps, { exact: true }).first();
    if (await gaps.count()) {
      const r3 = await rec.region("concepts-gaps", [capHeight(await cardOf(gaps), 420)]);
      beats.push(beat(r3, pad(r3.focus[0], 10), 3.2));
    }
    return beats;
  },

  async variables({ page, rec }) {
    if (CACHES && CACHES.variables !== "ready") throw new Skip(`the model variables are not written in ${LANG} yet`);
    await openSub(page, SECTION.variables, T.scenarioDetail.variablesModelTab.subData);
    const v = T.scenarioDetail.variables;
    const title = page.getByText(v.modelVariablesTitle).first();
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
    const focuses = [focus];
    // The predictors with their sources, and the algorithm.
    const variableHead = page.getByText(v.tableVariable, { exact: true }).first();
    if (await variableHead.count()) focuses.push(capHeight(await boxUp(variableHead, "table"), 520));
    const algorithm = page.getByText(v.recommendedAlgorithm).first();
    if (await algorithm.count()) focuses.push(capHeight(await cardOf(algorithm), 420));
    const r = await rec.region("variables", focuses);
    return r.focus.map((f, i) => beat(r, pad(f, 10), i === 0 ? 3.4 : 3.0));
  },

  async model({ page, rec }) {
    if (CACHES && CACHES.variables === "none") throw new Skip("no model specification");
    await openSub(page, SECTION.variables, T.scenarioDetail.variablesModelTab.subModel);
    const dash = page.getByText(T.scenarioDetail.model.dashboard.title).first();
    if (!(await shows(dash, 30_000))) throw new Skip("no trained model");
    await settle(page, 1200);
    const dashboard = await cardOf(dash);
    const trained = await cardOf(page.getByText(T.scenarioDetail.model.guardrails).first()).catch(() => dashboard);
    const r = await rec.region("model", [dashboard, trained]);
    return [beat(r, pad(r.focus[0], 10), 3.2), beat(r, pad(r.focus[1], 10), 3.2)];
  },

  async alerts({ page, rec }) {
    await openTab(page, SECTION.alerts);
    const email = page.getByText(T.scenarioDetail.alerts.emailAlerts).first();
    await expect(email).toBeVisible({ timeout: 30_000 });
    await settle(page);
    const both = union(await cardOf(email),
                       await cardOf(page.getByText(T.scenarioDetail.alerts.livingReviewPipeline).first()));
    const r = await rec.region("alerts", [both]);
    return [beat(r, pad(r.focus[0], 10), 4.2)];
  },

  async takeaway({ rec }) {
    // A card of its own, not a screen: every format the application hands out, in one place.
    const img = await rec.render("takeaway-card", takeawayHtml(S.takeawayCard), false);
    return [{ image: img.image, size: [FRAME.width, FRAME.height], focus: null, spot: false, hold: 6.0 }];
  },
};

/** Scenes filmed on the scenario page: it is opened first when the tour is elsewhere. */
const ON_SCENARIO = new Set<SceneId>((Object.keys(FILMS) as SceneId[])
  .filter((id) => !["ask", "scenarios", "takeaway"].includes(id)));

/** "Recorded read-only from <host> on <date>", for the end card of a filmed deployment. */
function filmedOn(): string {
  const host = new URL(process.env.DEMO_BASE_URL ?? "http://localhost").host;
  const date = new Date().toLocaleDateString(LOCALE, { year: "numeric", month: "long", day: "numeric" });
  return S.filmedOn.replace("{host}", host).replace("{date}", date);
}

test("records the main pipeline", async ({ page, browser }) => {
  test.setTimeout(40 * 60_000);
  fs.mkdirSync(path.join(OUT, "frames"), { recursive: true });
  fs.mkdirSync(path.join(OUT, "overlays"), { recursive: true });
  await page.setViewportSize(VIEWPORT);
  await page.addInitScript(({ lang, key }) => {
    localStorage.setItem("literev-lang", lang);
    if (key) localStorage.setItem("api_key", key);
  }, { lang: LANG, key: API_KEY });
  if (REMOTE) {
    CACHES = await readCaches(page);
    await installReadOnlyGuard(page, CACHES);
    console.log(`read-only on ${process.env.DEMO_BASE_URL} (${LANG}): model variables ${CACHES.variables}, `
                + `cluster summaries ${CACHES.clusters}, SEIR projection ${CACHES.seir ? "stored" : "not stored"}`);
  }
  FACTS = await readFacts(page);
  console.log(`figures: ${FACT_NAMES.map((n) => `${n} ${FACTS[n] ?? "-"}`).join(", ")}; `
              + `brief in ${LANG}: ${FACTS.brief ? "yes" : "no"}; kept answers: ${FACTS.answers}`);
  const rec = new Recorder(page, browser);

  const card = async (id: "intro" | "outro", c: Card, extra: string[], hold: number) => {
    const img = await rec.render(id, cardHtml(c, extra), false);
    rec.scenes.push({ id, kind: "card", caption: null, narration: c.narration, overlay: null, band: 0,
                      beats: [{ image: img.image, size: [FRAME.width, FRAME.height], focus: null, spot: false, hold }] });
  };

  if (!ONLY.length) await card("intro", INTRO, [], 5.0);
  // DEMO_SCENES picks scenes by id, in the order of FILMS, including those the default
  // order leaves out.
  const order = ONLY.length ? (Object.keys(FILMS) as SceneId[]).filter((id) => ONLY.includes(id)) : SCENE_ORDER;
  for (const id of order) {
    const started = Date.now();
    try {
      const words = S.scenes[id];
      const caption = { step: words.step, title: words.title, body: withFacts(words.body) };
      const narration = withFacts(words.narration || words.body, true);
      if (ON_SCENARIO.has(id)) await openScenario(page);
      const beats = fitToCaption(await FILMS[id]({ page, rec }), Math.max(caption.body.length, narration.length));
      const overlay = await rec.render(id, captionHtml(caption), true);
      const band = Math.min(0.45, (FRAME.height - overlay.top + 18) / FRAME.height);
      rec.scenes.push({ id, kind: "screen", caption, narration, overlay: overlay.image, band, beats });
      console.log(`scene ${id}: ${beats.length} beat(s) in ${((Date.now() - started) / 1000).toFixed(1)} s`);
    } catch (e) {
      const reason = e instanceof Error ? e.message.split("\n")[0] : String(e);
      if (STRICT && !(e instanceof Skip)) throw e;
      rec.skipped.push({ id, reason });
      console.log(`skipped scene ${id}: ${reason}`);
    }
  }
  if (!ONLY.length) await card("outro", S.outro, SYNTHETIC ? [S.demoNote] : REMOTE ? [filmedOn()] : [], 5.5);

  if (REMOTE) {
    console.log(`read-only guard: ${GUARD.blocked.length} request(s) blocked, ${GUARD.redirected.length} read from the stored copy`);
    for (const line of [...GUARD.blocked, ...GUARD.redirected]) console.log(`  ${line}`);
  }
  const manifest = {
    version: 1, lang: LANG, scenario: SCENARIO_ID, synthetic: SYNTHETIC,
    base_url: process.env.DEMO_BASE_URL ?? null,
    viewport: VIEWPORT, scale: SCALE, frame: FRAME,
    recorded_at: new Date().toISOString(),
    facts: FACTS, caches: CACHES, guard: REMOTE ? GUARD : null,
    scenes: rec.scenes, skipped: rec.skipped,
  };
  fs.writeFileSync(path.join(OUT, "manifest.json"), JSON.stringify(manifest, null, 2));
  expect(rec.scenes.length, "scenes recorded").toBeGreaterThan(0);
});
