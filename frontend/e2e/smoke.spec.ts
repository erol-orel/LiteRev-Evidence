// Browser smoke test of the search page and the scenario page, driven by
// scripts/smoke_e2e.py (repository root): it boots the API on a throwaway database,
// seeds one synthetic scenario and serves the built frontend. Nothing here reaches a
// live source: the search runs on the local corpus only and the LLM generators are
// off (no OpenAI key), so the test is deterministic and works offline.
import { expect, test, type Page } from "@playwright/test";
import { en } from "../src/i18n/locales/en";
import { fr } from "../src/i18n/locales/fr";

const SCENARIO_NAME = process.env.E2E_SCENARIO_NAME ?? "Smoke corpus";
const SCENARIO_QUERY = process.env.E2E_SCENARIO_QUERY ?? "influenza AND surveillance";
const ARTICLES = Number(process.env.E2E_ARTICLES ?? 40);
const API_KEY = process.env.E2E_API_KEY ?? "smoke-key";

/** Any uncaught page error or 5xx answer from the API fails the test at the end. */
function watchBrowser(page: Page): () => void {
  const problems: string[] = [];
  page.on("pageerror", (e) => problems.push(`page error: ${e.message}`));
  page.on("response", (r) => {
    if (r.status() >= 500) problems.push(`HTTP ${r.status()} on ${r.url()}`);
  });
  return () => expect(problems, "errors seen by the browser").toEqual([]);
}

/** Open the app with the interface language (and the admin key) already chosen. */
async function openApp(page: Page, opts: { lang?: "fr" | "en"; withKey?: boolean } = {}) {
  await page.addInitScript(({ lang, key }) => {
    if (lang) localStorage.setItem("literev-lang", lang);
    if (key) localStorage.setItem("api_key", key);
  }, { lang: opts.lang, key: opts.withKey ? API_KEY : "" });
  await page.goto("/");
}

const nav = (page: Page, label: string) => page.getByRole("button", { name: label, exact: true }).first();
const seededCard = (page: Page) => page.getByTestId("scenario-card").filter({ hasText: SCENARIO_NAME }).first();

test("lists the seeded scenario and keeps the chosen language across reloads", async ({ page }) => {
  const check = watchBrowser(page);
  await openApp(page);                                   // fr-FR browser → French interface
  await nav(page, fr.nav.scenarios).click();
  await expect(seededCard(page)).toBeVisible();
  await expect(seededCard(page)).toContainText(`${fr.scenarios.savedSearchPrefix}${SCENARIO_QUERY}`);
  await expect(seededCard(page)).toContainText(`${ARTICLES} ${fr.scenarios.articles}`);

  await page.getByRole("button", { name: "EN", exact: true }).click();
  await expect(nav(page, en.nav.scenarios)).toBeVisible();
  await expect(seededCard(page)).toContainText(`${en.scenarios.savedSearchPrefix}${SCENARIO_QUERY}`);

  await page.reload();
  await expect(nav(page, en.nav.search)).toBeVisible();  // the choice is persisted
  await expect(nav(page, en.nav.scenarios)).toBeVisible();
  check();
});

test("opens the scenario page: header, corpus, PRISMA flow and clustering", async ({ page }) => {
  const check = watchBrowser(page);
  await openApp(page, { lang: "en" });
  await nav(page, en.nav.scenarios).click();
  await seededCard(page).getByRole("button", { name: en.scenarios.detailPage, exact: true }).click();

  // Header: title, corpus size and the saved query.
  await expect(page.getByRole("heading", { level: 2, name: SCENARIO_NAME })).toBeVisible();
  await expect(page.getByText(`${ARTICLES} ${en.scenarioDetail.page.articles}`, { exact: true })).toBeVisible();
  await expect(page.getByText(`${en.scenarios.savedSearchPrefix}${SCENARIO_QUERY}`)).toBeVisible();

  // Corpus & Review → Corpus: the article list with the seeded titles.
  await expect(page.getByText(/Benchmark article \d+/).first()).toBeVisible();
  await expect(page.getByText(`${en.scenarioDetail.corpus.corpusTitlePrefix} (${ARTICLES} ${en.scenarioDetail.corpus.corpusTitleArticles})`)).toBeVisible();

  // PRISMA: the flow is computed and its first stage shows the corpus size.
  await page.getByRole("button", { name: en.scenarioDetail.review.subPrisma, exact: true }).click();
  await expect(page.getByText(en.scenarioDetail.prisma.title)).toBeVisible();
  await expect(page.getByText(en.scenarioDetail.prisma.stage1)).toBeVisible();
  await expect(page.getByText(en.scenarioDetail.prisma.duplicatesRemoved)).toBeVisible();
  await expect(page.getByText(en.scenarioDetail.prisma.stage2)).toBeVisible();

  // Visualization: the cached clustering is rendered in the interface language.
  await page.getByRole("button", { name: en.scenarioDetail.page.sections.viz, exact: true }).click();
  await expect(page.getByText("Cluster 1", { exact: true }).first()).toBeVisible();
  await expect(page.getByText("Synthetic summary.").first()).toBeVisible();

  // Knowledge graph: the concept map is the default mode and answers without an LLM
  // (structured fields only, or an explicit empty state), the similarity network stays.
  await page.getByRole("button", { name: en.scenarioDetail.vizTab.subKnowledgeGraph, exact: true }).click();
  await expect(page.getByRole("button", { name: en.scenarioDetail.knowledgeGraph.modeConcepts, exact: true })).toBeVisible();
  await expect(page.getByText(en.scenarioDetail.knowledgeGraph.conceptTitle)
    .or(page.getByText(en.scenarioDetail.knowledgeGraph.noConcepts)).first()).toBeVisible({ timeout: 15_000 });
  await page.getByRole("button", { name: en.scenarioDetail.knowledgeGraph.modeArticles, exact: true }).click();
  await expect(page.getByText(en.scenarioDetail.knowledgeGraph.title)
    .or(page.getByText(en.scenarioDetail.knowledgeGraph.noArticleEmbeddings)).first()).toBeVisible({ timeout: 15_000 });
  check();
});

test("runs a two-facet search on the local corpus and names the scenario with the AND", async ({ page }) => {
  const check = watchBrowser(page);
  await openApp(page, { lang: "en", withKey: true });

  const query = page.getByPlaceholder(en.search.queryPlaceholder);
  await query.fill("influenza AND surveillance");
  // The kind of the main query is detected from its syntax.
  await expect(page.getByRole("button", { name: new RegExp(`^${en.search.subQueryKindBoolean}`) })).toBeVisible();

  // Second facet, intersected with the first.
  await page.getByRole("button", { name: `+ ${en.search.addSubQuery}` }).click();
  await page.getByPlaceholder(en.search.subQueryPlaceholder).fill("wastewater");
  await page.getByRole("button", { name: en.search.combinatorIntersection, exact: true }).click();

  // Live sources stay off (the checkbox is unchecked by default): local corpus only.
  await expect(page.getByRole("checkbox")).not.toBeChecked();
  await page.locator("#search-btn").click();             // the nav tab is also named "Search"

  // The corpus is built server-side, then the most relevant articles are shown.
  await expect(page.getByText(new RegExp(`\\d+ ${en.search.documents} ${en.search.relevantPlural}`)).first())
    .toBeVisible({ timeout: 60_000 });
  await expect(page.getByText(/Benchmark article \d+/).first()).toBeVisible();

  // The saved search carries the whole expression, AND included.
  const combined = "(influenza AND surveillance) AND (wastewater)";
  await nav(page, en.nav.scenarios).click();
  const card = page.getByTestId("scenario-card").filter({ hasText: combined }).first();
  await expect(card).toBeVisible();
  await expect(card).toContainText(`${en.scenarios.savedSearchPrefix}${combined}`);
  check();
});
