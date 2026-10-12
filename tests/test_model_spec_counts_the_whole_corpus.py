"""The model spec speaks for the whole relevant corpus, and says when it had to be repaired.

Four things the spec did wrong (api/variables.py):

- each variable's "evidence_level" was asked of the model, "Nombre d'études qui la
  mentionnent", and the model counted among the 25 articles it was shown. It is now
  counted in SQL over every relevant article, by a rule the spec states in `_meta`;
- the cache fingerprint covered the first 25 relevant ids only, so a change further down
  reused the old spec and skipped the epidemiological-parameter extraction, which reads
  the whole corpus;
- 90 seconds for an 8,000-token answer, and a truncated spec silently repaired;
- a failed translation was cached as the translation.

The generation runs against the test database with the LLM stubbed; the translation
tests are pure."""
import json
import types

import pytest

import llm_usage
import main
from conftest import ensure_document_columns, patch_app  # noqa: E402

SID = "usr-spec-count-test"
RELEVANT = list(range(8701, 8731))                    # 30 relevant articles
EXCLUDED, BELOW = 8740, 8741
IDS = RELEVANT + [EXCLUDED, BELOW]


def _completion(content: str, finish: str = "stop"):
    msg = types.SimpleNamespace(content=content)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg, finish_reason=finish)])


def _engine_ok() -> bool:
    try:
        with main.engine.connect():
            return True
    except Exception:
        return False


@pytest.fixture()
def seeded(db_conn):
    """Thirty relevant articles: 12 say "temperature" in their abstract, 2 "heat waves",
    1 has "heat exposure" in its PICO only. An excluded article and one below the
    threshold say "temperature" too, and must not count."""
    if not _engine_ok():
        pytest.skip("main.engine cannot reach the database")
    with db_conn.cursor() as cur:
        cur.execute("SELECT to_regclass('user_scenarios') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_user_scenarios_table()
        cur.execute("CREATE TABLE IF NOT EXISTS literature_document (id BIGINT PRIMARY KEY)")
        cur.execute("CREATE TABLE IF NOT EXISTS article_scenarios ("
                    "scenario_id TEXT, document_id BIGINT, PRIMARY KEY (scenario_id, document_id))")
        ensure_document_columns(cur)
        for col, typ in (("title", "TEXT"), ("abstract", "TEXT"), ("is_duplicate", "BOOLEAN DEFAULT FALSE"),
                         ("authors", "TEXT"), ("doi", "TEXT"), ("journal", "TEXT"), ("citation_count", "INTEGER")):
            cur.execute(f"ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS {col} {typ}")
        cur.execute("ALTER TABLE article_scenarios ADD COLUMN IF NOT EXISTS similarity_score DOUBLE PRECISION")
        cur.execute("SELECT to_regclass('scenario_settings') IS NULL")
        if cur.fetchone()[0]:
            main._ensure_scenario_settings_table()
        main._ensure_spec_proposal_columns()
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM document_chunk WHERE document_id = ANY(%s)", (IDS,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (IDS,))
        cur.execute("INSERT INTO user_scenarios (id, name, query, mode, filters) "
                    "VALUES (%s, 'Dengue and weather', 'dengue weather', 'boolean', '{}')", (SID,))
        for k, doc_id in enumerate(RELEVANT):
            if k < 12:
                abstract = "Daily Temperature was associated with dengue incidence in the region."
            elif k < 14:
                abstract = "Heat waves preceded the dengue outbreaks by three weeks."
            else:
                abstract = "Mosquito surveillance in the region over five seasons."
            pico = json.dumps({"P": "residents", "I": "heat exposure" if k == 14 else "surveillance",
                               "O": "dengue incidence"})
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate, pico_json) VALUES (%s, %s, %s, 2024, 'pubmed', 'literev', false, %s)",
                        (doc_id, f"Study {k}", abstract, pico))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score) "
                        "VALUES (%s, %s, %s)", (SID, doc_id, round(0.95 - 0.01 * k, 2)))
        for doc_id, sim, status in ((EXCLUDED, 0.99, "excluded"), (BELOW, 0.10, None)):
            cur.execute("INSERT INTO literature_document (id, title, abstract, year, source, project_context, "
                        "is_duplicate) VALUES (%s, 'Outside', 'Temperature and dengue.', 2024, 'pubmed', "
                        "'literev', false)", (doc_id,))
            cur.execute("INSERT INTO article_scenarios (scenario_id, document_id, similarity_score, "
                        "screening_status) VALUES (%s, %s, %s, %s)", (SID, doc_id, sim, status))
        cur.execute("INSERT INTO scenario_settings (scenario_id, similarity_threshold) VALUES (%s, 0.45) "
                    "ON CONFLICT (scenario_id) DO UPDATE SET similarity_threshold = 0.45", (SID,))
    yield db_conn
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM article_scenarios WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM scenario_settings WHERE scenario_id = %s", (SID,))
        cur.execute("DELETE FROM user_scenarios WHERE id = %s", (SID,))
        cur.execute("DELETE FROM literature_document WHERE id = ANY(%s)", (IDS,))


_SPEC = {
    "primary_outcome": {"name": "Dengue incidence", "task_type": "regression", "provenance": [8701]},
    "predictor_variables": [
        {"name": "Température maximale", "keywords": ["temperature", "heat"], "type": "continuous",
         "importance": "high", "evidence_level": "3 études", "provenance": [8701]},
        {"name": "Rainfall", "keywords": ["rainfall", "precipitation"], "type": "continuous",
         "importance": "medium", "provenance": []},
    ],
    "recommended_algorithm": {"primary": "Random forest", "family": "random_forest", "provenance": []},
    "implementation_notes": "A long note that the ceiling cuts in the middle of its sentence " * 5,
}


class _FakeSpecLLM:
    """The model, answering the spec with `content` cut at its ceiling (`finish`)."""

    def __init__(self, content: str, finish: str):
        self.calls: list[dict] = []
        self.timeouts: list = []
        outer = self

        def _create(**kw):
            outer.calls.append(kw)
            return _completion(content, finish)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))

    def __call__(self, *a, **kw):
        self.timeouts.append(kw.get("timeout"))
        return self


def _cut_spec() -> str:
    full = json.dumps(_SPEC, ensure_ascii=False)
    return full[:full.index("implementation_notes") + 40]


def test_each_variable_is_counted_over_every_relevant_article(seeded, monkeypatch):
    fake = _FakeSpecLLM(json.dumps(_SPEC, ensure_ascii=False), "stop")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    out = main._generate_variables_from_pico(SID, lang="fr")
    assert "error" not in out, out
    temperature, rainfall = out["predictor_variables"]
    # 12 abstracts say temperature, 2 say heat, 1 PICO says heat: 15 of the 30 relevant.
    # The excluded article and the one below the threshold say temperature too.
    assert (temperature["evidence_count"], temperature["evidence_level"]) == (15, "15/30")
    assert temperature["evidence_terms"] == ["Température maximale", "temperature", "heat"]
    assert (rainfall["evidence_count"], rainfall["evidence_level"]) == (0, "0/30")
    meta = out["_meta"]
    assert meta["evidence_count_denominator"] == 30
    assert "relevant articles" in meta["evidence_count_rule"]
    assert meta["truncated"] is False
    # The model is no longer asked to count: it gives the terms the server counts with.
    prompt = fake.calls[0]["messages"][1]["content"]
    assert '"keywords"' in prompt and "Nombre d'études qui la mentionnent" not in prompt
    assert fake.timeouts == [300.0]


def test_a_spec_cut_twice_is_repaired_and_says_so(seeded, monkeypatch):
    fake = _FakeSpecLLM(_cut_spec(), "length")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    out = main._generate_variables_from_pico(SID, lang="fr")
    assert "error" not in out, out
    assert len(fake.calls) == 2                                    # asked again, shorter
    assert "CUT OFF" in fake.calls[1]["messages"][0]["content"]
    assert [pv["name"] for pv in out["predictor_variables"]] == ["Température maximale", "Rainfall"]
    meta = out["_meta"]
    assert meta["truncated"] is True
    assert 0 < meta["truncated_repair"]["kept_chars"] <= meta["truncated_repair"]["total_chars"]


def test_the_fingerprint_covers_every_relevant_article(seeded, monkeypatch):
    """A change below the first 25 used to reuse the old spec, and skip the parameter
    extraction that reads the whole corpus."""
    fake = _FakeSpecLLM(json.dumps(_SPEC, ensure_ascii=False), "stop")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", fake)
    first = main._generate_variables_from_pico(SID, lang="fr")
    assert first["_meta"]["reused"] is False and len(fake.calls) == 1

    again = main._generate_variables_from_pico(SID, lang="fr")
    assert again["_meta"]["reused"] is True and len(fake.calls) == 1        # same evidence
    assert again["_meta"]["evidence_count_denominator"] == 30               # carried over

    with seeded.cursor() as cur:                                            # the 28th article goes
        cur.execute("UPDATE article_scenarios SET screening_status = 'excluded' "
                    "WHERE scenario_id = %s AND document_id = %s", (SID, RELEVANT[27]))
    moved = main._generate_variables_from_pico(SID, lang="fr")
    assert moved["_meta"]["reused"] is False and len(fake.calls) == 2
    assert moved["_meta"]["evidence_fingerprint"] != first["_meta"]["evidence_fingerprint"]
    assert moved["predictor_variables"][0]["evidence_level"] == "15/29"


# ── Translation ──────────────────────────────────────────────────────────────
class _Writes:
    def __init__(self):
        self.statements: list = []

    def execute(self, *a, **k):
        self.statements.append(a)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def test_a_failed_translation_is_served_untranslated_and_not_cached(monkeypatch):
    writes = _Writes()
    monkeypatch.setattr(main.engine, "begin", lambda: writes)
    base = {"primary_outcome": {"name": "Incidence de la dengue", "definition": "Le taux hebdomadaire"}}

    patch_app(monkeypatch, "_llm_translate_strings", lambda texts, lang: None)
    out = main._get_localized_variables("usr-x", "en", base_variables=base, variables_lang="fr",
                                        variables_i18n=None)
    assert out["primary_outcome"]["name"] == "Incidence de la dengue"
    assert writes.statements == []                                 # the next display tries again

    patch_app(monkeypatch, "_llm_translate_strings", lambda texts, lang: [f"[en] {t}" for t in texts])
    out = main._get_localized_variables("usr-x", "en", base_variables=base, variables_lang="fr",
                                        variables_i18n=None)
    assert out["primary_outcome"]["name"] == "[en] Incidence de la dengue"
    assert len(writes.statements) == 1


def test_a_cut_translation_is_split_in_two_and_a_half_cut_again_is_no_translation(monkeypatch):
    calls: list = []

    def _create(**kw):
        items = json.loads(kw["messages"][1]["content"])["items"]
        calls.append(len(items))
        if len(items) == 4 or (cut_halves and len(items) == 2):
            return _completion('{"items": ["a', "length")
        return _completion(json.dumps({"items": [f"T {x}" for x in items]}))

    client = types.SimpleNamespace(chat=types.SimpleNamespace(completions=types.SimpleNamespace(create=_create)))
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda **kw: client)
    cut_halves = False
    assert main._llm_translate_strings(["a", "b", "c", "d"], "en") == ["T a", "T b", "T c", "T d"]
    assert calls == [4, 2, 2]
    cut_halves = True
    assert main._llm_translate_strings(["a", "b", "c", "d"], "en") is None
