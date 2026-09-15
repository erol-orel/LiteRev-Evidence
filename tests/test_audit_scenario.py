"""scripts/audit_scenario.py cross-checks the numbers the interface shows for one
scenario across endpoints. Pure test: the endpoints are canned."""
import importlib.util
import os

import pytest

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "audit_scenario.py")
_spec = importlib.util.spec_from_file_location("audit_scenario", _PATH)
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)

SID = "usr-audit"


def _payloads(corpus: int, card: int, screened: int, running: bool = False, lang: str = "fr"):
    return {
        "/activity": {"running": ([{"scenario_id": SID, "kind": "pipeline", "step": "embed"}] if running else [])},
        f"/user-scenarios/{SID}/counts": {
            "corpus_links": corpus, "article_count": card, "prisma_screened": screened,
            "above_threshold": corpus - 3, "below_threshold": 3, "embedded": corpus,
            "consistent": card == corpus == screened,
            "mismatches": ([] if card == corpus == screened else [{"field": "article_count"}]),
        },
        f"/user-scenarios/{SID}/detail": {
            "corpus_stats": {"total": corpus},
            "facets": [{"kind": "boolean", "text": "A", "op": None}, {"kind": "boolean", "text": "B", "op": "and"}],
            "combined_query": "(A) AND (B)",
        },
        "/user-scenarios": [{"id": SID, "article_count": card, "populate_status": "done", "pipeline_status": "done"}],
        f"/user-scenarios/{SID}/prisma": {"identification": {
            "total_records": screened + 110, "duplicates_removed": 100, "unique_records": screened + 10,
            "removed_no_abstract": 7, "removed_not_matching": 3, "removed_other_reasons": 0,
            "records_screened": screened, "figures_from": "search_run", "federation_incomplete": False}},
        f"/user-scenarios/{SID}/embedding-status": {"corpus_total": corpus, "ranking": {"scored": corpus, "total": corpus},
                                                     "score_availability": {"semantic": True}},
        f"/user-scenarios/{SID}/clustering?lang={lang}": {
            "n_docs": 50, "n_docs_total": corpus, "lang": lang, "method": "embeddings_umap_hdbscan",
            "points_shown": 50, "points_total": 50,
            "clusters": [{"cluster_id": 0, "n_docs": 30, "is_noise": False, "summary": "s"},
                         {"cluster_id": 1, "n_docs": 15, "is_noise": False, "summary": "t"},
                         {"cluster_id": -1, "n_docs": 5, "is_noise": True, "summary": ""}]},
        f"/user-scenarios/{SID}/pico-stats": {"total": corpus, "with_pico": 40, "without_pico": corpus - 40, "coverage_pct": 1.0},
        f"/scenarios/{SID}/settings": {"similarity_threshold": 0.45, "cached": {"clustering": True, "evidence_brief": False}},
        f"/user-scenarios/{SID}/evidence-brief": {"scenario_id": SID, "corpus_stats": {}},
    }


def _run(payloads):
    def get(path):
        return (200, payloads[path]) if path in payloads else (404, None)
    a = audit.Audit(get, SID, "fr")
    rc = a.run()
    return rc, {check: level for level, check, _ in a.rows}


def test_consistent_scenario_passes_every_check(capsys):
    rc, levels = _run(_payloads(corpus=200, card=200, screened=200))
    assert rc == 0
    assert levels["counts (server verdict)"] == "OK"
    assert levels["prisma arithmetic"] == "OK" and levels["prisma screened = corpus"] == "OK"
    assert levels["clustering sizes"] == "OK" and levels["clustering language"] == "OK"
    assert levels["multi-facet expression"] == "OK"
    assert levels["pico coverage arithmetic"] == "OK"
    assert "ALL NUMBERS AGREE" in capsys.readouterr().out


def test_disagreeing_counts_fail_when_nothing_is_running():
    rc, levels = _run(_payloads(corpus=200, card=180, screened=195))
    assert rc == 1
    assert levels["counts (server verdict)"] == "FAIL"
    assert levels["scenario list card"] == "FAIL"
    assert levels["prisma screened = corpus"] == "FAIL"


def test_disagreements_are_only_warnings_while_the_scenario_runs():
    rc, levels = _run(_payloads(corpus=200, card=180, screened=195, running=True))
    assert rc == 0
    assert levels["counts (server verdict)"] == "WARN" and levels["scenario list card"] == "WARN"
    assert levels["activity"] == "INFO"


def test_wrong_language_and_broken_prisma_arithmetic_fail():
    p = _payloads(corpus=200, card=200, screened=200)
    p[f"/user-scenarios/{SID}/clustering?lang=fr"]["lang"] = "en"
    p[f"/user-scenarios/{SID}/prisma"]["identification"]["unique_records"] = 999
    rc, levels = _run(p)
    assert rc == 1
    assert levels["clustering language"] == "FAIL" and levels["prisma arithmetic"] == "FAIL"


def test_missing_endpoint_is_reported_not_raised():
    p = _payloads(corpus=200, card=200, screened=200)
    del p[f"/user-scenarios/{SID}/prisma"]
    rc, levels = _run(p)
    assert rc == 1 and levels[f"/user-scenarios/{SID}/prisma"] == "FAIL"
