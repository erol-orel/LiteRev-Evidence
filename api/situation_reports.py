"""ReliefWeb situation reports: ingestion, listing, budget.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text

from .core import RELIEFWEB_APPNAME, app, engine, logger, require_api_key

# ═════════════════════════════════════════════════════════════════════════════
# ReliefWeb — rapports de situation humanitaires, EN PARALLÈLE des articles
# ═════════════════════════════════════════════════════════════════════════════
# Flux SÉPARÉ, volontairement : table `situation_report` distincte, ingestion distincte,
# budget d'appels distinct. Ni la fédération 12 sources (_run_user_scenario_populate), ni
# PRISMA, ni /corpus/stats ne sont touchés — ce que ReliefWeb apporte (la vitesse, le
# terrain) ne doit pas se mélanger à un corpus revu par les pairs, ni en gonfler les
# compteurs. Voir reliefweb_source.py pour le pourquoi détaillé.
def _ensure_reliefweb_tables():
    """DDL de démarrage, en doublure de la migration f3a9c1d4e8b2 (convention maison :
    cf. _ensure_dedup_columns). Idempotent."""
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS situation_report (
                id SERIAL PRIMARY KEY, rw_id TEXT NOT NULL,
                rw_kind TEXT NOT NULL DEFAULT 'report',
                title TEXT NOT NULL, body TEXT, url TEXT,
                published_at TIMESTAMPTZ, changed_at TIMESTAMPTZ,
                fetched_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                sources JSONB NOT NULL DEFAULT '[]'::jsonb, format TEXT,
                primary_country TEXT, primary_iso3 TEXT,
                countries JSONB NOT NULL DEFAULT '[]'::jsonb, glide TEXT,
                disaster_names JSONB NOT NULL DEFAULT '[]'::jsonb,
                disaster_types JSONB NOT NULL DEFAULT '[]'::jsonb,
                themes JSONB NOT NULL DEFAULT '[]'::jsonb,
                language TEXT, status TEXT,
                credibility DOUBLE PRECISION NOT NULL DEFAULT 0.2,
                title_norm TEXT, series_key TEXT)"""))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_sitrep_rw "
                          "ON situation_report (rw_kind, rw_id)"))
        for _sql in (
            "CREATE INDEX IF NOT EXISTS ix_sitrep_published ON situation_report (published_at DESC)",
            "CREATE INDEX IF NOT EXISTS ix_sitrep_iso3 ON situation_report (primary_iso3)",
            "CREATE INDEX IF NOT EXISTS ix_sitrep_glide ON situation_report (glide)",
            "CREATE INDEX IF NOT EXISTS ix_sitrep_series ON situation_report (series_key)",
        ):
            conn.execute(text(_sql))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS situation_report_scenarios (
                id SERIAL PRIMARY KEY,
                report_id INTEGER NOT NULL REFERENCES situation_report(id) ON DELETE CASCADE,
                scenario_id TEXT NOT NULL, relevance DOUBLE PRECISION,
                linked_at TIMESTAMPTZ NOT NULL DEFAULT NOW())"""))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_sitrep_scenario "
                          "ON situation_report_scenarios (report_id, scenario_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS ix_sitrep_scenario_sid "
                          "ON situation_report_scenarios (scenario_id)"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS reliefweb_quota (
                day DATE PRIMARY KEY, calls_used INTEGER NOT NULL DEFAULT 0)"""))
    logger.info("Tables ReliefWeb (situation_report) vérifiées/créées.")


try:
    _ensure_reliefweb_tables()
except Exception as _e:
    logger.warning(f"_ensure_reliefweb_tables: {_e}")


def _rw_calls_used_today() -> int:
    with engine.begin() as conn:
        row = conn.execute(text(
            "SELECT calls_used FROM reliefweb_quota WHERE day = CURRENT_DATE")).mappings().first()
    return int(row["calls_used"]) if row else 0


def _rw_budget_remaining() -> int:
    """Appels encore disponibles aujourd'hui. L'API documente 1000 appels/jour mais
    n'expose AUCUN en-tête de quota, aucune heure de remise à zéro et aucun code d'erreur
    dédié : le budget doit donc être compté ici, sinon il ne peut pas être respecté."""
    import reliefweb_source
    return max(0, reliefweb_source.QUOTA_CALLS_PER_DAY - _rw_calls_used_today())


def _rw_note_calls(n: int) -> None:
    if n <= 0:
        return
    with engine.begin() as conn:
        conn.execute(text("""
            INSERT INTO reliefweb_quota (day, calls_used) VALUES (CURRENT_DATE, :n)
            ON CONFLICT (day) DO UPDATE SET calls_used = reliefweb_quota.calls_used + :n"""),
            {"n": int(n)})


def _rw_upsert_reports(reports) -> tuple[int, int]:
    """Insère/actualise des rapports. Clé naturelle (rw_kind, rw_id) : un rapport RÉVISÉ
    par ReliefWeb met à jour sa ligne au lieu d'en créer une seconde. Renvoie (insérés,
    actualisés)."""
    inserted = updated = 0
    with engine.begin() as conn:
        for r in reports or []:
            row = r.to_row()
            res = conn.execute(text("""
                INSERT INTO situation_report (
                    rw_id, rw_kind, title, body, url, published_at, changed_at,
                    sources, format, primary_country, primary_iso3, countries, glide,
                    disaster_names, disaster_types, themes, language, status,
                    credibility, title_norm, series_key)
                VALUES (:rw_id, 'report', :title, :body, :url,
                        CAST(NULLIF(:published_at,'') AS TIMESTAMPTZ),
                        CAST(NULLIF(:changed_at,'') AS TIMESTAMPTZ),
                        CAST(:sources AS JSONB), :format, :primary_country, :primary_iso3,
                        CAST(:countries AS JSONB), :glide, CAST(:disaster_names AS JSONB),
                        CAST(:disaster_types AS JSONB), CAST(:themes AS JSONB),
                        :language, :status, :credibility, :title_norm, :series_key)
                ON CONFLICT (rw_kind, rw_id) DO UPDATE SET
                    title = EXCLUDED.title, body = EXCLUDED.body,
                    changed_at = EXCLUDED.changed_at, status = EXCLUDED.status,
                    fetched_at = NOW()
                RETURNING (xmax = 0) AS is_new"""), {
                    **{k: row[k] for k in ("rw_id", "title", "body", "url", "format",
                                           "primary_country", "primary_iso3", "glide",
                                           "language", "status", "credibility",
                                           "title_norm", "series_key")},
                    "published_at": row["published_at"] or "",
                    "changed_at": row["changed_at"] or "",
                    "sources": json.dumps(row["sources"]),
                    "countries": json.dumps(row["countries"]),
                    "disaster_names": json.dumps(row["disaster_names"]),
                    "disaster_types": json.dumps(row["disaster_types"]),
                    "themes": json.dumps(row["themes"]),
                }).mappings().first()
            if res and res["is_new"]:
                inserted += 1
            else:
                updated += 1
    return inserted, updated


def _rw_link_to_scenario(scenario_id: str, rw_ids: list[str]) -> int:
    if not rw_ids:
        return 0
    with engine.begin() as conn:
        res = conn.execute(text("""
            INSERT INTO situation_report_scenarios (report_id, scenario_id)
            SELECT id, :sid FROM situation_report
             WHERE rw_kind = 'report' AND rw_id = ANY(:ids)
            ON CONFLICT (report_id, scenario_id) DO NOTHING"""),
            {"sid": scenario_id, "ids": list(rw_ids)})
    return int(res.rowcount or 0)


class ReliefWebIngestIn(BaseModel):
    pathogens: list[str] | None = None
    countries: list[str] | None = None          # ISO3
    date_from: str | None = None
    date_to: str | None = None
    languages: list[str] | None = ["en"]
    limit: int = 100                            # par appel (max 1000, cf. quotas)
    max_calls: int = 3                          # plafond volontaire pour ce run
    keep_per_series: int = 1
    scenario_id: str | None = None              # rattachement facultatif à un scénario


@app.post("/reliefweb/ingest")
def reliefweb_ingest(payload: ReliefWebIngestIn,
                     _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Ingère des rapports de situation ReliefWeb dans le flux SÉPARÉ `situation_report`.

    Respecte le quota documenté (1000 appels/jour) en le comptant localement, déduplique
    AVANT insertion (4 000+ sources couvrent le même événement, et le même document existe
    en EN/FR/ES/AR sous des ids distincts), et signale explicitement une couverture
    partielle (`truncated`) — un balayage écourté par le quota ne doit jamais se lire
    comme exhaustif."""
    import reliefweb_source
    if not RELIEFWEB_APPNAME:
        raise HTTPException(status_code=503, detail=(
            "RELIEFWEB_APPNAME non configuré. Depuis le 1er novembre 2025 l'API ReliefWeb "
            "exige un appname PRÉ-APPROUVÉ (formulaire d'inscription sur apidoc.reliefweb.int) ; "
            "une chaîne arbitraire est rejetée."))
    budget = _rw_budget_remaining()
    if budget <= 0:
        raise HTTPException(status_code=429, detail=(
            f"Quota ReliefWeb épuisé pour aujourd'hui "
            f"({reliefweb_source.QUOTA_CALLS_PER_DAY} appels/jour)."))

    max_calls = max(1, min(int(payload.max_calls or 1), budget))
    limit = max(1, min(int(payload.limit or 100), reliefweb_source.MAX_LIMIT_PER_CALL))
    collected, total_available, calls = [], 0, 0
    try:
        for page in range(max_calls):
            q = reliefweb_source.build_reports_query(
                pathogens=payload.pathogens, countries=payload.countries,
                date_from=payload.date_from, date_to=payload.date_to,
                languages=payload.languages, limit=limit, offset=page * limit)
            reports, total = reliefweb_source.fetch_reports(q, RELIEFWEB_APPNAME)
            calls += 1
            total_available = max(total_available, total)
            collected.extend(reports)
            if len(reports) < limit:
                break                       # dernière page atteinte
    except Exception as e:
        _rw_note_calls(calls)
        raise HTTPException(status_code=502, detail=f"ReliefWeb: {e}") from e
    finally:
        _rw_note_calls(calls)

    kept = reliefweb_source.dedupe(collected, keep_per_series=payload.keep_per_series)
    inserted, updated = _rw_upsert_reports(kept)
    linked = (_rw_link_to_scenario(payload.scenario_id, [r.rw_id for r in kept])
              if payload.scenario_id else 0)
    out = reliefweb_source.summarise_ingest(collected, kept, total_available, calls)
    out.update({"inserted": inserted, "updated": updated, "linked_to_scenario": linked,
                "calls_remaining_today": _rw_budget_remaining()})
    return out


@app.get("/reliefweb/reports")
def reliefweb_reports(scenario_id: str | None = None, iso3: str | None = None,
                      glide: str | None = None, q: str | None = None,
                      limit: int = 50, offset: int = 0) -> dict[str, Any]:
    """Liste les rapports de situation ingérés (flux séparé — n'interroge JAMAIS
    literature_document). Lecture seule, sans appel réseau."""
    import reliefweb_source
    limit = max(1, min(int(limit or 50), 200))
    where, params = ["1=1"], {"limit": limit, "offset": max(0, int(offset or 0))}
    join = ""
    if scenario_id:
        join = "JOIN situation_report_scenarios l ON l.report_id = s.id AND l.scenario_id = :sid"
        params["sid"] = scenario_id
    if iso3:
        where.append("s.primary_iso3 = :iso3")
        params["iso3"] = str(iso3).lower()
    if glide:
        where.append("s.glide = :glide")
        params["glide"] = glide
    if q:
        where.append("(s.title ILIKE :q OR s.body ILIKE :q)")
        params["q"] = f"%{q}%"
    sql = f"""
        SELECT s.id, s.rw_id, s.title, s.url, s.published_at, s.sources, s.format,
               s.primary_country, s.primary_iso3, s.glide, s.disaster_types, s.themes,
               s.language, s.credibility, LEFT(COALESCE(s.body,''), 400) AS excerpt
          FROM situation_report s {join}
         WHERE {' AND '.join(where)}
         ORDER BY s.published_at DESC NULLS LAST, s.id DESC
         LIMIT :limit OFFSET :offset"""
    with engine.begin() as conn:
        rows = conn.execute(text(sql), params).mappings().all()
        total = conn.execute(text(
            f"SELECT COUNT(*) AS n FROM situation_report s {join} "
            f"WHERE {' AND '.join(where)}"),
            {k: v for k, v in params.items() if k not in ("limit", "offset")}
        ).mappings().first()
    return {
        "total": int(total["n"]) if total else 0,
        "limit": limit, "offset": params["offset"],
        "reports": [{
            "id": r["id"], "rw_id": r["rw_id"], "title": r["title"], "url": r["url"],
            "published_at": r["published_at"].isoformat() if r["published_at"] else None,
            "sources": r["sources"], "format": r["format"],
            "primary_country": r["primary_country"], "primary_iso3": r["primary_iso3"],
            "glide": r["glide"], "disaster_types": r["disaster_types"],
            "themes": r["themes"], "language": r["language"],
            "credibility": r["credibility"], "excerpt": r["excerpt"],
        } for r in rows],
        # Rappel affiché dans l'UI : ce flux est de la littérature GRISE.
        "grey_literature": True,
        "max_credibility": reliefweb_source.MAX_CREDIBILITY,
    }


@app.get("/reliefweb/status")
def reliefweb_status() -> dict[str, Any]:
    """État du flux ReliefWeb : configuration, quota consommé, volume ingéré."""
    import reliefweb_source
    try:
        with engine.begin() as conn:
            agg = conn.execute(text("""
                SELECT COUNT(*) AS n,
                       COUNT(DISTINCT primary_iso3) AS countries,
                       MAX(published_at) AS newest
                  FROM situation_report""")).mappings().first()
    except Exception as e:
        return {"configured": bool(RELIEFWEB_APPNAME), "error": str(e)}
    used = _rw_calls_used_today()
    return {
        "configured": bool(RELIEFWEB_APPNAME),
        "base_url": reliefweb_source.BASE_URL,
        "appname_required_since": "2025-11-01",
        "quota_calls_per_day": reliefweb_source.QUOTA_CALLS_PER_DAY,
        "calls_used_today": used,
        "calls_remaining_today": max(0, reliefweb_source.QUOTA_CALLS_PER_DAY - used),
        "reports": int(agg["n"]) if agg else 0,
        "countries": int(agg["countries"]) if agg else 0,
        "newest_published_at": (agg["newest"].isoformat()
                                if agg and agg["newest"] else None),
    }
