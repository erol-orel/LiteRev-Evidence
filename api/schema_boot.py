"""Startup DDL: tables, columns and indexes the app adds on top of schema.sql.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
import re

from sqlalchemy import text, bindparam

import lexical_search as _lex
import llm_usage as _llm_usage

from .core import _is_openai_quota_error, _openai_in_cooldown, _trip_openai_cooldown, app, engine, logger
from .documents import _embed_chunks_resilient
from llm_usage import model_for as _model

# ─────────────────────────────────────────────────────────────────────────────
# Index de performance
# ─────────────────────────────────────────────────────────────────────────────
# Instructions de DDL de démarrage qui ont ÉCHOUÉ, accumulées au fil des _ensure_*.
# Le DDL de démarrage échoue OUVERT par conception (un serveur qui refuse de démarrer
# est pire) - mais jusqu'ici la seule trace était une ligne de log, et /health
# continuait d'annoncer « ok » sur une base incomplète. Cette liste rend l'état
# dégradé LISIBLE (cf. /health).
_SCHEMA_DDL_FAILURES: list[str] = []

# Tables sans lesquelles l'application ne peut pas servir ses écrans principaux.
_REQUIRED_TABLES = (
    "literature_document", "document_chunk", "article_scenarios",
    "user_scenarios", "user_scenario_folders", "scenario_settings",
)


def _exec_ddl_isolated(statements, label: str, record: bool = True) -> list[str]:
    """Exécute des instructions DDL CHACUNE DANS SA PROPRE TRANSACTION.

    Pourquoi : Postgres avorte la transaction ENTIÈRE à la première erreur. Regroupées
    dans un seul `with engine.begin()`, une instruction qui échoue (p. ex. un ALTER sur
    une table absente) annulait les CREATE TABLE réussis qui la précédaient dans le même
    bloc - la base ressortait SANS les tables que la fonction venait de créer, et le seul
    indice était un avertissement dans les logs. C'est exactement ce qui rendait toute
    base NEUVE inutilisable (voir tests/test_fresh_db_bootstrap.py).

    Renvoie la liste des instructions ayant échoué (vide = tout est passé).

    `record=False` : l'échec est journalisé et renvoyé, mais PAS inscrit dans
    _SCHEMA_DDL_FAILURES - donc sans effet sur `schema.ok`, qui est BLOQUANT au
    déploiement (scripts/check_health.py). À réserver aux objets dont l'absence
    dégrade (recherche plus lente) sans rien casser."""
    failed: list[str] = []
    for _sql in statements:
        try:
            with engine.begin() as _c:
                _c.execute(text(_sql))
        except Exception as _e:                     # noqa: BLE001 - best-effort par design
            _first = _sql.strip().split("\n")[0][:120]
            failed.append(_first)
            if record:
                _SCHEMA_DDL_FAILURES.append(f"{label}: {_first}")
            logger.warning(f"{label}: DDL ignorée ({_e.__class__.__name__}): {_first}")
    return failed


def _ensure_llm_usage_table() -> None:
    """Table de COMPTABILITÉ des appels OpenAI (cf. llm_usage.py).

    Créée AVANT le worker d'arrière-plan (lancé à l'import, plus bas) : sans elle, les
    premiers cycles d'embedding/PICO dépenseraient sans laisser de trace. `configure()`
    donne au module l'engine ; sans cet appel il n'enregistre rien (et ne casse rien)."""
    _llm_usage.configure(engine)
    _exec_ddl_isolated(_llm_usage.DDL, "_ensure_llm_usage_table")


try:
    _ensure_llm_usage_table()
except Exception as _e:                               # jamais bloquant : c'est de la mesure
    logger.warning(f"_ensure_llm_usage_table: {_e}")


def _ensure_document_search() -> None:
    """Table `document_search` + fonction + triggers de la recherche PLEIN TEXTE
    (cf. lexical_search.py) : un tsvector par document, tenu à jour par triggers, qui
    remplace les LIKE '%terme%' du match booléen (55 à 240 s par requête en prod).

    `record=False` : si ces objets ne peuvent pas être créés (droits sur les tables,
    par ex.), la recherche booléenne reste sur le chemin LIKE - plus lente, pas
    cassée. Ce n'est donc pas une dégradation du schéma au sens de `schema.ok`
    (bloquant au déploiement) ; l'état est exposé dans /health → lexical_search."""
    _lex.configure(engine)
    _lex.DDL_FAILURES[:] = _exec_ddl_isolated(_lex.DDL, "_ensure_document_search", record=False)


try:
    _ensure_document_search()
except Exception as _e:
    logger.warning(f"_ensure_document_search: {_e}")


#: L'index unique qui ferme la course entre fetchers parallèles sur le titre normalisé,
#: avec la règle de dédup de _ingest_doc_direct : même titre (≥ 20 caractères normalisés)
#: ET même DOI normalisé, un DOI absent valant ''. Il portait sur (project_context,
#: title_norm) seul : deux notices de même titre générique et de DOI différents ne
#: pouvaient pas coexister, la seconde était refusée et repliée sur la première. La clé
#: nouvelle est strictement plus lâche que l'ancienne : construite sur une table que
#: l'ancienne tenait sans doublon, elle ne peut pas en trouver.
TITLE_DOI_INDEX = "uq_litdoc_title_norm_doi"
_OLD_TITLE_INDEX = "uq_litdoc_title_norm"
_TITLE_DOI_INDEX_SQL = (
    f"CREATE UNIQUE INDEX CONCURRENTLY IF NOT EXISTS {TITLE_DOI_INDEX} "
    "ON literature_document (project_context, title_norm, lower(btrim(COALESCE(doi, '')))) "
    "WHERE title_norm IS NOT NULL AND length(title_norm) >= 20 "
    "AND (is_duplicate IS NULL OR is_duplicate = FALSE)")


def _ensure_title_doi_unique_index() -> str:
    """Remplace uq_litdoc_title_norm par uq_litdoc_title_norm_doi, au démarrage.

    Sur la table de production (≈ 494 000 lignes), sans bloquer les écritures : le nouvel
    index est bâti CONCURRENTLY, donc hors de toute transaction (AUTOCOMMIT), AVANT que
    l'ancien soit retiré, et l'ancien n'est retiré que si le nouveau est VALIDE. Un
    échec de construction laisse un index invalide, retiré aussitôt ; l'ancien reste
    alors en place et la relance suivante réessaie. Idempotent et peu coûteux à rejouer :
    une lecture du catalogue, un IF NOT EXISTS, un DROP IF EXISTS. Renvoie « ok »,
    « invalid » ou « failed »."""
    def _state(conn):
        return conn.execute(text("SELECT i.indisvalid FROM pg_index i "
                                 "WHERE i.indexrelid = to_regclass(:n)"),
                            {"n": TITLE_DOI_INDEX}).scalar()
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            if _state(conn) is False:                 # reste d'une construction interrompue
                conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {TITLE_DOI_INDEX}"))
            try:
                conn.execute(text(_TITLE_DOI_INDEX_SQL))
            except Exception as e:                       # noqa: BLE001
                logger.warning(f"{TITLE_DOI_INDEX} : construction impossible ({e}) ; "
                               f"{_OLD_TITLE_INDEX} reste en place")
                conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {TITLE_DOI_INDEX}"))
                return "failed"
            if not _state(conn):
                conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {TITLE_DOI_INDEX}"))
                return "invalid"
            conn.execute(text(f"DROP INDEX CONCURRENTLY IF EXISTS {_OLD_TITLE_INDEX}"))
        return "ok"
    except Exception as e:                               # noqa: BLE001
        logger.warning(f"_ensure_title_doi_unique_index : {e}")
        return "failed"


def _ensure_performance_indexes() -> None:
    """Crée les index manquants sur les colonnes chaudes + un index ANN pgvector.

    Exécuté EN ARRIÈRE-PLAN (hors du chemin de démarrage) : la 1re création de
    l'index vectoriel peut durer plusieurs minutes sur un gros corpus et ne doit
    pas retarder le health check du déploiement. Tout est IF NOT EXISTS (idempotent)
    et chaque instruction est isolée : un échec est journalisé sans rien casser
    (au pire, la requête concernée reste en scan séquentiel, comme aujourd'hui)."""
    # 1) Index B-tree sur les colonnes de filtre / jointure les plus fréquentes.
    btree = [
        # Extension trigramme : permet aux index GIN gin_trgm_ops (créés plus bas)
        # d'assister les LIKE '%terme%' du match booléen. Tolérant : si le rôle DB ne
        # peut pas créer l'extension, les index GIN échouent et on reste en scan.
        "CREATE EXTENSION IF NOT EXISTS pg_trgm",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scenario ON article_scenarios (scenario_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_document ON article_scenarios (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_article_scenarios_scen_sim ON article_scenarios (scenario_id, similarity_score)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_screening_status ON literature_document (screening_status)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_is_duplicate ON literature_document (is_duplicate)",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_source ON literature_document (source)",
        # PMID et external_id : les DEUX clés par lesquelles le populate PubMed demande
        # « ces 1 000 identifiants sont-ils déjà en base ? ». Sans index, chaque lot
        # balayait toute la table, une fois par lot et par recherche ; et comme la
        # requête réunit maintenant deux recherches par UNION au lieu d'un OR sur deux
        # colonnes, il faut un index pour chacune.
        "CREATE INDEX IF NOT EXISTS ix_litdoc_pmid ON literature_document (pmid) "
        "WHERE pmid IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_external_id ON literature_document (external_id) "
        "WHERE external_id IS NOT NULL",
        # La cible que l'INSERT de la living review nomme : `ON CONFLICT (external_id,
        # source)`. Aucun index unique n'y correspondait, donc Postgres rejetait CHAQUE
        # insertion (« there is no unique or exclusion constraint matching the ON
        # CONFLICT specification »), l'exception était avalée par un `except` qui
        # journalise, et le compteur d'erreurs montait pendant que la tâche se déclarait
        # réussie. La living review n'a jamais pu ingérer un article.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_litdoc_external_source "
        "ON literature_document (external_id, source) "
        "WHERE external_id IS NOT NULL AND external_id <> ''",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_project_context ON literature_document (project_context)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_document ON document_chunk (document_id)",
        "CREATE INDEX IF NOT EXISTS ix_doc_chunk_type ON document_chunk (chunk_type)",
        # Dédup par titre : backfill title_norm des lignes existantes (idempotent -
        # WHERE title_norm IS NULL) puis index. Même normalisation que _normalize_title.
        "UPDATE literature_document SET title_norm = btrim(regexp_replace(lower(title), '[^a-z0-9]+', ' ', 'g')) "
        "WHERE title_norm IS NULL AND title IS NOT NULL",
        "CREATE INDEX IF NOT EXISTS ix_litdoc_title_norm ON literature_document (project_context, title_norm)",
        # Index UNIQUES rendant l'INSERT de _ingest_doc_direct atomique (ON CONFLICT
        # DO NOTHING) - ferment la course entre fetchers parallèles. best-effort : le
        # try/except par instruction ci-dessous journalise un échec sans rien casser.
        #  • DOI : présent en prod via un script ad-hoc mais NON versionné (cf.
        #    PIPELINE_AUDIT A1) - (re)créé ici pour garantir une cible à ON CONFLICT,
        #    y compris après une reconstruction depuis les seules migrations.
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_literature_document_doi "
        "ON literature_document (doi) WHERE doi IS NOT NULL",
        #  • Titre normalisé : cf. _ensure_title_doi_unique_index, juste après.
    ]
    for ddl in btree:
        try:
            with engine.begin() as conn:
                conn.execute(text(ddl))
        except Exception as e:
            logger.warning(f"_ensure_performance_indexes (btree) « {ddl[:60]}… » : {e}")
    _ensure_title_doi_unique_index()

    # 2) Index ANN pgvector (HNSW, cosinus). CONCURRENTLY → hors transaction, ne
    # bloque pas les écritures. Accélère « ORDER BY embedding <=> q LIMIT k » qui
    # sinon scanne TOUS les embeddings à chaque requête.
    ann_ddl = ("CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_doc_chunk_embedding_hnsw "
               "ON document_chunk USING hnsw (embedding vector_cosine_ops)")
    try:
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
            conn.execute(text(ann_ddl))
        logger.info("Index ANN pgvector (HNSW) vérifié/créé.")
    except Exception as e:
        logger.warning(f"Index ANN pgvector indisponible ({e}); recherche vectorielle en scan séquentiel.")

    # 3) Index GIN TRIGRAMME (pg_trgm) sur titre / résumé / contenu : rendent les
    # `LOWER(COALESCE(x,'')) LIKE '%terme%'` du match booléen index-assistés. Le
    # wildcard EN TÊTE interdit tout index B-tree → sans ceci, chaque terme scanne
    # séquentiellement 207k docs (× termes × facettes) - cause des recherches à
    # plusieurs minutes. CONCURRENTLY + AUTOCOMMIT : build hors transaction, sans
    # bloquer les écritures ; chaque index isolé/tolérant. L'expression indexée est
    # IDENTIQUE à celle de la requête (sinon le planificateur n'utilise pas l'index).
    trgm_ddl = [
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_litdoc_title_trgm "
        "ON literature_document USING gin (LOWER(COALESCE(title,'')) gin_trgm_ops)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_litdoc_abstract_trgm "
        "ON literature_document USING gin (LOWER(COALESCE(abstract,'')) gin_trgm_ops)",
        "CREATE INDEX CONCURRENTLY IF NOT EXISTS ix_docchunk_content_trgm "
        "ON document_chunk USING gin (LOWER(COALESCE(content,'')) gin_trgm_ops)",
    ]
    for _tddl in trgm_ddl:
        try:
            with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as conn:
                conn.execute(text(_tddl))
            logger.info(f"Index trigramme vérifié/créé : « {_tddl[:55]}… ».")
        except Exception as e:
            logger.warning(f"Index trigramme indisponible « {_tddl[:55]}… » ({e}); LIKE en scan séquentiel.")


# ─────────────────────────────────────────────────────────────────────────────
# Startup
# ─────────────────────────────────────────────────────────────────────────────
def _warm_clustering_kernels() -> None:
    """Compile les noyaux numba d'UMAP et d'HDBSCAN sur un jeu minuscule, une fois par
    processus. Sans cela, la PREMIÈRE visualisation après un redémarrage payait ≈ 30 s de
    compilation sous les yeux de l'utilisateur (et le préchauffage figurait au runbook
    comme geste manuel). Best-effort : n'échoue jamais, ne bloque pas le démarrage."""
    import time as _t
    t0 = _t.time()
    try:
        import numpy as _np
        X = _np.random.RandomState(0).rand(60, 16).astype("float32")
        try:
            import umap as _umap
            # MÊME configuration que le clustering réel (_cluster_core) : métrique cosinus,
            # low_memory, n_epochs. Préchauffer la métrique euclidienne par défaut
            # compilait un noyau que l'application n'utilise jamais, et la première
            # visualisation payait quand même la compilation.
            _umap.UMAP(n_neighbors=8, n_components=2, metric="cosine", random_state=42,
                       low_memory=True, n_epochs=200).fit_transform(X)
        except Exception as _e:                              # noqa: BLE001
            logger.info(f"warm-up UMAP: {_e}")
        try:
            import hdbscan as _hdb
            _hdb.HDBSCAN(min_cluster_size=5).fit(X)
        except Exception as _e:                              # noqa: BLE001
            logger.info(f"warm-up HDBSCAN: {_e}")
        logger.info(f"Warm-up clustering kernels: {_t.time() - t0:.1f} s")
    except Exception as _e:                                  # noqa: BLE001
        logger.warning(f"warm-up: {_e}")


def _startup_flag(name: str, default: str = "1") -> bool:
    return os.getenv(name, default).strip().lower() not in ("0", "false", "no", "off")


# ─────────────────────────────────────────────────────────────────────────────
# Backfill des résumés (worker d'arrière-plan)
# ─────────────────────────────────────────────────────────────────────────────
EPMC_REST_SEARCH = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
#: Europe PMC rend au plus 1 000 notices par page. Une page à la taille du lot (20 DOI)
#: poussait hors de la page les notices d'un DOI qui en a plusieurs, et l'absence d'un
#: DOI de la page se lisait « pas de résumé » : la notice était marquée tentée à jamais.
_EPMC_ABSTRACT_PAGE = 1000
_EPMC_ABSTRACT_GROUP = 20          # identifiants par requête


def _clean_epmc_abstract(raw: str | None) -> str | None:
    """Le résumé Europe PMC sans son balisage JATS/HTML, ou None s'il n'en est pas un."""
    if not raw:
        return None
    ab = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", raw)).strip()
    return ab if len(ab) >= 30 else None


def _epmc_abstract_lookup(kind: str, values: list[str], http_get=None) -> tuple[dict[str, str | None], bool]:
    """Les résumés de `values` (DOI si kind='doi', PMID si kind='pmid') demandés à Europe
    PMC en UNE requête. Renvoie ({valeur: résumé ou None} pour chaque valeur RÉPONDUE, ok).

    Une valeur est répondue quand la requête a abouti et qu'elle figure dans la page, ou
    que la page tient TOUTES les notices trouvées (son absence est alors un vrai « pas de
    notice »). `ok=False` : la requête a échoué (réseau, 5xx, 429) ; rien n'est répondu,
    rien ne sera marqué tenté, le prochain cycle réessaie. Un refus 4xx est propre à la
    requête : un lot est redemandé valeur par valeur, et une valeur seule refusée est
    répondue « sans résumé » (sinon elle bloquerait la file à jamais)."""
    import requests as _rq
    http_get = http_get or _rq.get
    vals = [v for v in dict.fromkeys(values) if v]
    if not vals:
        return {}, True
    if kind == "doi":
        # Ni guillemet ni barre oblique inverse dans la phrase : ils casseraient la requête.
        clauses = ['DOI:"' + re.sub(r'["\\]', "", v) + '"' for v in vals]
    else:
        clauses = [f"(EXT_ID:{v} AND SRC:MED)" for v in vals if v.isdigit()]
        if not clauses:
            return {v: None for v in vals}, True
    try:
        r = http_get(EPMC_REST_SEARCH,
                     params={"query": " OR ".join(clauses), "resultType": "core",
                             "format": "json", "pageSize": _EPMC_ABSTRACT_PAGE},
                     timeout=30)
    except Exception as e:                               # noqa: BLE001 - réseau : réessayé plus tard
        logger.info(f"EuropePMC abstract lookup ({kind}, {len(vals)}): {e}")
        return {}, False
    status = int(getattr(r, "status_code", 200) or 200)
    if 400 <= status < 500 and status != 429:
        if len(vals) == 1:
            return {vals[0]: None}, True
        out: dict[str, str | None] = {}
        ok_all = True
        for v in vals:
            sub, ok = _epmc_abstract_lookup(kind, [v], http_get)
            out.update(sub)
            ok_all = ok_all and ok
        return out, ok_all
    if status >= 400:
        logger.info(f"EuropePMC abstract lookup ({kind}, {len(vals)}): HTTP {status}")
        return {}, False
    try:
        data = r.json() or {}
    except Exception as e:                               # noqa: BLE001
        logger.info(f"EuropePMC abstract lookup ({kind}): {e}")
        return {}, False
    results = (data.get("resultList") or {}).get("result") or []
    found: dict[str, str | None] = {}
    for res in results:
        key = ((res.get("doi") or "").lower().strip() if kind == "doi"
               else str(res.get("pmid") or "").strip())
        if not key:
            continue
        ab = _clean_epmc_abstract(res.get("abstractText"))
        if ab or key not in found:
            found[key] = ab or found.get(key)
    try:
        complete = int(data.get("hitCount") or 0) <= len(results)
    except (TypeError, ValueError):
        complete = False
    answered: dict[str, str | None] = {}
    for v in vals:
        k = v.lower().strip() if kind == "doi" else v
        if k in found:
            answered[v] = found[k]
        elif complete:
            answered[v] = None
    return answered, True


def _backfill_abstracts_once(limit: int = 50, http_get=None) -> dict[str, int]:
    """Un passage du backfill des résumés : les notices `literev` sans résumé qui ont un
    DOI ou, à défaut, un PMID, demandées à Europe PMC.

    Seules les notices dont l'identifiant a été RÉPONDU sont marquées tentées. La version
    d'avant avalait l'erreur d'une requête échouée et marquait quand même les cinquante
    notices du lot : une panne passagère d'Europe PMC les privait de résumé à jamais. Un
    résumé trouvé reconstruit le chunk titre+résumé de la notice (cf.
    sources._refresh_title_abstract_chunks), pour qu'elle soit scorée sur son résumé et
    non sur son titre, et la rend éligible au sous-ensemble pertinent d'elle-même."""
    from .relevance import _backfill_title_abstract_chunks  # lazy: relevance is loaded after this module
    from .sources import _refresh_title_abstract_chunks  # lazy: sources is loaded after this module
    stats = {"candidates": 0, "filled": 0, "marked": 0, "failed_requests": 0}
    with engine.connect() as conn:
        rows = conn.execute(text("""
            SELECT id, doi, pmid::text AS pmid FROM literature_document
            WHERE project_context = 'literev'
              AND (NULLIF(btrim(doi), '') IS NOT NULL OR NULLIF(btrim(pmid::text), '') IS NOT NULL)
              AND (abstract IS NULL OR length(trim(abstract)) < 30)
              AND abstract_backfill_attempted IS NOT TRUE
            ORDER BY id LIMIT :lim
        """), {"lim": int(limit)}).mappings().fetchall()
    stats["candidates"] = len(rows)
    if not rows:
        return stats
    by_doi: dict[str, list[int]] = {}
    by_pmid: dict[str, list[int]] = {}
    for r in rows:
        doi = (r["doi"] or "").strip()
        pmid = re.sub(r"\D", "", str(r["pmid"] or ""))
        if doi:
            by_doi.setdefault(doi.lower(), []).append(int(r["id"]))
        elif pmid:
            by_pmid.setdefault(pmid, []).append(int(r["id"]))
    filled_ids: list[int] = []
    marked_ids: list[int] = []
    for kind, groups in (("doi", by_doi), ("pmid", by_pmid)):
        keys = list(groups)
        for k in range(0, len(keys), _EPMC_ABSTRACT_GROUP):
            chunk = keys[k:k + _EPMC_ABSTRACT_GROUP]
            answered, ok = _epmc_abstract_lookup(kind, chunk, http_get)
            if not ok:
                stats["failed_requests"] += 1
            if not answered:
                if not ok:
                    break                                # service en panne : le cycle suivant réessaie
                continue
            chunk_filled: list[int] = []
            with engine.begin() as cu:
                for value, abstract in answered.items():
                    ids = groups.get(value, [])
                    marked_ids.extend(ids)
                    if not abstract:
                        continue
                    for doc_id in ids:
                        n = cu.execute(text("""
                            UPDATE literature_document SET abstract = :ab
                            WHERE id = :id AND (abstract IS NULL OR length(trim(abstract)) < 30)
                        """), {"ab": abstract, "id": doc_id}).rowcount
                        if n:
                            chunk_filled.append(doc_id)
                if chunk_filled:
                    _refresh_title_abstract_chunks(cu, chunk_filled)
            filled_ids.extend(chunk_filled)
            if not ok:
                break
    if marked_ids:
        with engine.begin() as cu:
            cu.execute(text("UPDATE literature_document SET abstract_backfill_attempted = TRUE "
                            "WHERE id = ANY(CAST(:ids AS bigint[]))"), {"ids": marked_ids})
    stats["filled"] = len(filled_ids)
    stats["marked"] = len(marked_ids)
    if filled_ids:
        # Les notices qui n'avaient AUCUN chunk (titre trop court) en reçoivent un.
        _backfill_title_abstract_chunks()
        logger.info(f"BG abstract backfill: {len(filled_ids)} résumés récupérés (EuropePMC).")
    return stats


def background_pico_sql() -> str:
    """Le lot du PICO automatique : les articles PERTINENTS d'au moins un scénario.

    Le worker extrayait le PICO de TOUT article lié à un scénario, pertinent ou non. Sur
    HPAI_last : 15 547 PICO écrits pour 1 075 articles pertinents, soit quatorze appels
    payés pour un qui sert. Le PICO ne sert qu'au sous-ensemble pertinent (digest, brief,
    extraction) ; la porte est celle de tout le reste (`scenario_scope_sql('relevant')`),
    avec le seuil de CHAQUE scénario lu dans la requête. Un article qui devient pertinent
    (seuil abaissé, inclusion à la main) entre au cycle suivant ; un lot explicite
    (POST /pico/extract, portée « all ») reste possible pour tout le corpus.

    On extrait le PICO des articles sans PICO, PUIS on ré-extrait ceux dont le PICO venait
    du résumé alors que le texte intégral est désormais disponible (pico_source !=
    'fulltext'). Les articles jamais traités passent en premier. Pur : aucune connexion."""
    from .scenario_store import scenario_scope_sql   # lazy: une seule définition de la porte
    gate = scenario_scope_sql("relevant", doc="literature_document", link="ars", sid="ars.scenario_id")
    return f"""
        SELECT id, title, abstract, has_fulltext
        FROM literature_document
        WHERE project_context = 'literev'
          AND abstract IS NOT NULL
          AND LENGTH(abstract) > 50
          AND COALESCE(pico_attempts, 0) < :max_attempts  -- borne les échecs déterministes
          AND EXISTS (
              SELECT 1 FROM article_scenarios ars
              WHERE ars.document_id = literature_document.id
                AND {gate}
          )
          AND (
            pico_json IS NULL
            OR (
                has_fulltext IS TRUE
                AND (pico_json->>'pico_source') IS DISTINCT FROM 'fulltext'
                AND pico_fulltext_attempted IS NOT TRUE  -- une seule tentative
            )
          )
        ORDER BY (pico_json IS NULL) DESC, id
        LIMIT :lim
    """


@app.on_event("startup")
def startup_event() -> None:
    from .scenarios import _pipeline_jobs_lock, _user_scenario_pipeline_jobs  # lazy: scenarios is loaded after this module
    from .pipeline import _run_user_scenario_full_pipeline  # lazy: pipeline is loaded after this module
    from .model_training import _seed_demo_scenarios  # lazy: model_training is loaded after this module
    with engine.connect() as conn:
        conn.execute(text("SELECT 1"))
    logger.info("Database connection OK")

    # Index de performance (B-tree + ANN pgvector) : en arrière-plan pour ne pas
    # retarder le démarrage / le health check ; idempotent et sans effet de bord
    # en cas d'échec (au pire, on reste sur le comportement actuel).
    try:
        import threading as _perf_threading
        _perf_threading.Thread(
            target=_ensure_performance_indexes, daemon=True, name="ensure-perf-indexes"
        ).start()
    except Exception as _e:
        logger.warning(f"spawn _ensure_performance_indexes: {_e}")

    # Préchauffage des noyaux UMAP/HDBSCAN (≈ 30 s de compilation numba, en arrière-plan),
    # pour que la première visualisation après ce redémarrage ne les paie pas.
    # WARM_ON_STARTUP=0 pour désactiver (tests, CI).
    if _startup_flag("WARM_ON_STARTUP"):
        try:
            import threading as _warm_threading
            _warm_threading.Thread(target=_warm_clustering_kernels, daemon=True, name="warm-umap").start()
        except Exception as _e:
            logger.warning(f"spawn warm-up: {_e}")

    # Recherche plein texte : remplissage initial puis rafraîchissement de
    # document_search en arrière-plan (cf. lexical_search.py). Tant que le remplissage
    # n'est pas COMPLET, la recherche booléenne reste sur le chemin LIKE.
    try:
        _lex.start_worker()
    except Exception as _e:
        logger.warning(f"spawn document_search worker: {_e}")

    # Scénario de démonstration intégré (dataset RÉEL grippe + modèle entraîné) : rend
    # l'essai « données réelles » visible dans la liste. Idempotent (id stable) et
    # best-effort - en arrière-plan pour ne jamais retarder/casser le démarrage.
    try:
        import threading as _seed_threading
        _seed_threading.Thread(
            target=_seed_demo_scenarios, daemon=True, name="seed-demo"
        ).start()
    except Exception as _e:
        logger.warning(f"spawn _seed_demo_scenarios: {_e}")

    # Pipelines orphelins : tout pipeline marqué 'running' ou 'starting' au
    # démarrage du serveur est forcément mort (le thread a été tué lors du
    # redémarrage précédent).
    # Stratégie : on les relance automatiquement en arrière-plan plutôt que
    # de les marquer 'failed' et forcer l'utilisateur à les relancer manuellement.
    try:
        with engine.connect() as _startup_conn:
            _orphan_rows = _startup_conn.execute(text("""
                SELECT id, query, filters, pipeline_lang
                FROM user_scenarios
                WHERE pipeline_status IN ('running', 'starting')
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).mappings().fetchall()
            _pop_orphan_rows = _startup_conn.execute(text("""
                SELECT id, query, filters, pipeline_lang FROM user_scenarios
                WHERE populate_status = 'running'
                  AND COALESCE(is_system, FALSE) = FALSE
            """)).mappings().fetchall()

        # Recherches orphelines (populate 'running' au moment du redémarrage) : RELANCÉES,
        # même requête et même langue, plutôt que passées à 'error' - la page de recherche
        # qui les attendait retrouve un job en cours, et le pipeline complet enchaîne
        # ensuite comme après toute recherche. RESUME_ON_STARTUP=0 : marquées 'error'.
        _relaunched_pop: set[str] = set()
        if _pop_orphan_rows:
            if _startup_flag("RESUME_ON_STARTUP"):
                from .scenarios import LIVE_MAX_PER_SOURCE, _launch_populate_job  # lazy: scenarios is loaded after this module
                for _po in _pop_orphan_rows:
                    if not _po.get("query"):
                        continue
                    try:
                        _launch_populate_job(_po["id"], _po["query"], _po.get("filters") or {},
                                             LIVE_MAX_PER_SOURCE, True, _po.get("pipeline_lang"))
                        _relaunched_pop.add(_po["id"])
                        logger.warning(f"Startup: recherche {_po['id']} interrompue par le redémarrage → relancée.")
                    except Exception as _e_po:
                        logger.warning(f"Startup: relance populate {_po['id']}: {_e_po}")
            else:
                with engine.begin() as _c:
                    _c.execute(text("""
                        UPDATE user_scenarios
                        SET populate_status = 'error', updated_at = NOW()
                        WHERE populate_status = 'running'
                          AND COALESCE(is_system, FALSE) = FALSE
                    """))
                logger.warning(f"Startup: {len(_pop_orphan_rows)} populate(s) orphelin(s) reinitialisé(s) à 'error'.")

        # Relancer automatiquement les pipelines interrompus (sauf ceux dont la recherche
        # vient d'être relancée : le pipeline enchaînera à la fin de celle-ci).
        _orphan_rows = [r for r in _orphan_rows if r["id"] not in _relaunched_pop]
        if _orphan_rows:
            import threading as _startup_threading
            logger.warning(
                f"Startup: {len(_orphan_rows)} pipeline(s) interrompu(s) détecté(s) - "
                f"relance automatique en arrière-plan."
            )
            for _orphan in _orphan_rows:
                _oid = _orphan["id"]
                _oquery = _orphan["query"] or ""
                _ofilters = _orphan["filters"] or {}
                if not _oquery:
                    # Pas de requête → on ne peut pas relancer, marquer failed
                    with engine.begin() as _c:
                        _c.execute(text("""
                            UPDATE user_scenarios
                            SET pipeline_status = 'failed',
                                pipeline_step   = NULL,
                                updated_at      = NOW()
                            WHERE id = :sid
                        """), {"sid": _oid})
                    logger.warning(f"Startup: pipeline {_oid} sans requête → marqué 'failed'.")
                    continue
                # Initialiser le job en mémoire
                _olang = (_orphan.get("pipeline_lang") or "fr").lower()
                with _pipeline_jobs_lock:
                    _user_scenario_pipeline_jobs[_oid] = {
                        "overall_status": "starting",
                        "current_step": "ingest",
                        "auto_restarted": True,
                        "lang": _olang,
                        "steps": {k: {"status": "pending"} for k in (
                            "ingest", "fulltext", "embed", "rerank", "pico", "metadata",
                            "clustering", "knowledge_graph", "evidence", "variables", "actions")},
                    }
                _t = _startup_threading.Thread(
                    target=_run_user_scenario_full_pipeline,
                    args=(_oid, _oquery, _ofilters),
                    kwargs={"lang": _olang},              # même langue qu'avant l'interruption
                    daemon=True,
                )
                _t.start()
                logger.info(f"Startup: pipeline {_oid} relancé automatiquement (query={_oquery[:60]!r}).")
    except Exception as _se:
        logger.error(f"Startup cleanup/relance pipelines orphelins: {_se}")

    import threading
    # ── Worker d'enrichissement automatique (embedding + PICO) ──────────────
    # Tourne en permanence en arrière-plan. Chaque cycle :
    #   1. Embède tous les chunks title_abstract/fulltext_section sans embedding
    #   2. Extrait le PICO des articles PERTINENTS d'un scénario (background_pico_sql)
    def _background_enrichment_worker():
        import time as _time
        from llm_usage import MeteredOpenAI as _OAI_bg
        from concurrent.futures import ThreadPoolExecutor as _TPE
        from datetime import datetime, timezone

        _EMBED_BATCH   = 100   # chunks par appel OpenAI embeddings
        _PICO_WORKERS  = 5     # threads parallèles pour extraction PICO
        _CYCLE_SLEEP   = 30    # secondes entre deux cycles
        _PICO_BATCH    = 50    # articles PICO par cycle
        _PICO_MAX_ATTEMPTS = 3 # essais max par article (borne les échecs déterministes)
        _ABS_BATCH     = 50    # notices sans résumé traitées par cycle (backfill)

        def _extract_pico_one(row, client):
            from .pico_map import PicoUnreadable, extract_pico, pico_fulltext_chunks
            try:
                title    = row["title"] or ""
                abstract = row["abstract"] or ""
                # Le texte lu : TOUJOURS le résumé, plus, quand le texte intégral existe,
                # ses méthodes et ses résultats (api/pico_map.pico_source_text). Il
                # remplaçait le résumé et n'en lisait que les 14 000 premiers caractères :
                # l'en-tête, l'introduction, le début des méthodes. On marque la source
                # (`pico_source`) pour ré-extraire plus tard les articles dont le PICO
                # venait du résumé seul.
                chunks = []
                if row.get("has_fulltext"):
                    with engine.connect() as _ftc:
                        chunks = pico_fulltext_chunks(_ftc, row["id"])
                # Appel LLM isolé : une erreur d'API (quota/réseau) est TRANSITOIRE
                # → on ne compte PAS de tentative (le cooldown s'en charge) et on
                # réessaiera quand l'API sera saine.
                #
                # Une RÉPONSE inutilisable (coupée deux fois, JSON invalide), elle, COMPTE
                # la tentative : un article dont la sortie est malformée de façon
                # déterministe (temperature=0/seed=42) serait sinon ré-extrait à l'infini.
                # pico_attempts (plafonné par _PICO_MAX_ATTEMPTS dans le sélecteur) borne
                # ces échecs. Les clés manquantes, elles, sont complétées (TOLÉRANT).
                try:
                    # `pico_source` (fulltext ou abstract) est porté par le PICO lui-même.
                    _pico, _src = extract_pico(client, title, abstract, chunks,
                                               label=f"BG PICO {row['id']}")
                except PicoUnreadable as _bad:
                    logger.debug(f"BG PICO doc {row['id']}: {_bad}")
                    _pico = None
                except Exception as _api_e:
                    if _is_openai_quota_error(_api_e):
                        _trip_openai_cooldown()
                    logger.debug(f"BG PICO API doc {row['id']}: {_api_e}")
                    return None

                with engine.begin() as _c:
                    if isinstance(_pico, dict):
                        _c.execute(text("""
                            UPDATE literature_document
                            SET pico_json = CAST(:pico AS jsonb),
                                pico_extracted_at = :ts,
                                pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                               ELSE pico_fulltext_attempted END,
                                pico_attempts = COALESCE(pico_attempts, 0) + 1
                            WHERE id = :doc_id
                        """), {
                            "pico":   json.dumps(_pico),
                            "ts":     datetime.now(timezone.utc),
                            "hf":     bool(row.get("has_fulltext")),
                            "doc_id": row["id"],
                        })
                        return row["id"]
                    # JSON invalide : on compte la tentative (+ fulltext_attempted)
                    # pour borner les ré-essais → fin de la boucle de tokens.
                    _c.execute(text("""
                        UPDATE literature_document
                        SET pico_attempts = COALESCE(pico_attempts, 0) + 1,
                            pico_fulltext_attempted = CASE WHEN :hf THEN TRUE
                                                           ELSE pico_fulltext_attempted END
                        WHERE id = :doc_id
                    """), {"hf": bool(row.get("has_fulltext")), "doc_id": row["id"]})
                    return None
            except Exception as _pe:
                logger.debug(f"BG PICO doc {row['id']}: {_pe}")
                return None

        # Colonne de suivi : évite de re-tenter indéfiniment les notices dont
        # EuropePMC n'a pas de résumé (sinon le même lot bloquerait la file).
        try:
            with engine.begin() as _cc:
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS abstract_backfill_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Garde-fou anti-boucle : ne ré-extraire le PICO « texte intégral »
                # qu'UNE SEULE fois par article. Sans cela, tout article avec
                # has_fulltext=TRUE mais SANS chunk fulltext exploitable garde
                # pico_source='abstract' et re-matche le sélecteur PICO à CHAQUE
                # cycle (30 s) → ré-extraction gpt-4.1-mini infinie = fuite de tokens.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_fulltext_attempted BOOLEAN DEFAULT FALSE"
                ))
                # Compteur de tentatives PICO. Un article dont l'extraction échoue
                # de façon DÉTERMINISTE (sortie LLM malformée : clé manquante ou JSON
                # invalide) revenait dans le sélecteur à CHAQUE cycle et était ré-envoyé
                # à gpt-4.1-mini indéfiniment (même échec, temperature=0/seed=42) →
                # 2e fuite de tokens. On borne à _PICO_MAX_ATTEMPTS essais par article.
                _cc.execute(text(
                    "ALTER TABLE literature_document "
                    "ADD COLUMN IF NOT EXISTS pico_attempts INTEGER DEFAULT 0"
                ))
        except Exception as _ce:
            logger.warning(f"ensure backfill/pico-attempt columns: {_ce}")

        logger.info("Background enrichment worker started (abstract backfill + embedding + PICO).")
        while True:
            try:
                openai_key = os.getenv("OPENAI_API_KEY")
                if not openai_key:
                    _time.sleep(_CYCLE_SLEEP)
                    continue

                _client = _OAI_bg(api_key=openai_key, timeout=90.0)

                # ── 0. BACKFILL DES RÉSUMÉS (notices sans abstract, via DOI ou PMID) ──
                # Beaucoup de notices Crossref/OpenAlex arrivent sans résumé. On
                # tente de le récupérer via EuropePMC (par DOI, sinon par PMID) pour les
                # rendre exploitables (puis embedding + PICO par les étapes suivantes).
                # Cf. _backfill_abstracts_once : seules les notices répondues sont
                # marquées tentées, et le chunk titre+résumé est reconstruit.
                try:
                    _backfill_abstracts_once(_ABS_BATCH)
                except Exception as _abe:
                    logger.warning(f"BG abstract backfill error: {_abe}")

                # ── 1. EMBEDDING ──────────────────────────────────────────────
                # On embède TOUS les chunks standard sans embedding - y compris le
                # title_abstract des docs à texte intégral : le résumé sert de vecteur
                # représentatif CONSTANT au clustering (même base pour chaque document)
                # pour un coût négligeable. (Avant, on le sautait pour les docs full-text
                # → incohérence : la plupart embeddés, ~153 non, et clustering par 1re
                # section au lieu du résumé.)
                with engine.connect() as _conn:
                    _chunks = _conn.execute(text("""
                        SELECT c.id, c.content
                        FROM document_chunk c
                        WHERE c.embedding IS NULL
                          AND c.chunk_type IN ('title_abstract', 'fulltext_section')
                          AND LENGTH(c.content) > 20
                          AND COALESCE(c.embedding_attempts, 0) < 3
                        ORDER BY c.chunk_type DESC, c.id
                        LIMIT 500
                    """)).mappings().fetchall()

                if _chunks and not _openai_in_cooldown():
                    try:
                        _emb_done, _failed = _embed_chunks_resilient(_client, list(_chunks))
                    except Exception:               # quota propagé → pause + réessai après cooldown
                        _trip_openai_cooldown()
                        _emb_done, _failed = 0, []
                    if _failed:
                        # Chunks refusés par l'API (hors quota) : incrémenter le compteur
                        # pour les sortir de la file après 3 essais (sinon « en attente »
                        # éternel + lot ré-échoué à chaque cycle).
                        with engine.begin() as _cu:
                            _cu.execute(
                                text("UPDATE document_chunk SET embedding_attempts = "
                                     "COALESCE(embedding_attempts,0)+1 WHERE id IN :ids")
                                .bindparams(bindparam("ids", expanding=True)),
                                {"ids": _failed},
                            )
                    if _emb_done:
                        logger.info(f"BG worker: {_emb_done} chunks embedded.")

                # ── 2. PICO ───────────────────────────────────────────────────
                with engine.connect() as _conn:
                    _pico_rows = _conn.execute(text(background_pico_sql()), {
                        "lim": _PICO_BATCH, "max_attempts": _PICO_MAX_ATTEMPTS,
                    }).mappings().fetchall()

                # Coupe-circuit : PICO_AUTOEXTRACT_ENABLED=0 dans /etc/literev-api.env
                # (puis restart) met en pause l'extraction PICO automatique sans
                # toucher au code - utile pour stopper net la dépense OpenAI.
                _pico_enabled = os.getenv("PICO_AUTOEXTRACT_ENABLED", "1").strip().lower() not in ("0", "false", "no", "off")
                if _pico_rows and _pico_enabled and not _openai_in_cooldown():
                    _pico_done = 0
                    with _TPE(max_workers=_PICO_WORKERS) as _pool:
                        _futs = {_pool.submit(_extract_pico_one, r, _client): r["id"] for r in _pico_rows}
                        for _f in _futs:
                            if _f.result() is not None:
                                _pico_done += 1
                    if _pico_done:
                        logger.info(f"BG worker: {_pico_done} PICO extracted.")

            except Exception as _we:
                logger.error(f"BG enrichment worker error: {_we}")

            _time.sleep(_CYCLE_SLEEP)

    threading.Thread(target=_background_enrichment_worker, daemon=True, name="bg-enrichment").start()
    logger.info("Background enrichment worker launched.")


# ─── DOUBLE-AVEUGLE SCREENING + KAPPA DE COHEN ───────────────────────────────

def _ensure_double_blind_columns():
    """Crée les colonnes reviewer_1_status/reviewer_2_status si elles n'existent pas."""
    with engine.begin() as conn:
        conn.execute(text("""
            ALTER TABLE literature_document
            ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20) DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT        DEFAULT NULL,
            ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN     DEFAULT FALSE,
            ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20) DEFAULT NULL
        """))
        # Par SCÉNARIO (article_scenarios) : mêmes colonnes, car un document
        # appartenant à plusieurs scénarios doit porter une décision double-aveugle
        # DISTINCTE par scénario (sinon les kappa se contaminent entre scénarios).
        # Miroir de la migration Alembic e2f6a8b3c5d7 (belt-and-suspenders au boot).
        if conn.execute(text("SELECT to_regclass('public.article_scenarios')")).scalar():
            conn.execute(text("""
                ALTER TABLE article_scenarios
                ADD COLUMN IF NOT EXISTS reviewer_1_status  VARCHAR(20),
                ADD COLUMN IF NOT EXISTS reviewer_1_reason  TEXT,
                ADD COLUMN IF NOT EXISTS reviewer_2_status  VARCHAR(20),
                ADD COLUMN IF NOT EXISTS reviewer_2_reason  TEXT,
                ADD COLUMN IF NOT EXISTS kappa_resolved     BOOLEAN DEFAULT FALSE,
                ADD COLUMN IF NOT EXISTS kappa_final_status VARCHAR(20),
                -- Quand chaque relecteur a voté, et combien de fois il est revenu
                -- dessus : un screening se relit, et une décision se corrige. Sans
                -- l'horodatage, l'historique d'un relecteur n'a pas d'ordre.
                ADD COLUMN IF NOT EXISTS reviewer_1_decided_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS reviewer_2_decided_at TIMESTAMP,
                ADD COLUMN IF NOT EXISTS reviewer_1_changes INTEGER DEFAULT 0,
                ADD COLUMN IF NOT EXISTS reviewer_2_changes INTEGER DEFAULT 0
            """))
    logger.info("Colonnes double-aveugle (global + par scénario) vérifiées/créées.")

# Appel au démarrage
try:
    _ensure_double_blind_columns()
except Exception as _e:
    logger.warning(f"_ensure_double_blind_columns: {_e}")


def _ensure_prior_screening_columns() -> None:
    """De quoi ANNULER un découpage sans détruire ce qu'il a recouvert.

    `undo` réécrivait littéralement 'pending' et NULL. Un article qu'un relecteur avait
    rescapé à la main, puis que le découpage avait exclu, revenait donc « en attente,
    sans motif » : sous le seuil et sans inclusion, il disparaissait du corpus pertinent,
    et le travail du relecteur avec lui. L'état antérieur est maintenant conservé."""
    with engine.begin() as conn:
        if not conn.execute(text("SELECT to_regclass('public.article_scenarios')")).scalar():
            return
        conn.execute(text("""
            ALTER TABLE article_scenarios
            ADD COLUMN IF NOT EXISTS prior_screening_status VARCHAR(20),
            ADD COLUMN IF NOT EXISTS prior_screening_reason TEXT
        """))
    logger.info("Colonnes d'état antérieur du screening (découpage) vérifiées/créées.")


try:
    _ensure_prior_screening_columns()
except Exception as _e:
    logger.warning(f"_ensure_prior_screening_columns: {_e}")


def _backfill_ars_screening_from_document() -> None:
    """Recopie UNE FOIS, sur les liens, les décisions qui ne vivaient que globalement.

    Le statut de screening se lisait `COALESCE(ars.screening_status,
    d.screening_status)` : à défaut de décision dans cette revue, la décision prise dans
    une AUTRE. `literature_document` étant partagé, un article exclu dans une revue
    sortait du sous-ensemble pertinent de toutes les autres qui le contiennent, sans
    qu'un seul de leurs écrans ne le dise.

    Les lectures ne regardent plus que le lien. Les décisions antérieures à l'écriture
    par scénario n'existent que sur la ligne globale : sans cette recopie, elles
    seraient silencieusement annulées, et un relecteur verrait ses exclusions revenir.
    On les recopie donc sur les liens EXISTANTS, ce qui ne change rien à ce qui est
    affiché aujourd'hui, et on arrête là : un scénario créé après ce démarrage part de
    liens vierges, donc d'une revue qui n'hérite de rien.

    Idempotent (ne touche que les liens sans statut) et sans effet sur une base neuve."""
    with engine.begin() as conn:
        if not conn.execute(text("SELECT to_regclass('public.article_scenarios')")).scalar():
            return
        n = conn.execute(text("""
            UPDATE article_scenarios ars
            SET screening_status = d.screening_status,
                screening_reason = COALESCE(ars.screening_reason, d.screening_reason)
            FROM literature_document d
            WHERE d.id = ars.document_id
              AND ars.screening_status IS NULL
              AND d.screening_status IN ('included', 'excluded')
        """)).rowcount or 0
    if n:
        logger.warning(
            f"Screening par revue : {n} décision(s) recopiée(s) depuis la ligne globale "
            f"du document vers les liens. Les lectures ne regardent plus que le lien, "
            f"donc une décision prise dans une revue ne déplace plus le corpus d'une autre.")
    else:
        logger.info("Screening par revue : aucune décision globale à recopier.")


# L'APPEL est plus bas, après `_ensure_bibliographic_columns`, qui est la fonction qui
# CRÉE `literature_document.screening_status` (absente de schema.sql). Appelé ici, sur
# une base neuve, il échouait sur « column d.screening_status does not exist » et n'était
# retenté qu'au démarrage suivant.


def _ensure_dedup_columns():
    """Colonne title_norm (titre normalisé) pour la déduplication inter-sources par
    TITRE - en complément du DOI, afin de capter les doublons SANS DOI (préprints,
    essais) ou dont le DOI diffère d'une source à l'autre. L'ALTER est instantané ; le
    backfill des lignes existantes + l'index se font en arrière-plan
    (_ensure_performance_indexes)."""
    with engine.begin() as conn:
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS title_norm TEXT"))
        # Colonnes de déduplication historiquement posées par des scripts ad-hoc
        # (scripts/archive/*.sql) - garanties ici pour que TOUTE base (prod comme base
        # reconstruite/CI) porte : `pmid` (renseignée à l'ingest → dédup PMID), et
        # `is_duplicate`/`canonical_id` (marquage + statut de dédup, filtrés par ~20
        # requêtes existantes). ADD COLUMN IF NOT EXISTS : no-op si déjà présentes.
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pmid TEXT"))
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS is_duplicate BOOLEAN DEFAULT FALSE"))
        conn.execute(text(
            "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS canonical_id BIGINT"))
    logger.info("Colonnes de déduplication (title_norm, pmid, is_duplicate, canonical_id) vérifiées/créées.")


try:
    _ensure_dedup_columns()
except Exception as _e:
    logger.warning(f"_ensure_dedup_columns: {_e}")


def _ensure_bibliographic_columns():
    """Colonnes bibliographiques / d'affichage historiquement posées par un script ad-hoc
    (scripts/archive/migrate_add_bibliographic_columns.sql) - donc ABSENTES d'une base
    reconstruite depuis schema.sql seul (CI incluse). On les garantit ici pour que le
    détail document enrichi (auteurs / revue / DOI / pays / devis / type d'article) ET les
    requêtes corpus existantes qui les lisent fonctionnent partout. ADD COLUMN IF NOT
    EXISTS : no-op si déjà présentes."""
    stmts = [
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS authors TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS journal TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS doi TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS country TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS study_design TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS publication_type TEXT",
        # ── Colonnes de screening / qualité / enrichissement ────────────────────
        # Même situation que ci-dessus : lues massivement par le code (screening_status
        # ≈95 références, pico_json ≈24) mais absentes de schema.sql, donc introuvables
        # sur toute base neuve. `screening_reason`/`notes` sont, elles, lues par la
        # migration c8d4e2f1a9b3 qui recopie le verdict global vers article_scenarios.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_status TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_reason TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screening_notes TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS screened_at TIMESTAMP",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pico_json JSONB",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS metadata_json JSONB",
        # Concepts typés normalisés par le LLM (carte des concepts) - une fois par article.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS concepts_json JSONB",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS citation_count INTEGER",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS quality_score DOUBLE PRECISION",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS keywords TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS language TEXT",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS sample_size INTEGER",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS open_access BOOLEAN",
        # Lues par de vraies requêtes SQL (SELECT id, title, abstract, has_fulltext… ;
        # UPDATE … SET pico_extracted_at = :ts) et créées nulle part non plus.
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS has_fulltext BOOLEAN DEFAULT FALSE",
        "ALTER TABLE literature_document ADD COLUMN IF NOT EXISTS pico_extracted_at TIMESTAMP",
    ]
    # Une instruction par transaction : une seule qui échoue ne doit pas faire annuler
    # les précédentes (cf. _exec_ddl_isolated).
    _exec_ddl_isolated(stmts, "_ensure_bibliographic_columns")
    logger.info("Colonnes bibliographiques, de screening et de qualité vérifiées/créées.")


try:
    _ensure_bibliographic_columns()
except Exception as _e:
    logger.warning(f"_ensure_bibliographic_columns: {_e}")


# Après la création de `literature_document.screening_status` : la recopie des décisions
# qui ne vivaient que sur la ligne globale (cf. sa docstring) a besoin de cette colonne.
try:
    _backfill_ars_screening_from_document()
except Exception as _e:
    logger.warning(f"_backfill_ars_screening_from_document: {_e}")
