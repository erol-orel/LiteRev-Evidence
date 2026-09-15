"""UMAP/HDBSCAN clustering of a scenario corpus and its per-language summaries.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import os
from typing import Any

from fastapi import Query
from sqlalchemy import text

from .core import _norm_lang, app, engine, logger
from .documents import _llm_lang_directive
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404
from .gesica import _gesica_title, _get_db_gesica_scenario_or_404, _get_scenario_name

# ── Encoder JSON pour types numpy ────────────────────────────────────────────
class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        import numpy as np
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# ── Tâches de clustering en cours ────────────────────────────────────────────
_clustering_jobs: dict[str, dict] = {}  # scenario_id -> {"status": "running"|"done"|"error", "result": ...}


def _cluster_core(
    docs: list,
    texts: list[str],
    *,
    openai_key: str | None = None,
    allow_openai_embeddings: bool = False,
    tfidf_min_df: int = 2,
) -> dict:
    """Cœur partagé du clustering (utilisé par l'endpoint à la demande ET le pipeline).

    Chaîne : embeddings (pgvector DB → OpenAI optionnel → repli TF-IDF) → UMAP 2D
    (thread, timeout 60 s) → HDBSCAN, avec repli K-Means+SVD si UMAP/HDBSCAN échoue.
    Retourne labels, projection 2D, méthode, et les artefacts TF-IDF nécessaires à la
    construction des clusters (feature_names, matrice dense).
    """
    import numpy as np
    import threading
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.cluster import KMeans
    from sklearn.decomposition import TruncatedSVD, PCA

    vectorizer = TfidfVectorizer(max_features=800, stop_words="english",
                                 min_df=tfidf_min_df, max_df=0.9, ngram_range=(1, 2))
    X_tfidf = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    # 1) Embeddings pgvector stockés en DB (privilégiés)
    embeddings_matrix = None
    embedding_source = "tfidf"
    if any(d.get("embedding_str") for d in docs):
        try:
            # Parsing numpy en float32 (4 octets/valeur) : la liste de floats Python
            # (~24 octets/valeur, 1536 par doc) pesait ~900 Mo pour 25 000 docs.
            vecs = []
            for d in docs:
                es = d.get("embedding_str")
                vecs.append(np.fromstring(es.strip("[]"), sep=",", dtype=np.float32) if es else None)
            valid = [v for v in vecs if v is not None and v.size]
            if valid:
                mean_vec = np.mean(np.stack(valid), axis=0)
                vecs = [v if (v is not None and v.size == mean_vec.size) else mean_vec for v in vecs]
                embeddings_matrix = np.stack(vecs).astype(np.float32, copy=False)
                embedding_source = "db_pgvector"
        except Exception as e:
            logger.warning(f"_cluster_core: embeddings DB inutilisables: {e}")

    # 2) Sinon, génération OpenAI (optionnelle)
    if embeddings_matrix is None and allow_openai_embeddings and openai_key:
        try:
            from llm_usage import MeteredOpenAI as _OAI
            _oai = _OAI(api_key=openai_key, timeout=90.0)
            all_vecs: list = []
            batch_texts = [t[:2000] for t in texts]
            for i in range(0, len(batch_texts), 100):
                resp = _oai.embeddings.create(model="text-embedding-3-small",
                                              input=batch_texts[i:i + 100])
                all_vecs.extend([e.embedding for e in resp.data])
            embeddings_matrix = np.array(all_vecs, dtype=np.float32)
            embedding_source = "openai_api"
        except Exception as e:
            logger.warning(f"_cluster_core: embeddings OpenAI échoués: {e}")

    umap_input = embeddings_matrix if embeddings_matrix is not None else X_tfidf.toarray()

    # 3) UMAP 2D dans un thread avec timeout 60 s
    umap_result: dict = {"embedding": None}
    def _run_umap():
        try:
            import umap as umap_lib
            reducer = umap_lib.UMAP(
                n_neighbors=min(10, len(docs) - 1), n_components=2,
                metric="cosine", random_state=42, low_memory=True, n_epochs=200,
            )
            umap_result["embedding"] = reducer.fit_transform(umap_input)
        except Exception as e:
            logger.warning(f"_cluster_core UMAP: {e}")
    _t = threading.Thread(target=_run_umap, daemon=True)
    _t.start()
    _t.join(timeout=60)

    embedding_2d = umap_result["embedding"]
    labels = None
    method_used = "kmeans_fallback"

    # 4) HDBSCAN sur la projection 2D
    if embedding_2d is not None:
        try:
            import hdbscan as hdbscan_lib
            clusterer = hdbscan_lib.HDBSCAN(
                min_cluster_size=max(3, len(docs) // 15), min_samples=2,
                metric="euclidean", cluster_selection_method="eom",
            )
            labels = clusterer.fit_predict(embedding_2d)
            method_used = "embeddings_umap_hdbscan" if embeddings_matrix is not None else "tfidf_umap_hdbscan"
        except Exception as e:
            logger.warning(f"_cluster_core HDBSCAN: {e}")

    # 5) Repli K-Means + SVD (si UMAP a expiré ou HDBSCAN a échoué)
    if labels is None:
        n_clusters = max(3, min(8, len(docs) // 15))
        svd = TruncatedSVD(n_components=min(50, X_tfidf.shape[1] - 1, len(docs) - 1), random_state=42)
        X_reduced = svd.fit_transform(X_tfidf)
        labels = KMeans(n_clusters=n_clusters, random_state=42, n_init=5, max_iter=100).fit_predict(X_reduced)
        if embedding_2d is None:
            embedding_2d = PCA(n_components=2, random_state=42).fit_transform(X_reduced)
        method_used = "kmeans_fallback"

    return {
        "labels": labels,
        "embedding_2d": embedding_2d,
        "method": method_used,
        "feature_names": feature_names,
        # Matrice TF-IDF gardée CREUSE (la version dense pesait 160 Mo pour
        # 25 000 docs × 800 termes) ; les moyennes par cluster se font dessus.
        "X_dense": X_tfidf,
        "embedding_source": embedding_source,
    }


CLUSTER_MAX_DOCS = max(200, int(os.getenv("CLUSTER_MAX_DOCS", "3000")))


def _clustering_docs(scenario_id: str, threshold: float, cap: int | None = None) -> tuple[list, int]:
    """Documents à clusteriser pour un scénario : le sous-ensemble PERTINENT (≥ seuil
    sémantique OU inclus manuellement ; jamais les exclus), plafonné aux `cap` plus
    pertinents (inclus d'abord, puis score décroissant), avec le vecteur du résumé.
    Renvoie (docs, nombre total de documents éligibles).

    Sans plafond, 25 000 documents = 25 000 embeddings à parser puis UMAP sur une
    matrice 25 000 × 1536 : 88 s et un pic de 3 Go de RAM — de quoi faire tuer le
    processus API sur le serveur. Les clusters sont visuellement identiques sur les
    3 000 articles les plus pertinents."""
    cap = CLUSTER_MAX_DOCS if cap is None else max(5, int(cap))
    _relevant = """
        FROM literature_document d
        JOIN article_scenarios asn ON asn.document_id = d.id
        WHERE asn.scenario_id = :sid
          AND d.project_context = 'literev'
          AND (d.is_duplicate IS NULL OR d.is_duplicate = FALSE)
          AND d.abstract IS NOT NULL
          AND LENGTH(d.abstract) > 50
          AND COALESCE(asn.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
          AND (COALESCE(asn.screening_status, d.screening_status) = 'included'
               OR COALESCE(asn.similarity_score, 0) >= :thr)
    """
    with engine.connect() as conn:
        n_total = int(conn.execute(text(f"SELECT COUNT(*) {_relevant}"),
                                   {"sid": scenario_id, "thr": threshold}).scalar() or 0)
        docs = list(conn.execute(text(f"""
            SELECT d.id, d.title, d.abstract, d.year, d.journal,
                   (
                       SELECT c.embedding::text
                       FROM document_chunk c
                       WHERE c.document_id = d.id
                         AND c.embedding IS NOT NULL
                       -- Vecteur représentatif = le résumé (title_abstract) en
                       -- priorité pour TOUS les docs (cohérent) ; repli 1er chunk.
                       ORDER BY (c.chunk_type = 'title_abstract') DESC, c.id
                       LIMIT 1
                   ) AS embedding_str
            {_relevant}
            ORDER BY (COALESCE(asn.screening_status, d.screening_status) = 'included') DESC,
                     asn.similarity_score DESC NULLS LAST, d.year DESC NULLS LAST, d.id
            LIMIT :cap
        """), {"sid": scenario_id, "thr": threshold, "cap": cap}).mappings().all())
    return docs, n_total


# ── Caches de visualisation persistés en DB (scenario_settings) ───────────────
# Un SEUL couple load/save par visualisation, partagé par le pipeline, le
# précalcul et les endpoints — plus de duplication ni de cache /tmp éphémère.

def _save_viz_cache(scenario_id: str, col: str, payload: dict) -> None:
    """Upsert un JSON de visualisation dans scenario_settings.{col}_json (+ _at)."""
    _at = "clustering_generated_at" if col == "clustering" else "kg_generated_at"
    _jc = f"{col}_json" if col == "clustering" else "knowledge_graph_json"
    try:
        with engine.begin() as _c:
            _c.execute(text(f"""
                INSERT INTO scenario_settings (scenario_id, {_jc}, {_at}, updated_at)
                VALUES (:sid, CAST(:p AS jsonb), NOW(), NOW())
                ON CONFLICT (scenario_id) DO UPDATE
                SET {_jc} = CAST(:p AS jsonb), {_at} = NOW(), updated_at = NOW()
            """), {"sid": scenario_id, "p": json.dumps(payload, default=str)})
    except Exception as _e:
        logger.warning(f"_save_viz_cache {col} {scenario_id}: {_e}")


def _load_viz_cache(scenario_id: str, col: str, ttl: int = 86400) -> dict | None:
    """Lit le JSON de visualisation en cache s'il est frais (< ttl secondes)."""
    _at = "clustering_generated_at" if col == "clustering" else "kg_generated_at"
    _jc = f"{col}_json" if col == "clustering" else "knowledge_graph_json"
    try:
        with engine.connect() as _c:
            row = _c.execute(text(
                f"SELECT {_jc} AS j, {_at} AS at FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
        if row and row["j"]:
            fresh = True
            if row["at"]:
                from datetime import datetime as _dt, timezone as _tz
                _ts = row["at"]
                if _ts.tzinfo is None:
                    _ts = _ts.replace(tzinfo=_tz.utc)
                fresh = (_dt.now(_tz.utc) - _ts).total_seconds() < ttl
            if fresh:
                data = dict(row["j"])
                data["from_cache"] = True
                return data
    except Exception as _e:
        logger.warning(f"_load_viz_cache {col} {scenario_id}: {_e}")
    return None


_CLUSTER_NOISE_TEXT = {
    "fr": ("Non-classés", "Bruit de fond (articles non regroupés)."),
    "en": ("Unclassified", "Background noise (articles not grouped)."),
}
_CLUSTER_MESSAGES = {
    "insufficient_corpus": {
        "fr": "Corpus insuffisant pour le clustering (minimum 5 articles avec abstract requis)",
        "en": "Not enough articles for clustering (at least 5 articles with an abstract are required)",
    },
    "running": {
        "fr": "Calcul en cours. Revenez dans 30-60s.",
        "en": "Computing. Check back in 30-60 s.",
    },
}


def _cluster_summary_llm(client, title: str | None, docs: list, lang: str | None) -> str:
    """Résumé LLM d'UN cluster à partir de ses articles représentatifs (titre+résumé),
    dans la langue demandée. Partagé par le clustering complet et la (re)génération
    des résumés seuls quand la langue du cache ne correspond pas à celle demandée."""
    english = _norm_lang(lang) == "en"
    lang_word = "in English" if english else "en français"
    llm_ctx = "\n\n".join(
        f"Titre: {d.get('title') or ''}\nRésumé: {(d.get('abstract') or '')[:350]}"
        for d in docs
    )
    completion = client.chat.completions.create(
        model="gpt-4.1-mini",
        messages=[{"role": "user", "content": (
            f"Scénario : {title or 'scénario'}.\n"
            f"Articles représentatifs du cluster :\n{llm_ctx}\n\n"
            f"Rédigez un résumé concis (3-4 phrases, max 120 mots) {lang_word} : "
            f"thématique commune, évidences clés, valeur opérationnelle pour la pratique clinique et la santé publique."
        ) + _llm_lang_directive(lang)}],
        max_tokens=200, temperature=0.3,
    )
    return (completion.choices[0].message.content or "").strip()


def _clusters_have_lang(payload: dict | None, lang: str) -> bool:
    """True si CHAQUE cluster dense du payload porte un résumé dans `lang`
    (clusters[].summaries[lang]). Un cache d'avant le suivi de la langue (résumé
    unique `summary`, langue inconnue) ou un cache de pipeline (sans résumés) → False :
    les résumés sont alors (re)générés dans la langue demandée, sans re-clusteriser."""
    if not isinstance(payload, dict):
        return False
    dense = [c for c in (payload.get("clusters") or []) if isinstance(c, dict) and not c.get("is_noise")]
    return all(lang in (c.get("summaries") or {}) for c in dense)


CLUSTER_MAX_POINTS = max(500, int(os.getenv("CLUSTER_MAX_POINTS", "4000")))


def _sample_points(points: list, keep: int) -> list:
    """Sous-échantillon DÉTERMINISTE de `keep` points, à pas régulier (les points sont
    dans l'ordre du corpus, donc l'échantillon couvre tout le nuage)."""
    n = len(points)
    if keep >= n or keep <= 0:
        return list(points)
    step = n / float(keep)
    return [points[int(i * step)] for i in range(keep)]


def _localize_clusters_payload(payload: dict, lang: str | None, max_points: int | None = None) -> dict:
    """Vue du payload de clustering dans la langue demandée : `summary` de chaque
    cluster = son résumé dans cette langue (repli : le résumé existant), libellés du
    cluster « bruit » et message localisés. Ne modifie pas l'objet en cache.

    Les points de la projection UMAP sont plafonnés à `max_points` au total (répartis
    au prorata des clusters, ≥ 5 par cluster) : 25 000 points = 2,5 Mo de JSON et
    25 000 cercles SVG dans le navigateur, pour un nuage visuellement identique à
    4 000 points. Le cache garde tous les points (`points_total` par cluster)."""
    want = _norm_lang(lang) or "fr"
    noise_name, noise_summary = _CLUSTER_NOISE_TEXT[want]
    cap = CLUSTER_MAX_POINTS if max_points is None else max(0, int(max_points))
    out = dict(payload)
    clusters = []
    total_points = sum(len(c.get("points") or []) for c in (out.get("clusters") or []) if isinstance(c, dict))
    ratio = (cap / float(total_points)) if (cap and total_points > cap) else 1.0
    shown = 0
    for c in out.get("clusters") or []:
        if not isinstance(c, dict):
            continue
        cc = dict(c)
        if cc.get("is_noise"):
            cc["cluster_name"] = noise_name
            cc["summary"] = noise_summary
        else:
            summaries = cc.get("summaries") or {}
            if want in summaries:
                cc["summary"] = summaries[want]
        pts = cc.get("points") or []
        cc["points_total"] = len(pts)
        if ratio < 1.0 and len(pts) > 50:            # small clusters are served whole
            cc["points"] = _sample_points(pts, max(50, int(len(pts) * ratio)))
        shown += len(cc.get("points") or [])
        clusters.append(cc)
    out["clusters"] = clusters
    out["points_total"] = total_points
    out["points_shown"] = shown
    code = out.get("message_code")
    if code in _CLUSTER_MESSAGES:
        out["message"] = _CLUSTER_MESSAGES[code][want]
    out["lang"] = want
    return out


def _clustering_running_payload(scenario_id: str, lang: str | None) -> dict:
    want = _norm_lang(lang) or "fr"
    return {"scenario_id": scenario_id, "status": "running", "message_code": "running",
            "message": _CLUSTER_MESSAGES["running"][want], "clusters": [], "lang": want}


def _summarize_clusters_in_lang(scenario_id: str, payload: dict, lang: str) -> dict:
    """(Re)génère les RÉSUMÉS des clusters dans `lang` SANS recalculer le clustering
    (embeddings/UMAP/HDBSCAN conservés) : les 5 articles les plus proches du centre de
    chaque cluster sont relus en base et résumés par le LLM. Renvoie une COPIE du
    payload avec clusters[].summaries[lang] (+ `summary` dans cette langue)."""
    from concurrent.futures import ThreadPoolExecutor as _TPE
    want = _norm_lang(lang) or "fr"
    out = json.loads(json.dumps(payload, cls=_NumpyEncoder, default=str))
    dense = [c for c in (out.get("clusters") or []) if isinstance(c, dict) and not c.get("is_noise")]
    picks: dict[int, list[int]] = {}
    need: set[int] = set()
    for c in dense:
        cx, cy = float(c.get("center_x") or 0.0), float(c.get("center_y") or 0.0)
        pts = [p for p in (c.get("points") or []) if isinstance(p, dict) and p.get("id") is not None]
        pts.sort(key=lambda p: (float(p.get("x") or 0.0) - cx) ** 2 + (float(p.get("y") or 0.0) - cy) ** 2)
        ids = [int(p["id"]) for p in pts[:5]]
        rep_id = (c.get("representative_doc") or {}).get("id")
        if not ids and rep_id is not None:
            ids = [int(rep_id)]
        picks[int(c["cluster_id"])] = ids
        need.update(ids)
    rows: dict[int, dict] = {}
    if need:
        with engine.connect() as conn:
            for r in conn.execute(text(
                "SELECT id, title, abstract FROM literature_document WHERE id = ANY(CAST(:ids AS bigint[]))"
            ), {"ids": sorted(need)}).mappings():
                rows[int(r["id"])] = dict(r)
    try:
        title = _get_scenario_name(scenario_id)
    except Exception:
        title = scenario_id
    openai_key = os.getenv("OPENAI_API_KEY")
    client = None
    if openai_key:
        from llm_usage import MeteredOpenAI as _OAI
        client = _OAI(api_key=openai_key, timeout=90.0)

    def _one(c: dict) -> str:
        docs = [rows[i] for i in picks.get(int(c["cluster_id"]), []) if i in rows]
        if client is None or not docs:
            return ""
        try:
            return _cluster_summary_llm(client, title, docs, want)
        except Exception as _e:
            logger.error(f"Résumé cluster {c.get('cluster_id')} ({want}) {scenario_id}: {_e}")
            return ""

    if dense:
        with _TPE(max_workers=4) as ex:
            texts = list(ex.map(_one, dense))
    else:
        texts = []
    for c, s in zip(dense, texts):
        summaries = dict(c.get("summaries") or {})
        summaries[want] = s
        c["summaries"] = summaries
        c["summary"] = s
    out["lang"] = want
    out["from_cache"] = False
    return out


def _persist_clustering_result(scenario_id: str, result: dict) -> None:
    """Cache DB (durable) + /tmp (compat) d'un payload de clustering."""
    _save_viz_cache(scenario_id, "clustering", json.loads(json.dumps(result, cls=_NumpyEncoder, default=str)))
    try:
        cache_dir = "/tmp/literev_clustering_cache"
        os.makedirs(cache_dir, exist_ok=True)
        with open(os.path.join(cache_dir, f"{scenario_id}.json"), "w") as f:
            json.dump(result, f, cls=_NumpyEncoder, default=str)
    except Exception:
        pass


def _relocalize_clustering_background(scenario_id: str, payload: dict, lang: str) -> None:
    """Thread : résumés des clusters dans la langue demandée (structure conservée),
    puis mise en cache — la page interroge /clustering/status jusqu'à « done »."""
    try:
        result = _summarize_clusters_in_lang(scenario_id, payload, lang)
        _persist_clustering_result(scenario_id, result)
        _clustering_jobs[scenario_id] = {"status": "done", "result": result}
    except Exception as e:
        logger.error(f"Clustering {scenario_id} résumés {lang}: {e}", exc_info=True)
        _clustering_jobs[scenario_id] = {"status": "error", "error": str(e)}


def _build_clusters_payload(scenario_id: str, docs: list, cc: dict, *,
                            with_summaries: bool = False, openai_key: str | None = None,
                            title: str | None = None, lang: str | None = None,
                            n_docs_total: int | None = None) -> dict:
    """Construit le payload de clustering CANONIQUE (un seul format, partagé par le
    pipeline ET le calcul en arrière-plan). `with_summaries` active le résumé LLM
    par cluster, rangé par LANGUE dans clusters[].summaries (`summary` = celui de la
    langue demandée) pour que le cache serve chaque langue sans mélange.
    Schéma figé : clusters[].representative_doc + embedding_source."""
    import numpy as np
    labels = cc["labels"]; embedding_2d = cc["embedding_2d"]; method_used = cc["method"]
    feature_names = cc["feature_names"]; X_dense = cc["X_dense"]; embedding_source = cc["embedding_source"]
    want = _norm_lang(lang) or "fr"
    noise_name, noise_summary = _CLUSTER_NOISE_TEXT[want]
    _client = None
    if with_summaries and openai_key:
        from llm_usage import MeteredOpenAI as _OAI
        _client = _OAI(api_key=openai_key, timeout=90.0)
    clusters = []
    for label in sorted(set(labels)):
        label_int = int(label)
        idxs = [i for i, l in enumerate(labels) if int(l) == label_int]
        coords = embedding_2d[idxs]
        cluster_tfidf = np.asarray(X_dense[idxs].mean(axis=0)).ravel()   # dense ou creuse
        top_indices = cluster_tfidf.argsort()[-10:][::-1]
        top_words = [str(feature_names[i]) for i in top_indices if cluster_tfidf[i] > 0]
        center = np.mean(coords, axis=0)
        distances = np.linalg.norm(coords - center, axis=1)
        rep = docs[idxs[int(np.argmin(distances))]]
        points = [
            {"id": int(docs[i]["id"]), "title": str(docs[i]["title"] or ""),
             "year": int(docs[i]["year"]) if docs[i].get("year") else None,
             "x": float(embedding_2d[i, 0]), "y": float(embedding_2d[i, 1])}
            for i in idxs
        ]
        resume = noise_summary if label_int == -1 else ""
        summaries: dict[str, str] = {}
        if _client is not None and label_int != -1:
            try:
                top5 = np.argsort(distances)[:5]
                resume = _cluster_summary_llm(
                    _client, title or scenario_id, [docs[idxs[int(t)]] for t in top5], want)
            except Exception as _e:
                logger.error(f"Résumé cluster {label_int}: {_e}")
            summaries[want] = resume
        clusters.append({
            "cluster_id": label_int,
            "cluster_name": f"Cluster {label_int + 1}" if label_int != -1 else noise_name,
            "is_noise": label_int == -1,
            "n_docs": len(idxs),
            "center_x": float(center[0]), "center_y": float(center[1]),
            "top_words": top_words,
            "summary": resume,
            "summaries": summaries,
            "representative_doc": {
                "id": int(rep["id"]),
                "title": str(rep["title"] or ""),
                "year": int(rep["year"]) if rep.get("year") else None,
                "journal": str(rep.get("journal") or ""),
            },
            "points": points,
        })
    return {
        "scenario_id": scenario_id,
        "n_docs": len(docs),
        # Documents éligibles au total ; > n_docs quand le clustering a été plafonné
        # aux CLUSTER_MAX_DOCS plus pertinents (l'interface l'indique).
        "n_docs_total": int(n_docs_total) if n_docs_total is not None else len(docs),
        "n_clusters": len([c for c in clusters if not c["is_noise"]]),
        "method": method_used,
        "embedding_source": embedding_source,
        "clusters": sorted(clusters, key=lambda x: (x["is_noise"], -x["n_docs"])),
        "lang": want if with_summaries else None,
        "from_cache": False,
    }


def _run_clustering_background(scenario_id: str, force_refresh: bool = False, lang: str | None = None) -> None:
    """Calcule le clustering dans un thread séparé et stocke le résultat en cache."""
    import time as _time

    cache_dir = "/tmp/literev_clustering_cache"
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{scenario_id}.json")
    TTL = 86400
    want = _norm_lang(lang) or "fr"

    # Vérifier le cache d'abord — dans la LANGUE demandée : un cache frais dont les
    # résumés sont dans l'autre langue (ou sans résumés) garde sa structure, seuls les
    # résumés sont régénérés. Avant, le cache était servi tel quel → résumés en
    # français sous le toggle anglais.
    if not force_refresh and os.path.exists(cache_file):
        try:
            mtime = os.path.getmtime(cache_file)
            if _time.time() - mtime < TTL:
                with open(cache_file, "r") as f:
                    cached = json.load(f)
                if _clusters_have_lang(cached, want) or not cached.get("clusters") or not os.getenv("OPENAI_API_KEY"):
                    cached["from_cache"] = True
                    _clustering_jobs[scenario_id] = {"status": "done", "result": cached}
                else:
                    _relocalize_clustering_background(scenario_id, cached, want)
                return
        except Exception:
            pass

    try:
        meta_for_cluster = {}
        try:
            meta_for_cluster = _get_db_gesica_scenario_or_404(scenario_id)
        except Exception:
            pass  # Scénario utilisateur ou non trouvé — on continue sans métadonnées
    except Exception:
        meta_for_cluster = {}
    try:

        # Clustering sur le SOUS-ENSEMBLE PERTINENT (≥ seuil sémantique OU inclus
        # manuellement ; jamais les exclus) — comme le knowledge graph et l'Assistant
        # RAG. Sinon les topics étaient dilués par les centaines d'articles hors-sujet
        # ramenés par la fédération.
        _thr = _get_scenario_threshold(scenario_id)
        docs, _n_total = _clustering_docs(scenario_id, _thr)

        if len(docs) < 5:
            result = {
                "scenario_id": scenario_id, "n_docs": len(docs),
                "message_code": "insufficient_corpus",
                "message": _CLUSTER_MESSAGES["insufficient_corpus"][want],
                "clusters": [], "from_cache": False, "lang": want,
            }
            _clustering_jobs[scenario_id] = {"status": "done", "result": result}
            return

        texts = [f"{d['title']} {d['abstract'] or ''}" for d in docs]

        # ── Embeddings → UMAP → HDBSCAN (cœur partagé _cluster_core) ────────
        openai_key = os.getenv("OPENAI_API_KEY")
        _cc = _cluster_core(docs, texts, openai_key=openai_key,
                            allow_openai_embeddings=True, tfidf_min_df=2)
        labels = _cc["labels"]
        embedding_2d = _cc["embedding_2d"]
        method_used = _cc["method"]
        feature_names = _cc["feature_names"]
        X_dense = _cc["X_dense"]
        embedding_source = _cc["embedding_source"]
        logger.info(f"Clustering {scenario_id}: {len(docs)} docs, source={embedding_source}, method={method_used}")

        # Construction du payload (helper PARTAGÉ avec le pipeline — plus de copie).
        result = _build_clusters_payload(
            scenario_id, docs, _cc, with_summaries=True, openai_key=openai_key,
            title=(_gesica_title(meta_for_cluster) if meta_for_cluster else None),
            lang=want, n_docs_total=_n_total,
        )
        # Cache DB (durable) + /tmp (compat) + mémoire.
        _persist_clustering_result(scenario_id, result)
        _clustering_jobs[scenario_id] = {"status": "done", "result": result}

    except Exception as e:
        logger.error(f"Clustering {scenario_id} error: {e}", exc_info=True)
        _clustering_jobs[scenario_id] = {"status": "error", "error": str(e)}


@app.get("/user-scenarios/{scenario_id}/clustering")
def get_user_scenario_clustering(scenario_id: str, force_refresh: bool = False, lang: str | None = Query(None)) -> dict[str, Any]:
    """Clustering pour un scénario utilisateur, dans la langue demandée (`lang`).

    Le cache (DB, puis job en mémoire) n'est servi tel quel que s'il porte les résumés
    dans CETTE langue. Sinon la structure (embeddings/UMAP/HDBSCAN) est conservée et
    seuls les résumés sont régénérés en arrière-plan → réponse « running », la page
    interroge /clustering/status. Auparavant le cache était renvoyé quelle que soit la
    langue : résumés en français sous le toggle anglais."""
    import threading
    _get_user_scenario_or_404(scenario_id)
    want = _norm_lang(lang) or "fr"
    if not force_refresh:
        cached = _load_viz_cache(scenario_id, "clustering")
        if not cached:
            job = _clustering_jobs.get(scenario_id)
            if job and job.get("status") == "done" and isinstance(job.get("result"), dict):
                cached = job["result"]
        if cached:
            if (_clusters_have_lang(cached, want) or not cached.get("clusters")
                    or not os.getenv("OPENAI_API_KEY")):
                return _localize_clusters_payload(cached, want)
            job = _clustering_jobs.get(scenario_id)
            if not job or job.get("status") != "running":
                _clustering_jobs[scenario_id] = {"status": "running"}
                threading.Thread(target=_relocalize_clustering_background,
                                 args=(scenario_id, cached, want), daemon=True).start()
            return _clustering_running_payload(scenario_id, want)
    job = _clustering_jobs.get(scenario_id)
    if not job or job.get("status") not in ("running",) or force_refresh:
        _clustering_jobs[scenario_id] = {"status": "running"}
        t = threading.Thread(target=_run_clustering_background, args=(scenario_id, force_refresh, want), daemon=True)
        t.start()
    return _clustering_running_payload(scenario_id, want)


@app.get("/user-scenarios/{scenario_id}/clustering/status")
def get_user_scenario_clustering_status(scenario_id: str, lang: str | None = Query(None)) -> dict:
    """Statut du clustering pour un scénario utilisateur (résultat dans la langue demandée)."""
    _get_user_scenario_or_404(scenario_id)
    want = _norm_lang(lang) or "fr"
    job = _clustering_jobs.get(scenario_id)
    if not job:
        _db = _load_viz_cache(scenario_id, "clustering")
        if _db:
            return _localize_clusters_payload(_db, want)
        return {"scenario_id": scenario_id, "status": "not_started",
                "message": "Aucun calcul lancé." if want == "fr" else "No computation started."}
    if job["status"] == "running":
        return {"scenario_id": scenario_id, "status": "running",
                "message": _CLUSTER_MESSAGES["running"][want]}
    if job["status"] == "error":
        return {"scenario_id": scenario_id, "status": "error", "error": job.get("error", "Erreur inconnue")}
    return _localize_clusters_payload(job["result"], want)
