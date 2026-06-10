#!/usr/bin/env python3
"""
Remplace le bloc populate de main.py par :
1. _run_user_scenario_populate sans limite (max_results=500, pagination NCBI)
2. _run_user_scenario_full_pipeline (PubMed → PICO → métadonnées → fulltext → clustering)
3. Endpoints /pipeline et /pipeline/status
4. Correction pico/extract + metadata/extract pour article_scenarios (user scenarios)
"""

with open("main.py", "r") as f:
    content = f.read()

# ── 1. Remplacer _run_user_scenario_populate + endpoint populate ──────────────
OLD_POPULATE = '''# ── Populate : ingestion PubMed en arrière-plan ───────────────────────────────

_user_scenario_populate_jobs: dict[str, dict] = {}


def _run_user_scenario_populate(scenario_id: str, query: str, filters: dict, max_results: int = 30) -> None:
    """
    Ingère des articles PubMed pour un scénario utilisateur en arrière-plan.
    Assigne chaque article ingéré à article_scenarios avec le scenario_id utilisateur.
    """
    import time as _time
    import xml.etree.ElementTree as ET
    import requests as _requests

    _user_scenario_populate_jobs[scenario_id] = {"status": "running", "ingested": 0, "errors": 0}

    ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    EMAIL = os.getenv("PUBMED_EMAIL", "literev@example.com")
    WRITE_KEY = os.getenv("WRITE_API_KEY", "")
    HEADERS_LOCAL = {"X-Api-Key": WRITE_KEY}
    API_LOCAL = "http://127.0.0.1:8000"

    try:
        # 1. Recherche PubMed
        r = _requests.get(
            f"{ENTREZ_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": max_results,
                "retmode": "json",
                "email": EMAIL,
            },
            timeout=30,
        )
        r.raise_for_status()
        pmids = r.json()["esearchresult"]["idlist"]'''

assert OLD_POPULATE in content, "❌ Bloc OLD_POPULATE non trouvé"
print("✅ Bloc populate trouvé")

# Trouver la fin du bloc populate (jusqu'à @app.post("/user-scenarios/{scenario_id}/populate"))
# et l'endpoint populate lui-même (jusqu'au prochain bloc commentaire)
end_marker = '''# ── Proxy endpoints : rediriger les appels /gesica/scenarios/{usr-*}/... ──────'''
start_idx = content.index(OLD_POPULATE)
end_idx = content.index(end_marker)

old_block = content[start_idx:end_idx]
print(f"Bloc à remplacer: {len(old_block)} caractères")

NEW_BLOCK = '''# ── Populate : ingestion PubMed en arrière-plan ───────────────────────────────

_user_scenario_populate_jobs: dict[str, dict] = {}
_user_scenario_pipeline_jobs: dict[str, dict] = {}


def _run_user_scenario_populate(
    scenario_id: str,
    query: str,
    filters: dict,
    max_results: int = 500,
    _pipeline_callback=None,
) -> int:
    """
    Ingère des articles PubMed pour un scénario utilisateur en arrière-plan.
    Utilise usehistory=y pour paginer sans limite NCBI (retmax max=10000 par page).
    Retourne le nombre d'articles ingérés.
    """
    import time as _time
    import xml.etree.ElementTree as ET
    import requests as _requests
    import math

    if _pipeline_callback is None:
        _user_scenario_populate_jobs[scenario_id] = {"status": "running", "ingested": 0, "errors": 0, "total_found": 0}

    ENTREZ_BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
    EMAIL = os.getenv("PUBMED_EMAIL", "literev@example.com")
    WRITE_KEY = os.getenv("WRITE_API_KEY", "")
    HEADERS_LOCAL = {"X-Api-Key": WRITE_KEY}
    API_LOCAL = "http://127.0.0.1:8000"
    BATCH_SIZE = 200  # efetch max par requête

    try:
        # 1. Recherche PubMed avec usehistory pour paginer
        r = _requests.get(
            f"{ENTREZ_BASE}/esearch.fcgi",
            params={
                "db": "pubmed",
                "term": query,
                "retmax": 0,          # On veut juste le count + WebEnv
                "retmode": "json",
                "usehistory": "y",
                "email": EMAIL,
            },
            timeout=30,
        )
        r.raise_for_status()
        search_result = r.json()["esearchresult"]
        total_found = int(search_result.get("count", 0))
        web_env = search_result.get("webenv", "")
        query_key = search_result.get("querykey", "1")

        effective_max = min(max_results, total_found)
        if _pipeline_callback:
            _pipeline_callback("pubmed_found", total_found)
        else:
            _user_scenario_populate_jobs[scenario_id]["total_found"] = total_found

        if total_found == 0:
            if _pipeline_callback is None:
                _user_scenario_populate_jobs[scenario_id] = {
                    "status": "done", "ingested": 0, "errors": 0, "total_found": 0,
                    "message": "Aucun article trouvé sur PubMed pour cette requête."
                }
            return 0

        ingested = 0
        errors = 0
        n_batches = math.ceil(effective_max / BATCH_SIZE)

        for batch_idx in range(n_batches):
            retstart = batch_idx * BATCH_SIZE
            retmax_batch = min(BATCH_SIZE, effective_max - retstart)
            if retmax_batch <= 0:
                break

            # 2. Fetch XML PubMed (par batch)
            r2 = _requests.post(
                f"{ENTREZ_BASE}/efetch.fcgi",
                data={
                    "db": "pubmed",
                    "WebEnv": web_env,
                    "query_key": query_key,
                    "retstart": retstart,
                    "retmax": retmax_batch,
                    "rettype": "xml",
                    "retmode": "xml",
                    "email": EMAIL,
                },
                timeout=90,
            )
            r2.raise_for_status()
            root = ET.fromstring(r2.content)

            for article_elem in root.findall(".//PubmedArticle"):
                pmid = article_elem.findtext(".//PMID") or ""
                title_elem = article_elem.find(".//ArticleTitle")
                title = "".join(title_elem.itertext()).strip() if title_elem is not None else ""
                abstract_parts = []
                for node in article_elem.findall(".//Abstract/AbstractText"):
                    txt = "".join(node.itertext()).strip()
                    if txt:
                        abstract_parts.append(txt)
                abstract = " ".join(abstract_parts).strip()

                year = None
                year_text = (
                    article_elem.findtext(".//PubDate/Year")
                    or article_elem.findtext(".//ArticleDate/Year")
                    or ""
                )
                if year_text[:4].isdigit():
                    year = int(year_text[:4])

                # Auteurs
                authors_list = []
                for author in article_elem.findall(".//AuthorList/Author"):
                    last = author.findtext("LastName") or ""
                    first = author.findtext("ForeName") or ""
                    if last:
                        authors_list.append(f"{last} {first}".strip())
                authors = "; ".join(authors_list[:6]) if authors_list else None

                # Journal
                journal = article_elem.findtext(".//Journal/Title") or article_elem.findtext(".//ISOAbbreviation") or None

                # DOI
                doi = None
                for id_elem in article_elem.findall(".//ArticleIdList/ArticleId"):
                    if id_elem.get("IdType") == "doi":
                        doi = id_elem.text
                        break

                if not pmid or not title:
                    continue

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                content_text = f"{title}\\n\\n{abstract}".strip()
                if len(content_text) < 30:
                    continue

                try:
                    # Vérifier si l'article existe déjà en DB
                    with engine.connect() as conn:
                        existing = conn.execute(text("""
                            SELECT id FROM literature_document
                            WHERE external_id = :pmid AND project_context = 'literev'
                            LIMIT 1
                        """), {"pmid": pmid}).mappings().first()

                    if existing:
                        doc_id = existing["id"]
                    else:
                        # Ingérer via l'API locale
                        doc_r = _requests.post(
                            f"{API_LOCAL}/documents",
                            headers=HEADERS_LOCAL,
                            json={
                                "source": "pubmed",
                                "title": title,
                                "abstract": abstract or None,
                                "year": year,
                                "url": url,
                                "external_id": pmid,
                                "project_context": "literev",
                                "source_type": "article",
                                "disease_or_condition": None,
                                "scenario_type": scenario_id,
                                "geographic_scope": None,
                                "evidence_category": None,
                                "authors": authors,
                                "journal": journal,
                                "doi": doi,
                            },
                            timeout=30,
                        )
                        doc_r.raise_for_status()
                        doc_id = doc_r.json()["id"]

                        # Créer le chunk
                        _requests.post(
                            f"{API_LOCAL}/chunks",
                            headers=HEADERS_LOCAL,
                            json={
                                "document_id": doc_id,
                                "chunk_index": 0,
                                "content": content_text,
                                "chunk_type": "title_abstract",
                                "section_label": None,
                                "char_start": None,
                                "char_end": None,
                                "token_count": len(content_text.split()),
                                "chunk_weight": 1.0,
                                "metadata_json": {},
                            },
                            timeout=60,
                        )

                    # Assigner l'article au scénario utilisateur dans article_scenarios
                    with engine.begin() as conn:
                        conn.execute(text("""
                            INSERT INTO article_scenarios (document_id, scenario_id, similarity_score)
                            VALUES (:doc_id, :sid, 1.0)
                            ON CONFLICT (document_id, scenario_id) DO NOTHING
                        """), {"doc_id": doc_id, "sid": scenario_id})

                    ingested += 1
                    if _pipeline_callback is None:
                        _user_scenario_populate_jobs[scenario_id]["ingested"] = ingested

                except Exception as e:
                    logger.warning(f"Populate user_scenario {scenario_id} - PMID {pmid}: {e}")
                    errors += 1

                _time.sleep(0.1)

            # Pause entre batches pour respecter les limites NCBI
            if batch_idx < n_batches - 1:
                _time.sleep(0.5)

        # Mettre à jour le compteur dans user_scenarios
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE user_scenarios
                SET result_count = (
                    SELECT COUNT(DISTINCT document_id) FROM article_scenarios WHERE scenario_id = :sid
                ),
                updated_at = NOW()
                WHERE id = :sid
            """), {"sid": scenario_id})

        if _pipeline_callback is None:
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "done",
                "ingested": ingested,
                "errors": errors,
                "total_found": total_found,
                "message": f"{ingested} articles ingérés depuis PubMed ({total_found} trouvés), {errors} erreurs.",
            }
        logger.info(f"Populate user_scenario {scenario_id}: {ingested}/{total_found} articles ingérés.")
        return ingested

    except Exception as e:
        logger.error(f"Populate user_scenario {scenario_id} fatal: {e}", exc_info=True)
        if _pipeline_callback is None:
            _user_scenario_populate_jobs[scenario_id] = {
                "status": "error",
                "error": str(e),
                "ingested": _user_scenario_populate_jobs.get(scenario_id, {}).get("ingested", 0),
            }
        return 0


def _run_user_scenario_full_pipeline(scenario_id: str, query: str, filters: dict, max_results: int = 500) -> None:
    """
    Pipeline complet d'enrichissement pour un scénario utilisateur :
    1. Ingestion PubMed (sans limite)
    2. Extraction PICO (LLM batch)
    3. Extraction métadonnées (LLM batch)
    4. Récupération full-text (Unpaywall)
    5. Clustering thématique
    Chaque étape met à jour _user_scenario_pipeline_jobs[scenario_id].
    """
    import time as _time

    def update_step(step: str, status: str, **kwargs):
        job = _user_scenario_pipeline_jobs.get(scenario_id, {})
        job["current_step"] = step
        job["steps"] = job.get("steps", {})
        job["steps"][step] = {"status": status, **kwargs}
        job["overall_status"] = "running"
        _user_scenario_pipeline_jobs[scenario_id] = job
        logger.info(f"Pipeline {scenario_id} [{step}]: {status} {kwargs}")

    def pubmed_callback(event: str, value):
        if event == "pubmed_found":
            update_step("pubmed", "running", found=value)

    _user_scenario_pipeline_jobs[scenario_id] = {
        "overall_status": "running",
        "current_step": "pubmed",
        "steps": {
            "pubmed": {"status": "pending"},
            "pico": {"status": "pending"},
            "metadata": {"status": "pending"},
            "fulltext": {"status": "pending"},
            "clustering": {"status": "pending"},
        },
    }

    try:
        # ── Étape 1 : Ingestion PubMed ────────────────────────────────────────
        update_step("pubmed", "running")
        ingested = _run_user_scenario_populate(
            scenario_id, query, filters, max_results, _pipeline_callback=pubmed_callback
        )
        update_step("pubmed", "done", ingested=ingested)

        if ingested == 0:
            _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
            _user_scenario_pipeline_jobs[scenario_id]["message"] = "Aucun article trouvé sur PubMed."
            return

        # ── Étape 2 : Extraction PICO ─────────────────────────────────────────
        update_step("pico", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI
                from datetime import datetime, timezone
                _client = _OAI(api_key=openai_key)
                system_prompt_pico = (
                    "You are a systematic review expert in emergency medicine. "
                    "Extract PICO elements and return ONLY valid JSON:\\n"
                    '{"P":"Population","I":"Intervention","C":"Comparator or Not specified",'
                    '"O":"Outcome(s)","study_design":"RCT|Cohort|Systematic review|etc",'
                    '"pico_confidence":0.0-1.0,"pico_notes":""}\\n'
                    "Be concise (max 2 sentences per field). Return ONLY the JSON."
                )
                with engine.connect() as conn:
                    pico_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (ld.pico_json IS NULL OR (ld.pico_json->>'pico_confidence')::float < 0.5)
                          AND ld.abstract IS NOT NULL AND length(ld.abstract) > 50
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                pico_extracted = 0
                pico_errors = 0
                for row in pico_rows:
                    try:
                        response = _client.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_pico},
                                {"role": "user", "content": f"Title: {row['title']}\\n\\nAbstract: {(row['abstract'] or '')[:3000]}"},
                            ],
                            temperature=0.1,
                            max_tokens=400,
                            response_format={"type": "json_object"},
                        )
                        pico = json.loads(response.choices[0].message.content)
                        required = {"P", "I", "C", "O", "study_design", "pico_confidence"}
                        if required.issubset(pico.keys()):
                            pico["pico_confidence"] = float(pico.get("pico_confidence", 0.5))
                            with engine.begin() as conn:
                                conn.execute(text("""
                                    UPDATE literature_document
                                    SET pico_json = CAST(:pico AS jsonb), pico_extracted_at = :ts
                                    WHERE id = :article_id
                                """), {"pico": json.dumps(pico), "ts": datetime.now(timezone.utc), "article_id": row["id"]})
                            pico_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline PICO article {row['id']}: {e}")
                        pico_errors += 1
                    _time.sleep(0.05)
                update_step("pico", "done", extracted=pico_extracted, errors=pico_errors)
            else:
                update_step("pico", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("pico", "error", error=str(e))

        # ── Étape 3 : Extraction métadonnées ─────────────────────────────────
        update_step("metadata", "running")
        try:
            openai_key = os.getenv("OPENAI_API_KEY")
            if openai_key:
                from openai import OpenAI as _OAI2
                from datetime import datetime, timezone
                _client2 = _OAI2(api_key=openai_key)
                system_prompt_meta = (
                    "You are a biomedical librarian. Extract metadata from this article and return ONLY valid JSON:\\n"
                    '{"study_type":"RCT|Cohort|Case-control|Cross-sectional|Systematic review|Meta-analysis|Case report|Editorial|Other",'
                    '"sample_size":null,"country":"ISO2 or null","setting":"hospital|prehospital|community|other|null",'
                    '"primary_outcome":"brief description or null","funding":"public|industry|mixed|not reported",'
                    '"bias_risk":"low|moderate|high|unclear","metadata_confidence":0.0-1.0}\\n'
                    "Return ONLY the JSON."
                )
                with engine.connect() as conn:
                    meta_rows = conn.execute(text("""
                        SELECT ld.id, ld.title, ld.abstract, ld.source, ld.year
                        FROM literature_document ld
                        JOIN article_scenarios asn ON asn.document_id = ld.id
                        WHERE asn.scenario_id = :sid
                          AND ld.project_context = 'literev'
                          AND (ld.metadata_json IS NULL OR ld.metadata_json = '{}'::jsonb)
                        ORDER BY ld.id
                    """), {"sid": scenario_id}).mappings().fetchall()

                meta_extracted = 0
                meta_errors = 0
                for row in meta_rows:
                    try:
                        response = _client2.chat.completions.create(
                            model="gpt-4.1-mini",
                            messages=[
                                {"role": "system", "content": system_prompt_meta},
                                {"role": "user", "content": f"Title: {row['title']}\\n\\nAbstract: {(row['abstract'] or '')[:2000]}"},
                            ],
                            temperature=0.1,
                            max_tokens=300,
                            response_format={"type": "json_object"},
                        )
                        metadata = json.loads(response.choices[0].message.content)
                        metadata["metadata_confidence"] = float(metadata.get("metadata_confidence", 0.5))
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET metadata_json = CAST(:meta AS jsonb)
                                WHERE id = :article_id
                            """), {"meta": json.dumps(metadata), "article_id": row["id"]})
                        meta_extracted += 1
                    except Exception as e:
                        logger.warning(f"Pipeline metadata article {row['id']}: {e}")
                        meta_errors += 1
                    _time.sleep(0.05)
                update_step("metadata", "done", extracted=meta_extracted, errors=meta_errors)
            else:
                update_step("metadata", "skipped", reason="Clé OpenAI non configurée")
        except Exception as e:
            update_step("metadata", "error", error=str(e))

        # ── Étape 4 : Full-text (Unpaywall) ──────────────────────────────────
        update_step("fulltext", "running")
        try:
            import urllib.request as _urllib_req
            with engine.connect() as conn:
                ft_rows = conn.execute(text("""
                    SELECT ld.id, ld.doi
                    FROM literature_document ld
                    JOIN article_scenarios asn ON asn.document_id = ld.id
                    WHERE asn.scenario_id = :sid
                      AND ld.project_context = 'literev'
                      AND (ld.has_fulltext IS NULL OR ld.has_fulltext = false)
                      AND ld.doi IS NOT NULL
                    ORDER BY ld.id
                """), {"sid": scenario_id}).mappings().fetchall()

            ft_fetched = 0
            ft_errors = 0
            for row in ft_rows:
                try:
                    unpaywall_url = f"https://api.unpaywall.org/v2/{row['doi']}?email=literev@gesica.ch"
                    req = _urllib_req.Request(unpaywall_url, headers={"User-Agent": "LiteRev/1.0"})
                    with _urllib_req.urlopen(req, timeout=8) as resp:
                        data = json.loads(resp.read())
                    oa_url = None
                    if data.get("is_oa") and data.get("best_oa_location"):
                        oa_url = data["best_oa_location"].get("url_for_pdf") or data["best_oa_location"].get("url")
                    if oa_url:
                        with engine.begin() as conn:
                            conn.execute(text("""
                                UPDATE literature_document
                                SET has_fulltext = true, url = :url
                                WHERE id = :article_id
                            """), {"url": oa_url, "article_id": row["id"]})
                        ft_fetched += 1
                except Exception as e:
                    ft_errors += 1
                _time.sleep(0.1)
            update_step("fulltext", "done", fetched=ft_fetched, errors=ft_errors)
        except Exception as e:
            update_step("fulltext", "error", error=str(e))

        # ── Étape 5 : Clustering ──────────────────────────────────────────────
        update_step("clustering", "running")
        try:
            import numpy as np
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.cluster import KMeans
            from sklearn.decomposition import TruncatedSVD

            with engine.connect() as conn:
                cl_docs = list(conn.execute(text("""
                    SELECT d.id, d.title, d.abstract, d.year, d.journal
                    FROM literature_document d
                    JOIN article_scenarios asn ON asn.document_id = d.id
                    WHERE asn.scenario_id = :sid
                      AND d.project_context = 'literev'
                      AND d.abstract IS NOT NULL
                      AND LENGTH(d.abstract) > 50
                    ORDER BY d.year DESC NULLS LAST
                    LIMIT 500
                """), {"sid": scenario_id}).mappings().all())

            if len(cl_docs) >= 5:
                texts = [f"{d['title']} {d['abstract'] or ''}" for d in cl_docs]
                n_clusters = min(max(3, len(cl_docs) // 15), 12)
                vectorizer = TfidfVectorizer(max_features=500, stop_words="english", min_df=1)
                X = vectorizer.fit_transform(texts)
                if X.shape[1] > 50:
                    X = TruncatedSVD(n_components=min(50, X.shape[1] - 1)).fit_transform(X)
                km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = km.fit_predict(X)

                # Sauvegarder en cache
                import os as _os
                cache_dir = "/tmp/literev_clustering_cache"
                _os.makedirs(cache_dir, exist_ok=True)
                clusters_data = []
                for ci in range(n_clusters):
                    idxs = [i for i, l in enumerate(labels) if l == ci]
                    cluster_docs = [cl_docs[i] for i in idxs]
                    top_terms = vectorizer.get_feature_names_out()
                    clusters_data.append({
                        "id": ci,
                        "size": len(idxs),
                        "articles": [{"id": d["id"], "title": d["title"], "year": d["year"]} for d in cluster_docs[:5]],
                    })
                result_cache = {
                    "scenario_id": scenario_id,
                    "n_docs": len(cl_docs),
                    "n_clusters": n_clusters,
                    "clusters": clusters_data,
                    "from_cache": False,
                }
                with open(f"{cache_dir}/{scenario_id}.json", "w") as f:
                    json.dump(result_cache, f)
                update_step("clustering", "done", n_clusters=n_clusters, n_docs=len(cl_docs))
            else:
                update_step("clustering", "skipped", reason=f"Corpus insuffisant ({len(cl_docs)} articles)")
        except Exception as e:
            update_step("clustering", "error", error=str(e))

        # ── Fin du pipeline ───────────────────────────────────────────────────
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "done"
        _user_scenario_pipeline_jobs[scenario_id]["message"] = (
            f"Pipeline terminé : {ingested} articles ingérés et enrichis."
        )
        logger.info(f"Pipeline complet {scenario_id}: terminé.")

    except Exception as e:
        logger.error(f"Pipeline user_scenario {scenario_id} fatal: {e}", exc_info=True)
        _user_scenario_pipeline_jobs[scenario_id]["overall_status"] = "error"
        _user_scenario_pipeline_jobs[scenario_id]["error"] = str(e)


@app.post("/user-scenarios/{scenario_id}/populate")
def populate_user_scenario(
    scenario_id: str,
    max_results: int = 500,
) -> dict[str, Any]:
    """
    Déclenche l'ingestion PubMed en arrière-plan pour un scénario utilisateur.
    Sans limite fixe — max_results par défaut à 500, peut aller jusqu'à 10000.
    """
    import threading
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]

    job = _user_scenario_populate_jobs.get(scenario_id)
    if job and job.get("status") == "running":
        return {
            "scenario_id": scenario_id,
            "status": "already_running",
            "message": "Une ingestion est déjà en cours pour ce scénario.",
            "ingested": job.get("ingested", 0),
        }

    _user_scenario_populate_jobs[scenario_id] = {"status": "running", "ingested": 0, "errors": 0, "total_found": 0}
    t = threading.Thread(
        target=_run_user_scenario_populate,
        args=(scenario_id, query, row.get("filters") or {}, max_results, None),
        daemon=True,
    )
    t.start()

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": query,
        "max_results": max_results,
        "message": f"Ingestion PubMed lancée en arrière-plan pour '{row['name']}'. "
                   "Utilisez /user-scenarios/{id}/populate/status pour suivre la progression.",
    }


@app.get("/user-scenarios/{scenario_id}/populate/status")
def get_user_scenario_populate_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état de l'ingestion PubMed en cours pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_populate_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "status": "not_started",
            "message": "Aucune ingestion lancée. Appelez POST /user-scenarios/{id}/populate.",
        }
    return {"scenario_id": scenario_id, **job}


@app.post("/user-scenarios/{scenario_id}/pipeline")
def start_user_scenario_pipeline(
    scenario_id: str,
    max_results: int = 500,
) -> dict[str, Any]:
    """
    Déclenche le pipeline complet d'enrichissement en arrière-plan :
    PubMed → PICO → Métadonnées → Full-text → Clustering.
    Idéalement appelé dès qu'une recherche est validée en scénario épinglé.
    """
    import threading
    row = _get_user_scenario_or_404(scenario_id)
    query = row["query"]

    job = _user_scenario_pipeline_jobs.get(scenario_id)
    if job and job.get("overall_status") == "running":
        return {
            "scenario_id": scenario_id,
            "status": "already_running",
            "message": "Un pipeline est déjà en cours pour ce scénario.",
            "current_step": job.get("current_step"),
        }

    _user_scenario_pipeline_jobs[scenario_id] = {
        "overall_status": "starting",
        "current_step": "pubmed",
        "steps": {
            "pubmed": {"status": "pending"},
            "pico": {"status": "pending"},
            "metadata": {"status": "pending"},
            "fulltext": {"status": "pending"},
            "clustering": {"status": "pending"},
        },
    }

    t = threading.Thread(
        target=_run_user_scenario_full_pipeline,
        args=(scenario_id, query, row.get("filters") or {}, max_results),
        daemon=True,
    )
    t.start()

    return {
        "scenario_id": scenario_id,
        "status": "started",
        "query": query,
        "max_results": max_results,
        "message": f"Pipeline complet lancé pour '{row['name']}'. "
                   "Suivez la progression via GET /user-scenarios/{id}/pipeline/status.",
        "steps": ["pubmed", "pico", "metadata", "fulltext", "clustering"],
    }


@app.get("/user-scenarios/{scenario_id}/pipeline/status")
def get_user_scenario_pipeline_status(scenario_id: str) -> dict[str, Any]:
    """Retourne l'état détaillé du pipeline d'enrichissement pour un scénario utilisateur."""
    _get_user_scenario_or_404(scenario_id)
    job = _user_scenario_pipeline_jobs.get(scenario_id)
    if not job:
        return {
            "scenario_id": scenario_id,
            "overall_status": "not_started",
            "message": "Aucun pipeline lancé. Appelez POST /user-scenarios/{id}/pipeline.",
            "steps": {},
        }
    return {"scenario_id": scenario_id, **job}


'''

new_content = content[:start_idx] + NEW_BLOCK + content[end_idx:]
print(f"Nouveau contenu: {len(new_content)} caractères (original: {len(content)})")

with open("main.py", "w") as f:
    f.write(new_content)

print("✅ main.py mis à jour avec le pipeline complet")
