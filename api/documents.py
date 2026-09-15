"""Documents and chunks: ingestion models, embeddings, quality score, detail endpoints.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json
import re
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .core import _is_openai_quota_error, app, engine, logger, require_api_key

# ── Embedding résilient (anti « lot empoisonné ») ────────────────────────────
# L'API embeddings rejette TOUT le lot si UN SEUL input est invalide — le plus
# souvent trop de tokens : l'ancienne garde tronquait à 8000 CARACTÈRES (pas
# tokens), et du texte dense (CJK, identifiants, références) dépasse la limite de
# 8191 tokens. Sans isolation ni borne de tentatives, le lot fautif était
# ré-échoué à chaque cycle → chunks « en attente » bloqués indéfiniment. Ces
# helpers (1) tronquent par TOKENS, (2) mappent la réponse par index, et (3) si un
# lot échoue hors quota, ré-essaient chunk par chunk pour n'isoler QUE le fautif.
# None = pas encore tenté ; False = tiktoken indisponible (repli caractères, ne plus
# réessayer — sinon un téléchargement BPE qui échoue serait retenté à chaque appel) ;
# sinon = l'encodeur. Repli SÛR : 6000 caractères restent < 8191 tokens pour du texte
# réel (tiktoken, quand présent, préserve bien plus de contenu jusqu'à la vraie limite).
_TIKTOKEN_ENC: list = [None]


def _truncate_to_tokens(s: str, max_tokens: int = 8000) -> str:
    if not s:
        return s
    if _TIKTOKEN_ENC[0] is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC[0] = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TIKTOKEN_ENC[0] = False
    enc = _TIKTOKEN_ENC[0]
    if not enc:
        return s[:6000]
    try:
        toks = enc.encode(s)
        return s if len(toks) <= max_tokens else enc.decode(toks[:max_tokens])
    except Exception:
        return s[:6000]


#: Caractères de contrôle C0 hors tabulation/saut de ligne/retour chariot. NUL (0x00)
#: est le seul que PostgreSQL REFUSE dans un champ `text` (« PostgreSQL text fields
#: cannot contain NUL (0x00) bytes ») ; les autres sont acceptés mais n'ont aucun sens
#: dans du texte extrait — ils viennent d'un PDF mal formé ou d'un décodage raté, et
#: polluent aussi bien les prompts LLM que l'affichage.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def sanitize_db_text(s):
    """Retire les caractères de contrôle qui rendent un texte INSTOCKABLE en base.

    Bug corrigé : `pdftotext` renvoie parfois des octets NUL sur un PDF mal formé.
    L'INSERT du chunk lève alors psycopg.DataError, et comme
    `_insert_fulltext_chunks_ft` fait DELETE puis INSERT dans UNE transaction, tout
    est annulé : l'article n'obtient JAMAIS son texte intégral, avec pour seule trace
    un WARNING. Observé 3 fois en 45 min en production.

    Le nettoyage se fait ICI, à l'entrée, et pas à l'INSERT : un NUL casse aussi le
    découpage, le prompt PICO et l'appel d'embedding en aval.

    ATTENTION : `re.sub(r"\\s+", " ", …)`, appliqué juste avant dans les extracteurs,
    ne protège de RIEN — en Python `\\s` ne couvre pas `\\x00`. Vérifié.

    Pur : renvoie `s` inchangé si ce n'est pas une chaîne (None, bytes…)."""
    if not isinstance(s, str) or not s:
        return s
    return _CONTROL_CHARS_RE.sub("", s)


def _embed_one_call(client, batch: list) -> None:
    """Embède `batch` (liste de {id, content}) en UN appel et écrit les vecteurs,
    en mappant chaque vecteur par SON index de réponse (pas positionnel)."""
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=[_truncate_to_tokens(r["content"]) for r in batch],
    )
    by_index = {d.index: d.embedding for d in resp.data}
    with engine.begin() as cu:
        for k, r in enumerate(batch):
            emb = by_index.get(k)
            if emb is None:
                continue
            vec = "[" + ",".join(str(x) for x in emb) + "]"
            cu.execute(
                text("UPDATE document_chunk SET embedding = CAST(:vec AS vector) WHERE id = :cid"),
                {"vec": vec, "cid": r["id"]},
            )


def _embed_chunks_resilient(client, rows: list, batch_size: int = 100) -> tuple[int, list]:
    """Embède rows=[{id, content}] par lots ; si un lot échoue HORS quota, ré-essaie
    chunk par chunk pour ne pas bloquer les chunks sains. Renvoie (nb_embeddés,
    [ids_en_échec]). Propage l'erreur de quota pour que l'appelant déclenche le cooldown."""
    embedded, failed = 0, []
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i + batch_size]
        try:
            _embed_one_call(client, batch)
            embedded += len(batch)
        except Exception as e:
            if _is_openai_quota_error(e):
                raise
            for r in batch:  # lot empoisonné → on isole le fautif
                try:
                    _embed_one_call(client, [r])
                    embedded += 1
                except Exception as e2:
                    if _is_openai_quota_error(e2):
                        raise
                    failed.append(r["id"])
                    logger.warning(f"embed chunk {r['id']} échec définitif: {e2}")
    return embedded, failed


def _strategy_is_degraded(strategy: object, query: str | None = None) -> bool:
    """Vrai si une stratégie de recherche est un repli dégradé (échec LLM) :
    marquée degraded, ou dont la requête booléenne 'general' est vide / identique
    au texte brut / sans opérateur booléen — donc à régénérer."""
    if not isinstance(strategy, dict):
        return True
    if strategy.get("degraded"):
        return True
    general = (strategy.get("general") or "").strip()
    if not general:
        return True
    # Une vraie requête booléenne contient des opérateurs ou des guillemets.
    has_operators = any(op in general for op in (" AND ", " OR ", " NOT ", '"')) or "[" in general
    if not has_operators:
        return True
    # NB : on NE traite PLUS `general == query` comme dégradé. Si on atteint ici,
    # `general` contient des opérateurs booléens ; un utilisateur qui saisit un
    # booléen valide que le LLM préserve à l'identique produit légitimement
    # general == query — le marquer dégradé forçait une régénération inutile et
    # jetait les champs pubmed/synonyms fournis (une requête NL sans opérateur est
    # déjà captée par le garde `not has_operators` ci-dessus).
    return False


# Normalisation des types d'étude : le PICO LLM produit du texte libre (des
# centaines de variantes uniques). On regroupe en un jeu canonique fixe au moment
# de l'affichage (les valeurs brutes study_design / pico_json restent intactes).
# `d` = libellé brut en minuscules (cf. _study_design_distinct_cte). 1er match gagne.
_STUDY_DESIGN_CASE = """CASE
        WHEN d = '' THEN 'Non spécifié'
        WHEN d LIKE '%systematic review%' OR d LIKE '%meta-analysis%' OR d LIKE '%meta analysis%' OR d LIKE '%scoping review%' OR d LIKE '%umbrella review%' THEN 'Revue systématique / Méta-analyse'
        WHEN d LIKE '%non-randomi%' OR d LIKE '%non randomi%' OR d LIKE '%quasi-experimental%' OR d LIKE '%quasi experimental%' OR d LIKE '%interrupted time series%' OR d LIKE '%controlled before%' THEN 'Essai non randomisé / Quasi-expérimental'
        WHEN d LIKE '%randomi%' OR d LIKE 'rct%' THEN 'Essai contrôlé randomisé (RCT)'
        WHEN d LIKE '%controlled trial%' OR d LIKE '%clinical trial%' THEN 'Essai non randomisé / Quasi-expérimental'
        WHEN d LIKE '%case-control%' OR d LIKE '%case control%' THEN 'Cas-témoins'
        WHEN d LIKE '%cross-sectional%' OR d LIKE '%cross sectional%' THEN 'Transversale'
        WHEN d LIKE '%case report%' OR d LIKE '%case series%' THEN 'Cas clinique / Série de cas'
        WHEN d LIKE '%cohort%' OR d LIKE '%longitudinal%' OR d LIKE '%observational%' OR d LIKE '%retrospective%' OR d LIKE '%prospective%' OR d LIKE '%registry%' OR d LIKE '%surveillance%' THEN 'Cohorte / Observationnelle'
        WHEN d LIKE '%model%' OR d LIKE '%simulation%' OR d LIKE '%forecast%' OR d LIKE '%machine learning%' OR d LIKE '%in silico%' OR d LIKE '%predictive%' THEN 'Modélisation / Simulation'
        WHEN d LIKE '%qualitative%' OR d LIKE '%interview%' OR d LIKE '%focus group%' THEN 'Qualitative'
        WHEN d LIKE '%narrative review%' OR d LIKE '%literature review%' OR d LIKE '%guideline%' OR d LIKE '%review%' THEN 'Revue narrative / Recommandation'
        WHEN d LIKE '%in vitro%' OR d LIKE '%in vivo%' OR d LIKE '%animal%' OR d LIKE '%experimental%' OR d LIKE '%laboratory%' OR d LIKE '%murine%' OR d LIKE '% mice%' THEN 'Expérimentale / Préclinique'
        ELSE 'Autre'
      END"""

# Niveau de preuve GRADE (strict), déterminé par le DEVIS d'étude — PAS par un
# score composite (citations/récence/échantillon servent au classement, pas à la
# certitude). En GRADE :
#   - essais randomisés + synthèses d'essais  → certitude ÉLEVÉE  (« Forte »)
#   - essais contrôlés non randomisés / quasi-expérimental / recommandations
#                                             → certitude MODÉRÉE (« Modérée »)
#   - TOUTES les études observationnelles (cohortes, cas-témoins, transversales,
#     séries/rapports de cas, registres, surveillance…) partent en certitude
#     FAIBLE (« Faible ») ; idem revues narratives / avis d'experts.
# `d` = libellé brut du devis en minuscules. Le 1er match gagne : on teste
# « non-randomi… » avant « randomi… » pour ne pas surclasser les essais non
# randomisés. Devis inconnus / modélisation / qualitatif → « Non évaluée ».
_GRADE_LEVEL_CASE = """CASE
        WHEN d = '' THEN 'Non évaluée'
        WHEN d LIKE '%non-randomi%' OR d LIKE '%non randomi%' OR d LIKE '%quasi-experimental%' OR d LIKE '%quasi experimental%' OR d LIKE '%interrupted time series%' OR d LIKE '%controlled before%' THEN 'Modérée'
        WHEN d LIKE '%systematic review%' OR d LIKE '%meta-analysis%' OR d LIKE '%meta analysis%' OR d LIKE '%umbrella review%' THEN 'Forte'
        WHEN d LIKE '%randomi%' OR d LIKE 'rct%' THEN 'Forte'
        WHEN d LIKE '%controlled trial%' OR d LIKE '%clinical trial%' OR d LIKE '%guideline%' OR d LIKE '%recommendation%' THEN 'Modérée'
        WHEN d LIKE '%cohort%' OR d LIKE '%longitudinal%' OR d LIKE '%observational%' OR d LIKE '%retrospective%' OR d LIKE '%prospective%' OR d LIKE '%registry%' OR d LIKE '%surveillance%' OR d LIKE '%case-control%' OR d LIKE '%case control%' OR d LIKE '%cross-sectional%' OR d LIKE '%cross sectional%' OR d LIKE '%case report%' OR d LIKE '%case series%' OR d LIKE '%ecological%' OR d LIKE '%survey%' THEN 'Faible'
        WHEN d LIKE '%narrative%' OR d LIKE '%literature review%' OR d LIKE '%scoping review%' OR d LIKE '%editorial%' OR d LIKE '%commentary%' OR d LIKE '%opinion%' OR d LIKE '%review%' THEN 'Faible'
        ELSE 'Non évaluée'
      END"""

# Type d'article normalisé → UN vocabulaire contrôlé aligné sur ce que PubMed / Crossref /
# OpenAlex expriment (liste fusionnée validée). Le champ « Type » brut (source_type) est
# quasi toujours « article » (chaque fetcher jette le type natif de l'API) et study_design
# est du texte libre : ce CASE ramène les deux à une valeur unique et stable. `p` = signal
# brut en minuscules = COALESCE(publication_type, study_design, source_type). Le 1er match
# gagne, du plus spécifique au plus générique (non-randomisé avant randomisé, revue
# systématique avant revue simple, rapport de cas avant cohorte).
_PUB_TYPE_CASE = """CASE
        WHEN p = '' THEN 'Journal article'
        WHEN p LIKE '%systematic review%' OR p LIKE '%meta-analysis%' OR p LIKE '%meta analysis%' OR p LIKE '%umbrella review%' THEN 'Systematic review / Meta-analysis'
        WHEN p LIKE '%practice guideline%' OR p LIKE '%guideline%' OR p LIKE '%recommendation%' OR p LIKE '%consensus statement%' THEN 'Practice guideline'
        WHEN p LIKE '%non-random%' OR p LIKE '%non random%' OR p LIKE '%quasi-experimental%' OR p LIKE '%quasi experimental%' THEN 'Clinical trial (other)'
        WHEN p LIKE '%randomized controlled trial%' OR p LIKE '%randomised controlled trial%' OR p LIKE '%randomi%' OR p LIKE 'rct%' THEN 'Randomized controlled trial'
        WHEN p LIKE '%clinical trial%' OR p LIKE '%controlled trial%' OR p LIKE '%clinical_trial%' THEN 'Clinical trial (other)'
        WHEN p LIKE '%case report%' OR p LIKE '%case series%' OR p LIKE '%case-report%' THEN 'Case report / series'
        WHEN p LIKE '%cohort%' OR p LIKE '%case-control%' OR p LIKE '%case control%' OR p LIKE '%cross-sectional%' OR p LIKE '%cross sectional%' OR p LIKE '%observational%' OR p LIKE '%longitudinal%' OR p LIKE '%retrospective%' OR p LIKE '%prospective%' OR p LIKE '%registry%' OR p LIKE '%surveillance%' OR p LIKE '%ecological%' THEN 'Observational study'
        WHEN p LIKE '%editorial%' OR p LIKE '%letter%' OR p LIKE '%comment%' OR p LIKE '%opinion%' OR p LIKE '%correspondence%' THEN 'Editorial / Letter'
        WHEN p LIKE '%preprint%' OR p LIKE '%posted-content%' OR p LIKE '%posted content%' THEN 'Preprint'
        WHEN p LIKE '%conference%' OR p LIKE '%proceedings%' THEN 'Conference paper'
        WHEN p LIKE '%book%' OR p LIKE '%chapter%' OR p LIKE '%monograph%' OR p LIKE '%reference-entry%' THEN 'Book / Chapter'
        WHEN p LIKE '%dataset%' OR p LIKE '%data paper%' OR p LIKE '%data-set%' THEN 'Dataset / Other'
        WHEN p LIKE '%scoping review%' OR p LIKE '%narrative review%' OR p LIKE '%literature review%' OR p LIKE '%review%' THEN 'Review (narrative / scoping)'
        WHEN p LIKE '%journal%' OR p LIKE '%article%' THEN 'Journal article'
        ELSE 'Journal article'
      END"""

# ─────────────────────────────────────────────────────────────────────────────
# Pydantic models
# ─────────────────────────────────────────────────────────────────────────────
def _normalize_doi(doi: str | None) -> str | None:
    """Normalise un DOI en retirant les préfixes URL courants.
    Exemples : 'https://doi.org/10.1016/...' → '10.1016/...'
               'http://dx.doi.org/10.1016/...' → '10.1016/...'
    """
    if not doi:
        return doi
    doi = doi.strip()
    for prefix in ("https://doi.org/", "http://doi.org/",
                   "https://dx.doi.org/", "http://dx.doi.org/"):
        if doi.lower().startswith(prefix):
            return doi[len(prefix):]
    return doi


def _normalize_title(title: str | None) -> str:
    """Forme canonique d'un titre pour la déduplication inter-sources (minuscules,
    ponctuation → espace, espaces compactés). Le MÊME calcul est reproduit en SQL pour
    le backfill : btrim(regexp_replace(lower(title), '[^a-z0-9]+', ' ', 'g'))."""
    return re.sub(r"[^a-z0-9]+", " ", (title or "").lower()).strip()


# Hiérarchie des devis d'étude (evidence pyramid) → score 0–1.
_STUDY_DESIGN_TIERS = (
    (("meta-analysis", "méta-analyse", "metaanalysis"), 1.00),
    (("systematic review", "revue systématique", "systematic"), 0.92),
    # ⚠ « non-randomi… » AVANT « randomi… » : "non-randomized" CONTIENT "randomized",
    # donc sans ce garde-fou un essai NON randomisé serait surclassé au rang d'ECR.
    # Quasi-expérimental → certitude intermédiaire, cohérent avec _GRADE_LEVEL_CASE
    # qui les classe « Modérée » (au-dessus de l'observationnel, sous l'ECR).
    (("non-randomized", "non-randomised", "non randomi", "quasi-experimental",
      "quasi experimental", "interrupted time series", "controlled before"), 0.68),
    (("randomized", "randomised", "rct", "essai randomisé"), 0.85),
    (("cohort", "cohorte", "longitudinal"), 0.62),
    (("case-control", "cas-témoins", "case control"), 0.52),
    (("cross-sectional", "transversale", "survey", "observational"), 0.42),
    (("case series", "case report", "cas clinique", "série de cas"), 0.28),
    (("editorial", "commentary", "opinion", "letter", "éditorial"), 0.18),
)
_BIAS_RISK_FACTOR = {"low": 1.0, "faible": 1.0, "moderate": 0.85, "modéré": 0.85,
                     "unclear": 0.75, "incertain": 0.75, "high": 0.55, "élevé": 0.55}


def _coerce_int(value: Any) -> int | None:
    """Convertit prudemment une valeur (str/float/None) en int positif, sinon None."""
    if value is None:
        return None
    try:
        n = int(float(str(value).replace(",", "").strip()))
        return n if n > 0 else None
    except (ValueError, TypeError):
        return None


def _llm_lang_directive(lang: str | None) -> str:
    """Output-language instruction appended to LLM system prompts.
    Defaults to French (the app's default) so existing behaviour is unchanged
    when no language is supplied."""
    # Defensive: only a real string carries a language. Anything else — None, or a
    # FastAPI Query/Depends object leaked by an internal *direct* call to an endpoint
    # whose param defaults to Query(...) — falls back to the French default instead of
    # crashing on `.strip()` ("'Query' object has no attribute 'strip'").
    if not isinstance(lang, str):
        lang = None
    if (lang or "fr").strip().lower().startswith("en"):
        return ("\n\nRESPOND ENTIRELY IN ENGLISH. Every sentence, heading, bullet, "
                "and JSON string value you produce must be written in English, even if "
                "the source articles or the instructions above are in French.")
    return ("\n\nRéponds intégralement en FRANÇAIS. Toutes les phrases, titres, puces "
            "et valeurs de chaîne JSON que tu produis doivent être en français.")


def _design_tier_score(study_design: str | None) -> float | None:
    """Score 0–1 du devis d'étude d'après la pyramide des preuves, ou None si inconnu."""
    if not study_design:
        return None
    s = study_design.strip().lower()
    for keywords, score in _STUDY_DESIGN_TIERS:
        if any(k in s for k in keywords):
            return score
    return None


def _compute_quality_score(
    study_design: str | None = None,
    year: int | None = None,
    sample_size: int | None = None,
    citation_count: int | None = None,
    open_access: bool | None = None,
    bias_risk: str | None = None,
) -> float | None:
    """
    Score de qualité méthodologique déterministe et reproductible, dans [0, 1].

    Combinaison pondérée de signaux objectifs (aucun appel LLM) :
      - devis d'étude (pyramide des preuves)  — poids 0.50
      - taille d'échantillon (log)            — poids 0.18
      - citations (log)                       — poids 0.12
      - récence                               — poids 0.12
      - accès ouvert                          — poids 0.08
    Le score du devis est en outre modulé par le risque de biais s'il est connu.

    Renvoie None si AUCUN signal n'est disponible (on ne fabrique pas une note).
    Les poids sont renormalisés sur les seuls signaux présents, afin qu'un article
    bien documenté et un article peu documenté restent comparables.
    """
    import math
    from datetime import datetime, timezone

    components: list[tuple[float, float]] = []  # (sous-score 0–1, poids)

    design = _design_tier_score(study_design)
    if design is not None:
        if bias_risk:
            design *= _BIAS_RISK_FACTOR.get(str(bias_risk).strip().lower(), 1.0)
        components.append((min(1.0, design), 0.50))

    if sample_size and sample_size > 0:
        # 10 → 0.25, 1k → ~0.6, 100k → 1.0
        components.append((min(1.0, math.log10(sample_size) / 5.0), 0.18))

    if citation_count is not None and citation_count >= 0:
        # 0 → 0, ~30 → 0.5, 1000 → 1.0
        components.append((min(1.0, math.log10(citation_count + 1) / 3.0), 0.12))

    if year and year > 1950:
        current = datetime.now(timezone.utc).year
        age = max(0, current - int(year))
        # ≤2 ans → 1.0, dégrade linéairement, 0 au-delà de 25 ans
        components.append((max(0.0, min(1.0, (25 - age) / 23.0)), 0.12))

    if open_access is not None:
        components.append((1.0 if open_access else 0.0, 0.08))

    if not components:
        return None
    total_weight = sum(w for _, w in components)
    score = sum(sub * w for sub, w in components) / total_weight
    score = max(0.0, min(1.0, score))
    # Sans devis d'étude connu, on ne peut pas affirmer une qualité « Forte » :
    # on plafonne à 0.55 (au mieux « Modérée ») pour qu'un article récent mais
    # non caractérisé ne soit jamais classé au sommet de la pyramide des preuves.
    if design is None:
        score = min(score, 0.55)
    return round(score, 4)


class DocumentIn(BaseModel):
    source: str = Field(..., min_length=1)
    title: str = Field(..., min_length=1)
    abstract: str | None = None
    year: int | None = None
    url: str | None = None
    external_id: str | None = None
    project_context: str | None = None
    source_type: str | None = None
    disease_or_condition: str | None = None
    scenario_type: str | None = None
    geographic_scope: str | None = None
    evidence_category: str | None = None
    # Champs bibliographiques enrichis
    doi: str | None = None
    pmid: str | None = None
    authors: str | None = None
    journal: str | None = None
    open_access: bool | None = None

    @field_validator("doi", mode="before")
    @classmethod
    def _clean_doi(cls, v: str | None) -> str | None:
        return _normalize_doi(v)

class ChunkIn(BaseModel):
    document_id: int = Field(..., ge=1)
    chunk_index: int = Field(..., ge=0)
    content: str = Field(..., min_length=1)
    chunk_type: str | None = None
    section_label: str | None = None
    char_start: int | None = Field(None, ge=0)
    char_end: int | None = Field(None, ge=0)
    token_count: int | None = Field(None, ge=0)
    chunk_weight: float | None = Field(None, ge=0)
    metadata_json: dict[str, Any] | None = None

# ─────────────────────────────────────────────────────────────────────────────
# Write endpoints (protected)
# ─────────────────────────────────────────────────────────────────────────────
@app.post("/documents")
def create_document(
    doc: DocumentIn, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    sql = text("""
        INSERT INTO literature_document (
            source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category,
            doi, pmid, authors, journal, open_access
        )
        VALUES (
            :source, :title, :abstract, :year, :url, :external_id,
            :project_context, :source_type, :disease_or_condition,
            :scenario_type, :geographic_scope, :evidence_category,
            :doi, :pmid, :authors, :journal, :open_access
        )
        ON CONFLICT (doi) WHERE doi IS NOT NULL DO NOTHING
        RETURNING id
    """)
    params = doc.model_dump()
    with engine.begin() as conn:
        new_id = conn.execute(sql, params).scalar()
        deduplicated = False
        if new_id is None:
            # DOI already present (UNIQUE(doi) partial index) — return the existing row
            new_id = conn.execute(
                text("SELECT id FROM literature_document WHERE doi = :doi ORDER BY id LIMIT 1"),
                {"doi": params.get("doi")},
            ).scalar()
            deduplicated = True
    return {"id": new_id, "deduplicated": deduplicated}

@app.post("/chunks")
def create_chunk(
    chunk: ChunkIn, _: None = Depends(require_api_key)
) -> dict[str, Any]:
    sql = text("""
        INSERT INTO document_chunk (
            document_id, chunk_index, content, chunk_type, section_label,
            char_start, char_end, token_count, chunk_weight, metadata_json
        )
        VALUES (
            :document_id, :chunk_index, :content, :chunk_type, :section_label,
            :char_start, :char_end, :token_count, :chunk_weight,
            CAST(:metadata_json AS jsonb)
        )
        RETURNING id
    """)
    payload = chunk.model_dump()
    # Serialize metadata_json to a JSON string for the CAST(:x AS jsonb) binding
    meta = payload.get("metadata_json")
    if meta is None or meta == {}:
        payload["metadata_json"] = "{}"
    elif isinstance(meta, dict):
        payload["metadata_json"] = json.dumps(meta)
    # else already a string : leave as-is

    with engine.begin() as conn:
        new_id = conn.execute(sql, payload).scalar_one()
    return {"id": new_id}


# ─────────────────────────────────────────────────────────────────────────────
# Document detail
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/documents/{document_id}")
def get_document_detail(document_id: int) -> dict[str, Any]:
    # Détail enrichi ET recentré « santé publique » : on ajoute les champs
    # bibliographiques utiles (auteurs / revue / DOI / pays) et on NORMALISE les deux
    # axes de type à un vocabulaire contrôlé — `study_design` via _STUDY_DESIGN_CASE,
    # `article_type` via _PUB_TYPE_CASE — au lieu du texte libre. `d`/`p` = signaux
    # bruts en minuscules attendus par ces CASE. (source_type/scenario_type restent
    # disponibles mais le front ne montre plus le scénario ni les ids internes.)
    sql_doc = text(f"""
        WITH _doc AS (
            SELECT ld.id, ld.source, ld.title, ld.abstract, ld.year, ld.url, ld.external_id,
                   ld.project_context, ld.source_type, ld.disease_or_condition,
                   ld.scenario_type, ld.geographic_scope, ld.evidence_category,
                   ld.authors, ld.journal, ld.doi, ld.country,
                   lower(coalesce(nullif(trim(ld.study_design), ''), '')) AS d,
                   lower(coalesce(nullif(trim(ld.publication_type), ''),
                                  nullif(trim(ld.study_design), ''),
                                  ld.source_type, '')) AS p
            FROM literature_document ld
            WHERE ld.id = :document_id
            LIMIT 1
        )
        SELECT id, source, title, abstract, year, url, external_id,
               project_context, source_type, disease_or_condition,
               scenario_type, geographic_scope, evidence_category,
               authors, journal, doi, country,
               ({_STUDY_DESIGN_CASE}) AS study_design,
               ({_PUB_TYPE_CASE}) AS article_type
        FROM _doc
    """)
    sql_chunks = text("""
        SELECT
            id, document_id, chunk_index, content, chunk_type,
            section_label, char_start, char_end, token_count,
            chunk_weight, metadata_json
        FROM document_chunk
        WHERE document_id = :document_id
        ORDER BY chunk_index ASC
    """)
    with engine.connect() as conn:
        doc = conn.execute(sql_doc, {"document_id": document_id}).mappings().first()
        if not doc:
            raise HTTPException(status_code=404, detail="Document not found")
        chunks = conn.execute(
            sql_chunks, {"document_id": document_id}
        ).mappings().all()
    return {
        "document": dict(doc),
        "chunks": [dict(c) for c in chunks],
    }

# ─────────────────────────────────────────────────────────────────────────────
# GESICA Evidence Signals Extraction Engine
# ─────────────────────────────────────────────────────────────────────────────
def _extract_gesica_evidence(
    title: str | None, abstract: str | None, chunks: list[dict[str, Any]]
) -> dict[str, Any]:
    text_blob = " ".join([
        title or "",
        abstract or "",
        " ".join([c.get("content", "") for c in chunks]),
    ]).lower()

    demand_patterns = [
        "call volume", "demand forecasting", "arrival rate", "forecast",
        "predict", "ambulance demand", "ems demand", "workload", "hourly",
        "daily", "temporal", "timeseries", "time series", "xgboost", "lstm",
        "prophet", "random forest", "neural network", "regression", "mae", "mape", "rmse",
    ]
    resource_patterns = [
        "ambulance", "dispatch", "allocation", "fleet", "staffing",
        "crew", "response time", "location", "coverage", "optimization",
        "heuristics", "genetic algorithm", "simulation", "queuing", "chuv", "hug",
    ]
    crisis_patterns = [
        "disaster", "mass casualty", "mci", "crisis", "sanitarian",
        "epidemic", "pandemic", "influenza", "heatwave", "canicule",
        "flood", "evacuation", "surge", "capacity", "coordination",
    ]
    intervention_patterns = [
        "triage", "priority", "protocol", "diversion", "routing",
        "transfer", "telemedicine", "dispatch policy", "resource allocation",
    ]
    geography_patterns = [
        "geneva", "geneve", "vaud", "lausanne", "neuchatel", "france",
        "switzerland", "suisse", "cross-border", "transfrontalier", "rhone", "alps",
    ]

    setting_patterns = {
        "dispatch_center": ["dispatch", "regulation", "centre 15", "144", "call center"],
        "pre_hospital": ["ambulance", "smur", "paramedic", "ems", "pre-hospital", "rescue"],
        "hospital_er": ["emergency department", "er", "urgences", "hospital", "icu", "bed"],
    }
    scenario_rules = {
        "epidemic-surge": ["pandemic", "epidemic", "influenza", "covid", "outbreak", "virus"],
        "extreme-weather": ["heatwave", "canicule", "cold", "winter", "flood", "storm", "weather"],
        "mass-casualty": ["mci", "mass casualty", "terrorist", "accident", "explosion", "disaster"],
        "daily-operations": ["daily", "routine", "hourly", "weekday", "seasonal", "demand"],
    }

    metrics_patterns = [
        "auc", "auroc", "accuracy", "sensitivity", "specificity",
        "f1-score", "precision", "recall", "rmse", "mae", "mape",
    ]
    uncertainty_patterns = [
        "confidence interval", "uncertainty", "calibration",
        "probabilistic", "bayesian", "ensemble",
    ]

    def matched(patterns: list[str]) -> list[str]:
        return sorted({p for p in patterns if p in text_blob})

    horizon_matches = re.findall(
        r"\b(\d+\s*(?:hour|hours|day|days|week|weeks|month|months|year|years))\b",
        text_blob,
    )
    horizon_match_single = re.search(
        r"(\d+)\s*(hour|hours|day|days|week|weeks|month|months|year|years)",
        text_blob,
    )

    detected_settings = [
        s for s, keys in setting_patterns.items() if any(k in text_blob for k in keys)
    ]
    detected_scenarios = [
        s for s, keys in scenario_rules.items() if any(k in text_blob for k in keys)
    ]

    evidence_strength = "weak"
    if matched(metrics_patterns):
        evidence_strength = "moderate"
    if matched(metrics_patterns) and matched(uncertainty_patterns):
        evidence_strength = "strong"

    return {
        "demand_signals": matched(demand_patterns),
        "resource_types": matched(resource_patterns),
        "intervention_types": matched(intervention_patterns),
        "operational_settings": detected_settings,
        "scenario_tags": detected_scenarios,
        "forecast_horizon": horizon_match_single.group(0) if horizon_match_single else None,
        "forecast_horizons": horizon_matches[:10],
        "cross_border": any(x in text_blob for x in geography_patterns),
        "cross_border_signals": matched(geography_patterns),
        "crisis_signals": matched(crisis_patterns),
        "evidence_strength": evidence_strength,
        "uncertainty_handling": matched(uncertainty_patterns),
        "reported_metrics": matched(metrics_patterns),
        "is_ems_or_crisis_relevant": bool(
            matched(demand_patterns) or matched(resource_patterns) or matched(crisis_patterns)
        ),
    }

@app.get("/evidence-summary/{document_id}")
def get_evidence_summary(document_id: int) -> dict[str, Any]:
    sql_doc = text("""
        SELECT
            id, source, title, abstract, year, url, external_id,
            project_context, source_type, disease_or_condition,
            scenario_type, geographic_scope, evidence_category
        FROM literature_document
        WHERE id = :document_id
        LIMIT 1
    """)
    sql_chunks = text("""
        SELECT id, document_id, chunk_index, content
        FROM document_chunk
        WHERE document_id = :document_id
        ORDER BY chunk_index
    """)

    with engine.connect() as conn:
        doc_row = conn.execute(sql_doc, {"document_id": document_id}).mappings().first()
        if not doc_row:
            raise HTTPException(status_code=404, detail="Document not found")
        chunk_rows = conn.execute(
            sql_chunks, {"document_id": document_id}
        ).mappings().all()

    document = dict(doc_row)
    chunks = [dict(r) for r in chunk_rows]
    signals = _extract_gesica_evidence(
        document.get("title"), document.get("abstract"), chunks
    )

    return {
        "document": document,
        "summary": {
            "project_context": document.get("project_context"),
            "scenario_type": document.get("scenario_type"),
            "evidence_category": document.get("evidence_category"),
            "geographic_scope": document.get("geographic_scope"),
            "disease_or_condition": document.get("disease_or_condition"),
        },
        "gesica_signals": signals,
        "chunk_count": len(chunks),
    }


@app.post("/admin/recompute-quality-scores")
def recompute_quality_scores(
    limit: int = 5000,
    only_missing: bool = True,
    _: None = Depends(require_api_key),
) -> dict[str, Any]:
    """
    (Re)calcule le quality_score déterministe sur le corpus existant à partir des
    colonnes structurées et de metadata_json (study_type, sample_size, bias_risk).
    Idempotent. `only_missing=True` ne traite que les documents sans score
    (quality_score NULL ou 0). À appeler par lots (`limit`) pour le backfill.
    """
    where_missing = "AND (quality_score IS NULL OR quality_score = 0)" if only_missing else ""
    with engine.connect() as conn:
        rows = conn.execute(text(f"""
            SELECT id, year, citation_count, open_access, study_design, sample_size,
                   metadata_json
            FROM literature_document
            WHERE project_context = 'literev'
              {where_missing}
            ORDER BY id
            LIMIT :limit
        """), {"limit": max(1, min(limit, 50000))}).mappings().fetchall()

    updated = 0
    skipped_no_signal = 0
    for r in rows:
        meta = r["metadata_json"] if isinstance(r["metadata_json"], dict) else {}
        study_design = r["study_design"] or (meta.get("study_type") if meta else None)
        sample_size = r["sample_size"] or _coerce_int(meta.get("sample_size") if meta else None)
        score = _compute_quality_score(
            study_design=study_design,
            year=r["year"],
            sample_size=sample_size,
            citation_count=r["citation_count"],
            open_access=r["open_access"],
            bias_risk=(meta.get("bias_risk") if meta else None),
        )
        if score is None:
            skipped_no_signal += 1
            continue
        with engine.begin() as conn:
            conn.execute(text("""
                UPDATE literature_document
                SET quality_score = :score,
                    study_design = COALESCE(:study_design, study_design),
                    sample_size = COALESCE(:sample_size, sample_size)
                WHERE id = :id
            """), {"score": score, "study_design": study_design,
                   "sample_size": sample_size, "id": r["id"]})
        updated += 1

    return {
        "scanned": len(rows),
        "updated": updated,
        "skipped_no_signal": skipped_no_signal,
        "only_missing": only_missing,
        "limit": limit,
        "message": "Relancez l'endpoint tant que 'scanned' == 'limit' pour traiter tout le corpus.",
    }
