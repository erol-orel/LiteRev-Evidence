from __future__ import annotations

from typing import Iterable

PUBMED_TYPE_MAP = {
    "systematic review": ("systematic_review", "systematic_review"),
    "meta-analysis": ("meta_analysis", "systematic_review"),
    "review": ("review", "review"),
    "clinical trial": ("article", "interventional_study"),
    "randomized controlled trial": ("article", "interventional_study"),
    "guideline": ("guideline", "guideline"),
    "practice guideline": ("guideline", "guideline"),
    "editorial": ("editorial", "commentary"),
    "letter": ("letter", "commentary"),
    "protocol": ("protocol", "methodological"),
}

OPENALEX_TYPE_MAP = {
    "article": ("article", None),
    "review": ("review", "review"),
    "preprint": ("preprint", None),
    "book-chapter": ("book_chapter", None),
    "dissertation": ("article", "methodological"),
    "editorial": ("editorial", "commentary"),
    "letter": ("letter", "commentary"),
    "report": ("article", "surveillance_report"),
}

CROSSREF_TYPE_MAP = {
    "journal-article": ("article", None),
    "proceedings-article": ("conference_paper", None),
    "posted-content": ("preprint", None),
    "book-chapter": ("book_chapter", None),
    "report": ("article", "surveillance_report"),
    "dissertation": ("article", "methodological"),
}

SCENARIO_KEYWORDS = {
    "outbreak_detection": ["outbreak detection", "detect outbreak", "early detection", "outbreak alert"],
    "surveillance": ["surveillance", "monitoring", "sentinel"],
    "epidemic_intelligence": ["epidemic intelligence", "event-based surveillance", "event based surveillance"],
    "forecasting": ["forecast", "forecasting", "prediction model", "predictive model", "phylodynamic"],
    "preparedness": ["preparedness", "pandemic preparedness"],
    "response": ["response", "response strategy", "intervention"],
    "hospital_surveillance": ["nosocomial", "hospital surveillance", "emergency department", "intra-hospital"],
    "genomic_epidemiology": ["genomic epidemiology", "pathogen genomics", "whole-genome sequencing", "wgs"],
    "wastewater_surveillance": ["wastewater", "sewage surveillance", "wbe"],
}

DISEASE_KEYWORDS = {
    "covid-19": ["covid", "sars-cov-2", "coronavirus disease 2019"],
    "influenza": ["influenza", "flu"],
    "mpox": ["mpox", "monkeypox"],
    "tuberculosis": ["tuberculosis", "tb"],
    "hiv": ["hiv", "human immunodeficiency virus"],
    "malaria": ["malaria"],
    "mrsa": ["mrsa", "methicillin-resistant staphylococcus aureus"],
}

GEO_KEYWORDS = {
    "global": ["global", "worldwide", "international"],
    "africa": ["africa", "african"],
    "europe": ["europe", "european"],
    "france": ["france", "french"],
    "bangladesh": ["bangladesh"],
    "usa": ["united states", "usa", "u.s."],
}

def _clean(v: str | None) -> str | None:
    if v is None:
        return None
    v = str(v).strip()
    return v or None

def _find_keyword(text: str, mapping: dict[str, list[str]]) -> str | None:
    t = text.lower()
    for label, keywords in mapping.items():
        if any(k in t for k in keywords):
            return label
    return None

def _infer_evidence(text: str) -> str | None:
    t = text.lower()
    if "systematic review" in t:
        return "systematic_review"
    if "meta-analysis" in t or "meta analysis" in t:
        return "systematic_review"
    if "review" in t:
        return "review"
    if "guideline" in t:
        return "guideline"
    if any(k in t for k in ["method", "framework", "pipeline", "toolkit", "algorithm"]):
        return "methodological"
    if any(k in t for k in ["cohort", "case-control", "cross-sectional", "observational"]):
        return "observational_study"
    if any(k in t for k in ["trial", "randomized", "intervention"]):
        return "interventional_study"
    return None

def normalize_pubmed(publication_types: Iterable[str] | None, title: str | None = None, abstract: str | None = None) -> dict[str, str | None]:
    source_type = "article"
    evidence_category = None
    pts = [p.strip().lower() for p in (publication_types or []) if p]
    for key in [
        "systematic review",
        "meta-analysis",
        "review",
        "clinical trial",
        "randomized controlled trial",
        "guideline",
        "practice guideline",
        "editorial",
        "letter",
        "protocol",
    ]:
        if key in pts:
            source_type, evidence_category = PUBMED_TYPE_MAP[key]
            break
    combined = " ".join(filter(None, [title, abstract]))
    return {
        "source_type": source_type,
        "evidence_category": evidence_category or _infer_evidence(combined),
        "scenario_type": _find_keyword(combined, SCENARIO_KEYWORDS),
        "disease_or_condition": _find_keyword(combined, DISEASE_KEYWORDS),
        "geographic_scope": _find_keyword(combined, GEO_KEYWORDS),
    }

def normalize_pmc(article_type: str | None, pmid: str | None = None, title: str | None = None, abstract: str | None = None) -> dict[str, str | None]:
    st = _clean(article_type)
    if st:
        st = st.lower().replace(" ", "_").replace("-", "_")
    combined = " ".join(filter(None, [title, abstract]))
    source_type = st or ("article" if pmid else "unknown")
    return {
        "source_type": source_type,
        "evidence_category": _infer_evidence(combined),
        "scenario_type": _find_keyword(combined, SCENARIO_KEYWORDS),
        "disease_or_condition": _find_keyword(combined, DISEASE_KEYWORDS),
        "geographic_scope": _find_keyword(combined, GEO_KEYWORDS),
    }

def normalize_openalex(work_type: str | None, title: str | None = None, abstract: str | None = None, concepts: Iterable[str] | None = None) -> dict[str, str | None]:
    wt = _clean(work_type)
    source_type, evidence_category = OPENALEX_TYPE_MAP.get((wt or "").lower(), ("article", None))
    combined = " ".join(filter(None, [title, abstract, " ".join(concepts or [])]))
    return {
        "source_type": source_type,
        "evidence_category": evidence_category or _infer_evidence(combined),
        "scenario_type": _find_keyword(combined, SCENARIO_KEYWORDS),
        "disease_or_condition": _find_keyword(combined, DISEASE_KEYWORDS),
        "geographic_scope": _find_keyword(combined, GEO_KEYWORDS),
    }

def normalize_crossref(work_type: str | None, title: str | None = None, abstract: str | None = None, subjects: Iterable[str] | None = None) -> dict[str, str | None]:
    wt = _clean(work_type)
    source_type, evidence_category = CROSSREF_TYPE_MAP.get((wt or "").lower(), ("article", None))
    combined = " ".join(filter(None, [title, abstract, " ".join(subjects or [])]))
    return {
        "source_type": source_type,
        "evidence_category": evidence_category or _infer_evidence(combined),
        "scenario_type": _find_keyword(combined, SCENARIO_KEYWORDS),
        "disease_or_condition": _find_keyword(combined, DISEASE_KEYWORDS),
        "geographic_scope": _find_keyword(combined, GEO_KEYWORDS),
    }

def normalize_record(source: str, metadata: dict | None = None, title: str | None = None, abstract: str | None = None) -> dict[str, str | None]:
    metadata = metadata or {}
    s = (source or "").lower()
    if s == "pubmed":
        return normalize_pubmed(metadata.get("publication_types"), title=title, abstract=abstract)
    if s in {"pmc", "pubmedcentral", "pubmed_central"}:
        return normalize_pmc(metadata.get("article_type"), pmid=metadata.get("pmid"), title=title, abstract=abstract)
    if s == "openalex":
        return normalize_openalex(metadata.get("type"), title=title, abstract=abstract, concepts=metadata.get("concepts"))
    if s == "crossref":
        return normalize_crossref(metadata.get("type"), title=title, abstract=abstract, subjects=metadata.get("subjects"))
    combined = " ".join(filter(None, [title, abstract]))
    return {
        "source_type": "unknown",
        "evidence_category": _infer_evidence(combined),
        "scenario_type": _find_keyword(combined, SCENARIO_KEYWORDS),
        "disease_or_condition": _find_keyword(combined, DISEASE_KEYWORDS),
        "geographic_scope": _find_keyword(combined, GEO_KEYWORDS),
    }
