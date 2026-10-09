"""Questions asked of a scenario: kept, re-readable, exportable, and able to
propose a change to the scenario.

An answer from the assistant used to be a chat turn that scrolled away. It is in
fact a small piece of research: it has a SCOPE (which scenario, which threshold,
which narrowing was in force), a DATE, and a SET OF SOURCES. Throwing those three
away each time means the same question re-asked next month cannot be compared with
the last answer, cannot be cited, and cannot be audited.

So every question is stored with what it was asked of and what it answered, can be
exported as a document, and, where its answer names a value the scenario already
holds differently, can propose that difference as a change for a reviewer to accept.

`main` re-exports everything for the scripts, tools and tests.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .scenario_store import _get_scenario_threshold, _get_user_scenario_or_404

# ── Table ────────────────────────────────────────────────────────────────────

_DDL = """
CREATE TABLE IF NOT EXISTS scenario_question (
    id            BIGSERIAL PRIMARY KEY,
    scenario_id   VARCHAR(100) NOT NULL,
    question      TEXT NOT NULL,
    answer        TEXT NOT NULL DEFAULT '',
    lang          VARCHAR(8),
    -- La PORTÉE : de quoi la réponse parle. Sans elle, deux réponses au même
    -- libellé ne sont pas comparables, parce qu'elles peuvent porter sur deux
    -- sous-ensembles différents du même scénario.
    threshold     DOUBLE PRECISION,
    scope_json    JSONB,
    -- Les SOURCES réellement citées, et les compteurs affichés sous la réponse.
    sources_json  JSONB,
    papers_used   INTEGER,
    papers_quoted INTEGER,
    digest_complete BOOLEAN DEFAULT FALSE,
    -- Les propositions tirées de la réponse, et leur sort.
    proposals_json JSONB,
    owner_email   VARCHAR(255),
    created_at    TIMESTAMP DEFAULT NOW()
)
"""
_DDL_INDEX = ("CREATE INDEX IF NOT EXISTS scenario_question_scenario_idx "
              "ON scenario_question (scenario_id, created_at DESC)")


def _ensure_question_table(conn) -> None:
    conn.execute(text(_DDL))
    conn.execute(text(_DDL_INDEX))


try:
    with engine.begin() as _c:
        _ensure_question_table(_c)
    logger.info("Table scenario_question vérifiée/créée.")
except Exception as _e:                                           # noqa: BLE001
    logger.warning(f"scenario_question DDL: {_e}")


# ── La portée d'une question, en une phrase ──────────────────────────────────

def describe_scope(scope: dict | None, threshold: float | None) -> str:
    """La portée d'une question en une phrase lisible, pour l'en-tête d'un export
    et pour la liste. Pure : aucune connexion, testable hors base.

    Sans portée explicite, la réponse porte sur le sous-ensemble pertinent entier,
    ce que la phrase doit DIRE plutôt que laisser deviner."""
    scope = scope or {}
    parts: list[str] = []
    if threshold is not None:
        parts.append(f"threshold {float(threshold):.2f}")
    for key, label in (("clusters", "cluster"), ("designs", "study design"),
                       ("levels", "evidence level"), ("concepts", "concept")):
        values = [str(v) for v in (scope.get(key) or []) if str(v).strip()]
        if values:
            noun = label if len(values) == 1 else label + "s"
            parts.append(f"{noun}: {', '.join(values)}")
    if scope.get("relevant_only"):
        parts.append("relevant subset only")
    if scope.get("query"):
        parts.append(f"corpus search: {scope['query']}")
    if not parts:
        return "the whole relevant subset"
    return " · ".join(parts)


# ── Propositions tirées d'une réponse ────────────────────────────────────────
# Un nombre cité dans une réponse n'est une PROPOSITION que s'il porte sur une
# grandeur que le scénario tient déjà, et qu'il en diffère. Le reste est du texte.
#
# L'extraction est volontairement littérale et sans modèle : on cherche les
# grandeurs nommées, telles qu'elles s'écrivent dans la littérature. Un faux
# positif coûte une ligne qu'un relecteur rejette ; un modèle qui « comprend » la
# réponse coûterait une proposition inventée, ce qui est bien pire.

_PARAM_PATTERNS: tuple[tuple[str, str, str], ...] = (
    ("r0", r"\bR0\b|\bbasic reproduction number\b|\bnombre de reproduction de base\b", ""),
    ("incubation_period", r"\bincubation period\b|\bpériode d'incubation\b", "days"),
    ("serial_interval", r"\bserial interval\b|\bintervalle sériel\b", "days"),
    ("latent_period", r"\blatent period\b|\bpériode de latence\b", "days"),
    ("infectious_period", r"\binfectious period\b|\bpériode infectieuse\b", "days"),
    # « létalité » manquait : le motif ne portait que « letalité » (sans accent) et
    # « léthalité » (avec un h), si bien que l'orthographe courante n'était pas reconnue.
    ("case_fatality_rate",
     r"\bcase[- ]fatality (?:rate|ratio)\b|\bCFR\b|\bl[eé]talit[eé]\b|\bléthalité\b"
     r"|\btaux de l[eé]talit[eé]\b", "%"),
    ("attack_rate", r"\battack rate\b|\btaux d'attaque\b", "%"),
)
# Un nombre, éventuellement décimal, éventuellement un intervalle « 2.1 to 3.4 ».
_NUMBER = r"(\d+(?:[.,]\d+)?)"
_RANGE = rf"{_NUMBER}\s*(?:-|–|to|à)\s*{_NUMBER}"


def _as_float(raw: str) -> float | None:
    try:
        return float(str(raw).replace(",", "."))
    except (TypeError, ValueError):
        return None


#: Les mots qui comptent des OBJETS, pas une grandeur. « Sur 24 études, le nombre de
#: reproduction de base n'est pas rapporté » rendait R0 = 24, parce que le nombre était
#: cherché dans TOUTE la phrase, y compris avant le libellé, et qu'aucun mot n'était
#: récusé. Cette valeur était ensuite proposée pour adoption dans la spécification du
#: modèle, sans que la phrase soit montrée.
_COUNTING_WORDS = (
    "article", "articles", "study", "studies", "étude", "études", "paper", "papers",
    "patient", "patients", "cas", "case", "cases", "country", "countries", "pays",
    "review", "reviews", "revue", "revues", "n", "total", "sample", "échantillon",
    "cohort", "cohorte", "participants", "participant", "sujets", "sujet",
    "outbreak", "outbreaks", "foyer", "foyers", "reference", "references",
)
#: Ce qui FERME une proposition : au-delà, le nombre ne porte plus sur le libellé.
_CLAUSE_END = re.compile(r"[;:]|,\s*(?:and|but|while|whereas|et|mais|alors que|tandis que)\b",
                         re.IGNORECASE)
#: Au plus ce nombre de caractères entre le libellé et son nombre. « La période
#: infectieuse est de 5 jours » en compte 9 ; une phrase entière en compte cent.
_MAX_GAP_CHARS = 60
#: Une NÉGATION entre le libellé et le nombre : « la létalité n'est pas rapportée dans
#: 24 études » ne dit pas que la létalité vaut 24.
_NEGATION = re.compile(r"\b(?:not|no|never|pas|aucun|aucune|non|sans)\b", re.IGNORECASE)


def _value_after_label(tail: str) -> re.Match | None:
    """Le nombre qui suit le libellé, s'il lui appartient vraiment.

    Trois conditions, toutes nécessaires : il est APRÈS le libellé (et non n'importe où
    dans la phrase), avant la fin de la proposition, et pas précédé d'un mot qui compte
    des objets. Sinon, rien : une valeur inventée proposée pour adoption dans la
    spécification du modèle est pire qu'une absence de valeur."""
    cut = _CLAUSE_END.search(tail)
    window = tail[: cut.start()] if cut else tail
    for m in (re.search(_RANGE, window), re.search(_NUMBER, window)):
        if not m:
            continue
        gap = window[: m.start()]
        if len(gap) > _MAX_GAP_CHARS:
            return None
        if _NEGATION.search(gap):
            return None
        words = re.findall(r"[\wÀ-ÿ=]+", gap.lower())
        if words and words[-1].strip("=") in _COUNTING_WORDS:
            return None
        return m
    return None


def extract_parameter_claims(answer: str) -> list[dict[str, Any]]:
    """Les grandeurs épidémiologiques nommées dans une réponse, avec leur valeur.

    Pure. Ne renvoie QUE ce qui est écrit : une grandeur reconnue SUIVIE, dans la même
    proposition, d'un nombre ou d'un intervalle qui lui appartient. Rien n'est déduit.

    Le nombre était cherché dans toute la phrase, libellé masqué mais texte AVANT le
    libellé inclus : « Sur 24 études, le nombre de reproduction de base n'est pas
    rapporté » rendait R0 = 24, et l'historique proposait cette valeur pour adoption
    dans la spécification du modèle sans montrer la phrase."""
    out: list[dict[str, Any]] = []
    if not answer:
        return out
    # Une phrase à la fois : un R0 dans une phrase et un taux dans la suivante ne
    # doivent pas se mélanger.
    for sentence in re.split(r"(?<=[.!?])\s+|\n+", answer):
        for key, pattern, unit in _PARAM_PATTERNS:
            label = re.search(pattern, sentence, re.IGNORECASE)
            if not label:
                continue
            m = _value_after_label(sentence[label.end():])
            if m is None:
                break
            if m.re.pattern == _RANGE:
                lo, hi = _as_float(m.group(1)), _as_float(m.group(2))
                if lo is not None and hi is not None:
                    out.append({"key": key, "unit": unit, "low": lo, "high": hi,
                                "value": round((lo + hi) / 2, 4),
                                "quote": sentence.strip()[:400]})
            else:
                v = _as_float(m.group(1))
                if v is not None:
                    out.append({"key": key, "unit": unit, "low": None, "high": None,
                                "value": v, "quote": sentence.strip()[:400]})
            break
    # Une grandeur citée plusieurs fois : on garde la première occurrence, qui est
    # celle que la réponse met en avant.
    seen: set[str] = set()
    unique: list[dict[str, Any]] = []
    for claim in out:
        if claim["key"] in seen:
            continue
        seen.add(claim["key"])
        unique.append(claim)
    return unique


def diff_against_current(claims: list[dict], current: dict[str, Any] | None,
                         tolerance: float = 0.05) -> list[dict[str, Any]]:
    """Les claims qui DIFFÈRENT de ce que le scénario tient déjà.

    Une valeur égale à celle en place n'est pas une proposition, c'est une
    confirmation, et l'afficher comme un changement à accepter userait la
    confiance du relecteur. `tolerance` est une différence relative en deçà de
    laquelle deux valeurs sont tenues pour la même. Pure."""
    current = current or {}
    out: list[dict[str, Any]] = []
    for claim in claims:
        key = claim["key"]
        have = current.get(key)
        have_value = have.get("value") if isinstance(have, dict) else have
        have_value = _as_float(have_value) if have_value is not None else None
        new_value = claim.get("value")
        if have_value is not None and new_value is not None:
            scale = max(abs(have_value), abs(new_value), 1e-9)
            if abs(have_value - new_value) / scale <= tolerance:
                continue                     # même valeur : rien à proposer
        out.append({**claim, "current": have_value,
                    "kind": "update" if have_value is not None else "new"})
    return out


# ── Lecture et écriture ──────────────────────────────────────────────────────

class QuestionIn(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000)
    answer: str = Field("", max_length=200_000)
    lang: str | None = None
    threshold: float | None = None
    scope: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    papers_used: int | None = None
    # Les articles RAPATRIÉS, pas ceux que la réponse cite. La colonne en base garde son
    # ancien nom (`papers_quoted`) : renommer une colonne n'ajoute rien, mais le champ
    # servi et affiché doit dire ce qu'il compte.
    papers_retrieved: int | None = None
    digest_complete: bool = False
    owner_email: str | None = None


def _row_to_question(row) -> dict[str, Any]:
    d = dict(row)
    for key in ("scope_json", "sources_json", "proposals_json"):
        value = d.pop(key)
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:                                     # noqa: BLE001
                value = None
        d[key[:-5]] = value
    d["created_at"] = d["created_at"].isoformat() if d.get("created_at") else None
    d["scope_label"] = describe_scope(d.get("scope"), d.get("threshold"))
    return d


@app.post("/user-scenarios/{scenario_id}/questions")
def save_scenario_question(scenario_id: str, payload: QuestionIn,
                           _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Enregistre une question et sa réponse, avec la portée sur laquelle elle a
    été posée. Appelé par l'interface une fois la réponse reçue en entier.

    La clé est exigée, comme elle l'est déjà pour SUPPRIMER la même ligne : écrire dans
    l'historique de n'importe quel scénario, avec un texte arbitraire de 200 000
    caractères et une adresse de propriétaire choisie, ne demandait rien, pendant que
    l'interface affichait un badge « lecture seule »."""
    _get_user_scenario_or_404(scenario_id)
    threshold = (payload.threshold if payload.threshold is not None
                 else _get_scenario_threshold(scenario_id))
    proposals = diff_against_current(
        extract_parameter_claims(payload.answer), _current_parameters(scenario_id))
    with engine.begin() as conn:
        _ensure_question_table(conn)
        row = conn.execute(text("""
            INSERT INTO scenario_question
                (scenario_id, question, answer, lang, threshold, scope_json,
                 sources_json, papers_used, papers_quoted, digest_complete,
                 proposals_json, owner_email)
            VALUES
                (:sid, :q, :a, :lang, :thr, CAST(:scope AS jsonb),
                 CAST(:sources AS jsonb), :used, :quoted, :complete,
                 CAST(:proposals AS jsonb), :email)
            RETURNING *
        """), {
            "sid": scenario_id, "q": payload.question.strip(), "a": payload.answer or "",
            "lang": payload.lang, "thr": threshold,
            "scope": json.dumps(payload.scope or {}),
            "sources": json.dumps(payload.sources or []),
            "used": payload.papers_used, "quoted": payload.papers_retrieved,
            "complete": bool(payload.digest_complete),
            "proposals": json.dumps(proposals),
            "email": payload.owner_email,
        }).mappings().first()
    return _row_to_question(row)


@app.get("/user-scenarios/{scenario_id}/questions")
def list_scenario_questions(scenario_id: str, limit: int = 50,
                            offset: int = 0) -> dict[str, Any]:
    """L'historique des questions posées à ce scénario, la plus récente d'abord."""
    _get_user_scenario_or_404(scenario_id)
    limit = max(1, min(int(limit), 500))
    offset = max(0, int(offset))
    with engine.connect() as conn:
        _ensure_question_table(conn)
        total = conn.execute(text(
            "SELECT COUNT(*) FROM scenario_question WHERE scenario_id = :sid"),
            {"sid": scenario_id}).scalar() or 0
        rows = conn.execute(text("""
            SELECT * FROM scenario_question
            WHERE scenario_id = :sid
            ORDER BY created_at DESC, id DESC
            LIMIT :limit OFFSET :offset
        """), {"sid": scenario_id, "limit": limit, "offset": offset}).mappings().all()
    items = [_row_to_question(r) for r in rows]
    # Les reprises d'une MÊME question : ce qui permet de voir bouger une réponse
    # quand le corpus a grandi. Comparé sur le libellé normalisé.
    counts: dict[str, int] = {}
    for it in items:
        key = re.sub(r"\s+", " ", (it["question"] or "").strip().lower())
        counts[key] = counts.get(key, 0) + 1
    for it in items:
        key = re.sub(r"\s+", " ", (it["question"] or "").strip().lower())
        it["asked_times_in_page"] = counts[key]
    return {"scenario_id": scenario_id, "total": int(total), "items": items}


@app.get("/user-scenarios/{scenario_id}/questions/{question_id}")
def get_scenario_question(scenario_id: str, question_id: int) -> dict[str, Any]:
    with engine.connect() as conn:
        _ensure_question_table(conn)
        row = conn.execute(text(
            "SELECT * FROM scenario_question WHERE id = :qid AND scenario_id = :sid"),
            {"qid": question_id, "sid": scenario_id}).mappings().first()
    if not row:
        raise HTTPException(status_code=404, detail="Question non trouvée")
    return _row_to_question(row)


@app.delete("/user-scenarios/{scenario_id}/questions/{question_id}")
def delete_scenario_question(scenario_id: str, question_id: int,
                             _: None = Depends(require_api_key)) -> dict[str, Any]:
    with engine.begin() as conn:
        _ensure_question_table(conn)
        n = conn.execute(text(
            "DELETE FROM scenario_question WHERE id = :qid AND scenario_id = :sid"),
            {"qid": question_id, "sid": scenario_id}).rowcount
    if not n:
        raise HTTPException(status_code=404, detail="Question non trouvée")
    return {"deleted": True, "id": question_id}


def _settings_variables(scenario_id: str) -> dict[str, Any]:
    """Le bloc `variables_json` du scénario (qui porte le model_spec), ou {}."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT variables_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
    except Exception:                                             # noqa: BLE001
        return {}
    blob = row["variables_json"] if row else None
    if isinstance(blob, str):
        try:
            blob = json.loads(blob)
        except Exception:                                         # noqa: BLE001
            return {}
    return blob if isinstance(blob, dict) else {}


def _current_parameters(scenario_id: str) -> dict[str, Any]:
    """Les paramètres épidémiologiques que le scénario tient déjà, s'il en a.

    Ils vivent LÀ OÙ le SEIR les lit, dans le model_spec du scénario, et nulle part
    ailleurs : une proposition acceptée doit changer la projection, pas se poser à
    côté dans un champ que rien ne consulte. Absents, toute grandeur citée dans une
    réponse est une proposition NOUVELLE plutôt qu'une mise à jour."""
    spec = (_settings_variables(scenario_id).get("model_spec") or {})
    epi = spec.get("epidemic_parameters") or {}
    params = epi.get("params")
    return params if isinstance(params, dict) else {}


class ProposalDecision(BaseModel):
    key: str = Field(..., min_length=1, max_length=64)
    decision: str = Field(..., pattern="^(accepted|rejected)$")


@app.post("/user-scenarios/{scenario_id}/questions/{question_id}/proposals")
def decide_proposal(scenario_id: str, question_id: int, payload: ProposalDecision,
                    _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Accepte ou rejette UNE proposition tirée d'une réponse.

    Accepter écrit la valeur dans les paramètres du scénario, en notant d'où elle
    vient : la provenance est ce qui distingue un paramètre d'un nombre."""
    question = get_scenario_question(scenario_id, question_id)
    proposals = question.get("proposals") or []
    target = next((p for p in proposals if p.get("key") == payload.key), None)
    if target is None:
        raise HTTPException(status_code=404, detail=f"Proposition '{payload.key}' non trouvée")
    target["decision"] = payload.decision
    target["decided_at"] = datetime.now(timezone.utc).isoformat()

    if payload.decision == "accepted":
        # On écrit DANS le model_spec, à l'endroit que la projection SEIR lit. La
        # provenance accompagne la valeur : c'est ce qui distingue un paramètre d'un
        # nombre, et ce qui permet de défaire la décision en la relisant.
        blob = _settings_variables(scenario_id)
        spec = blob.get("model_spec")
        if not isinstance(spec, dict):
            raise HTTPException(
                status_code=409,
                detail="Ce scénario n'a pas encore de spécification de modèle : "
                       "générez les variables avant d'adopter un paramètre.")
        epi = spec.setdefault("epidemic_parameters", {"applicable": True, "disease": None,
                                                      "params": {}})
        if not isinstance(epi.get("params"), dict):
            epi["params"] = {}
        epi["applicable"] = True
        epi["params"][payload.key] = {
            "value": target.get("value"), "low": target.get("low"),
            "high": target.get("high"), "unit": target.get("unit") or None,
            "source": "assistant_answer", "question_id": question_id,
            "quote": target.get("quote"),
            "adopted_at": target["decided_at"],
        }
        blob["model_spec"] = spec
        try:
            with engine.begin() as conn:
                conn.execute(text("""
                    INSERT INTO scenario_settings (scenario_id, variables_json)
                    VALUES (:sid, CAST(:blob AS jsonb))
                    ON CONFLICT (scenario_id) DO UPDATE
                        SET variables_json = CAST(:blob AS jsonb)
                """), {"sid": scenario_id, "blob": json.dumps(blob)})
        except Exception as e:                                    # noqa: BLE001
            logger.warning(f"decide_proposal store {scenario_id}: {e}")
            raise HTTPException(status_code=500,
                                detail="Paramètre non enregistré") from e

    with engine.begin() as conn:
        conn.execute(text(
            "UPDATE scenario_question SET proposals_json = CAST(:p AS jsonb) WHERE id = :qid"),
            {"p": json.dumps(proposals), "qid": question_id})
    return {"scenario_id": scenario_id, "question_id": question_id,
            "key": payload.key, "decision": payload.decision, "proposals": proposals}


# ── Export d'une réponse ─────────────────────────────────────────────────────

def question_markdown(q: dict[str, Any], scenario_name: str = "") -> str:
    """Une réponse en Markdown, avec sa portée, sa date et ses sources.

    C'est le format pivot : le document Word et la page imprimable en descendent,
    de sorte qu'ils ne peuvent pas dire trois choses différentes. Pure."""
    out: list[str] = []
    out.append(f"# {q.get('question', '').strip()}")
    out.append("")
    meta = []
    if scenario_name:
        meta.append(scenario_name)
    if q.get("created_at"):
        meta.append(str(q["created_at"])[:19].replace("T", " "))
    meta.append(q.get("scope_label") or describe_scope(q.get("scope"), q.get("threshold")))
    out.append("*" + "  ·  ".join(meta) + "*")
    out.append("")
    used, retrieved = q.get("papers_used"), q.get("papers_quoted")
    if used is not None:
        line = f"Answered over {used} relevant articles"
        if retrieved is not None:
            line += f", {retrieved} of them retrieved for the answer"
        out.append(line + ".")
        out.append("")
    out.append(q.get("answer") or "")
    sources = q.get("sources") or []
    if sources:
        out.append("")
        out.append("## Sources")
        out.append("")
        for i, src in enumerate(sources, 1):
            bits = [str(src.get("title") or "Untitled").strip().rstrip(".")]
            if src.get("authors"):
                bits.insert(0, str(src["authors"]).strip().rstrip("."))
            if src.get("year"):
                bits.append(str(src["year"]))
            entry = ". ".join(b for b in bits if b)
            if src.get("doi"):
                entry += f". https://doi.org/{src['doi']}"
            out.append(f"{i}. {entry}")
    proposals = [p for p in (q.get("proposals") or []) if p.get("decision") != "rejected"]
    if proposals:
        out.append("")
        out.append("## Values this answer proposes")
        out.append("")
        for p in proposals:
            rng = (f" ({p['low']} to {p['high']})"
                   if p.get("low") is not None and p.get("high") is not None else "")
            was = (f", currently {p['current']}" if p.get("current") is not None
                   else ", not currently held")
            state = f" [{p['decision']}]" if p.get("decision") else ""
            out.append(f"- **{p['key']}**: {p.get('value')}{rng}"
                       f"{(' ' + p['unit']) if p.get('unit') else ''}{was}{state}")
    return "\n".join(out).rstrip() + "\n"


def _docx_bytes(markdown: str, title: str) -> bytes:
    """Un .docx minimal mais valide, écrit sans dépendance.

    Un fichier Word est un zip d'XML : pour un document de texte courant, l'écrire
    directement évite d'ajouter une bibliothèque à un service qui se déploie à
    chaque fusion. Les styles restent ceux de Word par défaut."""
    import io
    import zipfile
    from xml.sax.saxutils import escape

    def para(txt: str, style: str | None = None, bold: bool = False) -> str:
        pr = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style else ""
        rpr = "<w:rPr><w:b/></w:rPr>" if bold else ""
        return (f"<w:p>{pr}<w:r>{rpr}<w:t xml:space=\"preserve\">"
                f"{escape(txt)}</w:t></w:r></w:p>")

    body: list[str] = []
    for raw in markdown.split("\n"):
        line = raw.rstrip()
        if not line:
            body.append("<w:p/>")
        elif line.startswith("# "):
            body.append(para(line[2:], style="Heading1"))
        elif line.startswith("## "):
            body.append(para(line[3:], style="Heading2"))
        elif line.startswith("### "):
            body.append(para(line[4:], style="Heading3"))
        elif line.startswith("- "):
            body.append(para("• " + re.sub(r"\*\*(.+?)\*\*", r"\1", line[2:])))
        elif line.startswith("*") and line.endswith("*") and len(line) > 2:
            body.append(para(line.strip("*")))
        else:
            body.append(para(re.sub(r"\*\*(.+?)\*\*", r"\1", line)))

    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f'<w:body>{"".join(body)}'
        '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1134" w:right="1134" w:bottom="1134" w:left="1134"/>'
        '</w:sectPr></w:body></w:document>'
    )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-'
        'officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/docProps/core.xml" ContentType="application/vnd.openxmlformats-'
        'package.core-properties+xml"/>'
        '</Types>'
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/'
        'relationships/officeDocument" Target="word/document.xml"/>'
        '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/package/2006/'
        'relationships/metadata/core-properties" Target="docProps/core.xml"/>'
        '</Relationships>'
    )
    core = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties xmlns:cp="http://schemas.openxmlformats.org/package/2006/'
        'metadata/core-properties" xmlns:dc="http://purl.org/dc/elements/1.1/">'
        f'<dc:title>{escape(title)}</dc:title></cp:coreProperties>'
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", content_types)
        z.writestr("_rels/.rels", rels)
        z.writestr("docProps/core.xml", core)
        z.writestr("word/document.xml", document)
    return buf.getvalue()


_EXPORT_TYPES = {
    "md": "text/markdown; charset=utf-8",
    "docx": ("application/vnd.openxmlformats-officedocument."
             "wordprocessingml.document"),
    "pdf": "application/pdf",
}


def _slugify(value: str, limit: int = 60) -> str:
    import unicodedata
    folded = unicodedata.normalize("NFKD", value or "").encode("ascii", "ignore").decode()
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", folded).strip("-").lower()
    return (slug[:limit].rstrip("-")) or "question"


def _pdf_bytes(markdown: str, title: str) -> bytes:
    """Un PDF du même Markdown, via reportlab (déjà une dépendance du service)."""
    import io
    from reportlab.lib.enums import TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import mm
    from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer
    from xml.sax.saxutils import escape

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5,
                          leading=15, alignment=TA_LEFT, spaceAfter=6)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=17, leading=21,
                        spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=13, leading=17,
                        spaceBefore=10, spaceAfter=6)
    meta = ParagraphStyle("meta", parent=body, fontSize=9, textColor="#666666")

    def inline(txt: str) -> str:
        return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", escape(txt))

    flow: list[Any] = []
    for raw in markdown.split("\n"):
        line = raw.rstrip()
        if not line:
            flow.append(Spacer(1, 4))
        elif line.startswith("# "):
            flow.append(Paragraph(inline(line[2:]), h1))
        elif line.startswith("## "):
            flow.append(Paragraph(inline(line[3:]), h2))
        elif line.startswith("- "):
            flow.append(Paragraph("&bull; " + inline(line[2:]), body))
        elif line.startswith("*") and line.endswith("*") and len(line) > 2:
            flow.append(Paragraph(inline(line.strip("*")), meta))
        else:
            flow.append(Paragraph(inline(line), body))

    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4, title=title,
                      leftMargin=20 * mm, rightMargin=20 * mm,
                      topMargin=18 * mm, bottomMargin=18 * mm).build(flow)
    return buf.getvalue()


@app.get("/user-scenarios/{scenario_id}/questions/{question_id}/export")
def export_scenario_question(scenario_id: str, question_id: int,
                             format: str = "pdf") -> Response:
    """Une réponse comme document : Markdown, Word ou PDF.

    Les trois descendent du même Markdown, donc ils disent la même chose."""
    fmt = (format or "pdf").lower().strip()
    if fmt not in _EXPORT_TYPES:
        raise HTTPException(status_code=422,
                            detail=f"format doit être l'un de {', '.join(_EXPORT_TYPES)}")
    question = get_scenario_question(scenario_id, question_id)
    row = _get_user_scenario_or_404(scenario_id)
    md = question_markdown(question, scenario_name=row.get("name") or "")
    title = (question.get("question") or "question").strip()
    if fmt == "md":
        data = md.encode("utf-8")
    elif fmt == "docx":
        data = _docx_bytes(md, title)
    else:
        data = _pdf_bytes(md, title)
    name = f"{_slugify(row.get('name') or 'scenario', 32)}_{_slugify(title)}.{fmt}"
    return Response(content=data, media_type=_EXPORT_TYPES[fmt],
                    headers={"Content-Disposition": f'attachment; filename="{name}"'})
