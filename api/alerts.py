"""Email alert subscriptions and digests.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import json as _json
import os
import re
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .gesica import _get_scenario_name
from .scenario_store import _get_scenario_threshold

# ─── ALERTES EMAIL ────────────────────────────────────────────────────────────

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _clean_email(v) -> str | None:
    """Normalise (trim + minuscules) et valide une adresse email ; None si vide/invalide."""
    if not isinstance(v, str):
        return None
    v = v.strip().lower()
    return v if (v and _EMAIL_RE.match(v)) else None


def _ensure_alert_subscription(conn, email: str, scenario_id: str, frequency: str = "weekly") -> None:
    """Crée la table si besoin puis (ré)active l'abonnement email↔scénario (idempotent)."""
    conn.execute(text("""
        CREATE TABLE IF NOT EXISTS alert_subscriptions (
            id SERIAL PRIMARY KEY,
            email VARCHAR(255) NOT NULL,
            scenario_id VARCHAR(100) NOT NULL,
            frequency VARCHAR(20) DEFAULT 'weekly',
            created_at TIMESTAMP DEFAULT NOW(),
            last_notified_at TIMESTAMP DEFAULT NULL,
            is_active BOOLEAN DEFAULT TRUE,
            UNIQUE(email, scenario_id)
        )
    """))
    # `last_notified_at = NOW()` DÈS l'abonnement : le premier digest porte alors sur ce
    # qui est arrivé APRÈS l'inscription, ce que l'abonné attend. Laissé à NULL, « depuis
    # la dernière notification » n'avait pas de borne basse et le premier envoi comptait
    # le corpus ENTIER comme nouveau (2170 articles annoncés « nouveaux » sur un scénario
    # qui n'en avait pas gagné un seul depuis l'inscription).
    # COALESCE sur le conflit : on ne REMET PAS la pendule à zéro pour un abonnement déjà
    # actif (changer la fréquence sauterait tout ce qui est arrivé depuis son dernier
    # digest) ; on ne la pose que si elle manque.
    conn.execute(text("""
        INSERT INTO alert_subscriptions (email, scenario_id, frequency, last_notified_at)
        VALUES (:email, :scenario_id, :frequency, NOW())
        ON CONFLICT (email, scenario_id) DO UPDATE SET
            frequency = :frequency,
            is_active = TRUE,
            last_notified_at = COALESCE(alert_subscriptions.last_notified_at, NOW())
    """), {"email": email, "scenario_id": scenario_id, "frequency": frequency})


class AlertSubscriptionIn(BaseModel):
    email: str = Field(..., max_length=255)
    scenario_id: str = Field(..., min_length=1, max_length=100)
    frequency: str = "weekly"  # "daily" | "weekly" | "immediate"

    @field_validator("email")
    @classmethod
    def _validate_email(cls, v: str) -> str:
        v = v.strip()
        if not re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", v):
            raise ValueError("Adresse email invalide")
        return v

    @field_validator("frequency")
    @classmethod
    def _validate_frequency(cls, v: str) -> str:
        if v not in ("daily", "weekly", "immediate"):
            raise ValueError("frequency doit être 'daily', 'weekly' ou 'immediate'")
        return v


@app.post("/alerts/subscribe")
def subscribe_alerts(payload: AlertSubscriptionIn, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Enregistre une alerte email pour un scénario.

    ATTENTION, ce que cet endpoint fait vraiment : il ENREGISTRE l'abonnement, il
    n'envoie rien et ne programme rien. L'envoi est fait par `/alerts/run-digests`,
    qu'il faut appeler depuis un cron (ou un timer systemd) : l'API n'a pas de
    planificateur interne. Sans cron, ou sans `SMTP_HOST`, l'abonnement est bien
    stocké mais aucun email ne partira jamais. La réponse porte donc `delivery`, qui
    dit lequel des deux prérequis manque, plutôt que de promettre un envoi.
    """
    email = _clean_email(payload.email) or payload.email
    with engine.begin() as conn:
        _ensure_alert_subscription(conn, email, payload.scenario_id, payload.frequency)
        # S'abonner à SON scénario utilisateur = en devenir propriétaire (si aucun),
        # pour que la living review le crawle (run_user_scenarios filtre sur owner_email).
        owner_set = False
        if payload.scenario_id.startswith("usr-"):
            res = conn.execute(text(
                "UPDATE user_scenarios SET owner_email = :e, updated_at = NOW() "
                "WHERE id = :id AND (owner_email IS NULL OR owner_email = '')"
            ), {"e": email, "id": payload.scenario_id})
            owner_set = (res.rowcount or 0) > 0

    smtp_ok = bool(os.getenv("SMTP_HOST", ""))
    return {
        "status": "subscribed",
        "email": email,
        "scenario_id": payload.scenario_id,
        "frequency": payload.frequency,
        "owner_set": owner_set,
        # État RÉEL de la distribution, pas une promesse : l'abonnement est stocké dans
        # tous les cas, mais l'email ne part que si SMTP est configuré ET si
        # /alerts/run-digests est appelé par un cron (l'API ne planifie rien).
        "delivery": {
            "smtp_configured": smtp_ok,
            "requires_scheduled_runner": True,
            "runner_endpoint": "POST /alerts/run-digests",
        },
        "message": (
            f"Abonnement {payload.frequency} enregistré pour le scénario "
            f"'{payload.scenario_id}'. L'envoi dépend de l'appel périodique de "
            f"/alerts/run-digests"
            + ("" if smtp_ok else " et de la configuration de SMTP_HOST, absente ici")
            + "."
        ),
    }


@app.delete("/alerts/unsubscribe")
def unsubscribe_alerts(email: str, scenario_id: str, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Désabonnement des alertes email."""
    # subscribe stocke l'email NORMALISÉ (_clean_email : trim + minuscule). Comparer
    # l'email brut échouait silencieusement (0 ligne) tout en renvoyant "unsubscribed"
    # → l'utilisateur continuait de recevoir les digests.
    email = _clean_email(email)
    with engine.begin() as conn:
        res = conn.execute(text("""
            UPDATE alert_subscriptions
            SET is_active = FALSE
            WHERE email = :email AND scenario_id = :scenario_id
        """), {"email": email, "scenario_id": scenario_id})
        if res.rowcount == 0:
            raise HTTPException(status_code=404, detail="Abonnement non trouvé")
    return {"status": "unsubscribed", "email": email, "scenario_id": scenario_id}


@app.get("/alerts/subscriptions")
def list_subscriptions(email: str) -> list[dict[str, Any]]:
    """Liste les abonnements actifs pour un email."""
    email = _clean_email(email)   # même normalisation qu'à l'abonnement (sinon 0 résultat)
    try:
        with engine.connect() as conn:
            rows = conn.execute(text("""
                SELECT scenario_id, frequency, created_at, last_notified_at, is_active
                FROM alert_subscriptions
                WHERE email = :email AND is_active = TRUE
                ORDER BY created_at DESC
            """), {"email": email}).mappings().all()
        return [dict(r) for r in rows]
    except Exception:
        return []


def _new_articles_for_scenario(conn, scenario_id: str, since, limit: int = 25) -> list[dict]:
    """Articles du corpus d'un scénario ingérés APRÈS `since`. Couvre les deux modèles
    d'appartenance : préréglé (literature_document.scenario_type) ET utilisateur
    (article_scenarios). since=None (1re notification) → les plus récents (bornés)."""
    rows = conn.execute(text("""
        SELECT DISTINCT d.id, d.title, d.year, d.doi, d.url, d.created_at
        FROM literature_document d
        WHERE (d.scenario_type = :sid
               OR EXISTS (SELECT 1 FROM article_scenarios a
                          WHERE a.document_id = d.id AND a.scenario_id = :sid))
          AND (CAST(:since AS timestamp) IS NULL OR d.created_at > CAST(:since AS timestamp))
        ORDER BY d.created_at DESC
        LIMIT :lim
    """), {"sid": scenario_id, "since": since, "lim": int(limit)}).mappings().all()
    return [dict(r) for r in rows]


def _count_new_articles_for_scenario(conn, scenario_id: str, since) -> int:
    """Combien d'articles nouveaux AU TOTAL, sans la borne d'affichage.

    `_new_articles_for_scenario` en renvoie au plus 25, pour ne pas mettre 300 lignes
    dans un email. Ce total-ci est ce que l'email ANNONCE. Les confondre faisait écrire
    « 25 nouveaux articles » à un abonné dont le scénario en avait gagné 300, et rendait
    inatteignable la mention « et N de plus » du gabarit (elle compare le total au nombre
    de lignes affichées : si le total EST le nombre de lignes, elle ne s'affiche jamais).
    Mêmes conditions exactement que la requête ci-dessus."""
    return int(conn.execute(text("""
        SELECT COUNT(DISTINCT d.id)
        FROM literature_document d
        WHERE (d.scenario_type = :sid
               OR EXISTS (SELECT 1 FROM article_scenarios a
                          WHERE a.document_id = d.id AND a.scenario_id = :sid))
          AND (CAST(:since AS timestamp) IS NULL OR d.created_at > CAST(:since AS timestamp))
    """), {"sid": scenario_id, "since": since}).scalar() or 0)


# ── Ce que les nouveaux articles CHANGENT (signaux, sans LLM) ─────────────────
# Le digest disait combien d'articles étaient arrivés. Un relecteur veut savoir deux
# choses de plus, et ces deux-là décident s'il ouvre l'application ou non : combien de
# ces articles sont PERTINENTS (le compte brut inclut ce que le seuil écarte), et
# lesquels valent une relecture du modèle.
#
# Ces signaux sont calculés en SQL sur les faits DÉJÀ extraits par article (pico_json,
# concepts_json, study_design) : aucun LLM, aucun coût par envoi. Ils SIGNALENT, ils
# n'affirment jamais que le modèle a changé : seule une régénération peut le dire, et
# elle se demande explicitement. Un email quotidien qui annoncerait « votre SEIR a
# changé » sur la foi d'une expression régulière serait exactement le genre d'affirmation
# non tenue que ce projet retire partout ailleurs.

# Gate de pertinence : le MÊME prédicat que partout (jamais les exclus, inclus par un
# relecteur ou au-dessus du seuil).
_RELEVANT_GATE = """
      AND COALESCE(a.screening_status, d.screening_status) IS DISTINCT FROM 'excluded'
      AND (COALESCE(a.screening_status, d.screening_status) = 'included'
           OR COALESCE(a.similarity_score, 0) >= :thr)
"""

# Devis qui déplacent le niveau de preuve d'une revue : s'ils arrivent, le brief mérite
# une relecture même si tout le reste est inchangé.
_STRONG_DESIGNS = ("systematic review", "meta-analysis", "meta analysis",
                   "randomized controlled trial", "randomised controlled trial", "rct")


def _new_relevant_rows(conn, scenario_id: str, since, threshold: float) -> list[dict]:
    """Les articles NOUVEAUX ET PERTINENTS, avec les faits déjà extraits sur chacun."""
    return [dict(r) for r in conn.execute(text(f"""
        SELECT d.id, d.title, d.abstract, d.year, d.study_design, d.concepts_json
        FROM literature_document d
        JOIN article_scenarios a ON a.document_id = d.id AND a.scenario_id = :sid
        WHERE d.is_duplicate IS NOT TRUE
          AND (CAST(:since AS timestamp) IS NULL OR d.created_at > CAST(:since AS timestamp))
          {_RELEVANT_GATE}
        ORDER BY COALESCE(a.rerank_score, a.similarity_score, 0) DESC NULLS LAST, d.id
    """), {"sid": scenario_id, "since": since, "thr": float(threshold)}).mappings().all()]


def _known_concept_labels(scenario_id: str) -> set[str] | None:
    """Les concepts DÉJÀ sur la carte en cache, en minuscules. None si aucune carte :
    sans référence, « nouveau concept » ne veut rien dire et le signal se tait."""
    try:
        with engine.connect() as conn:
            row = conn.execute(text(
                "SELECT concept_graph_json FROM scenario_settings WHERE scenario_id = :sid"
            ), {"sid": scenario_id}).mappings().first()
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"alert signals, concept map {scenario_id}: {_e}")
        return None
    payload = (row or {}).get("concept_graph_json")
    if isinstance(payload, str):
        try:
            payload = _json.loads(payload)
        except Exception:
            payload = None
    if not isinstance(payload, dict) or not payload.get("nodes"):
        return None
    out: set[str] = set()
    for n in payload["nodes"]:
        lab = (n.get("label") or {}) if isinstance(n, dict) else {}
        for v in (lab.get("en"), lab.get("fr")):
            if v:
                out.add(str(v).strip().lower())
    return out or None


def change_signals(scenario_id: str, since, threshold: float) -> dict[str, Any]:
    """Ce qui, parmi les nouveaux articles pertinents, mérite une relecture.

    Renvoie `{"new_relevant": n, "signals": [{code, n, detail, tab}, ...]}`. Chaque signal
    est une INVITATION À VÉRIFIER, jamais un constat de changement. Jamais bloquant : une
    panne de calcul rend une liste vide, le digest part quand même."""
    out: dict[str, Any] = {"new_relevant": 0, "signals": []}
    try:
        with engine.connect() as conn:
            rows = _new_relevant_rows(conn, scenario_id, since, threshold)
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"alert signals {scenario_id}: {_e}")
        return out
    out["new_relevant"] = len(rows)
    if not rows:
        return out

    # 1. Paramètres épidémiologiques : un terme de MESURE dans le titre ou le résumé.
    #    Même détecteur que l'onglet des paramètres, donc les deux disent la même chose.
    try:
        from .variables import params_mentioned          # lazy: variables charge après alerts
        _params: dict[str, int] = {}
        _n_param_articles = 0
        for r in rows:
            found = params_mentioned(f"{r.get('title') or ''} {r.get('abstract') or ''}")
            if found:
                _n_param_articles += 1
                for p in found:
                    _params[p] = _params.get(p, 0) + 1
        if _n_param_articles:
            _top = ", ".join(k for k, _ in sorted(_params.items(), key=lambda kv: -kv[1])[:4])
            out["signals"].append({
                "code": "epidemic_parameters", "n": _n_param_articles, "tab": "variables",
                "detail": _top,
            })
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"alert signals, parameters {scenario_id}: {_e}")

    # 2. Concepts absents de la carte en cache.
    try:
        known = _known_concept_labels(scenario_id)
        if known is not None:
            fresh: dict[str, int] = {}
            for r in rows:
                cj = r.get("concepts_json")
                if isinstance(cj, str):
                    try:
                        cj = _json.loads(cj)
                    except Exception:
                        cj = None
                for c in ((cj or {}).get("concepts") or []) if isinstance(cj, dict) else []:
                    lab = str((c or {}).get("en") or "").strip()
                    if lab and lab.lower() not in known:
                        fresh[lab] = fresh.get(lab, 0) + 1
            if fresh:
                _top = ", ".join(k for k, _ in sorted(fresh.items(), key=lambda kv: -kv[1])[:4])
                out["signals"].append({
                    "code": "new_concepts", "n": len(fresh), "tab": "concepts", "detail": _top,
                })
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"alert signals, concepts {scenario_id}: {_e}")

    # 3. Devis qui relèvent le niveau de preuve.
    try:
        _strong: dict[str, int] = {}
        for r in rows:
            d = str(r.get("study_design") or "").strip()
            if d and d.lower() in _STRONG_DESIGNS:
                _strong[d] = _strong.get(d, 0) + 1
        if _strong:
            _top = ", ".join(f"{v} {k}" for k, v in sorted(_strong.items(), key=lambda kv: -kv[1])[:3])
            out["signals"].append({
                "code": "strong_designs", "n": sum(_strong.values()), "tab": "brief",
                "detail": _top,
            })
    except Exception as _e:                                   # noqa: BLE001
        logger.warning(f"alert signals, designs {scenario_id}: {_e}")
    return out


def _digest_is_due(frequency: str | None, last_notified, now) -> bool:
    """Un abonnement est-il dû ? immediate = toujours ; daily/weekly selon le délai
    depuis la dernière notification ; jamais notifié = dû."""
    from datetime import timedelta
    if last_notified is None:
        return True
    if (frequency or "weekly") == "immediate":
        return True
    delta = now - last_notified
    return delta >= (timedelta(days=1) if frequency == "daily" else timedelta(days=7))


def _render_alert_digest(scenario_id: str, articles: list[dict], total_new: int,
                         base_url: str = "https://literev-scenario.com",
                         scenario_name: str | None = None,
                         first_digest: bool = False,
                         n_relevant: int | None = None,
                         signals: list[dict] | None = None) -> tuple[str, str, str]:
    """(subject, html, text) d'un digest - liste les VRAIS nouveaux articles. Pur/testable.
    Utilise le NOM lisible du scénario (pas l'ID) et un lien PROFOND vers sa page
    (?scenario=<id>, ouvert directement par le front).

    `first_digest=True` : l'abonnement n'a JAMAIS été notifié, il n'y a donc pas de borne
    basse et rien n'est « nouveau » au sens de « depuis la dernière fois ». L'email dit
    alors ce qu'il est vraiment, un aperçu des articles les plus récents du scénario, au
    lieu d'annoncer le corpus entier comme une arrivée du jour.

    `n_relevant` : combien de ces articles passent le seuil de pertinence. Le total brut
    seul ne dit pas si trois d'entre eux comptent ou aucun.

    `signals` : ce qui mérite une relecture (paramètres épidémiologiques rapportés,
    concepts absents de la carte, devis qui relèvent le niveau de preuve). Formulés comme
    des INVITATIONS À VÉRIFIER : l'email ne peut pas savoir si le modèle a changé, seule
    une régénération le peut, et il le dit."""
    import html as _html
    from urllib.parse import quote as _q
    label = (scenario_name or scenario_id).strip() or scenario_id
    if first_digest:
        # Accord FR au singulier comme au pluriel, sans « (s) ».
        noun = "1 article récent" if total_new == 1 else f"{total_new} articles récents"
        subj = f"[LiteRev] Alertes activées - {label} : {noun}"
        intro = ("Première notification pour ce scénario : voici ses articles les plus "
                 "récents. Les prochaines ne porteront que sur les nouveautés.")
    else:
        # Accord FR : 1 → « nouvel article » ; ≥2 → « nouveaux articles ».
        _n = "nouvel article" if total_new == 1 else "nouveaux articles"
        noun = f"{total_new} {_n}"
        subj = f"[LiteRev] {noun} - {label}"
        intro = ""
    scen_url = f"{base_url}/?scenario={_q(scenario_id, safe='')}"

    def _row(a):
        title = _html.escape(str(a.get("title") or "Article"))
        yr = f" ({a['year']})" if a.get("year") else ""
        href = a.get("url") or (f"https://doi.org/{a['doi']}" if a.get("doi") else scen_url)
        return f'<li style="margin:4px 0"><a href="{_html.escape(str(href))}" style="color:#16a34a">{title}</a>{yr}</li>'

    # Combien passent le seuil, et ce qui mérite une relecture.
    _rel_txt = ""
    if n_relevant is not None and total_new:
        _rel_txt = (f"{n_relevant} sur {total_new} passent le seuil de pertinence."
                    if n_relevant != total_new else
                    f"Les {total_new} passent le seuil de pertinence.")
    # Accord singulier/pluriel écrit, pas de « (s) » : ces lignes partent dans un email
    # que des collègues lisent.
    _SIGNAL_LABELS = {
        "epidemic_parameters": (
            "article rapportant un paramètre épidémiologique ({detail})"
            " : la spécification du modèle mérite une relecture",
            "articles rapportant un paramètre épidémiologique ({detail})"
            " : la spécification du modèle mérite une relecture"),
        "new_concepts": (
            "concept absent de la carte ({detail})",
            "concepts absents de la carte ({detail})"),
        "strong_designs": (
            "devis qui relève le niveau de preuve ({detail})"
            " : le brief mérite une relecture",
            "devis qui relèvent le niveau de preuve ({detail})"
            " : le brief mérite une relecture"),
    }
    _sig_lines = []
    for sg in (signals or []):
        _tpl = _SIGNAL_LABELS.get(sg.get("code"))
        _n = sg.get("n")
        if not _tpl or not _n:
            continue
        _sig_lines.append(f"{_n} " + _tpl[0 if _n == 1 else 1]
                          .replace("{detail}", str(sg.get("detail") or "")))
    # La phrase qui empêche l'email de se lire comme un constat. Elle accompagne TOUJOURS
    # les signaux : ils sont calculés sans LLM, sur des faits déjà extraits, et ne peuvent
    # pas établir qu'une conclusion a changé.
    _sig_caveat = ("Ces points sont des pistes de relecture, pas un constat : "
                   "seule une régénération du brief, des variables ou du SEIR peut dire "
                   "si une conclusion change.")

    shown = articles[:25]
    # `noun` porte DÉJÀ le nombre : le répéter donnait « 30 30 nouveaux articles ».
    more = f'<p style="font-size:12px;color:#6b7280">… et {total_new - len(shown)} de plus.</p>' if total_new > len(shown) else ""
    _intro_html = (f'<p style="font-size:13px;color:#4b5563">{_html.escape(intro)}</p>'
                   if intro else "")
    html_body = (
        '<html><body style="font-family:system-ui,Arial,sans-serif;color:#111">'
        f'<h2 style="color:#14532d">LiteRev - {noun}</h2>'
        f'<p>Scénario <strong>{_html.escape(label)}</strong> :</p>'
        f'{_intro_html}'
        + (f'<p style="font-size:13px;color:#166534;font-weight:600">{_html.escape(_rel_txt)}</p>'
           if _rel_txt else "")
        + (('<div style="margin:10px 0;padding:10px 12px;border-left:3px solid #f59e0b;'
            'background:#fffbeb">'
            '<p style="margin:0 0 6px;font-size:12px;font-weight:700;color:#92400e">'
            'À relire</p><ul style="margin:0;padding-left:18px">'
            + "".join(f'<li style="font-size:12px;color:#78350f">{_html.escape(x)}</li>'
                      for x in _sig_lines)
            + f'</ul><p style="margin:6px 0 0;font-size:11px;color:#a16207">'
              f'{_html.escape(_sig_caveat)}</p></div>')
           if _sig_lines else "")
        + f'<ul>{"".join(_row(a) for a in shown)}</ul>{more}'
        f'<p><a href="{_html.escape(scen_url)}" style="color:#16a34a;font-weight:600">Ouvrir le scénario →</a></p>'
        '<hr><p style="font-size:11px;color:#6b7280">Vous recevez cet email car vous êtes abonné aux alertes LiteRev pour ce scénario.</p>'
        '</body></html>'
    )
    text_body = (f"LiteRev - {noun} pour le scénario « {label} » :\n\n"
                 + (f"{intro}\n\n" if intro else "")
                 + (f"{_rel_txt}\n\n" if _rel_txt else "")
                 + (("A RELIRE :\n" + "\n".join(f"- {x}" for x in _sig_lines)
                     + f"\n{_sig_caveat}\n\n") if _sig_lines else "")
                 + "\n".join(f"- {a.get('title', '')}" + (f" ({a['year']})" if a.get("year") else "") for a in shown)
                 + (f"\n… et {total_new - len(shown)} de plus." if total_new > len(shown) else "")
                 + f"\n\n{scen_url}\n")
    return subj, html_body, text_body


def _smtp_mode_for(port, security: str | None) -> str:
    """Mode de connexion SMTP : 'ssl' | 'starttls' | 'none'. Un SMTP_SECURITY
    explicite l'emporte ; sinon on l'infère du port (587/25 → STARTTLS, sinon SSL).
    Couvre GoDaddy Professional Email (smtpout.secureserver.net:465 SSL) ET
    Microsoft 365 (smtp.office365.com:587 STARTTLS)."""
    mode = (security or "").strip().lower()
    if mode in ("ssl", "starttls", "none"):
        return mode
    try:
        p = int(port)
    except (TypeError, ValueError):
        p = 465
    return "starttls" if p in (587, 25) else "ssl"


def _send_email_smtp(host: str, user: str, pw: str, to: str, subject: str, html_body: str, text_body: str,
                     port=None, security: str | None = None) -> None:
    """Envoi SMTP, port/mode configurables (env SMTP_PORT / SMTP_SECURITY ; défaut
    465 SSL). Lève en cas d'échec (l'appelant journalise)."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    msg = MIMEMultipart("alternative")
    msg["Subject"], msg["From"], msg["To"] = subject, user, to
    msg.attach(MIMEText(text_body, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    port = int(port or os.getenv("SMTP_PORT") or 465)
    mode = _smtp_mode_for(port, security if security is not None else os.getenv("SMTP_SECURITY"))
    if mode == "ssl":
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            if user:
                server.login(user, pw)
            server.sendmail(user, to, msg.as_string())
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.ehlo()
            if mode == "starttls":
                server.starttls()
                server.ehlo()
            if user:
                server.login(user, pw)
            server.sendmail(user, to, msg.as_string())


def _process_alert_digests(scenario_id: str | None, dry_run: bool, respect_frequency: bool) -> dict[str, Any]:
    """Cœur commun de l'envoi des digests : pour chaque abonnement actif (et dû si
    respect_frequency), calcule les NOUVEAUX articles depuis last_notified_at, envoie
    l'email (sauf dry_run) et met à jour last_notified_at. Ne notifie pas si 0 nouveauté."""
    from datetime import datetime
    smtp_host, smtp_user, smtp_pass = os.getenv("SMTP_HOST", ""), os.getenv("SMTP_USER", ""), os.getenv("SMTP_PASS", "")
    # NAÏF (pas tz-aware) : `last_notified_at` est une colonne TIMESTAMP (sans fuseau)
    # → SQLAlchemy renvoie un datetime naïf ; soustraire un `now` aware levait
    # TypeError dans _digest_is_due dès le 2e passage (tous les digests daily/weekly
    # cessaient d'être envoyés). Cohérent avec les timestamps naïfs de la base.
    now = datetime.utcnow()
    try:
        with engine.connect() as conn:
            q = "SELECT id, email, scenario_id, frequency, last_notified_at FROM alert_subscriptions WHERE is_active = TRUE"
            params: dict[str, Any] = {}
            if scenario_id:
                q += " AND scenario_id = :sid"
                params["sid"] = scenario_id
            subs = [dict(r) for r in conn.execute(text(q), params).mappings().all()]
    except Exception as e:
        return {"status": "error", "error": str(e), "processed": 0, "sent": 0}

    results, sent = [], 0
    for sub in subs:
        if respect_frequency and not _digest_is_due(sub.get("frequency"), sub.get("last_notified_at"), now):
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "skipped": "not_due"})
            continue
        # Jamais notifié : pas de borne basse, donc RIEN n'est « nouveau depuis la
        # dernière fois ». Le digest est un aperçu des articles les plus récents, et son
        # total est ce qu'il montre. Compter le corpus entier faisait annoncer « 2170
        # nouveaux articles » sur un scénario qui n'en avait pas gagné un seul depuis
        # l'inscription. Les abonnements créés à partir de maintenant partent avec
        # last_notified_at = NOW(), donc ce cas ne concerne que les lignes héritées.
        _since = sub.get("last_notified_at")
        _first = _since is None
        try:
            with engine.connect() as conn:
                arts = _new_articles_for_scenario(conn, sub["scenario_id"], _since)
                # Le TOTAL, pas le nombre de lignes affichées : c'est lui que l'email annonce.
                total_new = (len(arts) if _first else
                             _count_new_articles_for_scenario(conn, sub["scenario_id"], _since))
        except Exception as e:
            logger.warning(f"digest new-articles {sub['scenario_id']}: {e}")
            arts, total_new = [], 0
        # Combien de ces articles PASSENT le seuil, et ce qui mérite une relecture.
        # « 300 nouveaux articles » sans cette ligne ne dit pas si trois d'entre eux
        # comptent ou aucun : c'est le seul chiffre sur lequel un relecteur décide
        # d'ouvrir l'application un matin chargé.
        try:
            _thr = _get_scenario_threshold(sub["scenario_id"])
            _sig = change_signals(sub["scenario_id"], _since, _thr)
        except Exception as e:                                # noqa: BLE001 - jamais bloquant
            logger.warning(f"digest signals {sub['scenario_id']}: {e}")
            _sig = {"new_relevant": None, "signals": []}
        if not arts:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": 0, "sent": False})
            continue
        if dry_run:
            # `would_send` : la question que l'aperçu doit trancher. Sans elle, le dry run
            # court-circuitait AVANT le contrôle SMTP et répondait « dry_run » aussi bien
            # sur un serveur qui enverra que sur un serveur qui n'enverra jamais rien : le
            # seul prérequis qu'on venait vérifier était le seul qu'il taisait.
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"],
                            "new": total_new, "listed": len(arts), "sent": False,
                            "reason": "dry_run", "would_send": bool(smtp_host),
                            "first_digest": _first,
                            "new_relevant": _sig.get("new_relevant"),
                            "signals": _sig.get("signals") or []})
            continue
        if not smtp_host:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"],
                            "new": total_new, "listed": len(arts), "sent": False,
                            "reason": "smtp_not_configured"})
            continue
        try:
            _scen_name = _get_scenario_name(sub["scenario_id"])
        except Exception:
            _scen_name = None
        # `total_new` et NON len(arts) : le gabarit compare le total au nombre de lignes
        # pour écrire « et N de plus ». Lui passer le nombre de lignes comme total rendait
        # cette branche morte (N valait toujours 0) et faisait annoncer 25 au lieu de 300.
        subj, html_body, text_body = _render_alert_digest(
            sub["scenario_id"], arts, total_new, scenario_name=_scen_name,
            first_digest=_first, n_relevant=_sig.get("new_relevant"),
            signals=_sig.get("signals") or [])
        try:
            _send_email_smtp(smtp_host, smtp_user, smtp_pass, sub["email"], subj, html_body, text_body)
            with engine.begin() as conn:
                conn.execute(text("UPDATE alert_subscriptions SET last_notified_at = NOW() WHERE id = :id"), {"id": sub["id"]})
            sent += 1
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"],
                            "new": total_new, "listed": len(arts), "sent": True,
                            "first_digest": _first,
                            "new_relevant": _sig.get("new_relevant")})
        except Exception as e:
            logger.error(f"digest email {sub['email']}: {e}")
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"],
                            "new": total_new, "listed": len(arts), "sent": False,
                            "reason": f"error: {e}"})

    status = "sent" if (not dry_run and sent) else ("dry_run" if dry_run else ("not_configured" if not smtp_host else "no_new_articles"))
    return {"status": status, "subscriptions": len(subs), "processed": len(results), "sent": sent,
            "dry_run": dry_run,
            # Au niveau de la réponse aussi : un aperçu qui ne dit pas si SMTP est
            # configuré ne permet pas de décider s'il est sûr d'activer le cron.
            "smtp_configured": bool(smtp_host), "results": results}


@app.post("/alerts/send-digest")
def send_alert_digest(scenario_id: str | None = None, dry_run: bool = True,
                      _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Envoie (ou simule en dry_run) les digests aux abonnés d'un scénario (ou de tous) :
    liste les VRAIS nouveaux articles depuis la dernière notification et met à jour
    last_notified_at. Ignore les abonnements sans nouveauté. Ignore la fréquence
    (envoi forcé) - pour la cadence automatique, voir /alerts/run-digests."""
    return _process_alert_digests(scenario_id, dry_run=dry_run, respect_frequency=False)


@app.post("/alerts/run-digests")
def run_due_alert_digests(dry_run: bool = False, _: None = Depends(require_api_key)) -> dict[str, Any]:
    """Traite tous les abonnements DUS selon leur fréquence (daily/weekly/immediate).
    Appelable par cron après la living review (ex. `curl -XPOST .../alerts/run-digests`)
    pour notifier les utilisateurs des nouveaux articles de LEURS scénarios."""
    return _process_alert_digests(None, dry_run=dry_run, respect_frequency=True)
