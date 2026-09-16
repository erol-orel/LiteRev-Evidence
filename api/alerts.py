"""Email alert subscriptions and digests.

Extracted from main.py (LiteRev API); `main` re-exports everything for the scripts,
tools and tests.
"""
from __future__ import annotations

import os
import re
from typing import Any

from fastapi import Depends, HTTPException
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import text

from .core import app, engine, logger, require_api_key
from .gesica import _get_scenario_name

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
    conn.execute(text("""
        INSERT INTO alert_subscriptions (email, scenario_id, frequency)
        VALUES (:email, :scenario_id, :frequency)
        ON CONFLICT (email, scenario_id) DO UPDATE SET frequency = :frequency, is_active = TRUE
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
    """
    Enregistre une alerte email pour un scénario.
    L'utilisateur sera notifié quand de nouveaux articles sont ajoutés.
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

    return {
        "status": "subscribed",
        "email": email,
        "scenario_id": payload.scenario_id,
        "frequency": payload.frequency,
        "owner_set": owner_set,
        "message": f"Vous recevrez des alertes {payload.frequency} pour le scénario '{payload.scenario_id}'.",
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
                         scenario_name: str | None = None) -> tuple[str, str, str]:
    """(subject, html, text) d'un digest - liste les VRAIS nouveaux articles. Pur/testable.
    Utilise le NOM lisible du scénario (pas l'ID) et un lien PROFOND vers sa page
    (?scenario=<id>, ouvert directement par le front)."""
    import html as _html
    from urllib.parse import quote as _q
    label = (scenario_name or scenario_id).strip() or scenario_id
    # Accord FR : 1 → « nouvel article » ; ≥2 → « nouveaux articles ».
    noun = "nouvel article" if total_new == 1 else "nouveaux articles"
    subj = f"[LiteRev] {total_new} {noun} - {label}"
    scen_url = f"{base_url}/?scenario={_q(scenario_id, safe='')}"

    def _row(a):
        title = _html.escape(str(a.get("title") or "Article"))
        yr = f" ({a['year']})" if a.get("year") else ""
        href = a.get("url") or (f"https://doi.org/{a['doi']}" if a.get("doi") else scen_url)
        return f'<li style="margin:4px 0"><a href="{_html.escape(str(href))}" style="color:#16a34a">{title}</a>{yr}</li>'

    shown = articles[:25]
    more = f'<p style="font-size:12px;color:#6b7280">… et {total_new - len(shown)} de plus.</p>' if total_new > len(shown) else ""
    html_body = (
        '<html><body style="font-family:system-ui,Arial,sans-serif;color:#111">'
        f'<h2 style="color:#14532d">LiteRev - {total_new} {noun}</h2>'
        f'<p>Scénario <strong>{_html.escape(label)}</strong> :</p>'
        f'<ul>{"".join(_row(a) for a in shown)}</ul>{more}'
        f'<p><a href="{_html.escape(scen_url)}" style="color:#16a34a;font-weight:600">Ouvrir le scénario →</a></p>'
        '<hr><p style="font-size:11px;color:#6b7280">Vous recevez cet email car vous êtes abonné aux alertes LiteRev pour ce scénario.</p>'
        '</body></html>'
    )
    text_body = (f"LiteRev - {total_new} {noun} pour le scénario « {label} » :\n\n"
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
        try:
            with engine.connect() as conn:
                arts = _new_articles_for_scenario(conn, sub["scenario_id"], sub.get("last_notified_at"))
        except Exception as e:
            logger.warning(f"digest new-articles {sub['scenario_id']}: {e}")
            arts = []
        if not arts:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": 0, "sent": False})
            continue
        if dry_run:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": "dry_run"})
            continue
        if not smtp_host:
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": "smtp_not_configured"})
            continue
        try:
            _scen_name = _get_scenario_name(sub["scenario_id"])
        except Exception:
            _scen_name = None
        subj, html_body, text_body = _render_alert_digest(sub["scenario_id"], arts, len(arts), scenario_name=_scen_name)
        try:
            _send_email_smtp(smtp_host, smtp_user, smtp_pass, sub["email"], subj, html_body, text_body)
            with engine.begin() as conn:
                conn.execute(text("UPDATE alert_subscriptions SET last_notified_at = NOW() WHERE id = :id"), {"id": sub["id"]})
            sent += 1
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": True})
        except Exception as e:
            logger.error(f"digest email {sub['email']}: {e}")
            results.append({"email": sub["email"], "scenario_id": sub["scenario_id"], "new": len(arts), "sent": False, "reason": f"error: {e}"})

    status = "sent" if (not dry_run and sent) else ("dry_run" if dry_run else ("not_configured" if not smtp_host else "no_new_articles"))
    return {"status": status, "subscriptions": len(subs), "processed": len(results), "sent": sent, "dry_run": dry_run, "results": results}


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
