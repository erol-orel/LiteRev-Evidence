# Infrastructure LiteRev

## Serveurs Hetzner

| Nom | IP publique | IP privée | Rôle |
|-----|-------------|-----------|------|
| literev-app-01 | 62.238.39.50 | — | API FastAPI + Frontend Nginx |
| literev-db-01 | 62.238.34.180 | 10.10.1.10 | PostgreSQL + pgvector |

## Connexion DB

```bash
PGPASSWORD='...' psql -h 10.10.1.10 -U literev -d literev
```

## Déploiement

### Automatique (CI/CD — GitHub Actions)

Deux workflows GitHub Actions :

1. **`deploy.yml`** — sur chaque PR : CI (syntaxe Python `compileall` + résolution `requirements.txt` + build frontend `npm ci && npm run build`). Sur chaque push/merge sur `main` : déploiement SSH (exécute `deploy.sh` : git pull, deps, migrations, build, bascule nginx atomique, restart + health check bloquant).
2. **`server-command.yml`** — déclenchement manuel (onglet *Actions* → *Server command (manual)* → *Run workflow*) pour lancer une commande ponctuelle et journalisée sur le serveur : `diagnose`, `migrate`, `restart`, `logs`, `deploy`, ou une commande libre (`custom`).

#### Configuration initiale (une seule fois)

1. **Clé SSH** : la clé publique `claude-code-session` est déjà installée dans `/root/.ssh/authorized_keys` sur app-01. La clé privée correspondante sert de secret GitHub. *(Pour une rotation propre, régénérer une paire `ssh-keygen -t ed25519`, remplacer la ligne dans `authorized_keys`, et mettre à jour le secret.)*

2. Dans GitHub → repo → **Settings → Secrets and variables → Actions → New repository secret**, créer 3 secrets :

   | Secret | Valeur |
   |--------|--------|
   | `DEPLOY_SSH_KEY` | clé privée correspondant à `claude-code-session` (bloc complet `-----BEGIN...END-----`) |
   | `DEPLOY_HOST` | `62.238.39.50` |
   | `DEPLOY_USER` | `root` |

   > La clé n'est volontairement PAS restreinte à `deploy.sh` (pas de `command=`) car le workflow manuel a besoin d'exécuter `migrate`/`logs`/`custom`. Toute commande passe donc par GitHub Actions et y est journalisée.

#### Secrets serveur (env)

Le service `literev-api` lit ses secrets depuis `/opt/literev-api/secrets.env` (mode 600), chargé par l'unité systemd. Variables attendues :

| Variable | Rôle |
|----------|------|
| `DB_URL` | `postgresql+psycopg://literev:<pass>@10.10.1.10:5432/literev` |
| `WRITE_API_KEY` | clé d'authentification des endpoints de mutation |
| `OPENAI_API_KEY` | embeddings RAG + triage LLM |
| `CDS_API_KEY` | (optionnel) Copernicus Climate Data Store |

Après modification : `systemctl restart literev-api`.

### Manuel (secours)

```bash
cd /opt/literev-api && ./deploy.sh
```

## Stack

- **Backend** : FastAPI + SQLAlchemy + pgvector (Python 3.10)
- **Frontend** : React 19 + Vite 8 + Tailwind 3 + TypeScript
- **DB** : PostgreSQL 15 + pgvector
- **Web** : Nginx (reverse proxy + static files)
- **Embeddings** : OpenAI `text-embedding-3-small`

## Tables DB

| Table | Description |
|-------|-------------|
| `literature_document` | Documents indexés (96 docs, 3 projets) |
| `document_chunk` | Chunks vectorisés |
| `alembic_version` | Migrations |
