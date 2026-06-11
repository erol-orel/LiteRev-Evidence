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

Chaque push/merge sur `main` déclenche `.github/workflows/deploy.yml` :

1. **CI** : vérification syntaxe Python (`main.py`, `backend_additions.py`, `app/`) + build frontend (`npm ci && npm run build`). Les pull requests vers `main` exécutent la CI seule (pas de déploiement).
2. **Deploy** : connexion SSH au serveur app et exécution de `deploy.sh` (git pull, build, copie nginx, restart `literev-api`).

Déclenchement manuel possible : onglet *Actions* → *CI / Deploy* → *Run workflow*.

#### Configuration initiale (une seule fois)

1. Générer une clé SSH dédiée au déploiement (sur votre machine) :

   ```bash
   ssh-keygen -t ed25519 -f ~/.ssh/literev_deploy -N "" -C "github-actions-deploy"
   ```

2. Installer la clé publique sur le serveur app, restreinte au seul script de déploiement :

   ```bash
   ssh root@62.238.39.50 "echo 'command=\"cd /opt/literev-api && bash deploy.sh\",no-port-forwarding,no-agent-forwarding,no-X11-forwarding $(cat ~/.ssh/literev_deploy.pub)' >> /root/.ssh/authorized_keys"
   ```

   La restriction `command="..."` garantit que cette clé ne peut QUE déployer, rien d'autre.

3. Dans GitHub → repo → **Settings → Secrets and variables → Actions → New repository secret**, créer 3 secrets :

   | Secret | Valeur |
   |--------|--------|
   | `DEPLOY_SSH_KEY` | contenu du fichier `~/.ssh/literev_deploy` (clé privée, bloc complet `-----BEGIN...END-----`) |
   | `DEPLOY_HOST` | `62.238.39.50` |
   | `DEPLOY_USER` | `root` |

### Manuel (secours)

```bash
cd /opt/literev-api && ./deploy.sh
```

## Stack

- **Backend** : FastAPI + SQLAlchemy + pgvector (Python 3.10)
- **Frontend** : React 19 + Vite 8 + Tailwind 3 + TypeScript
- **DB** : PostgreSQL 15 + pgvector
- **Web** : Nginx (reverse proxy + static files)
- **Embeddings** : BGE-M3

## Tables DB

| Table | Description |
|-------|-------------|
| `literature_document` | Documents indexés (96 docs, 3 projets) |
| `document_chunk` | Chunks vectorisés |
| `alembic_version` | Migrations |
