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
