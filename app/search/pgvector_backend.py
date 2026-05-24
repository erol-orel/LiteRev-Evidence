from __future__ import annotations

from typing import Any

from sqlalchemy import text


class PgvectorSearchBackend:
    def __init__(self, db_engine, embedder):
        self.db_engine = db_engine
        self.embedder = embedder

    def _embed_query(self, query: str) -> list[float]:
        if hasattr(self.embedder, "encode_query"):
            vec = self.embedder.encode_query(query)
        else:
            vec = self.embedder.encode(query)
        return vec.tolist()

    def search(
        self,
        query: str,
        filters: dict[str, Any] | None = None,
        mode: str = "semantic",
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        filters = filters or {}
        query_embedding = self._embed_query(query)

        where_clauses = []
        params: dict[str, Any] = {
            "query_embedding": str(query_embedding),
            "limit": limit,
        }

        field_map = {
            "source_type": "d.source_type",
            "disease_or_condition": "d.disease_or_condition",
            "scenario_type": "d.scenario_type",
            "geographic_scope": "d.geographic_scope",
            "evidence_category": "d.evidence_category",
            "source": "d.source",
            "year": "d.year",
        }

        for key, value in filters.items():
            if value in (None, ""):
                continue
            if key not in field_map:
                continue
            where_clauses.append(f"{field_map[key]} = :{key}")
            params[key] = value

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = text(f"""
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.chunk_type,
                c.section_label,
                c.char_start,
                c.char_end,
                c.token_count,
                c.chunk_weight,
                c.metadata_json,
                d.title,
                d.abstract,
                d.year,
                d.url,
                d.external_id,
                d.source,
                d.project_context,
                d.source_type,
                d.disease_or_condition,
                d.scenario_type,
                d.geographic_scope,
                d.evidence_category,
                1 - (c.embedding <=> CAST(:query_embedding AS vector)) AS score
            FROM document_chunk c
            JOIN literature_document d ON d.id = c.document_id
            {where_sql}
            ORDER BY c.embedding <=> CAST(:query_embedding AS vector)
            LIMIT :limit
        """)

        with self.db_engine.connect() as conn:
            rows = conn.execute(sql, params).mappings().all()

        return [dict(row) for row in rows]

    def get_filter_options(self) -> dict[str, list[dict[str, Any]]]:
        fields = {
            "source": "source",
            "source_type": "source_type",
            "disease_or_condition": "disease_or_condition",
            "scenario_type": "scenario_type",
            "geographic_scope": "geographic_scope",
            "evidence_category": "evidence_category",
            "year": "year",
        }

        label_overrides = {
            "pubmed": "PubMed",
            "pmc": "PubMed Central",
            "pubmedcentral": "PubMed Central",
            "pubmed_central": "PubMed Central",
            "openalex": "OpenAlex",
            "crossref": "Crossref",
            "systematic_review": "Systematic review",
            "meta_analysis": "Meta-analysis",
            "genomic_epidemiology": "Genomic epidemiology",
            "outbreak_detection": "Outbreak detection",
            "epidemic_intelligence": "Epidemic intelligence",
            "hospital_surveillance": "Hospital surveillance",
            "wastewater_surveillance": "Wastewater surveillance",
            "observational_study": "Observational study",
            "interventional_study": "Interventional study",
            "surveillance_report": "Surveillance report",
            "book_chapter": "Book chapter",
            "conference_paper": "Conference paper",
            "covid-19": "COVID-19",
            "hiv": "HIV",
            "usa": "USA",
        }

        out: dict[str, list[dict[str, Any]]] = {}

        with self.db_engine.connect() as conn:
            for api_name, column_name in fields.items():
                sql = text(f"""
                    SELECT DISTINCT {column_name} AS value
                    FROM literature_document
                    WHERE source IN ('pubmed', 'pmc', 'openalex', 'crossref')
                      AND {column_name} IS NOT NULL
                      AND TRIM(CAST({column_name} AS TEXT)) <> ''
                    ORDER BY value
                """)
                rows = conn.execute(sql).fetchall()

                values = []
                for row in rows:
                    value = row[0]
                    if value is None:
                        continue
                    if isinstance(value, int):
                        label = str(value)
                    else:
                        label = label_overrides.get(str(value), str(value).replace("_", " ").title())
                    values.append({"value": value, "label": label})

                out[api_name] = values

        return out

    def hybrid_search(
        self,
        query: str,
        boolean_filters: dict[str, Any] | None = None,
        vector_weight: float = 0.65,
        bm25_weight: float = 0.25,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        return self.search(
            query=query,
            filters=boolean_filters,
            mode="hybrid",
            limit=limit,
        )
