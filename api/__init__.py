"""LiteRev API, one module per domain. Import order matters: the startup DDL of each
module runs when it is imported, in the order MODULES lists them, and modules only
import from the ones before them at module level (later ones are imported lazily inside
functions). `main` at the repository root is the composition root and entry point.
"""
MODULES = (
    "core",
    "documents",
    "scenario_store",
    "schema_boot",
    "system",
    "search",
    "sources",
    "corpus",
    "enrichment",
    "gesica",
    "terrain",
    "living_review",
    "clustering",
    "knowledge_graph",
    "double_blind",
    "alerts",
    "scenarios",
    "pipeline",
    "relevance",
    "exports",
    "review",
    "evidence",
    "assistant",
    "variables",
    "model_spec",
    "actions",
    "model_data",
    "seir",
    "situation_reports",
    "model_training",
    "gesica_routes",
)
