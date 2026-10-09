"""Le modèle dit ce qu'il fait tourner, et sur quoi.

Quatre affirmations que la moitié prédictive tenait sans les vérifier.

1. La période infectieuse était posée à 7 jours par le défaut d'un champ de dataclass,
   donc appliquée en silence. Elle fixe gamma, donc le jour du pic, sa hauteur, le taux
   de croissance et la durée de l'épidémie : un corpus qui rapporte un R0 sans période
   infectieuse - tous les scénarios de production qui portent une projection - voyait sa
   trajectoire entièrement décidée par une constante qu'aucun article n'a fournie, sous
   une étiquette « issue de la littérature ».

2. « Aucun paramètre épidémiologique extrait de la littérature », suivi d'un bloc de
   zéros, sur des corpus où 55 à 73 articles pertinents rapportent un R0. Les deux
   compteurs venaient de `variables_json._meta`, écrit par la seule génération et
   seulement quand elle ne réutilisait pas un spec existant ; `int(... or 0)` changeait
   « absent » en 0.

3. Un paramètre mesuré par UNE étude ne survivait que si la passe narrative avait aussi
   deviné un nombre, et c'était le nombre deviné qui était gardé.

4. « Prêt » et « utilisable » venaient de la présence d'un CHEMIN en base, pas d'un
   fichier sur le disque. Le dossier des modèles était sous la racine de déploiement, et
   `git clean -fd` de deploy.sh l'effaçait à chaque fusion sur main.
"""
import pytest

pytest.importorskip("fastapi")

import seir_model as sm  # noqa: E402


# ── 1. La période infectieuse ────────────────────────────────────────────────

def test_the_default_infectious_period_is_a_named_fallback_not_a_field_default():
    assert sm.SeirParams().infectious_period_days is None, (
        "le champ porte encore une valeur par défaut : elle s'appliquerait sans que "
        "rien ne le dise")
    assert sm.DEFAULT_INFECTIOUS_PERIOD_DAYS == 7.0


def test_the_projection_says_which_period_it_used_and_where_it_comes_from():
    assumed = sm.simulate(sm.SeirParams(r0=2.0), days=30)["summary"]
    assert assumed["infectious_period_source"] == "assumed"
    assert assumed["infectious_period_days"] == sm.DEFAULT_INFECTIOUS_PERIOD_DAYS
    given = sm.simulate(sm.SeirParams(r0=2.0, infectious_period_days=4), days=30)["summary"]
    assert given["infectious_period_source"] == "literature"
    assert given["infectious_period_days"] == 4.0


def test_the_assumed_period_changes_the_whole_trajectory():
    """Pourquoi la provenance compte : la constante ne décore pas, elle décide."""
    a = sm.simulate(sm.SeirParams(r0=2.0, infectious_period_days=4), days=200)["summary"]
    b = sm.simulate(sm.SeirParams(r0=2.0, infectious_period_days=14), days=200)["summary"]
    assert a["peak_incidence_day"] != b["peak_incidence_day"]
    assert a["peak_incidence"] != b["peak_incidence"]


def test_an_implausible_period_is_still_refused():
    for bad in (0.0, -1.0, 1e-9):
        with pytest.raises(ValueError):
            sm.simulate(sm.SeirParams(r0=2.0, infectious_period_days=bad), days=10)


# ── 3. Une seule règle pour tous les paramètres ──────────────────────────────

def test_a_single_observation_is_authoritative_and_carries_no_invented_interval():
    blk = {"applicable": True,
           "r0": {"value": 9.0, "provenance": [7],
                  "observations": [{"article_id": 7, "value": 1.8}]}}
    out = sm.normalize_extracted_parameters(blk, valid_ids={7}, quality_by_id={7: 0.6})
    r0 = out["params"]["r0"]
    assert r0["value"] == pytest.approx(1.8)
    assert r0["value_source"] == "pooled" and r0["n_studies"] == 1
    assert r0["ci_low"] is None and r0["ci_high"] is None


def test_two_observations_still_pool_with_an_interval():
    blk = {"applicable": True,
           "r0": {"value": None, "provenance": [1, 2],
                  "observations": [{"article_id": 1, "value": 2.0},
                                   {"article_id": 2, "value": 2.4}]}}
    out = sm.normalize_extracted_parameters(blk, valid_ids={1, 2}, quality_by_id={1: 1.0, 2: 1.0})
    r0 = out["params"]["r0"]
    assert r0["n_studies"] == 2 and r0["value_source"] == "pooled"
    assert r0["ci_low"] is not None and r0["ci_low"] < r0["value"] < r0["ci_high"]


# ── 4. « Prêt » veut dire que le fichier existe ──────────────────────────────

def test_the_model_data_directory_is_outside_the_deployment_tree():
    """`git clean -fd` dans /opt/literev-api effaçait tous les CSV et tous les modèles à
    chaque fusion sur main, pendant que les lignes survivaient avec leurs chemins."""
    from api.model_data import MODEL_DATA_DIR
    assert "/opt/literev-api" not in str(MODEL_DATA_DIR), (
        "le dossier des données est de nouveau dans l'arbre de déploiement, que le "
        "script de déploiement nettoie")


def test_the_data_directory_is_also_ignored_by_git_as_a_second_belt():
    """`git clean -fd` n'enlève pas les fichiers IGNORÉS (il faudrait -x) : lister
    l'ancien chemin protège les installations qui le gardent."""
    import pathlib
    gi = (pathlib.Path(__file__).resolve().parent.parent / ".gitignore").read_text()
    assert "uploads_datasets" in gi


# ── 2. Les compteurs du scanner de paramètres ────────────────────────────────

def test_a_count_that_was_never_measured_is_not_zero():
    """`articles_with_values` à None dit « jamais mesuré » ; 0 dirait « mesuré, rien
    trouvé ». Les deux se lisaient 0, et le panneau affirmait que la littérature ne
    rapporte aucun paramètre sur des corpus qui en rapportent des dizaines."""
    import inspect

    from api import seir as S
    src = inspect.getsource(S._seir_projection_payload)
    assert "values_never_extracted" in src
    assert "_parameter_candidate_articles" in src, (
        "le nombre d'articles qui mentionnent un paramètre est une question sur le "
        "corpus, sans LLM : elle doit être posée maintenant, pas lue dans un spec")
