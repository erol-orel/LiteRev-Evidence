"""Aucun nom indéfini dans le backend.

Trouvé sur le scénario HPAI_last de production : le pipeline complet rangeait trois de
ses étapes en « error » à chaque exécution.

    cross_encoder  name '_run_cross_encoder_rerank' is not defined   (depuis #322)
    pico           name '_llm_lang_directive' is not defined         (depuis #278)
    metadata       name '_llm_lang_directive' is not defined         (depuis #278)

Les deux fonctions existent (api/relevance.py, api/documents.py) mais n'étaient importées
nulle part dans api/pipeline.py. Python ne résout un nom global qu'au moment où la ligne
s'exécute : `compileall` passe, l'import du module passe, et aucun test ne faisait tourner
ces étapes jusqu'à la ligne fautive. Chaque étape étant enveloppée dans son propre try,
le pipeline continuait et livrait le brief, les variables et les actions recommandées
avec le statut « done ». Le PICO, extrait aussi en arrière-plan à l'ingestion, ne
manquait guère ; les métadonnées (type d'étude, effectif, risque de biais), qui n'ont pas
d'autre chemin automatique, ne couvraient que 23 % des articles de HPAI_last, et les
niveaux de preuve du brief se calculaient sur ce quart.

Ce test passe tout le paquet au crible de pyflakes et n'en retient que les noms
indéfinis : les imports inutilisés de `main.py` sont des ré-exports voulus, et les
variables non lues ne cassent rien. pyflakes est installé en CI ; en local, sans lui,
le test est sauté plutôt que de mentir.
"""
import pathlib

import pytest

pyflakes_checker = pytest.importorskip("pyflakes.checker")
pyflakes_messages = pytest.importorskip("pyflakes.messages")

ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Tout ce qui tourne en production : le paquet, son point d'entrée, et les deux modules
#: racine qu'il importe.
FILES = sorted(ROOT.glob("api/*.py")) + [ROOT / "main.py", ROOT / "llm_usage.py", ROOT / "env_files.py"]

UNDEFINED = (pyflakes_messages.UndefinedName, pyflakes_messages.UndefinedLocal,
             pyflakes_messages.UndefinedExport)


def _undefined_names(path: pathlib.Path) -> list[str]:
    import ast
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    checker = pyflakes_checker.Checker(tree, filename=str(path))
    return [f"{path.relative_to(ROOT)}:{m.lineno}: {m.message % m.message_args}"
            for m in checker.messages if isinstance(m, UNDEFINED)]


def test_the_files_under_check_exist():
    """Une liste vide passerait toujours : le crible doit porter sur le vrai code."""
    assert len(FILES) > 20 and all(f.exists() for f in FILES), FILES


@pytest.mark.parametrize("path", FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_no_name_is_used_without_being_defined(path):
    found = _undefined_names(path)
    assert not found, (
        "nom(s) indéfini(s) : la ligne lèvera un NameError le jour où elle s'exécutera\n  "
        + "\n  ".join(found))


def test_the_guard_catches_the_bug_it_was_written_for():
    """Le crible voit bien la forme exacte du défaut : un nom défini dans un autre module
    du paquet, utilisé sans import à l'intérieur d'une fonction."""
    import ast
    src = ("def step(lang):\n"
           "    return 'prompt' + _llm_lang_directive(lang)\n")
    checker = pyflakes_checker.Checker(ast.parse(src), filename="probe.py")
    assert any(isinstance(m, pyflakes_messages.UndefinedName) for m in checker.messages)
