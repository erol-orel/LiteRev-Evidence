"""Le PDF Evidence Brief répondait 500 sur le scénario HPAI, et sur lui seul.

Un Paragraph reportlab n'est pas du texte, c'est un mini-balisage : il lit `<i>`, `<b>`,
`<sub>`. Une balise NON FERMÉE ne s'affiche pas de travers, elle fait lever la
construction du document entier.

Le titre fautif, tel qu'il est en base :

    Infection Prevention and Control Strategies According to the Type of
    Multidrug-Resistant Bacteria and <i>Candida auris</i>

Le code tronquait à 120 caractères À L'INTÉRIEUR d'un `<b>...</b>`, ce qui coupait la
balise fermante en deux et produisait `<b>... and <i>Candida auris</</b>`. Un seul article
sur les 602 du scénario suffisait à casser l'export, et rien ne disait lequel.

Ces tests épinglent le comportement de `_pdf_text`, et surtout qu'un document contenant
ces titres se CONSTRUIT vraiment : la vérification à la main des balises ne vaut rien, la
seule preuve est que reportlab accepte.
"""
import io

import pytest

pytest.importorskip("reportlab")
pytest.importorskip("fastapi")

from reportlab.lib.pagesizes import A4  # noqa: E402
from reportlab.lib.styles import getSampleStyleSheet  # noqa: E402
from reportlab.platypus import Paragraph, SimpleDocTemplate  # noqa: E402

from api.evidence import _pdf_text  # noqa: E402

# Le vrai titre de l'article 133623, celui qui cassait l'export.
CANDIDA = ("Infection Prevention and Control Strategies According to the Type of "
           "Multidrug-Resistant Bacteria and <i>Candida auris</i>")

# Les autres formes réellement présentes dans le corpus HPAI.
REAL_TITLES = [
    CANDIDA,
    "Landscape of H5 Infections in ASEAN Region: Past Insights, Present Realities, &amp; Future Strategies.",
    "Research progress on &lt;i&gt;Avibacterium paragallinarum&lt;/i&gt; and related diseases in poultry",
    "Employing drug delivery strategies to create safe and effective pharmaceuticals for <scp>COVID</scp>-19",
    "&lt;p&gt;Knowledge, Attitudes, and Practice Regarding COVID-19 among Patients with Rheumatic Diseases",
    "Can <i>Echinacea</i> be a potential candidate to target immunity, inflammation, and infection",
    "<i>Ofeleein i mi Vlaptin</i>-Volume II: Immunity Following Infection or mRNA Vaccination",
    "Insights, Realities & Strategies",          # esperluette nue
    "Unclosed <i>italic that never ends",        # la forme fatale, directement
]


def _builds(*texts: str) -> bool:
    """Construit un vrai document. C'est la seule vérification qui prouve quoi que ce
    soit : `Paragraph(...)` seul ne parse pas, il diffère au build."""
    style = getSampleStyleSheet()["Normal"]
    SimpleDocTemplate(io.BytesIO(), pagesize=A4).build(
        [Paragraph(t, style) for t in texts])
    return True


def test_the_exact_title_that_broke_production_builds_now():
    assert _builds(f"<b>1. {_pdf_text(CANDIDA, 120, 'Sans titre')}</b>")


def test_the_old_code_really_did_break_on_it():
    """Sans quoi ce fichier ne prouverait rien : on montre la panne d'origine."""
    with pytest.raises(Exception):
        _builds(f"<b>1. {CANDIDA[:120]}</b>")       # exactement l'ancien code


@pytest.mark.parametrize("title", REAL_TITLES)
def test_every_shape_found_in_the_corpus_builds(title):
    assert _builds(f"<b>1. {_pdf_text(title, 120, 'Sans titre')}</b>",
                   _pdf_text(title, 80),
                   f"2026 · {_pdf_text(title, 120, 'Journal inconnu')} · {_pdf_text(None, 60)}")


def test_truncation_can_never_split_a_tag_or_an_entity():
    """Le coeur de la correction : on tronque le texte BRUT, on échappe ensuite, donc la
    sortie ne peut contenir ni balise ni entité coupée."""
    for n in range(0, len(CANDIDA) + 5):
        out = _pdf_text(CANDIDA, n)
        assert "<" not in out and ">" not in out, f"une balise a survécu à limit={n}"
        assert _builds(f"<b>{out}</b>")


def test_markup_is_shown_rather_than_interpreted():
    """Une balise que l'auteur avait vraiment écrite doit se voir, pas disparaître."""
    out = _pdf_text("Candida <i>auris</i>", 200)
    assert "&lt;i&gt;" in out


def test_an_ampersand_becomes_one_whole_entity():
    assert _pdf_text("Realities & Strategies", 200) == "Realities &amp; Strategies"


def test_empty_and_none_fall_back_without_raising():
    assert _pdf_text(None, 50, "Sans titre") == "Sans titre"
    assert _pdf_text("   ", 50, "Sans titre") == "Sans titre"
    assert _pdf_text(None, 50) == ""
    assert _builds(_pdf_text(None, 50, "Sans titre"))


def test_a_fallback_containing_markup_is_escaped_too():
    """Le repli vient du code aujourd'hui, mais il ne doit pas être la prochaine faille."""
    assert _pdf_text(None, 50, "<i>none</i>") == "&lt;i&gt;none&lt;/i&gt;"


def test_a_long_title_is_marked_as_cut():
    out = _pdf_text("x" * 300, 120)
    assert out.endswith("…") and len(out) <= 121
