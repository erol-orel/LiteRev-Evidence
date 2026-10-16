"""L'interface a deux thèmes, et rien ne doit pouvoir en échapper en silence.

Le mode sombre est l'interface d'origine ; le mode clair est obtenu en changeant la valeur
de variables CSS, sans toucher aux ~3 000 classes de couleur déjà écrites. Cela tient à
trois conditions, et chacune se casse par simple distraction :

1. une teinte employée dans `src/` doit être déclarée dans `tailwind.config.js`, sinon elle
   garde la valeur fixe de Tailwind et reste claire sur fond clair ;
2. la même teinte doit avoir ses deux jeux de valeurs dans `src/theme.css` ;
3. une couleur écrite en dur dans une classe (`bg-[#0a1410]`) ne suit aucun thème.

Ces tests lisent les fichiers, sans navigateur : ils disent ce qui manque et où.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "frontend" / "src"
CONFIG = ROOT / "frontend" / "tailwind.config.js"
THEME = SRC / "theme.css"
INDEX_HTML = ROOT / "frontend" / "index.html"

pytestmark = pytest.mark.skipif(not SRC.exists(), reason="pas de frontend dans cette copie")

#: Les préfixes d'utilitaires Tailwind qui portent une couleur.
PREFIX = (r"(?:bg|text|border|from|to|via|ring|fill|stroke|divide|placeholder|accent|"
          r"decoration|outline|shadow|caret)")
#: `text-rose-300`, `bg-brand-500/20`... La teinte est le groupe 1.
HUE_CLASS = re.compile(rf"\b{PREFIX}-([a-z]{{3,12}})-(?:50|[1-9]00|950)\b")
#: Une couleur écrite en dur dans une classe : `bg-[#0a1410]`, `text-[#fff]`.
FIXED_CLASS = re.compile(rf"\b{PREFIX}-\[#[0-9a-fA-F]{{3,8}}\]")

#: Les noms qui ressemblent à une teinte sans en être une (`border-l-2`, `top-1/2`...).
NOT_HUES = {"none", "transparent", "current", "inherit", "auto", "full", "screen"}


def _tsx_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.tsx") if ".test." not in p.name)


def _hues_in_use() -> dict[str, set[str]]:
    """Teinte -> fichiers qui l'emploient."""
    found: dict[str, set[str]] = {}
    for f in _tsx_files():
        for hue in HUE_CLASS.findall(f.read_text(encoding="utf-8")):
            if hue in NOT_HUES:
                continue
            found.setdefault(hue, set()).add(str(f.relative_to(ROOT)))
    return found


def _declared_hues() -> set[str]:
    text = CONFIG.read_text(encoding="utf-8")
    block = re.search(r"const HUES = \[(.*?)\]", text, re.S)
    assert block, "tailwind.config.js ne déclare plus de liste HUES"
    return set(re.findall(r"'([a-z]+)'", block.group(1)))


def test_every_hue_in_use_follows_the_theme():
    """Une teinte oubliée garde la valeur fixe de Tailwind : claire sur fond clair."""
    missing = {h: sorted(f)[:3] for h, f in _hues_in_use().items() if h not in _declared_hues()}
    assert not missing, (
        "ces teintes sont employées mais absentes de HUES dans tailwind.config.js, "
        f"donc elles ne suivent pas le thème : {missing}")


def test_every_declared_hue_has_both_sets_of_values():
    """Déclarer une teinte sans la générer donnerait une variable vide, donc du noir."""
    css = THEME.read_text(encoding="utf-8")
    dark, light = css.split("[data-theme='light']", 1)
    for hue in sorted(_declared_hues()):
        for shade in (50, 300, 500, 700, 950):
            var = f"--c-{hue}-{shade}:"
            assert var in dark, f"{var} manque au jeu sombre de theme.css"
            assert var in light, f"{var} manque au jeu clair de theme.css"


def test_the_light_set_is_not_a_copy_of_the_dark_one():
    """Garde-fou sur le générateur : un mode clair identique au sombre serait silencieux."""
    css = THEME.read_text(encoding="utf-8")
    dark, light = css.split("[data-theme='light']", 1)
    found = re.search(r"--c-brand-300: ([^;]+);", dark)
    light_found = re.search(r"--c-brand-300: ([^;]+);", light)
    assert found and light_found
    assert found.group(1) != light_found.group(1), "les deux jeux donnent la même couleur"


def test_no_colour_is_written_in_stone_in_a_class():
    """`bg-[#0a1410]` reste sombre sous tous les thèmes : c'est exactement le piège."""
    offenders = {
        str(f.relative_to(ROOT)): sorted(set(FIXED_CLASS.findall(f.read_text(encoding="utf-8"))))
        for f in _tsx_files()
        if FIXED_CLASS.search(f.read_text(encoding="utf-8"))
    }
    assert not offenders, (
        "ces classes portent une couleur fixe ; passer par une variable du thème "
        f"(`bg-[rgb(var(--surface))]`) : {offenders}")


def test_the_page_is_painted_before_react_starts():
    """Sans ce script, une page choisie en clair s'affiche sombre puis saute."""
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "literev-theme" in html, "index.html ne lit plus le thème choisi"
    assert "data-theme" in html, "index.html ne pose plus le thème avant le rendu"
    provider = (SRC / "theme" / "ThemeProvider.tsx").read_text(encoding="utf-8")
    assert 'STORAGE_KEY = "literev-theme"' in provider, (
        "la clé de stockage a changé dans le provider mais pas dans index.html")


def test_the_selected_pill_reads_in_both_themes():
    """Or sur vert foncé dans les deux thèmes : inversé, ce serait or sur vert pâle."""
    css = THEME.read_text(encoding="utf-8")
    dark, light = css.split("[data-theme='light']", 1)
    for part, name in ((dark, "sombre"), (light, "clair")):
        assert "--pill:" in part and "--on-pill:" in part, f"jeton de pastille absent en {name}"
    for f in _tsx_files():
        text = f.read_text(encoding="utf-8")
        assert "bg-brand-700 text-gold-400" not in text, (
            f"{f.relative_to(ROOT)} utilise encore l'ancienne pastille, illisible en clair")
