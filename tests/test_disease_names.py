"""Disease names, one key per disease (api/diseases.py).

Pooled estimates combine studies within one disease, and the disease is free text: on HPAI_last
one infection, A(H5N1), was pooled six times under six spellings. These tests pin the mapping from
a spelling to its key. It is a pure function of the text: no database, no web framework."""
import ast
import pathlib

import pytest

from api import diseases
from api.diseases import NOT_STATED, disease_key, disease_spellings, fold_disease

#: The six spellings HPAI_last pooled apart (immune_status > seropositive, sheet human_susc), as
#: the papers write them and as the panel showed them once folded.
HPAI_LAST = ("HPAI H5N1", "Influenza A(H5N1)", "Avian influenza A(H5N1)", "HPAI A(H5N1)", "H5N1", "H5N1 avian influenza")
HPAI_LAST_FOLDED = ("hpai h5n1", "influenza a h5n1", "avian influenza a h5n1", "hpai a h5n1", "h5n1", "h5n1 avian influenza")


def test_the_six_spellings_of_hpai_last_are_one_disease():
    assert {disease_key(s) for s in HPAI_LAST + HPAI_LAST_FOLDED} == {"A(H5N1)"}
    # They made six pools because folding case and punctuation leaves six names.
    assert {fold_disease(s) for s in HPAI_LAST} == set(HPAI_LAST_FOLDED)


def test_h9n2_never_joins_h5n1_and_no_subtype_joins_another():
    subtypes = ("H5N1", "H9N2", "H5N8", "H5N6", "H5Nx", "H7N9", "H3N8", "H7N7", "H10N8")
    assert len({disease_key(s) for s in subtypes}) == len(subtypes)
    assert disease_key("Avian influenza A(H9N2)") == disease_key("LPAI H9N2") == "A(H9N2)"
    assert disease_key("H9N2") != disease_key("HPAI A(H5N1)")


@pytest.mark.parametrize("spelling", [
    "A/H5N1", "A (H5N1)", "(H5N1)", "H5 N1", "H5-N1", "AH5N1", "Influenza A virus subtype H5N1",
    "Highly pathogenic avian influenza (HPAI) H5N1", "HPAIV H5N1", "bird flu (H5N1)", "H5N1 virus infection",
    "H5N1 clade 2.3.4.4b", "HPAI H5N1 2.3.4.4b", "Avian influenza A(H5N1) virus, clade 2.3.4.4b, genotype B3.13",
    "influenza A(H5N1) in dairy cattle", "ＨＰＡＩ Ｈ５Ｎ１", "H₅N₁",
])
def test_the_words_around_a_subtype_are_not_part_of_the_key(spelling):
    assert disease_key(spelling) == "A(H5N1)"


def test_any_n_and_a_bare_h_are_one_key_never_a_named_n():
    assert {disease_key(s) for s in ("H5Nx", "H5N?", "HPAI H5", "H5", "avian influenza H5 clade 2.3.4.4b",
                                     "HPAI clade 2.3.4.4b", "Gs/GD H5")} == {"A(H5Nx)"}
    assert disease_key("H5Nx") != disease_key("H5N1")
    # Within one H, any N covers the named ones, and a named one covers a bare H.
    assert disease_key("H5Nx (H5N8, H5N5)") == "A(H5Nx)" and disease_key("HPAI H5 (H5N1)") == "A(H5N1)"


def test_a_lineage_suffix_is_dropped_like_the_clade():
    assert disease_key("A(H1N1)pdm09") == disease_key("H1N1") == "A(H1N1)"
    assert disease_key("H3N2v") == "A(H3N2)"


def test_low_pathogenicity_is_kept_apart_for_h5_and_h7_only():
    assert disease_key("LPAI H7N9") == "A(H7N9) LPAI"
    assert disease_key("HPAI H7N9") == disease_key("H7N9") == "A(H7N9)"
    assert disease_key("low pathogenic avian influenza H5N2") == "A(H5N2) LPAI"
    assert disease_key("LPAI H9N2") == "A(H9N2)"            # every H9 virus is low pathogenic: nothing to keep apart
    # Both stated for an H5 or H7 name: neither pool is the right one.
    assert disease_key("HPAI and LPAI H7N9") == "hpai and lpai h7n9"


def test_subtypes_named_together_make_a_key_that_joins_neither():
    both = disease_key("H5N1 and H9N2")
    assert both == disease_key("H9N2/H5N1") == disease_key("avian influenza A(H9N2) or A(H5N1)") == "A(H5N1) + A(H9N2)"
    assert both not in (disease_key("H5N1"), disease_key("H9N2"))


def test_another_disease_beside_a_subtype_or_a_negated_subtype_is_kept_as_written():
    assert disease_key("Newcastle disease and H9N2") == "newcastle disease and h9n2"
    assert disease_key("H5N1 and COVID-19") == "h5n1 and covid 19"
    assert disease_key("H5N1 and seasonal influenza") == "h5n1 and seasonal influenza"
    assert disease_key("non-H5N1 influenza") == "non h5n1 influenza"
    # A bare H outside influenza is no subtype: the H7 of E. coli O157:H7 is a flagellar antigen.
    assert disease_key("E. coli O157:H7") == "e coli o157 h7"


def test_influenza_without_a_subtype_says_what_it_can_and_joins_no_subtype():
    assert {disease_key(s) for s in ("HPAI", "HPAIV", "highly pathogenic avian influenza",
                                     "High pathogenicity avian influenza", "IAHP")} == {"HPAI"}
    assert {disease_key(s) for s in ("LPAI", "low pathogenic avian influenza", "LPAIV")} == {"LPAI"}
    assert {disease_key(s) for s in ("avian influenza", "Avian Influenza (AI)", "bird flu", "grippe aviaire", "AIV")} == {"avian influenza"}
    assert {disease_key(s) for s in ("influenza A", "Influenza A virus", "IAV")} == {"influenza A"}
    assert disease_key("influenza") == disease_key("flu") == "influenza"
    assert disease_key("HPAI and LPAI") == "hpai and lpai"
    # A clade of the goose/Guangdong lineage is H5; "clade 1b" alone could be mpox.
    assert disease_key("HPAI clade 2.3.4.4b") == "A(H5Nx)"
    assert disease_key("clade 1b") == "clade 1b" and disease_key("mpox clade Ib") == "mpox clade ib"


def test_any_other_disease_is_kept_as_written_folded_for_case_and_punctuation_only():
    assert disease_key("COVID-19") == disease_key("covid 19") == disease_key(" Covid-19 ") == "covid 19"
    assert disease_key("SARS-CoV-2") == "sars cov 2" != disease_key("COVID-19")
    assert disease_key("Rabies") == "rabies" and disease_key("Measles") == "measles"      # not singularised
    assert disease_key("seasonal influenza") == "seasonal influenza" and disease_key("influenza B") == "influenza b"
    assert disease_key("HIV-1") == "hiv 1" and disease_key("H. pylori") == "h pylori"
    assert disease_key("禽流感") == "禽流感"                      # stated, in another script


@pytest.mark.parametrize("empty", [None, "", "  ", "-", "?", "N/A", "NA", "unknown", "Not reported", "not stated",
                                   "non précisé"])
def test_a_disease_not_given_is_not_stated(empty):
    assert disease_key(empty) == NOT_STATED


def test_a_key_passed_back_in_gives_itself():
    for s in HPAI_LAST + ("LPAI H7N9", "H5/H7 LPAI", "H5N1, H5N6 and H5N8", "H5Nx", "HPAI", "LPAI", "bird flu", "IAV", "flu",
                          "HPAI and LPAI H7N9", "Newcastle disease and H9N2", "non-H5N1 influenza", "E. coli O157:H7",
                          "COVID-19", "N/A", "禽流感"):
        key = disease_key(s)
        assert disease_key(key) == key, s


def test_the_mapping_is_the_same_whatever_was_asked_before():
    first = [disease_key(s) for s in HPAI_LAST + ("H9N2", "COVID-19")]
    diseases._disease_key.cache_clear()
    assert [disease_key(s) for s in reversed(HPAI_LAST + ("H9N2", "COVID-19"))][::-1] == first


def test_the_mapping_needs_no_database_and_no_web_framework():
    tree = ast.parse(pathlib.Path(diseases.__file__).read_text(encoding="utf-8"))
    imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert not any(isinstance(n, ast.ImportFrom) and n.level for n in imports)          # nothing from the api package
    names = {a.name.split(".")[0] for n in imports if isinstance(n, ast.Import) for a in n.names}
    names |= {n.module.split(".")[0] for n in imports if isinstance(n, ast.ImportFrom)}
    assert names <= {"__future__", "re", "unicodedata", "collections", "functools", "typing"}


def test_the_spellings_are_folded_counted_and_shown_as_most_often_written():
    sp = disease_spellings(["HPAI H5N1", "HPAI H5N1", "hpai  h5n1", "Influenza A(H5N1)", "H5N1", None, "",
                            "influenza a (h5n1)", "Influenza A(H5N1)"])
    assert sp == [{"spelling": "HPAI H5N1", "n": 3}, {"spelling": "Influenza A(H5N1)", "n": 3}, {"spelling": "H5N1", "n": 1}]
    assert disease_spellings([None, ""]) == []
