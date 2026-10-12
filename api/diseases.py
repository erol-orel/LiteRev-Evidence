"""Disease names: one key per disease, decided when the extraction is read.

The extraction writes the disease as the paper words it, and papers word one infection in many
ways: "HPAI H5N1", "influenza A(H5N1)", "avian influenza A (H5N1) clade 2.3.4.4b", "H5N1".
Pooled estimates combine studies within one disease, so one disease spelled six ways made six
pools: on HPAI_last the seropositive humans of one codebook label were pooled six times, about 23
studies that belong in one pool. Merging every disease is no fix: it pools A(H9N2) and COVID-19
with them.

Like the codebook labels, the disease is NORMALISED AT READ TIME: the stored extraction is never
rewritten, and a change to these rules applies to every extraction already made, with no new
model call. The mapping is a pure function of the text, deterministic and conservative (two
spellings share a key only when they name the same thing):

  - an influenza A subtype is recognised however it is written (H5N1, A(H5N1), A/H5N1, H5 N1,
    AH5N1, H1N1pdm09, H3N2v) and the key is the subtype, written the WHO way: A(H5N1). The words
    around it are not part of the key: HPAI, highly pathogenic, avian, bird, influenza, flu,
    virus, the clade (2.3.4.4b), the lineage, the host. H5Nx, H5N? and a bare H5 are A(H5Nx);
  - low pathogenicity stated for an H5 or an H7 virus is kept apart, as in A(H7N9) LPAI: the low
    and the highly pathogenic viruses of one subtype are different diseases. For the other
    subtypes, which are all low pathogenic, it says nothing more and is dropped;
  - several subtypes named together (H5N1 and H9N2) make a key of their own, which joins neither;
  - with no subtype, an influenza name is HPAI, LPAI, avian influenza, influenza A or influenza,
    by what it says; a Gs/GD clade such as 2.3.4.4b names an A(H5Nx) virus;
  - any other disease is kept as written, folded only for case, accents and punctuation:
    "COVID-19" and "covid 19" are one key, "SARS-CoV-2" another. So is a name that joins another
    disease to a subtype ("Newcastle disease and H9N2"), or that negates one ("non-H5N1"): it
    pools only with itself;
  - an empty field, or one that says the disease is unknown (n/a, not reported), is
    "(not stated)".

A key passed back in gives itself.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from functools import lru_cache
from typing import Any, Iterable

NOT_STATED = "(not stated)"

#: An influenza A subtype in ASCII lower case: H1 to H19, then N1 to N11 or any N (x, ?, *),
#: glued to the type or not (AH5N1), with or without a space, hyphen or slash (H5 N1). Letters
#: may follow the N (H1N1pdm09, H3N2v), a digit may not; a bare H must stand alone (H5, H5/H7).
_SUBTYPE = re.compile(r"(?<![a-z0-9])(?P<neg>non[\s-]*)?a?h(?P<h>[1-9][0-9]?)"
                      r"(?:[\s/-]*n(?P<n>[1-9][0-9]?|x|\?|\*)(?![0-9])|(?![a-z0-9]))")
#: Words that join two names: a part between them that is not influenza is another disease.
_CONNECTOR = re.compile(r"\s+(?:and|with|plus|or|versus|vs\.?|et|ou|avec)\s+")
#: What an influenza name is made of beside its subtype. A name made only of these words and of
#: numbers names influenza and nothing else.
_FLU_WORDS = frozenset("""
    a type subtype subtypes strain strains lineage lineages clade clades genotype genotypes
    virus viruses viral infection infections infected disease diseases case cases
    outbreak outbreaks epizootic epizootics panzootic epidemic epidemics
    influenza flu grippe aviaire avian bird birds hautement faiblement pathogene
    highly high low lowly pathogenic pathogenicity hpai hpaiv lpai lpaiv ai aiv iav iahp iafp
    of the in and or et ou de du la le des
""".split())
_LINK_WORDS = frozenset("a and or et ou with plus".split())
_NUMBERISH = re.compile(r"[0-9]+[a-z]?")
_FLU_MARKER = re.compile(r"\b(?:influenza|flu|grippe|avian|aviaire|birds?|hpaiv?|lpaiv?|aiv?|iav|iahp|iafp)\b")
_HP = re.compile(r"\b(?:hpaiv?|iahp|highly pathogenic|high pathogenicity|high pathogenic|hautement pathogene)\b")
_LP = re.compile(r"\b(?:lpaiv?|iafp|low pathogenic|low pathogenicity|lowly pathogenic|faiblement pathogene)\b")
#: The goose/Guangdong lineage, all of whose clades are H5: a clade with two dots or more
#: (2.3.4.4b, 2.3.2.1c) or the lineage named. "Clade 1" alone could be mpox: see `_CLADE`.
_GSGD = re.compile(r"\b[0-9]\.[0-9]+(?:\.[0-9]+)+[a-z]?\b|\bgs\s*/\s*gd\b|\bgoose\s*/\s*guangdong\b")
_CLADE = re.compile(r"\bclade\s*[0-9][0-9a-z.]*")
_NULL_MARKERS = frozenset({
    "", "na", "n a", "nan", "none", "null", "nil", "nr", "ns", "unknown", "unspecified", "unclear",
    "not stated", "not specified", "not reported", "not applicable", "not available", "not given",
    "not mentioned", "no disease", "inconnu", "inconnue", "non precise", "non precisee",
    "non renseigne", "non specifie", "non specifiee", "aucun", "aucune",
})
_H5_H7 = (5, 7)


def _ascii(s: Any) -> str:
    """Lower-case ASCII: accents dropped, full-width and subscript characters made plain."""
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def fold_disease(s: Any) -> str:
    """A disease name folded for comparison: lower-case ASCII words, punctuation as spaces. Not
    singularised, unlike a codebook label ("rabies" stays "rabies"). A name in a script with no
    ASCII form is folded in its own script rather than lost."""
    if s is None:
        return ""
    t = " ".join(re.sub(r"[^a-z0-9]+", " ", _ascii(s)).split())
    if not t and any(ch.isalnum() for ch in str(s)):
        t = " ".join(re.sub(r"[\W_]+", " ", str(s).casefold()).split())
    return t


def _bare_h_ok(text: str) -> bool:
    """A bare H (H5, H7) is a subtype only beside an influenza word, or alone: in "E. coli
    O157:H7" it is a flagellar antigen."""
    if _FLU_MARKER.search(fold_disease(text)) or _GSGD.search(text):
        return True
    rest = fold_disease(_SUBTYPE.sub(" ", text)).split()
    return all(w in _LINK_WORDS or _NUMBERISH.fullmatch(w) for w in rest)


def _subtypes(text: str) -> list[tuple[int, str, bool]]:
    """(H, N, negated) for every subtype written in `text`, in ASCII lower case. N is the number,
    "x" for any N (H5Nx), or "" for a bare H."""
    found: list[tuple[int, str, bool]] = []
    bare_ok: bool | None = None
    for m in _SUBTYPE.finditer(text):
        h, n = int(m["h"]), m["n"] or ""
        n = "x" if n in ("?", "*") else n
        if h > 19 or (n not in ("", "x") and int(n) > 11):
            continue
        if not n:
            bare_ok = _bare_h_ok(text) if bare_ok is None else bare_ok
            if not bare_ok:
                continue
        found.append((h, n, bool(m["neg"])))
    return found


def _flu_only(text: str) -> bool:
    """True when `text` (ASCII lower case) is made only of influenza words and numbers."""
    words = fold_disease(_CLADE.sub(" ", _GSGD.sub(" ", text))).split()
    return all(w in _FLU_WORDS or _NUMBERISH.fullmatch(w) for w in words)


def _key_of(found: list[tuple[int, str, bool]], lp: bool) -> str:
    """A(H5N1), A(H5Nx), A(H5N1) + A(H9N2), A(H7N9) LPAI. Within one H, any N (x) covers the
    named ones and a named one covers a bare H."""
    by_h: dict[int, set[str]] = {}
    for h, n, _neg in found:
        by_h.setdefault(h, set()).add(n)
    parts = []
    for h in sorted(by_h):
        named = sorted((n for n in by_h[h] if n not in ("", "x")), key=int)
        parts += [f"A(H{h}Nx)"] if "x" in by_h[h] or not named else [f"A(H{h}N{n})" for n in named]
    return " + ".join(parts) + (" LPAI" if lp and any(h in _H5_H7 for h in by_h) else "")


@lru_cache(maxsize=8192)
def _disease_key(raw: str) -> str:
    text = _ascii(raw)
    folded = fold_disease(raw)
    if folded in _NULL_MARKERS:
        return NOT_STATED
    hp, lp = bool(_HP.search(folded)), bool(_LP.search(folded))
    tokens = set(folded.split())
    found = _subtypes(text)
    if found:
        if any(neg for _h, _n, neg in found):
            return folded
        if any(not _subtypes(p) and not _flu_only(p) for p in _CONNECTOR.split(text)):
            return folded
    elif not _flu_only(text):
        return folded
    elif _GSGD.search(text) or (_CLADE.search(text) and _FLU_MARKER.search(folded)):
        found = [(5, "x", False)]
    elif hp != lp:
        return "HPAI" if hp else "LPAI"
    elif hp:
        return folded                                  # high and low together: neither pool
    elif tokens & {"avian", "aviaire", "bird", "birds", "ai", "aiv"}:
        return "avian influenza"
    elif "iav" in tokens or ("a" in tokens and tokens & {"influenza", "flu", "grippe"}):
        return "influenza A"
    elif tokens & {"influenza", "flu", "grippe"}:
        return "influenza"
    else:
        return folded
    if hp and lp and any(h in _H5_H7 for h, _n, _neg in found):
        return folded
    return _key_of(found, lp and not hp)


def disease_key(raw: Any) -> str:
    """The key of a disease as a paper wrote it (see the module note for the rules)."""
    return NOT_STATED if raw is None else _disease_key(str(raw))


def disease_spellings(raws: Iterable[Any]) -> list[dict[str, Any]]:
    """The spellings behind some rows, one per name folded for case, accents and punctuation:
    each shown as it is most often written (then alphabetically), with how many rows carry it,
    most frequent first. A row with no disease written has no spelling."""
    seen: dict[str, Counter] = {}
    for raw in raws:
        s = " ".join(str(raw).split()) if raw is not None else ""
        if s:
            seen.setdefault(fold_disease(s), Counter())[s] += 1
    out = [{"spelling": min(c, key=lambda s: (-c[s], s)), "n": sum(c.values())} for c in seen.values()]
    out.sort(key=lambda e: (-e["n"], e["spelling"].lower(), e["spelling"]))
    return out


def spellings_text(spellings: Iterable[dict[str, Any]]) -> str:
    """The spellings of `disease_spellings` in one table cell: "HPAI H5N1 (5); H5N1 (3)"."""
    return "; ".join(f"{s['spelling']} ({s['n']})" for s in spellings)
