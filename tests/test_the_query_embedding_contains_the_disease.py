"""Le plongement de la requête coupait à 2 000 caractères, AVANT le bloc des virus.

Trouvé par la relecture adversariale en évaluant si le scénario HPAI de production pouvait
servir de référence « avant » : `_run_semantic_rerank_inline` plongeait `query[:2000]`. La
requête du scénario fait 3 075 caractères, et son bloc « ET (virus aviaires) » commence au
caractère 2 239. Les 2 000 premiers ne contiennent ni « influenza » ni « H5N1 ».

Les 201 articles « pertinents » de ce scénario, et tout ce qui en découle, ont donc été
classés par similarité avec un plongement d'« exposition professionnelle, fomites,
aérosols, transmission, perception du risque », sans la maladie. La relecture a mesuré sur
le corpus réel que 36 % seulement des 201 mentionnent la grippe aviaire.

La même coupe était dans `_embed_query_vector`, le plongement des questions du RAG.

Ces tests MESURENT ce qui part vers le modèle : un faux client enregistre l'argument
`input` et lève, et la fonction, qui attrape tout, rend 0 ; on lit ensuite ce qui a été
enregistré. Aucun grep du source.
"""
import pytest

pytest.importorskip("fastapi")

from api import relevance as R  # noqa: E402

#: La forme de la requête de production : quatre blocs d'exposition en OU, longs, PUIS le
#: bloc des virus, le tout bien au-delà de 2 000 caractères.
HPAI_LIKE = (
    "( " + " OR ".join(f'"Exposure Concept Number {i:03d} With Long Name"[tiab]' for i in range(40))
    + ' ) AND ( "Influenza in Birds"[mh] OR "Influenza A Virus, H5N1 Subtype"[mh] '
    'OR "avian influenza"[tiab] OR "H5N1"[tiab] ) NOT ( "news"[Publication Type] )'
)
assert len(HPAI_LIKE) > 2000 and HPAI_LIKE.find("Influenza in Birds") > 2000


class _Captured(Exception):
    pass


class _FakeEmbeddings:
    def __init__(self, sink):
        self.sink = sink

    def create(self, model=None, input=None, **kw):
        self.sink.append(input)
        raise _Captured("captured, do not continue into SQL")


class _FakeClient:
    def __init__(self, sink):
        self.embeddings = _FakeEmbeddings(sink)


# ── La fonction pure ─────────────────────────────────────────────────────────

def test_the_whole_query_is_embedded_and_the_disease_block_is_in_it():
    text_ = R.embedding_text_for_query(HPAI_LIKE)
    assert "Influenza in Birds" in text_ and "H5N1" in text_ and "avian influenza" in text_, (
        "le bloc des virus n'atteint pas le plongement : le classement ne voit pas la maladie")
    assert len(text_) > 2000, "la requête est encore coupée à 2 000 caractères"


def test_field_tags_are_not_embedded():
    text_ = R.embedding_text_for_query(HPAI_LIKE)
    for tag in ("[tiab]", "[mh]", "[Publication Type]"):
        assert tag not in text_, f"{tag} gaspille la place du plongement"
    assert "  " not in text_


def test_the_bound_is_the_models_not_an_arbitrary_2000():
    assert R.EMBED_QUERY_MAX_CHARS >= 8_000
    long = "word " * 10_000
    assert len(R.embedding_text_for_query(long)) == R.EMBED_QUERY_MAX_CHARS


def test_an_empty_query_embeds_an_empty_string_rather_than_raising():
    assert R.embedding_text_for_query("") == ""
    assert R.embedding_text_for_query(None) == ""


# ── Ce qui part réellement vers le modèle ────────────────────────────────────

def test_the_rerank_sends_the_whole_query_to_the_embedding_model(monkeypatch):
    sent = []
    import llm_usage
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda *a, **k: _FakeClient(sent))
    n = R._run_semantic_rerank_inline("usr-embedtest0001", HPAI_LIKE)
    assert n == 0, "le faux client lève : la fonction doit rendre 0, pas propager"
    assert len(sent) == 1, "aucun appel au modèle de plongement n'a été capturé"
    payload = sent[0] if isinstance(sent[0], str) else sent[0][0]
    assert "H5N1" in payload and "Influenza in Birds" in payload, (
        "ce qui part vers le modèle ne contient pas le bloc des virus")
    assert "[tiab]" not in payload


def test_the_rag_question_embedding_uses_the_same_text(monkeypatch):
    sent = []
    import llm_usage
    monkeypatch.setattr(llm_usage, "MeteredOpenAI", lambda *a, **k: _FakeClient(sent))
    monkeypatch.setenv("OPENAI_API_KEY", "x")
    out = R._embed_query_vector(HPAI_LIKE)
    assert out is None, "le faux client lève : la fonction doit rendre None"
    assert sent and ("H5N1" in (sent[0][0] if isinstance(sent[0], list) else sent[0]))
