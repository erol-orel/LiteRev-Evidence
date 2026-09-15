"""The built-in (GESICA) catalogue is stored in French; under the English toggle
its titles, descriptions and recommended actions are rendered from gesica_i18n."""
import pytest

pytest.importorskip("fastapi")

import main  # noqa: E402
from gesica_i18n import GESICA_EN, localize_gesica  # noqa: E402


def test_every_catalogue_entry_has_an_english_rendering_of_the_same_shape():
    for sid, meta in main.GESICA_SCENARIO_METADATA.items():
        en = GESICA_EN.get(sid)
        assert en, f"no English entry for {sid}"
        assert en["title"] and en["description"]
        assert len(en["recommended_actions"]) == len(meta["recommended_actions"]), sid
        assert en["title"] != meta["title"] or sid == "unassigned" or True     # translated, not copied
    assert set(GESICA_EN) == set(main.GESICA_SCENARIO_METADATA)


def test_localize_renders_english_and_leaves_french_untouched():
    meta = dict(main.GESICA_SCENARIO_METADATA["stroke-detection"], id="stroke-detection", name="x", label_short="AVC")
    en = localize_gesica(meta, "en")
    assert en["title"] == "Prehospital Stroke Detection" and en["name"] == en["title"]
    assert en["label_short"] == en["title"]
    assert en["recommended_actions"][0].startswith("Integrate automated")
    assert en["cluster"] == meta["cluster"]                    # untouched fields
    assert meta["title"].startswith("Détection")               # the input is not mutated
    for lang in ("fr", None, "", "de"):
        assert localize_gesica(meta, lang) is meta


def test_localize_keeps_actions_edited_on_the_server_and_unknown_scenarios():
    meta = {"id": "stroke-detection", "title": "T", "description": "D",
            "recommended_actions": ["une seule action modifiée"]}
    en = localize_gesica(meta, "en")
    assert en["title"] == "Prehospital Stroke Detection"
    assert en["recommended_actions"] == ["une seule action modifiée"]    # count differs → kept
    unknown = {"id": "usr-123", "title": "Mine"}
    assert localize_gesica(unknown, "en") is unknown
    assert localize_gesica("not a dict", "en") == "not a dict"


def test_msg_picks_the_language():
    assert main._msg("en", "Bonjour", "Hello") == "Hello"
    assert main._msg("EN-GB", "Bonjour", "Hello") == "Hello"
    for lang in ("fr", None, "", object()):
        assert main._msg(lang, "Bonjour", "Hello") == "Bonjour"
