"""The demo video is code (docs/demo-video.md): what it shows, when, and on what data.

Pure tests: the synthetic corpus (scripts/demo_corpus.py) without a database, the
renderer's camera and timeline (scripts/demo_render.py) without a browser or an encoder,
and the storyboard (frontend/demo/storyboard.json) against the tour that films it."""
import importlib.util
import json
import os
import random
import re
import sys
import wave

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, os.path.join(ROOT, "scripts", f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod             # a dataclass looks its module up while it is defined
    spec.loader.exec_module(mod)
    return mod


corpus = _load("demo_corpus")
render = _load("demo_render")
Rect = render.Rect


# ─────────────────────────────────────────────────────────────────────────────
# The corpus: fiction, and it says so; coherent, so the screens mean something
# ─────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def records():
    return corpus.build_corpus()


def test_the_corpus_is_the_same_on_every_run(records):
    again = corpus.build_corpus()
    assert [(r.title, r.score, r.year, r.source) for r in records] == \
           [(r.title, r.score, r.year, r.source) for r in again]
    other = corpus.build_corpus(seed=corpus.DEFAULT_SEED + 1)
    assert [r.title for r in other] != [r.title for r in records]


def test_the_corpus_has_its_size_its_themes_and_unique_titles(records):
    assert len(records) == corpus.DEFAULT_ARTICLES
    assert len({r.title.lower() for r in records}) == len(records)
    assert {r.theme for r in records} == {name for name, *_ in corpus.THEMES}
    assert all(r.external_id.startswith(corpus.EXTERNAL_PREFIX) for r in records)


def test_the_default_threshold_keeps_the_topic_and_drops_the_rest(records):
    above = [r for r in records if r.score >= 0.45]
    assert 0.4 <= len(above) / len(records) <= 0.75
    assert not [r for r in above if r.theme == "off_topic"]
    on_topic = [r for r in records if r.theme == "wastewater"]
    assert sum(r.score >= 0.45 for r in on_topic) / len(on_topic) > 0.9


def test_no_record_contradicts_its_own_dates(records):
    """A study period ends before publication, SARS-CoV-2 exists only from 2020, a season
    is over before the paper about it is published."""
    for r in records:
        text = f"{r.title} {r.abstract}"
        for y1, y2 in re.findall(r"between (\d{4}) and (\d{4})", r.abstract):
            assert int(y1) <= int(y2) < r.year, (r.title, r.year)
        for y1, y2 in re.findall(r"\b(\d{4})-(\d{4})\b", r.title):
            assert int(y2) < r.year, (r.title, r.year)
        for start in re.findall(r"\b(\d{4})/\d{2}\b", text):
            if "Recommendations" not in r.title:            # a guideline is for the coming season
                assert int(start) + 1 < r.year, (r.title, r.year)
        if "SARS-CoV-2" in text or "COVID-19" in text:
            assert r.year >= 2021, (r.title, r.year)


def test_every_design_reads_as_a_design_of_the_app(records):
    from api.study_design import classify
    for r in records:
        if r.design != "Not stated":
            assert classify(r.design) != "not_stated", r.design


def test_the_pico_is_what_the_app_keeps(records):
    from api.pico_map import FINDING_DIRECTIONS, normalise_pico
    for r in records:
        kept = normalise_pico(r.pico, source="abstract")
        assert all(kept[k] for k in ("P", "I", "O")), r.title
        assert kept["finding_direction"] in FINDING_DIRECTIONS
        assert bool(kept["key_finding"]) == (kept["finding_direction"] != "none"), r.title
        assert 0.5 <= kept["pico_confidence"] <= 1


def test_the_concepts_are_typed_like_the_concept_pass_writes_them(records):
    allowed = {"pathogen", "vector", "host", "population", "exposure", "intervention", "outcome", "method"}
    for r in records:
        cj = corpus.concepts_json(r)
        assert cj["concepts"] and len(cj["concepts"]) <= 8
        for c in cj["concepts"]:
            if c["t"] == "place":
                assert re.fullmatch(r"[A-Z]{2}", c["en"])
            else:
                assert c["t"] in allowed and c["en"] and c["fr"], c
        # Keywords become Topic nodes: one repeating a concept would link it to itself.
        labels = {c["en"].lower() for c in cj["concepts"]}
        assert not labels & {k.strip().lower() for k in r.keywords.split(";")}, r.title


def test_no_generated_text_carries_an_em_dash(records):
    em = chr(0x2014)
    for r in records:
        blob = json.dumps([r.title, r.abstract, r.pico, r.concepts, r.keywords, r.design], ensure_ascii=False)
        assert em not in blob, r.title


def test_identification_counts_each_record_once_per_source_that_returned_it(records):
    by_source = corpus.records_by_source(records)
    assert sum(by_source.values()) == sum(len(r.found_in) for r in records) >= len(records)
    assert set(by_source) <= {s for s, *_ in corpus.SOURCES}


def test_a_full_text_is_laid_out_in_sections_only_when_there_is_one(records):
    with_text = [r for r in records if r.has_fulltext]
    assert 0.3 < len(with_text) / len(records) < 0.9
    for r in records:
        sections = corpus.fulltext_sections(r)
        if not r.has_fulltext:
            assert sections == []
            continue
        assert [label for label, _ in sections][:2] == ["Introduction", "Methods"]
        assert all(body and "Methods:" not in body for _, body in sections)


# ─────────────────────────────────────────────────────────────────────────────
# The camera: 16:9, inside the screenshot, the box above the caption
# ─────────────────────────────────────────────────────────────────────────────
def _close(a, b, tol=1e-6):
    return abs(a - b) <= tol


def test_without_a_box_the_camera_shows_the_top_of_the_page():
    assert render.camera((1440, 810), None) == (Rect(0, 0, 1440, 810), None)
    cam, end = render.camera((1440, 3000), None, band=0.25)
    assert cam == Rect(0, 0, 1440, 810) and end is None
    assert render.camera((1920, 1080), None)[0] == Rect(0, 0, 1920, 1080)


def test_a_small_box_is_framed_close_but_not_closer_than_the_limit():
    cam, end = render.camera((1440, 810), Rect(670, 380, 100, 50), band=0.25)
    assert end is None
    assert _close(cam.w, render.MIN_CAMERA_W) and _close(cam.w / cam.h, render.ASPECT)
    assert cam.x <= 670 and cam.x + cam.w >= 770


def test_the_camera_keeps_16_9_stays_in_the_image_and_keeps_the_box_above_the_caption():
    rnd = random.Random(7)
    for _ in range(500):
        w, h = 1440.0, rnd.choice([810.0, 1200.0, 2400.0, 3000.0])
        bw, bh = rnd.uniform(40, w - 20), rnd.uniform(20, min(h, 700))
        box = Rect(rnd.uniform(0, w - bw), rnd.uniform(0, h - bh), bw, bh)
        band = rnd.choice([0.0, 0.19, 0.25])
        cam, end = render.camera((w, h), box, band)
        for c in filter(None, (cam, end)):
            assert _close(c.w / c.h, render.ASPECT, 1e-6)
            assert c.x >= -1e-6 and c.y >= -1e-6 and c.x + c.w <= w + 1e-6 and c.y + c.h <= h + 1e-6
            assert c.w >= min(render.MIN_CAMERA_W, w) - 1e-6
        if end is None and cam.w >= box.w:
            # The whole box is in the part of the frame the caption leaves free.
            assert box.x >= cam.x - 1e-6 and box.x + box.w <= cam.x + cam.w + 1e-6
            free_bottom = cam.y + cam.h * (1 - band)
            assert box.y >= cam.y - 1e-6
            assert box.y + box.h <= free_bottom + 1e-6 or _close(cam.y + cam.h, h, 1e-6)


def test_a_box_too_tall_for_the_frame_is_shown_by_panning_down_it():
    box = Rect(40, 200, 1360, 1500)
    start, end = render.camera((1440, 2400), box, band=0.25)
    assert end is not None and end.y > start.y and end.x == start.x
    assert start.y <= box.y                                   # starts at the top of the box
    assert end.y + end.h * 0.75 >= box.y + box.h - 1e-6       # ends showing its bottom


def test_a_camera_overshooting_the_screenshot_by_a_fraction_is_moved_back_inside():
    assert render._inside((10, 2700, 2890, 4320), (2880, 4300)) == (0.0, 2680.0, 2880.0, 4300.0)
    assert render._inside((-5, -3, 100, 50), (200, 200)) == (0.0, 0.0, 105.0, 53.0)


# ─────────────────────────────────────────────────────────────────────────────
# The timeline
# ─────────────────────────────────────────────────────────────────────────────
def _scene(sid, beats, kind="screen", voice=0.0, caption=True):
    s = render.Scene(id=sid, kind=kind, narration=f"Narration of {sid}.", overlay=None, band=0.25,
                     beats=beats, caption={"step": "1", "title": sid.title(), "body": f"Body of {sid}."} if caption else None)
    s.voice = voice
    return s


def _beat(image="a.png", focus=None, spot=False, hold=2.0, cut=False, size=(1440, 810)):
    return render.Beat(image=image, size=size, focus=focus, spot=spot, hold=hold, cut=cut)


def test_scenes_dip_through_the_background_beats_move_typing_cuts():
    scenes = [
        _scene("intro", [_beat("card.png", size=(1920, 1080), hold=3.0)], kind="card", caption=False),
        _scene("ask", [_beat("t1.png", Rect(300, 200, 900, 300), hold=0.12, cut=True),
                       _beat("t2.png", Rect(300, 200, 900, 300), hold=0.12, cut=True),
                       _beat("t3.png", Rect(300, 200, 900, 300), spot=True, hold=3.0, cut=True)]),
        _scene("corpus", [_beat("c.png", Rect(40, 100, 1300, 160), spot=True, hold=2.0),
                          _beat("c.png", Rect(40, 500, 1300, 200), spot=True, hold=2.5)]),
    ]
    tl = render.plan(scenes)
    kinds = [s.kind for s in tl.segments]
    assert kinds[0] == "hold"                                  # the video opens on its first frame
    assert kinds.count("dip") == 2                             # one per scene change
    assert "move" in kinds and "fade" not in kinds             # same screenshot: the camera moves
    # Typing: cuts from frame to frame, then the spotlight fades in on the last one.
    assert kinds[kinds.index("dip") + 1:kinds.index("dip") + 5] == ["hold", "hold", "spot", "hold"]
    expected = 3.0 + render.FADE_SCENE + 0.12 * 2 + 3.0 + render.FADE_SCENE + 2.0 + render.MOVE + 2.5
    assert abs(tl.seconds - expected) < 1e-9
    starts = [start for _, start, _ in tl.scenes]
    ends = [end for _, _, end in tl.scenes]
    assert starts[0] == 0 and starts[1:] == ends[:-1] and abs(ends[-1] - tl.seconds) < 1e-9


def test_a_narrated_scene_lasts_as_long_as_its_voice():
    short = [_scene("a", [_beat(hold=2.0)]), _scene("b", [_beat("b.png", hold=2.0)], voice=6.0)]
    tl = render.plan(short)
    (_, a0, a1), (_, b0, b1) = tl.scenes
    assert abs((a1 - a0) - 2.0) < 1e-9
    assert b1 - b0 >= render.NARRATION_DELAY + 6.0 + render.NARRATION_TAIL - 1e-9
    quiet = render.plan([_scene("a", [_beat(hold=2.0)]), _scene("b", [_beat("b.png", hold=9.0)], voice=1.0)])
    assert abs(quiet.scenes[1][2] - quiet.scenes[1][1] - (render.FADE_SCENE + 9.0)) < 1e-9


def test_the_subtitles_follow_the_scenes():
    scenes = [_scene("intro", [_beat(hold=3.0)], kind="card", caption=False),
              _scene("ask", [_beat("b.png", hold=4.0)])]
    tl = render.plan(scenes)
    text = render.vtt(scenes, tl)
    assert text.startswith("WEBVTT\n")
    times = re.findall(r"(\d\d:\d\d:\d\d\.\d{3}) --> (\d\d:\d\d:\d\d\.\d{3})", text)
    second_end = 3.0 + render.FADE_SCENE + 4.0
    assert times == [("00:00:00.000", "00:00:03.000"), ("00:00:03.000", f"00:00:0{second_end:.3f}")]
    assert "Narration of intro." in text and "Ask\nBody of ask." in text


def test_frames_are_exactly_the_timeline_long(tmp_path):
    Image = pytest.importorskip("PIL.Image")
    os.makedirs(tmp_path / "frames")
    Image.new("RGB", (2880, 1620), (20, 60, 40)).save(tmp_path / "frames" / "a.png")
    Image.new("RGB", (1920, 1080), (11, 20, 17)).save(tmp_path / "frames" / "card.png")
    scenes = [_scene("intro", [_beat("frames/card.png", size=(1920, 1080), hold=0.3)], kind="card"),
              _scene("a", [_beat("frames/a.png", Rect(100, 100, 400, 200), spot=True, hold=0.5),
                           _beat("frames/a.png", Rect(600, 300, 400, 200), spot=True, hold=0.4)])]
    tl = render.plan(scenes)
    n = 0
    for data in render.frames(tl, render.Composer(str(tmp_path))):
        assert len(data) == render.OUT_W * render.OUT_H * 3
        n += 1
    assert n == round(tl.seconds * render.FPS)


def test_the_narration_track_places_each_clip_at_its_scene(tmp_path):
    rate = render.TTS_RATE
    clips = {}
    for sid in ("a", "b"):
        path = tmp_path / f"{sid}.pcm"
        path.write_bytes(b"\x10\x27" * int(rate * 0.5))       # half a second of a constant sample
        clips[sid] = str(path)
    tl = render.plan([_scene("a", [_beat(hold=2.0)]), _scene("b", [_beat("b.png", hold=2.0)])])
    out = tmp_path / "n.wav"
    render.narration_track(clips, tl, str(out))
    with wave.open(str(out)) as w:
        assert w.getframerate() == rate and w.getnchannels() == 1
        assert abs(w.getnframes() / rate - tl.seconds) < 0.01
        pcm = w.readframes(w.getnframes())
    for _, start, _ in tl.scenes:
        at = int((start + render.NARRATION_DELAY) * rate) * 2
        assert pcm[at:at + 2] == b"\x10\x27"
        assert pcm[at - 2:at] == b"\x00\x00"


# ─────────────────────────────────────────────────────────────────────────────
# The storyboard: the same scenes in both languages, each one filmed by the tour
# ─────────────────────────────────────────────────────────────────────────────
STORYBOARD = os.path.join(ROOT, "frontend", "demo", "storyboard.json")
TOUR = os.path.join(ROOT, "frontend", "demo", "tour.spec.ts")
#: A body longer than this takes a third line on the caption card and hides the screen.
MAX_BODY = 230


@pytest.fixture(scope="module")
def board():
    with open(STORYBOARD, encoding="utf-8") as f:
        return json.load(f)


def test_both_languages_caption_every_scene_the_tour_films(board):
    with open(TOUR, encoding="utf-8") as f:
        filmed = set(re.findall(r"^  async (\w+)\(\{ page, rec \}\)", f.read(), re.M))
    assert filmed, "no scene found in the tour"
    assert set(board["en"]["scenes"]) == set(board["fr"]["scenes"]) == filmed
    assert set(board["order"]) <= filmed and len(board["order"]) == len(set(board["order"]))


def test_every_caption_fits_its_card(board):
    for lang in ("en", "fr"):
        words = board[lang]
        assert words["question"] and words["demoBadge"] and words["demoNote"]
        for card in (words["intro"], words["outro"]):
            assert card["title"] and card["subtitle"] and card["narration"]
        for sid, c in words["scenes"].items():
            assert c["step"].isdigit() and c["title"] and c["body"], (lang, sid)
            assert len(c["title"]) <= 40 and len(c["body"]) <= MAX_BODY, (lang, sid)
            assert c["body"].endswith("."), (lang, sid)          # read aloud as a sentence


def test_a_narration_clip_is_asked_once_then_read_from_the_cache(tmp_path, monkeypatch):
    calls = []

    class Answer:
        def __init__(self, data):
            self.data = data

        def read(self):
            return self.data

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def fake_urlopen(req, timeout):
        calls.append((req.full_url, req.get_header("Authorization"), json.loads(req.data)))
        return Answer(b"\x01\x00" * render.TTS_RATE)               # one second of sound

    monkeypatch.setattr(render.urllib.request, "urlopen", fake_urlopen)
    voice = render.Voice(api_key="k", base_url="https://tts.example/v1/")
    first = render.speak("Each article gets a score.", voice, str(tmp_path))
    again = render.speak("Each article gets a score.", voice, str(tmp_path))
    assert first == again and len(calls) == 1
    url, auth, body = calls[0]
    assert url == "https://tts.example/v1/audio/speech" and auth == "Bearer k"
    assert body["input"] == "Each article gets a score." and body["response_format"] == "pcm"
    assert "instructions" in body                                  # a gpt-4o-mini-tts voice takes them
    assert abs(render.pcm_seconds(first) - 1.0) < 1e-9
    render.speak("Another sentence.", render.Voice(api_key="k", model="tts-1"), str(tmp_path))
    assert "instructions" not in calls[-1][2]                       # tts-1 refuses them


# ─────────────────────────────────────────────────────────────────────────────
# The site plays it: the build serves docs/demo/ under /demo/, the header opens it
# ─────────────────────────────────────────────────────────────────────────────
def test_the_site_serves_the_video_and_the_header_plays_it():
    """The video lived in the repository only (the README links docs/demo/), and nothing
    put it on the site. The frontend build copies docs/demo/ into its output under /demo/,
    without a second copy in git, and the header's "Demo video" button plays the one in
    the interface's language."""
    with open(os.path.join(ROOT, "frontend", "vite.config.ts"), encoding="utf-8") as f:
        vite = f.read()
    assert "'../docs/demo/'" in vite and "demoVideo()" in vite and "closeBundle" in vite
    for ext in (".mp4", ".vtt", ".jpg"):
        assert f"'{ext}'" in vite, ext
    with open(os.path.join(ROOT, "frontend", "src", "App.tsx"), encoding="utf-8") as f:
        app = f.read()
    assert "`/demo/literev-demo-${lang}.mp4`" in app and "`/demo/literev-demo-${lang}.jpg`" in app
    assert 't("header.demoVideo")' in app
    for lang in ("en", "fr"):
        assert os.path.isfile(os.path.join(ROOT, "docs", "demo", f"literev-demo-{lang}.mp4")), lang
        with open(os.path.join(ROOT, "frontend", "src", "i18n", "locales", f"{lang}.ts"), encoding="utf-8") as f:
            locale = f.read()
        for key in ("demoVideo:", "demoVideoTitle:", "demoVideoNote:", "demoVideoOpen:", "demoVideoClose:"):
            assert key in locale, (lang, key)
