#!/usr/bin/env python3
"""Render the demo video from what the tour recorded (frontend/demo/tour.spec.ts).

The tour writes screenshots, the boxes to frame in each of them, one caption overlay per
scene and manifest.json. This module turns them into a 1920 x 1080 MP4: a camera that
glides from one framed box to the next, a spotlight on the box being explained, the
caption on top, crossfades between scenes and, when a text-to-speech key is given, a
narration read from the captions, with WebVTT subtitles beside the video.

Frames are composed with Pillow and piped to ffmpeg (libx264). Everything that decides
what is on screen and when (`camera`, `plan`, `vtt`) is pure, so the tests pin it without
a browser or an encoder.

  python3 scripts/demo_render.py --frames /tmp/demo-en -o docs/demo/literev-demo-en.mp4
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request
import wave
from dataclasses import dataclass, field, replace

OUT_W, OUT_H = 1920, 1080
ASPECT = OUT_W / OUT_H
FPS = 30
#: The closest the camera gets: a third of a 1440 px page is about two words wide, and
#: below 900 px the zoom shows the pixels of a scale-2 screenshot.
MIN_CAMERA_W = 900.0
#: Room left around a framed box, as a share of its size.
MARGIN = 0.06
FADE_SCENE = 0.7       # between two scenes: out to the background, then in
BACKGROUND = (11, 20, 17)  # the app's darkest green, which the dip passes through
FADE_BEAT = 0.45       # crossfade between two screenshots of one scene
MOVE = 0.9             # camera move between two boxes of one screenshot
SPOT_IN = 0.35         # spotlight fading in once the camera has arrived
PAN_REST = 0.5         # pause before and after a pan over a box too tall for the frame
NARRATION_DELAY = 0.3  # the voice starts once the new scene has begun to show
NARRATION_TAIL = 0.7   # and the scene lasts that much longer than the voice
GOLD = (217, 179, 92)
DIM = 0.42             # darkening outside the spotlight


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def lerp(self, other: "Rect", t: float) -> "Rect":
        return Rect(*(a + (b - a) * t for a, b in zip((self.x, self.y, self.w, self.h),
                                                       (other.x, other.y, other.w, other.h))))

    @staticmethod
    def of(d: dict | None) -> "Rect | None":
        return Rect(float(d["x"]), float(d["y"]), float(d["w"]), float(d["h"])) if d else None


@dataclass
class Beat:
    image: str
    size: tuple[float, float]
    focus: Rect | None
    spot: bool
    hold: float
    cut: bool = False


@dataclass
class Scene:
    id: str
    kind: str
    narration: str
    overlay: str | None
    band: float
    beats: list[Beat]
    caption: dict | None = None
    #: Seconds of recorded narration, when there is one.
    voice: float = 0.0


@dataclass(frozen=True)
class Shot:
    """What one frame shows: a screenshot through a camera, a spotlight, an overlay."""
    image: str
    size: tuple[float, float]
    camera: Rect
    spot: Rect | None
    overlay: str | None
    spot_alpha: float = 1.0


@dataclass
class Segment:
    kind: str                  # hold | move | pan | fade | dip | spot
    a: Shot
    b: Shot | None
    seconds: float
    scene: str = ""


@dataclass
class Timeline:
    segments: list[Segment] = field(default_factory=list)
    #: (scene id, start, end) in seconds, for the narration and the subtitles.
    scenes: list[tuple[str, float, float]] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return sum(s.seconds for s in self.segments)


def load_manifest(frames_dir: str) -> tuple[dict, list[Scene]]:
    with open(os.path.join(frames_dir, "manifest.json"), encoding="utf-8") as f:
        m = json.load(f)
    scenes = []
    for s in m["scenes"]:
        beats = [Beat(image=b["image"], size=(float(b["size"][0]), float(b["size"][1])),
                      focus=Rect.of(b.get("focus")), spot=bool(b.get("spot")), hold=float(b["hold"]),
                      cut=bool(b.get("cut"))) for b in s["beats"]]
        scenes.append(Scene(id=s["id"], kind=s["kind"], narration=s.get("narration") or "",
                            overlay=s.get("overlay"), band=float(s.get("band") or 0), beats=beats,
                            caption=s.get("caption")))
    return m, scenes


# ─────────────────────────────────────────────────────────────────────────────
# Where the camera looks
# ─────────────────────────────────────────────────────────────────────────────
def camera(size: tuple[float, float], focus: Rect | None, band: float = 0.0) -> tuple[Rect, Rect | None]:
    """The camera for a box, in the CSS pixels of its screenshot, and where it ends when
    the box is too tall to show at once (it then pans down the box). PURE.

    The camera keeps the 16:9 of the output, never leaves the screenshot, never comes
    closer than MIN_CAMERA_W, and keeps the box in the part of the frame the caption does
    not cover: the top (1 - band)."""
    w_img, h_img = size
    visible = max(0.2, 1.0 - band)
    if focus is None:                                  # the whole width, from the top
        cw = w_img
        ch = cw / ASPECT
        if ch > h_img:
            ch = h_img
            cw = ch * ASPECT
        return Rect((w_img - cw) / 2, 0.0, cw, ch), None
    fw, fh = focus.w * (1 + 2 * MARGIN), focus.h * (1 + 2 * MARGIN)
    cw = max(fw, fh * ASPECT / visible, min(MIN_CAMERA_W, w_img))
    cw = min(cw, w_img, h_img * ASPECT)
    ch = cw / ASPECT
    x = min(max(focus.cx - cw / 2, 0.0), w_img - cw)
    window = ch * visible                              # the height the box can use
    if fh <= window:
        y = min(max(focus.cy - window / 2, 0.0), h_img - ch)
        return Rect(x, y, cw, ch), None
    top = focus.y - focus.h * MARGIN
    bottom = focus.y + focus.h * (1 + MARGIN) - window
    start = Rect(x, min(max(top, 0.0), h_img - ch), cw, ch)
    end = Rect(x, min(max(bottom, 0.0), h_img - ch), cw, ch)
    return start, (end if abs(end.y - start.y) > 1 else None)


def _ease(t: float) -> float:
    """Ease in and out: the camera starts and stops gently."""
    t = min(max(t, 0.0), 1.0)
    return t * t * (3 - 2 * t)


# ─────────────────────────────────────────────────────────────────────────────
# When
# ─────────────────────────────────────────────────────────────────────────────
def plan(scenes: list[Scene]) -> Timeline:
    """The segments of the video, scene by scene. PURE.

    A new scene crossfades in; inside a scene the camera moves between two boxes of the
    same screenshot, crossfades between two screenshots, and cuts between typing frames.
    A spotlight fades in once the camera has arrived. A narrated scene is held until its
    voice has finished."""
    tl = Timeline()
    prev: Shot | None = None
    prev_image = None
    for scene in scenes:
        start = tl.seconds
        for j, b in enumerate(scene.beats):
            cam, pan_end = camera(b.size, b.focus, scene.band)
            spot = b.focus if b.spot and b.focus else None
            shot = Shot(b.image, b.size, cam, spot, scene.overlay)
            bare = replace(shot, spot_alpha=0.0)
            if prev is not None:
                if j == 0:
                    tl.segments.append(Segment("dip", prev, bare, FADE_SCENE, scene.id))
                elif b.cut:
                    pass
                elif b.image == prev_image:
                    tl.segments.append(Segment("move", replace(prev, spot_alpha=0.0), bare, MOVE, scene.id))
                else:
                    tl.segments.append(Segment("fade", prev, bare, FADE_BEAT, scene.id))
            hold = b.hold
            if spot is not None:
                tl.segments.append(Segment("spot", bare, shot, SPOT_IN, scene.id))
                hold = max(0.0, hold - SPOT_IN)
            if pan_end is not None:
                end = replace(shot, camera=pan_end)
                rest = min(PAN_REST, hold / 4)
                tl.segments.append(Segment("hold", shot, None, rest, scene.id))
                tl.segments.append(Segment("pan", shot, end, max(0.0, hold - 2 * rest), scene.id))
                tl.segments.append(Segment("hold", end, None, rest, scene.id))
                shot = end
            else:
                tl.segments.append(Segment("hold", shot, None, hold, scene.id))
            prev, prev_image = shot, b.image
        # A narrated scene lasts as long as its voice, and a little more.
        needed = NARRATION_DELAY + scene.voice + NARRATION_TAIL if scene.voice else 0.0
        short = needed - (tl.seconds - start)
        if short > 0 and prev is not None:
            tl.segments.append(Segment("hold", prev, None, short, scene.id))
        tl.scenes.append((scene.id, start, tl.seconds))
    return tl


def poster_shot(tl: Timeline, prefer: tuple[str, ...] = ("corpus", "evidence")) -> Shot | None:
    """The still that stands for the video: the longest hold of the first preferred scene
    the video has, else of its first screen. Caption and badge included: a poster of
    synthetic data says so too. PURE."""
    holds = [s for s in tl.segments if s.kind == "hold" and s.a.overlay]
    for sid in prefer:
        mine = [s for s in holds if s.scene == sid]
        if mine:
            return max(mine, key=lambda s: s.seconds).a
    return holds[0].a if holds else None


def vtt(scenes: list[Scene], tl: Timeline) -> str:
    """WebVTT subtitles: each scene's caption (or a card's narration) for its duration."""
    def ts(t: float) -> str:
        ms = int(round(t * 1000))
        return f"{ms // 3_600_000:02d}:{ms // 60_000 % 60:02d}:{ms // 1000 % 60:02d}.{ms % 1000:03d}"
    by_id = {s.id: s for s in scenes}
    cues = ["WEBVTT", ""]
    for sid, start, end in tl.scenes:
        s = by_id[sid]
        text = (f"{s.caption['title']}\n{s.caption['body']}" if s.caption else s.narration).strip()
        if text:
            cues += [ts(start) + " --> " + ts(end), text, ""]
    return "\n".join(cues)


# ─────────────────────────────────────────────────────────────────────────────
# What: composing frames
# ─────────────────────────────────────────────────────────────────────────────
def _inside(box: tuple[float, float, float, float], size: tuple[int, int]) -> tuple[float, float, float, float]:
    """The box moved back inside the image, then cut to it. A recorded size is in CSS
    pixels and can be fractional, a screenshot is in whole device pixels: the camera can
    overshoot by a fraction of a pixel. PURE."""
    x0, y0, x1, y1 = box
    w, h = size
    dx = min(0.0, w - x1) or max(0.0, -x0)
    dy = min(0.0, h - y1) or max(0.0, -y0)
    x0, x1, y0, y1 = x0 + dx, x1 + dx, y0 + dy, y1 + dy
    return max(0.0, x0), max(0.0, y0), min(float(w), x1), min(float(h), y1)


class Composer:
    """Turns shots into 1920 x 1080 RGB frames. Keeps the last few screenshots in memory:
    a tall screenshot at scale 2 weighs tens of megabytes."""

    def __init__(self, frames_dir: str, keep: int = 3):
        from PIL import Image
        self.Image = Image
        self.dir = frames_dir
        self.keep = keep
        self._images: dict[str, object] = {}
        self._overlays: dict[str, object] = {}
        self._last_view: tuple | None = None

    def _image(self, rel: str):
        if rel not in self._images:
            if len(self._images) >= self.keep:
                self._images.pop(next(iter(self._images)))
            self._images[rel] = self.Image.open(os.path.join(self.dir, rel)).convert("RGB")
        return self._images[rel]

    def _overlay(self, rel: str):
        if rel not in self._overlays:
            img = self.Image.open(os.path.join(self.dir, rel)).convert("RGBA")
            if img.size != (OUT_W, OUT_H):
                img = img.resize((OUT_W, OUT_H), self.Image.LANCZOS)
            self._overlays[rel] = img
        return self._overlays[rel]

    def _view(self, shot: Shot, fast: bool):
        """The screenshot through the camera, at the output size. The still views are
        cached: a hold and a spotlight fading in reuse the same one."""
        key = (shot.image, shot.camera)
        if not fast and self._last_view and self._last_view[0] == key:
            return self._last_view[1]
        img = self._image(shot.image)
        k = img.size[0] / shot.size[0]                 # image pixels per CSS pixel
        c = shot.camera
        box = _inside((c.x * k, c.y * k, (c.x + c.w) * k, (c.y + c.h) * k), img.size)
        view = img.resize((OUT_W, OUT_H), self.Image.BILINEAR if fast else self.Image.LANCZOS, box=box)
        if not fast:
            self._last_view = (key, view)
        return view

    def frame(self, shot: Shot, fast: bool = False):
        from PIL import ImageDraw
        c = shot.camera
        out = self._view(shot, fast)
        if shot.spot is not None and shot.spot_alpha > 0:
            s = OUT_W / c.w
            x0, y0 = (shot.spot.x - c.x) * s, (shot.spot.y - c.y) * s
            x1, y1 = x0 + shot.spot.w * s, y0 + shot.spot.h * s
            a = shot.spot_alpha
            shade = self.Image.new("RGBA", (OUT_W, OUT_H), (0, 0, 0, int(255 * DIM * a)))
            ImageDraw.Draw(shade).rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(0, 0, 0, 0))
            ring = ImageDraw.Draw(shade)
            ring.rounded_rectangle((x0 - 3, y0 - 3, x1 + 3, y1 + 3), radius=21,
                                   outline=GOLD + (int(235 * a),), width=4)
            out = self.Image.alpha_composite(out.convert("RGBA"), shade).convert("RGB")
        if shot.overlay:
            out = self.Image.alpha_composite(out.convert("RGBA"), self._overlay(shot.overlay)).convert("RGB")
        return out


def frames(tl: Timeline, composer: Composer):
    """Every frame of the video, as RGB bytes, in order."""
    from PIL import Image
    clock = 0.0
    done = 0
    for seg in tl.segments:
        clock += seg.seconds
        n = int(round(clock * FPS)) - done             # frame-exact over the whole video
        done += n
        if n <= 0:
            continue
        if seg.kind == "hold":
            still = composer.frame(seg.a).tobytes()
            for _ in range(n):
                yield still
        elif seg.kind == "fade":
            a, b = composer.frame(seg.a), composer.frame(seg.b)
            for i in range(n):
                yield Image.blend(a, b, _ease((i + 1) / n)).tobytes()
        elif seg.kind == "dip":
            a, b = composer.frame(seg.a), composer.frame(seg.b)
            dark = Image.new("RGB", (OUT_W, OUT_H), BACKGROUND)
            for i in range(n):
                t = (i + 1) / n
                yield (Image.blend(a, dark, _ease(t * 2)) if t < 0.5
                       else Image.blend(dark, b, _ease(t * 2 - 1))).tobytes()
        elif seg.kind == "spot":
            for i in range(n):
                yield composer.frame(replace(seg.b, spot_alpha=_ease((i + 1) / n))).tobytes()
        else:                                          # move, pan: the camera travels
            linear = seg.kind == "pan"
            for i in range(n):
                t = (i + 1) / n
                t = 0.5 - 0.5 * math.cos(math.pi * t) if linear else _ease(t)
                shot = replace(seg.b, camera=seg.a.camera.lerp(seg.b.camera, t),
                               spot_alpha=seg.b.spot_alpha)
                yield composer.frame(shot, fast=True).tobytes()


# ─────────────────────────────────────────────────────────────────────────────
# Narration (optional): one clip per scene from a text-to-speech API
# ─────────────────────────────────────────────────────────────────────────────
TTS_RATE = 24_000           # the API's raw PCM: 24 kHz, 16-bit, mono
#: A clip is asked again when the API is slow or busy: a long video asks for dozens, and
#: one request left unanswered for minutes (as on 13 October) must not lose the recording.
#: An answer starts within seconds, so 30 s without a byte is a request the API dropped:
#: on 13 October 5 of 33 requests hung, and each time a later attempt answered at once.
TTS_ATTEMPTS = 6
TTS_TIMEOUT = 30            # seconds without a byte before an attempt is given up
TTS_RETRY_STATUS = {408, 409, 429, 500, 502, 503, 504}


@dataclass
class Voice:
    api_key: str
    model: str = "gpt-4o-mini-tts"
    voice: str = "coral"
    base_url: str = "https://api.openai.com/v1"
    instructions: str = ("Calm, clear and friendly, like the narrator of a short product demo. "
                         "Moderate pace.")


def speak(text: str, voice: Voice, cache_dir: str) -> str:
    """The clip of `text` as raw PCM, cached by its content: re-rendering a video after a
    visual change does not pay for the same narration twice."""
    key = hashlib.sha256(json.dumps([text, voice.model, voice.voice, voice.instructions]).encode()).hexdigest()[:16]
    path = os.path.join(cache_dir, f"{key}.pcm")
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    body = {"model": voice.model, "voice": voice.voice, "input": text, "response_format": "pcm"}
    if voice.model.startswith("gpt-"):                 # the older tts-1 models take no instructions
        body["instructions"] = voice.instructions
    req = urllib.request.Request(voice.base_url.rstrip("/") + "/audio/speech", data=json.dumps(body).encode(),
                                 headers={"Authorization": f"Bearer {voice.api_key}",
                                          "Content-Type": "application/json"}, method="POST")
    for attempt in range(1, TTS_ATTEMPTS + 1):
        try:
            with urllib.request.urlopen(req, timeout=TTS_TIMEOUT) as r:
                pcm = r.read()
            break
        except urllib.error.HTTPError as e:
            if e.code not in TTS_RETRY_STATUS or attempt == TTS_ATTEMPTS:
                # The error's code only: an OpenAI message can quote part of the key.
                raise RuntimeError(f"text-to-speech refused the clip: HTTP {e.code} ({_error_code(e)})") from None
            why = f"HTTP {e.code}"
        except (TimeoutError, ConnectionError, urllib.error.URLError, http.client.HTTPException) as e:
            if attempt == TTS_ATTEMPTS:
                raise RuntimeError(f"text-to-speech did not answer in {TTS_ATTEMPTS} attempts ({type(e).__name__})") from None
            why = type(e).__name__
        wait = min(5 * 2 ** (attempt - 1), 30)
        print(f"text-to-speech: attempt {attempt} failed ({why}), next in {wait} s", file=sys.stderr)
        time.sleep(wait)
    os.makedirs(cache_dir, exist_ok=True)
    with open(path, "wb") as f:
        f.write(pcm)
    return path


def _error_code(e: "urllib.error.HTTPError") -> str:
    """The machine-readable code of an API error (`invalid_api_key`, `insufficient_quota`),
    never its message."""
    try:
        err = json.loads(e.read().decode("utf-8", "replace")).get("error") or {}
        return str(err.get("code") or err.get("type") or "no code")
    except Exception:
        return "no code"


def pcm_seconds(path: str) -> float:
    return os.path.getsize(path) / 2 / TTS_RATE


def narration_track(clips: dict[str, str], tl: Timeline, out_wav: str) -> None:
    """One WAV as long as the video, each scene's clip placed at its scene's start."""
    total = int(math.ceil(tl.seconds * TTS_RATE))
    track = bytearray(total * 2)
    for sid, start, _end in tl.scenes:
        if sid not in clips:
            continue
        with open(clips[sid], "rb") as f:
            pcm = f.read()
        at = int((start + NARRATION_DELAY) * TTS_RATE) * 2
        pcm = pcm[: max(0, len(track) - at)]
        track[at:at + len(pcm)] = pcm
    with wave.open(out_wav, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(TTS_RATE)
        w.writeframes(bytes(track))


# ─────────────────────────────────────────────────────────────────────────────
# Encoding
# ─────────────────────────────────────────────────────────────────────────────
def render(frames_dir: str, out_mp4: str, voice: Voice | None = None, crf: int = 28,
           log=lambda m: print(m, file=sys.stderr)) -> dict:
    """The video of a recording, with its subtitles beside it (same name, .vtt)."""
    manifest, scenes = load_manifest(frames_dir)
    clips: dict[str, str] = {}
    if voice is not None:
        cache = os.path.join(frames_dir, "voice")
        for s in scenes:
            if s.narration:
                clips[s.id] = speak(s.narration, voice, cache)
                s.voice = pcm_seconds(clips[s.id])
        log(f"narration: {len(clips)} clip(s), {sum(s.voice for s in scenes):.1f} s of voice")
    tl = plan(scenes)
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{OUT_W}x{OUT_H}", "-r", str(FPS), "-i", "-"]
    wav = None
    if clips:
        wav = os.path.join(frames_dir, "narration.wav")
        narration_track(clips, tl, wav)
        cmd += ["-i", wav, "-c:a", "aac", "-b:a", "160k"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", str(crf), "-pix_fmt", "yuv420p",
            "-movflags", "+faststart", "-metadata", f"title=LiteRev demo ({manifest.get('lang', 'en')})",
            out_mp4]
    log(f"encoding {tl.seconds:.1f} s ({len(tl.scenes)} scenes) to {out_mp4}")
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    try:
        for data in frames(tl, Composer(frames_dir)):
            proc.stdin.write(data)
    finally:
        proc.stdin.close()
        rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"ffmpeg exited with code {rc}")
    sub = os.path.splitext(out_mp4)[0] + ".vtt"
    with open(sub, "w", encoding="utf-8") as f:
        f.write(vtt(scenes, tl))
    poster = os.path.splitext(out_mp4)[0] + ".jpg"
    shot = poster_shot(tl)
    if shot is not None:
        Composer(frames_dir).frame(shot).save(poster, quality=84, optimize=True)
    return {"video": out_mp4, "subtitles": sub, "poster": poster if shot else None, "seconds": round(tl.seconds, 1),
            "scenes": [sid for sid, _, _ in tl.scenes], "narrated": bool(clips),
            "skipped": manifest.get("skipped", [])}


def voice_from_env(required: bool = False) -> Voice | None:
    """The narration voice: DEMO_TTS_API_KEY, or OPENAI_API_KEY when narration is asked
    for explicitly. DEMO_TTS_MODEL, DEMO_TTS_VOICE and DEMO_TTS_BASE_URL change the rest."""
    key = os.environ.get("DEMO_TTS_API_KEY") or (os.environ.get("OPENAI_API_KEY") if required else "")
    if not key:
        if required:
            raise SystemExit("narration needs DEMO_TTS_API_KEY (or OPENAI_API_KEY)")
        return None
    v = Voice(api_key=key)
    v.model = os.environ.get("DEMO_TTS_MODEL") or v.model
    v.voice = os.environ.get("DEMO_TTS_VOICE") or v.voice
    v.base_url = os.environ.get("DEMO_TTS_BASE_URL") or v.base_url
    return v


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", required=True, help="directory the tour wrote (manifest.json, frames/, overlays/)")
    ap.add_argument("-o", "--out", required=True, help="the MP4 to write (subtitles go beside it, .vtt)")
    ap.add_argument("--narrate", action="store_true", help="add a narration (needs a text-to-speech key)")
    ap.add_argument("--crf", type=int, default=28, help="x264 quality: lower is sharper and bigger")
    args = ap.parse_args()
    voice = voice_from_env(required=args.narrate)
    print(json.dumps(render(args.frames, args.out, voice, args.crf), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
