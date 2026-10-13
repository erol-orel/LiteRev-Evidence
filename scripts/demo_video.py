#!/usr/bin/env python3
"""Record the demo video of the main pipeline from the real interface, and render it.

The video is code: the interface changes, the script runs again, the video follows.

  python3 scripts/demo_video.py                       # English, into docs/demo/
  python3 scripts/demo_video.py --lang fr             # French
  python3 scripts/demo_video.py --lang en --lang fr   # both

By default, like the browser smoke test (scripts/smoke_e2e.py), the script:

  1. creates a throwaway database next to DB_URL and applies schema.sql;
  2. boots the API on it WITHOUT any LLM key: what needs an LLM (the narrative brief, the
     assistant's answers, the cluster summaries) is shown as the app shows it without one,
     never faked;
  3. seeds the synthetic demo corpus into the built-in demo scenario, whose model the API
     trains at startup on the real EAWAG wastewater series (scripts/demo_corpus.py);
  4. builds the frontend and runs the tour with Playwright (frontend/demo/tour.spec.ts),
     which writes screenshots, the boxes to frame and one caption per scene;
  5. renders the MP4 and its WebVTT subtitles (scripts/demo_render.py);
  6. stops the API and drops the database (unless --keep-db).

With --base-url the tour films a running deployment instead, on --scenario and its real
data. It only reads: it types a question without launching the search, opens tabs and
expands an article. No database is touched and no corpus is seeded.

Narration: set DEMO_TTS_API_KEY (an OpenAI key, or the key of any API that speaks
/v1/audio/speech: DEMO_TTS_BASE_URL) and each caption is also read aloud. --narrate does
the same with OPENAI_API_KEY. DEMO_TTS_MODEL and DEMO_TTS_VOICE choose the voice. The
clips are cached in the work directory, so a re-render does not pay for them twice.

Needs: the backend dependencies plus uvicorn (as for the smoke test), Pillow, ffmpeg,
Node with the frontend dev dependencies (npm ci in frontend/) and a Playwright Chromium
(npx playwright install chromium in frontend/). The captions live in
frontend/demo/storyboard.ts; docs/demo-video.md explains the whole chain.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND = os.path.join(ROOT, "frontend")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import demo_render  # noqa: E402  (scripts/ on the path, as smoke_e2e does)

DEMO_SCENARIO_ID = "usr-deadbeef0001"   # demo_seed.DEMO_SCENARIO_ID, without importing the app


def _log(msg: str) -> None:
    print(f"[demo] {msg}", file=sys.stderr, flush=True)


def _get_json(url: str, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def _wait_for_health(base: str, proc: subprocess.Popen, timeout: float = 180) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"the API exited with code {proc.returncode} before answering /health")
        with contextlib.suppress(Exception):
            if _get_json(f"{base}/health", timeout=5).get("status"):
                return
        time.sleep(1)
    raise RuntimeError(f"the API did not answer /health within {timeout:.0f} s")


def _tail(path: str, n: int = 40) -> str:
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            return "".join(f.readlines()[-n:])
    except OSError:
        return ""


@contextlib.contextmanager
def throwaway_database(db_url: str, name: str | None, keep: bool):
    """A new database on the server of `db_url`, with schema.sql applied; dropped on exit."""
    import sqlalchemy as sa
    from sqlalchemy.engine import make_url
    import db_bootstrap

    url = make_url(db_url)
    db_name = name or f"literev_demo_{uuid.uuid4().hex[:8]}"
    admin = sa.create_engine(url.set(database="postgres"), isolation_level="AUTOCOMMIT", pool_pre_ping=True)

    def drop() -> None:
        with admin.connect() as c:
            c.execute(sa.text("SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                              "WHERE datname = :n AND pid <> pg_backend_pid()"), {"n": db_name})
            c.execute(sa.text(f'DROP DATABASE IF EXISTS "{db_name}"'))

    drop()
    with admin.connect() as c:
        c.execute(sa.text(f'CREATE DATABASE "{db_name}"'))
    demo_url = url.set(database=db_name)
    try:
        db_bootstrap.apply_schema(demo_url)
        _log(f"database {db_name} created")
        # render_as_string(hide_password=False): str(URL) masks the password as "***".
        yield demo_url.render_as_string(hide_password=False)
    finally:
        if keep:
            _log(f"database kept: {db_name}")
        else:
            drop()
            _log(f"database {db_name} dropped")
        admin.dispose()


@contextlib.contextmanager
def running_api(db_url: str, port: int, api_key: str, work: str):
    """The API on the throwaway database, with no LLM key and no live source key."""
    log_path = os.path.join(work, "api.log")
    env = {**os.environ, "DB_URL": db_url, "WRITE_API_KEY": api_key, "ADMIN_API_KEY": api_key,
           "OPENAI_API_KEY": "", "COHERE_API_KEY": "", "SEMANTIC_SCHOLAR_API_KEY": "",
           "OPENALEX_API_KEY": "", "WARM_ON_STARTUP": "0", "AUTO_PIPELINE_AFTER_SEARCH": "0",
           # The demo scenario's dataset and model are written here, not in /var/lib.
           "MODEL_DATA_DIR": os.path.join(work, "model_data"), "PYTHONUNBUFFERED": "1"}
    log = open(log_path, "w", encoding="utf-8")
    proc = subprocess.Popen([sys.executable, "-m", "uvicorn", "main:app", "--host", "127.0.0.1",
                             "--port", str(port), "--log-level", "warning"],
                            cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    try:
        _wait_for_health(base, proc)
        _log(f"API up on {base} (log: {log_path})")
        yield base
    except Exception:
        _log("last lines of the API log:\n" + _tail(log_path))
        raise
    finally:
        proc.terminate()
        try:
            proc.wait(15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log.close()


def seed(db_url: str, base: str, api_key: str, articles: int) -> dict:
    """The synthetic corpus, in a process of its own: the app's modules it imports run their
    startup DDL against DB_URL, which must be the throwaway database."""
    env = {**os.environ, "DB_URL": db_url, "WRITE_API_KEY": api_key, "OPENAI_API_KEY": ""}
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "demo_corpus.py"), "--db-url", db_url,
                          "--api", base, "--api-key", api_key, "--articles", str(articles)],
                         cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    return json.loads(out.stdout)


def warm_up(base: str, sid: str, langs: list[str], timeout: float = 240) -> None:
    """Compute the clustering before the tour films it (UMAP and HDBSCAN on a cold cache)."""
    for lang in langs:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with contextlib.suppress(Exception):
                if _get_json(f"{base}/user-scenarios/{sid}/clustering?lang={lang}", timeout=60).get("clusters"):
                    break
            time.sleep(3)
        else:
            _log(f"the clustering ({lang}) was not ready after {timeout:.0f} s: the tour will wait for it")


def record(lang: str, frames: str, *, base_url: str | None, web_port: int, api_base: str | None,
           scenario: str, api_key: str, synthetic: bool) -> None:
    """Run the tour: screenshots, overlays and manifest.json into `frames`."""
    shutil.rmtree(frames, ignore_errors=True)
    os.makedirs(frames)
    env = {**os.environ, "DEMO_OUT": frames, "DEMO_LANG": lang, "DEMO_SCENARIO_ID": scenario,
           "DEMO_API_KEY": api_key, "DEMO_SYNTHETIC": "1" if synthetic else "0",
           "DEMO_STRICT": "1" if synthetic else "0", "E2E_WEB_PORT": str(web_port)}
    if base_url:
        env["DEMO_BASE_URL"] = base_url
    else:
        env.pop("DEMO_BASE_URL", None)
        env["VITE_API_PROXY_TARGET"] = api_base or ""
    _log(f"recording the tour ({lang})")
    rc = subprocess.run(["npx", "playwright", "test", "-c", "playwright.demo.config.ts"],
                        cwd=FRONTEND, env=env).returncode
    if rc != 0:
        raise RuntimeError(f"the tour failed ({lang}): Playwright exited with code {rc}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lang", action="append", choices=("en", "fr"), help="interface and caption language (repeatable)")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "docs", "demo"), help="where the videos go")
    ap.add_argument("--base-url", default=None, help="film this running deployment instead (read only)")
    ap.add_argument("--scenario", default=None, help="scenario to film (default: the built-in demo scenario)")
    ap.add_argument("--db-url", default=os.environ.get("DB_URL"),
                    help="Postgres URL (any database on the server; the throwaway one is created next to it)")
    ap.add_argument("--db-name", default=None, help="name of the throwaway database")
    ap.add_argument("--keep-db", action="store_true", help="keep the throwaway database at the end")
    ap.add_argument("--api-port", type=int, default=8775)
    ap.add_argument("--web-port", type=int, default=4175)
    ap.add_argument("--articles", type=int, default=200, help="size of the synthetic corpus")
    ap.add_argument("--no-build", action="store_true", help="serve the existing frontend/dist")
    ap.add_argument("--work", default=None, help="working directory (default: a temporary one)")
    ap.add_argument("--keep-work", action="store_true", help="keep the frames, to re-render with demo_render.py")
    ap.add_argument("--narrate", action="store_true", help="read the captions aloud (DEMO_TTS_API_KEY or OPENAI_API_KEY)")
    ap.add_argument("--no-narration", action="store_true", help="never narrate, even with DEMO_TTS_API_KEY set")
    ap.add_argument("--crf", type=int, default=28, help="x264 quality: lower is sharper and bigger")
    args = ap.parse_args()
    langs = args.lang or ["en"]
    remote = bool(args.base_url)
    if not remote and not args.db_url:
        ap.error("--db-url or DB_URL is required (or --base-url to film a running deployment)")
    if remote and not args.scenario:
        ap.error("--scenario is required with --base-url")
    if not shutil.which("ffmpeg"):
        ap.error("ffmpeg is required (apt install ffmpeg)")
    if not os.path.isdir(os.path.join(FRONTEND, "node_modules")):
        ap.error("run `npm ci` in frontend/ first")
    voice = None if args.no_narration else demo_render.voice_from_env(required=args.narrate)
    # Absolute: the tour runs from frontend/ and would read a relative path from there.
    work = os.path.abspath(args.work or tempfile.mkdtemp(prefix="literev-demo-"))
    os.makedirs(work, exist_ok=True)
    out_dir = os.path.abspath(args.out_dir)
    scenario = args.scenario or DEMO_SCENARIO_ID
    results = []

    def film_and_render(api_base: str | None, api_key: str) -> None:
        for lang in langs:
            frames = os.path.join(work, lang)
            record(lang, frames, base_url=args.base_url, web_port=args.web_port, api_base=api_base,
                   scenario=scenario, api_key=api_key, synthetic=not remote)
            out = os.path.join(out_dir, f"literev-demo-{lang}.mp4")
            results.append(demo_render.render(frames, out, voice, args.crf, log=_log))

    try:
        if remote:
            film_and_render(None, os.environ.get("DEMO_API_KEY", ""))
        else:
            api_key = f"demo-{uuid.uuid4().hex[:12]}"
            with throwaway_database(args.db_url, args.db_name, args.keep_db) as db_url, \
                    running_api(db_url, args.api_port, api_key, work) as base:
                summary = seed(db_url, base, api_key, args.articles)
                _log(f"seeded {summary['articles']} synthetic articles into {summary['scenario']}")
                warm_up(base, scenario, langs)
                if not args.no_build:
                    _log("building the frontend")
                    subprocess.run(["npm", "run", "build"], cwd=FRONTEND, check=True,
                                   env={**os.environ, "VITE_API_PROXY_TARGET": base})
                film_and_render(base, api_key)
    finally:
        if args.keep_work or args.work:
            _log(f"frames kept in {work} (re-render: python3 scripts/demo_render.py --frames {work}/<lang> -o ...)")
        else:
            shutil.rmtree(work, ignore_errors=True)
    print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
