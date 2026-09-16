"""No em dash anywhere in the repository: code, comments, interface strings, prompts,
documentation, scripts, workflows. A house rule (see CLAUDE.md); this test is what keeps
it in force. Use a comma, a colon, a plain hyphen or a middle dot instead."""
import os
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TEXT_EXT = (".py", ".md", ".ts", ".tsx", ".js", ".sh", ".yml", ".yaml", ".sql", ".example", ".txt",
            ".json", ".css", ".html", ".toml", ".cfg", ".ini")
EM_DASH = chr(0x2014)
SKIP_PREFIXES = ("frontend/package-lock.json", "frontend/dist/", "frontend/node_modules/", ".git/")


def _tracked_files() -> list[str]:
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True, timeout=30).stdout
        files = [f for f in out.decode("utf-8", "replace").split("\0") if f]
        if files:
            return files
    except Exception:  # pragma: no cover - no git on the machine: walk the tree
        pass
    found = []
    for base, dirs, names in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in (".git", "node_modules", "dist", ".venv", "__pycache__")]
        for n in names:
            found.append(os.path.relpath(os.path.join(base, n), ROOT))
    return found


def test_no_em_dash_in_the_repository():
    offenders = []
    for rel in _tracked_files():
        if rel.startswith(SKIP_PREFIXES) or not rel.endswith(TEXT_EXT):
            continue
        path = os.path.join(ROOT, rel)
        try:
            with open(path, encoding="utf-8") as fh:
                for i, line in enumerate(fh, 1):
                    if EM_DASH in line:
                        offenders.append(f"{rel}:{i}")
                        if len(offenders) > 40:
                            break
        except (UnicodeDecodeError, FileNotFoundError):
            continue
    assert not offenders, "em dashes found (house rule: none, anywhere):\n" + "\n".join(offenders)
