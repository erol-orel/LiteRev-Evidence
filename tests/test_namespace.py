"""Every name an `api` module defines is re-exported by `main`, into ONE namespace.

`main.py` copies the names of each module, in order, into its own globals, so two modules that
define the same top-level name do not conflict where they are used (each uses its own) but the
LATER one silently replaces the other for everything that reads `main.<name>`: the tests, the
scripts, and `patch_app`. A new report module once defined `to_markdown` and quietly took it over
from the exports module. This pins that the modules of the structured extraction never define a
name that another module defines too."""
import ast
import glob
import os
from collections import defaultdict

NEW = {"codebook", "extraction", "extraction_review", "pooling", "extraction_report", "geography"}


def _top_level_names(path):
    names = []
    for node in ast.parse(open(path, encoding="utf-8").read()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, ast.Assign):
            names += [t.id for t in node.targets if isinstance(t, ast.Name)]
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.append(node.target.id)
    return [n for n in names if not n.startswith("__")]


def test_the_extraction_modules_define_no_name_that_another_module_defines():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    owners = defaultdict(set)
    for path in glob.glob(os.path.join(root, "api", "*.py")):
        mod = os.path.basename(path)[:-3]
        if mod == "__init__":
            continue
        for n in _top_level_names(path):
            owners[n].add(mod)
    clashes = {n: sorted(m) for n, m in owners.items() if len(m) > 1 and m & NEW}
    assert clashes == {}, f"a name defined in two api modules is taken over by the later one in `main`: {clashes}"


def test_main_still_exposes_the_functions_the_older_modules_define():
    import main
    from api import exports
    assert main.to_markdown is exports.to_markdown
