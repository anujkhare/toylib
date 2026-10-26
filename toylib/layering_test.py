"""Check the import direction between layers (see README.md, "Rules between layers").

    toylib.nn, toylib.data   core: import nothing else from the repo
    toylib.harness           may import core
    toylib.viz               imports nothing else from the repo
    toylib                   never imports toylib_projects
    toylib_projects.pNN_*    posts: may import toylib, never another project

Imports inside test files are exempt: a test may build fixtures from any layer.
"""

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Package (path relative to repo root) -> import prefixes it must not use.
FORBIDDEN = {
    "toylib/nn": ("toylib.harness", "toylib.viz", "toylib.data", "toylib_projects"),
    "toylib/data": ("toylib.harness", "toylib.viz", "toylib.nn", "toylib_projects"),
    "toylib/harness": ("toylib.viz", "toylib_projects"),
    "toylib/viz": ("toylib.harness", "toylib.nn", "toylib.data", "toylib_projects"),
}


def _imports(path: pathlib.Path) -> list[str]:
    tree = ast.parse(path.read_text(), filename=str(path))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.append(node.module)
            names.extend(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def _violations(package: str, forbidden: tuple[str, ...]) -> list[str]:
    found = []
    for path in sorted((ROOT / package).rglob("*.py")):
        if path.name.endswith("_test.py"):
            continue
        for name in _imports(path):
            if any(name == f or name.startswith(f + ".") for f in forbidden):
                found.append(f"{path.relative_to(ROOT)} imports {name}")
    return found


@pytest.mark.parametrize("package", sorted(FORBIDDEN))
def test_library_layers(package):
    assert _violations(package, FORBIDDEN[package]) == []


def test_posts_are_independent():
    """Each post imports only toylib and itself, so it can be read on its own."""
    projects_dir = ROOT / "toylib_projects"
    projects = sorted(
        p.name for p in projects_dir.iterdir() if p.is_dir() and (p / "__init__.py").exists()
    )
    posts = [p for p in projects if re.fullmatch(r"p\d\d_.*", p)]
    problems = []
    for post in posts:
        others = tuple(f"toylib_projects.{p}" for p in projects if p != post)
        problems += _violations(f"toylib_projects/{post}", others)
    assert problems == []
