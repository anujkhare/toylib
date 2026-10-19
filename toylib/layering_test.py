"""Check the import direction between layers (see README.md, "Rules between layers").

    toylib.nn, toylib.data   core: import nothing else from the repo
    toylib.harness           may import core
    toylib.viz               imports nothing else from the repo
    toylib_posts.<post>      may import toylib; not other posts, not toylib_projects

Imports inside test files are exempt: a test may build fixtures from any layer.
"""

import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parent.parent

# Package (path relative to repo root) -> import prefixes it must not use.
FORBIDDEN = {
    "toylib/nn": ("toylib.harness", "toylib.viz", "toylib.data", "toylib_posts", "toylib_projects"),
    "toylib/data": ("toylib.harness", "toylib.viz", "toylib.nn", "toylib_posts", "toylib_projects"),
    "toylib/harness": ("toylib.viz", "toylib_posts", "toylib_projects"),
    "toylib/viz": ("toylib.harness", "toylib.nn", "toylib.data", "toylib_posts", "toylib_projects"),
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
    posts_dir = ROOT / "toylib_posts"
    posts = sorted(p.name for p in posts_dir.iterdir() if p.is_dir() and p.name.startswith("p"))
    problems = []
    for post in posts:
        others = tuple(f"toylib_posts.{p}" for p in posts if p != post)
        problems += _violations(f"toylib_posts/{post}", others + ("toylib_projects",))
    assert problems == []
