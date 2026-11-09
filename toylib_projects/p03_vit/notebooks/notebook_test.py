"""Run the Colab notebook's cells locally, with the stand-in model and synthetic data.

Locally the notebook trains the ``smoke`` run, so this checks every cell end to end in
seconds. The Colab-only lines (shell commands and ``%cd``) are never reached here, so
they're replaced with a call that fails if it ever runs.
"""

import json
import pathlib
import re

import pytest

from toylib_projects.p03_vit import imagenette
from toylib_projects.p03_vit import testing
from toylib_projects.p03_vit import train

NOTEBOOK = pathlib.Path(__file__).parent / "01_imagenette_vit_colab.ipynb"
_MAGIC = re.compile(r"^(\s*)((?:\w+\s*=\s*)?)[!%](.*)$")


def _python(source: str) -> str:
    """Turn IPython's shell lines into calls to ``_colab_only``."""
    lines = []
    for line in source.splitlines():
        match = _MAGIC.match(line)
        if match:
            indent, assign, command = match.groups()
            line = f"{indent}{assign}_colab_only({command!r})"
        lines.append(line)
    return "\n".join(lines)


def _colab_only(command: str):
    raise AssertionError(f"A Colab-only line ran locally: {command}")


def test_notebook_runs_locally(monkeypatch, tmp_path):
    testing.use_stand_in(monkeypatch)
    monkeypatch.setattr(imagenette, "load", testing.SyntheticImagenette(n_train=64, n_test=64))
    monkeypatch.setattr(train, "OUTPUT_DIR", tmp_path)
    monkeypatch.delenv("RUNS_TO_TRAIN", raising=False)
    monkeypatch.delenv("COLAB_RELEASE_TAG", raising=False)

    cells = json.loads(NOTEBOOK.read_text())["cells"]
    namespace = {"_colab_only": _colab_only}
    for cell in cells:
        if cell["cell_type"] == "code":
            exec(compile(_python("".join(cell["source"])), str(NOTEBOOK), "exec"), namespace)

    assert namespace["IN_COLAB"] is False
    assert list(namespace["results"]) == ["smoke"]
    assert 0.0 <= namespace["results"]["smoke"]["full_test_accuracy"] <= 1.0
    assert (tmp_path / "smoke_e2" / "checkpoints").is_dir()


def test_notebook_has_no_outputs():
    """Outputs belong in the gitignored ``_output.ipynb`` that ``colab exec`` writes."""
    cells = json.loads(NOTEBOOK.read_text())["cells"]
    assert all(not cell.get("outputs") for cell in cells if cell["cell_type"] == "code")


@pytest.mark.parametrize("line, expected", [
    ("    !pip install x", "    _colab_only('pip install x')"),
    ("    %cd /content", "    _colab_only('cd /content')"),
    ("x = !ls", "x = _colab_only('ls')"),
    ("print('!')", "print('!')"),
])
def test_shell_lines_become_calls(line, expected):
    assert _python(line) == expected
