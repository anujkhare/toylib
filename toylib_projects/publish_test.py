"""Tests for turning a post into website files."""

import pytest

from toylib_projects import publish

POST = publish.Post(slug="001-x", date="2026-01-01", description="d", tags=("a", "b"))


def test_bundle_moves_the_heading_into_front_matter():
    page = publish.to_bundle(POST, "# Why? It's fine\n\nBody.\n")
    assert page.startswith("---\ntitle: \"Why? It's fine\"\n")
    assert 'tags: ["a", "b"]' in page
    assert "canonicalURL" not in page
    assert page.endswith("---\n\nBody.\n")


def test_bundle_needs_a_heading():
    with pytest.raises(ValueError):
        publish.to_bundle(POST, "Body.\n")


def test_embedded_figures_skip_links_and_other_images():
    text = "![a](figures/a.png) [b](figures/b.png) ![c](other/c.png) ![d](figures/d.png)"
    assert publish.embedded_figures(text) == ["a.png", "d.png"]


def test_standalone_page_gets_a_head_and_a_link_back():
    page = publish.to_standalone_page("<title>T</title><footer><p>x</p></footer>", "/posts/x/")
    assert page.startswith("<!doctype html>")
    assert '<meta name="viewport"' in page
    assert '<a href="/posts/x/">Read the post</a>' in page


def test_publish_writes_the_bundle_and_explorer(tmp_path, monkeypatch):
    source = tmp_path / "toy"
    (source / "figures").mkdir(parents=True)
    (source / "explorer").mkdir()
    (source / "POST.md").write_text("# T\n\n![a](figures/a.png)\n")
    (source / "figures" / "a.png").write_bytes(b"a")
    (source / "figures" / "unused.png").write_bytes(b"b")
    for file in publish.EXPLORER_FILES:
        (source / "explorer" / file).write_text("<footer></footer>")
    monkeypatch.setattr(publish, "PROJECTS", tmp_path)
    post = publish.Post(slug="001-x", date="d", description="d", tags=(), explorer="toy")
    monkeypatch.setitem(publish.POSTS, "toy", post)

    site = tmp_path / "site"
    publish.publish("toy", site)

    bundle = site / "content" / "posts" / "001-x"
    assert (bundle / "figures" / "a.png").exists()
    assert not (bundle / "figures" / "unused.png").exists()
    assert (site / "static" / "explorers" / "toy" / "worker.js").exists()
    assert "Read the post" in (site / "static" / "explorers" / "toy" / "index.html").read_text()
