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


def test_views_become_shortcodes():
    post = publish.Post(slug="x", date="d", description="d", tags=(), explorer="toy")
    page = publish.embed_views(post, "Text.\n\n<!-- explorer: stages -->\n\nMore.\n")
    assert '{{< explorer "toy" "stages" >}}' in page
    assert "<!--" not in page


def test_comments_inside_a_line_stay():
    post = publish.Post(slug="x", date="d", description="d", tags=(), explorer="toy")
    text = "See `<!-- explorer: stages -->` in the source.\n"
    assert publish.embed_views(post, text) == text


def test_redirect_page_points_at_the_post():
    assert 'url=/posts/x/"' in publish.redirect_page("/posts/x/")


def test_publish_writes_the_bundle_and_explorer(tmp_path, monkeypatch):
    source = tmp_path / "toy"
    (source / "figures").mkdir(parents=True)
    (source / "explorer").mkdir()
    (source / "POST.md").write_text("# T\n\n![a](figures/a.png)\n\n<!-- explorer: stages -->\n")
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
    assert '{{< explorer "toy" "stages" >}}' in (bundle / "index.md").read_text()
    assert "Read the post" in (site / "static" / "explorers" / "toy" / "index.html").read_text()
