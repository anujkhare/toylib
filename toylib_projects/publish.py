"""Copy a post and its explorer into the website repo, which is a Hugo site.

    uv run python -m toylib_projects.publish p01_cnn_mnist --site ../anujkhare.github.io

The post becomes a page bundle, so ``figures/`` links keep working. Only figures the post
embeds are copied. The explorer goes under ``static/``, with the document head that the
artifact host would otherwise add. Rebuilding and pushing the site is a separate step.
"""

import argparse
import dataclasses
import pathlib
import re
import shutil

PROJECTS = pathlib.Path(__file__).parent
EXPLORER_FILES = ("index.html", "data.js", "weights.js", "cnn.js", "worker.js")
EXPLORER_HEAD = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
"""


@dataclasses.dataclass
class Post:
    slug: str
    date: str
    description: str
    tags: tuple[str, ...]
    explorer: str | None = None  # Its folder under /explorers/ on the site.
    folder: str = "."  # Where the post's files sit inside the project.


POSTS = {
    "p01_cnn_mnist": Post(
        slug="004-why-does-a-cnn-work",
        date="2026-11-01T12:00:00-08:00",
        description="From a linear model to a ResNet on Imagenette, looking inside at each step.",
        tags=("CNN", "visualization", "from scratch"),
        explorer="why-cnn",
        folder="post",
    ),
}


def front_matter(post: Post, title: str) -> str:
    tags = ", ".join(f'"{tag}"' for tag in post.tags)
    return f"""---
title: {title!r}
date: '{post.date}'
tags: [{tags}]
author: "Anuj Khare"
showToc: true
TocOpen: false
draft: false
description: "{post.description}"
ShowReadingTime: true
ShowBreadCrumbs: true
ShowPostNavLinks: true
ShowWordCount: true
UseHugoToc: true
---
"""


def to_bundle(post: Post, text: str) -> str:
    """``POST.md`` -> a Hugo page; its first line is the ``# title``."""
    heading, _, body = text.partition("\n")
    if not heading.startswith("# "):
        raise ValueError(f"POST.md must start with a '# title' line, not {heading!r}")
    return front_matter(post, heading[2:]) + body


def embedded_figures(text: str) -> list[str]:
    return re.findall(r"!\[[^\]]*\]\(figures/([^)]+)\)", text)


def to_standalone_page(html: str, post_url: str) -> str:
    """Add the document head, and a link back to the post."""
    if "<footer>" not in html:
        raise ValueError("the explorer page has no <footer>")
    link = f'<footer>\n    <p><a href="{post_url}">Read the post</a></p>'
    return EXPLORER_HEAD + html.replace("<footer>", link, 1)


def publish(name: str, site: pathlib.Path) -> None:
    post = POSTS[name]
    source = PROJECTS / name / post.folder
    text = (source / "POST.md").read_text()
    bundle = site / "content" / "posts" / post.slug
    (bundle / "figures").mkdir(parents=True, exist_ok=True)
    (bundle / "index.md").write_text(to_bundle(post, text))
    for figure in embedded_figures(text):
        shutil.copy(source / "figures" / figure, bundle / "figures" / figure)

    if post.explorer is None:
        return
    target = site / "static" / "explorers" / post.explorer
    target.mkdir(parents=True, exist_ok=True)
    for file in EXPLORER_FILES:
        shutil.copy(source / "explorer" / file, target / file)
    page = (source / "explorer" / "index.html").read_text()
    (target / "index.html").write_text(to_standalone_page(page, f"/posts/{post.slug}/"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name", choices=sorted(POSTS))
    parser.add_argument("--site", type=pathlib.Path, required=True)
    args = parser.parse_args()
    publish(args.name, args.site)
