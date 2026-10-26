from toylib.data import download


def test_fetch_skips_existing_file(tmp_path):
    dest = tmp_path / "f"
    dest.write_text("cached")
    assert download.fetch("http://invalid.invalid/f", dest) == dest
    assert dest.read_text() == "cached"


def test_cache_dir_honours_env(tmp_path, monkeypatch):
    monkeypatch.setenv("TOYLIB_CACHE", str(tmp_path))
    assert download.cache_dir("x") == tmp_path / "x"
    assert (tmp_path / "x").is_dir()
