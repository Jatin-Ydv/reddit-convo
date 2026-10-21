from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
ASSETS_DIR = PACKAGE_ROOT / "assets"
SCENES_DIR = ASSETS_DIR / "scenes"
CHARACTERS_DIR = ASSETS_DIR / "characters"
DEFAULT_CACHE = Path("cache")
MODEL_CACHE = Path.home() / ".cache" / "reddit-convo"


def post_cache(cache_dir: Path, post_id: str) -> Path:
    path = cache_dir / post_id
    path.mkdir(parents=True, exist_ok=True)
    (path / "audio").mkdir(exist_ok=True)
    return path
