from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from reddit_convo.models import SceneName
from reddit_convo.paths import CHARACTERS_DIR, SCENES_DIR
from reddit_convo.render.figure import POSES, Act, render_figure
from reddit_convo.render.personality import CATALOG
from reddit_convo.render.scenes import draw_scene

WIDTH, HEIGHT = 1280, 720
WORLD_W, WORLD_H = 1920, 1080
SCENE_VERSION = 2
SCENES = ("park", "coffee_shop", "debate_table", "gym", "campfire", "office", "boutique", "living_room")

FONT_FILES = {
    "bold": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "regular": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "medium": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-M.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "italic": ["/usr/share/fonts/truetype/ubuntu/Ubuntu-RI.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "round": ["/usr/share/fonts/truetype/quicksand/Quicksand-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
    "serif": ["/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf", "/usr/share/fonts/truetype/freefont/FreeSerifBold.ttf"],
}


def ensure_assets() -> None:
    SCENES_DIR.mkdir(parents=True, exist_ok=True)
    CHARACTERS_DIR.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        path = SCENES_DIR / f"{name}_v{SCENE_VERSION}.png"
        if not path.exists():
            draw_scene(name).save(path, "PNG")
    sheet = CHARACTERS_DIR / "expressions.png"
    if not sheet.exists():
        character_sheet().save(sheet, "PNG")


def scene_path(name: SceneName) -> Path:
    ensure_assets()
    return SCENES_DIR / f"{name}_v{SCENE_VERSION}.png"


def character_sheet() -> Image.Image:
    exprs = ["neutral", "happy", "laugh", "sad", "angry", "smirk", "eyeroll", "surprised", "scared", "love", "thinking"]
    poses = ["relaxed", "talk", "point", "crossed", "hips", "shrug", "fist_pump", "chin", "heart", "wring", "pockets"]
    cell, height = 170, 230
    ids = list(CATALOG)
    sheet = Image.new("RGBA", (cell * len(exprs), height * len(ids) + 40), (238, 236, 242, 255))
    d = ImageDraw.Draw(sheet)
    font = load_font(18, "bold")
    for col, name in enumerate(exprs):
        d.text((col * cell + cell // 2, 20), name, font=font, fill=(30, 30, 36), anchor="mm")
    for row, pid in enumerate(ids):
        spec = CATALOG[pid]
        for col, expr in enumerate(exprs):
            act = Act(pose=POSES[poses[col % len(poses)]], expr=expr, mouth=0.5 if col % 2 else 0.0)
            sprite = render_figure(spec, (90, 140, 200), 200, 1, act)
            x = int(col * cell + cell / 2 - sprite.anchor[0])
            y = int(40 + row * height + 215 - sprite.anchor[1])
            sheet.alpha_composite(sprite.image, (max(0, x), max(0, y)))
    return sheet


@lru_cache(maxsize=64)
def load_font(size: int, kind: str | bool = "regular"):
    if isinstance(kind, bool):
        kind = "bold" if kind else "regular"
    for path in FONT_FILES.get(kind, FONT_FILES["regular"]):
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


SCENE_TINT = {
    "campfire": (255, 120, 40, 26),
    "living_room": (60, 60, 140, 22),
    "coffee_shop": (255, 190, 120, 18),
    "boutique": (255, 210, 220, 14),
    "park": (255, 240, 200, 10),
}


def make_overlay(scene: str, size: tuple[int, int] = (WIDTH, HEIGHT)) -> Image.Image:
    """Vignette plus a per-scene color grade, composited once per frame."""
    w, h = size
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).ellipse((-int(w * 0.1), -int(h * 0.16), int(w * 1.1), int(h * 1.16)), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(60))
    overlay = Image.new("RGBA", (w, h), (10, 8, 20, 0))
    overlay.putalpha(Image.eval(mask, lambda p: int((255 - p) * 0.38)))
    tint = SCENE_TINT.get(scene)
    if tint:
        overlay = Image.alpha_composite(Image.new("RGBA", (w, h), tint), overlay)
    return overlay


if __name__ == "__main__":
    ensure_assets()
    character_sheet().save(CHARACTERS_DIR / "expressions.png", "PNG")
    print(f"Wrote scenes to {SCENES_DIR}")
    print(f"Wrote characters to {CHARACTERS_DIR}")
