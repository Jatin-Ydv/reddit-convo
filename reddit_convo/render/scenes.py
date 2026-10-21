"""Painted 1920x1080 backdrops. Characters stand with feet at y~1000, heads near y~580."""

from __future__ import annotations

import math
import random

from PIL import Image, ImageDraw, ImageFilter

W, H = 1920, 1080
FLOOR_Y = 800
INK = (40, 34, 44)

# World-space points used by the animated ambient layer.
HOTSPOTS: dict[str, dict[str, object]] = {
    "coffee_shop": {"steam": [(1590, 548), (1730, 600)], "window": (120, 140, 760, 640)},
    "living_room": {"window": (300, 150, 760, 600)},
    "boutique": {"sparkles": []},
    "campfire": {"fire": (960, 975)},
    "park": {},
    "office": {"window": (100, 120, 900, 560)},
    "gym": {},
    "debate_table": {"spots": [(640, 0), (1280, 0)]},
}


def draw_scene(name: str) -> Image.Image:
    painter = {
        "park": _park,
        "coffee_shop": _coffee,
        "debate_table": _debate,
        "gym": _gym,
        "campfire": _campfire,
        "office": _office,
        "boutique": _boutique,
        "living_room": _living_room,
    }.get(name, _coffee)
    img = painter().convert("RGB")
    return img.filter(ImageFilter.GaussianBlur(1.2))


# ---------------------------------------------------------------- helpers


def _grad(top, bottom, w=W, h=H) -> Image.Image:
    img = Image.new("RGB", (w, h))
    d = ImageDraw.Draw(img)
    for y in range(h):
        t = y / max(h - 1, 1)
        d.line([(0, y), (w, y)], fill=tuple(int(a + (b - a) * t) for a, b in zip(top, bottom)))
    return img


def _paste_grad(img: Image.Image, box, top, bottom) -> None:
    x0, y0, x1, y1 = [int(v) for v in box]
    img.paste(_grad(top, bottom, x1 - x0, y1 - y0), (x0, y0))


def _glow(img: Image.Image, center, radius: float, color, strength: float = 0.6) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    md = ImageDraw.Draw(mask)
    x, y = center
    md.ellipse((x - radius, y - radius, x + radius, y + radius), fill=int(255 * strength))
    mask = mask.filter(ImageFilter.GaussianBlur(radius * 0.45))
    layer = Image.new("RGB", img.size, color)
    return Image.composite(layer, img, mask)


def _shade(img: Image.Image, box, color, alpha: int, blur: float = 0) -> Image.Image:
    mask = Image.new("L", img.size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=alpha)
    if blur:
        mask = mask.filter(ImageFilter.GaussianBlur(blur))
    return Image.composite(Image.new("RGB", img.size, color), img, mask)


def _wood_floor(img: Image.Image, base, line, y0=FLOOR_Y) -> None:
    d = ImageDraw.Draw(img)
    _paste_grad(img, (0, y0, W, H), tuple(int(c * 0.86) for c in base), base)
    vx, vy = W / 2, y0 - 900
    for i in range(-14, 15):
        xb = W / 2 + i * 150
        t = (y0 - vy) / (H - vy)
        xt = vx + (xb - vx) * t
        d.line([(xt, y0), (xb, H)], fill=line, width=3)
    y, gap = y0 + 18, 16
    while y < H:
        d.line([(0, y), (W, y)], fill=tuple(int(c * 0.95) for c in base), width=2)
        gap *= 1.3
        y += gap
    d.line([(0, y0), (W, y0)], fill=tuple(int(c * 0.6) for c in base), width=6)


def _plant(d: ImageDraw.ImageDraw, x: int, y: int, scale: float = 1.0, pot=(196, 110, 80)) -> None:
    s = scale
    for i in range(9):
        a = math.radians(-160 + i * 17)
        lx, ly = x + math.cos(a) * 120 * s, y - 70 * s + math.sin(a) * 140 * s
        d.ellipse((lx - 46 * s, ly - 22 * s, lx + 46 * s, ly + 22 * s), fill=(60 + i * 6, 140 + (i % 3) * 14, 80), outline=(36, 90, 52), width=3)
        d.line([(x, y - 30 * s), (lx, ly)], fill=(50, 110, 60), width=max(2, int(5 * s)))
    d.polygon([(x - 62 * s, y - 40 * s), (x + 62 * s, y - 40 * s), (x + 46 * s, y + 70 * s), (x - 46 * s, y + 70 * s)], fill=pot, outline=INK, width=3)
    d.rectangle((x - 68 * s, y - 52 * s, x + 68 * s, y - 34 * s), fill=tuple(int(c * 0.85) for c in pot), outline=INK, width=3)


def _font(size: int, kind: str = "bold"):
    from reddit_convo.render.art import load_font

    return load_font(size, kind)


# ---------------------------------------------------------------- coffee shop


def _coffee() -> Image.Image:
    img = _grad((238, 206, 170), (214, 172, 132))
    d = ImageDraw.Draw(img)
    rng = random.Random(4)
    for row in range(0, 40):
        y = row * 22
        if y > FLOOR_Y - 160:
            break
        off = 0 if row % 2 else 34
        for col in range(-1, 10):
            x = 1290 + off + col * 68
            tone = rng.randint(-14, 14)
            d.rectangle((x, y, x + 64, y + 18), fill=(176 + tone, 98 + tone // 2, 74 + tone // 2))
    d.line([(1290, 0), (1290, FLOOR_Y - 160)], fill=(150, 104, 70), width=6)

    x0, y0, x1, y1 = HOTSPOTS["coffee_shop"]["window"]  # type: ignore[misc]
    _paste_grad(img, (x0, y0, x1, y1), (150, 196, 228), (226, 238, 244))
    for i, (bx, bw, bh) in enumerate(((150, 120, 300), (290, 90, 220), (390, 140, 360), (550, 110, 260), (670, 90, 320))):
        col = (176 + i * 6, 192 + i * 4, 212)
        d.rectangle((bx, y1 - bh, bx + bw, y1), fill=col)
        for wy in range(y1 - bh + 20, y1 - 20, 36):
            for wx in range(bx + 14, bx + bw - 14, 28):
                d.rectangle((wx, wy, wx + 12, wy + 16), fill=(230, 238, 246))
    d.ellipse((560, 380, 800, 620), fill=(110, 160, 110))
    d.rectangle((x0 - 18, y0 - 18, x1 + 18, y0), fill=(86, 56, 38))
    d.rectangle((x0 - 18, y1, x1 + 18, y1 + 26), fill=(86, 56, 38))
    d.rectangle((x0 - 18, y0, x0, y1), fill=(86, 56, 38))
    d.rectangle((x1, y0, x1 + 18, y1), fill=(86, 56, 38))
    d.rectangle(((x0 + x1) // 2 - 7, y0, (x0 + x1) // 2 + 7, y1), fill=(86, 56, 38))
    d.rectangle((x0, (y0 + y1) // 2 - 6, x1, (y0 + y1) // 2 + 6), fill=(86, 56, 38))
    font = _font(46, "serif")
    d.text(((3 * x0 + x1) // 4, y0 + 70), "CAFÉ", font=font, fill=(255, 255, 255), anchor="mm")

    d.rectangle((860, 160, 1180, 440), fill=(92, 62, 40))
    d.rectangle((876, 176, 1164, 424), fill=(46, 54, 50))
    d.text((1020, 210), "MENU", font=_font(34, "round"), fill=(240, 240, 230), anchor="mm")
    for i, (item, price) in enumerate((("latte", "4.5"), ("mocha", "5"), ("cold brew", "4"), ("chai", "3.5"))):
        y = 260 + i * 40
        d.text((900, y), item, font=_font(24, "round"), fill=(236, 230, 210))
        d.text((1140, y), price, font=_font(24, "round"), fill=(250, 210, 120), anchor="ra")

    for lx in (560, 1250, 1640):
        d.line([(lx, 0), (lx, 150)], fill=INK, width=4)
        d.polygon([(lx - 60, 210), (lx + 60, 210), (lx + 26, 150), (lx - 26, 150)], fill=(48, 92, 74), outline=INK, width=3)
        img = _glow(img, (lx, 230), 170, (255, 226, 160), 0.55)
        d = ImageDraw.Draw(img)
        d.ellipse((lx - 18, 200, lx + 18, 236), fill=(255, 246, 210))

    for sy in (330, 480):
        d.rectangle((1360, sy, 1860, sy + 16), fill=(120, 80, 50), outline=INK, width=2)
        for i in range(7):
            mx = 1385 + i * 68
            col = [(230, 90, 80), (90, 150, 210), (240, 200, 90), (250, 250, 245), (120, 180, 120)][i % 5]
            if i % 3 == 2:
                d.rounded_rectangle((mx, sy - 82, mx + 44, sy), radius=8, fill=(200, 220, 230), outline=INK, width=2)
                d.rectangle((mx + 4, sy - 40, mx + 40, sy - 4), fill=(140, 92, 54))
            else:
                d.rectangle((mx, sy - 52, mx + 44, sy), fill=col, outline=INK, width=2)
                d.arc((mx + 34, sy - 44, mx + 58, sy - 16), 270, 90, fill=INK, width=4)

    d.rectangle((1240, 640, W, FLOOR_Y), fill=(120, 76, 48), outline=INK, width=3)
    d.rectangle((1240, 620, W, 646), fill=(70, 46, 34))
    for x in range(1270, W, 90):
        d.rectangle((x, 664, x + 64, 780), outline=(96, 60, 38), width=4)
    d.rounded_rectangle((1500, 520, 1690, 622), radius=12, fill=(190, 196, 204), outline=INK, width=3)
    d.rectangle((1530, 560, 1660, 600), fill=(60, 64, 72))
    d.rectangle((1560, 600, 1580, 622), fill=(40, 40, 46))
    d.rectangle((1610, 600, 1630, 622), fill=(40, 40, 46))
    d.rectangle((1700, 580, 1760, 622), fill=(250, 250, 245), outline=INK, width=2)

    d.rectangle((0, 640, 1240, FLOOR_Y), fill=(150, 98, 64))
    d.rectangle((0, 630, 1240, 648), fill=(110, 70, 46))
    for x in range(20, 1240, 110):
        d.rectangle((x, 668, x + 84, 780), outline=(124, 80, 52), width=4)

    _wood_floor(img, (168, 116, 78), (140, 94, 62))
    d = ImageDraw.Draw(img)
    _plant(d, 90, 740, 1.0)
    return img


# ---------------------------------------------------------------- living room


def _living_room() -> Image.Image:
    img = _grad((92, 92, 130), (74, 72, 104))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 64):
        d.rectangle((x, 0, x + 26, FLOOR_Y), fill=(98, 98, 138))

    x0, y0, x1, y1 = HOTSPOTS["living_room"]["window"]  # type: ignore[misc]
    _paste_grad(img, (x0, y0, x1, y1), (22, 30, 62), (52, 58, 96))
    rng = random.Random(9)
    for i in range(7):
        bx = x0 + i * 70
        bh = rng.randint(90, 230)
        d.rectangle((bx, y1 - bh, bx + 60, y1), fill=(30, 36, 66))
        for _ in range(6):
            wx, wy = rng.randint(bx + 6, bx + 50), rng.randint(y1 - bh + 8, y1 - 12)
            d.rectangle((wx, wy, wx + 7, wy + 9), fill=(250, 214, 120))
    img = _glow(img, (x1 - 110, y0 + 90), 60, (220, 226, 255), 0.7)
    d = ImageDraw.Draw(img)
    d.ellipse((x1 - 140, y0 + 60, x1 - 80, y0 + 120), fill=(236, 238, 250))
    for side, cx in ((-1, x0), (1, x1)):
        for i in range(5):
            fx = cx - side * 20 + side * i * 26 - (100 if side < 0 else 0)
            col = (126, 52, 70) if i % 2 else (146, 64, 84)
            d.rectangle((fx, y0 - 50, fx + 26, y1 + 90), fill=col)
    d.rectangle((x0 - 130, y0 - 64, x1 + 130, y0 - 46), fill=(60, 44, 40))
    d.rectangle((x0 - 10, y1, x1 + 10, y1 + 20), fill=(220, 214, 220))

    for i, (fx, fy, fw, fh) in enumerate(((1060, 190, 170, 130), (1260, 150, 120, 170), (1410, 220, 160, 120))):
        d.rectangle((fx, fy, fx + fw, fy + fh), fill=(200, 170, 120), outline=INK, width=3)
        d.rectangle((fx + 14, fy + 14, fx + fw - 14, fy + fh - 14), fill=(236, 226, 214))
        if i == 1:
            from reddit_convo.render.figure import heart

            heart(d, fx + fw / 2 - 10, fy + fh / 2 - 6, 22, (210, 80, 100))
            heart(d, fx + fw / 2 + 14, fy + fh / 2 + 4, 22, (210, 80, 100))
            d.line([(fx + fw / 2 + 2, fy + 40), (fx + fw / 2 - 8, fy + fh - 40)], fill=(236, 226, 214), width=6)
        else:
            d.polygon([(fx + 20, fy + fh - 20), (fx + fw / 2, fy + 40), (fx + fw - 20, fy + fh - 20)], fill=(140, 170, 200))

    img = _glow(img, (1720, 330), 300, (255, 196, 120), 0.55)
    d = ImageDraw.Draw(img)
    d.line([(1720, 340), (1720, 980)], fill=INK, width=8)
    d.ellipse((1660, 966, 1780, 996), fill=INK)
    d.polygon([(1650, 340), (1790, 340), (1756, 240), (1684, 240)], fill=(246, 218, 160), outline=INK, width=3)

    d.rounded_rectangle((420, 600, 1500, 760), radius=40, fill=(56, 104, 116), outline=INK, width=4)
    for i in range(3):
        cx = 520 + i * 330
        d.rounded_rectangle((cx, 630, cx + 300, 760), radius=28, fill=(66, 122, 134), outline=INK, width=3)
    d.rounded_rectangle((380, 660, 480, 860), radius=34, fill=(50, 94, 106), outline=INK, width=4)
    d.rounded_rectangle((1440, 660, 1540, 860), radius=34, fill=(50, 94, 106), outline=INK, width=4)
    d.rounded_rectangle((470, 740, 1450, 840), radius=20, fill=(60, 112, 124), outline=INK, width=4)
    d.rounded_rectangle((560, 650, 720, 760), radius=26, fill=(214, 120, 120), outline=INK, width=3)

    _wood_floor(img, (96, 70, 58), (82, 58, 48))
    d = ImageDraw.Draw(img)
    d.ellipse((300, 870, 1620, 1110), fill=(150, 84, 96), outline=(120, 64, 76), width=8)
    d.ellipse((440, 900, 1480, 1080), outline=(196, 130, 120), width=6)

    d.rectangle((150, 700, 300, 720), fill=(110, 76, 56), outline=INK, width=3)
    d.rectangle((166, 720, 182, 860), fill=(110, 76, 56))
    d.rectangle((268, 720, 284, 860), fill=(110, 76, 56))
    d.rectangle((170, 650, 240, 700), fill=(150, 196, 220), outline=INK, width=3)
    d.polygon([(190, 650), (205, 620), (222, 650)], fill=(250, 250, 250), outline=INK)
    d.rectangle((250, 662, 286, 700), fill=(236, 230, 214), outline=INK, width=2)

    return _shade(img, (0, 0, W, H), (30, 30, 70), 40)


# ---------------------------------------------------------------- boutique


def _bottle(d: ImageDraw.ImageDraw, x: float, base: float, kind: int, color) -> tuple[float, float]:
    lighter = tuple(min(255, int(c * 1.35)) for c in color)
    if kind == 0:
        w, h = 46, 70
        d.rounded_rectangle((x - w / 2, base - h, x + w / 2, base), radius=10, fill=color, outline=INK, width=3)
    elif kind == 1:
        w, h = 58, 58
        d.ellipse((x - w / 2, base - h, x + w / 2, base), fill=color, outline=INK, width=3)
    elif kind == 2:
        w, h = 34, 96
        d.rounded_rectangle((x - w / 2, base - h, x + w / 2, base), radius=6, fill=color, outline=INK, width=3)
    else:
        w, h = 60, 60
        d.polygon([(x, base - h), (x + w / 2, base - h / 2), (x, base), (x - w / 2, base - h / 2)], fill=color, outline=INK, width=3)
    d.rectangle((x - w / 2 + 8, base - h + 10, x - w / 2 + 14, base - 12), fill=lighter)
    cap = (212, 176, 90) if kind % 2 == 0 else (36, 32, 40)
    d.rectangle((x - 9, base - h - 10, x + 9, base - h), fill=(220, 220, 228), outline=INK, width=2)
    d.rounded_rectangle((x - 14, base - h - 34, x + 14, base - h - 10), radius=5, fill=cap, outline=INK, width=2)
    return x, base - h / 2


def _boutique() -> Image.Image:
    img = _grad((252, 238, 232), (236, 214, 208))
    d = ImageDraw.Draw(img)
    rng = random.Random(21)
    sparkles: list[tuple[float, float]] = []
    colors = [(236, 170, 70), (230, 120, 150), (120, 170, 230), (140, 200, 160), (190, 140, 220), (240, 210, 120), (200, 80, 90)]
    for nx in (140, 760, 1380):
        nw = 400
        top = 120
        d.rounded_rectangle((nx - 12, top - 12, nx + nw + 12, FLOOR_Y - 120), radius=200, fill=(214, 186, 140))
        d.rounded_rectangle((nx, top, nx + nw, FLOOR_Y - 130), radius=192, fill=(246, 226, 220))
        img = _glow(img, (nx + nw / 2, top + 90), 160, (255, 246, 220), 0.5)
        d = ImageDraw.Draw(img)
        for shelf_y in (330, 470, 610):
            d.rectangle((nx + 20, shelf_y, nx + nw - 20, shelf_y + 10), fill=(200, 226, 236), outline=(150, 170, 180), width=2)
            x = nx + 60
            while x < nx + nw - 50:
                kind = rng.randint(0, 3)
                sparkles.append(_bottle(d, x, shelf_y, kind, rng.choice(colors)))
                x += rng.randint(66, 84)
    HOTSPOTS["boutique"]["sparkles"] = sparkles
    d.rectangle((0, 40, W, 52), fill=(212, 176, 90))
    d.text((W / 2, 86), "PARFUMERIE", font=_font(44, "serif"), fill=(170, 130, 60), anchor="mm")
    d.rectangle((0, FLOOR_Y - 120, W, FLOOR_Y), fill=(236, 220, 214))
    d.rectangle((0, FLOOR_Y - 124, W, FLOOR_Y - 112), fill=(212, 176, 90))
    _paste_grad(img, (0, FLOOR_Y, W, H), (226, 220, 222), (246, 242, 240))
    d = ImageDraw.Draw(img)
    for i in range(-10, 11):
        xb = W / 2 + i * 220
        d.line([(W / 2 + i * 60, FLOOR_Y), (xb, H)], fill=(214, 206, 210), width=2)
    for y in (850, 920, 1010):
        d.line([(0, y), (W, y)], fill=(214, 206, 210), width=2)
    img = _shade(img, (0, FLOOR_Y, W, FLOOR_Y + 60), (180, 150, 140), 40, blur=20)
    d = ImageDraw.Draw(img)
    _plant(d, 70, 760, 1.1, pot=(236, 236, 230))
    _plant(d, 1850, 760, 1.1, pot=(236, 236, 230))
    return img


# ---------------------------------------------------------------- park


def _park() -> Image.Image:
    img = _grad((120, 186, 238), (210, 234, 246))
    img = _glow(img, (1640, 150), 200, (255, 246, 200), 0.8)
    d = ImageDraw.Draw(img)
    d.ellipse((1590, 100, 1690, 200), fill=(255, 244, 190))
    for cx, cy, s in ((300, 160, 1.0), (900, 110, 0.8), (1300, 220, 0.7)):
        for dx, dy, rr in ((-70, 10, 50), (-20, -20, 64), (40, 0, 56), (90, 16, 42)):
            d.ellipse((cx + (dx - rr) * s, cy + (dy - rr) * s, cx + (dx + rr) * s, cy + (dy + rr) * s), fill=(255, 255, 255))
    d.polygon([(0, 560), (300, 430), (640, 520), (980, 410), (1340, 500), (1700, 420), (W, 480), (W, FLOOR_Y), (0, FLOOR_Y)], fill=(150, 196, 150))
    d.polygon([(0, 640), (420, 560), (860, 620), (1300, 560), (W, 610), (W, FLOOR_Y), (0, FLOOR_Y)], fill=(112, 176, 108))
    for x in range(-40, W + 80, 120):
        r = 70 + (x * 7) % 40
        d.ellipse((x - r, 560 - r, x + r, 560 + r * 0.8), fill=(84, 146, 90))
    _paste_grad(img, (0, 680, W, H), (110, 180, 96), (86, 156, 74))
    d = ImageDraw.Draw(img)
    d.polygon([(860, 680), (1060, 680), (1500, H), (420, H)], fill=(222, 200, 160))
    d.line([(860, 680), (420, H)], fill=(190, 166, 126), width=4)
    d.line([(1060, 680), (1500, H)], fill=(190, 166, 126), width=4)
    for tx, r in ((140, 170), (1780, 190)):
        d.rectangle((tx - 22, 420, tx + 22, 860), fill=(110, 76, 52), outline=INK, width=3)
        for dx, dy, k in ((-80, -40, 1.0), (60, -70, 1.1), (0, -150, 1.0), (-40, 40, 0.8), (70, 30, 0.9)):
            rr = r * 0.6 * k
            d.ellipse((tx + dx - rr, 380 + dy - rr, tx + dx + rr, 380 + dy + rr), fill=(70, 140, 76), outline=(48, 104, 56), width=3)
    d.rectangle((1180, 700, 1520, 730), fill=(150, 98, 60), outline=INK, width=3)
    d.rectangle((1180, 650, 1520, 676), fill=(150, 98, 60), outline=INK, width=3)
    for x in (1200, 1500):
        d.rectangle((x - 6, 730, x + 6, 790), fill=INK)
    d.line([(1620, 470), (1620, 790)], fill=INK, width=10)
    d.rectangle((1596, 440, 1644, 480), fill=(250, 240, 200), outline=INK, width=4)
    rng = random.Random(3)
    for _ in range(140):
        x, y = rng.randint(0, W), rng.randint(720, H)
        if 420 < x < 1500 and y > 680 + (x - 860) * 0:
            continue
        col = rng.choice([(250, 220, 80), (250, 140, 160), (255, 255, 255), (190, 150, 240)])
        d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=col)
    return img


# ---------------------------------------------------------------- campfire


def _campfire() -> Image.Image:
    img = _grad((14, 20, 46), (58, 44, 86))
    d = ImageDraw.Draw(img)
    rng = random.Random(7)
    for _ in range(220):
        x, y = rng.randint(0, W), rng.randint(0, 560)
        s = rng.choice((1, 1, 2, 3))
        d.ellipse((x - s, y - s, x + s, y + s), fill=(255, 255, 230))
    img = _glow(img, (1500, 170), 120, (200, 210, 255), 0.6)
    d = ImageDraw.Draw(img)
    d.ellipse((1460, 130, 1540, 210), fill=(240, 240, 255))
    d.ellipse((1480, 120, 1560, 200), fill=(30, 34, 70))
    for layer, (col, base, hgt) in enumerate((((34, 46, 66), 640, 300), ((22, 32, 46), 720, 360))):
        for x in range(-60, W + 120, 90 + layer * 30):
            hh = hgt + rng.randint(-60, 60)
            d.polygon([(x - 70, base + 60), (x, base - hh), (x + 70, base + 60)], fill=col)
    _paste_grad(img, (0, 760, W, H), (46, 38, 44), (30, 26, 32))
    fx, fy = HOTSPOTS["campfire"]["fire"]  # type: ignore[misc]
    img = _glow(img, (fx, fy - 120), 520, (255, 140, 50), 0.5)
    d = ImageDraw.Draw(img)
    for i in range(9):
        a = math.radians(i * 40)
        d.ellipse((fx + math.cos(a) * 110 - 26, fy + math.sin(a) * 30 - 18, fx + math.cos(a) * 110 + 26, fy + math.sin(a) * 30 + 18), fill=(110, 104, 110), outline=INK, width=2)
    d.polygon([(fx - 90, fy - 10), (fx + 70, fy - 40), (fx + 80, fy - 16), (fx - 80, fy + 14)], fill=(110, 70, 40), outline=INK)
    d.polygon([(fx + 90, fy - 10), (fx - 70, fy - 40), (fx - 80, fy - 16), (fx + 80, fy + 14)], fill=(126, 80, 46), outline=INK)
    for lx in (240, 1680):
        d.rounded_rectangle((lx - 140, 860, lx + 140, 920), radius=28, fill=(120, 80, 50), outline=INK, width=4)
        d.ellipse((lx + 110, 860, lx + 160, 920), fill=(180, 140, 100), outline=INK, width=3)
    return img


# ---------------------------------------------------------------- office


def _office() -> Image.Image:
    img = _grad((222, 228, 236), (204, 212, 222))
    d = ImageDraw.Draw(img)
    x0, y0, x1, y1 = HOTSPOTS["office"]["window"]  # type: ignore[misc]
    _paste_grad(img, (x0, y0, x1, y1), (150, 200, 240), (220, 236, 248))
    rng = random.Random(5)
    x = x0
    while x < x1:
        bw, bh = rng.randint(60, 120), rng.randint(140, 380)
        d.rectangle((x, y1 - bh, x + bw, y1), fill=(150 + rng.randint(0, 30), 170, 196))
        x += bw + 6
    for mx in range(x0, x1 + 1, 200):
        d.rectangle((mx - 6, y0, mx + 6, y1), fill=(90, 96, 110))
    d.rectangle((x0, y0 - 8, x1, y0 + 6), fill=(90, 96, 110))
    d.rectangle((x0, y1 - 6, x1, y1 + 10), fill=(90, 96, 110))
    for lx in (480, 960, 1440):
        d.rectangle((lx - 140, 0, lx + 140, 22), fill=(250, 252, 255))
    d.rectangle((1080, 150, 1640, 480), fill=(250, 250, 252), outline=(150, 156, 170), width=8)
    for i, (h, c) in enumerate(((90, (90, 150, 220)), (150, (240, 150, 70)), (60, (220, 80, 90)), (200, (100, 190, 120)))):
        bx = 1130 + i * 70
        d.rectangle((bx, 440 - h, bx + 46, 440), fill=c)
    d.line([(1430, 400), (1490, 300), (1540, 340), (1600, 220)], fill=(220, 60, 70), width=6)
    for i, c in enumerate(((255, 230, 120), (255, 170, 200), (160, 220, 255))):
        d.rectangle((1440 + i * 60, 170, 1490 + i * 60, 220), fill=c)
    d.rectangle((0, 620, W, FLOOR_Y), fill=(184, 190, 202))
    for dx in (140, 1300):
        d.rectangle((dx, 640, dx + 480, 664), fill=(160, 120, 90), outline=INK, width=3)
        d.rectangle((dx + 140, 520, dx + 340, 640), fill=(40, 44, 54), outline=INK, width=4)
        d.rectangle((dx + 150, 530, dx + 330, 626), fill=(80, 150, 210))
        d.rectangle((dx + 230, 640, dx + 250, 664), fill=INK)
    _paste_grad(img, (0, FLOOR_Y, W, H), (120, 130, 150), (100, 108, 128))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 40):
        d.line([(x, FLOOR_Y), (x - 60, H)], fill=(112, 120, 140), width=1)
    _plant(d, 1840, 760, 1.0, pot=(240, 240, 240))
    d.rounded_rectangle((40, 560, 120, 790), radius=12, fill=(230, 236, 244), outline=INK, width=3)
    d.rounded_rectangle((52, 470, 108, 570), radius=20, fill=(160, 210, 240), outline=INK, width=3)
    return img


# ---------------------------------------------------------------- gym


def _gym() -> Image.Image:
    img = _grad((62, 64, 74), (44, 46, 54))
    d = ImageDraw.Draw(img)
    d.rectangle((0, 520, W, 560), fill=(210, 60, 60))
    d.rectangle((260, 140, 1660, 470), fill=(130, 150, 170))
    _paste_grad(img, (272, 152, 1648, 458), (156, 176, 196), (110, 128, 148))
    d = ImageDraw.Draw(img)
    for x in range(400, 1600, 260):
        d.line([(x, 152), (x + 120, 458)], fill=(186, 202, 220), width=8)
    d.text((W / 2, 92), "NO PAIN · NO GAIN", font=_font(54, "bold"), fill=(250, 210, 70), anchor="mm")
    for lx in (480, 960, 1440):
        img = _glow(img, (lx, 0), 160, (255, 255, 240), 0.5)
    d = ImageDraw.Draw(img)
    d.rectangle((60, 640, 520, 660), fill=(90, 90, 100), outline=INK, width=3)
    d.rectangle((60, 720, 520, 740), fill=(90, 90, 100), outline=INK, width=3)
    for i in range(6):
        x = 90 + i * 74
        for y in (620, 700):
            d.rectangle((x, y - 10, x + 50, y + 10), fill=(140, 140, 150))
            d.ellipse((x - 14, y - 22, x + 10, y + 22), fill=(40, 40, 46))
            d.ellipse((x + 40, y - 22, x + 64, y + 22), fill=(40, 40, 46))
    d.rectangle((1400, 690, 1800, 720), fill=(40, 40, 46), outline=INK, width=3)
    d.line([(1440, 500), (1440, 800)], fill=(120, 120, 130), width=12)
    d.line([(1760, 500), (1760, 800)], fill=(120, 120, 130), width=12)
    d.line([(1380, 580), (1820, 580)], fill=(170, 170, 180), width=10)
    for x in (1390, 1810):
        d.ellipse((x - 26, 520, x + 26, 640), fill=(200, 60, 60), outline=INK, width=3)
    _paste_grad(img, (0, FLOOR_Y, W, H), (52, 52, 58), (36, 36, 42))
    d = ImageDraw.Draw(img)
    for x in range(0, W, 160):
        d.line([(x, FLOOR_Y), (x - 120, H)], fill=(62, 62, 70), width=3)
    d.line([(0, 900), (W, 900)], fill=(240, 200, 60), width=6)
    return img


# ---------------------------------------------------------------- debate stage


def _debate() -> Image.Image:
    img = _grad((30, 34, 70), (60, 40, 90))
    d = ImageDraw.Draw(img)
    for i in range(12):
        x = 180 + i * 140
        d.ellipse((x - 50, 300 - 50, x + 50, 300 + 50), outline=(80, 90, 150), width=4)
    d.text((W / 2, 180), "VS", font=_font(160, "bold"), fill=(250, 210, 80), anchor="mm", stroke_width=6, stroke_fill=(120, 60, 30))
    for x in (640, 1280):
        mask = Image.new("L", img.size, 0)
        ImageDraw.Draw(mask).polygon([(x - 60, 0), (x + 60, 0), (x + 320, H), (x - 320, H)], fill=70)
        mask = mask.filter(ImageFilter.GaussianBlur(30))
        img = Image.composite(Image.new("RGB", img.size, (255, 250, 220)), img, mask)
    d = ImageDraw.Draw(img)
    for side in (0, 1):
        x0 = 0 if side == 0 else W - 200
        for i in range(5):
            d.rectangle((x0 + i * 40, 0, x0 + i * 40 + 40, FLOOR_Y + 40), fill=(170, 30, 50) if i % 2 else (140, 20, 40))
    d.rectangle((0, 0, W, 40), fill=(140, 20, 40))
    _paste_grad(img, (0, FLOOR_Y, W, H), (96, 70, 60), (60, 40, 40))
    d = ImageDraw.Draw(img)
    d.rectangle((0, FLOOR_Y, W, FLOOR_Y + 14), fill=(220, 180, 80))
    return img
