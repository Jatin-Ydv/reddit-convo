"""Stick-figure rig: big expressive head, capsule torso, posable stick limbs, emotes.

All drawing happens in a local frame where +x is the direction the character faces;
the head layer is mirrored for characters facing left.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from PIL import Image, ImageDraw

from reddit_convo.render.personality import Personality

INK = (32, 28, 38)
MOUTH = (110, 30, 44)
TONGUE = (236, 112, 122)
WHITE = (255, 255, 255)

# (upper-arm angle, forearm angle, hand). Angles are degrees from straight down;
# positive rotates outward (away from the body), 90 = horizontal, 180 = straight up.
Arm = tuple[float, float, str]


@dataclass(frozen=True)
class Pose:
    front: Arm
    back: Arm
    lean: float = 0.0
    tilt: float = 0.0
    shrug: float = 0.0
    slump: float = 0.0
    stance: float = 0.0
    crouch: float = 0.0


POSES: dict[str, Pose] = {
    "relaxed": Pose((10, 4, "open"), (10, 4, "open")),
    "clasp": Pose((22, -62, "open"), (22, -62, "open")),
    "talk": Pose((34, 100, "open"), (12, 6, "open")),
    "explain": Pose((40, 112, "open"), (40, 112, "open")),
    "explain_big": Pose((72, 140, "open"), (72, 140, "open"), stance=10),
    "point": Pose((84, 90, "finger"), (12, 6, "open"), lean=6),
    "point_up": Pose((28, 172, "finger"), (14, 6, "open")),
    "crossed": Pose((24, -96, "open"), (24, -96, "open"), tilt=4),
    "crossed_talk": Pose((40, 112, "open"), (24, -96, "open")),
    "hips": Pose((46, -40, "fist"), (46, -40, "fist"), stance=22),
    "shrug": Pose((44, 136, "open"), (44, 136, "open"), shrug=1.0, tilt=10),
    "fist_pump": Pose((150, 176, "fist"), (34, -40, "fist"), stance=22),
    "cheer": Pose((152, 168, "open"), (152, 168, "open"), stance=18, crouch=0.2),
    "chin": Pose((14, -168, "fist"), (26, -96, "open"), tilt=8),
    "heart": Pose((26, -128, "open"), (26, -128, "open")),
    "wring": Pose((16, -56, "fist"), (16, -56, "fist"), slump=0.35),
    "wring_talk": Pose((36, 96, "open"), (16, -56, "fist"), slump=0.2),
    "defend": Pose((46, -168, "open"), (46, -168, "open"), lean=-8, crouch=0.4),
    "cheeks": Pose((22, -176, "open"), (22, -176, "open"), shrug=0.6),
    "fists": Pose((62, 162, "fist"), (62, 162, "fist"), lean=6, stance=26, crouch=0.3),
    "slump": Pose((4, 0, "open"), (4, 0, "open"), slump=1.0, lean=4),
    "pockets": Pose((20, -22, "pocket"), (20, -22, "pocket"), stance=14),
    "pocket_talk": Pose((32, 96, "open"), (20, -22, "pocket"), stance=14),
    "wave": Pose((122, 174, "wave"), (10, 4, "open")),
    "facepalm": Pose((18, -172, "open"), (24, -96, "open"), slump=0.4, tilt=-6),
}


def blend_pose(a: Pose, b: Pose, t: float) -> Pose:
    if t >= 1.0:
        return b
    if t <= 0.0:
        return a

    def arm(x: Arm, y: Arm) -> Arm:
        return (x[0] + (y[0] - x[0]) * t, x[1] + (y[1] - x[1]) * t, y[2] if t > 0.5 else x[2])

    def f(x: float, y: float) -> float:
        return x + (y - x) * t

    return Pose(
        front=arm(a.front, b.front),
        back=arm(a.back, b.back),
        lean=f(a.lean, b.lean),
        tilt=f(a.tilt, b.tilt),
        shrug=f(a.shrug, b.shrug),
        slump=f(a.slump, b.slump),
        stance=f(a.stance, b.stance),
        crouch=f(a.crouch, b.crouch),
    )


@dataclass
class Act:
    pose: Pose
    expr: str = "neutral"
    mouth: float = 0.0
    blink: float = 0.0
    look: float = 1.0
    look_up: float = 0.0
    tilt: float = 0.0
    nod: float = 0.0
    bob: float = 0.0
    beat: float = 0.0
    wave: float = 0.0
    shake: float = 0.0


@dataclass
class Sprite:
    image: Image.Image
    anchor: tuple[float, float]
    head: tuple[float, float]
    head_r: float


BODY = {"round": (0.95, 0.17), "tall": (1.07, 0.11), "short": (0.86, 0.13), "athletic": (1.0, 0.145)}


def render_figure(
    spec: Personality,
    shirt: tuple[int, int, int],
    height: float,
    facing: int,
    act: Act,
    ss: int = 2,
) -> Sprite:
    H = height * ss
    mul, tw_f = BODY.get(spec.body, (1.0, 0.13))
    p = act.pose
    head_r = 0.19 * H
    leg = 0.29 * H * mul
    torso = 0.25 * H * mul
    W, Hh = int(H * 1.15), int(H * 1.5)
    ax, ay = W / 2, Hh - 0.05 * H
    img = Image.new("RGBA", (W, Hh), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    lw = max(2, int(0.03 * H))
    ol = max(2, int(0.012 * H))

    d.ellipse((ax - 0.2 * H, ay - 0.028 * H, ax + 0.2 * H, ay + 0.028 * H), fill=(0, 0, 0, 60))

    hip_x = ax + act.shake * ss
    hip_y = ay - leg + p.crouch * 0.06 * H - act.bob * H
    spread = (0.045 + p.stance * 0.0022) * H
    for side in (-1, 1):
        hx = hip_x + side * 0.03 * H
        fx = ax + side * spread
        knee = _knee((hx, hip_y), (fx, ay), leg / 2, facing)
        d.line([(hx, hip_y), knee, (fx, ay - 0.01 * H)], fill=INK, width=lw, joint="curve")
        _dot(d, knee, lw / 2, INK)
        foot_x = fx + facing * 0.022 * H
        d.ellipse((foot_x - 0.042 * H, ay - 0.032 * H, foot_x + 0.042 * H, ay + 0.004 * H), fill=INK)

    lean = math.radians(p.lean)
    sx = hip_x + facing * math.sin(lean) * torso
    sy = hip_y - math.cos(lean) * torso + p.slump * 0.03 * H
    tw = tw_f * H
    d.line([(hip_x, hip_y), (sx, sy)], fill=INK, width=int(tw + 2 * ol))
    _dot(d, (hip_x, hip_y), tw / 2 + ol, INK)
    _dot(d, (sx, sy), tw / 2 + ol, INK)
    d.line([(hip_x, hip_y), (sx, sy)], fill=shirt, width=int(tw))
    _dot(d, (hip_x, hip_y), tw / 2, shirt)
    _dot(d, (sx, sy), tw / 2, shirt)
    dark = tuple(int(c * 0.78) for c in shirt)
    d.line([(hip_x - tw * 0.5, hip_y), (hip_x + tw * 0.5, hip_y)], fill=dark, width=max(2, int(ol * 1.5)))

    neck_top = sy - 0.035 * H
    hc_x = sx + facing * p.slump * 0.035 * H
    hc_y = neck_top - head_r * 0.86 + act.nod * head_r * 0.22 + p.slump * 0.05 * H
    d.line([(sx, sy), (hc_x, hc_y)], fill=INK, width=lw)

    shoulder_y = sy + tw * 0.2 - p.shrug * 0.04 * H
    ua, la = 0.15 * H * mul**0.5, 0.14 * H * mul**0.5
    for which, arm, side in (("back", p.back, -facing), ("front", p.front, facing)):
        u, l, hand = arm
        if which == "front" and hand not in {"pocket", "wave"}:
            u += act.beat * 10
            l += act.beat * 38
        if hand == "wave":
            l += 24 * math.sin(act.wave)
        shx = sx + side * (tw / 2 - lw * 0.3)
        ur, lr = math.radians(u), math.radians(l)
        ex, ey = shx + side * math.sin(ur) * ua, shoulder_y + math.cos(ur) * ua
        hx2, hy2 = ex + side * math.sin(lr) * la, ey + math.cos(lr) * la
        d.line([(shx, shoulder_y), (ex, ey), (hx2, hy2)], fill=INK, width=lw, joint="curve")
        _dot(d, (ex, ey), lw / 2, INK)
        _hand(d, (hx2, hy2), hand, side, lr, H, spec.skin, ol)

    S = int(head_r * 3.5)
    head = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    hd = ImageDraw.Draw(head, "RGBA")
    c = S / 2
    draw_head(hd, spec, c, c, head_r, act, ol)
    if facing < 0:
        head = head.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    angle = -(p.tilt + act.tilt) * facing
    if abs(angle) > 0.4:
        head = head.rotate(angle, resample=Image.Resampling.BILINEAR, center=(c, c))
    img.alpha_composite(head, (max(0, int(hc_x - c)), max(0, int(hc_y - c))))

    out = img.reduce(ss) if ss > 1 else img
    return Sprite(out, (ax / ss, ay / ss), (hc_x / ss, hc_y / ss), head_r / ss)


def _knee(hip: tuple[float, float], foot: tuple[float, float], seg: float, facing: int) -> tuple[float, float]:
    dx, dy = foot[0] - hip[0], foot[1] - hip[1]
    dist = math.hypot(dx, dy) or 1.0
    mx, my = (hip[0] + foot[0]) / 2, (hip[1] + foot[1]) / 2
    if dist >= 2 * seg:
        return mx, my
    off = math.sqrt(seg * seg - (dist / 2) ** 2)
    px, py = -dy / dist, dx / dist
    if px * facing < 0:
        px, py = -px, -py
    return mx + px * off, my + py * off


def _dot(d: ImageDraw.ImageDraw, c: tuple[float, float], r: float, fill) -> None:
    d.ellipse((c[0] - r, c[1] - r, c[0] + r, c[1] + r), fill=fill)


def _hand(d, pos, hand: str, side: int, fore: float, H: float, skin, ol: int) -> None:
    if hand == "pocket":
        return
    x, y = pos
    if hand == "finger":
        dx, dy = side * math.sin(fore), math.cos(fore)
        tip = (x + dx * 0.075 * H, y + dy * 0.075 * H)
        w = int(0.022 * H)
        d.line([(x, y), tip], fill=INK, width=w + 2 * ol)
        _dot(d, tip, w / 2 + ol, INK)
        d.line([(x, y), tip], fill=skin, width=w)
        _dot(d, tip, w / 2, skin)
    r = 0.036 * H if hand == "fist" else 0.031 * H
    _dot(d, pos, r + ol, INK)
    _dot(d, pos, r, skin)


# ---------------------------------------------------------------- head & face


def _px(r: float, n: float) -> int:
    return max(1, int(n * r / 70))


EXPRESSIONS: dict[str, tuple[str, str, str]] = {
    # eyes, brows, mouth
    "happy": ("open", "raised", "smile"),
    "laugh": ("happy", "raised", "grin"),
    "sad": ("sad", "sad", "frown"),
    "cry": ("sad", "sad", "wobbly"),
    "angry": ("angry", "angry", "teeth"),
    "smirk": ("half", "one_up", "smirk"),
    "eyeroll": ("roll", "one_up", "flat"),
    "surprised": ("wide", "raised", "o"),
    "scared": ("pinned", "worried", "wobbly"),
    "love": ("heart", "raised", "grin"),
    "annoyed": ("half", "low", "flat"),
    "worried": ("open", "worried", "wobbly"),
    "thinking": ("up", "one_up", "side"),
    "sympathetic": ("sad", "sad", "small_smile"),
    "proud": ("happy", "raised", "smile"),
}

_BROW_REST = {"flat": "flat", "soft": "raised", "raised_right": "one_up", "thick": "flat", "worried": "worried"}


def face_parts(spec: Personality, expr: str) -> tuple[str, str, str]:
    if expr in EXPRESSIONS:
        return EXPRESSIONS[expr]
    eyes = "half" if spec.id in {"cool", "cynic"} else "open"
    mouth = "small_smile" if spec.rest_smile else ("smirk" if spec.id == "cynic" else "flat")
    return eyes, _BROW_REST.get(spec.brow_rest, "flat"), mouth


def draw_head(d: ImageDraw.ImageDraw, spec: Personality, cx: float, cy: float, r: float, act: Act, ol: int) -> None:
    _hair_back(d, spec, cx, cy, r, ol)
    d.ellipse((cx - r - ol, cy - r - ol, cx + r + ol, cy + r + ol), fill=INK)
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=spec.skin)
    draw_face(d, spec, cx, cy, r, act)
    _hair_front(d, spec, cx, cy, r, ol)
    _accessory(d, spec, cx, cy, r, ol)


def _eye_pos(cx: float, cy: float, r: float) -> tuple[float, float, float]:
    return cx + 0.12 * r, cy - 0.02 * r, 0.33 * r


def draw_face(d: ImageDraw.ImageDraw, spec: Personality, cx: float, cy: float, r: float, act: Act) -> None:
    eyes, brows, mouth_mode = face_parts(spec, act.expr)
    fx, ey, edx = _eye_pos(cx, cy, r)
    ew, eh = 0.15 * r, 0.19 * r
    lw = max(2, int(0.05 * r))
    thin = max(1, int(0.03 * r))

    if spec.blush or act.expr in {"love", "laugh", "proud"}:
        for side in (-1, 1):
            bx = fx + side * 0.56 * r
            d.ellipse((bx - 0.15 * r, cy + 0.17 * r, bx + 0.15 * r, cy + 0.29 * r), fill=(255, 120, 130, 110))

    for side in (-1, 1):
        ex = fx + side * edx
        inner = -side
        if eyes == "heart":
            heart(d, ex, ey, ew * 1.5, (230, 50, 80))
            continue
        if eyes == "happy" or (act.blink > 0.85 and eyes != "happy"):
            box = (ex - ew, ey - eh * 0.55, ex + ew, ey + eh * (0.95 if eyes == "happy" else 0.6))
            d.arc(box, 200 if eyes == "happy" else 20, 340 if eyes == "happy" else 160, fill=INK, width=lw)
            continue
        scale = 1.25 if eyes in {"wide", "pinned"} else 1.0
        w, h = ew * scale, eh * scale
        d.ellipse((ex - w - thin, ey - h - thin, ex + w + thin, ey + h + thin), fill=INK)
        d.ellipse((ex - w, ey - h, ex + w, ey + h), fill=WHITE)
        pr = w * (0.38 if eyes == "pinned" else 0.68)
        px = ex + act.look * 0.32 * w
        py = ey + 0.1 * h
        if eyes == "roll":
            px, py = ex + 0.1 * w, ey - 0.55 * h
        elif eyes == "up":
            px, py = ex + 0.35 * w * act.look, ey - 0.4 * h
        elif act.look_up:
            py -= act.look_up * 0.35 * h
        d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=spec.eye_color)
        d.ellipse((px - pr * 0.55, py - pr * 0.55, px + pr * 0.55, py + pr * 0.55), fill=INK)
        hl = pr * 0.32
        d.ellipse((px - pr * 0.45 - hl, py - pr * 0.45 - hl, px - pr * 0.45 + hl, py - pr * 0.45 + hl), fill=WHITE)

        top = ey - h - thin * 2
        lid = act.blink
        if eyes == "half":
            lid = max(lid, 0.42)
        elif eyes == "roll":
            lid = max(lid, 0.2)
        if eyes == "angry":
            lo = (ex + inner * w * 1.1, ey - 0.05 * h)
            hi = (ex - inner * w * 1.1, ey - 0.75 * h)
            d.polygon([(lo[0], top), lo, hi, (hi[0], top)], fill=spec.skin)
            d.line([lo, hi], fill=INK, width=lw)
        elif eyes == "sad":
            lo = (ex - inner * w * 1.1, ey - 0.1 * h)
            hi = (ex + inner * w * 1.1, ey - 0.72 * h)
            d.polygon([(lo[0], top), lo, hi, (hi[0], top)], fill=spec.skin)
            d.line([lo, hi], fill=INK, width=lw)
        elif lid > 0.05:
            ly = ey - h + 2 * h * lid
            d.rectangle((ex - w - thin * 2, top, ex + w + thin * 2, ly), fill=spec.skin)
            d.line([(ex - w * 1.05, ly), (ex + w * 1.05, ly)], fill=INK, width=lw)

    if act.expr == "cry":
        for side in (-1, 1):
            ex = fx + side * edx
            d.rectangle((ex - ew * 0.5, ey + eh * 0.6, ex + ew * 0.5, cy + 0.9 * r), fill=(140, 200, 255, 200))

    _brows(d, spec, fx, ey - eh - 0.13 * r, edx, r, brows)
    _mouth(d, fx, cy + 0.44 * r, r, mouth_mode, act.mouth)


def _brows(d, spec: Personality, fx: float, by: float, edx: float, r: float, kind: str) -> None:
    color = tuple(max(0, int(c * 0.65)) for c in spec.hair_color) if spec.hair != "bald" else INK
    width = max(2, int(0.075 * r * (1.5 if spec.brow_rest == "thick" else 1.0)))
    half = 0.17 * r
    for side in (-1, 1):
        bx = fx + side * edx
        inner_dy = outer_dy = 0.0
        if kind == "raised":
            inner_dy = outer_dy = -0.09 * r
        elif kind == "angry":
            inner_dy, outer_dy = 0.1 * r, -0.07 * r
        elif kind in {"sad", "worried"}:
            inner_dy, outer_dy = -0.11 * r, 0.04 * r
            if kind == "worried":
                inner_dy -= 0.05 * r
        elif kind == "one_up":
            if side == 1:
                inner_dy, outer_dy = -0.08 * r, -0.16 * r
            else:
                inner_dy = outer_dy = 0.04 * r
        elif kind == "low":
            inner_dy = outer_dy = 0.05 * r
        ix, ox = bx - side * half, bx + side * half
        d.line([(ix, by + inner_dy), (ox, by + outer_dy)], fill=color, width=width)
        _dot(d, (ix, by + inner_dy), width / 2, color)
        _dot(d, (ox, by + outer_dy), width / 2, color)


def _mouth(d, mx: float, my: float, r: float, mode: str, openness: float) -> None:
    lw = max(2, int(0.055 * r))
    mw = 0.5 * r
    o = max(0.0, min(1.0, openness))
    talking = o > 0.07
    if mode == "grin" and not talking:
        o, talking = 0.35, True

    if not talking:
        if mode in {"smile", "small_smile"}:
            w = mw * (1.0 if mode == "smile" else 0.7)
            d.arc((mx - w / 2, my - 0.22 * r, mx + w / 2, my + 0.12 * r), 25, 155, fill=INK, width=lw)
        elif mode == "frown":
            d.arc((mx - mw / 2.2, my - 0.02 * r, mx + mw / 2.2, my + 0.26 * r), 205, 335, fill=INK, width=lw)
        elif mode == "smirk":
            d.line([(mx - 0.2 * r, my + 0.04 * r), (mx + 0.08 * r, my + 0.02 * r), (mx + 0.24 * r, my - 0.08 * r)], fill=INK, width=lw, joint="curve")
        elif mode == "o":
            d.ellipse((mx - 0.1 * r, my - 0.1 * r, mx + 0.1 * r, my + 0.14 * r), fill=MOUTH, outline=INK, width=lw)
        elif mode == "wobbly":
            pts = [(mx - 0.22 * r + i * 0.088 * r, my + (0.03 if i % 2 else -0.03) * r) for i in range(6)]
            d.line(pts, fill=INK, width=lw, joint="curve")
        elif mode == "teeth":
            _teeth(d, mx, my, r, 0.16 * r, lw)
        elif mode == "side":
            d.line([(mx - 0.02 * r, my + 0.04 * r), (mx + 0.2 * r, my)], fill=INK, width=lw)
        else:
            d.line([(mx - 0.16 * r, my + 0.02 * r), (mx + 0.16 * r, my + 0.02 * r)], fill=INK, width=lw)
        return

    if mode in {"smile", "small_smile", "grin", "flat", "side"}:
        h = r * (0.1 + 0.34 * o)
        w = mw * (1.15 if mode == "grin" else 0.95)
        top = my - 0.06 * r
        box = (mx - w / 2, top - h, mx + w / 2, top + h)
        d.pieslice(box, 0, 180, fill=MOUTH)
        d.rectangle((mx - w * 0.38, top, mx + w * 0.38, top + h * 0.22), fill=WHITE)
        if o > 0.3:
            d.ellipse((mx - w * 0.2, top + h * 0.55, mx + w * 0.2, top + h * 0.92), fill=TONGUE)
        d.arc(box, 0, 180, fill=INK, width=lw)
        d.line([(mx - w / 2, top), (mx + w / 2, top)], fill=INK, width=lw)
    elif mode == "teeth":
        _teeth(d, mx, my, r, r * (0.14 + 0.2 * o), lw)
    elif mode == "o":
        s = r * (0.12 + 0.12 * o)
        d.ellipse((mx - s * 0.85, my - s, mx + s * 0.85, my + s * 1.1), fill=MOUTH, outline=INK, width=lw)
    elif mode == "smirk":
        h = r * (0.08 + 0.22 * o)
        pts = [(mx - 0.2 * r, my + 0.03 * r), (mx + 0.26 * r, my - 0.1 * r), (mx + 0.18 * r, my + h), (mx - 0.08 * r, my + h * 0.8)]
        d.polygon(pts, fill=MOUTH, outline=INK, width=lw)
    else:  # frown / wobbly: open but downturned
        h = r * (0.08 + 0.22 * o)
        w = mw * 0.7
        d.chord((mx - w / 2, my - h * 0.2, mx + w / 2, my + h * 1.6), 180, 360, fill=MOUTH, outline=INK, width=lw)


def _teeth(d, mx, my, r, h, lw) -> None:
    w = 0.52 * r
    box = (mx - w / 2, my - h / 2, mx + w / 2, my + h / 2)
    d.rounded_rectangle(box, radius=int(h * 0.35), fill=WHITE, outline=INK, width=lw)
    d.line([(mx - w / 2, my), (mx + w / 2, my)], fill=INK, width=max(1, lw // 2))
    for i in range(1, 4):
        x = mx - w / 2 + i * w / 4
        d.line([(x, my - h / 2), (x, my + h / 2)], fill=INK, width=max(1, lw // 2))


# ---------------------------------------------------------------- hair & accessories


def _hair_back(d, spec: Personality, cx, cy, r, ol) -> None:
    c = spec.hair_color
    if spec.hair == "long":
        d.rounded_rectangle((cx - 1.12 * r, cy - 0.7 * r, cx + 1.12 * r, cy + 1.25 * r), radius=int(r * 0.6), fill=c, outline=INK, width=ol)
    elif spec.hair == "ponytail":
        d.ellipse((cx - 1.55 * r, cy - 0.5 * r, cx - 0.65 * r, cy + 0.85 * r), fill=c, outline=INK, width=ol)
    elif spec.hair == "bun":
        d.ellipse((cx - 0.38 * r, cy - 1.55 * r, cx + 0.3 * r, cy - 0.82 * r), fill=c, outline=INK, width=ol)
    elif spec.hair == "curly":
        for i in range(9):
            a = math.radians(160 + i * 27.5)
            x, y = cx + math.cos(a) * r * 0.98, cy + math.sin(a) * r * 0.98
            if y > cy + 0.15 * r:
                continue
            d.ellipse((x - 0.33 * r, y - 0.33 * r, x + 0.33 * r, y + 0.33 * r), fill=c, outline=INK, width=ol)


def _hair_front(d, spec: Personality, cx, cy, r, ol) -> None:
    c = spec.hair_color
    h = spec.hair
    cap = (cx - 1.04 * r, cy - 1.1 * r, cx + 1.04 * r, cy + 0.55 * r)
    if h in {"tidy", "ponytail", "long", "bun"}:
        d.chord(cap, 192, 348, fill=c, outline=INK, width=ol)
        if h == "tidy":
            d.polygon([(cx + 0.1 * r, cy - 0.98 * r), (cx + 0.75 * r, cy - 0.55 * r), (cx + 0.2 * r, cy - 0.62 * r)], fill=c)
    elif h == "mop":
        d.chord((cx - 1.12 * r, cy - 1.2 * r, cx + 1.12 * r, cy + 0.2 * r), 180, 360, fill=c, outline=INK, width=ol)
        for i in range(5):
            x = cx - 0.8 * r + i * 0.4 * r
            d.ellipse((x - 0.24 * r, cy - 0.78 * r, x + 0.24 * r, cy - 0.38 * r), fill=c)
    elif h == "spiky":
        pts = [(cx - 1.0 * r, cy - 0.35 * r)]
        for i in range(6):
            a = math.radians(195 + i * 30)
            pts.append((cx + math.cos(a) * 1.45 * r, cy + math.sin(a) * 1.45 * r))
            b = math.radians(210 + i * 30)
            pts.append((cx + math.cos(b) * 0.9 * r, cy + math.sin(b) * 0.9 * r))
        pts.append((cx + 1.0 * r, cy - 0.35 * r))
        pts.append((cx, cy - 0.55 * r))
        d.polygon(pts, fill=c, outline=INK, width=ol)
    elif h == "messy":
        d.chord(cap, 186, 354, fill=c, outline=INK, width=ol)
        for dx, dy in ((-0.3, -1.35), (0.1, -1.45), (0.5, -1.25)):
            d.polygon([(cx + (dx - 0.18) * r, cy - 0.85 * r), (cx + dx * r, cy + dy * r), (cx + (dx + 0.18) * r, cy - 0.85 * r)], fill=c, outline=INK, width=ol)
    elif h == "buzz":
        d.chord((cx - 1.0 * r, cy - 1.04 * r, cx + 1.0 * r, cy + 0.3 * r), 196, 344, fill=c)
    elif h == "curly":
        d.chord(cap, 190, 350, fill=c)
    elif h == "bald":
        d.ellipse((cx - 0.55 * r, cy - 0.85 * r, cx - 0.2 * r, cy - 0.65 * r), fill=(255, 255, 255, 120))


def _accessory(d, spec: Personality, cx, cy, r, ol) -> None:
    a = spec.accessory
    fx, ey, edx = _eye_pos(cx, cy, r)
    if a in {"glasses", "round_glasses"}:
        rad = r * (0.26 if a == "round_glasses" else 0.24)
        w = max(2, int(0.05 * r))
        for side in (-1, 1):
            ex = fx + side * edx
            if a == "round_glasses":
                d.ellipse((ex - rad, ey - rad, ex + rad, ey + rad), outline=INK, width=w)
            else:
                d.rounded_rectangle((ex - rad, ey - rad * 0.8, ex + rad, ey + rad * 0.8), radius=int(rad * 0.3), outline=INK, width=w)
            d.arc((ex - rad * 0.75, ey - rad * 0.75, ex + rad * 0.75, ey + rad * 0.75), 200, 250, fill=WHITE, width=max(1, w // 2))
        d.line([(fx - edx + rad, ey), (fx + edx - rad, ey)], fill=INK, width=w)
        d.line([(fx - edx - rad, ey), (cx - 0.98 * r, ey - 0.06 * r)], fill=INK, width=w)
    elif a == "cap":
        col = (44, 52, 70)
        d.chord((cx - 1.06 * r, cy - 1.22 * r, cx + 1.06 * r, cy + 0.4 * r), 180, 360, fill=col, outline=INK, width=ol)
        d.polygon([(cx + 0.3 * r, cy - 0.45 * r), (cx + 1.55 * r, cy - 0.32 * r), (cx + 1.5 * r, cy - 0.2 * r), (cx + 0.6 * r, cy - 0.28 * r)], fill=col, outline=INK, width=ol)
        d.ellipse((cx - 0.12 * r, cy - 1.3 * r, cx + 0.12 * r, cy - 1.12 * r), fill=col)
    elif a == "beanie":
        col = (48, 64, 98)
        d.chord((cx - 1.08 * r, cy - 1.35 * r, cx + 1.08 * r, cy + 0.15 * r), 180, 360, fill=col, outline=INK, width=ol)
        d.rounded_rectangle((cx - 1.1 * r, cy - 0.68 * r, cx + 1.1 * r, cy - 0.4 * r), radius=int(0.1 * r), fill=(70, 88, 128), outline=INK, width=ol)
        d.ellipse((cx - 0.2 * r, cy - 1.62 * r, cx + 0.2 * r, cy - 1.22 * r), fill=(226, 84, 84), outline=INK, width=ol)
    elif a == "bow":
        bx, by = cx - 0.55 * r, cy - 0.92 * r
        col = (232, 84, 132)
        d.polygon([(bx, by), (bx - 0.45 * r, by - 0.3 * r), (bx - 0.45 * r, by + 0.3 * r)], fill=col, outline=INK, width=ol)
        d.polygon([(bx, by), (bx + 0.45 * r, by - 0.3 * r), (bx + 0.45 * r, by + 0.3 * r)], fill=col, outline=INK, width=ol)
        _dot(d, (bx, by), 0.12 * r, (190, 50, 100))
    elif a == "headphones":
        col = (60, 60, 72)
        d.arc((cx - 1.12 * r, cy - 1.22 * r, cx + 1.12 * r, cy + 0.9 * r), 190, 350, fill=INK, width=int(0.14 * r))
        for side in (-1, 1):
            x = cx + side * 1.0 * r
            d.rounded_rectangle((x - 0.16 * r, cy - 0.32 * r, x + 0.16 * r, cy + 0.22 * r), radius=int(0.1 * r), fill=col, outline=INK, width=ol)
    elif a == "earring":
        _dot(d, (cx - 0.97 * r, cy + 0.32 * r), 0.09 * r, (240, 196, 60))
    elif a == "bandana":
        col = (204, 48, 56)
        d.chord((cx - 1.03 * r, cy - 1.05 * r, cx + 1.03 * r, cy + 0.6 * r), 196, 344, fill=col, outline=INK, width=ol)
        d.polygon([(cx - 0.95 * r, cy - 0.45 * r), (cx - 1.45 * r, cy - 0.2 * r), (cx - 1.3 * r, cy + 0.05 * r)], fill=col, outline=INK, width=ol)
        for i in range(4):
            _dot(d, (cx - 0.5 * r + i * 0.35 * r, cy - 0.68 * r), 0.04 * r, WHITE)


# ---------------------------------------------------------------- emotes


def heart(d, x: float, y: float, s: float, fill, outline=None) -> None:
    r = s * 0.5
    d.ellipse((x - s, y - r * 1.4, x, y + r * 0.6), fill=fill, outline=outline)
    d.ellipse((x, y - r * 1.4, x + s, y + r * 0.6), fill=fill, outline=outline)
    d.polygon([(x - s * 0.96, y - r * 0.2), (x + s * 0.96, y - r * 0.2), (x, y + s * 1.05)], fill=fill)


def _pop(t: float) -> float:
    if t >= 0.3:
        return 1.0
    x = t / 0.3
    return 1 + 2.7 * (x - 1) ** 3 + 1.7 * (x - 1) ** 2


def draw_emote(d: ImageDraw.ImageDraw, kind: str, hx: float, hy: float, r: float, t: float, facing: int, font_for) -> None:
    """Draw an emote around a head at (hx, hy) with radius r; t is seconds since the emote began."""
    s = _pop(t)
    if s <= 0.02:
        return
    back = -facing
    if kind == "anger":
        x, y = hx + facing * 0.85 * r, hy - 0.95 * r
        q = 0.24 * r * s * (1 + 0.12 * math.sin(t * 14))
        w = max(3, int(0.08 * r))
        for sx_, sy_ in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            cx_, cy_ = x + sx_ * q, y + sy_ * q
            ang = math.degrees(math.atan2(-sy_, -sx_))
            box = (cx_ - q * 0.8, cy_ - q * 0.8, cx_ + q * 0.8, cy_ + q * 0.8)
            d.arc(box, ang - 50, ang + 50, fill=(220, 36, 48), width=w)
        if int(t * 1.6) % 2 == 0:
            for i in range(3):
                ph = (t * 1.6 + i * 0.33) % 1
                px = hx + back * (0.4 + 0.25 * i) * r
                py = hy - r * (1.25 + ph * 0.7)
                pr = r * (0.1 + 0.08 * ph)
                d.ellipse((px - pr, py - pr, px + pr, py + pr), fill=(255, 255, 255, int(200 * (1 - ph))))
    elif kind == "sweat":
        slide = (t * 0.35) % 0.45
        x, y = hx + back * 0.95 * r, hy - 0.55 * r + slide * r
        dr = 0.13 * r * s
        d.polygon([(x, y - dr * 2.4), (x - dr * 0.95, y - dr * 0.2), (x + dr * 0.95, y - dr * 0.2)], fill=(140, 205, 255), outline=(40, 110, 190))
        d.ellipse((x - dr, y - dr, x + dr, y + dr), fill=(140, 205, 255), outline=(40, 110, 190), width=max(2, int(r * 0.025)))
        d.ellipse((x - dr * 0.5, y - dr * 0.4, x - dr * 0.1, y), fill=WHITE)
    elif kind in {"exclaim", "question"}:
        glyph = "!" if kind == "exclaim" else "?"
        size = int(r * 0.9 * s)
        if size < 6:
            return
        font = font_for(size)
        bob = math.sin(t * 5) * 0.06 * r if kind == "question" else 0
        x, y = hx + facing * 0.95 * r, hy - 1.35 * r + bob
        color = (255, 214, 64) if kind == "exclaim" else (120, 200, 255)
        d.text((x, y), glyph, font=font, fill=color, anchor="mm", stroke_width=max(2, int(r * 0.06)), stroke_fill=INK)
    elif kind == "hearts":
        for i in range(3):
            ph = (t * 0.6 + i * 0.33) % 1
            x = hx + facing * (0.7 + 0.25 * math.sin(ph * 6 + i)) * r
            y = hy - r * (0.6 + ph * 1.2)
            hs = 0.16 * r * (1 - ph * 0.4) * s
            heart(d, x, y, hs, (236, 64, 96, int(255 * min(1, (1 - ph) * 2))))
    elif kind == "sparkle":
        for i, (ax_, ay_) in enumerate(((1.15, -0.9), (-1.1, -0.6), (0.9, 0.2), (-0.6, -1.3))):
            tw = (math.sin(t * 6 + i * 1.7) + 1) / 2
            x, y = hx + ax_ * r, hy + ay_ * r
            k = 0.16 * r * (0.4 + 0.6 * tw) * s
            d.polygon([(x, y - k), (x + k * 0.25, y - k * 0.25), (x + k, y), (x + k * 0.25, y + k * 0.25),
                       (x, y + k), (x - k * 0.25, y + k * 0.25), (x - k, y), (x - k * 0.25, y - k * 0.25)],
                      fill=(255, 226, 90, int(120 + 135 * tw)))
    elif kind == "gloom":
        w = max(2, int(0.045 * r))
        for i in range(4):
            x = hx + (-0.45 + i * 0.3) * r
            top = hy - 1.55 * r
            pts = [(x + 0.05 * r * math.sin(t * 3 + j + i), top + j * 0.1 * r) for j in range(5)]
            d.line(pts, fill=(110, 130, 200, 200), width=w, joint="curve")
        d.ellipse((hx - 0.6 * r, hy - 1.85 * r, hx + 0.6 * r, hy - 1.5 * r), fill=(90, 96, 124, 200))
    elif kind == "dots":
        x, y = hx + facing * 1.05 * r, hy - 1.15 * r
        bw, bh = 0.85 * r * s, 0.42 * r * s
        d.rounded_rectangle((x - bw / 2, y - bh / 2, x + bw / 2, y + bh / 2), radius=int(bh / 2), fill=WHITE, outline=INK, width=max(2, int(r * 0.03)))
        n = int(t * 3) % 4
        for i in range(n):
            dx = (-0.25 + i * 0.25) * r * s
            _dot(d, (x + dx, y), 0.06 * r * s, INK)
    elif kind == "shock":
        w = max(2, int(0.05 * r))
        for i in range(7):
            a = math.radians(-160 + i * 23)
            r0, r1 = 1.3 * r, (1.55 + 0.1 * math.sin(t * 20 + i)) * r
            d.line([(hx + math.cos(a) * r0, hy + math.sin(a) * r0), (hx + math.cos(a) * r1, hy + math.sin(a) * r1)], fill=INK, width=w)
