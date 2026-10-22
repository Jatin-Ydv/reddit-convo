from __future__ import annotations

import logging
import math
import random
import shutil
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from reddit_convo.models import Beat, DialogueTurn, Script, Speaker
from reddit_convo.preprocess.beats import dominant_mood
from reddit_convo.render.art import (
    HEIGHT,
    WIDTH,
    WORLD_H,
    WORLD_W,
    ensure_assets,
    load_font,
    make_overlay,
    scene_path,
)
from reddit_convo.render.figure import INK, POSES, Act, Pose, blend_pose, draw_emote, render_figure
from reddit_convo.render.personality import Personality, assign_looks
from reddit_convo.render.scenes import HOTSPOTS
from reddit_convo.voice.piper import TURN_GAP_MS, wav_rms_envelope

log = logging.getLogger(__name__)
FPS = 24

CHAR_H = 560.0
FEET_Y = 1000.0
HEAD_WORLD_Y = FEET_Y - CHAR_H * 0.74
SLOT_X = {"L": 640.0, "R": 1280.0, "C": 960.0}
HEADER_BOTTOM = 92
TEXT_W = 600
LINE_H = 38

TALK_POSES = {
    "host": ["talk", "explain", "talk"],
    "kind": ["explain", "heart", "talk"],
    "cynic": ["shrug", "crossed_talk", "talk"],
    "spark": ["fist_pump", "explain_big", "point"],
    "worry": ["wring_talk", "cheeks", "talk"],
    "cool": ["pocket_talk", "talk", "pocket_talk"],
    "bookish": ["point_up", "explain", "chin"],
}
IDLE_POSE = {
    "host": "relaxed",
    "kind": "clasp",
    "cynic": "crossed",
    "spark": "hips",
    "worry": "wring",
    "cool": "pockets",
    "bookish": "chin",
}


class RenderError(RuntimeError):
    pass


@dataclass
class Cam:
    x: float
    y: float
    zoom: float

    @property
    def k(self) -> float:
        return WIDTH * self.zoom / WORLD_W

    @property
    def box(self) -> tuple[float, float, float, float]:
        w, h = WORLD_W / self.zoom, WORLD_H / self.zoom
        return self.x - w / 2, self.y - h / 2, self.x + w / 2, self.y + h / 2

    def to_screen(self, wx: float, wy: float) -> tuple[float, float]:
        left, top, _, _ = self.box
        return (wx - left) * self.k, (wy - top) * self.k


@dataclass
class TurnPlan:
    turn: DialogueTurn
    speaker: Speaker
    listener: Speaker | None
    start: int
    speech: int
    total: int
    env: np.ndarray
    slots: dict[str, float]
    pages: list[list[str]]
    page_starts: list[int]
    page_chars: list[int] = field(default_factory=list)
    beat_spans: list[tuple[int, int, Beat]] = field(default_factory=list)
    speaker_keys: list[tuple[int, str]] = field(default_factory=list)
    react_start: int = 0
    reaction: tuple[str, str, str | None] = ("neutral", "relaxed", None)
    closeup: tuple[int, int] | None = None


def render_video(script: Script, audio_path: Path, dest: Path) -> Path:
    ensure_assets()
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RenderError("ffmpeg is not on PATH")
    dest.parent.mkdir(parents=True, exist_ok=True)
    world = Image.open(scene_path(script.scene)).convert("RGB")
    if world.size != (WORLD_W, WORLD_H):
        world = world.resize((WORLD_W, WORLD_H))
    overlay = make_overlay(script.scene)
    looks = assign_looks(script.speakers)
    plans = plan_turns(script, looks)
    crowd = crowd_slots(script.speakers)
    total_frames = sum(p.total for p in plans)

    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{WIDTH}x{HEIGHT}", "-r", str(FPS), "-i", "-",
        "-i", str(audio_path),
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "160k",
        str(dest),
    ]
    log_path = dest.with_suffix(".ffmpeg.log")
    with log_path.open("wb") as err_f:
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=err_f)
        assert proc.stdin is not None
        try:
            done = 0
            for plan in plans:
                calm = build_stage(world, plan, looks, crowd, reacting=False)
                react = build_stage(world, plan, looks, crowd, reacting=True) if plan.listener else calm
                for i in range(plan.total):
                    stage = react if i >= plan.react_start and plan.listener else calm
                    frame = render_frame(stage, overlay, script, plan, looks, i, len(plans))
                    proc.stdin.write(frame.tobytes())
                done += plan.total
                log.info("Rendered turn %d/%d (%d%%)", plan.turn.index + 1, len(plans), 100 * done // max(total_frames, 1))
            proc.stdin.close()
            code = proc.wait()
        except BrokenPipeError:
            code = proc.wait()
            if dest.exists() and dest.stat().st_size > 0:
                return dest
            raise RenderError(f"ffmpeg pipe broke; see {log_path}")
    if code != 0:
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RenderError(f"ffmpeg failed ({code}): {tail}")
    return dest


# ---------------------------------------------------------------- planning


def _wav_seconds(path: Path) -> float:
    try:
        with wave.open(str(path), "rb") as wav:
            return wav.getnframes() / max(wav.getframerate(), 1)
    except (OSError, wave.Error, EOFError):
        return 1.2


def plan_turns(script: Script, looks: dict[str, Personality]) -> list[TurnPlan]:
    by_id = {s.id: s for s in script.speakers}
    font = load_font(30, "round")
    plans: list[TurnPlan] = []
    prev_sides: dict[str, str] = {}
    clock = 0.0
    gap_s = TURN_GAP_MS / 1000
    for turn in script.turns:
        speaker = by_id[turn.speaker_id]
        listener = by_id.get(turn.reply_to or "") if turn.reply_to != turn.speaker_id else None
        wav = Path(turn.audio_path or "")
        seconds = _wav_seconds(wav) if wav.exists() else 1.2
        start = round(clock * FPS)
        clock += seconds
        speech = max(1, round(clock * FPS) - start)
        clock += gap_s
        total = round(clock * FPS) - start

        env = wav_rms_envelope(wav, FPS) if wav.exists() else np.zeros(speech, dtype=np.float32)
        env = np.pad(env, (0, max(0, total - len(env))))[:total]
        env = np.convolve(env, np.ones(3) / 3, mode="same")

        sides = _assign_sides(speaker.id, listener.id if listener else None, prev_sides)
        prev_sides = sides
        slots = {sid: SLOT_X[side] for sid, side in sides.items()}

        text = turn.spoken_text or turn.text
        pages = _paginate(text, font, TEXT_W)
        spans = beat_spans(turn, text, speech)
        page_chars = _page_char_starts(text, pages)
        page_starts = [0] + [_char_to_frame(spans, c) for c in page_chars[1:]]

        plan = TurnPlan(
            turn=turn,
            speaker=speaker,
            listener=listener,
            start=start,
            speech=speech,
            total=total,
            env=env,
            slots=slots,
            pages=pages,
            page_starts=page_starts,
            page_chars=page_chars,
            beat_spans=spans,
        )
        spec = looks[speaker.id]
        plan.speaker_keys = _speaker_keys(turn, spec, speech, listener is not None, spans)
        if listener:
            plan.react_start = int(max(speech * 0.45, speech - FPS * 4))
            plan.reaction = _reaction(turn, looks[listener.id])
            if speech > FPS * 7:
                plan.closeup = (int(speech * 0.4), int(speech * 0.7))
        plans.append(plan)
    return plans


def _assign_sides(spk: str, lis: str | None, prev: dict[str, str]) -> dict[str, str]:
    if lis is None:
        return {spk: "C"}
    other = {"L": "R", "R": "L"}
    if prev.get(lis) in other:
        ls = prev[lis]
        return {lis: ls, spk: other[ls]}
    if prev.get(spk) in other:
        ss = prev[spk]
        return {spk: ss, lis: other[ss]}
    return {spk: "R", lis: "L"}


def _paginate(text: str, font, max_w: int, per_page: int = 3) -> list[list[str]]:
    lines: list[str] = []
    cur = ""
    for word in text.split():
        while font.getlength(word) > max_w and len(word) > 4:
            cut = len(word) * max_w // int(font.getlength(word)) - 1
            if cur:
                lines.append(cur)
                cur = ""
            lines.append(word[:cut] + "-")
            word = word[cut:]
        test = f"{cur} {word}".strip()
        if font.getlength(test) <= max_w:
            cur = test
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = ["…"]
    return [lines[i : i + per_page] for i in range(0, len(lines), per_page)]


def _page_char_starts(text: str, pages: list[list[str]]) -> list[int]:
    """Char offset in ``text`` where each page begins."""
    cur, starts = 0, []
    for page in pages:
        first = None
        for line in page:
            for word in line.split(" "):
                word = word.rstrip("-")
                k = text.find(word, cur) if word else -1
                if k < 0:
                    continue
                if first is None:
                    first = k
                cur = k + len(word)
        starts.append(cur if first is None else first)
    return starts


def beat_spans(turn: DialogueTurn, text: str, speech: int) -> list[tuple[int, int, Beat]]:
    """Frame span of every beat inside the turn (relative to turn start)."""
    beats = turn.beats or [Beat(text=text, char_end=len(text), mood="sarcastic" if turn.sarcasm else "neutral",
                                expression="smirk" if turn.sarcasm else "neutral")]
    if all(b.end_ms > 0 for b in beats):
        spans = [(round(b.start_ms * FPS / 1000), round(b.end_ms * FPS / 1000), b) for b in beats]
    else:
        lead = int(0.28 * FPS) if turn.sarcasm else 0
        total = sum(max(b.char_end - b.char_start, 1) for b in beats)
        acc, spans = 0, []
        for b in beats:
            s0 = lead + (speech - lead) * acc // total
            acc += max(b.char_end - b.char_start, 1)
            spans.append((s0, lead + (speech - lead) * acc // total, b))
    return [(min(s0, speech), max(min(e0, speech), min(s0, speech) + 1), b) for s0, e0, b in spans]


def _beat_index(spans: list[tuple[int, int, Beat]], i: int, lag: int = 0) -> int:
    idx = 0
    for n, (s0, _, _) in enumerate(spans):
        if s0 + lag <= i:
            idx = n
    return idx


def _char_to_frame(spans: list[tuple[int, int, Beat]], c: int) -> int:
    for s0, e0, b in spans:
        if c < b.char_end:
            u = (c - b.char_start) / max(b.char_end - b.char_start, 1)
            return int(s0 + max(0.0, u) * (e0 - s0))
    return spans[-1][1] if spans else 0


def _frame_to_char(spans: list[tuple[int, int, Beat]], i: int) -> float:
    if not spans:
        return 0.0
    s0, e0, b = spans[_beat_index(spans, i)]
    u = max(0.0, min(1.0, (i - s0) / max(e0 - s0, 1)))
    return b.char_start + u * (b.char_end - b.char_start)


def _speaker_keys(
    turn: DialogueTurn,
    spec: Personality,
    speech: int,
    has_listener: bool,
    spans: list[tuple[int, int, Beat]],
) -> list[tuple[int, str]]:
    idle = IDLE_POSE.get(spec.id, "relaxed")
    talk = TALK_POSES.get(spec.id, TALK_POSES["host"])
    keys: list[tuple[int, str]] = [(-100, idle)]
    f = 0
    if turn.index == 0:
        keys.append((0, "wave"))
        f = int(FPS * 1.1)
    elif has_listener:
        keys.append((0, "point"))
        f = int(FPS * 0.7)
    seed = sum(ord(c) for c in turn.author)
    k = 0
    for s0, e0, b in spans:
        s0 = max(s0, f)
        if e0 - s0 < FPS // 3:
            continue
        keys.append((s0, b.pose or talk[k % len(talk)]))
        k += 1
        g = s0 + int(FPS * (1.5 + ((seed + k) % 3) * 0.4))
        while g < e0 - FPS // 2:
            alt = talk[k % len(talk)]
            keys.append((g, b.pose if (b.pose and k % 2 == 0) else alt))
            k += 1
            g += int(FPS * (1.4 + ((seed + k) % 3) * 0.45))
        f = s0 + 1
    keys.append((speech, idle))
    keys.sort(key=lambda kv: kv[0])
    return keys


POSITIVE_MOODS = {"happy", "excited", "laugh", "love", "grateful", "proud", "encouraging", "confident", "agree"}
NOD_MOODS = {"agree", "confident", "encouraging", "proud", "grateful", "sympathetic", "sad", "thinking"}
SHAKE_MOODS = {"disagree", "annoyed", "disgusted"}
EMOTE_HOLD = {"exclaim": 1.6, "shock": 1.4, "question": 2.4, "sparkle": 1.8, "dots": 2.2}


def _reaction_to(mood: str, intensity: float, spec: Personality) -> tuple[str, str, str | None]:
    """How a listener with this personality reacts to a beat: (expression, pose, emote)."""
    pid, hot = spec.id, intensity >= 0.7
    idle = IDLE_POSE.get(pid, "relaxed")
    if mood in {"laugh", "excited"}:
        if pid == "cynic":
            return ("smirk", "crossed", None)
        return ("laugh", "cheer" if spec.energy > 0.9 and hot else "belly", "sparkle" if hot else None)
    if mood == "happy":
        return ("happy", "clasp" if pid != "spark" else "hips", None)
    if mood in {"love", "grateful", "encouraging"}:
        if pid == "cynic":
            return ("smirk", "crossed", None)
        if mood == "encouraging":
            return ("proud", "heart" if hot else "clasp", "sparkle" if hot else None)
        return ("love" if hot else "happy", "heart", "hearts" if hot else None)
    if mood == "proud":
        return ("eyeroll", "crossed", None) if pid == "cynic" else ("happy", "clasp", "sparkle")
    if mood == "confident":
        return ("thinking" if not hot else "curious", "chin", None)
    if mood == "agree":
        return ("happy", "hips" if pid == "spark" else idle, None)
    if mood == "disagree":
        if pid == "spark":
            return ("annoyed", "hips", None)
        return ("skeptical", "crossed" if pid in {"cynic", "cool"} else "chin", None)
    if mood == "angry":
        if pid == "spark":
            return ("furious" if hot else "angry", "fists", "anger")
        if pid in {"cynic", "cool"}:
            return ("annoyed", "crossed", "dots")
        return ("scared" if hot else "nervous", "defend", "sweat")
    if mood == "annoyed":
        if pid in {"cynic", "cool"}:
            return ("smirk", "crossed", None)
        return ("nervous", "wring", "sweat")
    if mood == "disgusted":
        return ("laugh", "belly", None) if pid == "spark" else ("disgusted", "crossed", None)
    if mood == "sarcastic":
        return ("eyeroll" if pid == "cynic" else "annoyed", "crossed", "dots")
    if mood in {"sad", "sympathetic"}:
        return ("sad" if hot else "sympathetic", "heart", None)
    if mood == "worried":
        return ("sympathetic", "clasp", None) if pid != "worry" else ("worried", "wring", "sweat")
    if mood == "surprised":
        return ("surprised", "cheeks" if hot else idle, "exclaim")
    if mood == "curious":
        return ("thinking", "chin", "dots")
    if mood == "thinking":
        return ("curious", idle, None)
    return ("happy" if spec.rest_smile else "neutral", idle, None)


def _reaction(turn: DialogueTurn, spec: Personality) -> tuple[str, str, str | None]:
    """Overall reaction to a turn: listener reaction shot and bystander faces."""
    if turn.beats:
        mood, intensity = dominant_mood(turn.beats)
    else:
        mood = "sarcastic" if turn.sarcasm else EMOTION_MOOD.get(turn.emotion, "neutral")
        intensity = 0.6
    return _reaction_to(mood, intensity, spec)


EMOTION_MOOD = {"anger": "angry", "joy": "happy", "love": "love", "sadness": "sad", "fear": "worried", "surprise": "surprised"}


# ---------------------------------------------------------------- crowd

MAX_CROWD = 24
CROWD_ROWS = ((815.0, 0.4), (850.0, 0.46))  # (feet y, scale) back row first


def crowd_slots(speakers: list[Speaker]) -> dict[str, tuple[Speaker, float, float, float]]:
    """Fixed home spot for every cast member so the whole thread shares one room."""
    cast = speakers[:MAX_CROWD]
    rows: list[list[Speaker]] = [cast[0::2], cast[1::2]]
    slots: dict[str, tuple[Speaker, float, float, float]] = {}
    for (feet, scale), members in zip(CROWD_ROWS, rows):
        for j, who in enumerate(members):
            x = 110 + (WORLD_W - 220) * (j + 0.5) / max(len(members), 1)
            slots[who.id] = (who, x, feet, scale)
    return slots


def build_stage(
    world: Image.Image,
    plan: TurnPlan,
    looks: dict[str, Personality],
    crowd: dict[str, tuple[Speaker, float, float, float]],
    reacting: bool,
) -> Image.Image:
    """Background plus everyone not in the current exchange, watching the speaker."""
    active = {plan.speaker.id} | ({plan.listener.id} if plan.listener else set())
    focus_x = plan.slots[plan.speaker.id]
    layer = Image.new("RGBA", world.size, (0, 0, 0, 0))
    for sid, (who, x, feet, scale) in sorted(crowd.items(), key=lambda kv: kv[1][2]):
        if sid in active:
            continue
        spec = looks[sid]
        expr = "neutral"
        if reacting and who.sprite_index % 3 != 0:
            expr = _reaction(plan.turn, spec)[0]
        act = Act(
            pose=POSES[IDLE_POSE.get(spec.id, "relaxed")],
            expr=expr,
            look=0.9,
            tilt=4.0 * (who.sprite_index % 3 - 1),
        )
        sprite = render_figure(spec, who.color, CHAR_H * scale, 1 if focus_x >= x else -1, act)
        ox, oy = int(x - sprite.anchor[0]), int(feet - sprite.anchor[1])
        layer.alpha_composite(_dim(sprite.image, 0.82), (max(0, ox), max(0, oy)), (max(0, -ox), max(0, -oy)))
    stage = world.convert("RGBA")
    stage.alpha_composite(layer.filter(ImageFilter.GaussianBlur(1.4)))
    return stage.convert("RGB")


# ---------------------------------------------------------------- per-frame


def _ease(t: float) -> float:
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def _pose_at(keys: list[tuple[int, str]], i: int, blend: int = 6) -> Pose:
    idx = 0
    for n, (f, _) in enumerate(keys):
        if f <= i:
            idx = n
    f0, name = keys[idx]
    prev = POSES[keys[idx - 1][1]] if idx > 0 else POSES[name]
    return blend_pose(prev, POSES[name], _ease((i - f0) / blend))


def _blink(gf: int, seed: int) -> float:
    period = 68 + seed % 41
    ph = (gf + seed * 13) % period
    return 1.0 if ph < 2 else (0.5 if ph == 2 else 0.0)


def _camera(plan: TurnPlan, i: int, gf: int) -> Cam:
    sx = plan.slots[plan.speaker.id]
    if plan.listener is None:
        u = _ease(i / max(plan.speech, 1))
        z, x, hy = 1.08 + 0.2 * u, sx, 420.0
    else:
        lx = plan.slots[plan.listener.id]
        mid = (sx + lx) / 2
        if i >= plan.speech:
            u = (i - plan.speech) / max(plan.total - plan.speech, 1)
            z, x, hy = 1.5 + 0.05 * u, lx + (sx - lx) * 0.1, 430.0
        elif plan.closeup and plan.closeup[0] <= i < plan.closeup[1]:
            u = (i - plan.closeup[0]) / max(plan.closeup[1] - plan.closeup[0], 1)
            z, x, hy = 1.42 + 0.06 * u, sx + (lx - sx) * 0.12, 430.0
        else:
            u = _ease(i / max(plan.speech, 1))
            z, x, hy = 1.0 + 0.16 * u, mid + (sx - mid) * (0.1 + 0.3 * u), 405.0
    x += 5 * math.sin(gf * 0.015)
    hy += 3 * math.sin(gf * 0.022)
    k = WIDTH * z / WORLD_W
    y = HEAD_WORLD_Y - (hy - HEIGHT / 2) / k
    hw, hh = WORLD_W / (2 * z), WORLD_H / (2 * z)
    return Cam(min(max(x, hw), WORLD_W - hw), min(max(y, hh), WORLD_H - hh), z)


def _speaker_act(plan: TurnPlan, spec: Personality, i: int, gf: int) -> tuple[Act, str | None, float]:
    talking = i < plan.speech
    t = i / FPS
    rms = float(plan.env[i]) if talking else 0.0
    seed = plan.speaker.sprite_index * 7 + 3
    s0, e0, beat = plan.beat_spans[_beat_index(plan.beat_spans, i)]
    since = (i - s0) / FPS
    mood, power = beat.mood, beat.intensity

    if not talking:
        expr = "happy" if mood in POSITIVE_MOODS else ("smirk" if mood == "sarcastic" else "neutral")
    else:
        expr = beat.expression
        if mood == "sarcastic":
            expr = "eyeroll" if int(since / 1.6) % 3 == 2 else "smirk"
        elif mood in {"laugh", "excited", "happy"} and rms > 0.62:
            expr = "laugh"
        elif mood == "neutral" and beat.text.rstrip().endswith("?"):
            expr = "curious"

    accent = power * math.exp(-since * 3.0) if talking else 0.0
    head_shake = 5.0 * math.sin(since * 14) * math.exp(-since * 1.6) if talking and mood in SHAKE_MOODS else 0.0
    bounce = 0.012 * abs(math.sin(t * math.pi * 2.2)) * spec.energy if (spec.idle == "bounce" and talking) else 0.0
    shake = 0.0
    if expr in {"scared", "furious"}:
        shake = (1.4 if expr == "scared" else 1.0) * math.sin(t * 42)
    elif expr == "nervous" or spec.idle == "fidget":
        shake = 0.6 * math.sin(t * 17)
    act = Act(
        pose=_pose_at(plan.speaker_keys, i),
        expr=expr,
        mouth=min(1.0, rms * (1.1 + 0.4 * power)),
        blink=_blink(gf, seed) if since > 0.25 else max(_blink(gf, seed), 1.0 if 0.08 < since < 0.16 else 0.0),
        look=0.9 if plan.listener else 0.2,
        tilt=3 * math.sin(t * 1.3 + seed) * spec.energy + rms * 4 + 7 * accent + head_shake,
        nod=rms * (0.5 + 0.5 * math.sin(t * 7.5)) + 0.8 * accent,
        bob=0.004 * math.sin(t * 1.9) + bounce,
        beat=max(0.0, min(1.0, (rms - 0.25) * 1.6)) * min(1.3, spec.energy + 0.2 + 0.4 * power),
        wave=t * 12,
        shake=shake,
    )
    emote = beat.emote if talking else None
    if emote and since > EMOTE_HOLD.get(emote, 99.0):
        emote = None
    if talking and not emote and mood == "neutral" and beat.text.rstrip().endswith("?") and e0 - i < FPS * 1.6:
        emote, since = "question", (i - (e0 - FPS * 1.6)) / FPS
    return act, emote, since


def _listener_act(plan: TurnPlan, spec: Personality, i: int, gf: int) -> tuple[Act, str | None, float]:
    t = i / FPS
    seed = plan.listener.sprite_index * 7 + 3 if plan.listener else 0
    idle = IDLE_POSE.get(spec.id, "relaxed")
    lag = int(0.35 * FPS)
    spans = plan.beat_spans
    if i >= plan.speech:
        cur, start, prev_pose, mood = plan.reaction, plan.speech, POSES[_reaction_to(spans[-1][2].mood, spans[-1][2].intensity, spec)[1]], "final"
    elif i < spans[0][0] + lag:
        cur, start, prev_pose, mood = ("neutral", idle, None), 0, POSES[idle], "neutral"
    else:
        n = _beat_index(spans, i, lag)
        b = spans[n][2]
        cur, start, mood = _reaction_to(b.mood, b.intensity, spec), spans[n][0] + lag, b.mood
        prev_pose = POSES[_reaction_to(spans[n - 1][2].mood, spans[n - 1][2].intensity, spec)[1]] if n else POSES[idle]
    expr, pose_name, emote = cur
    since = (i - start) / FPS
    pose = blend_pose(prev_pose, POSES[pose_name], _ease(since * FPS / 8))
    nod_amp = 0.65 if mood in NOD_MOODS else (0.3 if expr in {"neutral", "happy"} else 0.0)
    shake_head = 4.5 * math.sin(since * 12) * math.exp(-since * 1.2) if (mood in {"disagree", "sarcastic"} and spec.id in {"cynic", "spark", "cool"}) else 0.0
    act = Act(
        pose=pose,
        expr=expr,
        mouth=0.0,
        blink=_blink(gf, seed),
        look=0.9,
        tilt=2.5 * math.sin(t * 0.9 + seed) + shake_head + (6.0 if expr in {"curious", "thinking"} else 0.0),
        nod=nod_amp * max(0.0, math.sin(t * 3.4)),
        bob=0.004 * math.sin(t * 1.7 + 1),
        shake=1.2 * math.sin(t * 38) if expr == "scared" else (0.5 * math.sin(t * 17) if (spec.idle == "fidget" or expr == "nervous") else 0.0),
    )
    if emote and since > EMOTE_HOLD.get(emote, 3.0):
        emote = None
    return act, emote, since


def _dim(img: Image.Image, f: float) -> Image.Image:
    r, g, b, a = img.split()
    lut = [int(v * f) for v in range(256)]
    return Image.merge("RGBA", (r.point(lut), g.point(lut), b.point(lut), a))


def render_frame(
    world: Image.Image,
    overlay: Image.Image,
    script: Script,
    plan: TurnPlan,
    looks: dict[str, Personality],
    i: int,
    n_turns: int,
) -> Image.Image:
    gf = plan.start + i
    cam = _camera(plan, i, gf)
    frame = world.resize((WIDTH, HEIGHT), Image.Resampling.BILINEAR, box=cam.box).convert("RGBA")
    d = ImageDraw.Draw(frame, "RGBA")
    _ambient_back(d, script.scene, gf, cam)

    talking = i < plan.speech
    char_h = CHAR_H * cam.k
    heads: dict[str, tuple[float, float, float, int]] = {}
    sx_world = plan.slots[plan.speaker.id]
    actors: list[tuple[Speaker, bool]] = []
    if plan.listener:
        actors.append((plan.listener, False))
    actors.append((plan.speaker, True))
    emotes: list[tuple[str, str, float]] = []

    for who, is_speaker in actors:
        x_world = plan.slots[who.id]
        px, py = cam.to_screen(x_world, FEET_Y)
        if px < -char_h * 0.6 or px > WIDTH + char_h * 0.6:
            continue
        spec = looks[who.id]
        if plan.listener is None:
            facing = 1
        else:
            other = plan.slots[plan.listener.id] if is_speaker else sx_world
            facing = 1 if other >= x_world else -1
        if is_speaker:
            act, emote, et = _speaker_act(plan, spec, i, gf)
            if talking:
                glow = char_h * 0.24
                d.ellipse((px - glow, py - glow * 0.22, px + glow, py + glow * 0.22), fill=(*who.color, 90))
        else:
            act, emote, et = _listener_act(plan, spec, i, gf)
        sprite = render_figure(spec, who.color, char_h, facing, act)
        img = sprite.image if (is_speaker or not talking) else _dim(sprite.image, 0.86)
        ox, oy = int(px - sprite.anchor[0]), int(py - sprite.anchor[1])
        frame.alpha_composite(img, (max(0, ox), max(0, oy)), (max(0, -ox), max(0, -oy)))
        heads[who.id] = (ox + sprite.head[0], oy + sprite.head[1], sprite.head_r, facing)
        if emote:
            emotes.append((who.id, emote, et))

    d = ImageDraw.Draw(frame, "RGBA")
    for sid, kind, et in emotes:
        hx, hy, r, facing = heads[sid]
        draw_emote(d, kind, hx, hy, r, et, facing, lambda s: load_font(s, "bold"))
    _ambient_front(d, script.scene, gf, cam)

    frame = Image.alpha_composite(frame, overlay)
    d = ImageDraw.Draw(frame, "RGBA")
    if plan.listener and plan.listener.id in heads:
        hx, hy, r, _ = heads[plan.listener.id]
        _name_tag(d, hx, hy - r * 1.3 - 16, plan.listener, big=not talking)
    if talking and plan.speaker.id in heads:
        _bubble(d, plan, heads[plan.speaker.id], i)
    _header(d, script, plan, n_turns)
    return frame.convert("RGB")


# ---------------------------------------------------------------- UI


def _text_color(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    lum = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return INK if lum > 170 else (255, 255, 255)


def _pill(d: ImageDraw.ImageDraw, x: float, y: float, text: str, color, font, pad: int = 12, h: int = 32) -> float:
    w = font.getlength(text) + pad * 2
    d.rounded_rectangle((x, y, x + w, y + h), radius=h // 2, fill=(*color, 255), outline=(255, 255, 255, 200), width=2)
    d.text((x + pad, y + h / 2), text, font=font, fill=_text_color(color), anchor="lm")
    return x + w


def _fit(text: str, font, max_w: float) -> str:
    text = " ".join(text.split())
    if font.getlength(text) <= max_w:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if font.getlength(text[:mid] + "…") <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + "…"


def _header(d: ImageDraw.ImageDraw, script: Script, plan: TurnPlan, n_turns: int) -> None:
    turn = plan.turn
    x0, y0, x1, y1 = 18, 12, WIDTH - 18, 12 + 72
    d.rounded_rectangle((x0, y0, x1, y1), radius=18, fill=(14, 14, 24, 215))
    bold = load_font(21, "bold")
    small = load_font(18, "medium")
    quote = load_font(18, "italic")

    depth = 0 if turn.index == 0 else turn.depth + 1
    x = x0 + 16
    for k in range(min(depth, 5)):
        d.line([(x + k * 9, y0 + 12), (x + k * 9, y1 - 12)], fill=(255, 255, 255, 70 + 30 * k), width=3)
    x += min(depth, 5) * 9 + (6 if depth else 0)
    y = y0 + 8
    x = _pill(d, x, y, f"u/{plan.speaker.author}", plan.speaker.color, bold, h=30)
    if turn.index == 0 or plan.listener is None:
        d.text((x + 10, y + 15), f"posted in r/{script.subreddit}", font=small, fill=(200, 200, 214), anchor="lm")
        snippet = script.title
    else:
        ax = x + 12
        d.line([(ax, y + 8), (ax, y + 18), (ax + 14, y + 18)], fill=(200, 200, 214), width=3)
        d.polygon([(ax + 14, y + 13), (ax + 20, y + 18), (ax + 14, y + 23)], fill=(200, 200, 214))
        d.text((ax + 26, y + 15), "replying to", font=small, fill=(200, 200, 214), anchor="lm")
        x = ax + 26 + small.getlength("replying to") + 10
        _pill(d, x, y, f"u/{plan.listener.author}", plan.listener.color, bold, h=30)
        snippet = turn.reply_to_text or ""
    if snippet:
        text = _fit(snippet, quote, x1 - x0 - 210)
        d.text((x0 + 16 + min(depth, 5) * 9 + (6 if depth else 0), y1 - 18), f"“{text}”", font=quote, fill=(170, 172, 190), anchor="lm")

    right = x1 - 16
    counter = f"{turn.index + 1}/{n_turns}"
    d.text((right, y + 15), counter, font=small, fill=(170, 172, 190), anchor="rm")
    score = f"{turn.score}"
    sx = right - small.getlength(score)
    d.text((right, y1 - 18), score, font=small, fill=(255, 140, 80), anchor="rm")
    d.polygon([(sx - 22, y1 - 12), (sx - 14, y1 - 25), (sx - 6, y1 - 12)], fill=(255, 140, 80))


def _name_tag(d: ImageDraw.ImageDraw, hx: float, y: float, who: Speaker, big: bool) -> None:
    font = load_font(22 if big else 18, "bold")
    text = f"u/{who.author}"
    h = 34 if big else 28
    w = font.getlength(text) + 24
    x = min(max(hx - w / 2, 10), WIDTH - 10 - w)
    y = max(HEADER_BOTTOM + 4, y - h)
    _pill(d, x, y, text, who.color, font, h=h)


def _bubble(d: ImageDraw.ImageDraw, plan: TurnPlan, head: tuple[float, float, float, int], i: int) -> None:
    hx, hy, r, _ = head
    page_idx = 0
    for n, s in enumerate(plan.page_starts):
        if s <= i:
            page_idx = n
    page = plan.pages[page_idx]
    p_start = plan.page_starts[page_idx]

    font = load_font(30, "round")
    color = plan.speaker.color
    pad_x, pad_y = 26, 22
    tw = max(font.getlength(line) for line in page)
    bw = max(tw + 2 * pad_x, 300)
    bh = len(page) * LINE_H + 2 * pad_y
    toward = 0 if plan.listener is None else (1 if hx < WIDTH / 2 else -1)
    cx = hx + toward * bw * 0.2
    bx0 = min(max(cx - bw / 2, 20), WIDTH - 20 - bw)
    head_top = hy - r * 1.3
    by1 = head_top - 30
    by0 = by1 - bh
    if by0 < HEADER_BOTTOM + 26:
        by0 = HEADER_BOTTOM + 26
        by1 = by0 + bh
    pop = 1 - _ease((i - p_start) / 4)
    by0 += 10 * pop
    by1 += 10 * pop

    base_x = min(max(hx, bx0 + 50), bx0 + bw - 50)
    tip = (hx + toward * r * 0.15, max(by1 + 18, hy - r * 1.05))
    tail = [(base_x - 20, by1 - 4), (base_x + 20, by1 - 4), tip]
    d.rounded_rectangle((bx0 + 5, by0 + 7, bx0 + bw + 5, by1 + 7), radius=24, fill=(0, 0, 0, 60))
    d.polygon(tail, fill=(255, 255, 255), outline=(*color, 255), width=5)
    d.rounded_rectangle((bx0, by0, bx0 + bw, by1), radius=24, fill=(255, 255, 255), outline=(*color, 255), width=5)
    d.polygon([(base_x - 14, by1 - 8), (base_x + 14, by1 - 8), (tip[0], tip[1] - 9)], fill=(255, 255, 255))

    spoken = _frame_to_char(plan.beat_spans, i) - plan.page_chars[page_idx] + 2
    y = by0 + pad_y + LINE_H / 2 - 2
    seen = 0
    for line in page:
        x = bx0 + pad_x
        for word in line.split(" "):
            done = seen + len(word) / 2 <= spoken
            d.text((x, y), word, font=font, fill=INK if done else (150, 150, 162), anchor="lm")
            x += font.getlength(word + " ")
            seen += len(word) + 1
        y += LINE_H

    tab_font = load_font(20, "bold")
    label = f"u/{plan.speaker.author}"
    tx = _pill(d, bx0 + 18, by0 - 17, label, color, tab_font, h=32)
    if plan.turn.index == 0:
        _pill(d, tx + 6, by0 - 15, "OP", (255, 112, 40), load_font(16, "bold"), pad=8, h=28)
    elif plan.turn.sarcasm:
        _pill(d, tx + 6, by0 - 15, "/s  sarcasm", (250, 196, 60), load_font(16, "bold"), pad=8, h=28)
    if len(plan.pages) > 1:
        dots_x = bx0 + bw - 18 - 14 * (len(plan.pages) - 1)
        for n in range(min(len(plan.pages), 12)):
            cxp = dots_x + n * 14 if len(plan.pages) <= 12 else dots_x
            fill = (*color, 255) if n == page_idx else (210, 210, 220, 255)
            d.ellipse((cxp - 4, by1 - 14, cxp + 4, by1 - 6), fill=fill)


# ---------------------------------------------------------------- ambient motion


def _ambient_back(d: ImageDraw.ImageDraw, scene: str, gf: int, cam: Cam) -> None:
    t = gf / FPS
    k = cam.k
    spots = HOTSPOTS.get(scene, {})
    if scene == "park":
        rng = random.Random(scene)
        for n in range(14):
            x0, speed, size = rng.uniform(0, WIDTH), rng.uniform(30, 70), rng.uniform(5, 10)
            y = (rng.uniform(0, HEIGHT) + t * speed) % (HEIGHT + 40) - 20
            x = (x0 + t * 25 + 30 * math.sin(t * 1.3 + n)) % WIDTH
            col = (250, 170, 190, 200) if n % 2 else (120, 180, 90, 200)
            d.ellipse((x - size, y - size * 0.5, x + size, y + size * 0.5), fill=col)
    elif scene == "living_room":
        x0, y0, x1, y1 = spots["window"]  # type: ignore[misc]
        rng = random.Random(1)
        for _ in range(46):
            wx = rng.uniform(x0, x1)
            speed = rng.uniform(500, 800)
            wy = y0 + (rng.uniform(0, y1 - y0) + t * speed) % (y1 - y0)
            ax, ay = cam.to_screen(wx, wy)
            bx, by = cam.to_screen(wx - 6, min(wy + 34, y1))
            d.line([(ax, ay), (bx, by)], fill=(180, 200, 255, 110), width=max(1, int(2 * k)))
    elif scene == "coffee_shop":
        for n, (wx, wy) in enumerate(spots["steam"]):  # type: ignore[union-attr]
            for j in range(4):
                ph = (t * 0.5 + j * 0.25 + n * 0.13) % 1
                sx, sy = cam.to_screen(wx + 14 * math.sin(ph * 6 + j), wy - ph * 120)
                rr = (10 + 16 * ph) * k
                d.ellipse((sx - rr, sy - rr, sx + rr, sy + rr), fill=(255, 255, 255, int(70 * (1 - ph))))
    elif scene == "campfire":
        fx, fy = spots["fire"]  # type: ignore[misc]
        sx, sy = cam.to_screen(fx, fy)
        pulse = 1 + 0.08 * math.sin(t * 9) + 0.05 * math.sin(t * 23)
        g = 300 * k * pulse
        d.ellipse((sx - g, sy - g * 0.7, sx + g, sy + g * 0.3), fill=(255, 150, 60, 26))
        for j, (col, hgt, wid) in enumerate((((255, 90, 30), 170, 70), ((255, 160, 40), 120, 50), ((255, 236, 140), 70, 28))):
            h = hgt * k * (1 + 0.14 * math.sin(t * 11 + j * 2))
            w = wid * k
            sway = 12 * k * math.sin(t * 7 + j)
            d.polygon([(sx - w, sy - 10 * k), (sx - w * 0.7, sy - h * 0.45), (sx + sway, sy - h), (sx + w * 0.7, sy - h * 0.45), (sx + w, sy - 10 * k)], fill=(*col, 235))
            d.ellipse((sx - w, sy - 40 * k, sx + w, sy + 6 * k), fill=(*col, 235))
    elif scene == "boutique":
        rng = random.Random(5)
        for n in range(26):
            nx = rng.choice((140, 760, 1380)) + rng.uniform(40, 360)
            wy = rng.choice((330, 470, 610)) - rng.uniform(30, 90)
            tw = max(0.0, math.sin(t * 2.2 + n * 1.9))
            if tw < 0.6:
                continue
            sx, sy = cam.to_screen(nx, wy)
            s = 9 * k * tw
            d.polygon([(sx, sy - s * 2), (sx + s * 0.4, sy), (sx, sy + s * 2), (sx - s * 0.4, sy)], fill=(255, 255, 255, 220))
            d.polygon([(sx - s * 2, sy), (sx, sy - s * 0.4), (sx + s * 2, sy), (sx, sy + s * 0.4)], fill=(255, 255, 255, 220))


def _ambient_front(d: ImageDraw.ImageDraw, scene: str, gf: int, cam: Cam) -> None:
    t = gf / FPS
    rng = random.Random(scene)
    if scene == "campfire":
        fx, fy = HOTSPOTS["campfire"]["fire"]  # type: ignore[misc]
        for n in range(18):
            ph = (t * rng.uniform(0.3, 0.6) + n / 18) % 1
            wx = fx + rng.uniform(-60, 60) + 40 * math.sin(ph * 8 + n)
            wy = fy - 60 - ph * 520
            sx, sy = cam.to_screen(wx, wy)
            s = 4 * cam.k * (1 - ph) + 1
            d.ellipse((sx - s, sy - s, sx + s, sy + s), fill=(255, 190, 80, int(230 * (1 - ph))))
    elif scene in {"coffee_shop", "office", "debate_table", "gym", "boutique", "living_room"}:
        for n in range(16):
            x = (rng.uniform(0, WIDTH) + 12 * math.sin(t * 0.4 + n)) % WIDTH
            y = (rng.uniform(0, HEIGHT) - t * rng.uniform(4, 12)) % HEIGHT
            s = rng.uniform(1.5, 3.2)
            d.ellipse((x - s, y - s, x + s, y + s), fill=(255, 250, 230, 60))
