"""Delivery settings per beat: how fast, how lively, how high, how loud."""

from __future__ import annotations

from dataclasses import dataclass, replace

from reddit_convo.models import Beat


@dataclass(frozen=True)
class Prosody:
    length: float = 1.0  # Piper length_scale (>1 slower)
    noise: float = 0.667  # Piper noise_scale: timbre/intonation variation
    noise_w: float = 0.8  # Piper noise_w_scale: rhythm variation
    pitch: float = 1.0  # resample pitch factor (>1 higher)
    gain: float = 1.0  # loudness relative to normal speech
    pause_ms: int = 260  # silence after the beat
    rise: float = 0.0  # pitch glide over the last third (questions)


NEUTRAL = Prosody()

MOOD: dict[str, Prosody] = {
    "neutral": NEUTRAL,
    "happy": Prosody(0.95, 0.76, 0.9, 1.04, 1.05, 230),
    "excited": Prosody(0.88, 0.82, 1.0, 1.08, 1.15, 200),
    "laugh": Prosody(0.9, 0.85, 1.05, 1.08, 1.12, 240),
    "love": Prosody(1.04, 0.72, 0.85, 1.04, 0.98, 300),
    "grateful": Prosody(1.03, 0.72, 0.85, 1.03, 1.0, 300),
    "proud": Prosody(0.97, 0.74, 0.85, 1.03, 1.1, 260),
    "encouraging": Prosody(0.97, 0.75, 0.9, 1.04, 1.1, 260),
    "confident": Prosody(0.96, 0.62, 0.75, 0.98, 1.1, 240),
    "agree": Prosody(0.96, 0.72, 0.85, 1.03, 1.05, 220),
    "disagree": Prosody(1.02, 0.66, 0.8, 0.97, 1.05, 300),
    "angry": Prosody(0.86, 0.85, 1.05, 0.95, 1.3, 170),
    "annoyed": Prosody(1.06, 0.6, 0.7, 0.96, 1.02, 340),
    "disgusted": Prosody(1.08, 0.7, 0.9, 0.95, 1.05, 320),
    "sarcastic": Prosody(1.2, 0.86, 1.12, 0.94, 1.02, 420, rise=0.06),
    "sad": Prosody(1.2, 0.52, 0.6, 0.93, 0.82, 450),
    "sympathetic": Prosody(1.12, 0.58, 0.68, 0.98, 0.88, 380),
    "worried": Prosody(0.95, 0.88, 1.15, 1.04, 0.9, 260),
    "surprised": Prosody(0.92, 0.82, 1.0, 1.1, 1.15, 230),
    "curious": Prosody(1.0, 0.72, 0.85, 1.02, 1.0, 320, rise=0.15),
    "thinking": Prosody(1.1, 0.6, 0.7, 0.99, 0.92, 360),
}

# Per-personality baseline, applied on top of the mood.
PERSONA: dict[str, Prosody] = {
    "host": NEUTRAL,
    "kind": Prosody(1.03, 0.7, 0.8, 1.02, 0.97, 280),
    "cynic": Prosody(1.06, 0.6, 0.75, 0.97, 1.0, 300),
    "spark": Prosody(0.92, 0.78, 0.95, 1.04, 1.08, 220),
    "worry": Prosody(0.96, 0.8, 1.05, 1.03, 0.92, 260),
    "cool": Prosody(1.07, 0.6, 0.72, 0.97, 0.96, 300),
    "bookish": Prosody(1.0, 0.64, 0.76, 1.0, 1.0, 280),
}


def _mix(a: float, b: float, base: float, k: float) -> float:
    return a + (b - base) * k


def prosody_for(beat: Beat, personality: str = "host", *, last: bool = False) -> Prosody:
    """Blend personality baseline with the beat's mood, scaled by intensity."""
    mood = MOOD.get(beat.mood, NEUTRAL)
    persona = PERSONA.get(personality, NEUTRAL)
    k = 0.45 + 0.9 * max(0.0, min(1.0, beat.intensity))
    p = Prosody(
        length=persona.length * (1 + (mood.length - 1) * k),
        noise=_mix(persona.noise, mood.noise, NEUTRAL.noise, k),
        noise_w=_mix(persona.noise_w, mood.noise_w, NEUTRAL.noise_w, k),
        pitch=persona.pitch * (1 + (mood.pitch - 1) * k),
        gain=persona.gain * (1 + (mood.gain - 1) * k),
        pause_ms=int((mood.pause_ms + persona.pause_ms) / 2),
        rise=mood.rise * k,
    )
    text = beat.text.rstrip()
    if text.endswith("?") and p.rise < 0.1:
        p = replace(p, rise=0.12, pause_ms=p.pause_ms + 80)
    elif text.endswith(("...", "…")):
        p = replace(p, pause_ms=p.pause_ms + 250, length=p.length * 1.04)
    if last:
        p = replace(p, pause_ms=0)
    return replace(
        p,
        length=min(max(p.length, 0.72), 1.4),
        noise=min(max(p.noise, 0.4), 1.0),
        noise_w=min(max(p.noise_w, 0.5), 1.25),
        pitch=min(max(p.pitch, 0.88), 1.14),
        gain=min(max(p.gain, 0.7), 1.4),
    )
