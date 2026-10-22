from __future__ import annotations

import logging
import shutil
import subprocess
import tempfile
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from reddit_convo.models import Beat, DialogueTurn, Script, Tone
from reddit_convo.paths import MODEL_CACHE
from reddit_convo.voice.prosody import Prosody, prosody_for
from reddit_convo.voice.voices import HOST_VOICE, PiperVoiceSpec, spec_by_name

log = logging.getLogger(__name__)

TONE_MOOD: dict[Tone, str] = {
    "heated": "angry",
    "somber": "sad",
    "witty": "sarcastic",
    "supportive": "encouraging",
    "calm": "neutral",
}
ESPEAK_VOICE = {
    "lessac": "en-us",
    "amy": "en-us+f3",
    "ryan": "en-us+m3",
    "joe": "en+m2",
    "kathleen": "en-us+f2",
    "norman": "en-gb+m3",
    "alba": "en-gb+f2",
}

TURN_GAP_MS = 500
SAMPLE_RATE = 22050
SARCASM_LEAD_MS = 280
TARGET_RMS = 0.1

_VOICE_MODELS: dict[str, object] = {}


class TtsError(RuntimeError):
    pass


def synthesize_script(script: Script, audio_dir: Path) -> str:
    audio_dir.mkdir(parents=True, exist_ok=True)
    backend = _detect_backend()
    script.tts_backend = backend
    by_id = {s.id: s for s in script.speakers}
    wav_paths: list[Path] = []
    for turn in script.turns:
        out = audio_dir / f"{turn.index:02d}_{turn.speaker_id}.wav"
        speaker = by_id[turn.speaker_id]
        if not turn.spoken_text.strip():
            _write_silence(out, ms=400)
        else:
            audio = synthesize_turn(turn, speaker.voice, speaker.personality, backend)
            _write_wav(out, audio)
        turn.audio_path = str(out)
        turn.duration_ms = _wav_duration_ms(out)
        wav_paths.append(out)
    concat_wavs(wav_paths, audio_dir / "full.wav", gap_ms=TURN_GAP_MS)
    return backend


def synthesize_turn(turn: DialogueTurn, voice_name: str, personality: str, backend: str) -> np.ndarray:
    """Render a turn beat by beat; fills each beat's start_ms/end_ms. Returns float32 audio."""
    beats = turn.beats or [
        Beat(text=turn.spoken_text, char_end=len(turn.spoken_text), mood=TONE_MOOD[turn.tone])
    ]
    lead = SARCASM_LEAD_MS if turn.sarcasm else 0
    parts: list[np.ndarray] = [np.zeros(_n(lead), dtype=np.float32)]
    cursor = _n(lead)
    for k, beat in enumerate(beats):
        p = prosody_for(beat, personality, last=k == len(beats) - 1)
        if backend == "piper":
            raw = _piper_beat(beat.text, spec_by_name(voice_name), p)
            shaped = shape_audio(raw, p)
        else:
            raw = _espeak_beat(beat.text, voice_name, p)
            shaped = shape_audio(raw, replace(p, pitch=1.0, rise=0.0))
        beat.start_ms = int(1000 * cursor / SAMPLE_RATE)
        cursor += shaped.size
        beat.end_ms = int(1000 * cursor / SAMPLE_RATE)
        pause = np.zeros(_n(p.pause_ms), dtype=np.float32)
        parts += [shaped, pause]
        cursor += pause.size
    return np.concatenate(parts)


def _n(ms: float) -> int:
    return int(SAMPLE_RATE * ms / 1000)


def _detect_backend() -> str:
    try:
        import piper  # noqa: F401

        return "piper"
    except ImportError:
        pass
    if shutil.which("espeak-ng") or shutil.which("espeak"):
        return "espeak"
    raise TtsError(
        "No TTS backend found. Install piper-tts (`pip install piper-tts`) "
        "or espeak-ng."
    )


def _piper_beat(text: str, spec: PiperVoiceSpec, p: Prosody) -> np.ndarray:
    from piper.config import SynthesisConfig

    voice = _load_piper_voice(spec)
    config = SynthesisConfig(
        length_scale=p.length * p.pitch,  # pitch-up resampling shortens audio by 1/pitch
        noise_scale=p.noise,
        noise_w_scale=p.noise_w,
        normalize_audio=True,
    )
    chunks: list[np.ndarray] = []
    rate = SAMPLE_RATE
    for chunk in voice.synthesize(text, syn_config=config):
        rate = chunk.sample_rate
        if chunks:
            chunks.append(np.zeros(int(rate * 0.14), dtype=np.float32))
        chunks.append(chunk.audio_float_array.astype(np.float32))
    if not chunks:
        return np.zeros(_n(200), dtype=np.float32)
    audio = np.concatenate(chunks)
    return _resample_float(audio, rate, SAMPLE_RATE)


def _espeak_beat(text: str, voice_name: str, p: Prosody) -> np.ndarray:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise TtsError("espeak-ng is not installed")
    speed = int(165 / p.length)
    pitch = int(min(max(50 * p.pitch + 120 * p.rise, 10), 90))
    with tempfile.TemporaryDirectory() as tmp:
        dest = Path(tmp) / "beat.wav"
        subprocess.run(
            [binary, "-w", str(dest), "-s", str(speed), "-p", str(pitch),
             "-v", ESPEAK_VOICE.get(voice_name, "en"), "--", text],
            check=True,
            capture_output=True,
        )
        audio, rate = _read_mono(dest)
    return _resample_float(audio.astype(np.float32) / 32768.0, rate, SAMPLE_RATE)


def shape_audio(audio: np.ndarray, p: Prosody) -> np.ndarray:
    """Pitch shift (with optional end-of-sentence glide) and loudness for one beat."""
    if audio.size < 2:
        return audio.astype(np.float32)
    n = audio.size
    if abs(p.pitch - 1.0) > 1e-3 or p.rise > 1e-3:
        u = np.linspace(0.0, 1.0, n, dtype=np.float32)
        ramp = np.clip((u - 0.7) / 0.3, 0.0, 1.0)
        rate = p.pitch * (1.0 + p.rise * ramp * ramp * (3 - 2 * ramp))
        out_pos = np.cumsum(1.0 / rate)
        src = np.interp(np.arange(int(out_pos[-1])), out_pos, np.arange(n))
        audio = np.interp(src, np.arange(n), audio).astype(np.float32)
    voiced = audio[np.abs(audio) > 0.02]
    rms = float(np.sqrt(np.mean(voiced**2))) if voiced.size else 0.0
    if rms > 1e-6:
        audio = audio * (TARGET_RMS * p.gain / rms)
    return np.tanh(audio).astype(np.float32)


def _resample_float(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or audio.size == 0:
        return audio.astype(np.float32)
    n = int(audio.size * dst / src)
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _write_wav(path: Path, audio: np.ndarray) -> None:
    pcm = (np.clip(audio, -1.0, 1.0) * 32000).astype(np.int16)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(pcm.tobytes())


def _load_piper_voice(spec: PiperVoiceSpec):
    if spec.name in _VOICE_MODELS:
        return _VOICE_MODELS[spec.name]
    from piper import PiperVoice

    model_path = ensure_voice(spec)
    log.info("Loading Piper voice %s", spec.voice_id)
    voice = PiperVoice.load(str(model_path))
    _VOICE_MODELS[spec.name] = voice
    return voice


def ensure_voice(spec: PiperVoiceSpec) -> Path:
    voice_dir = MODEL_CACHE / "piper"
    voice_dir.mkdir(parents=True, exist_ok=True)
    onnx = voice_dir / spec.filename
    cfg = voice_dir / f"{spec.filename}.json"
    if onnx.exists() and onnx.stat().st_size > 0 and cfg.exists():
        return onnx
    log.info("Downloading Piper voice %s", spec.voice_id)
    try:
        from piper.download_voices import download_voice

        download_voice(spec.voice_id, voice_dir)
        if onnx.exists() and onnx.stat().st_size > 0:
            return onnx
        raise TtsError(f"Piper download produced no file for {spec.voice_id}")
    except Exception as exc:
        if spec.name == HOST_VOICE.name:
            raise TtsError(f"Could not download host voice: {exc}") from exc
        log.warning("Voice %s failed (%s); falling back to %s", spec.name, exc, HOST_VOICE.name)
        return ensure_voice(HOST_VOICE)


def _write_silence(path: Path, ms: int, sample_rate: int = 22050) -> None:
    n = int(sample_rate * ms / 1000)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n)


def _wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        frames = wav.getnframes()
        rate = wav.getframerate()
    return int(1000 * frames / max(rate, 1))


def concat_wavs(paths: list[Path], dest: Path, gap_ms: int = 250) -> None:
    chunks: list[np.ndarray] = []
    target_rate = 22050
    for index, path in enumerate(paths):
        audio, rate = _read_mono(path)
        if rate != target_rate:
            audio = _resample(audio, rate, target_rate)
        chunks.append(audio)
        if gap_ms and index < len(paths) - 1:
            chunks.append(np.zeros(int(target_rate * gap_ms / 1000), dtype=np.int16))
    combined = np.concatenate(chunks) if chunks else np.zeros(1, dtype=np.int16)
    with wave.open(str(dest), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(target_rate)
        wav.writeframes(combined.tobytes())


def _read_mono(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wav:
        channels = wav.getnchannels()
        width = wav.getsampwidth()
        rate = wav.getframerate()
        frames = wav.readframes(wav.getnframes())
    if width == 2:
        audio = np.frombuffer(frames, dtype=np.int16)
    elif width == 1:
        audio = (np.frombuffer(frames, dtype=np.uint8).astype(np.int16) - 128) * 256
    elif width == 4:
        audio = (np.frombuffer(frames, dtype=np.int32) // 65536).astype(np.int16)
    else:
        raise TtsError(f"Unsupported sample width {width} in {path}")
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1).astype(np.int16)
    return audio, rate


def _resample(audio: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or audio.size == 0:
        return audio
    duration = audio.size / src
    n = int(duration * dst)
    if n <= 0:
        return audio
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n, endpoint=False)
    resampled = np.interp(x_new, x_old, audio.astype(np.float32))
    return np.clip(resampled, -32768, 32767).astype(np.int16)


def wav_rms_envelope(path: Path, fps: int) -> np.ndarray:
    audio, rate = _read_mono(path)
    samples_per_frame = max(int(rate / fps), 1)
    n_frames = int(np.ceil(audio.size / samples_per_frame))
    envelope = np.zeros(n_frames, dtype=np.float32)
    for i in range(n_frames):
        chunk = audio[i * samples_per_frame : (i + 1) * samples_per_frame]
        if chunk.size:
            envelope[i] = float(np.sqrt(np.mean(chunk.astype(np.float32) ** 2)))
    peak = envelope.max() if envelope.size else 1.0
    if peak > 0:
        envelope /= peak
    return envelope
