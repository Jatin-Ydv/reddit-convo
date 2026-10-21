from __future__ import annotations

import logging
import shutil
import subprocess
import wave
from pathlib import Path

import numpy as np

from reddit_convo.models import Script, Tone
from reddit_convo.paths import MODEL_CACHE
from reddit_convo.voice.voices import HOST_VOICE, PiperVoiceSpec, spec_by_name

log = logging.getLogger(__name__)

LENGTH_BY_TONE: dict[Tone, float] = {
    "heated": 0.85,
    "somber": 1.18,
    "witty": 1.05,
    "supportive": 1.0,
    "calm": 1.0,
}
ESPEAK_RATE: dict[Tone, int] = {
    "heated": 185,
    "somber": 125,
    "witty": 160,
    "supportive": 150,
    "calm": 155,
}

TURN_GAP_MS = 500

_VOICE_MODELS: dict[str, object] = {}


class TtsError(RuntimeError):
    pass


def synthesize_script(script: Script, audio_dir: Path) -> str:
    audio_dir.mkdir(parents=True, exist_ok=True)
    backend = _detect_backend()
    script.tts_backend = backend
    wav_paths: list[Path] = []
    for turn in script.turns:
        out = audio_dir / f"{turn.index:02d}_{turn.speaker_id}.wav"
        if not turn.spoken_text.strip():
            _write_silence(out, ms=400)
        else:
            extra_pause = 280 if turn.sarcasm else 0
            _synthesize_turn(
                text=turn.spoken_text,
                voice_name=next(s.voice for s in script.speakers if s.id == turn.speaker_id),
                tone=turn.tone,
                dest=out,
                backend=backend,
                leading_silence_ms=extra_pause,
            )
        turn.audio_path = str(out)
        turn.duration_ms = _wav_duration_ms(out)
        wav_paths.append(out)
    concat_wavs(wav_paths, audio_dir / "full.wav", gap_ms=TURN_GAP_MS)
    return backend


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


def _synthesize_turn(
    text: str,
    voice_name: str,
    tone: Tone,
    dest: Path,
    backend: str,
    leading_silence_ms: int,
) -> None:
    tmp = dest.with_suffix(".raw.wav")
    if backend == "piper":
        _piper_synth(text, spec_by_name(voice_name), tone, tmp)
    else:
        _espeak_synth(text, voice_name, tone, tmp)
    if leading_silence_ms:
        silence = dest.with_name(f"_silence_{leading_silence_ms}.wav")
        _write_silence(silence, leading_silence_ms)
        concat_wavs([silence, tmp], dest, gap_ms=0)
        tmp.unlink(missing_ok=True)
        silence.unlink(missing_ok=True)
    else:
        tmp.replace(dest)


def _piper_synth(text: str, spec: PiperVoiceSpec, tone: Tone, dest: Path) -> None:
    from piper.config import SynthesisConfig

    voice = _load_piper_voice(spec)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(dest), "wb") as wav_file:
        voice.synthesize_wav(
            text,
            wav_file,
            syn_config=SynthesisConfig(length_scale=LENGTH_BY_TONE[tone]),
        )


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


def _espeak_synth(text: str, voice_name: str, tone: Tone, dest: Path) -> None:
    binary = shutil.which("espeak-ng") or shutil.which("espeak")
    if not binary:
        raise TtsError("espeak-ng is not installed")
    voice = {
        "lessac": "en-us",
        "amy": "en-us+f3",
        "ryan": "en-us+m3",
        "joe": "en+m2",
        "kathleen": "en-us+f2",
        "norman": "en-gb+m3",
        "alba": "en-gb+f2",
    }.get(voice_name, "en")
    dest.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [binary, "-w", str(dest), "-s", str(ESPEAK_RATE[tone]), "-v", voice, "--", text],
        check=True,
        capture_output=True,
    )


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
