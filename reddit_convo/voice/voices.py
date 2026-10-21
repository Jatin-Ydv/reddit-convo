from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PiperVoiceSpec:
    name: str
    lang: str
    quality: str
    folder: str  # huggingface path after en/<lang>/

    @property
    def voice_id(self) -> str:
        return f"{self.lang}-{self.name}-{self.quality}"

    @property
    def filename(self) -> str:
        return f"{self.voice_id}.onnx"

    @property
    def hf_dir(self) -> str:
        return f"en/{self.lang}/{self.name}/{self.quality}"


HOST_VOICE = PiperVoiceSpec("lessac", "en_US", "medium", "lessac")

GUEST_VOICES = [
    PiperVoiceSpec("amy", "en_US", "medium", "amy"),
    PiperVoiceSpec("ryan", "en_US", "medium", "ryan"),
    PiperVoiceSpec("joe", "en_US", "medium", "joe"),
    PiperVoiceSpec("kathleen", "en_US", "low", "kathleen"),
    PiperVoiceSpec("norman", "en_US", "medium", "norman"),
    PiperVoiceSpec("alba", "en_GB", "medium", "alba"),
]

ALL_VOICES = [HOST_VOICE, *GUEST_VOICES]


def spec_by_name(name: str) -> PiperVoiceSpec:
    for spec in ALL_VOICES:
        if spec.name == name:
            return spec
    return HOST_VOICE
