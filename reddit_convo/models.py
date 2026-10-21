from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Emotion = Literal[
    "joy",
    "sadness",
    "love",
    "anger",
    "fear",
    "surprise",
    "neutral",
]
Tone = Literal["witty", "heated", "somber", "supportive", "calm"]
Expression = Literal[
    "smile",
    "frown",
    "scowl",
    "smirk",
    "wide-eyes",
    "hearts",
    "rest",
]
SceneName = Literal[
    "park",
    "coffee_shop",
    "debate_table",
    "gym",
    "campfire",
    "office",
    "boutique",
    "living_room",
]
SpeakerRole = Literal["host", "guest"]


class CommentNode(BaseModel):
    id: str
    author: str
    body: str
    score: int = 0
    parent_id: str
    depth: int = 0
    replies: list[CommentNode] = Field(default_factory=list)


class Thread(BaseModel):
    id: str
    subreddit: str
    title: str
    selftext: str = ""
    author: str
    score: int = 0
    permalink: str = ""
    url: str = ""
    comments: list[CommentNode] = Field(default_factory=list)


class Speaker(BaseModel):
    id: str
    author: str
    role: SpeakerRole
    voice: str
    color: tuple[int, int, int]
    sprite_index: int = 0
    personality: str = "host"


class GraphEdge(BaseModel):
    source: str
    target: str
    comment_id: str


class SpeakerGraph(BaseModel):
    speakers: list[Speaker]
    edges: list[GraphEdge] = Field(default_factory=list)
    author_to_speaker: dict[str, str] = Field(default_factory=dict)


class DialogueTurn(BaseModel):
    index: int
    speaker_id: str
    author: str
    comment_id: str = ""
    text: str
    spoken_text: str
    reply_to: str | None = None
    reply_to_author: str | None = None
    reply_to_text: str | None = None
    depth: int = 0
    emotion: Emotion = "neutral"
    sarcasm: bool = False
    sarcasm_confidence: float = 0.0
    emotion_confidence: float = 0.0
    tone: Tone = "calm"
    expression: Expression = "rest"
    score: int = 0
    audio_path: str | None = None
    duration_ms: int = 0


class Script(BaseModel):
    post_id: str
    subreddit: str
    title: str
    scene: SceneName
    speakers: list[Speaker]
    edges: list[GraphEdge] = Field(default_factory=list)
    turns: list[DialogueTurn] = Field(default_factory=list)
    nlp_backend: str = "heuristic"
    tts_backend: str = "pending"
