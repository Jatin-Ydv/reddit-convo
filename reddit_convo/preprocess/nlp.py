from __future__ import annotations

import logging
import os
import re
from functools import lru_cache
from typing import TYPE_CHECKING

from reddit_convo.models import Emotion, Expression, SceneName, Tone
from reddit_convo.paths import MODEL_CACHE

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

EMOTION_MODEL = "AdamCodd/distilbert-base-uncased-finetuned-emotion-balanced"
SARCASM_MODEL = "dima806/sarcasm-detection-distilbert"

EMOTION_FLOOR = 0.45
SARCASM_FLOOR = 0.75

SCENE_KEYWORDS: dict[SceneName, tuple[str, ...]] = {
    "park": (
        "happy",
        "happier",
        "grateful",
        "gratitude",
        "mindful",
        "positivity",
        "positive",
        "self improvement",
        "self-improvement",
        "getbetter",
        "decidingtobebetter",
        "selfimprovement",
        "mademesmile",
        "kind",
        "kindness",
        "growth",
        "healing",
        "wellness",
        "meditation",
    ),
    "gym": (
        "gym",
        "fitness",
        "workout",
        "lift",
        "lifting",
        "weight",
        "bodybuilding",
        "loseweight",
        "run",
        "running",
        "marathon",
        "protein",
        "muscle",
    ),
    "debate_table": (
        "aita",
        "amitheasshole",
        "unpopularopinion",
        "debate",
        "politics",
        "cmv",
        "changemyview",
        "wrong",
        "argument",
        "disagree",
        "controversial",
        "legaladvice",
    ),
    "office": (
        "cscareerquestions",
        "jobs",
        "job",
        "work",
        "career",
        "programming",
        "interview",
        "salary",
        "boss",
        "coworker",
        "layoff",
        "experiencedevs",
        "tech",
    ),
    "campfire": (
        "tifu",
        "story",
        "nosleep",
        "offmychest",
        "confession",
        "remember",
        "once",
        "happened",
        "ghost",
        "scary",
    ),
    "coffee_shop": (
        "advice",
        "casual",
        "explainlikeimfive",
        "eli5",
        "help",
        "should i",
        "friend",
    ),
    "living_room": (
        "breakup",
        "break up",
        "broke up",
        "relationship",
        "dating",
        "marriage",
        "married",
        "divorce",
        "girlfriend",
        "boyfriend",
        "husband",
        "wife",
        "my ex",
        "lonely",
        "heartbreak",
        "family",
        "parents",
    ),
    "boutique": (
        "perfume",
        "fragrance",
        "cologne",
        "scent",
        "skincare",
        "makeup",
        "fashion",
        "sneaker",
        "watches",
        "outfit",
        "shopping",
        "haul",
        "clone",
        "dupe",
        "buyitforlife",
        "gadgets",
    ),
}

_JOY = re.compile(
    r"\b(happy|glad|love this|awesome|amazing|yay|congrats|proud|delight|win)\b",
    re.I,
)
_SAD = re.compile(r"\b(sad|sorry|depress|miserable|cry|hurt|lonely|miss you)\b", re.I)
_ANGER = re.compile(r"\b(angry|furious|hate|stupid|idiot|rage|worst|bs|bullshit)\b", re.I)
_FEAR = re.compile(r"\b(afraid|scared|worry|anxious|panic|terrified|nervous)\b", re.I)
_SURPRISE = re.compile(r"\b(wow|whoa|unbelievable|shocked|no way|wait what)\b", re.I)
_LOVE = re.compile(r"\b(love|heart|adorable|sweet|wholesome|bless)\b", re.I)
_SARCASM = re.compile(
    r"(/s\b|yeah right|oh great|sure jan|totally not|as if|wow another|"
    r"just what i needed|can't wait|imagine thinking)",
    re.I,
)


class TagResult(dict):
    emotion: Emotion
    emotion_confidence: float
    sarcasm: bool
    sarcasm_confidence: float
    tone: Tone
    expression: Expression
    backend: str


@lru_cache(maxsize=1)
def _load_pipelines() -> tuple[object, object] | None:
    if os.environ.get("REDDIT_CONVO_NLP", "auto").lower() in {"heuristic", "off", "0"}:
        return None
    try:
        from transformers import pipeline
    except ImportError:
        log.info("transformers not installed; using heuristic NLP")
        return None
    os.environ.setdefault("HF_HOME", str(MODEL_CACHE / "hf"))
    try:
        emotion = pipeline(
            "text-classification",
            model=EMOTION_MODEL,
            truncation=True,
            device="cpu",
        )
        sarcasm = pipeline(
            "text-classification",
            model=SARCASM_MODEL,
            truncation=True,
            device="cpu",
        )
        log.info("Loaded DistilBERT emotion + sarcasm classifiers")
        return emotion, sarcasm
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not load DistilBERT models (%s); using heuristic NLP", exc)
        return None


def _is_sarcastic_label(label: str) -> bool:
    normalized = label.lower().replace("_", " ").replace("-", " ").strip()
    if any(
        token in normalized
        for token in ("no sarcasm", "not sarcasm", "not sarcastic", "non sarcasm")
    ):
        return False
    if normalized in {"0", "label 0", "normal", "sincere"}:
        return False
    return "sarcas" in normalized or normalized in {"1", "label 1", "ironic", "irony"}


def _model_tags(text: str) -> tuple[Emotion, float, bool, float] | None:
    pipes = _load_pipelines()
    if pipes is None:
        return None
    emotion_pipe, sarcasm_pipe = pipes
    clipped = text[:512]
    try:
        emo = emotion_pipe(clipped)[0]
        sar = sarcasm_pipe(clipped)[0]
    except Exception as exc:  # noqa: BLE001
        log.warning("NLP inference failed (%s)", exc)
        return None
    emotion_label = str(emo.get("label", "neutral")).lower()
    emotion_score = float(emo.get("score", 0.0))
    if emotion_label not in {"joy", "sadness", "love", "anger", "fear", "surprise"}:
        emotion_label = "neutral"
        emotion_score = 0.0
    if emotion_score < EMOTION_FLOOR:
        emotion_label = "neutral"
    sarcasm_flag = _is_sarcastic_label(str(sar.get("label", "")))
    sarcasm_score = float(sar.get("score", 0.0))
    if sarcasm_score < SARCASM_FLOOR:
        sarcasm_flag = False
    if _SARCASM.search(clipped) or "/s" in clipped.lower():
        sarcasm_flag = True
        sarcasm_score = max(sarcasm_score, 0.9)
    return emotion_label, emotion_score, sarcasm_flag, sarcasm_score  # type: ignore[return-value]


def _heuristic_tags(text: str) -> tuple[Emotion, float, bool, float]:
    sarcasm = bool(_SARCASM.search(text)) or "/s" in text.lower()
    sarcasm_score = 0.9 if sarcasm else 0.1
    checks: list[tuple[Emotion, re.Pattern[str]]] = [
        ("anger", _ANGER),
        ("sadness", _SAD),
        ("fear", _FEAR),
        ("love", _LOVE),
        ("surprise", _SURPRISE),
        ("joy", _JOY),
    ]
    for emotion, pattern in checks:
        if pattern.search(text):
            return emotion, 0.7, sarcasm, sarcasm_score
    return "neutral", 0.55, sarcasm, sarcasm_score


def derive_tone(
    emotion: Emotion,
    sarcasm: bool,
    score: int,
    replies_to_host: bool,
) -> Tone:
    if sarcasm:
        return "heated" if emotion == "anger" else "witty"
    if emotion == "anger":
        return "heated"
    if emotion == "sadness":
        return "somber"
    if emotion in {"love", "joy"} and replies_to_host:
        return "supportive"
    return "calm"


def emotion_to_expression(emotion: Emotion, sarcasm: bool) -> Expression:
    if sarcasm:
        return "smirk"
    mapping: dict[Emotion, Expression] = {
        "joy": "smile",
        "sadness": "frown",
        "anger": "scowl",
        "fear": "wide-eyes",
        "surprise": "wide-eyes",
        "love": "hearts",
        "neutral": "rest",
    }
    return mapping[emotion]


def pick_scene(subreddit: str, title: str, selftext: str) -> SceneName:
    blob = f"{subreddit} {title} {selftext}".lower()
    scores: dict[SceneName, int] = {name: 0 for name in SCENE_KEYWORDS}
    for scene, keywords in SCENE_KEYWORDS.items():
        for word in keywords:
            if word in blob:
                scores[scene] += 1
    best = max(scores, key=lambda name: scores[name])
    if scores[best] == 0:
        return "coffee_shop"
    return best


def tag_text(
    text: str,
    *,
    score: int = 0,
    replies_to_host: bool = False,
) -> dict:
    modeled = _model_tags(text)
    if modeled is None:
        emotion, emotion_conf, sarcasm, sarcasm_conf = _heuristic_tags(text)
        backend = "heuristic"
    else:
        emotion, emotion_conf, sarcasm, sarcasm_conf = modeled
        backend = "distilbert"
    tone = derive_tone(emotion, sarcasm, score, replies_to_host)
    expression = emotion_to_expression(emotion, sarcasm)
    return {
        "emotion": emotion,
        "emotion_confidence": emotion_conf,
        "sarcasm": sarcasm,
        "sarcasm_confidence": sarcasm_conf,
        "tone": tone,
        "expression": expression,
        "backend": backend,
    }
