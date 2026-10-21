from __future__ import annotations

from dataclasses import dataclass, replace

from reddit_convo.models import DialogueTurn, Speaker

TRAIT_ORDER = ["cynic", "kind", "spark", "worry", "cool", "bookish"]
INDEX_DEFAULTS = ["host", "kind", "cynic", "spark", "worry", "cool", "bookish"]


@dataclass(frozen=True)
class Personality:
    id: str
    label: str
    body: str
    hair: str
    hair_color: tuple[int, int, int]
    accessory: str
    skin: tuple[int, int, int]
    eye_color: tuple[int, int, int]
    blush: bool
    energy: float
    idle: str
    brow_rest: str
    rest_smile: bool


CATALOG: dict[str, Personality] = {
    "host": Personality(
        id="host",
        label="Host",
        body="round",
        hair="tidy",
        hair_color=(72, 48, 32),
        accessory="round_glasses",
        skin=(252, 224, 190),
        eye_color=(92, 64, 40),
        blush=False,
        energy=0.7,
        idle="bob",
        brow_rest="flat",
        rest_smile=True,
    ),
    "kind": Personality(
        id="kind",
        label="Kind",
        body="round",
        hair="mop",
        hair_color=(210, 150, 70),
        accessory="bow",
        skin=(255, 228, 200),
        eye_color=(70, 120, 90),
        blush=True,
        energy=0.65,
        idle="bob",
        brow_rest="soft",
        rest_smile=True,
    ),
    "cynic": Personality(
        id="cynic",
        label="Cynic",
        body="tall",
        hair="messy",
        hair_color=(70, 64, 72),
        accessory="beanie",
        skin=(238, 208, 178),
        eye_color=(90, 110, 130),
        blush=False,
        energy=0.45,
        idle="sway",
        brow_rest="raised_right",
        rest_smile=False,
    ),
    "spark": Personality(
        id="spark",
        label="Spark",
        body="athletic",
        hair="spiky",
        hair_color=(220, 90, 40),
        accessory="none",
        skin=(214, 170, 128),
        eye_color=(60, 90, 160),
        blush=False,
        energy=1.15,
        idle="bounce",
        brow_rest="thick",
        rest_smile=True,
    ),
    "worry": Personality(
        id="worry",
        label="Worry",
        body="short",
        hair="bun",
        hair_color=(48, 36, 30),
        accessory="round_glasses",
        skin=(250, 222, 188),
        eye_color=(80, 70, 110),
        blush=True,
        energy=0.55,
        idle="fidget",
        brow_rest="worried",
        rest_smile=False,
    ),
    "cool": Personality(
        id="cool",
        label="Cool",
        body="athletic",
        hair="buzz",
        hair_color=(28, 22, 20),
        accessory="cap",
        skin=(138, 92, 64),
        eye_color=(50, 36, 28),
        blush=False,
        energy=0.5,
        idle="sway",
        brow_rest="flat",
        rest_smile=False,
    ),
    "bookish": Personality(
        id="bookish",
        label="Bookish",
        body="tall",
        hair="ponytail",
        hair_color=(90, 50, 30),
        accessory="glasses",
        skin=(255, 230, 208),
        eye_color=(70, 90, 70),
        blush=False,
        energy=0.6,
        idle="bob",
        brow_rest="soft",
        rest_smile=True,
    ),
}


def get_personality(speaker: Speaker | str) -> Personality:
    key = speaker if isinstance(speaker, str) else speaker.personality
    return CATALOG.get(key, CATALOG["kind"])


def default_for_index(sprite_index: int) -> str:
    return INDEX_DEFAULTS[sprite_index % len(INDEX_DEFAULTS)]


def _preferred_trait(turns: list[DialogueTurn]) -> str:
    if any(t.sarcasm or t.tone == "witty" for t in turns):
        return "cynic"
    if any(t.tone == "supportive" or t.emotion in {"love", "joy"} for t in turns):
        return "kind"
    if any(t.tone == "heated" or t.emotion == "anger" for t in turns):
        return "spark"
    if any(t.emotion in {"fear", "sadness"} or t.tone == "somber" for t in turns):
        return "worry"
    if any(t.emotion == "surprise" for t in turns):
        return "cool"
    return "bookish"


def refine_personalities(speakers: list[Speaker], turns: list[DialogueTurn]) -> None:
    for speaker in speakers:
        if speaker.role == "host":
            speaker.personality = "host"
    guests = [s for s in speakers if s.role != "host"]
    scores: dict[str, dict[str, int]] = {}
    for speaker in guests:
        theirs = [t for t in turns if t.speaker_id == speaker.id]
        blob = " ".join(t.text.lower() for t in theirs)
        scores[speaker.id] = {
            "cynic": sum(3 for t in theirs if t.sarcasm) + sum(1 for t in theirs if t.tone == "witty"),
            "kind": sum(1 for t in theirs if t.tone == "supportive" or t.emotion in {"love", "joy"}),
            "spark": sum(1 for t in theirs if t.tone == "heated" or t.emotion == "anger"),
            "worry": sum(2 for t in theirs if t.emotion in {"fear", "sadness"} or t.tone == "somber"),
            "cool": sum(1 for t in theirs if t.emotion == "surprise"),
            "bookish": 1,
        }
        if any(w in blob for w in ("scared", "panic", "afraid", "anxious", "nervous", "worried")):
            scores[speaker.id]["worry"] += 6
        if any(w in blob for w in ("squat", "gym", "lift", "pr ", "bro")):
            scores[speaker.id]["spark"] += 4
        if "/s" in blob or "yeah right" in blob:
            scores[speaker.id]["cynic"] += 4
    used = {"host"}
    assigned: dict[str, str] = {}
    for trait in TRAIT_ORDER:
        best_id = None
        best_score = 0
        for speaker in guests:
            if speaker.id in assigned:
                continue
            score = scores[speaker.id].get(trait, 0)
            if score > best_score:
                best_score = score
                best_id = speaker.id
        if best_id and best_score > 0:
            assigned[best_id] = trait
            used.add(trait)
    for speaker in guests:
        if speaker.id not in assigned:
            unused = next((p for p in TRAIT_ORDER if p not in used), None)
            if unused:
                assigned[speaker.id] = unused
            else:
                seed = _seed(speaker.author)
                evidence = {p: v for p, v in scores[speaker.id].items() if p != "bookish" and v > 0}
                if evidence:
                    assigned[speaker.id] = max(evidence, key=lambda p: (evidence[p], (seed + TRAIT_ORDER.index(p)) % 7))
                else:
                    assigned[speaker.id] = TRAIT_ORDER[seed % len(TRAIT_ORDER)]
            used.add(assigned[speaker.id])
        speaker.personality = assigned[speaker.id]


HAIR_STYLES = ["tidy", "mop", "spiky", "messy", "bun", "buzz", "ponytail", "curly", "long", "bald"]
HAIR_COLORS = [
    (30, 24, 22), (72, 48, 32), (120, 72, 40), (210, 160, 80), (180, 70, 40),
    (150, 150, 158), (90, 60, 140), (40, 110, 160), (230, 120, 160), (245, 230, 200),
]
SKINS = [(255, 228, 200), (246, 210, 175), (226, 180, 140), (196, 140, 100), (150, 100, 70), (110, 72, 50)]
ACCESSORIES = ["none", "glasses", "round_glasses", "cap", "beanie", "bow", "headphones", "earring", "bandana"]


def _seed(text: str) -> int:
    value = 2166136261
    for ch in text.encode("utf-8"):
        value = ((value ^ ch) * 16777619) & 0xFFFFFFFF
    return value


def assign_looks(speakers: list[Speaker]) -> dict[str, Personality]:
    """The first speaker of each trait keeps the canonical look; repeats get a stable per-author variant."""
    looks: dict[str, Personality] = {}
    seen: set[str] = set()
    for speaker in speakers:
        base = get_personality(speaker)
        if base.id not in seen:
            seen.add(base.id)
            looks[speaker.id] = base
            continue
        s = _seed(speaker.author)
        looks[speaker.id] = replace(
            base,
            hair=HAIR_STYLES[s % len(HAIR_STYLES)],
            hair_color=HAIR_COLORS[(s >> 4) % len(HAIR_COLORS)],
            skin=SKINS[(s >> 8) % len(SKINS)],
            accessory=ACCESSORIES[(s >> 12) % len(ACCESSORIES)],
            body=("round", "tall", "short", "athletic")[(s >> 16) % 4],
        )
    return looks
