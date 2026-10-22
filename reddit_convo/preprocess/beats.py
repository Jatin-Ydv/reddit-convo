"""Split a turn into sentence-sized beats and read the mood of each one.

Each beat drives the face, gesture, emote and voice delivery for that stretch of
speech, so a comment like "lol no. That's the worst clone. Try Azzaro instead!"
reads as amused -> annoyed -> confident instead of one flat emotion.
"""

from __future__ import annotations

import re

from reddit_convo.models import Beat

MOODS = (
    "neutral", "happy", "excited", "laugh", "love", "grateful", "confident",
    "angry", "annoyed", "disgusted", "sarcastic", "sad", "sympathetic",
    "worried", "surprised", "curious", "thinking", "agree", "disagree",
    "proud", "encouraging",
)


def _rx(words: str, emoji: str = "") -> re.Pattern:
    body = rf"(?<![\w'])(?:{words})(?![\w'])"
    return re.compile(body + (f"|{emoji}" if emoji else ""), re.I)


# (mood, pattern, weight). Strongest total score wins per sentence.
CUES: list[tuple[str, re.Pattern, float]] = [
    ("laugh", _rx(r"lol+|lmao+|lmfao|rofl|ha(ha)+h?|he(he)+|hilarious|i'?m dying|i'?m dead|cracked me up|funniest", "😂|🤣|💀|😆"), 3.0),
    ("angry", _rx(r"hate|stupid|idiot\w*|moron\w*|dumb|bs|bullshit|wtf|trash|garbage|scam\w*|rip[- ]?off|ridiculous|pathetic|furious|pissed|shut up|disgrace|unacceptable|how dare|f+u+c+k\w*|screw (this|that|you)|worst|wasted|waste of (money|time)|awful|terrible|horrible|useless|shit\w*|shi+|crap\w*|never again|regret buying", "😡|🤬|😠"), 3.0),
    ("disgusted", _rx(r"gross|disgusting|nasty|smells? like (piss|sweat|bug spray|chemicals|cheap)|stinks?|vomit|cringe|ew+|yuck|rotten|burns? my (skin|nose)|headache|bathroom freshener|toilet", "🤢|🤮"), 2.8),
    ("annoyed", _rx(r"ugh+|smh|meh|seriously|annoying|again\?|stop (it|this|spamming)|spamming|over(rated|hyped|priced)|not worth|waste of|why do people|idk why|enough|tired of|sick of|whatever|let (people|them|him|her|me) (be|enjoy|live|have)|not everything|nobody asked|who asked|get over it|chill", "🙄|😒|,\\s*wow\\.?\\s*$"), 2.8),
    ("disagree", _rx(r"no,|nope|nah|wrong|disagree|not really|not true|that'?s not|doesn'?t (work|last|smell)|but honestly|i doubt|not at all|hard pass|i wouldn'?t", r"^\s*actually\b"), 1.8),
    ("encouraging", _rx(r"you got this|you can do (it|this)|keep (going|it up|at it)|don'?t give up|rooting for you|proud of you|(that|it) (still )?counts|small wins?|one (day|step) at a time|it'?s (ok|okay) to|be proud|progress is progress|good for you|well done|nice work|go you"), 2.8),
    ("proud", _rx(r"proud of (myself|me)|i did it|nailed it|finally (did|made|got)|feel like a (person|human|new person)|feel (so )?(much )?better|best i'?ve (felt|been)|personal best|\bpr\b"), 2.8),
    ("agree", _rx(r"agree[ds]?|exactly|same( here)?|this\.|this!|so true|100%|facts|you'?re right|true that|yes+|yep|yup|absolutely|definitely this|\+1"), 2.0),
    ("grateful", _rx(r"thanks?( you| so much| a lot)?|thx|ty|appreciate[ds]?|grateful|helpful|that helps", "🙏"), 2.4),
    ("love", _rx(r"love (this|it|that|you|them|how)|in love|wholesome|adorable|so sweet|bless\w*|my heart|obsessed|favou?rite", "❤️|❤|♥|🥰|😍|💕"), 2.6),
    ("sympathetic", _rx(r"i'?m (so )?sorry|that sucks|hugs?|sending (love|hugs)|i feel (you|for you)|you'?re not alone|hang in there|it'?ll get better|take care", "🫂"), 2.8),
    ("sad", _rx(r"sad|miss(ing)? (you|him|her|them|it)|lonely|alone|cry\w*|tears|hurts?|heart ?broken|broke up|breakup|depress\w*|lost (my|him|her)|grief|it'?s hard|painful|miserable|gave up|regret|barely (get|got|make|made|sleep)|couldn'?t (get|sleep|stop|even)|struggl\w*|rough (time|day|week|year)|bad days?|hard time|exhausted|burn(ed|t)? out", "😢|😭|💔|😞"), 2.4),
    ("worried", _rx(r"scared|afraid|anxious|anxiety|nervous|worr(y|ied|ies)|panic\w*|what if|terrified|freaking out|stress(ed|ful)?|concerned|risky|careful", "😰|😟|😬"), 2.8),
    ("surprised", _rx(r"wow+|whoa+|omg|oh my god|no way|wait,? what|holy|unbelievable|shocked|insane|mind ?blown|can'?t believe|really\?|seriously\?|plot twist", "😮|😱|🤯|😲"), 2.6),
    ("excited", _rx(r"can'?t wait|so excited|let'?s go+|hype|stoked|pumped|finally|yay+|woo+|beast( ?mode)?|banger|fire|insane performance|blew me away", "🔥|🎉|🙌"), 2.6),
    ("happy", _rx(r"glad|happy|awesome|amazing|great|nice|cool|perfect|beautiful|excellent|fantastic|lovely|congrats\w*|good luck|enjoy|fun|good"), 1.4),
    ("confident", _rx(r"recommend\w*|go (for|with)|try (the|it|this|out)?|get (the|a|some|something|yourself)|just (buy|get|go)|buy (it|the|this|one)|best (one|is|option|options|bet|value|clone)|you should|you need|must[- ]have|trust me|hands down|easily|worth (it|the|every)|i'?d (go|say|suggest|pick)|my (pick|go-to)|stick with|skip it|not even close|by far"), 1.6),
    ("thinking", _rx(r"i think|i guess|maybe|probably|not sure|idk|i don'?t know|hmm+|depends|might|perhaps|it seems|kind of|sort of|i wonder"), 1.1),
    ("curious", _rx(r"anyone|which (one|is)|what'?s|how (do|does|long|much)|has anyone|does it|is it|any (tips|recs|suggestions)|thoughts\?|curious"), 1.0),
]

SARCASM = re.compile(
    r"/s\b|\byeah,? right\b|\boh,? (great|sure|wonderful|perfect)\b|\bsure,? jan\b|\bwhat a surprise\b"
    r"|\bthanks for nothing\b|\bslow clap\b|\bwow,? (just )?wow\b|\btotally\b.*\bnot\b|\bbecause that worked\b",
    re.I,
)
NEGATOR = re.compile(r"(?<![\w'])(not|never|no|isn'?t|wasn'?t|don'?t|doesn'?t|didn'?t|hardly|barely)\s+(\w+\s+)?$", re.I)
INTENSIFIER = re.compile(r"(?<![\w'])(so|very|really|super|extremely|absolutely|totally|literally|insanely|incredibly|way too|hella|crazy)(?![\w'])", re.I)
POSITIVE = {"happy", "excited", "love", "grateful", "confident", "agree", "laugh", "proud", "encouraging"}
CONTRAST = re.compile(r"(?<![\w'])(but|though|however|yet|still)(?![\w'])", re.I)

# mood -> (low-intensity expression, high-intensity expression)
FACE: dict[str, tuple[str, str]] = {
    "neutral": ("neutral", "neutral"),
    "happy": ("happy", "excited"),
    "excited": ("excited", "laugh"),
    "laugh": ("laugh", "laugh"),
    "love": ("happy", "love"),
    "grateful": ("happy", "proud"),
    "confident": ("confident", "confident"),
    "angry": ("annoyed", "furious"),
    "annoyed": ("annoyed", "eyeroll"),
    "disgusted": ("disgusted", "disgusted"),
    "sarcastic": ("smirk", "eyeroll"),
    "sad": ("sad", "cry"),
    "sympathetic": ("sympathetic", "sad"),
    "worried": ("nervous", "scared"),
    "surprised": ("surprised", "surprised"),
    "curious": ("curious", "curious"),
    "thinking": ("thinking", "thinking"),
    "agree": ("happy", "proud"),
    "disagree": ("skeptical", "annoyed"),
    "proud": ("proud", "proud"),
    "encouraging": ("happy", "proud"),
}
GESTURE: dict[str, tuple[str | None, str | None]] = {
    "neutral": (None, None),
    "happy": ("explain", "cheer"),
    "excited": ("fist_pump", "cheer"),
    "laugh": ("belly", "belly"),
    "love": ("heart", "heart"),
    "grateful": ("heart", "heart"),
    "confident": ("point_up", "point"),
    "angry": ("point", "fists"),
    "annoyed": ("crossed_talk", "facepalm"),
    "disgusted": ("stop", "stop"),
    "sarcastic": ("shrug", "shrug"),
    "sad": ("wring_talk", "slump"),
    "sympathetic": ("clasp", "heart"),
    "worried": ("wring", "defend"),
    "surprised": ("cheeks", "explain_big"),
    "curious": ("palms_up", "palms_up"),
    "thinking": ("chin", "chin"),
    "agree": ("explain", "point"),
    "disagree": ("stop", "crossed_talk"),
    "proud": ("hips", "fist_pump"),
    "encouraging": ("fist_pump", "fist_pump"),
}
EMOTE: dict[str, tuple[str | None, str | None]] = {
    "angry": (None, "anger"),
    "annoyed": (None, "dots"),
    "disgusted": ("gloom", "gloom"),
    "sad": (None, "gloom"),
    "worried": ("sweat", "sweat"),
    "surprised": ("exclaim", "shock"),
    "excited": ("sparkle", "sparkle"),
    "love": (None, "hearts"),
    "grateful": (None, "sparkle"),
    "curious": ("question", "question"),
    "thinking": (None, "dots"),
    "laugh": (None, "sparkle"),
    "proud": ("sparkle", "sparkle"),
    "encouraging": (None, "sparkle"),
}

_SENT = re.compile(r"[^.!?…\n]+(?:[.!?…]+|\n|$)")


def split_sentences(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of sentences; very short fragments merge forward."""
    spans = [(m.start(), m.end()) for m in _SENT.finditer(text) if m.group().strip()]
    merged: list[tuple[int, int]] = []
    carry: int | None = None
    for s, e in spans:
        if carry is not None:
            s = carry
            carry = None
        if len(text[s:e].strip()) < 14 and (s, e) != spans[-1]:
            carry = s
            continue
        merged.append((s, e))
    if carry is not None:
        if merged:
            merged[-1] = (merged[-1][0], len(text))
        else:
            merged.append((carry, len(text)))
    return merged or [(0, len(text))]


def read_sentence(sentence: str, *, sarcastic: bool = False, model_emotion: str | None = None) -> tuple[str, float]:
    """Return (mood, intensity 0..1) for one sentence."""
    text = sentence.strip().replace("’", "'").replace("‘", "'")
    scores: dict[str, float] = {}
    pivots = [m.start() for m in CONTRAST.finditer(text) if m.start() > 0]
    pivot = pivots[-1] if pivots else None
    for mood, pattern, weight in CUES:
        for m in pattern.finditer(text):
            target = mood
            if mood in POSITIVE and NEGATOR.search(text[: m.start()][-24:]):
                target = "disagree"
            w = weight
            if pivot is not None:
                w *= 1.5 if m.start() > pivot else 0.7
            scores[target] = scores.get(target, 0.0) + w
    if SARCASM.search(text):
        scores["sarcastic"] = scores.get("sarcastic", 0.0) + 4.0
    elif sarcastic:
        scores["sarcastic"] = scores.get("sarcastic", 0.0) + 2.0
    if text.endswith("?"):
        for mood in POSITIVE & scores.keys():
            scores[mood] *= 0.4
        scores["curious"] = scores.get("curious", 0.0) + 1.2
        if scores.get("annoyed") or scores.get("angry"):
            scores["annoyed"] = scores.get("annoyed", 0.0) + 1.0
    if model_emotion and model_emotion != "neutral":
        mapped = {"joy": "happy", "sadness": "sad", "anger": "angry", "fear": "worried",
                  "surprise": "surprised", "love": "love"}.get(model_emotion)
        if mapped:
            scores[mapped] = scores.get(mapped, 0.0) + 1.5

    bangs = min(text.count("!"), 3)
    if bangs and not scores:
        return "emphatic", round(min(1.0, 0.45 + 0.13 * bangs), 2)
    if not scores:
        return "neutral", 0.3

    mood = max(scores, key=lambda k: scores[k])
    letters = [c for c in text if c.isalpha()]
    caps = sum(c.isupper() for c in letters) / max(len(letters), 1)
    shouty = sum(1 for w in re.findall(r"\b[A-Z]{3,}\b", text) if w not in {"EDP", "EDT", "EDC", "OP", "USA", "LOL", "LMAO", "TBH", "IMO", "IMHO"})
    intensity = 0.35 + 0.08 * scores[mood]
    intensity += 0.13 * bangs
    intensity += 0.1 * min(len(INTENSIFIER.findall(text)), 2)
    intensity += 0.25 if caps > 0.6 and len(letters) > 8 else 0.08 * min(shouty, 2)
    if mood in {"sad", "sympathetic", "thinking"}:
        intensity -= 0.08 * bangs
    return mood, round(max(0.15, min(1.0, intensity)), 2)


def build_beats(
    text: str,
    *,
    sarcastic: bool = False,
    reply_mood: str | None = None,
    model_tagger=None,
) -> list[Beat]:
    """Beats for a turn's spoken text. ``reply_mood`` biases the opener when the
    turn obviously reacts to its parent (e.g. ``disagree`` -> skeptical start)."""
    beats: list[Beat] = []
    for n, (s, e) in enumerate(split_sentences(text)):
        sentence = text[s:e]
        model_emotion = model_tagger(sentence) if model_tagger else None
        mood, intensity = read_sentence(sentence, sarcastic=sarcastic, model_emotion=model_emotion)
        if n == 0 and mood == "neutral" and reply_mood:
            mood, intensity = reply_mood, 0.4
        beats.append(make_beat(sentence, s, e, mood, intensity))
    _resolve_emphatic(beats)
    _smooth(beats)
    return beats


def _resolve_emphatic(beats: list[Beat]) -> None:
    """A bare '!' sentence borrows the feeling of its neighbours (prev, then next)."""
    flat = {"neutral", "curious", "thinking", "emphatic"}
    for n, b in enumerate(beats):
        if b.mood != "emphatic":
            continue
        near = [beats[k] for k in (n - 1, n + 1) if 0 <= k < len(beats) and beats[k].mood not in flat]
        mood = near[0].mood if near else "excited"
        intensity = max([b.intensity] + [x.intensity for x in near[:1]])
        beats[n] = make_beat(b.text, b.char_start, b.char_end, mood, intensity)


def make_beat(text: str, start: int, end: int, mood: str, intensity: float) -> Beat:
    hi = intensity >= 0.7
    face = FACE.get(mood, ("neutral", "neutral"))[hi]
    pose = GESTURE.get(mood, (None, None))[hi]
    emote = EMOTE.get(mood, (None, None))[hi]
    return Beat(
        text=text.strip(), char_start=start, char_end=end, mood=mood, intensity=intensity,
        expression=face, emote=emote, pose=pose,
    )


def _smooth(beats: list[Beat]) -> None:
    """A neutral sentence sandwiched in an emotional run keeps the feeling, softer."""
    for n, b in enumerate(beats):
        if b.mood != "neutral" or n == 0:
            continue
        prev = beats[n - 1]
        if prev.mood not in {"neutral", "curious", "thinking"} and prev.intensity >= 0.55 and prev.emote is not None:
            soft = make_beat(b.text, b.char_start, b.char_end, prev.mood, round(prev.intensity * 0.6, 2))
            soft.emote = None
            beats[n] = soft


def dominant_mood(beats: list[Beat]) -> tuple[str, float]:
    best, score = "neutral", 0.0
    for b in beats:
        w = (b.char_end - b.char_start) * b.intensity * (0.3 if b.mood == "neutral" else 1.0)
        if w > score:
            best, score = b.mood, w
    top = max((b.intensity for b in beats if b.mood == best), default=0.3)
    return best, top


def reply_stance(text: str) -> str | None:
    """How the opening words relate to the parent comment."""
    head = text.strip()[:40].lower()
    if re.match(r"^(no+\b|nope|nah|wrong|not really|disagree|actually|eh+\b|hard disagree|idk about)", head):
        return "disagree"
    if re.match(r"^(yes+|yep|yup|agreed?|exactly|same|this\b|so true|100%|facts|\+1|true)", head):
        return "agree"
    if re.match(r"^(thanks?|thank you|ty\b|thx)", head):
        return "grateful"
    if re.match(r"^(lol|lmao|haha)", head):
        return "laugh"
    return None
