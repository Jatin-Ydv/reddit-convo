from __future__ import annotations

from reddit_convo.models import DialogueTurn, Script, Thread
from reddit_convo.preprocess.graph import assign_speakers, select_comments
from reddit_convo.preprocess.beats import build_beats, dominant_mood, reply_stance
from reddit_convo.preprocess.nlp import (
    MOOD_TO_EMOTION,
    derive_tone,
    emotion_to_expression,
    pick_scene,
    sentence_emotion,
    tag_text,
)
from reddit_convo.render.personality import refine_personalities
from reddit_convo.textutil import sanitize_spoken, truncate_body


def build_script(
    thread: Thread,
    max_comments: int = 8,
    max_depth: int = 2,
) -> Script:
    comments = select_comments(thread, max_comments=max_comments, max_depth=max_depth)
    graph = assign_speakers(thread, comments)
    scene = pick_scene(thread.subreddit, thread.title, thread.selftext)
    author_to_speaker = graph.author_to_speaker
    speaker_by_id = {s.id: s for s in graph.speakers}

    host_body = thread.title.strip()
    if thread.selftext.strip():
        sep = " " if host_body.endswith((".", "!", "?")) else ". "
        host_body = f"{host_body}{sep}{truncate_body(thread.selftext.strip())}"
    host_tags = tag_text(host_body, score=thread.score, replies_to_host=False)

    turns: list[DialogueTurn] = [
        DialogueTurn(
            index=0,
            speaker_id="host",
            author=thread.author,
            comment_id=thread.id,
            text=host_body,
            spoken_text=sanitize_spoken(host_body),
            reply_to=None,
            emotion=host_tags["emotion"],
            sarcasm=host_tags["sarcasm"],
            sarcasm_confidence=host_tags["sarcasm_confidence"],
            emotion_confidence=host_tags["emotion_confidence"],
            tone=host_tags["tone"],
            expression=host_tags["expression"],
            score=thread.score,
        )
    ]

    backends = {host_tags["backend"]}
    by_id = {c.id: c for c in comments}
    for comment in comments:
        speaker_id = author_to_speaker[comment.author]
        parent = None if comment.depth == 0 else by_id.get(comment.parent_id)
        reply_to = author_to_speaker.get(parent.author if parent else thread.author, "host")
        reply_text = parent.body if parent else thread.title
        tags = tag_text(
            comment.body,
            score=comment.score,
            replies_to_host=reply_to == "host",
        )
        backends.add(tags["backend"])
        reply_author = speaker_by_id[reply_to].author if reply_to in speaker_by_id else thread.author
        turns.append(
            DialogueTurn(
                index=len(turns),
                speaker_id=speaker_id,
                author=comment.author,
                comment_id=comment.id,
                text=comment.body,
                spoken_text=sanitize_spoken(comment.body),
                reply_to=reply_to,
                reply_to_author=reply_author,
                reply_to_text=reply_text,
                depth=comment.depth,
                emotion=tags["emotion"],
                sarcasm=tags["sarcasm"],
                sarcasm_confidence=tags["sarcasm_confidence"],
                emotion_confidence=tags["emotion_confidence"],
                tone=tags["tone"],
                expression=tags["expression"],
                score=comment.score,
            )
        )

    tagger = sentence_emotion if "distilbert" in backends else None
    for turn in turns:
        attach_beats(turn, tagger)

    refine_personalities(graph.speakers, turns)
    return Script(
        post_id=thread.id,
        subreddit=thread.subreddit,
        title=thread.title,
        scene=scene,
        speakers=graph.speakers,
        edges=graph.edges,
        turns=turns,
        nlp_backend="+".join(sorted(backends)),
    )


def attach_beats(turn: DialogueTurn, tagger=None) -> None:
    """Sentence-level acting beats; also sharpens a turn-level 'neutral' read."""
    text = turn.spoken_text or turn.text
    stance = reply_stance(text) if turn.reply_to else None
    turn.beats = build_beats(text, sarcastic=turn.sarcasm, reply_mood=stance, model_tagger=tagger)
    mood, strength = dominant_mood(turn.beats)
    if mood == "sarcastic" and not turn.sarcasm:
        turn.sarcasm = True
        turn.sarcasm_confidence = max(turn.sarcasm_confidence, 0.7)
    trust_beats = tagger is None or turn.emotion == "neutral"
    if trust_beats and strength >= 0.45 and mood != "neutral":
        turn.emotion = MOOD_TO_EMOTION.get(mood, "neutral")
        turn.emotion_confidence = max(turn.emotion_confidence, strength)
    turn.tone = derive_tone(turn.emotion, turn.sarcasm, turn.score, turn.reply_to == "host")
    turn.expression = emotion_to_expression(turn.emotion, turn.sarcasm)
