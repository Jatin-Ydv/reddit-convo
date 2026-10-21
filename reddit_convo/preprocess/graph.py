from __future__ import annotations

import colorsys

from reddit_convo.models import CommentNode, GraphEdge, Speaker, SpeakerGraph, Thread
from reddit_convo.render.personality import default_for_index
from reddit_convo.voice.voices import GUEST_VOICES, HOST_VOICE

SPEAKER_COLORS: list[tuple[int, int, int]] = [
    (46, 125, 90),
    (70, 130, 180),
    (196, 86, 86),
    (210, 150, 40),
    (142, 110, 210),
    (230, 120, 70),
    (70, 150, 155),
]


def speaker_color(index: int) -> tuple[int, int, int]:
    if index < len(SPEAKER_COLORS):
        return SPEAKER_COLORS[index]
    hue = (index * 0.618034) % 1.0
    sat = 0.55 + 0.15 * ((index * 7) % 3) / 2
    val = 0.72 + 0.12 * ((index * 5) % 2)
    r, g, b = colorsys.hsv_to_rgb(hue, sat, val)
    return int(r * 255), int(g * 255), int(b * 255)


def flatten_comments(nodes: list[CommentNode], max_depth: int) -> list[CommentNode]:
    """Depth-first flatten, keeping comments with depth < max_depth."""
    out: list[CommentNode] = []

    def walk(node: CommentNode) -> None:
        if node.depth >= max_depth:
            return
        out.append(node)
        for reply in node.replies:
            walk(reply)

    for node in nodes:
        walk(node)
    return out


def select_comments(
    thread: Thread,
    max_comments: int = 8,
    max_depth: int = 2,
) -> list[CommentNode]:
    """Walk the comment tree by score.

    Depth 2 is a top-level comment plus its replies. Depth 3 includes
    replies-to-replies. ``max_comments <= 0`` means take every node in range.
    """
    depth_limit = max(1, max_depth)
    unlimited = max_comments <= 0
    selected: list[CommentNode] = []

    def room() -> bool:
        return unlimited or len(selected) < max_comments

    def walk(node: CommentNode, depth: int, parent_id: str) -> None:
        if not room() or depth >= depth_limit:
            return
        selected.append(
            CommentNode(
                id=node.id,
                author=node.author,
                body=node.body,
                score=node.score,
                parent_id=parent_id,
                depth=depth,
                replies=[],
            )
        )
        for reply in sorted(node.replies, key=lambda r: r.score, reverse=True):
            walk(reply, depth + 1, node.id)
            if not room():
                return

    for comment in sorted(thread.comments, key=lambda c: c.score, reverse=True):
        walk(comment, 0, comment.parent_id or thread.id)
        if not room():
            break
    return selected


def assign_speakers(
    thread: Thread,
    comments: list[CommentNode],
) -> SpeakerGraph:
    host = Speaker(
        id="host",
        author=thread.author,
        role="host",
        voice=HOST_VOICE.name,
        color=SPEAKER_COLORS[0],
        sprite_index=0,
        personality="host",
    )
    speakers = [host]
    author_to_speaker = {thread.author: "host"}
    guest_count = 0

    for comment in comments:
        if comment.author in author_to_speaker:
            continue
        sprite_index = guest_count + 1
        speaker_id = f"guest_{guest_count}"
        voice = GUEST_VOICES[guest_count % len(GUEST_VOICES)]
        speakers.append(
            Speaker(
                id=speaker_id,
                author=comment.author,
                role="guest",
                voice=voice.name,
                color=speaker_color(sprite_index),
                sprite_index=sprite_index,
                personality=default_for_index(sprite_index),
            )
        )
        author_to_speaker[comment.author] = speaker_id
        guest_count += 1

    id_to_author = {thread.id: thread.author}
    for comment in comments:
        id_to_author[comment.id] = comment.author

    edges: list[GraphEdge] = []
    for comment in comments:
        parent_author = id_to_author.get(comment.parent_id, thread.author)
        source = author_to_speaker.get(comment.author)
        target = author_to_speaker.get(parent_author, "host")
        if source:
            edges.append(
                GraphEdge(source=source, target=target, comment_id=comment.id)
            )

    return SpeakerGraph(
        speakers=speakers,
        edges=edges,
        author_to_speaker=author_to_speaker,
    )
