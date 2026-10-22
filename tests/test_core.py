import os
from pathlib import Path

os.environ["REDDIT_CONVO_NLP"] = "heuristic"

from reddit_convo.fetch.reddit import (
    is_share_url,
    load_thread,
    parse_comment_rss,
    parse_context_rss,
    parse_reddit_url,
    thread_from_flat,
)
from reddit_convo.models import CommentNode, Thread
from reddit_convo.preprocess.graph import assign_speakers, select_comments
from reddit_convo.preprocess.nlp import _is_sarcastic_label, pick_scene, tag_text
from reddit_convo.preprocess.script import build_script
from reddit_convo.textutil import sanitize_spoken


FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "sample_thread.json"
LISTING = Path(__file__).resolve().parents[1] / "fixtures" / "sample_listing.json"


def test_parse_reddit_urls() -> None:
    assert parse_reddit_url("https://www.reddit.com/r/python/comments/abc123/hello/") == (
        "python",
        "abc123",
    )
    assert parse_reddit_url("https://old.reddit.com/r/AskReddit/comments/xyz987")[1] == "xyz987"
    assert parse_reddit_url("https://redd.it/abc123") == (None, "abc123")
    assert parse_reddit_url("abc123") == (None, "abc123")
    assert is_share_url("https://www.reddit.com/r/BreakUps/s/pD3wDnULXa") is True
    assert is_share_url("https://www.reddit.com/r/python/comments/abc123/hello/") is False


def test_sanitize_spoken_strips_markup_and_slash_s() -> None:
    text = "Check [this](https://example.com) out **now** /s"
    cleaned = sanitize_spoken(text)
    assert "example.com" not in cleaned
    assert "/s" not in cleaned.lower()
    assert "this" in cleaned
    assert "**" not in cleaned


def test_spoken_and_body_are_not_truncated() -> None:
    from reddit_convo.textutil import truncate_body

    long = ("This is a long Reddit comment about fragrance clones. " * 40).strip()
    assert len(long) > 400
    assert sanitize_spoken(long) == long
    assert truncate_body(long) == long
    assert "…" not in sanitize_spoken(long)


def test_sarcasm_labels_do_not_match_no_sarcasm() -> None:
    assert _is_sarcastic_label("No sarcasm") is False
    assert _is_sarcastic_label("Sarcasm") is True
    assert _is_sarcastic_label("LABEL_0") is False
    assert _is_sarcastic_label("not_sarcastic") is False


def test_scene_and_sarcasm_heuristics() -> None:
    assert pick_scene("DecidingToBeBetter", "I started walking", "") == "park"
    tags = tag_text("Oh great, another park montage. /s", score=10, replies_to_host=True)
    assert tags["sarcasm"] is True
    assert tags["expression"] == "smirk"
    assert tags["tone"] == "witty"


def test_parse_comment_rss_skips_post_entry() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <author><name>/u/op</name></author>
        <link href="https://www.reddit.com/r/test/comments/abc123/hello/"/>
        <content type="html">&lt;p&gt;Post body&lt;/p&gt;</content>
      </entry>
      <entry>
        <author><name>/u/guest</name></author>
        <link href="https://www.reddit.com/r/test/comments/abc123/hello/cmt99/"/>
        <content type="html">&lt;p&gt;Nice scent.&lt;/p&gt;</content>
      </entry>
    </feed>
    """
    comments = parse_comment_rss(xml, "abc123")
    assert len(comments) == 1
    assert comments[0]["id"] == "cmt99"
    assert comments[0]["author"] == "guest"
    assert "Nice scent" in comments[0]["body"]


def test_parse_context_rss_builds_parent_chain() -> None:
    xml = b"""<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom">
      <entry>
        <author><name>/u/op</name></author>
        <link href="https://www.reddit.com/r/test/comments/abc123/hello/"/>
        <content type="html">&lt;p&gt;Post&lt;/p&gt;</content>
      </entry>
      <entry>
        <author><name>/u/alice</name></author>
        <link href="https://www.reddit.com/r/test/comments/abc123/hello/cmt1/"/>
        <content type="html">&lt;p&gt;Top level&lt;/p&gt;</content>
      </entry>
      <entry>
        <author><name>/u/bob</name></author>
        <link href="https://www.reddit.com/r/test/comments/abc123/hello/cmt2/"/>
        <content type="html">&lt;p&gt;Replying to alice&lt;/p&gt;</content>
      </entry>
    </feed>
    """
    chain = parse_context_rss(xml, "abc123", "cmt2")
    assert [c["id"] for c in chain] == ["cmt1", "cmt2"]
    assert chain[0]["parent_id"] == "t3_abc123"
    assert chain[1]["parent_id"] == "t1_cmt1"


def test_script_wires_guest_to_guest_replies() -> None:
    thread = Thread(
        id="post",
        subreddit="test",
        title="Root post",
        author="op",
        comments=[
            CommentNode(
                id="a",
                author="alice",
                body="Top take",
                score=10,
                parent_id="post",
                depth=0,
                replies=[
                    CommentNode(
                        id="b",
                        author="bob",
                        body="Replying to alice",
                        score=5,
                        parent_id="a",
                        depth=1,
                    )
                ],
            )
        ],
    )
    script = build_script(thread, max_comments=0, max_depth=5)
    bob = next(t for t in script.turns if t.author == "bob")
    assert bob.reply_to_author == "alice"
    assert bob.reply_to != "host"
    assert bob.depth == 1
    assert bob.reply_to_text == "Top take"


def test_select_comments_walks_nested_depth() -> None:
    leaf = CommentNode(id="c", author="third", body="deeper", score=1, parent_id="b", depth=2)
    mid = CommentNode(id="b", author="second", body="reply", score=4, parent_id="a", depth=1, replies=[leaf])
    top = CommentNode(id="a", author="first", body="top", score=9, parent_id="post", depth=0, replies=[mid])
    thread = Thread(
        id="post",
        subreddit="test",
        title="Hello",
        author="op",
        comments=[top],
    )
    shallow = select_comments(thread, max_comments=10, max_depth=2)
    assert [c.id for c in shallow] == ["a", "b"]
    deep = select_comments(thread, max_comments=10, max_depth=3)
    assert [c.id for c in deep] == ["a", "b", "c"]
    all_comments = select_comments(thread, max_comments=0, max_depth=5)
    assert [c.id for c in all_comments] == ["a", "b", "c"]


def test_script_from_fixture() -> None:
    thread = Thread.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    comments = select_comments(thread, max_comments=8, max_depth=2)
    assert comments
    graph = assign_speakers(thread, comments)
    assert graph.speakers[0].role == "host"
    assert graph.speakers[0].author == "sunrise_again"
    script = build_script(thread)
    assert script.scene == "park"
    assert script.turns[0].speaker_id == "host"
    assert any(t.sarcasm for t in script.turns)
    authors = {t.author for t in script.turns}
    assert "cynical_owl" in authors
    assert len(script.turns) <= 9
    personalities = [s.personality for s in script.speakers]
    assert "host" in personalities
    assert len(set(personalities)) == len(personalities)
    cynic = next(s for s in script.speakers if s.author == "cynical_owl")
    assert cynic.personality == "cynic"
    worry = next(s for s in script.speakers if s.author == "anxious_tea")
    assert worry.personality == "worry"
    spark = next(s for s in script.speakers if s.author == "gym_rat_99")
    assert spark.personality == "spark"


def test_thread_from_flat_archive_comments() -> None:
    post = {
        "id": "abc123",
        "subreddit": "BreakUps",
        "title": "Remember this",
        "selftext": "Stay strong.",
        "author": "op_user",
        "score": 10,
        "permalink": "/r/BreakUps/comments/abc123/remember_this/",
    }
    comments = [
        {"id": "c1", "author": "kind_pal", "body": "You got this.", "score": 8, "parent_id": "t3_abc123"},
        {"id": "c2", "author": "op_user", "body": "Thank you.", "score": 3, "parent_id": "t1_c1"},
        {"id": "c3", "author": "cynic_pal", "body": "Sure. /s", "score": 2, "parent_id": "t3_abc123"},
    ]
    thread = thread_from_flat(post, comments)
    assert thread.author == "op_user"
    assert [c.author for c in thread.comments] == ["kind_pal", "cynic_pal"]
    assert thread.comments[0].replies[0].author == "op_user"
    script = build_script(thread)
    assert script.turns[0].author == "op_user"
    assert any(t.author == "kind_pal" for t in script.turns)


def test_load_native_reddit_listing() -> None:
    thread = load_thread(LISTING)
    assert thread.subreddit == "relationship_advice"
    assert thread.author == "office_lurker"
    assert any(c.author == "spicy_intern" for c in thread.comments)
    script = build_script(thread)
    assert script.scene in {"coffee_shop", "office"}
    assert script.turns[0].speaker_id == "host"


def test_turns_carry_reply_context() -> None:
    thread = Thread.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    script = build_script(thread, max_comments=0, max_depth=5)
    by_comment = {c.id: c for c in select_comments(thread, max_comments=0, max_depth=5)}
    for turn in script.turns[1:]:
        comment = by_comment[turn.comment_id]
        if comment.depth == 0:
            assert turn.reply_to_text == thread.title
        else:
            assert turn.reply_to_text == by_comment[comment.parent_id].body
        assert turn.depth == comment.depth


def test_new_scenes_are_picked() -> None:
    assert pick_scene("PerfumeIndia", "Best clone of Aventus?", "") == "boutique"
    assert pick_scene("BreakUps", "Day 30 after the breakup", "") == "living_room"


def test_speaker_colors_stay_distinct_for_big_threads() -> None:
    from reddit_convo.preprocess.graph import speaker_color

    colors = [speaker_color(i) for i in range(30)]
    assert len(set(colors)) == len(colors)


def test_sides_persist_across_back_and_forth() -> None:
    from reddit_convo.render.compose import _assign_sides

    first = _assign_sides("a", "host", {})
    second = _assign_sides("host", "a", first)
    assert second == first
    third = _assign_sides("b", "a", second)
    assert third["a"] == second["a"] and third["b"] != third["a"]


def test_bubble_pagination_fits_width() -> None:
    from reddit_convo.render.art import load_font
    from reddit_convo.render.compose import _paginate

    font = load_font(30, "round")
    pages = _paginate("word " * 120 + "x" * 80, font, 600)
    assert all(len(page) <= 3 for page in pages)
    assert all(font.getlength(line) <= 600 for page in pages for line in page)


def test_whole_cast_has_a_spot_in_the_room() -> None:
    from reddit_convo.render.compose import crowd_slots

    thread = Thread.model_validate_json(FIXTURE.read_text(encoding="utf-8"))
    script = build_script(thread, max_comments=0, max_depth=5)
    slots = crowd_slots(script.speakers)
    assert set(slots) == {s.id for s in script.speakers}
    spots = {(x, feet) for _, x, feet, _ in slots.values()}
    assert len(spots) == len(slots)


def test_beats_follow_mood_shifts_within_a_comment() -> None:
    from reddit_convo.preprocess.beats import build_beats

    text = "I had Pineapple Fume! It's the worst dupe I've ever tried! Is Italian Suede good?"
    moods = [b.mood for b in build_beats(text)]
    assert moods[0] == "angry"  # bare "!" opener borrows the rant that follows
    assert moods[1] == "angry"
    assert moods[-1] == "curious"  # a question about something "good" is not happiness


def test_beats_context_rules() -> None:
    from reddit_convo.preprocess.beats import build_beats, reply_stance

    assert build_beats("Let people be happy, wow.")[0].mood == "annoyed"
    assert build_beats("Then quit on day four and start again. That still counts.")[-1].mood == "encouraging"
    assert build_beats("This is not good at all honestly.")[0].mood == "disagree"
    assert build_beats("I actually feel like a person again.")[0].mood == "proud"
    assert reply_stance("Nah, that one fades in an hour") == "disagree"
    assert reply_stance("Exactly what I was going to say") == "agree"


def test_share_link_is_not_sarcasm() -> None:
    assert tag_text("https://www.reddit.com/r/DesiFragranceAddicts/s/g3FxbbvFm7")["sarcasm"] is False
    assert tag_text("Totally worth it /s")["sarcasm"] is True


def test_prosody_and_pitch_shaping() -> None:
    import numpy as np

    from reddit_convo.models import Beat
    from reddit_convo.voice.piper import SAMPLE_RATE, shape_audio
    from reddit_convo.voice.prosody import prosody_for

    excited = prosody_for(Beat(text="Let's go!", mood="excited", intensity=0.9), "spark")
    sad = prosody_for(Beat(text="I miss her.", mood="sad", intensity=0.8), "kind")
    question = prosody_for(Beat(text="Does it last?", mood="neutral", intensity=0.3))
    assert excited.length < 1 < sad.length
    assert excited.pitch > 1 > sad.pitch
    assert excited.gain > sad.gain
    assert question.rise > 0.05
    assert prosody_for(Beat(text="End.", mood="neutral"), last=True).pause_ms == 0

    tone = np.sin(2 * np.pi * 200 * np.arange(SAMPLE_RATE) / SAMPLE_RATE).astype(np.float32) * 0.3
    shifted = shape_audio(tone, excited)
    assert abs(shifted.size - SAMPLE_RATE / excited.pitch) < SAMPLE_RATE * 0.01
    assert np.abs(shifted).max() <= 1.0


def test_beat_spans_drive_bubble_timing() -> None:
    from reddit_convo.models import Beat, DialogueTurn
    from reddit_convo.render.compose import FPS, _char_to_frame, _frame_to_char, beat_spans

    text = "First sentence here. Second one is longer than the first."
    cut = text.index("Second")
    turn = DialogueTurn(
        index=1, speaker_id="g1", author="a", text=text, spoken_text=text,
        beats=[
            Beat(text=text[:cut], char_start=0, char_end=cut, start_ms=0, end_ms=1000),
            Beat(text=text[cut:], char_start=cut, char_end=len(text), start_ms=1500, end_ms=4000),
        ],
    )
    spans = beat_spans(turn, text, speech=FPS * 4)
    assert spans[1][0] == round(1.5 * FPS)
    assert _char_to_frame(spans, cut) == spans[1][0]
    assert _frame_to_char(spans, int(1.2 * FPS)) == cut  # pause holds the end of beat one
    assert _frame_to_char(spans, FPS * 4) == len(text)
