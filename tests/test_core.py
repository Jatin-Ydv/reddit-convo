import os
from pathlib import Path

os.environ["REDDIT_CONVO_NLP"] = "heuristic"

from reddit_convo.fetch.reddit import is_share_url, load_thread, parse_comment_rss, parse_reddit_url, thread_from_flat
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
