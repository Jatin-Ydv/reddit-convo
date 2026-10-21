from __future__ import annotations

import logging
from pathlib import Path

from reddit_convo.fetch.reddit import fetch_thread, load_thread
from reddit_convo.paths import DEFAULT_CACHE, post_cache
from reddit_convo.preprocess.script import build_script
from reddit_convo.render.art import ensure_assets
from reddit_convo.render.compose import render_video
from reddit_convo.voice.piper import synthesize_script

log = logging.getLogger(__name__)


def run(
    url: str | None = None,
    fixture: Path | None = None,
    out: Path | None = None,
    cache_dir: Path = DEFAULT_CACHE,
    max_comments: int = 8,
    max_depth: int = 2,
    script_only: bool = False,
) -> Path:
    if not url and not fixture:
        raise ValueError("Provide a Reddit URL or --fixture path.")
    ensure_assets()
    fetch_limit = 500 if max_comments <= 0 else max(100, max_comments)
    thread = load_thread(fixture) if fixture else fetch_thread(url or "", comment_limit=fetch_limit)
    work = post_cache(cache_dir, thread.id or "thread")
    (work / "thread.json").write_text(thread.model_dump_json(indent=2), encoding="utf-8")

    script = build_script(thread, max_comments=max_comments, max_depth=max_depth)
    script_path = work / "script.json"
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    log.info("Wrote script to %s (%s, scene=%s)", script_path, script.nlp_backend, script.scene)

    if script_only:
        return script_path

    audio_dir = work / "audio"
    synthesize_script(script, audio_dir)
    script_path.write_text(script.model_dump_json(indent=2), encoding="utf-8")
    full_audio = audio_dir / "full.wav"
    dest = out or (work / "convo.mp4")
    render_video(script, full_audio, dest)
    log.info("Wrote video to %s", dest)
    return dest
