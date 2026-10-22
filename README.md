# reddit-convo

Turn a Reddit post into an animated conversation video.

> **TRY IT OUT, IT IS FUN** — this isn’t a solemn utility. Paste a spicy, wholesome, or chaotic thread, watch stick figures argue about it, and share the clip. Use it because it’s fun.

Paste a thread URL. The tool fetches the discussion, tags tone and emotion, assigns stick-figure personalities, synthesizes speech, and renders an MP4 where you can see **who is talking** and **who they are replying to**.

No Reddit API key required.

**Why this tool?** See [WHY.md](WHY.md) — when it shines, when to skip it, and how it compares to screenshots and hosted “AI Reddit video” sites.

## Features

- **Link in, video out** — works with normal post URLs and share links (`/r/sub/s/...`)
- **Thread-aware staging** — speaker + listener up front, the rest of the cast stays in the room
- **Clear reply context** — name pills, “replying to” header, parent quote, depth markers
- **Expressive stick figures** — personalities (host, kind, cynic, spark, worry, cool, bookish), poses, emotes
- **Sentence-level acting** — every sentence is read for its own mood (laughing, ranting, disgusted, encouraging, skeptical, asking…), so faces, gestures and the listener's reaction change mid-comment
- **Scene matching** — park, café, gym, office, debate stage, campfire, perfume boutique, living room
- **Expressive local TTS** — Piper voices with per-sentence pace, pitch, loudness, pauses and rising questions shaped by mood and personality; espeak-ng fallback
- **Optional NLP** — DistilBERT emotion/sarcasm tagging when you install the `nlp` extras

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/) on your `PATH`
- Optional but recommended: [espeak-ng](https://github.com/espeak-ng/espeak-ng) (used if Piper is missing)

## Install

```bash
git clone https://github.com/Jatin-Ydv/reddit-convo.git
cd reddit-convo

python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

pip install -e ".[tts,nlp]"
```

Minimal install (heuristic NLP + espeak TTS):

```bash
pip install -e .
```

Piper voices download on first use into `~/.cache/reddit-convo/`.

## Usage

```bash
# Activate the venv first
source .venv/bin/activate

# Any Reddit post URL (share links work too)
reddit-convo "https://www.reddit.com/r/someSub/comments/abc123/title/" --out out.mp4

# Full thread, deeper replies
reddit-convo "https://www.reddit.com/r/PerfumeIndia/s/AL4Mo3KFqg" \
  --max-comments 0 \
  --max-depth 5 \
  --out perfume.mp4

# Offline dry-run with the sample fixture
reddit-convo --fixture fixtures/sample_thread.json --out sample.mp4
```

You can also run the module form:

```bash
python -m reddit_convo.cli "https://www.reddit.com/r/..." --out out.mp4
```

### Options

| Flag | Default | Meaning |
|------|---------|---------|
| `--max-comments` | `8` | Max guest comments after the host. `0` = every fetched comment within depth |
| `--max-depth` | `2` | How deep to walk (`2` = top comments + replies, `3` = replies-to-replies, …) |
| `--out` | `cache/<post_id>/video.mp4` | Output MP4 path |
| `--script-only` | off | Stop after writing the tagged script JSON |
| `--fixture` | — | Local thread JSON instead of fetching |
| `--cache-dir` | `cache` | Where intermediates (audio, script, thread) are stored |
| `-v` / `--verbose` | off | Debug logging |

### Example flow

1. Fetch the post (Reddit JSON when available; archive/RSS fallbacks otherwise)
2. Build a dialogue script with emotion, sarcasm, speaker personalities, and per-sentence mood beats
3. Synthesize each beat with Piper (or espeak) using mood-specific delivery, and record its timing
4. Render stick-figure animation + speech bubbles into an H.264 MP4

Intermediates land in `cache/<post_id>/` so you can inspect `script.json`, WAVs, and logs.

## How it works

```
URL → fetch thread → select comments → NLP tags
     → assign speakers / personalities
     → Piper / espeak TTS
     → Pillow stick-figure renderer → ffmpeg MP4
```

Scenes are chosen from subreddit + title keywords (e.g. fragrance threads → boutique, breakups → living room).

## Development

```bash
source .venv/bin/activate
pip install -e ".[tts,nlp]"
pip install pytest
pytest -q
```

Regenerate scene/character previews:

```bash
python -m reddit_convo.render.art
```

## Notes

- Reply edges follow Reddit’s real parent/child links: a comment that replies to another commenter faces that person on screen, not always the OP. When Reddit blocks JSON, we recover parents from per-comment context RSS (slower, and rate limits may leave some replies flat).
- Reddit sometimes rate-limits or blocks datacenter IPs. The fetcher falls back to public archives and RSS when that happens.
- Long threads take a while: rendering is roughly a few seconds of wall time per second of video on a typical laptop CPU.
- Comment text is shown as written, including profanity from the original thread.
- First NLP/TTS run may download models (hundreds of MB).

## License

MIT — see [LICENSE](LICENSE).

## Author

**Jatin Yadav** ([@Jatin-Ydv](https://github.com/Jatin-Ydv))
