# Why use reddit-convo?

Most Reddit tools scrape text or dump screenshots. This one turns a thread into something you can **watch**: a short animated conversation where the cast, replies, and tone are obvious without reading a wall of comments.

## What you get

**A video, not a transcript.** Share an MP4 in Discord, Twitter/X, YouTube Shorts, or a slide deck. People get the vibe of the thread in under a few minutes.

**Who said what to whom.** Sticky name pills, “replying to” headers, parent quotes, and depth markers mean you never lose the reply chain the way you do in a flat screenshot.

**The whole room stays in frame.** Commenters don’t vanish when someone else speaks. The active speaker and listener step forward; everyone else watches from the back of the scene.

**Personality without a design team.** Stick figures with distinct looks, poses, and emotes (sarcasm smirk, anger vein, sweat drop, heart-eyes) make heated takes and wholesome replies readable at a glance.

**Zero API keys for Reddit.** Paste a link (including share URLs). Fetching uses public endpoints and archive/RSS fallbacks when Reddit blocks automated JSON.

**Runs on your machine.** TTS (Piper / espeak) and rendering (Pillow + ffmpeg) are local. No paid video API, no uploading the thread to a third-party “AI video” service.

## When it’s a good fit

- You want a **shareable clip** of a funny, spicy, or wholesome thread
- You’re making **content** (reaction videos, roundups, podcast B-roll) and need a quick visual of the discussion
- You’re demoing **product / community feedback** and a video lands better than a spreadsheet of comments
- You care about **reply structure**, not just top-level takes

## When to skip it

- You need a **full archive** or legal-grade record of a thread → use Reddit’s own export / official API
- You want photoreal characters or Hollywood motion graphics → this is deliberately stick-figure and fast
- The post is **image/video-only** with almost no text discussion → there’s little dialogue to animate

## Compared to alternatives

| Approach | Gap |
|----------|-----|
| Screenshot the thread | Hard to follow nested replies; ugly on mobile |
| Read-aloud TTS of one comment | No cast, no reply graph, no scene |
| Manual After Effects / CapCut edit | Looks great, takes hours |
| Hosted “AI Reddit video” sites | Uploads your content, paywalls, less control |

**reddit-convo** is the middle path: one CLI command, local render, conversation-shaped output.

## Bottom line

Use it when you want people to *see* a Reddit conversation — who’s talking, who’s roasting whom, and how the room reacts — without asking them to scroll.
