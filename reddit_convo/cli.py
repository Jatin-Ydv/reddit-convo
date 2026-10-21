from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import typer

from reddit_convo.pipeline import run


def main(
    url: Optional[str] = typer.Argument(None, help="Reddit post URL or id"),
    max_comments: int = typer.Option(
        8,
        help="Max comments after the host. 0 = every fetched comment within --max-depth.",
    ),
    max_depth: int = typer.Option(
        2,
        help="How deep to walk. 2 = comment + replies, 3 = also replies-to-replies.",
    ),
    out: Optional[Path] = typer.Option(None, "--out", help="Output MP4 path."),
    script_only: bool = typer.Option(False, "--script-only", help="Stop after tagged script JSON."),
    fixture: Optional[Path] = typer.Option(None, "--fixture", help="Local thread JSON instead of fetching."),
    cache_dir: Path = typer.Option(Path("cache"), "--cache-dir", help="Where to write intermediates."),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Turn a Reddit post into an animated conversation video."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    if url is None and fixture is None:
        raise typer.BadParameter("Provide a Reddit URL or --fixture.")
    try:
        result = run(
            url=url,
            fixture=fixture,
            out=out,
            cache_dir=cache_dir,
            max_comments=max_comments,
            max_depth=max_depth,
            script_only=script_only,
        )
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("reddit_convo").error("%s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(str(result))


def app() -> None:
    typer.run(main)


if __name__ == "__main__":
    app()
