from __future__ import annotations

import html as htmlmod
import json
import logging
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

import httpx

from reddit_convo.models import CommentNode, Thread
from reddit_convo.textutil import looks_unusable, truncate_body

log = logging.getLogger(__name__)

USER_AGENTS = [
    "reddit-convo/0.1 (local slideshow generator)",
    "Mozilla/5.0 (compatible; reddit-convo/0.1; +https://localhost)",
]
BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/129.0.0.0 Safari/537.36"
)
_ATOM_NS = {"a": "http://www.w3.org/2005/Atom"}
_COMMENT_RE = re.compile(
    r"(?:old\.|www\.)?reddit\.com/r/(?P<sub>[\w]+)/comments/(?P<id>[a-z0-9]+)",
    re.IGNORECASE,
)
_SHARE_RE = re.compile(
    r"(?:old\.|www\.)?reddit\.com/r/(?P<sub>[\w]+)/s/(?P<code>[\w-]+)",
    re.IGNORECASE,
)
_SHORT_RE = re.compile(r"redd\.it/(?P<id>[a-z0-9]+)", re.IGNORECASE)
_BARE_RE = re.compile(r"^(?:t3_)?(?P<id>[a-z0-9]{5,8})$", re.IGNORECASE)


class RedditFetchError(RuntimeError):
    pass


def is_share_url(value: str) -> bool:
    return bool(_SHARE_RE.search(value.strip()))


def parse_reddit_url(value: str) -> tuple[str | None, str]:
    """Return (subreddit or None, post_id) from a URL or bare id."""
    text = value.strip()
    match = _COMMENT_RE.search(text)
    if match:
        return match.group("sub"), match.group("id")
    match = _SHORT_RE.search(text)
    if match:
        return None, match.group("id")
    match = _BARE_RE.match(text)
    if match:
        return None, match.group("id")
    if is_share_url(text):
        raise RedditFetchError(
            f"Share link is not a post id yet: {value}. "
            "fetch_thread() will resolve /s/ links."
        )
    raise RedditFetchError(f"Not a Reddit post URL: {value}")


def _json_urls(subreddit: str | None, post_id: str, limit: int = 100) -> list[str]:
    path = f"/comments/{post_id}.json"
    if subreddit:
        path = f"/r/{subreddit}/comments/{post_id}.json"
    query = f"raw_json=1&limit={max(25, limit)}"
    return [
        f"https://www.reddit.com{path}?{query}",
        f"https://old.reddit.com{path}?{query}",
    ]


def resolve_share_url(url: str, timeout: float = 20.0) -> str:
    """Follow a /r/sub/s/CODE share link to the /comments/ post URL."""
    last_error: Exception | None = None
    for user_agent in USER_AGENTS:
        headers = {"User-Agent": user_agent, "Accept": "text/html,application/json"}
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=False) as client:
            try:
                response = client.get(url)
                location = response.headers.get("location") or ""
                if location.startswith("/"):
                    location = str(response.url.join(location))
                if _COMMENT_RE.search(location):
                    return location
                if response.status_code in {401, 403}:
                    last_error = RedditFetchError(f"Reddit returned {response.status_code} resolving share link.")
                    continue
            except httpx.HTTPError as exc:
                last_error = exc
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
            try:
                response = client.get(url)
                final = str(response.url)
                if _COMMENT_RE.search(final):
                    return final
                match = _COMMENT_RE.search(response.text or "")
                if match:
                    return f"https://www.reddit.com/r/{match.group('sub')}/comments/{match.group('id')}/"
            except httpx.HTTPError as exc:
                last_error = exc
    raise RedditFetchError(
        "Could not resolve that Reddit share link to a post. "
        "Open it in a browser and paste the /comments/ URL instead. "
        f"({last_error})"
    )


def fetch_thread(url: str, timeout: float = 20.0, comment_limit: int = 100) -> Thread:
    if is_share_url(url):
        url = resolve_share_url(url, timeout=timeout)
    subreddit, post_id = parse_reddit_url(url)
    try:
        return _fetch_reddit_json(subreddit, post_id, url, timeout=timeout, comment_limit=comment_limit)
    except RedditFetchError as exc:
        log.info("Reddit JSON unavailable (%s); trying public archives", exc)
        return _fetch_archive(post_id, url, timeout=timeout, comment_limit=comment_limit)


def _fetch_reddit_json(
    subreddit: str | None,
    post_id: str,
    url: str,
    timeout: float,
    comment_limit: int = 100,
) -> Thread:
    last_error: Exception | None = None
    blocked = False
    for user_agent in USER_AGENTS:
        headers = {"User-Agent": user_agent, "Accept": "application/json"}
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
            for endpoint in _json_urls(subreddit, post_id, limit=comment_limit):
                try:
                    response = client.get(endpoint)
                    if response.status_code in {401, 403}:
                        blocked = True
                        last_error = RedditFetchError(
                            f"Reddit returned {response.status_code} for {endpoint}."
                        )
                        continue
                    if response.status_code == 429:
                        last_error = RedditFetchError("Reddit rate-limited the request (429).")
                        continue
                    if "text/html" in response.headers.get("content-type", "") and not response.text.lstrip().startswith("["):
                        blocked = True
                        last_error = RedditFetchError("Reddit returned an HTML block page instead of JSON.")
                        continue
                    response.raise_for_status()
                    payload = response.json()
                    return parse_listing(payload, requested_url=url)
                except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                    last_error = exc
    if blocked:
        raise RedditFetchError(
            "Reddit blocked unauthenticated JSON from this network (403)."
        )
    raise RedditFetchError(f"Could not fetch Reddit thread: {last_error}")


def _fetch_archive(
    post_id: str,
    requested_url: str,
    timeout: float,
    comment_limit: int = 100,
) -> Thread:
    headers = {"User-Agent": USER_AGENTS[0], "Accept": "application/json"}
    with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
        post = _archive_post(client, post_id)
        comments = _archive_comments(client, post_id, limit=comment_limit)
    if not post:
        raise RedditFetchError(
            "Reddit blocked this network and public archives had no copy of the post. "
            "Pass --fixture with saved thread JSON."
        )
    if not comments:
        comments = _rss_comments(post.get("subreddit") or "", post_id, timeout=timeout)
    thread = thread_from_flat(post, comments, requested_url=requested_url)
    if not thread.comments:
        log.warning("No comments found for %s; video will be host-only", post_id)
    log.info("Loaded r/%s via archive (%s comments)", thread.subreddit, len(thread.comments))
    return thread


def _archive_post(client: httpx.Client, post_id: str) -> dict | None:
    urls = [
        f"https://arctic-shift.photon-reddit.com/api/posts/ids?ids={post_id}",
        f"https://api.pullpush.io/reddit/search/submission/?ids={post_id}",
    ]
    for url in urls:
        try:
            response = client.get(url)
            if response.status_code != 200:
                continue
            payload = response.json()
            data = payload.get("data") if isinstance(payload, dict) else None
            if isinstance(data, list) and data:
                return data[0]
        except (httpx.HTTPError, json.JSONDecodeError, ValueError):
            continue
    return None


def _archive_comments(client: httpx.Client, post_id: str, limit: int = 100) -> list[dict]:
    size = min(100, max(25, limit if limit > 0 else 100))
    tree_limit = 9999 if limit <= 0 else max(size, min(limit, 9999))
    urls = [
        f"https://arctic-shift.photon-reddit.com/api/comments/tree?link_id=t3_{post_id}&limit={tree_limit}",
        f"https://arctic-shift.photon-reddit.com/api/comments/search?link_id={post_id}&limit={size}",
        f"https://api.pullpush.io/reddit/comment/search/?link_id={post_id}&size={size}",
    ]
    for url in urls:
        try:
            response = client.get(url)
            if response.status_code != 200:
                log.debug("Archive comments %s -> %s", url, response.status_code)
                continue
            payload = response.json()
            rows = _flatten_comment_payload(payload)
            if rows:
                return rows
        except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
            log.debug("Archive comments failed %s: %s", url, exc)
            continue
    return []


def _flatten_comment_payload(payload: object) -> list[dict]:
    data = payload.get("data") if isinstance(payload, dict) else payload
    rows: list[dict] = []

    def walk(node: object) -> None:
        if isinstance(node, list):
            for item in node:
                walk(item)
            return
        if not isinstance(node, dict):
            return
        if node.get("kind") == "more":
            return
        inner = node.get("data") if node.get("kind") in {"t1", "Listing"} else node
        if node.get("kind") == "Listing" and isinstance(inner, dict):
            walk(inner.get("children") or [])
            return
        if isinstance(inner, dict) and inner.get("body") and inner.get("id"):
            rows.append(inner)
            replies = inner.get("replies")
            if isinstance(replies, dict):
                walk(replies)
            elif isinstance(replies, list):
                walk(replies)
            return
        if isinstance(inner, dict):
            walk(inner.get("children") or inner.get("data") or [])

    walk(data)
    return rows


def _rss_comments(subreddit: str, post_id: str, timeout: float) -> list[dict]:
    if subreddit:
        url = f"https://www.reddit.com/r/{subreddit}/comments/{post_id}.rss"
    else:
        url = f"https://www.reddit.com/comments/{post_id}.rss"
    headers = {
        "User-Agent": BROWSER_UA,
        "Accept": "application/atom+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }
    try:
        with httpx.Client(headers=headers, timeout=timeout, follow_redirects=True) as client:
            response = client.get(url)
            response.raise_for_status()
            if "xml" not in response.headers.get("content-type", "") and not response.text.lstrip().startswith("<?xml"):
                return []
            comments = parse_comment_rss(response.content, post_id)
            if comments:
                log.info("Loaded %s comments from Reddit RSS", len(comments))
            return comments
    except (httpx.HTTPError, ET.ParseError, ValueError) as exc:
        log.debug("RSS comments failed: %s", exc)
        return []


def parse_comment_rss(xml_bytes: bytes, post_id: str) -> list[dict]:
    root = ET.fromstring(xml_bytes)
    entries = root.findall("a:entry", _ATOM_NS)
    comments: list[dict] = []
    for index, entry in enumerate(entries):
        href = ""
        for link in entry.findall("a:link", _ATOM_NS):
            if link.get("rel") in {None, "alternate"}:
                href = link.get("href") or ""
                break
        comment_id = _comment_id_from_permalink(href, post_id)
        if index == 0 and (not comment_id or comment_id == post_id):
            continue
        author = entry.findtext("a:author/a:name", default="", namespaces=_ATOM_NS) or "[deleted]"
        author = re.sub(r"^/u/", "", author).lstrip("/")
        if author.startswith("u/"):
            author = author[2:]
        content = entry.findtext("a:content", default="", namespaces=_ATOM_NS) or ""
        body = _html_to_text(content)
        if not comment_id:
            comment_id = f"rss{index}"
        if looks_unusable(author, body):
            continue
        comments.append(
            {
                "id": comment_id,
                "author": author,
                "body": body,
                "score": 1,
                "parent_id": f"t3_{post_id}",
            }
        )
    return comments


def _comment_id_from_permalink(href: str, post_id: str) -> str:
    parts = [p for p in urlparse(href).path.strip("/").split("/") if p]
    try:
        idx = parts.index("comments")
    except ValueError:
        return ""
    if len(parts) > idx + 3:
        cid = parts[idx + 3]
        if cid and cid != post_id:
            return cid
    return ""


class _HTMLText(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self._skip = False

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in {"script", "style"}:
            self._skip = True
        if tag in {"p", "br", "div", "li"}:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style"}:
            self._skip = False

    def handle_data(self, data: str) -> None:
        if not self._skip:
            self.parts.append(data)


def _html_to_text(raw: str) -> str:
    parser = _HTMLText()
    parser.feed(htmlmod.unescape(raw))
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


def thread_from_flat(post: dict, comments: list[dict], requested_url: str = "") -> Thread:
    permalink = post.get("permalink") or ""
    post_id = post.get("id") or ""
    return Thread(
        id=post_id,
        subreddit=post.get("subreddit") or "",
        title=post.get("title") or "",
        selftext=post.get("selftext") or "",
        author=post.get("author") or "[deleted]",
        score=int(post.get("score") or 0),
        permalink=permalink,
        url=f"https://www.reddit.com{permalink}" if permalink else requested_url,
        comments=_tree_from_flat(post_id, comments),
    )


def _tree_from_flat(post_id: str, comments: list[dict]) -> list[CommentNode]:
    by_parent: dict[str, list[dict]] = {}
    for row in comments:
        parent = _bare_id(str(row.get("parent_id") or post_id))
        by_parent.setdefault(parent, []).append(row)

    def build(parent_id: str, depth: int) -> list[CommentNode]:
        kids = sorted(by_parent.get(parent_id, []), key=lambda c: int(c.get("score") or 0), reverse=True)
        nodes: list[CommentNode] = []
        for row in kids:
            author = row.get("author") or "[deleted]"
            body = row.get("body") or ""
            if looks_unusable(author, body):
                continue
            cid = row.get("id") or ""
            nodes.append(
                CommentNode(
                    id=cid,
                    author=author,
                    body=truncate_body(body),
                    score=int(row.get("score") or 0),
                    parent_id=parent_id,
                    depth=depth,
                    replies=build(cid, depth + 1),
                )
            )
        return nodes

    return build(post_id, 0)


def load_thread(path: Path) -> Thread:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return parse_listing(data, requested_url=str(path))
    return Thread.model_validate(data)


def parse_listing(payload: object, requested_url: str = "") -> Thread:
    if not isinstance(payload, list) or len(payload) < 2:
        raise RedditFetchError("Unexpected Reddit JSON shape (need [post, comments]).")
    post_listing, comment_listing = payload[0], payload[1]
    children = post_listing.get("data", {}).get("children", [])
    if not children:
        raise RedditFetchError("Reddit JSON had no post.")
    post = children[0].get("data", {})
    comments = [
        node
        for child in comment_listing.get("data", {}).get("children", [])
        if (node := _parse_comment(child, depth=0)) is not None
    ]
    permalink = post.get("permalink") or ""
    return Thread(
        id=post.get("id") or "",
        subreddit=post.get("subreddit") or "",
        title=post.get("title") or "",
        selftext=post.get("selftext") or "",
        author=post.get("author") or "[deleted]",
        score=int(post.get("score") or 0),
        permalink=permalink,
        url=f"https://www.reddit.com{permalink}" if permalink else requested_url,
        comments=comments,
    )


def _parse_comment(child: dict, depth: int) -> CommentNode | None:
    if child.get("kind") != "t1":
        return None
    data = child.get("data") or {}
    author = data.get("author") or "[deleted]"
    body = data.get("body") or ""
    if looks_unusable(author, body):
        return None
    replies = _parse_replies(data, depth + 1)
    return CommentNode(
        id=data.get("id") or "",
        author=author,
        body=truncate_body(body),
        score=int(data.get("score") or 0),
        parent_id=_bare_id(data.get("parent_id") or ""),
        depth=depth,
        replies=replies,
    )


def _parse_replies(data: dict, depth: int) -> list[CommentNode]:
    replies = data.get("replies")
    if not replies or not isinstance(replies, dict):
        return []
    nodes: list[CommentNode] = []
    for child in replies.get("data", {}).get("children", []):
        node = _parse_comment(child, depth)
        if node is not None:
            nodes.append(node)
    return nodes


def _bare_id(reddit_fullname: str) -> str:
    if "_" in reddit_fullname:
        return reddit_fullname.split("_", 1)[1]
    return reddit_fullname
