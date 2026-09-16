"""The ``social://`` family: a post read without an account, and what it links to.

Spec section 6.2, and every platform on its list is answered for. Two are readable
by anyone with no key and no login, so those two are first class: Bluesky through
``public.api.bsky.app`` and Hacker News through its Firebase export. The other three
each answer differently, and the difference is the point (product rule 2):

* **Reddit** takes a free app the user registers under their own account. When the
  two variables are absent the post is reported as ``UNVERIFIED (credentials
  missing)`` — its own spec section 15 state — and *nobody is asked*: a source no
  request was ever made for is not a source that failed.
* **Mastodon** is thousands of independent instances, so it is read best effort from
  whichever one hosts the status. An instance that requires a login says so and the
  status is ``BLOCKED``: it is there, and this reader may not have it.
* **X** has no read-only API at all. It is answered without a request, in its own
  words: paste the post's text and the links inside it are verified as usual.

What a post *is*, for this tool, is decided by section 6.2 too: a post cites by
linking, so the addresses inside it are the sources and the post's own words are the
claim standing on them. :func:`read_post` is therefore the whole of the platform
knowledge — one URL in, one :class:`Post` or one :class:`Unreadable` out — and
``ingest.from_post`` turns the result into an ordinary document. The provider below
uses the same reader for the other direction, where a bibliography *cites* a post.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import SplitResult, parse_qs, urlsplit

from proofpath.document import Reference
from proofpath.fetch import Outcome
from proofpath.polite import PoliteClient, ProviderError
from proofpath.providers import NO_TEXT, URL_PREFIX, EvidenceDoc, FetchesUrl, Scheme, is_social
from proofpath.providers.web import WebProvider
from proofpath.resolve import Candidate, ResolveResult, Retraction, State, find_url
from proofpath.secrets import (
    CREDENTIALS_MISSING,
    REDDIT_CLIENT_ID_ENV,
    REDDIT_CLIENT_SECRET_ENV,
    REDDIT_HINT,
    ApiKey,
    resolve_api_key,
)

BSKY_API = "https://public.api.bsky.app/xrpc"
HN_API = "https://hacker-news.firebaseio.com/v0"
REDDIT_TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
REDDIT_API = "https://oauth.reddit.com"

BLUESKY = "Bluesky"
HACKER_NEWS = "Hacker News"
REDDIT = "Reddit"
MASTODON = "Mastodon"
X = "X"

#: What an address on a platform this version does not read is answered with. Every
#: platform spec section 6.2 lists now has a reader, so nothing routed by
#: ``providers.is_social`` reaches this; it is what :func:`read_post` answers when it
#: is handed an address on some sixth platform — one honest answer for "nobody asked
#: anyone anything" instead of a borrowed fetch outcome.
NOT_READ_HERE = "UNVERIFIED (platform not read in this version)"
UNSUPPORTED_HINT = (
    "this version reads Bluesky, Hacker News, Reddit and Mastodon; "
    "paste the post's own text instead"
)

#: X is answered without a request. There is no read-only API to ask, and a post the
#: tool never asked anyone about must not be reported as one that could not be
#: reached (product rule 2) — the post is there, this reader may not have it, which
#: is what ``BLOCKED`` means in spec section 15.
X_HINT = "X cannot be read; paste the post text — the links inside it will be verified"

#: What a comment permalink is answered with when the thread came back without that
#: comment anywhere in it. ``UNAVAILABLE`` and not ``UNREACHABLE``: the provider
#: answered in a shape this reader could not use, which says nothing about whether the
#: comment is there -- and "there may be no such comment" is exactly the claim product
#: rule 2 forbids making from a silence.
COMMENT_NOT_LISTED = "the thread was returned without the comment this address names"

#: An instance that serves statuses only to people logged into it. 401 and 403 say so
#: outright; 422 is what Mastodon answers when it will not process the request for an
#: anonymous reader at all.
MASTODON_LOGIN_HINT = "this instance requires a login"
_MASTODON_LOGIN = frozenset({401, 403, 422})

#: The hosts with a reader bound to them. Mastodon is not here and cannot be: the
#: host is whichever instance carries the status, so it is recognised by the shape of
#: its address instead (:func:`_mastodon_status`) — which is also why the four below
#: are checked first, so an instance-shaped path on one of them cannot be mistaken
#: for a status.
_READ_HOSTS = frozenset({"bsky.app", "news.ycombinator.com"})
_REDDIT_HOSTS = frozenset({"reddit.com"})
_X_HOSTS = frozenset({"x.com", "twitter.com"})

#: What a quoted record says when it is not a post the reader may have. Bluesky names
#: each case in the view's own ``$type``, so each is reported as itself rather than
#: collapsed into "there was no quote" (spec section 15, product rule 2).
_QUOTE_UNREAD = {
    "app.bsky.embed.record#viewNotFound": "it was not found",
    "app.bsky.embed.record#viewBlocked": "it is blocked",
    "app.bsky.embed.record#viewDetached": "its author detached it",
}
QUOTE_UNREAD = "a quoted post could not be read: {detail}"

# ``at://<did>/app.bsky.feed.post/<rkey>`` is the real name of a Bluesky post; the
# bsky.app address is a rendering of it, which is why the rkey has to be lifted out.
_BSKY_COLLECTION = "app.bsky.feed.post"
_LINK_FACET = "app.bsky.richtext.facet#link"
# ``getPostThread`` answers for a post nobody may read with a stub in place of the
# thread. Spec section 15 keeps "there is no such post" and "this post is there and
# you may not have it" apart, and so does the API, so neither does this collapse them.
_BLOCKED_POST = "app.bsky.feed.defs#blockedPost"
_NOT_FOUND_POST = "app.bsky.feed.defs#notFoundPost"

# Hacker News and Mastodon both hand over fragments of HTML: entities, ``<a href>``
# and a tag between paragraphs. The links are read before the tags are taken out.
# The paragraph break differs -- Hacker News opens one and never closes it, Mastodon
# closes every one -- so each platform brings its own, and nothing else is shared
# with the other's output.
_ANCHOR = re.compile(r'<a\s[^>]*href="([^"]+)"', re.I)
_HN_BREAK = re.compile(r"<p>", re.I)
_MASTODON_BREAK = re.compile(r"</p>\s*<p[^>]*>|<br\s*/?>", re.I)
_TAG = re.compile(r"<[^>]+>")

# An address in a Reddit markdown body, written bare or as ``[text](address)``. The
# closing bracket and paren are excluded from the match, so the second form ends
# where the link does; sentence punctuation is trimmed off the end the way
# ``resolve.find_url`` trims it.
_MARKDOWN_URL = re.compile(r"https?://[^\s<>\"'\]\)]+")
_TRAILING = ".,;:!?"


@dataclass(frozen=True)
class Post:
    """One post as its platform gave it, and nothing added to it.

    ``links`` is what the post cites — facet links and the address of an embedded
    card on Bluesky, the submitted URL and the anchors in the body on Hacker News —
    in order of appearance and without repeats. ``quoted`` holds the posts this one
    quotes, one level deep, each with its own links: a quote is another author's
    words, so it carries its own sources rather than inheriting its quoter's.

    ``notes`` names what the platform would not hand over — a quoted post that is
    blocked, deleted or detached. Such a quote costs the document a paragraph and
    every link inside it, and dropping it silently would leave a run that read less
    than it appears to have read (product rule 2).
    """

    platform: str
    url: str
    author: str
    text: str
    links: tuple[str, ...]
    quoted: tuple[Post, ...]
    fetched_at: str
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Unreadable:
    """A post that was not read, and which of spec section 15's states that is.

    Separate from :class:`Post` rather than an empty one, because "the post says
    nothing" and "nobody could read the post" are the two things product rule 2
    exists to keep apart.
    """

    platform: str
    url: str
    state: str
    hint: str


def read_post(url: str, client: PoliteClient) -> Post | Unreadable:
    """One post address, read through whatever its platform offers a reader.

    Bluesky, Hacker News and Mastodon need no key, no account and no contact address,
    so ``mailto`` is kept off all of them: it is a courtesy Crossref and OpenAlex ask
    for and these would only log it. Reddit needs the user's own app and says so when
    it has none. X needs nothing, because there is nobody to ask.

    The order is the order the matchers are specific in: the four host-bound
    platforms first, then Mastodon, which is recognised by the shape of its address
    on any instance.
    """
    bluesky = _bluesky_post(url)
    if bluesky is not None:
        return _read_bluesky(url, bluesky, client)
    item = _hn_item(url)
    if item is not None:
        return _read_hn(url, item, client)
    article = _reddit_article(url)
    if article is not None:
        return _read_reddit(url, article, client)
    if _x_status(url):
        return _blocked(X, url, X_HINT)
    status = _mastodon_status(url)
    if status is not None:
        return _read_mastodon(url, status, client)
    return Unreadable(platform="", url=url, state=NOT_READ_HERE, hint=UNSUPPORTED_HINT)


def is_post_url(url: str) -> bool:
    """Whether this address names a single post one of the readers can answer for.

    A profile, a subreddit, a feed and a front page are not posts: there is no one
    thing to read and no one set of links to verify. ``providers.is_social`` answers
    the broader question of whether an address belongs to a platform at all.
    """
    return (
        _bluesky_post(url) is not None
        or _hn_item(url) is not None
        or _reddit_article(url) is not None
        or _x_status(url)
        or _mastodon_status(url) is not None
    )


def is_read_here(url: str) -> bool:
    """Whether some reader in this module answers for this address's platform.

    Broader than :func:`is_post_url`, which asks whether the address names one post:
    a profile, a subreddit and an instance's front page are all read here, and the
    answer they get is "that address is not one post", not "nobody reads this".

    Since task 10.3 every platform ``providers.is_social`` routes has a reader — X's
    answers that the post cannot be read, which is still an answer about that post —
    so this is true wherever that is, plus the instances its conservative host test
    misses and :func:`_mastodon_status` recognises by shape. The two are kept as
    separate questions because they are separate questions: the sixth platform
    someone adds to ``SOCIAL_HOSTS`` is social before it is read, and ``verify`` has
    to be able to say so rather than call a post a page.
    """
    return is_social(url) or _mastodon_status(url) is not None


class SocialProvider:
    """Posts: a source whose text is short, whole, and cited by linking.

    The other direction from ``check --url``. Here a bibliography cites a post, and
    what the post says is the passage a verdict would rest on — so the post is read
    through the same API, and its quoted posts travel with it, because a quote is
    part of what the citing entry pointed the reader at.

    An address on a platform this version cannot read goes up the fetch ladder
    instead, exactly as v0.3.0 read it. Answering "unreachable" for a page nobody
    tried to fetch would be a fact invented about the source (product rule 2).
    """

    scheme: Scheme = "social"

    def __init__(self, client: PoliteClient, fetcher: FetchesUrl) -> None:
        self._client = client
        self._web = WebProvider(fetcher)

    def resolve(self, ref: Reference) -> ResolveResult:
        """Not in the indexes, and never was -- the same fact ``WebProvider`` reports.

        No bibliographic index covers posts, so absence from one says nothing about
        this one: it is never a ghost (product rule 3).
        """
        url = find_url(ref.raw)
        if url is None:
            return ResolveResult(State.NOT_INDEXED, None, [])
        best = Candidate(
            doi="", title="", first_author="", year=None, venue="", provider="social", url=url
        )
        return ResolveResult(State.NOT_INDEXED, best, [best])

    def retraction(self, resolved: ResolveResult) -> Retraction | None:
        """Nobody publishes retraction notices about posts, so nobody is asked."""
        return None

    def fetch(self, ref: Reference, resolved: ResolveResult) -> list[EvidenceDoc]:
        """The post, or the reason there is no post. One document either way."""
        url = find_url(ref.raw) or _address(resolved)
        if not url:
            return []
        if not is_post_url(url):
            return self._web.fetch(ref, resolved)
        read = read_post(url, self._client)
        if isinstance(read, Unreadable):
            return [
                EvidenceDoc(
                    source_id=f"{URL_PREFIX}{url}",
                    text="",
                    text_kind="none",
                    state=read.state,
                    url=url,
                    step=None,
                    notes=(read.hint,) if read.hint else (),
                    winner=read.platform,
                    title="the post could not be read",
                )
            ]
        text = _rendered(read)
        return [
            EvidenceDoc(
                source_id=f"{URL_PREFIX}{url}",
                # A post is short and it is *whole*: the API returns all of it, so
                # there is no abstract-grade reading of one. Grading it by the page
                # length bar would file every post ever cited as "abstract only"
                # and cost the run a coverage grade it did not lose (product rule 6).
                text=text,
                text_kind="fulltext" if text else "none",
                state="" if text else NO_TEXT,
                url=read.url,
                step=None,
                # A quote the platform withheld is missing from ``text``, so the
                # source's own entry says so rather than reading as the whole post.
                notes=read.notes,
                winner=read.platform,
                title="the post was reached but held no text",
            )
        ]


# --- Bluesky ----------------------------------------------------------------------


@dataclass(frozen=True)
class _BlueskyPost:
    """The two halves of a bsky.app address: who posted, and which post."""

    actor: str  # a handle, or the DID itself
    rkey: str


def _bluesky_post(url: str) -> _BlueskyPost | None:
    """``https://bsky.app/profile/<handle-or-did>/post/<rkey>``, or ``None``."""
    split = urlsplit(url)
    host = (split.hostname or "").lower().removeprefix("www.")
    if host != "bsky.app":
        return None
    parts = [part for part in split.path.split("/") if part]
    if len(parts) != 4 or parts[0] != "profile" or parts[2] != "post":
        return None
    return _BlueskyPost(actor=parts[1], rkey=parts[3])


def _read_bluesky(url: str, target: _BlueskyPost, client: PoliteClient) -> Post | Unreadable:
    """Resolve the handle if it is one, then read the post and nothing under it."""
    did = target.actor
    try:
        if not did.startswith("did:"):
            resolved = client.get(
                f"{BSKY_API}/com.atproto.identity.resolveHandle",
                {"handle": did},
                mailto=False,
            )
            did = _text_field(_json(resolved), "did")
            if not did:
                return _gone(BLUESKY, url, f"no account answers to {target.actor}")
        # depth=0: the replies under a post are other people's words, and this is a
        # question about what *this* post says.
        response = client.get(
            f"{BSKY_API}/app.bsky.feed.getPostThread",
            {"uri": f"at://{did}/{_BSKY_COLLECTION}/{target.rkey}", "depth": 0},
            mailto=False,
        )
    except ProviderError as error:
        return _failed(BLUESKY, url, error)
    if response.status_code == 404:
        return _gone(BLUESKY, url, "the post was not found")
    thread = _mapping(_json(response), "thread")
    kind = _text_field(thread, "$type")
    if kind == _BLOCKED_POST or (kind != _NOT_FOUND_POST and thread.get("blocked")):
        return _blocked(BLUESKY, url, "the post is there, and its author does not serve it here")
    view = _mapping(thread, "post")
    if not view:
        return _gone(BLUESKY, url, "the post was not found")
    return _bluesky_view(url, view)


def _bluesky_view(url: str, view: dict[str, Any]) -> Post:
    """One ``postView`` (or one quoted ``viewRecord``) as a :class:`Post`."""
    record = _mapping(view, "record") or _mapping(view, "value")
    author = _text_field(_mapping(view, "author"), "handle")
    embed = _mapping(view, "embed")
    links = [
        *_facet_links(record),
        *_embedded_links(embed),
        *_embedded_links(_mapping(record, "embed")),
    ]
    for embedded in _embed_list(view):
        links.extend(_embedded_links(embedded))
    quotes, notes = _quoted(view)
    return Post(
        platform=BLUESKY,
        url=url,
        author=author,
        text=_text_field(record, "text").strip(),
        links=_ordered(links),
        quoted=tuple(_bluesky_view(_bluesky_url(quoted), quoted) for quoted in quotes),
        fetched_at=_now(),
        notes=tuple(notes),
    )


def _facet_links(record: dict[str, Any]) -> list[str]:
    """Every ``#link`` facet's target, in the order the post's own byte spans give."""
    facets = [facet for facet in _sequence(record, "facets") if isinstance(facet, dict)]
    facets.sort(key=lambda facet: _int_field(_mapping(facet, "index"), "byteStart"))
    return [
        uri
        for facet in facets
        for feature in _sequence(facet, "features")
        if isinstance(feature, dict)
        and _text_field(feature, "$type") == _LINK_FACET
        and (uri := _text_field(feature, "uri"))
    ]


def _embedded_links(embed: dict[str, Any]) -> list[str]:
    """The address of an embedded link card, wherever the view shape buries it."""
    if not embed:
        return []
    found = []
    external = _mapping(embed, "external")
    if uri := _text_field(external, "uri"):
        found.append(uri)
    # ``recordWithMedia#view`` puts the card under ``media`` and the quote under
    # ``record``; ``external#view`` puts it at the top. Both shapes are walked.
    found.extend(_embedded_links(_mapping(embed, "media")))
    return found


def _quoted(view: dict[str, Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """The quoted post's own view, one level deep, and what stood in its place.

    A quote the reader may not have comes back as a stub -- ``#viewNotFound``,
    ``#viewBlocked``, ``#viewDetached`` -- with no ``value`` under it. Returning
    nothing for one would take a paragraph and every link in it out of the document
    with nobody told, so the stub becomes a note on the post instead (rule 2).
    """
    embed = _mapping(view, "embed")
    record = _mapping(embed, "record")
    inner = _mapping(record, "record") or record
    if _mapping(inner, "value"):
        return [inner], []
    detail = _QUOTE_UNREAD.get(_text_field(inner, "$type"))
    return [], [QUOTE_UNREAD.format(detail=detail)] if detail else []


def _embed_list(view: dict[str, Any]) -> list[dict[str, Any]]:
    """A quoted record's own embeds, which the view returns as a list beside it."""
    return [item for item in _sequence(view, "embeds") if isinstance(item, dict)]


def _bluesky_url(view: dict[str, Any]) -> str:
    """The bsky.app address of a quoted post, rebuilt from its ``at://`` name."""
    handle = _text_field(_mapping(view, "author"), "handle")
    rkey = _text_field(view, "uri").rsplit("/", 1)[-1]
    if not handle or not rkey:
        return ""
    return f"https://bsky.app/profile/{handle}/post/{rkey}"


# --- Hacker News --------------------------------------------------------------------


def _hn_item(url: str) -> int | None:
    """``https://news.ycombinator.com/item?id=<n>``, or ``None``."""
    split = urlsplit(url)
    host = (split.hostname or "").lower().removeprefix("www.")
    if host != "news.ycombinator.com" or split.path.rstrip("/") != "/item":
        return None
    values = parse_qs(split.query).get("id", [])
    return int(values[0]) if values and values[0].isdigit() else None


def _read_hn(url: str, item: int, client: PoliteClient) -> Post | Unreadable:
    """One item from the Firebase export. A comment's parent is never followed:
    the thread above it is other people's words, not this comment's sources."""
    try:
        response = client.get(f"{HN_API}/item/{item}.json", mailto=False)
    except ProviderError as error:
        return _failed(HACKER_NEWS, url, error)
    if response.status_code == 404:
        return _gone(HACKER_NEWS, url, "the item was not found")
    payload = _json(response)
    # The export names the two separately, so they are reported separately: ``dead``
    # is an item the site still has and no longer shows, ``deleted`` is one that is
    # gone. Spec section 15 keeps BLOCKED and UNREACHABLE apart for exactly this.
    if payload and payload.get("dead") and not payload.get("deleted"):
        return _blocked(HACKER_NEWS, url, "the item was killed by moderation")
    if not payload or payload.get("deleted"):
        return _gone(HACKER_NEWS, url, "the item was deleted or is no longer served")
    body, anchors = _html_text(_text_field(payload, "text"), _HN_BREAK)
    title = _text_field(payload, "title").strip()
    submitted = _text_field(payload, "url")
    return Post(
        platform=HACKER_NEWS,
        url=url,
        author=_text_field(payload, "by"),
        text="\n".join(part for part in (title, body) if part),
        # A story's submitted address is the thing it is about, so it comes first.
        links=_ordered([submitted, *anchors] if submitted else anchors),
        quoted=(),
        fetched_at=_now(),
    )


def _html_text(markup: str, breaks: re.Pattern[str]) -> tuple[str, list[str]]:
    """A post body as words, and the addresses its anchors pointed at.

    The anchors are read first, because taking the tags out is what makes the rest
    readable and would take the ``href``s with it.
    """
    if not markup:
        return "", []
    anchors = [html.unescape(href) for href in _ANCHOR.findall(markup)]
    text = breaks.sub("\n", markup)
    text = _TAG.sub("", text)
    return html.unescape(text).strip(), anchors


# --- Reddit ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _RedditArticle:
    """Which thread an address names, and which comment in it, if it names one."""

    article: str
    comment: str  # "" when the address names the post at the top of the thread


def _reddit_article(url: str) -> _RedditArticle | None:
    """``reddit.com/r/<sub>/comments/<id>[/<slug>[/<comment>]]``, or ``None``.

    ``/comments/<id>`` without the subreddit is the same thread and is accepted too.
    A subreddit, a user page and the front page are not posts and return ``None``.
    """
    split = urlsplit(url)
    if not _in((split.hostname or "").lower(), _REDDIT_HOSTS):
        return None
    parts = [part for part in split.path.split("/") if part]
    if len(parts) >= 4 and parts[0] == "r" and parts[2] == "comments":
        return _RedditArticle(parts[3], parts[5] if len(parts) >= 6 else "")
    if len(parts) >= 2 and parts[0] == "comments":
        return _RedditArticle(parts[1], parts[3] if len(parts) >= 4 else "")
    return None


def _read_reddit(url: str, target: _RedditArticle, client: PoliteClient) -> Post | Unreadable:
    """One thread from the user's own free app, or the reason there was no app.

    The token is fetched per read rather than kept: it is a credential, and a
    credential that is never stored is one that cannot leak out of a long-lived
    object into a log, a cache or another run's report. A run reads one post or a
    handful, and Reddit's client-credentials grant is one small request each.
    """
    app = _reddit_app()
    if app is None:
        # Nobody was asked. Not unreachable, not blocked: those say something about
        # the post, and this says something about this machine (product rule 2).
        return Unreadable(platform=REDDIT, url=url, state=CREDENTIALS_MISSING, hint=REDDIT_HINT)
    token = _reddit_token(url, app, client)
    if isinstance(token, Unreadable):
        return token
    params: dict[str, Any] = {"raw_json": 1}
    if target.comment:
        # The comment and nothing above it: the thread over a comment is other
        # people's words, not this comment's sources.
        params |= {"comment": target.comment, "context": 0, "depth": 1}
    try:
        response = client.get(
            f"{REDDIT_API}/comments/{target.article}.json",
            params,
            mailto=False,
            headers={"Authorization": f"bearer {token}"},
        )
    except ProviderError as error:
        return _failed(REDDIT, url, error)
    if response.status_code == 404:
        return _gone(REDDIT, url, "the thread was not found")
    listings = _listings(response)
    if target.comment:
        found = _reddit_comment(listings, target.comment)
        # Not ``_gone``: the comment was looked for in what Reddit sent and was not in
        # it, which is a fact about the answer's shape and not about the comment.
        return (
            _reddit_view(url, found, comment=True)
            if found
            else Unreadable(
                platform=REDDIT,
                url=url,
                state=Outcome.UNAVAILABLE.value,
                hint=COMMENT_NOT_LISTED,
            )
        )
    post = _reddit_post(listings)
    if not post:
        return _gone(REDDIT, url, "the thread was reached and held no such post")
    return _reddit_view(url, post, comment=False)


def _reddit_app() -> tuple[ApiKey, ApiKey] | None:
    """The user's client id and secret, or ``None`` when either is absent.

    Both or neither: an id without a secret cannot authenticate, so sending it would
    hand Reddit half a credential for a refusal that was never about the post.
    """
    ident = resolve_api_key(REDDIT_CLIENT_ID_ENV)
    secret = resolve_api_key(REDDIT_CLIENT_SECRET_ENV)
    if ident is None or secret is None:
        return None
    return ident, secret


def _reddit_token(url: str, app: tuple[ApiKey, ApiKey], client: PoliteClient) -> str | Unreadable:
    """One ``client_credentials`` grant. The two values go in HTTP basic auth only."""
    ident, secret = app
    try:
        response = client.post(
            REDDIT_TOKEN_URL,
            data={"grant_type": "client_credentials"},
            auth=(ident.value, secret.value),
        )
    except ProviderError as error:
        # A rejected app is Reddit refusing this reader, which ``_failed`` files as
        # BLOCKED. The error carries a status and a message, never the credential.
        return _failed(REDDIT, url, error)
    token = _text_field(_json(response), "access_token")
    if not token:
        return Unreadable(
            platform=REDDIT,
            url=url,
            state=Outcome.UNAVAILABLE.value,
            hint="Reddit issued no access token",
        )
    return token


def _listings(response: Any) -> list[dict[str, Any]]:
    """``[post listing, comment listing]``, as the article endpoint returns them."""
    try:
        payload = response.json()
    except ValueError:
        return []
    return [item for item in payload if isinstance(item, dict)] if isinstance(payload, list) else []


def _reddit_post(listings: list[dict[str, Any]]) -> dict[str, Any]:
    children = _reddit_children(listings[0]) if listings else []
    return children[0] if children else {}


def _reddit_comment(listings: list[dict[str, Any]], comment: str) -> dict[str, Any]:
    """The comment the permalink names, from the second listing.

    Matched on its id rather than taken positionally: ``context`` and ``depth`` are
    requests, not guarantees, and reading a neighbouring comment as the cited one
    would put another author's words behind the citing entry (product rule 1).

    For the same reason the search does not stop at the top of the listing. Reddit
    honours the two hints as it sees fit, and a live comment it chose to return one
    level down is a comment that exists -- calling it missing would be absence read as
    evidence (product rule 2). One level of ``replies`` is walked because that is what
    ``depth=1`` can hand back; deeper is a thread this reader never asked for.
    """
    if len(listings) < 2:
        return {}
    children = _reddit_children(listings[1])
    for data in children:
        if _text_field(data, "id") == comment:
            return data
    for data in children:
        for reply in _reddit_children(_mapping(data, "replies")):
            if _text_field(reply, "id") == comment:
                return reply
    return {}


def _reddit_children(listing: dict[str, Any]) -> list[dict[str, Any]]:
    """Every child's own ``data`` in one listing, in the order Reddit returned them.

    A comment with no replies carries ``""`` rather than a listing, which
    :func:`_mapping` turns into an empty one, so no caller has to know that.
    """
    return [
        data
        for child in _sequence(_mapping(listing, "data"), "children")
        if isinstance(child, dict) and (data := _mapping(child, "data"))
    ]


def _reddit_view(url: str, payload: dict[str, Any], *, comment: bool) -> Post | Unreadable:
    """One thing Reddit returned as a :class:`Post`, or why its words are not there.

    ``[removed]`` and ``[deleted]`` are the two words Reddit puts where a body was,
    and they are different facts: a moderator withheld this one, its author took that
    one away. Spec section 15 keeps BLOCKED and UNREACHABLE apart for exactly that.
    """
    body = _text_field(payload, "body" if comment else "selftext").strip()
    if body == "[removed]" or _text_field(payload, "removed_by_category"):
        return _blocked(REDDIT, url, "the post was removed by moderation")
    if body == "[deleted]" or _text_field(payload, "author") == "[deleted]":
        return _gone(REDDIT, url, "the post was deleted by its author")
    title = "" if comment else _text_field(payload, "title").strip()
    submitted = "" if comment or payload.get("is_self") else _text_field(payload, "url")
    return Post(
        platform=REDDIT,
        url=url,
        author=_text_field(payload, "author"),
        text="\n".join(part for part in (title, body) if part),
        # A link post's submitted address is the thing it is about, so it comes
        # first. A self post's ``url`` is the post itself, and citing that would
        # send the ladder after the page this post was read from.
        links=_ordered([submitted, *_markdown_links(body)] if submitted else _markdown_links(body)),
        quoted=(),
        fetched_at=_now(),
    )


def _markdown_links(markdown: str) -> list[str]:
    """Every address in a markdown body, whether written bare or behind link text."""
    return [link.rstrip(_TRAILING) for link in _MARKDOWN_URL.findall(markdown)]


# --- Mastodon -------------------------------------------------------------------------


def _mastodon_status(url: str) -> tuple[str, str] | None:
    """``(instance, status id)`` for the two addresses Mastodon publishes, else ``None``.

    Matched on the shape and not on a list of hosts: there is no list of Mastodon
    instances to keep, and one that is missing from a list is a post read as a page.
    ``providers.is_social`` keeps its conservative host test for *routing* — a miss
    there costs a cited status the social family and it is fetched as a page, which
    is what v0.3.0 did — where this is what a reader needs to make the call.
    """
    split = urlsplit(url)
    host = (split.hostname or "").lower()
    # Subdomain-aware, like the other two exclusions: ``www.bsky.app`` is Bluesky
    # whatever its path looks like, and asking it for a Mastodon status would send a
    # request to an endpoint that was never there.
    if not host or _in(host, _READ_HOSTS) or _in(host, _REDDIT_HOSTS) or _in(host, _X_HOSTS):
        return None
    instance = _instance(split)
    if not instance:
        return None
    parts = [part for part in split.path.split("/") if part]
    if len(parts) == 2 and parts[0].startswith("@") and parts[1].isdigit():
        return instance, parts[1]
    if len(parts) == 4 and parts[0] == "users" and parts[2] == "statuses" and parts[3].isdigit():
        return instance, parts[3]
    return None


def _instance(split: SplitResult) -> str:
    """``host`` or ``host:port``: the instance to ask, and nothing else from the netloc.

    ``netloc`` carries the userinfo too, and an address written ``user@host`` handed to
    ``httpx`` becomes a basic-auth header -- a credential this reader never had, sent
    to an instance that never asked for one. The port is kept because it is part of the
    address; a port that is not a number names no instance at all, so nothing is read.
    """
    host = (split.hostname or "").lower()
    try:
        port = split.port
    except ValueError:
        return ""
    return host if port is None else f"{host}:{port}"


def _read_mastodon(url: str, target: tuple[str, str], client: PoliteClient) -> Post | Unreadable:
    """One status from the instance that hosts it. No key, no account, best effort."""
    instance, status_id = target
    try:
        response = client.get(f"https://{instance}/api/v1/statuses/{status_id}", mailto=False)
    except ProviderError as error:
        if error.status in _MASTODON_LOGIN:
            # The status is there and this reader may not have it. ``_failed`` would
            # file 422 as "not found", which is a claim about the post instead.
            return _blocked(MASTODON, url, MASTODON_LOGIN_HINT)
        return _failed(MASTODON, url, error)
    if response.status_code == 404:
        return _gone(MASTODON, url, "the status was not found")
    payload = _json(response)
    if not payload:
        return _gone(MASTODON, url, "the instance answered with no status")
    text, anchors = _html_text(_text_field(payload, "content"), _MASTODON_BREAK)
    return Post(
        platform=MASTODON,
        url=url,
        author=_text_field(_mapping(payload, "account"), "acct"),
        text=text,
        # The anchors in the author's own words first, then the preview card, which
        # the instance builds from the last link and is usually one of them anyway.
        links=_ordered([*anchors, _text_field(_mapping(payload, "card"), "url")]),
        quoted=(),
        fetched_at=_now(),
    )


# --- X ----------------------------------------------------------------------------------


def _x_status(url: str) -> bool:
    """``x.com``/``twitter.com`` ``/<user>/status/<id>``, the one shape X publishes."""
    split = urlsplit(url)
    if not _in((split.hostname or "").lower(), _X_HOSTS):
        return False
    parts = [part for part in split.path.split("/") if part]
    return any(
        part in {"status", "statuses"} and index + 1 < len(parts) and parts[index + 1].isdigit()
        for index, part in enumerate(parts)
    )


# --- shared helpers -------------------------------------------------------------------


def _in(host: str, hosts: frozenset[str]) -> bool:
    """Whether a host is one of these, or a subdomain of one (``old.reddit.com``)."""
    return any(host == known or host.endswith(f".{known}") for known in hosts)


def _rendered(post: Post) -> str:
    """A post and the posts it quotes, as the one passage a verdict could rest on."""
    parts = [post.text, *(quoted.text for quoted in post.quoted)]
    return "\n\n".join(part for part in parts if part)


def _gone(platform: str, url: str, hint: str) -> Unreadable:
    """A post that is not there. ``UNREACHABLE`` is a fact about this address."""
    return Unreadable(platform=platform, url=url, state=Outcome.UNREACHABLE.value, hint=hint)


def _blocked(platform: str, url: str, hint: str) -> Unreadable:
    """A post that is there and is not served here. ``BLOCKED``, never ``UNREACHABLE``.

    Spec section 15 keeps the two apart, and so must this: "nobody may read it" says
    the post exists and its words are unknown, where "unreachable" says there may be
    no such post at all. Reporting the first as the second is the absence-as-evidence
    product rule 2 forbids.
    """
    return Unreadable(platform=platform, url=url, state=Outcome.BLOCKED.value, hint=hint)


def _failed(platform: str, url: str, error: ProviderError) -> Unreadable:
    """A request that ended in an error, filed by what the error was about.

    A 4xx is the platform answering about this post -- Bluesky says a deleted post
    is gone with a 400, not a 404 -- and a 5xx, a rate limit or a dead socket is the
    platform saying nothing at all. Collapsing the two would report an outage as a
    missing post (product rule 2).

    401 and 403 are the one 4xx that names itself: the platform is saying the post is
    there and this reader may not have it, which is the "403/bot protection" spec
    section 15 reserves ``BLOCKED`` for. The rest of the 4xx range stays
    ``UNREACHABLE`` even when a takedown is what caused it: the error body's own word
    for a deleted post and for a suspended account is the same ``NotFound``, and
    ``ProviderError`` carries only the status and the message. The blocked case the
    API *does* name arrives as a ``#blockedPost`` thread instead, which
    :func:`_read_bluesky` reads before it gets here.
    """
    status = error.status
    if status in (401, 403):
        return _blocked(platform, url, f"the platform would not serve the post ({error})")
    if status is not None and 400 <= status < 500 and status != 429:
        return _gone(platform, url, f"the post was not found ({error})")
    return Unreadable(platform=platform, url=url, state=Outcome.UNAVAILABLE.value, hint=str(error))


def _address(resolved: ResolveResult) -> str | None:
    """The address a resolve result recorded, when the record carries one."""
    best = resolved.best
    return best.url if best is not None and best.url else None


def _now() -> str:
    """When the post was read, to the second, in UTC."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _ordered(urls: list[str]) -> tuple[str, ...]:
    """The addresses in order of first appearance, without repeats."""
    return tuple(dict.fromkeys(url for url in urls if url))


def _json(response: Any) -> dict[str, Any]:
    """A JSON body as a mapping. A body that is not one is a body with nothing in it."""
    try:
        payload = response.json()
    except ValueError:
        return {}
    return payload if isinstance(payload, dict) else {}


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    return value if isinstance(value, dict) else {}


def _sequence(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key)
    return list(value) if isinstance(value, list) else []


def _text_field(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    return value if isinstance(value, str) else ""


def _int_field(payload: dict[str, Any], key: str) -> int:
    value = payload.get(key)
    return value if isinstance(value, int) else 0
