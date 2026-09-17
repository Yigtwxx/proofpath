"""The ``social://`` family: a post read without an account, and what it cites.

Spec section 6.2. Nothing here touches the network — the two read-only APIs are
``respx`` routes over the fixtures in ``tests/fixtures/social/`` — so what is tested
is the reading: which links a post carries, what a post that is gone says instead of
a verdict (product rule 2), and that a post with no links is reported as having
nothing to verify rather than passing silently (product rule 6).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest
import respx
from typer.testing import CliRunner

from proofpath import ingest
from proofpath import secrets as secrets_mod
from proofpath.claims import extract
from proofpath.cli import app
from proofpath.fetch import Outcome
from proofpath.polite import MIN_INTERVAL, PoliteClient
from proofpath.providers import NO_TEXT, provider_for, reader_for
from proofpath.providers.social import (
    BSKY_API,
    COMMENT_NOT_LISTED,
    HN_API,
    MASTODON_LOGIN_HINT,
    NOT_READ_HERE,
    REDDIT_API,
    REDDIT_TOKEN_URL,
    UNSUPPORTED_HINT,
    X_HINT,
    Post,
    SocialProvider,
    Unreadable,
    is_post_url,
    is_read_here,
    read_post,
)
from proofpath.report import Kind
from proofpath.resolve import State
from proofpath.secrets import (
    CREDENTIALS_MISSING,
    REDDIT_CLIENT_ID_ENV,
    REDDIT_CLIENT_SECRET_ENV,
)
from proofpath.verify import POST_READER, prepare
from tests.test_providers import FULLTEXT, StubFetcher, built, fetched, ref
from tests.test_verify import StubFetcher as VerifyStubFetcher
from tests.test_verify import StubResolver as VerifyStubResolver
from tests.test_verify import engine as build_engine
from tests.test_verify import fetched as verify_fetched

runner = CliRunner()

FIX = Path(__file__).parent / "fixtures" / "social"

HANDLE_URL = "https://bsky.app/profile/bsky.app/post/3lxmodreport"
DID = "did:plc:z72i7hdynmk6r22z27h6tvur"
DID_URL = f"https://bsky.app/profile/{DID}/post/3lxmodreport"
AT_URI = f"at://{DID}/app.bsky.feed.post/3lxmodreport"
QUOTE_URL = "https://bsky.app/profile/press.example.test/post/3lxpressnote"

REPORT_LINK = "https://example.test/moderation-report-2025"
APPENDIX_LINK = "https://example.test/appendix.pdf"
REGULATOR_LINK = "https://example.test/regulator-tally"

HN_STORY_URL = "https://news.ycombinator.com/item?id=8863"
HN_COMMENT_URL = "https://news.ycombinator.com/item?id=8952"
HN_STORY_LINK = "https://example.test/air-quality-guidance"
PAPER_LINK = "https://example.test/paper.pdf"
DATA_LINK = "https://example.test/data.csv"


def fixture(name: str) -> Any:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


@pytest.fixture
def client() -> Any:
    polite = PoliteClient(client=httpx.Client(timeout=5.0))
    yield polite
    polite.client.close()


def resolve_route() -> Any:
    return respx.get(f"{BSKY_API}/com.atproto.identity.resolveHandle").mock(
        return_value=httpx.Response(200, json=fixture("bsky-resolve-handle.json"))
    )


def thread_route(name: str = "bsky-thread.json", status: int = 200) -> Any:
    return respx.get(f"{BSKY_API}/app.bsky.feed.getPostThread").mock(
        return_value=httpx.Response(status, json=fixture(name))
    )


def hn_route(item: int, name: str) -> Any:
    return respx.get(f"{HN_API}/item/{item}.json").mock(
        return_value=httpx.Response(200, json=fixture(name))
    )


# --- which addresses are posts at all ----------------------------------------


def test_the_two_first_class_platforms_are_recognised_by_their_post_shape() -> None:
    assert is_post_url(HANDLE_URL)
    assert is_post_url(DID_URL)
    assert is_post_url(HN_STORY_URL)
    # A profile is not a post, and neither is the front page.
    assert not is_post_url("https://bsky.app/profile/bsky.app")
    assert not is_post_url("https://news.ycombinator.com/")
    assert not is_post_url("https://example.test/report.html")


def test_an_address_on_no_platform_at_all_is_not_a_post() -> None:
    """The platforms are spec section 6.2's list and nothing else. An ordinary page
    is not a post that could not be read -- it was never a post (product rule 2)."""
    for url in (
        "https://example.test/report.html",
        "https://www.tiktok.com/@someone/video/1234567890",
    ):
        assert not is_post_url(url), url
        assert not is_read_here(url), url


def test_both_apis_are_paced_at_the_interval_the_spec_asks_for() -> None:
    assert MIN_INTERVAL["public.api.bsky.app"] == 0.5
    assert MIN_INTERVAL["hacker-news.firebaseio.com"] == 0.5


# --- Bluesky ------------------------------------------------------------------


@respx.mock
def test_a_bluesky_post_carries_its_facet_link_its_embed_and_its_quote(client: Any) -> None:
    handle = resolve_route()
    thread = thread_route()

    post = read_post(HANDLE_URL, client)

    assert isinstance(post, Post)
    assert handle.calls.last.request.url.params["handle"] == "bsky.app"
    assert thread.calls.last.request.url.params["uri"] == AT_URI
    assert thread.calls.last.request.url.params["depth"] == "0"
    assert post.platform == "Bluesky"
    assert post.author == "bsky.app"
    assert post.text.startswith("Our 2025 moderation report is out.")
    # Facet first, then the external embed: order of appearance, not of discovery.
    assert post.links == (REPORT_LINK, APPENDIX_LINK)
    assert post.fetched_at
    # One level of quoting, with the quoted post's own links on the quoted post.
    assert len(post.quoted) == 1
    quote = post.quoted[0]
    assert (quote.author, quote.url) == ("press.example.test", QUOTE_URL)
    assert quote.text == "The regulator published its own tally this morning."
    assert quote.links == (REGULATOR_LINK,)
    assert quote.quoted == ()


@respx.mock
def test_a_post_addressed_by_did_skips_the_handle_lookup(client: Any) -> None:
    handle = resolve_route()
    thread_route()

    post = read_post(DID_URL, client)

    assert isinstance(post, Post)
    assert handle.call_count == 0


@respx.mock
def test_a_deleted_bluesky_post_is_unreachable_and_never_a_verdict(client: Any) -> None:
    resolve_route()
    thread_route("bsky-not-found.json", status=400)

    unreadable = read_post(HANDLE_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNREACHABLE.value
    assert unreadable.platform == "Bluesky"
    assert "not found" in unreadable.hint.lower()


@respx.mock
def test_a_bluesky_outage_is_unavailable_not_unreachable(client: Any) -> None:
    """A provider that is down says nothing about the post. Reporting it as
    unreachable would turn an outage into a fact about the source (product rule 2)."""
    resolve_route()
    respx.get(f"{BSKY_API}/app.bsky.feed.getPostThread").mock(
        return_value=httpx.Response(503, json={"error": "Unavailable"})
    )

    unreadable = read_post(HANDLE_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNAVAILABLE.value


# --- Hacker News --------------------------------------------------------------


@respx.mock
def test_an_hn_story_carries_its_title_and_the_address_it_submitted(client: Any) -> None:
    route = hn_route(8863, "hn-story.json")

    post = read_post(HN_STORY_URL, client)

    assert isinstance(post, Post)
    assert route.call_count == 1
    assert post.platform == "Hacker News"
    assert post.author == "dhouston"
    assert post.text == "Air quality guidance was revised this week"
    assert post.links == (HN_STORY_LINK,)
    assert post.quoted == ()


@respx.mock
def test_an_hn_comment_reads_its_own_text_and_never_follows_its_parent(client: Any) -> None:
    route = hn_route(8952, "hn-comment.json")

    post = read_post(HN_COMMENT_URL, client)

    assert isinstance(post, Post)
    assert [str(call.request.url) for call in route.calls] == [f"{HN_API}/item/8952.json"]
    assert post.links == (PAPER_LINK, DATA_LINK)
    # HTML entities are decoded and the tags are gone; the words are the author's.
    assert "&quot;" not in post.text and "<a href" not in post.text
    assert '"revised" is the author\'s word & not mine' in post.text


@respx.mock
def test_a_deleted_hn_item_and_a_missing_one_are_both_unreachable(client: Any) -> None:
    hn_route(8999, "hn-deleted.json")
    hn_route(1, "hn-missing.json")

    for url, item in (("https://news.ycombinator.com/item?id=8999", 8999),
                      ("https://news.ycombinator.com/item?id=1", 1)):  # fmt: skip
        unreadable = read_post(url, client)
        assert isinstance(unreadable, Unreadable), item
        assert unreadable.state == Outcome.UNREACHABLE.value


# --- an unsupported host -------------------------------------------------------


@respx.mock
def test_an_unsupported_platform_says_so_rather_than_inventing_a_state(client: Any) -> None:
    unreadable = read_post("https://www.tiktok.com/@someone/video/1234567890", client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.hint == UNSUPPORTED_HINT
    assert respx.calls.call_count == 0


# --- ingest.from_post ----------------------------------------------------------


@respx.mock
def test_a_post_becomes_one_paragraph_per_post_and_one_reference_per_link(
    client: Any,
) -> None:
    resolve_route()
    thread_route()
    post = read_post(HANDLE_URL, client)
    assert isinstance(post, Post)

    document, _ = ingest.from_post(post)

    assert (document.kind, document.name, document.pages) == ("post", HANDLE_URL, 1)
    assert len(document.paragraphs) == 2
    assert document.paragraphs[0].text.startswith("Our 2025 moderation report")
    assert document.paragraphs[1].text.startswith("The regulator published")
    # Numbered in order of appearance, across the post and the post it quotes.
    assert [(r.number, r.raw) for r in document.references] == [
        (1, REPORT_LINK),
        (2, APPENDIX_LINK),
        (3, REGULATOR_LINK),
    ]


@respx.mock
def test_every_sentence_of_a_post_stands_behind_that_posts_links(client: Any) -> None:
    """The post as a whole is what its links are supposed to back, so no single
    sentence is given a confident verdict on its own (spec section 6.2)."""
    resolve_route()
    thread_route()
    post = read_post(HANDLE_URL, client)
    assert isinstance(post, Post)

    claims = extract(ingest.from_post(post)[0])

    first = [claim for claim in claims.claims if claim.paragraph == 0]
    assert len(first) == 3
    assert all(claim.cited_refs == (1, 2) for claim in first)
    assert all(claim.paragraph_scoped for claim in first)
    assert len({claim.group for claim in first}) == 1
    # The quoted post carries its own link and nothing of its quoter's.
    quoted = [claim for claim in claims.claims if claim.paragraph == 1]
    assert quoted and all(claim.cited_refs == (3,) for claim in quoted)
    # Coverage is counted from the markers, so a post that cites must not read as
    # a post that cites nothing (product rule 6).
    assert len(claims.markers) == 2


@respx.mock
def test_a_post_with_no_links_has_nothing_to_verify_and_says_so(client: Any) -> None:
    hn_route(8863, "hn-story.json")
    story = read_post(HN_STORY_URL, client)
    assert isinstance(story, Post)
    bare = Post(
        platform=story.platform, url=story.url, author=story.author,
        text="No source for this, just a feeling.", links=(), quoted=(),
        fetched_at=story.fetched_at,
    )  # fmt: skip

    document, _ = ingest.from_post(bare)

    assert document.references == ()
    assert extract(document).claims == ()


# --- a post is never its own evidence ------------------------------------------


NOW = "2026-09-16T09:00:00+00:00"


def posted(url: str, *, text: str, links: tuple[str, ...], quoted: tuple[Post, ...] = ()) -> Post:
    """One post as a platform would have handed it over, with no reading involved."""
    return Post(
        platform="Bluesky", url=url, author="someone.test", text=text,
        links=links, quoted=quoted, fetched_at=NOW,
    )  # fmt: skip


def test_a_post_that_links_to_itself_is_not_listed_as_its_own_source() -> None:
    """Product rule 1: the passage behind a claim may never be the claim itself.
    Left in, the link routes straight back to ``SocialProvider.fetch``, which reads
    this same post again and hands its own words back as the evidence for them."""
    post = posted(
        HANDLE_URL,
        text="The vaccine reduced hospitalisation by 80 percent.",
        links=(HANDLE_URL, REPORT_LINK),
    )

    document, notes = ingest.from_post(post)

    assert [reference.raw for reference in document.references] == [REPORT_LINK]
    # Told, not silently dropped: a reader counting the post's links must be able to
    # see which one is not in the report and why (product rule 6).
    assert notes == (ingest.SELF_LINK_DROPPED.format(url=HANDLE_URL),)


@pytest.mark.parametrize(
    "written",
    [
        HANDLE_URL,
        f"{HANDLE_URL}/",
        HANDLE_URL.replace("https://bsky.app", "https://BSKY.App"),
        HANDLE_URL.replace("https://bsky.app", "https://www.bsky.app"),
        HANDLE_URL.replace("https://", "http://"),
        f"{HANDLE_URL}#comments",
        "https://news.ycombinator.com/item?id=8863",
    ],
)
def test_a_self_link_is_recognised_however_the_post_spelled_it(written: str) -> None:
    """Host case, a ``www.``, the scheme, a trailing slash and a fragment naming a
    place inside the post all leave the address naming the same post. The Hacker
    News form is in here because a story's ``url`` field carries its own permalink
    when the submitter had nothing else to submit, and its item number is a query."""
    owner = HANDLE_URL if "ycombinator" not in written else HN_STORY_URL
    post = posted(owner, text="One sentence with a source.", links=(written, REPORT_LINK))

    document, notes = ingest.from_post(post)

    assert [reference.raw for reference in document.references] == [REPORT_LINK]
    assert notes == (ingest.SELF_LINK_DROPPED.format(url=written),)


def test_a_post_linking_to_a_different_post_keeps_it() -> None:
    """Only the post's own address goes. Another post is another author's words and
    a real source; dropping one would be product rule 3's failure in miniature."""
    other_rkey = HANDLE_URL.replace("3lxmodreport", "3LXMODREPORT")
    post = posted(
        HANDLE_URL,
        text="Two other posts said the same thing.",
        links=(QUOTE_URL, other_rkey),
    )

    document, notes = ingest.from_post(post)

    # The rkey's own case is part of which post it names, so these are two posts.
    assert [reference.raw for reference in document.references] == [QUOTE_URL, other_rkey]
    assert notes == ()


def test_a_quoted_post_that_links_to_itself_is_not_listed_as_its_own_source() -> None:
    """A quote is another author's post, read as its own paragraph with its own
    links -- so a link of its own back at itself is the same failure, one level in."""
    quoted = posted(
        QUOTE_URL,
        text="The regulator published its own tally this morning.",
        links=(QUOTE_URL, REGULATOR_LINK),
    )
    quoter = posted(
        HANDLE_URL,
        text="Our 2025 moderation report is out.",
        links=(REPORT_LINK,),
        quoted=(quoted,),
    )

    document, notes = ingest.from_post(quoter)
    claims = extract(document)

    assert [reference.raw for reference in document.references] == [REPORT_LINK, REGULATOR_LINK]
    assert notes == (ingest.SELF_LINK_DROPPED.format(url=QUOTE_URL),)
    # The numbering closes over the gap, so the quoted author's sentences still cite
    # the link that is theirs and nothing of their quoter's.
    quoted_claims = [claim for claim in claims.claims if claim.paragraph == 1]
    assert quoted_claims and all(claim.cited_refs == (2,) for claim in quoted_claims)


@respx.mock
def test_a_run_over_a_story_submitted_at_its_own_address_says_the_link_was_dropped() -> None:
    """End to end: the note has to reach the report, or the reader is left to
    notice that a link they can see in the post is not in the run."""
    story = fixture("hn-story.json")
    story["url"] = HN_STORY_URL
    respx.get(f"{HN_API}/item/8863.json").mock(return_value=httpx.Response(200, json=story))
    built_engine = build_engine(resolver=VerifyStubResolver(), fetcher=VerifyStubFetcher({}))

    ready = prepare(HN_STORY_URL, built_engine)

    titles = [finding.title for finding in ready.findings if finding.kind is Kind.PARSE_ERROR]
    assert ingest.SELF_LINK_DROPPED.format(url=HN_STORY_URL) in titles


# --- pasted text ---------------------------------------------------------------


def test_pasted_text_turns_its_links_into_numbered_references() -> None:
    from proofpath.verify import target_document

    text = (
        f"The count rose 17% last year, see {REPORT_LINK} for the tally.\n"
        f"The method is in {APPENDIX_LINK}.\n"
        "\n"
        f"A separate body disagreed: {REGULATOR_LINK}\n"
    )

    document = target_document(text, name="stdin")

    assert [(r.number, r.raw) for r in document.references] == [
        (1, REPORT_LINK),
        (2, APPENDIX_LINK),
        (3, REGULATOR_LINK),
    ]
    claims = extract(document)
    first = [claim for claim in claims.claims if claim.paragraph == 0]
    assert first and all(claim.cited_refs == (1, 2) for claim in first)
    second = [claim for claim in claims.claims if claim.paragraph == 1]
    assert second and all(claim.cited_refs == (3,) for claim in second)
    # The address is the citation, not the assertion: it is out of the claim text.
    assert all("http" not in claim.text for claim in claims.claims)


def test_a_pasted_draft_that_prints_a_bibliography_is_untouched() -> None:
    from proofpath.verify import target_document

    text = f"A claim [1].\n\nReferences\n\n[1] A paper. {REPORT_LINK}\n"

    document = target_document(text)

    assert [r.raw for r in document.references] == [f"[1] A paper. {REPORT_LINK}"]
    assert [claim.cited_refs for claim in extract(document).claims] == [(1,)]


# --- SocialProvider ------------------------------------------------------------


@respx.mock
def test_the_social_provider_answers_not_indexed_and_carries_the_address(client: Any) -> None:
    provider = SocialProvider(client, StubFetcher())

    result = provider.resolve(ref(HANDLE_URL))

    assert result.state is State.NOT_INDEXED
    assert result.best is not None and result.best.url == HANDLE_URL
    assert provider.scheme == "social"
    assert provider.retraction(result) is None


@respx.mock
def test_a_cited_post_is_read_through_the_api_and_quoted_in_its_authors_words(
    client: Any,
) -> None:
    resolve_route()
    thread_route()
    fetcher = StubFetcher()
    provider = SocialProvider(client, fetcher)

    doc = provider.fetch(ref(HANDLE_URL), provider.resolve(ref(HANDLE_URL)))[0]

    assert fetcher.calls == []  # the ladder is not asked about a post it cannot read
    assert doc.source_id == f"url:{HANDLE_URL}"
    assert (doc.text_kind, doc.state) == ("fulltext", "")
    assert "moderation report" in doc.text
    assert "regulator published" in doc.text  # the quote is part of what the post says
    assert doc.winner == "Bluesky"


@respx.mock
def test_a_cited_post_that_is_gone_is_unverified_with_the_platforms_own_words(
    client: Any,
) -> None:
    resolve_route()
    thread_route("bsky-not-found.json", status=400)
    provider = SocialProvider(client, StubFetcher())

    doc = provider.fetch(ref(HANDLE_URL), provider.resolve(ref(HANDLE_URL)))[0]

    assert doc.text_kind == "none"
    assert doc.state == Outcome.UNREACHABLE.value
    assert doc.title == "the post could not be read"
    assert doc.text == ""


@respx.mock
def test_a_post_with_no_text_is_not_a_post_that_could_not_be_reached(client: Any) -> None:
    hn_route(8863, "hn-story.json")
    provider = SocialProvider(client, StubFetcher())
    empty = respx.get(f"{HN_API}/item/9000.json").mock(
        return_value=httpx.Response(200, json={"id": 9000, "type": "story", "by": "x"})
    )
    url = "https://news.ycombinator.com/item?id=9000"

    doc = provider.fetch(ref(url), SocialProvider(client, StubFetcher()).resolve(ref(url)))[0]

    assert empty.call_count == 1
    assert (doc.text_kind, doc.state) == ("none", NO_TEXT)


@respx.mock
def test_a_platform_this_version_cannot_read_still_goes_up_the_ladder(client: Any) -> None:
    """v0.3.0 read a subreddit's front page as an ordinary page. Answering
    "unreachable" instead would report a page nobody tried to fetch as one that
    failed -- and a front page is not one post with one set of links."""
    reddit = "https://old.reddit.com/r/science/"
    fetcher = StubFetcher(fetched(reddit, FULLTEXT))
    provider = SocialProvider(client, fetcher)

    doc = provider.fetch(ref(reddit), provider.resolve(ref(reddit)))[0]

    assert fetcher.calls == [reddit]
    assert (doc.text_kind, doc.state) == ("fulltext", "")
    assert doc.winner == "httpx"


# --- routing -------------------------------------------------------------------


def test_a_numbered_entry_that_prints_a_post_is_still_read_as_a_post() -> None:
    """``provider_for`` routes on the entry and keeps the printed ``[n] `` marker, so
    a numbered entry goes to the indexes exactly as v0.3.0 sent it. It is
    ``reader_for`` that puts the post back on the social family once the resolving
    stage has placed it under its own address — the marker never has to be stripped,
    and no academic entry changes route."""
    providers = built(social=object())
    assert providers.social is not None
    assert provider_for(ref(f"[3] {HANDLE_URL}"), providers) is providers.academic
    assert provider_for(ref(HANDLE_URL), providers) is providers.social
    assert reader_for(f"url:{HANDLE_URL}", providers.academic, providers) is providers.social
    assert reader_for(f"url:{HN_STORY_URL}", providers.web, providers) is providers.social
    # Everything that is not a post is still the web family's.
    assert reader_for("url:https://example.test/x", providers.academic, providers) is providers.web


# --- end to end through verify and the CLI --------------------------------------


@respx.mock
def test_a_post_is_verified_against_the_pages_its_links_point_at() -> None:
    """The whole point of spec section 6.2: the post is the claim, the links are the
    sources, and every link is read up the ordinary fetch ladder."""
    resolve_route()
    thread_route()
    fetcher = VerifyStubFetcher(
        {
            REPORT_LINK: verify_fetched(REPORT_LINK, FULLTEXT),
            APPENDIX_LINK: verify_fetched(APPENDIX_LINK, FULLTEXT),
            REGULATOR_LINK: verify_fetched(REGULATOR_LINK, FULLTEXT),
        }
    )
    built = build_engine(resolver=VerifyStubResolver(), fetcher=fetcher)

    ready = prepare(HANDLE_URL, built)

    assert ready.document.kind == "post"
    assert {stage.name: stage.by for stage in ready.stages}["Parsing"] == POST_READER
    # Every link was asked of the ladder, and nothing was asked of the indexes.
    assert sorted(fetcher.calls) == sorted([REPORT_LINK, APPENDIX_LINK, REGULATOR_LINK])
    assert [status.source_id for status in ready.sources.values()] == [
        f"url:{REPORT_LINK}",
        f"url:{APPENDIX_LINK}",
        f"url:{REGULATOR_LINK}",
    ]
    assert all(status.text_kind == "fulltext" for status in ready.sources.values())


@respx.mock
def test_a_post_whose_link_cannot_be_read_is_never_counted_as_verified() -> None:
    """Product rules 2 and 6: the post still has its claims, and the source behind
    them is reported in the ladder's own words instead of quietly passing."""
    resolve_route()
    thread_route()
    built = build_engine(resolver=VerifyStubResolver(), fetcher=VerifyStubFetcher({}))

    ready = prepare(HANDLE_URL, built)

    assert ready.claims.claims  # the post does make claims
    assert ready.texts == {}  # and not one of them has a passage behind it
    assert all(status.text_kind == "none" for status in ready.sources.values())
    unverified = [finding for finding in ready.findings if finding.kind is Kind.UNVERIFIED]
    assert len(unverified) == 3
    assert {finding.state for finding in unverified} == {Outcome.UNREACHABLE.value}
    fetching = {stage.name: stage.summary for stage in ready.stages}["Fetching"]
    assert fetching == "0 full text, 0 abstract, 3 unverified"


@respx.mock
def test_a_post_that_links_to_nothing_is_reported_rather_than_read_as_clean() -> None:
    """A post with no link has nothing behind it. Without the note, a run over one
    would print a report with no findings, which is what a clean run looks like."""
    respx.get(f"{HN_API}/item/9001.json").mock(
        return_value=httpx.Response(
            200, json={"id": 9001, "type": "story", "by": "someone", "title": "A feeling."}
        )
    )
    built = build_engine(resolver=VerifyStubResolver(), fetcher=VerifyStubFetcher({}))

    ready = prepare("https://news.ycombinator.com/item?id=9001", built)

    assert ready.document.references == ()
    assert ready.claims.claims == ()
    notes = [finding for finding in ready.findings if finding.kind is Kind.PARSE_ERROR]
    assert [finding.title for finding in notes] == [
        "post carries no links; nothing to verify against"
    ]


def test_check_takes_exactly_one_of_a_target_and_a_url() -> None:
    assert runner.invoke(app, ["check"]).exit_code == 2
    both = runner.invoke(app, ["check", "-", "--url", HANDLE_URL])
    assert both.exit_code == 2
    assert "exactly one" in both.output


def test_check_url_on_a_page_that_is_not_a_post_says_what_would_work() -> None:
    result = runner.invoke(app, ["check", "--url", "https://example.test/report.html"])

    assert result.exit_code == 2
    assert "is a page, not a post" in result.output
    assert "give the claim as text with the address inside it" in result.output
    assert "paste it in the TUI, or proofpath check - on the command line" in result.output


# --- fix round 1: the network permission, blocked posts, unread quotes ------------


@respx.mock
def test_a_denied_run_reads_no_post_at_all_and_says_so() -> None:
    """Spec section 7.1 and product rule 4: the permission is consulted before
    anything is read, and a post is read over the network like everything else."""
    handle = resolve_route()
    thread = thread_route()
    built_engine = build_engine(
        resolver=VerifyStubResolver(),
        fetcher=VerifyStubFetcher({}, network_allowed=False, network_note="network: deny"),
    )

    refused: Exception | None = None
    try:
        prepare(HANDLE_URL, built_engine)
    except ingest.IngestError as error:
        refused = error

    assert handle.call_count == 0, "resolveHandle was called before the permission was read"
    assert thread.call_count == 0, "getPostThread was called before the permission was read"
    # The denied run's own words, not a second wording for the same fact.
    assert refused is not None, "a denied run read nothing and said nothing"
    assert Outcome.NETWORK_DENIED.value in str(refused)
    assert HANDLE_URL in str(refused)


def test_check_url_on_a_page_on_a_platform_still_says_it_is_not_a_post() -> None:
    """A profile on a platform that *is* read keeps the other answer: there is one
    post to check at a post address and none at a profile or a subreddit."""
    for url in ("https://bsky.app/profile/bsky.app", "https://old.reddit.com/r/science/"):
        result = runner.invoke(app, ["check", "--url", url])
        assert result.exit_code == 2, url
        assert "is a page, not a post" in result.output, url


@respx.mock
def test_a_blocked_bluesky_post_is_blocked_and_not_merely_unreachable(client: Any) -> None:
    """Spec section 15 keeps the two apart: "nobody may read this" is a different
    fact about a live post from "there is no such post" (product rule 2)."""
    resolve_route()
    thread_route("bsky-blocked.json")

    unreadable = read_post(HANDLE_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.BLOCKED.value
    assert "not found" not in unreadable.hint


@respx.mock
def test_a_bluesky_post_the_api_says_is_gone_stays_unreachable(client: Any) -> None:
    resolve_route()
    thread_route("bsky-thread-gone.json")

    unreadable = read_post(HANDLE_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNREACHABLE.value


@respx.mock
def test_a_quoted_post_that_could_not_be_read_is_named_rather_than_dropped(
    client: Any,
) -> None:
    """Product rule 2: the document loses that paragraph and its links either way,
    so the run has to say which quote it could not read."""
    resolve_route()
    thread_route("bsky-quote-blocked.json")

    post = read_post(HANDLE_URL, client)

    assert isinstance(post, Post)
    assert post.quoted == ()
    assert post.notes and "quote" in " ".join(post.notes)
    # The document keeps the quoter's own words; what is gone is the quote's.
    assert len(ingest.from_post(post)[0].paragraphs) == 1


@respx.mock
def test_a_run_over_a_post_with_an_unread_quote_reports_it() -> None:
    resolve_route()
    thread_route("bsky-quote-blocked.json")
    built_engine = build_engine(resolver=VerifyStubResolver(), fetcher=VerifyStubFetcher({}))

    ready = prepare(HANDLE_URL, built_engine)

    notes = [finding for finding in ready.findings if finding.kind is Kind.PARSE_ERROR]
    assert notes and "quote" in notes[0].title


def test_sarif_names_where_the_document_came_from_not_what_it_said(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The artifact URI is a location, so a document read from stdin is ``-``.
    Putting the pasted text there percent-encodes the whole document into the log."""
    from tests.test_check_cli import TROUBLED_BODY, draft, install

    monkeypatch.chdir(tmp_path)
    install(monkeypatch)
    result = runner.invoke(
        app, ["check", "-", "--format", "sarif"], input=draft(TROUBLED_BODY).encode("utf-8")
    )

    assert result.exit_code == 1, result.output
    location = json.loads(result.stdout)["runs"][0]["results"][0]["locations"][0]
    assert location["physicalLocation"]["artifactLocation"]["uri"] == "-"


@respx.mock
def test_a_killed_hn_item_is_blocked_where_a_deleted_one_is_unreachable(client: Any) -> None:
    """The export names ``dead`` and ``deleted`` separately, so this reports them
    separately: one item is still held and not shown, the other is gone."""
    hn_route(9002, "hn-dead.json")
    hn_route(8999, "hn-deleted.json")

    killed = read_post("https://news.ycombinator.com/item?id=9002", client)
    deleted = read_post("https://news.ycombinator.com/item?id=8999", client)

    assert isinstance(killed, Unreadable) and killed.state == Outcome.BLOCKED.value
    assert isinstance(deleted, Unreadable) and deleted.state == Outcome.UNREACHABLE.value


def test_only_the_platforms_with_a_reader_are_read_here() -> None:
    """A profile and a subreddit are read here too: the platform has a reader, and
    "that address is not one post" is the answer, not "nobody reads this"."""
    for url in (
        HANDLE_URL,
        "https://bsky.app/profile/bsky.app",
        HN_STORY_URL,
        "https://old.reddit.com/r/x/comments/1/y/",
        "https://www.reddit.com/r/science/",
        MASTODON_URL,
        # An instance's own pages, and one instance the routing host test misses.
        "https://mastodon.social/explore",
        "https://hachyderm.io/users/someone/statuses/109384756",
    ):
        assert is_read_here(url), url
    assert not is_read_here("https://example.test/report.html")


def test_check_url_on_an_instances_front_page_says_it_is_not_a_post() -> None:
    """Mastodon is read now, so its front page is a page and not a platform nobody
    reads. The two refusals say different things and only one of them is true."""
    result = runner.invoke(app, ["check", "--url", "https://mastodon.social/explore"])

    assert result.exit_code == 2
    assert "is a page, not a post" in result.output


# --- fix round 2: a refused post, a text-less quoter, "[n]" in a post's words -------


@respx.mock
def test_a_post_the_platform_refuses_to_serve_is_blocked_not_unreachable(client: Any) -> None:
    """Spec section 15 reserves BLOCKED for "403/bot protection", and 401 and 403 are
    the API saying the post is there and this reader may not have it. Filing that as
    "not found" reports a live post as one that may not exist (product rule 2)."""
    resolve_route()
    thread = respx.get(f"{BSKY_API}/app.bsky.feed.getPostThread")

    for status in (401, 403):
        thread.mock(return_value=httpx.Response(status, json={"error": "Forbidden"}))

        unreadable = read_post(HANDLE_URL, client)

        assert isinstance(unreadable, Unreadable), status
        assert unreadable.state == Outcome.BLOCKED.value, status
        assert "not found" not in unreadable.hint, status


def test_a_text_less_quoting_post_never_lends_its_links_to_the_quoted_author() -> None:
    """``recordWithMedia#view`` carries a quote and a link card at once, and Bluesky
    allows the quoter to write nothing at all. Its link then has no paragraph of its
    own, and filing it under the quoted post's would back that author's sentences
    with a source they never posted (product rule 1). Uncited is the honest answer."""
    quoted = Post(
        platform="Bluesky", url=QUOTE_URL, author="press.example.test",
        text="The regulator published its own tally this morning.",
        links=(REGULATOR_LINK,), quoted=(), fetched_at="2026-09-16T09:00:00+00:00",
    )  # fmt: skip
    quoter = Post(
        platform="Bluesky", url=HANDLE_URL, author="bsky.app", text="",
        links=(REPORT_LINK,), quoted=(quoted,), fetched_at="2026-09-16T09:00:00+00:00",
    )  # fmt: skip

    document, _ = ingest.from_post(quoter)
    claims = extract(document)

    # Both links are still listed: nothing the post cited disappears (product rule 6).
    assert [reference.raw for reference in document.references] == [REPORT_LINK, REGULATOR_LINK]
    assert len(document.paragraphs) == 1
    assert document.paragraphs[0].text.startswith("The regulator published")
    # The quoted author's sentences cite the quoted post's link and nothing else.
    assert claims.claims and all(claim.cited_refs == (2,) for claim in claims.claims)


def test_a_bracketed_number_in_a_posts_words_still_leaves_its_links_paired() -> None:
    """A post prints no numbered bibliography, so "[2024]" in it names nothing. It
    used to switch the document onto the numbered branch, which lost every link
    pairing and left a post that cites reading as a post that cites nothing."""
    post = Post(
        platform="Bluesky", url=HANDLE_URL, author="bsky.app",
        text="The [2024] tally rose 17%. The method is written up in the appendix.",
        links=(REPORT_LINK,), quoted=(), fetched_at="2026-09-16T09:00:00+00:00",
    )  # fmt: skip

    claims = extract(ingest.from_post(post)[0])

    assert claims.claims and all(claim.cited_refs == (1,) for claim in claims.claims)


# --- Task 10.3: Reddit, Mastodon, X -----------------------------------------------


REDDIT_POST_URL = (
    "https://www.reddit.com/r/science/comments/1abcdef/our_2025_moderation_report_is_out/"
)
REDDIT_COMMENT_URL = f"{REDDIT_POST_URL}c0ffee1/"
REDDIT_LINK_POST_URL = (
    "https://old.reddit.com/r/science/comments/2linkid/air_quality_guidance_was_revised_this_week/"
)
REDDIT_REMOVED_URL = (
    "https://www.reddit.com/r/science/comments/3removed/our_2025_moderation_report_is_out/"
)
MASTODON_URL = "https://mastodon.social/@someone/109384756"
MASTODON_API = "https://mastodon.social/api/v1/statuses/109384756"
X_URL = "https://x.com/someone/status/1234567890"

FAKE_ID = "fake-client-id"
FAKE_SECRET = "fake-client-secret-never-printed"


@pytest.fixture
def no_reddit_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """A machine with no Reddit app registered, whatever the developer's own .env says."""
    monkeypatch.delenv(REDDIT_CLIENT_ID_ENV, raising=False)
    monkeypatch.delenv(REDDIT_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.setattr(secrets_mod, "default_dotenv_paths", list)


@pytest.fixture
def reddit_app(monkeypatch: pytest.MonkeyPatch) -> None:
    """The user's own free app, as two environment variables and nothing else."""
    monkeypatch.setenv(REDDIT_CLIENT_ID_ENV, FAKE_ID)
    monkeypatch.setenv(REDDIT_CLIENT_SECRET_ENV, FAKE_SECRET)
    monkeypatch.setattr(secrets_mod, "default_dotenv_paths", list)


def token_route(status: int = 200) -> Any:
    return respx.post(REDDIT_TOKEN_URL).mock(
        return_value=httpx.Response(status, json=fixture("reddit-token.json"))
    )


def reddit_route(article: str, name: str, status: int = 200) -> Any:
    return respx.get(f"{REDDIT_API}/comments/{article}").mock(
        return_value=httpx.Response(status, json=fixture(name))
    )


def mastodon_route(name: str, status: int = 200) -> Any:
    return respx.get(MASTODON_API).mock(return_value=httpx.Response(status, json=fixture(name)))


def test_the_three_platforms_task_103_adds_are_posts_this_version_answers_for() -> None:
    """Reddit, Mastodon and X each have a reader now. X's answer is that it cannot be
    read -- which is still an answer about that post, not a silence."""
    for url in (REDDIT_POST_URL, REDDIT_COMMENT_URL, MASTODON_URL, X_URL):
        assert is_post_url(url), url
        assert is_read_here(url), url
    # A subreddit, a profile and an instance's front page are not single posts.
    for url in (
        "https://old.reddit.com/r/science/",
        "https://www.reddit.com/user/someone",
        "https://mastodon.social/@someone",
        "https://x.com/someone",
    ):
        assert not is_post_url(url), url


# --- Reddit -------------------------------------------------------------------


@respx.mock
def test_a_reddit_post_is_read_with_the_users_own_app_and_carries_its_links(
    client: Any, reddit_app: None
) -> None:
    token = token_route()
    article = reddit_route("1abcdef.json", "reddit-post.json")

    post = read_post(REDDIT_POST_URL, client)

    assert isinstance(post, Post)
    assert token.call_count == 1
    # Client credentials: the grant in the body, the two values in HTTP basic auth.
    assert b"grant_type=client_credentials" in token.calls.last.request.content
    assert token.calls.last.request.headers["Authorization"].startswith("Basic ")
    assert article.calls.last.request.url.params["raw_json"] == "1"
    assert article.calls.last.request.headers["Authorization"] == "bearer TOKEN-FROM-THE-FIXTURE"
    assert post.platform == "Reddit"
    assert post.author == "someone"
    assert post.text.startswith("Our 2025 moderation report is out")
    assert "The tally is in" in post.text
    # The markdown link first, then the bare address: order of appearance.
    assert post.links == (REPORT_LINK, APPENDIX_LINK)
    assert post.quoted == ()


@respx.mock
def test_a_reddit_link_post_cites_the_address_it_submitted(client: Any, reddit_app: None) -> None:
    token_route()
    reddit_route("2linkid.json", "reddit-link-post.json")

    post = read_post(REDDIT_LINK_POST_URL, client)

    assert isinstance(post, Post)
    assert post.links == (HN_STORY_LINK,)


@respx.mock
def test_a_self_post_never_cites_its_own_permalink(client: Any, reddit_app: None) -> None:
    """``url`` on a self post is the post itself. Listing it would make every text
    post look like it cited a source, and send the ladder after its own page."""
    token_route()
    reddit_route("1abcdef.json", "reddit-post.json")

    post = read_post(REDDIT_POST_URL, client)

    assert isinstance(post, Post)
    assert not any("reddit.com" in link for link in post.links)


@respx.mock
def test_a_reddit_comment_permalink_reads_that_comment_and_not_the_thread(
    client: Any, reddit_app: None
) -> None:
    token_route()
    article = reddit_route("1abcdef.json", "reddit-comment.json")

    post = read_post(REDDIT_COMMENT_URL, client)

    assert isinstance(post, Post)
    assert article.calls.last.request.url.params["comment"] == "c0ffee1"
    assert post.author == "reader"
    assert post.text.startswith("The paper is at")
    assert post.links == (PAPER_LINK, DATA_LINK)


@respx.mock
def test_a_comment_reddit_chose_to_nest_is_still_the_comment_that_was_cited(
    client: Any, reddit_app: None
) -> None:
    """``context=0&depth=1`` is a request, not a guarantee. A live comment returned
    one level down is a comment that exists, and reporting it as one that may not
    would be absence taken for evidence (product rule 2)."""
    token_route()
    reddit_route("1abcdef.json", "reddit-nested-comment.json")

    post = read_post(REDDIT_COMMENT_URL, client)

    assert isinstance(post, Post)
    assert post.author == "reader"
    assert post.links == (PAPER_LINK, DATA_LINK)


@respx.mock
def test_a_comment_the_listing_never_carried_is_unavailable_not_unreachable(
    client: Any, reddit_app: None
) -> None:
    """Reddit answered in a shape this reader could not use, which says nothing about
    whether the comment is there. ``UNREACHABLE`` would say it may not be."""
    token_route()
    reddit_route("1abcdef.json", "reddit-comment.json")

    unreadable = read_post(f"{REDDIT_POST_URL}deadbee/", client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNAVAILABLE.value
    assert unreadable.state != Outcome.UNREACHABLE.value
    assert unreadable.hint == COMMENT_NOT_LISTED


@respx.mock
def test_reddit_without_credentials_is_reported_and_nobody_is_asked(
    client: Any, no_reddit_app: None
) -> None:
    """Product rule 2: a source nobody asked for is not a source that failed. The
    state is its own, and it names the two variables that would fix it."""
    unreadable = read_post(REDDIT_POST_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == CREDENTIALS_MISSING
    assert unreadable.state not in {Outcome.UNREACHABLE.value, Outcome.BLOCKED.value}
    assert REDDIT_CLIENT_ID_ENV in unreadable.hint
    assert REDDIT_CLIENT_SECRET_ENV in unreadable.hint
    assert respx.calls.call_count == 0, "a run with no credentials called Reddit anyway"


@respx.mock
def test_half_a_reddit_app_is_still_no_app(client: Any, monkeypatch: Any) -> None:
    """An id with no secret cannot authenticate. Sending it would leak the id to
    Reddit for nothing and report the refusal as a fact about the post."""
    monkeypatch.setenv(REDDIT_CLIENT_ID_ENV, FAKE_ID)
    monkeypatch.delenv(REDDIT_CLIENT_SECRET_ENV, raising=False)
    monkeypatch.setattr(secrets_mod, "default_dotenv_paths", list)

    unreadable = read_post(REDDIT_POST_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == CREDENTIALS_MISSING
    assert respx.calls.call_count == 0


@respx.mock
def test_reddit_credentials_are_never_printed_in_a_state_a_hint_or_a_repr(
    client: Any, reddit_app: None
) -> None:
    """A rejected app is a refusal by the platform, and the refusal must not carry
    the secret it was refused with."""
    token_route(status=401)

    unreadable = read_post(REDDIT_POST_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.BLOCKED.value
    for text in (unreadable.hint, unreadable.state, repr(unreadable)):
        assert FAKE_SECRET not in text
        assert FAKE_ID not in text


@respx.mock
def test_a_removed_reddit_post_is_blocked_and_a_missing_one_is_unreachable(
    client: Any, reddit_app: None
) -> None:
    """The same distinction Hacker News's ``dead``/``deleted`` gets: a post held back
    by moderation exists and is withheld; a 404 may be a post that never was."""
    token_route()
    reddit_route("3removed.json", "reddit-removed.json")
    respx.get(f"{REDDIT_API}/comments/9nothere.json").mock(return_value=httpx.Response(404))

    removed = read_post(REDDIT_REMOVED_URL, client)
    missing = read_post("https://www.reddit.com/r/science/comments/9nothere/gone/", client)

    assert isinstance(removed, Unreadable) and removed.state == Outcome.BLOCKED.value
    assert isinstance(missing, Unreadable) and missing.state == Outcome.UNREACHABLE.value


@respx.mock
def test_a_reddit_outage_is_unavailable_not_unreachable(client: Any, reddit_app: None) -> None:
    token_route()
    respx.get(f"{REDDIT_API}/comments/1abcdef.json").mock(
        return_value=httpx.Response(503, json={"error": 503})
    )

    unreadable = read_post(REDDIT_POST_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNAVAILABLE.value


# --- Mastodon -----------------------------------------------------------------


@respx.mock
def test_a_mastodon_status_is_read_from_the_instance_that_hosts_it(client: Any) -> None:
    route = mastodon_route("mastodon-status.json")

    post = read_post(MASTODON_URL, client)

    assert isinstance(post, Post)
    assert route.call_count == 1
    assert post.platform == "Mastodon"
    assert post.author == "someone"
    # The HTML is stripped to the author's words, paragraph breaks kept.
    assert post.text.startswith("Our 2025 moderation report is out.")
    assert "<p>" not in post.text and "&amp;" not in post.text
    assert "its own tally & we agree" in post.text
    # The anchor first, then the preview card: order of appearance.
    assert post.links == (REGULATOR_LINK, REPORT_LINK)


@respx.mock
def test_a_mastodon_instance_that_requires_a_login_is_blocked_not_unreachable(
    client: Any,
) -> None:
    """Spec section 15: the status is there and this reader may not have it."""
    for status in (401, 403, 422):
        respx.calls.reset()
        mastodon_route("mastodon-login-required.json", status=status)

        unreadable = read_post(MASTODON_URL, client)

        assert isinstance(unreadable, Unreadable), status
        assert unreadable.state == Outcome.BLOCKED.value, status
        assert unreadable.hint == MASTODON_LOGIN_HINT, status


@respx.mock
def test_a_mastodon_status_that_is_gone_is_unreachable(client: Any) -> None:
    mastodon_route("mastodon-login-required.json", status=404)

    unreadable = read_post(MASTODON_URL, client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == Outcome.UNREACHABLE.value


@respx.mock
def test_a_mastodon_status_is_read_on_any_instance_not_a_list_of_them(client: Any) -> None:
    """There is no list of Mastodon instances to keep, only the shape they share."""
    route = respx.get("https://hachyderm.io/api/v1/statuses/109384756").mock(
        return_value=httpx.Response(200, json=fixture("mastodon-status.json"))
    )

    post = read_post("https://hachyderm.io/users/someone/statuses/109384756", client)

    assert isinstance(post, Post)
    assert route.call_count == 1


@respx.mock
def test_a_mastodon_address_with_userinfo_never_hands_it_to_the_instance(client: Any) -> None:
    """``netloc`` carries ``user@host``, and a client handed that derives a basic-auth
    header from it -- a credential nobody had, sent to an instance that never asked."""
    route = mastodon_route("mastodon-status.json")

    post = read_post("https://someone@mastodon.social/@someone/109384756", client)

    assert isinstance(post, Post)
    assert route.call_count == 1
    assert "Authorization" not in route.calls.last.request.headers


@respx.mock
def test_a_mastodon_instance_on_its_own_port_keeps_the_port(client: Any) -> None:
    route = respx.get("https://mastodon.example:8443/api/v1/statuses/109384756").mock(
        return_value=httpx.Response(200, json=fixture("mastodon-status.json"))
    )

    post = read_post("https://mastodon.example:8443/@someone/109384756", client)

    assert isinstance(post, Post)
    assert route.call_count == 1


@respx.mock
def test_a_subdomain_of_a_host_read_here_is_never_read_as_a_mastodon_instance(
    client: Any,
) -> None:
    """``www.bsky.app`` is Bluesky however the path is shaped. Falling through to the
    Mastodon reader would ask Bluesky for a status it has no endpoint for."""
    unreadable = read_post("https://www.bsky.app/@someone/109384756", client)

    assert isinstance(unreadable, Unreadable)
    assert unreadable.state == NOT_READ_HERE
    assert unreadable.hint == UNSUPPORTED_HINT
    assert respx.calls.call_count == 0


# --- X ------------------------------------------------------------------------


@respx.mock
def test_an_x_status_says_it_cannot_be_read_and_asks_nobody(client: Any) -> None:
    """X has no read-only API. Saying "unreachable" would be a claim about the post
    rather than about this tool, and the user's own paste still verifies the links."""
    for url in (X_URL, "https://twitter.com/someone/status/1234567890"):
        unreadable = read_post(url, client)

        assert isinstance(unreadable, Unreadable), url
        assert unreadable.platform == "X"
        assert unreadable.state == Outcome.BLOCKED.value
        assert unreadable.hint == X_HINT
    assert respx.calls.call_count == 0


def test_check_url_on_an_x_status_says_to_paste_the_text_instead() -> None:
    result = runner.invoke(app, ["check", "--url", X_URL])

    assert result.exit_code == 2
    assert "paste the post text" in result.output


def test_check_url_on_a_reddit_post_names_the_two_variables(no_reddit_app: None) -> None:
    result = runner.invoke(app, ["check", "--url", REDDIT_POST_URL])

    assert result.exit_code == 2
    assert REDDIT_CLIENT_ID_ENV in result.output
    assert REDDIT_CLIENT_SECRET_ENV in result.output


# --- the coverage block a missing credential must reach ------------------------


@respx.mock
def test_a_cited_reddit_post_with_no_app_is_counted_under_its_own_reason(
    no_reddit_app: None, client: Any
) -> None:
    """Product rule 6 on its main surface: the run says why the source is unread,
    in a word that is neither "unreachable" nor "blocked"."""
    provider = SocialProvider(client, StubFetcher())

    doc = provider.fetch(ref(REDDIT_POST_URL), provider.resolve(ref(REDDIT_POST_URL)))[0]

    assert (doc.text_kind, doc.state) == ("none", CREDENTIALS_MISSING)
    assert doc.notes and REDDIT_CLIENT_ID_ENV in doc.notes[0]
    assert respx.calls.call_count == 0
