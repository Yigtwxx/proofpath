# A page as the target — "paste any link, it just works"

**Status:** approved 2026-09-17; implementation follows this document. Amends spec
§13 (`check --url`) of `2026-09-10-proofpath-design.md`. No honesty state changes;
one refusal (`NOT_A_POST`) narrows to the platforms it was written for.

## 1. Why

Pasting one address into the TUI bar, or `check --url`, routed it as a *social post*.
Only a Bluesky, Hacker News, Reddit or Mastodon post address passed; every other host
— a journal's article page, a news story, a blog, Wikipedia — was refused before any
request with `IngestError: <url> is a page, not a post …`, and the run showed as
*failed*. The bar's own placeholder says "paste a file path or URL", so the refusal
read as a bug. Links *inside* pasted text already worked for any site.

## 2. Decision (2026-09-17, with the user)

A bare address on a host that is not a platform is **the document itself**: the page
is read up the fetch ladder, its paragraphs are the claims, and its links (or the
reference list it prints) are its sources. The alternatives — ask for the claim after
fetching, or only reword the error — were offered and declined.

## 3. The changes

| Where | What |
|---|---|
| `verify._bare_address` / `_address_document` | one address → `_post_document` when `is_social(url)` **or** `social.is_read_here(url)` (a Mastodon status on an unlisted instance is known by shape), else `_page_document` |
| `verify._page_document` | network denied → refused in `NETWORK_DENIED`'s words; else `fetcher.fetch(url, counts_as_source=False, use_cache=False)`; a ladder failure → `IngestError("<url> could not be read: <Outcome>: <deciding step>")` — the Wayback miss every wall ends on is not the deciding step — plus `config set permissions.install_browser …` when the browser was the missing step; else `ingest.from_page` and, when the page prints no reference list, `ingest.with_carried_links`. Unlike `_pasted`, a citation marker alone does not decide it: "[1]" in one paragraph of a long page must not cost the page every link it carries, and a `linked` document reads a bracketed number as prose (`claims.extract`). The override is reported as a parse note (`MARKERS_SET_ASIDE`), so a list under a heading the finder does not know ("Sources") is not set aside silently (rule 6) |
| `fetch.Fetcher.fetch(use_cache=)` | a cache hit is text alone (`body=b""`); the page needs its markup because its links are its bibliography |
| `fetch.CONTENT_ROOTS` | the `article` → `main` → `body` choice, applied by `extract_text` (via `content_root`) and by the page reader on its own tree, so a source's text and a page's paragraphs are cut from one element |
| `ingest.from_page` → `Page` | HTML walked with lxml: one paragraph per block (`PAGE_BLOCK_TAGS`) that holds no other block; a container's words *between* its blocks (a `div` used as a paragraph, text around `<br>`) are paragraphs of their own; `aside`/`form`/`button`/`svg`/`template` contribute neither words nor links on top of the ladder's boilerplate. Each paragraph's `href`s are resolved against the address the page was served from (against the asked-for address for a Wayback copy, whose raw snapshot keeps links unrewritten) and kept per line, together with any address the paragraph prints in full. A link to the page itself — either address, query ignored (`_page_key`) — is dropped and named (`SELF_LINK_DROPPED`). PDF: `_pdf_document` from bytes, the same reader as a file. Plain text: `from_text` plus printed addresses. Anything else: refused by content type. `document.name` is the address asked for |
| `ingest.with_carried_links` | `with_link_references` for a page: the carried links, one reference per distinct address in order of first appearance, placed at the paragraph's first line (what `claims.pair_links` keys on); kind `linked` |
| `document.Kind` | `+ "page"` (own bibliography or nothing; **not** in `LINK_CITED`) |
| `verify.prepare` | "`<kind>` carries no links; nothing to verify against" for a post *or a page* with no references (rule 6) |
| `verify._parser_for` / stage line | a page opens and closes as `fetch ladder`; pasted text with links (kind `linked` too) still closes as `text` |

Unchanged: the ladder, the consent gate, permissions, per-site rules (none exist and
none are added). A profile, subreddit or platform front page keeps `NOT_A_POST`.

## 4. Product rules, checked

- Rule 2: every ladder outcome reaches the user verbatim; a blocked page ends the
  run, it never becomes a silent verdict.
- Rule 4: the network permission is read before the page is fetched, as for a post.
- Rule 6: a page with no links is reported, not passed as clean.
- Rule 1: a page's own address — as an anchor, with a tracking query, printed in
  full, or under the address a redirect or Wayback copy served it from — is never
  one of its sources.

## 5. Open

- No cap on link-heavy pages (a Wikipedia article can carry hundreds of links): every
  paragraph's links are fetched. Coverage is reported and the run can be cancelled.
  Recorded in OPEN-ITEMS.
- Scheme-less links (`nature.com/…`) in pasted text are still not references
  (`resolve._HTTP_URL`); separate.
