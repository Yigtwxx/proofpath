/* README "What the states mean", one row per printed state. Every one of these
   is reported as itself (spec §15); none is folded into a verdict. The reserved
   UNSUPPORTED CITATION STYLE is left out: no detected style produces it. */
export interface State {
    name: string;
    cause: string;
}

export const states: readonly State[] = [
    { name: 'LOW CONFIDENCE (abstract only)', cause: 'full text unavailable, abstract used' },
    { name: 'UNVERIFIED (blocked)', cause: '403 or bot protection' },
    {
        name: 'UNVERIFIED (blocked, robots.txt)',
        cause: "the site's robots.txt disallows the fetch",
    },
    {
        name: 'UNVERIFIED (blocked, browser not permitted)',
        cause: 'steps 1–2 blocked and the browser consent was denied or impossible',
    },
    { name: 'UNVERIFIED (unreachable)', cause: 'dead link, Wayback miss' },
    {
        name: 'UNVERIFIED (reached, no text extracted)',
        cause: '200 answered, nothing readable came back',
    },
    { name: 'UNVERIFIED (network not permitted)', cause: 'permissions.network = deny' },
    {
        name: 'UNVERIFIED (provider unavailable)',
        cause: 'API down, rate limited after backoff, or answering with a page instead of a record',
    },
    {
        name: 'UNVERIFIED (credentials missing)',
        cause: 'the platform reads only with a credential this machine lacks (Reddit: REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET) — nobody was asked, so it is not unreachable',
    },
    {
        name: 'UNVERIFIED (not in bibliographic indexes)',
        cause: 'web page, blog, report — the indexes do not cover it, so absence proves nothing',
    },
    { name: 'AMBIGUOUS', cause: 'several plausible records, all listed' },
    { name: 'NEI', cause: 'the source was read and neither supports nor contradicts' },
    {
        name: 'PARAGRAPH-SCOPED',
        cause: 'the citation covers a paragraph; each sentence is judged separately',
    },
];
