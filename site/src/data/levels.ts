/* The three questions proofpath asks of every citation, in order.
   Each level is a chapter on the page: which raven does the work, which
   services are consulted, what a finding at that level looks like. */
export type Verdict = {
    label: string;
    tone: 'ok' | 'bad' | 'slate' | 'ink';
};

export interface Finding {
    line: string;
    ref: string;
    verdict: Verdict;
    tier?: 'high' | 'medium' | 'low';
    note?: string;
    claim?: string;
    passage?: string;
}

export interface Level {
    n: 1 | 2 | 3;
    slug: 'exists' | 'valid' | 'supports';
    question: string;
    raven: 'huginn' | 'muninn';
    ravenDoes: string;
    sources: readonly string[];
    imageId: string;
    body: readonly string[];
    example: Finding;
    states: readonly { label: string; tone: Verdict['tone'] }[];
}

export const levels: readonly Level[] = [
    {
        n: 1,
        slug: 'exists',
        question: 'Does the source exist?',
        raven: 'huginn',
        ravenDoes:
            'Huginn takes the reference as written and asks five indexes whether they have ever seen it.',
        sources: ['Crossref', 'Semantic Scholar', 'arXiv', 'Open Library', 'OpenAlex'],
        imageId: 'bust',
        body: [
            'A reference is resolved against Crossref, Semantic Scholar, arXiv, Open Library and OpenAlex. A DOI that resolves, an arXiv id that matches, a title and author list that line up: that is a *RESOLVED* record.',
            'Several plausible records and no clear winner is *AMBIGUOUS*, and every candidate is listed. Only a reference that none of the indexes has seen is called a *GHOST REFERENCE* — and even then, the report names the indexes it asked before saying so.',
        ],
        example: {
            line: 'L41',
            ref: '[7] Marchetti, L. R., Osei, K. & Lind…',
            verdict: { label: 'GHOST REFERENCE', tone: 'bad' },
            note: 'arxiv and openlibrary consulted before the ghost call',
        },
        states: [
            { label: 'RESOLVED', tone: 'ok' },
            { label: 'AMBIGUOUS', tone: 'slate' },
            { label: 'GHOST REFERENCE', tone: 'bad' },
        ],
    },
    {
        n: 2,
        slug: 'valid',
        question: 'Is it still valid?',
        raven: 'huginn',
        ravenDoes:
            'Huginn checks whether the record has since been retracted, corrected or withdrawn.',
        sources: ['Crossref (Retraction Watch data)', 'OpenAlex'],
        imageId: 'door',
        body: [
            'A paper that exists can still have been *pulled*. Every resolved record is checked against the Retraction Watch data that Crossref carries, and against OpenAlex.',
            'A retraction is its own finding, not a footnote: *RETRACTED* is reported on the line where the citation sits, with the notice that says why.',
        ],
        example: {
            line: 'L18',
            ref: '[3] Wakefield, A. J. et al. Ileal-lymphoid…',
            verdict: { label: 'RETRACTED', tone: 'bad' },
            note: 'retraction notice 2010-02-02, The Lancet, via Crossref',
        },
        states: [
            { label: 'VALID', tone: 'ok' },
            { label: 'RETRACTED', tone: 'bad' },
        ],
    },
    {
        n: 3,
        slug: 'supports',
        question: 'Does it support the claim?',
        raven: 'muninn',
        ravenDoes:
            'Muninn reads the source, finds the passage closest to the sentence you wrote, and brings it back — word for word.',
        sources: ['full text or abstract', 'retrieval', 'a numeric rule', 'an entailment model'],
        imageId: 'lamp',
        body: [
            'The sentence that carries the citation is compared with the text of the source itself. The nearest passages are retrieved, *a numeric rule runs first* — if the claim says 21 days and the source says 3.5, no model is asked — and an entailment model judges the rest, on your machine.',
            '*SUPPORTED* and *NOT SUPPORTED* always show the passage they rest on. When the source neither supports nor contradicts, the finding is *NEI*, and it says so instead of guessing. Most tools stop at step 1. *This step is the point.*',
        ],
        example: {
            line: 'L6',
            ref: '[1] Vaswani, A., Shazeer, N., Parmar,…',
            verdict: { label: 'NOT SUPPORTED', tone: 'bad' },
            tier: 'high',
            note: 'numeric mismatch: claim says 21 days, source says 3.5 days',
            claim: 'On the WMT 2014 English-to-French translation task the Transformer reached…',
            passage: 'On the WMT 2014 English-to-French translation task, our model establishes …',
        },
        states: [
            { label: 'SUPPORTED', tone: 'ok' },
            { label: 'NOT SUPPORTED', tone: 'bad' },
            { label: 'NEI', tone: 'ink' },
            { label: 'LOW CONFIDENCE (abstract only)', tone: 'slate' },
        ],
    },
] as const;
