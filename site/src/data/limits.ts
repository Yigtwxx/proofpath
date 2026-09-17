/* README "Known limitations", one row per open item, with the run that found
   it where there is one. The judge row is left out: Judge.astro says it. */
export interface Limit {
    id: string;
    title: string;
    /** Passed through mark(): *word* is emphasis. */
    body: string;
    /** Path under docs/eval, when a run documents the limit. */
    doc?: string;
}

export const limits: readonly Limit[] = [
    {
        id: 'superscript',
        title: 'Superscript citations only when the PDF marks them.',
        body: 'A PDF that draws them as ordinary digits loses them, and *km²* can be read as *[2]*.',
    },
    {
        id: 'abstract',
        title: 'Three sentences is not a source.',
        body: 'When only an abstract is reachable the verdict is labelled *LOW CONFIDENCE (abstract only)* and counts as a finding.',
    },
    {
        id: 'two-column',
        title: 'An unnumbered two-column bibliography is cut at line breaks.',
        body: "RoBERTa's ACL list came out as 103 entries for about 50, half with no author, so author-year pairing resolved *15 claims* where a rejoined list would give 65. The largest open item.",
    },
    {
        id: 'heading',
        title: 'Without a References heading, only the last block is read.',
        body: 'A paged document is read by its shape — the last contiguous run of numbered paragraphs — which recovers that block alone (AlphaFold: *17 of 84*).',
    },
    {
        id: 'ghost-recall',
        title: 'The 0 % false-ghost rate is paid for in recall.',
        body: 'A fabricated entry with no year between the names and the title is reported as *not in bibliographic indexes*, not as a ghost, and one proceedings-volume record accepted a fabricated paper cited into it (0.9 %).',
        doc: '2026-09-12-ghosts.md',
    },
    {
        id: 'serial',
        title: 'A cold run of a long bibliography is serial.',
        body: 'The providers are asked one reference at a time (129 references: minutes). Only the TUI runs several documents’ network stages at once.',
    },
    {
        id: 'browser',
        title: 'A refused browser install is reported, not hidden.',
        body: 'The source is *UNVERIFIED (blocked, browser not permitted)* with the install log, never silently counted as unreachable.',
    },
    {
        id: 'coverage',
        title: 'Coverage is not perfectly reproducible.',
        body: 'Two runs minutes apart can read a different number of sources, depending on which providers answered.',
    },
];
