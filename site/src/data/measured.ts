/* The README's "Measured" table, one row per evaluation, each with the run
   that produced it. Numbers are copied, not rounded up; the AVeriTeC row is
   the one that says where the tool is weak, and it is kept for that reason. */
import { repo } from './commands';

export interface Measure {
    id: string;
    /** The figure set large. */
    figure: string;
    /** The unit or reading of the figure. */
    reading: string;
    what: string;
    set: string;
    /** Path under docs/eval. */
    doc: string;
    /** Second figure, when the row carries a comparison. */
    against?: string;
    /** A mono footnote under the row, for the caveat that belongs next to the number. */
    note?: string;
    /** True for the row that reports a weakness. */
    weak?: boolean;
}

export const measures: readonly Measure[] = [
    {
        id: 'ghosts',
        figure: '0.0 %',
        reading: 'false-ghost rate',
        against: '99.1 % ghost recall',
        what: 'Reference resolution',
        set: 'hand-built ghost set, 274 references',
        doc: '2026-09-12-ghosts.md',
    },
    {
        id: 'scifact',
        figure: '0.609',
        reading: 'accuracy · 0.597 macro-F1',
        against: 'against a 0.406 trivial baseline',
        what: 'Retrieval and entailment',
        set: 'SciFact dev, 340 pairs',
        doc: '2026-09-12-scifact-dev.md',
    },
    {
        id: 'pairing',
        figure: '0.99',
        reading: 'numeric · 0.940 author-year',
        what: 'Citation pairing',
        set: '61 and 55 hand-built passages',
        doc: '2026-09-12-pairing-author-year.md',
    },
    {
        id: 'speed',
        figure: '1.35 s',
        reading: 'second run · 79.3 s cold',
        against:
            'the second run asks the network for nothing; a 19-page arXiv PDF with 68 references: 12 m 57 s, then 3 m 42 s',
        what: 'Speed',
        set: '1-page draft, 7 references, Apple Silicon',
        doc: '2026-09-15-v0.2-live.md',
    },
    {
        id: 'averitec',
        figure: '0.270',
        reading: '3-way accuracy',
        against: 'against a 0.708 majority baseline — worse than always guessing "refuted"',
        what: 'End to end on real web claims',
        set: 'AVeriTeC dev, 100 claims',
        doc: '2026-09-16-averitec.md',
        weak: true,
    },
];

/** Source access on 50 DOIs, as the coverage bar draws it. */
export const accessCoverage = { full: 72, abstract: 18, unverified: 10 } as const;
export const accessDoc = '2026-09-11-coverage.md';

export const evalUrl = (doc: string): string => `${repo}/blob/main/docs/eval/${doc}`;

export const averitecNote =
    'A third of those claims had no readable source at all: 32 URLs needed the browser step, 29 were unreachable, 14 were refused by robots.txt. On the claims that did have one the score is 0.361 — still below the baseline. The models were calibrated on scientific abstracts; a fact-check page is a different object.';
