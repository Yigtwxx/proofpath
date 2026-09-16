/* Coverage as proofpath reports it: the share of sources read as full text,
   abstract only, or not at all, with the reasons for the last group. */
export interface Coverage {
    full: number;
    abstract: number;
    unverified: number;
    reasons?: readonly { label: string; count: number }[];
}

/** The spec's worked example, with reasons. */
export const specCoverage: Coverage = {
    full: 62,
    abstract: 21,
    unverified: 17,
    reasons: [
        { label: 'blocked', count: 3 },
        { label: 'credentials missing', count: 1 },
    ],
};

export const weakCoverageLine =
    'coverage is weak: unread sources may hold more, so this is a lower bound';
