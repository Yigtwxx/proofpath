/* The six product rules from CLAUDE.md, in the order the page tells them.
   `raven` ties a rule to the bird whose job it constrains; null for rules about
   the tool's manners rather than its findings. */
export interface Rule {
    id: string;
    title: string;
    body: string;
    raven: 'huginn' | 'muninn' | null;
    /** Short line the margin note quotes. */
    margin: string;
}

export const rules: readonly Rule[] = [
    {
        id: 'passage',
        title: 'No passage, no verdict.',
        body: 'A *SUPPORTED* or *NOT SUPPORTED* verdict always carries the quoted sentence it rests on. If the passage cannot be shown, *the verdict is not given*.',
        raven: 'muninn',
        margin: 'Muninn shows you what he read.',
    },
    {
        id: 'absence',
        title: 'Absence of evidence is not evidence of absence.',
        body: '*Unreachable*, *blocked*, *paywalled* and *ambiguous* are separate, reported states. None of them is quietly folded into a verdict.',
        raven: 'muninn',
        margin: 'A raven that comes back empty says "nothing found", not "nothing exists".',
    },
    {
        id: 'ghost',
        title: 'A real reference is never called a ghost.',
        body: 'When the indexes disagree or the match is uncertain, the finding is *AMBIGUOUS*. *GHOST REFERENCE* is reserved for a record no index has ever seen. The false-ghost rate is a *release gate*.',
        raven: 'huginn',
        margin: 'Huginn doubts before he accuses.',
    },
    {
        id: 'tty',
        title: 'No prompt without a terminal.',
        body: 'In CI or behind a pipe, a permission set to *ask* is treated as *deny*, and the report says so. A job never hangs waiting for a human who is not there.',
        raven: null,
        margin: 'Piped means quiet.',
    },
    {
        id: 'consent',
        title: 'Nothing large is installed without asking.',
        body: 'The browser fetcher and the GPU extra are *opt-in*. The base install stays small enough to put in a pipeline without thinking about it.',
        raven: null,
        margin: 'The base install is the whole install.',
    },
    {
        id: 'coverage',
        title: 'Every report states its own coverage.',
        body: 'How much was read as *full text*, how much as *abstract only*, how much *not at all* — and why. A run that read little cannot look like a run that read everything.',
        raven: 'huginn',
        margin: 'A clean report on 40% coverage is not a clean report.',
    },
] as const;
