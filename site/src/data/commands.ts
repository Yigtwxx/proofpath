/* The one-shot surface, as the README shows it. */
export interface Command {
    cmd: string;
    comment: string;
}

export const install: readonly { label: string; cmd: string }[] = [
    { label: 'uv', cmd: 'uv tool install proofpath' },
    { label: 'pipx', cmd: 'pipx install proofpath' },
    { label: 'pip', cmd: 'pip install proofpath' },
];

export const commands: readonly Command[] = [
    { cmd: 'proofpath check paper.pdf', comment: 'report → report.md, exit 0/1/2' },
    { cmd: "proofpath check draft.md --format json | jq '.coverage'", comment: 'machine-readable' },
    {
        cmd: 'proofpath check draft.md --format sarif --out draft.sarif',
        comment: 'any SARIF 2.1.0 viewer',
    },
    { cmd: 'proofpath -q check - < draft.md', comment: 'stdin; findings and coverage only' },
];

/* Two flags that add a model at the end of a run, and only there. */
export const judgeCommands: readonly Command[] = [
    {
        cmd: 'proofpath check paper.pdf --judge',
        comment: 'a second opinion on the low-confidence verdicts',
    },
    {
        cmd: 'proofpath check paper.pdf --summarize',
        comment: 'one model-written paragraph over the finished report',
    },
    { cmd: 'proofpath config check', comment: 'proves the key works before you spend a run on it' },
];

/* The judge's line from the live run of 2026-09-15, beside the local verdict
   it did not replace (docs/eval/2026-09-15-judge-live.md). */
export const judgeLine = {
    local: 'error[not-supported]: claim is not supported by the cited source  (confidence: low)',
    judge: '= judge (groq openai/gpt-oss-120b): NEI — Passage states "SciPy provides fundamental algorithms for scientific computing" but does not mention SciPy building on anything.',
} as const;

export const exitCodes: readonly { code: 0 | 1 | 2; meaning: string }[] = [
    { code: 0, meaning: 'clean' },
    { code: 1, meaning: 'findings — every UNVERIFIED and LOW CONFIDENCE counts' },
    { code: 2, meaning: 'the run itself failed' },
];

export const version = '0.4.3';
export const repo = 'https://github.com/Yigtwxx/proofpath';
export const pypi = 'https://pypi.org/project/proofpath/';
