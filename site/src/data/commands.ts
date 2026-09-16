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

export const exitCodes: readonly { code: 0 | 1 | 2; meaning: string }[] = [
    { code: 0, meaning: 'clean' },
    { code: 1, meaning: 'findings — every UNVERIFIED and LOW CONFIDENCE counts' },
    { code: 2, meaning: 'the run itself failed' },
];

export const version = '0.4.0';
export const repo = 'https://github.com/Yigtwxx/proofpath';
export const pypi = 'https://pypi.org/project/proofpath/';
