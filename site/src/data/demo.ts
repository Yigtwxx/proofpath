/* A real run, replayed. Lines are from docs/eval/2026-09-15-tui-v2-live.md
   (`/check tests/data/draft-live.md`, rich theme), abbreviated to fit a page.
   A frame either appends a line or, when it names an `id` that already exists,
   replaces that line — that is how a stage flips from running to done. The
   banner above the output is static. */
import { version } from './commands';

export type Tone = 'ok' | 'warn' | 'bad' | 'dim' | 'accent' | 'plain' | 'you' | 'source';

export interface Span {
    text: string;
    tone?: Tone;
}

export interface Frame {
    /** Milliseconds to wait before this frame lands. */
    wait: number;
    /** Lines with the same id replace each other. */
    id?: string;
    /** `type` frames are typed character by character. */
    type?: boolean;
    spans: Span[];
}

/**
 * The `rich` banner: `proofpath.tui.wordmark.WORDMARK` verbatim — six lines, 66
 * columns, `█` for the letters and `╗╔═╝║╚` for their shadow. The page draws the
 * same cells in an SVG rather than as text, because block and box-drawing glyphs
 * leave gaps between lines in a browser font.
 */
export const WORDMARK_BLOCK = '█';
export const wordmark: readonly string[] = [
    '██████╗                        █████╗██████╗               ██╗',
    '██╔══██╗                      ██╔═══╝██╔══██╗         ██╗  ██║',
    '██████╔╝██╗██╗ █████╗  █████╗ █████╗ ██████╔╝ █████╗ █████╗██████╗',
    '██╔═══╝ ████╔╝██╔══██╗██╔══██╗██╔══╝ ██╔═══╝ ██╔══██╗╚██╔═╝██╔═██╗',
    '██║     ██╔═╝ ╚█████╔╝╚█████╔╝██║    ██║     ███████║ ██║  ██║ ██║',
    '╚═╝     ╚═╝    ╚════╝  ╚════╝ ╚═╝    ╚═╝     ╚═╝  ╚═╝ ╚═╝  ╚═╝ ╚═╝',
];
export const WORDMARK_COLUMNS = Math.max(...wordmark.map((line) => line.length));
/** The TUI's five gradient bands, left to right, over the mark's own width. */
export const WORDMARK_BANDS = ['#e0455f', '#d02f4c', '#c4173a', '#a91330', '#8f0f2b'] as const;
/** The shadow glyphs and the rule under the banner. */
export const WORDMARK_SHADOW = '#6b0a20';

/** The two text lines beside or under the mark, as the TUI shows them at start. */
export const banner = {
    version: `proofpath v${version}`,
    context: 'academic . online . coreml',
    hint: 'paste a file path, a URL, or a claim.',
    commands: '/help  /config  /quit',
} as const;

const stage = (
    glyph: string,
    name: string,
    detail: string,
    provider: string,
    t: string,
): Span[] => [
    { text: glyph, tone: glyph === '✓' ? 'ok' : 'accent' },
    { text: ` ${name.padEnd(12)}` },
    { text: detail.padEnd(52), tone: 'plain' },
    { text: provider.padEnd(8), tone: 'dim' },
    { text: t, tone: 'dim' },
];

export const frames: Frame[] = [
    {
        wait: 300,
        id: 'prompt',
        type: true,
        spans: [{ text: '› ', tone: 'dim' }, { text: '/check tests/data/draft-live.md' }],
    },
    {
        wait: 700,
        id: 'hdr',
        spans: [
            { text: '#1  /check tests/data/draft-live.md', tone: 'accent' },
            { text: '  running', tone: 'dim' },
        ],
    },
    { wait: 250, id: 's1', spans: stage('⏺', 'Parsing', '', '', '') },
    {
        wait: 450,
        id: 's1',
        spans: stage('✓', 'Parsing', '1 pages · 7 refs', 'text', '0.0s'),
    },
    { wait: 150, id: 's2', spans: stage('⏺', 'Claims', '', '', '') },
    {
        wait: 450,
        id: 's2',
        spans: stage('✓', 'Claims', '7 citations · 0 unresolved', 'rules', '0.0s'),
    },
    { wait: 150, id: 's3', spans: stage('⏺', 'Resolving', '', '', '') },
    {
        wait: 700,
        id: 's3',
        spans: stage('⏺', 'Resolving', '6 ok · 0 amb · 1 ghost', 'cache', '0.0s'),
    },
    { wait: 200, id: 's4', spans: stage('⏺', 'Retractions', '', '', '') },
    { wait: 450, id: 's4', spans: stage('✓', 'Retractions', 'none', 'cache', '0.0s') },
    { wait: 150, id: 's5', spans: stage('⏺', 'Fetching', '', '', '') },
    {
        wait: 700,
        id: 's5',
        spans: stage('✓', 'Fetching', '5 full text · 1 abstract · 0 unverified', 'cache', '0.0s'),
    },
    { wait: 150, id: 's6', spans: stage('⏺', 'Verifying', '', '', '') },
    {
        wait: 900,
        id: 's6',
        spans: stage(
            '✓',
            'Verifying',
            '10 claims: 1 supported · 2 not supported · 7 NEI',
            'coreml',
            '0.8s',
        ),
    },
    {
        wait: 100,
        id: 'hdr',
        spans: [
            { text: '#1  /check tests/data/draft-live.md', tone: 'accent' },
            { text: '  done', tone: 'ok' },
        ],
    },
    { wait: 250, spans: [{ text: '─'.repeat(78), tone: 'dim' }] },
    {
        wait: 350,
        spans: [
            { text: '✗ ', tone: 'bad' },
            { text: 'L41  [7] Marchetti, L. R., Osei, K. & Lind…   ' },
            { text: ' GHOST REFERENCE ', tone: 'bad' },
        ],
    },
    {
        wait: 250,
        spans: [
            { text: '       arxiv and openlibrary consulted before the ghost call', tone: 'dim' },
        ],
    },
    {
        wait: 400,
        spans: [
            { text: '· ', tone: 'dim' },
            { text: 'L40  [6] Pearl, J. & Mackenzie, D. The B…      ' },
            { text: ' LOW CONFIDENCE (abstract only) ', tone: 'warn' },
        ],
    },
    {
        wait: 400,
        spans: [
            { text: '✗ ', tone: 'bad' },
            { text: 'L6   [1] Vaswani, A., Shazeer, N., Parmar,…    ' },
            { text: ' NOT SUPPORTED ', tone: 'bad' },
            { text: '  high', tone: 'dim' },
        ],
    },
    {
        wait: 150,
        spans: [
            {
                text: '       numeric mismatch: claim says 21 days, source says 3.5 days',
                tone: 'dim',
            },
        ],
    },
    {
        wait: 200,
        spans: [
            { text: '       you     ', tone: 'you' },
            {
                text: '"On the WMT 2014 English-to-French translation task the Transformer reached…"',
            },
        ],
    },
    {
        wait: 200,
        spans: [
            { text: '       source  ', tone: 'source' },
            {
                text: '"On the WMT 2014 English-to-French translation task, our model establishes …" ⧉',
            },
        ],
    },
    {
        wait: 400,
        spans: [
            { text: '· ', tone: 'dim' },
            { text: 'L11  [2] Jumper, J., Evans, R., Pritzel, A…   ' },
            { text: ' NEI ', tone: 'dim' },
            { text: '  low', tone: 'dim' },
        ],
    },
    { wait: 300, spans: [{ text: '─'.repeat(78), tone: 'dim' }] },
    {
        wait: 200,
        spans: [
            { text: '██████████████████████', tone: 'ok' },
            { text: '▓▓▓▓', tone: 'warn' },
            { text: '░░░░', tone: 'dim' },
            { text: '  72% full text · 14% abstract · 14% unverified', tone: 'plain' },
        ],
    },
    { wait: 200, spans: [{ text: '7 refs: 1 ghost, 2 unsupported, 4 ok', tone: 'plain' }] },
    {
        wait: 150,
        spans: [{ text: 'no report written · 0 API calls · 0.8s', tone: 'dim' }],
    },
];
