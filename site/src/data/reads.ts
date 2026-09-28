/* What proofpath can read, in plain words, as three piles: what it reads with nothing
   set up, what needs one thing from you, and what it cannot read (and says so).
   Mirrors README "What it reads"; the details live further down the page. */
import type { IconName } from '../components/sketch/SketchIcon.astro';

export interface ReadItem {
    icon: IconName;
    name: string;
    note: string;
}

export interface ReadColumn {
    title: string;
    lede: string;
    tone: 'ink' | 'crimson' | 'slate';
    items: readonly ReadItem[];
}

export const readColumns: readonly ReadColumn[] = [
    {
        title: 'Reads it',
        lede: 'No key, no account.',
        tone: 'ink',
        items: [
            { icon: 'doc', name: 'Your files', note: 'PDF, Word (.docx), Markdown, .txt' },
            { icon: 'book', name: 'Papers', note: 'DOI, arXiv, books' },
            { icon: 'globe', name: 'Web pages', note: 'News, blogs, Wikipedia, any page' },
            {
                icon: 'bubble',
                name: 'Posts',
                note: 'Bluesky, Hacker News, Lobste.rs, Mastodon, Lemmy',
            },
        ],
    },
    {
        title: 'Needs one thing from you',
        lede: 'Free, set up once.',
        tone: 'crimson',
        items: [
            { icon: 'key', name: 'Reddit', note: 'a free Reddit app of your own' },
            {
                icon: 'spark',
                name: 'LLM second opinion',
                note: 'a free Groq key, or Ollama with no key',
            },
            {
                icon: 'window',
                name: 'Sites that block robots',
                note: 'your OK for a 280 MB browser, asked first',
            },
        ],
    },
    {
        title: "Can't read, and says so",
        lede: 'Never skipped quietly.',
        tone: 'slate',
        items: [
            { icon: 'scan', name: 'Scanned PDFs', note: 'no OCR yet' },
            {
                icon: 'oldfile',
                name: 'Old formats',
                note: '.doc, .odt, .rtf: save as .docx or PDF',
            },
            { icon: 'lock', name: 'Paywalls', note: 'reported as unread' },
            { icon: 'nopost', name: 'X / Twitter', note: 'paste the text instead' },
        ],
    },
];
