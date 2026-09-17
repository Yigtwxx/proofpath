/* The v0.4 surface: a post read without an account, the links inside it, and
   since v0.4.4 any other page as the document itself.
   Platforms and their answers follow README "Posts and the links inside them"
   and providers/social.py; the states are spec §15's, not paraphrases. */
export interface Platform {
    name: string;
    how: string;
    /** The state a run reports when the platform cannot be read, if any. */
    state?: { label: string; tone: 'ok' | 'slate' | 'ink' };
}

export const postCommands: readonly { cmd: string; comment: string }[] = [
    {
        cmd: 'proofpath check --url https://bsky.app/profile/bsky.app/post/3movpwtbjgs2d',
        comment: 'a post: every link inside it is a source',
    },
    {
        cmd: 'proofpath check --url https://news.ycombinator.com/item?id=8863',
        comment: "a story's URL, and the links in a comment",
    },
    { cmd: 'proofpath check -', comment: 'paste the text of a post that cannot be read' },
    {
        cmd: 'proofpath check --url https://en.wikipedia.org/wiki/AlphaFold',
        comment: 'any other address is a page, read as the document itself',
    },
];

export const platforms: readonly Platform[] = [
    {
        name: 'Bluesky',
        how: 'public.api.bsky.app, no account. Link cards, rich-text links and one level of quoted post.',
        state: { label: 'read', tone: 'ink' },
    },
    {
        name: 'Hacker News',
        how: "The official Firebase API, no account. A story's URL and the links in a comment.",
        state: { label: 'read', tone: 'ink' },
    },
    {
        name: 'Reddit',
        how: 'With a free app you register yourself: REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET in .env. Without them no request is made: the post is UNVERIFIED (credentials missing), and the run names both variables.',
        state: { label: 'credentials missing', tone: 'slate' },
    },
    {
        name: 'Mastodon',
        how: 'Best effort, per instance. An instance that wants a login says so, and that answer is reported as UNVERIFIED (blocked) — not as a missing post.',
        state: { label: 'blocked', tone: 'slate' },
    },
    {
        name: 'X',
        how: 'Cannot be read at all. The run says so and asks you to paste the text; the links inside it are then verified as usual.',
        state: { label: 'paste the text', tone: 'ink' },
    },
];
