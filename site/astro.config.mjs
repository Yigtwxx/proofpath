// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// `site` is required by the sitemap integration and for canonical / OG URLs;
// `base` is the sub-path this page will live at on the hub site. Both are
// placeholders until the hub exists — set them together when it does, and
// every asset path follows (see src/lib/base.ts).
export default defineConfig({
    site: 'https://example.com',
    base: '/',
    output: 'static',
    integrations: [sitemap({ filter: (page) => !page.endsWith('/og') })],
    build: { inlineStylesheets: 'auto' },
});
