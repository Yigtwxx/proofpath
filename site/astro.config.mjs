// @ts-check
import { defineConfig } from 'astro/config';
import sitemap from '@astrojs/sitemap';

// `site` is required by the sitemap integration and for canonical / OG URLs.
// Placeholder until a domain is chosen; Vercel previews still work without it.
export default defineConfig({
    site: 'https://proofpath.vercel.app',
    output: 'static',
    integrations: [sitemap({ filter: (page) => !page.endsWith('/og') })],
    build: { inlineStylesheets: 'auto' },
});
