/* robots.txt, built from `site` so the sitemap line always names the address the
   page is served at (astro.config.mjs), never a stale one. */
import type { APIRoute } from 'astro';
import { withBase } from '../lib/base';

export const GET: APIRoute = ({ site }) => {
    const sitemap = new URL(withBase('sitemap-index.xml'), site).href;
    return new Response(`User-agent: *\nAllow: /\n\nSitemap: ${sitemap}\n`, {
        headers: { 'Content-Type': 'text/plain; charset=utf-8' },
    });
};
