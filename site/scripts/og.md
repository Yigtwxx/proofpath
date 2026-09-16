# Regenerating `public/og.png`

The social card is the `/og` page rendered at 1200×630 in a real browser, so it
uses the site's own fonts and halftones. To regenerate after a copy or image change:

1. `npm run dev`
2. Open `http://localhost:4321/og` in Chrome, set the viewport to 1200×630
   (DevTools → device toolbar → responsive), and remove the dev toolbar element
   (`document.querySelector('astro-dev-toolbar')?.remove()` in the console).
3. Capture the viewport (DevTools → ⋮ → _Capture screenshot_) and save it as
   `public/og.png`.

The page is `noindex` and excluded from the sitemap.
