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

## Regenerating `.github/social-preview.jpg`

The GitHub social preview is the `/social` page, 1280×640. It is captured at 2×
and scaled down, which keeps the dithered engraving sharp without moiré and the
file under GitHub's 1 MB limit:

1. `npm run build && npm run preview`, open `http://localhost:4321/social` in Chrome.
2. Set the viewport to 1280×640 at device pixel ratio 2 and capture it (2560×1280).
3. Scale it to 1280×640 and save it as JPEG, quality 90:
   `sips -z 640 1280 capture.png --out small.png && sips -s format jpeg -s formatOptions 90 small.png --out ../.github/social-preview.jpg`
4. Upload it in the repository's Settings → General → Social preview. GitHub has no
   API for this.
