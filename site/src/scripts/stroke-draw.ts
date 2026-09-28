/* Draws pencil strokes when they scroll into view.

   Each stroke is hidden with a dash as long as itself and revealed by sliding
   the dash offset to zero. The length has to be measured in *screen* space:
   several primitives stretch their SVG with `preserveAspectRatio="none"` and
   keep the pen width constant with `non-scaling-stroke`, and Chrome then lays
   the dash pattern out in screen pixels, ignoring `pathLength`. So this walks
   each path through its screen CTM and writes the result to `--len`.

   The stylesheet only hides strokes under `html.js` and inside
   `prefers-reduced-motion: no-preference`, so with no script or with reduced
   motion every line is simply there. */

type Stroke = SVGGeometryElement;

function screenLength(el: Stroke): number {
    const ctm = el.getScreenCTM();
    const total = el.getTotalLength();
    if (!ctm || total === 0) return 0;
    const steps = Math.max(8, Math.min(120, Math.ceil(total / 4)));
    let length = 0;
    let prev = el.getPointAtLength(0).matrixTransform(ctm);
    for (let i = 1; i <= steps; i++) {
        const point = el.getPointAtLength((total * i) / steps).matrixTransform(ctm);
        length += Math.hypot(point.x - prev.x, point.y - prev.y);
        prev = point;
    }
    return length;
}

function measure(svg: Element): void {
    const strokes = svg.querySelectorAll<Stroke>('path, line, polyline, circle, ellipse');
    for (const stroke of strokes) {
        const len = Math.ceil(screenLength(stroke) * 1.04) + 2;
        stroke.style.setProperty('--len', String(len));
        // Commit the hidden state now. Without a style pass in between, the drawn
        // state that follows would transition from whatever was computed last --
        // the 4000 fallback -- and the dash would cycle on the way to 0.
        void getComputedStyle(stroke).strokeDashoffset;
    }
}

// Longest `--delay` a stroke is given (the echo line of a box), plus a margin: when
// no `transitionend` arrives (a stroke with nothing left to animate), the drawing
// still settles.
const SETTLE_SLACK_MS = 400;

/** Drop the dash once every stroke has finished drawing (see global.css). */
function settle(svg: Element): void {
    const strokes = svg.querySelectorAll('path, line, polyline, circle, ellipse').length;
    let pending = strokes;
    const done = (): void => {
        svg.classList.add('is-settled');
        svg.removeEventListener('transitionend', onEnd);
    };
    const onEnd = (event: Event): void => {
        if ((event as TransitionEvent).propertyName !== 'stroke-dashoffset') return;
        pending -= 1;
        if (pending <= 0) done();
    };
    svg.addEventListener('transitionend', onEnd);
    const duration = parseFloat(getComputedStyle(svg).getPropertyValue('--draw-duration')) || 900;
    window.setTimeout(done, duration + SETTLE_SLACK_MS);
}

export function initStrokeDraw(root: ParentNode = document): void {
    if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const targets = Array.from(root.querySelectorAll<SVGElement>('[data-draw]'));
    if (targets.length === 0) return;

    for (const svg of targets) measure(svg);

    if (!('IntersectionObserver' in window)) {
        for (const svg of targets) svg.classList.add('is-drawn', 'is-settled');
        return;
    }
    const observer = new IntersectionObserver(
        (entries) => {
            for (const entry of entries) {
                if (!entry.isIntersecting) continue;
                // Re-measure right before drawing: fonts may have loaded since.
                measure(entry.target);
                requestAnimationFrame(() => {
                    entry.target.classList.add('is-drawn');
                    settle(entry.target);
                });
                observer.unobserve(entry.target);
            }
        },
        { threshold: 0.3, rootMargin: '0px 0px -5% 0px' },
    );
    for (const svg of targets) observer.observe(svg);
}
