/* Cleans and re-emits the favicon set from the 256 px bird embedded in
   `public/favicon.svg`: every white pixel not connected to the bird itself
   (halftone specks left over from the crop) is painted crimson, then the SVG,
   the two PNG sizes and a three-size `favicon.ico` are written from that one
   image. The bird's pixels are not touched. */

import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import sharp from 'sharp';

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)));
const out = (name) => path.join(root, 'public', name);
const CRIMSON = [0xc4, 0x17, 0x3a];

const svgIn = await readFile(out('favicon.svg'), 'utf8');
const embedded = /base64,([^"]+)/.exec(svgIn)?.[1];
if (!embedded) throw new Error('favicon.svg carries no embedded PNG');
const { data, info } = await sharp(Buffer.from(embedded, 'base64'))
    .ensureAlpha()
    .raw()
    .toBuffer({ resolveWithObject: true });
const { width: W, height: H, channels: C } = info;

// Label the light pixels' 8-connected components; keep only the largest.
const light = new Uint8Array(W * H);
for (let i = 0; i < W * H; i++) light[i] = data[i * C] > 200 ? 1 : 0;
const label = new Int32Array(W * H).fill(-1);
const sizes = [];
for (let seed = 0; seed < W * H; seed++) {
    if (!light[seed] || label[seed] >= 0) continue;
    const id = sizes.length;
    const stack = [seed];
    label[seed] = id;
    let n = 0;
    while (stack.length) {
        const p = stack.pop();
        n++;
        const x = p % W;
        const y = (p - x) / W;
        for (let dy = -1; dy <= 1; dy++) {
            for (let dx = -1; dx <= 1; dx++) {
                const nx = x + dx;
                const ny = y + dy;
                if (nx < 0 || ny < 0 || nx >= W || ny >= H) continue;
                const q = ny * W + nx;
                if (light[q] && label[q] < 0) {
                    label[q] = id;
                    stack.push(q);
                }
            }
        }
    }
    sizes.push(n);
}
const bird = sizes.indexOf(Math.max(...sizes));
let painted = 0;
for (let i = 0; i < W * H; i++) {
    if (light[i] && label[i] !== bird) {
        data.set([...CRIMSON, 255], i * C);
        painted++;
    }
}

const base = sharp(data, { raw: { width: W, height: H, channels: C } });
const png256 = await base.clone().png({ palette: true }).toBuffer();
const png = (px) =>
    sharp(png256).resize(px, px, { kernel: 'lanczos3' }).png({ palette: true }).toBuffer();

await writeFile(
    out('favicon.svg'),
    `<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" viewBox="0 0 ${W} ${H}"><image width="${W}" height="${H}" xlink:href="data:image/png;base64,${png256.toString('base64')}"/></svg>\n`,
);
await writeFile(out('favicon-32.png'), await png(32));
await writeFile(out('apple-touch-icon.png'), await png(180));

// .ico as a directory of PNG-encoded entries, which every modern browser reads.
const entries = await Promise.all([16, 32, 48].map(async (px) => [px, await png(px)]));
const header = Buffer.alloc(6);
header.writeUInt16LE(1, 2);
header.writeUInt16LE(entries.length, 4);
const dirs = [];
let offset = 6 + 16 * entries.length;
for (const [px, buf] of entries) {
    const d = Buffer.alloc(16);
    d.writeUInt8(px, 0);
    d.writeUInt8(px, 1);
    d.writeUInt16LE(1, 4);
    d.writeUInt16LE(32, 6);
    d.writeUInt32LE(buf.length, 8);
    d.writeUInt32LE(offset, 12);
    offset += buf.length;
    dirs.push(d);
}
await writeFile(out('favicon.ico'), Buffer.concat([header, ...dirs, ...entries.map(([, b]) => b)]));
console.log(`favicon: ${sizes.length - 1} specks (${painted} px) painted crimson, bird kept`);
