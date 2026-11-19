#!/usr/bin/env python3
"""Nhip 90 experiment: 2K vs 4K decided by DISPLAY SIZE, at equal file size.

Two synthetic 2160x3240 truths (72 frames, slow pan):
  detail -> noise band-limited to 0.485 cyc/px (energy to 4K Nyquist)
  soft   -> noise band-limited to 0.215 cyc/px (nothing above 2K Nyquist)

Both arms are encoded at matched bitrate targets (maxrate+bufsize so the target binds),
then scored against the reference for each DISPLAY size, so neither arm is scored against
a reference that already discarded the detail.

Run from an empty dir. Takes ~20 min (72 frames x 3 sizes x ~20 encodes).
Numbers reported on https://celticgemini001.github.io/upgolab-2k-vs-4k-decision/
"""
import json, os, re, subprocess, sys

import numpy as np
from PIL import Image

H, W = 3240, 2160
NFRAME = 72
FPS = 25
BR = ['4M', '5M', '6M', '7M', '8M', '8.5M', '9M', '10M', '11M', '12M', '13M', '14M', '16M']


def bandlimit(n, cut):
    F = np.fft.rfft2(n)
    r = np.sqrt(np.fft.fftfreq(H)[:, None] ** 2 + np.fft.rfftfreq(W)[None, :] ** 2)
    F[r > cut] = 0
    return np.fft.irfft2(F, s=(H, W)).astype(np.float32)


def build(cut, name):
    rng = np.random.default_rng(20261119)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    base = 0.45 + 0.12 * np.sin(2 * np.pi * xx / (W * 0.7)) + 0.10 * np.cos(2 * np.pi * yy / (H * 0.55))
    t = bandlimit(rng.standard_normal((H, W)).astype(np.float32), cut)
    t /= t.std()
    edges = ((xx % 90) < 45) * 0.10 - ((yy % 70) < 35) * 0.08
    disc = (((xx - W * .5) ** 2 + (yy - H * .42) ** 2) < (W * .18) ** 2).astype(np.float32)
    img = np.clip(base + 0.055 * t + edges + 0.09 * disc, 0, 1)
    im = Image.fromarray((img * 255).astype(np.uint8))
    for i in range(NFRAME):
        c = im.crop((0, 0, 2160 - int(i * .5), 3240 - int(i * .5)))
        for d, s in [(f'seq_{name}', (2160, 3240)), (f'ref2k_{name}', (1440, 2160)),
                     (f'ref1080_{name}', (1080, 1620))]:
            os.makedirs(d, exist_ok=True)
            c.resize(s, Image.Resampling.LANCZOS).save(f'{d}/f{i:03d}.png')


def enc(src, out, br):
    subprocess.run(['ffmpeg', '-v', 'error', '-y', '-framerate', str(FPS), '-i', src,
                    '-c:v', 'libx264', '-preset', 'slow', '-b:v', br, '-maxrate', br,
                    '-bufsize', f'2*{br}', '-pix_fmt', 'yuv420p', '-g', '25',
                    '-threads', '1', out], check=True)


def ssim(a, b, extra=None):
    # settb=AVTB on BOTH branches or the per-frame pairing drifts
    fa = '[0:v]format=rgb24,settb=AVTB' + (f',{extra}' if extra else '') + '[d]'
    cmd = ['ffmpeg', '-hide_banner', '-i', a, '-framerate', str(FPS), '-i', b, '-lavfi',
           f'{fa};[1:v]format=rgb24,settb=AVTB[r];[d][r]ssim', '-f', 'null', '-']
    # NOTE: do not pass -v error, the SSIM summary prints at info level
    e = subprocess.run(cmd, capture_output=True, text=True).stderr
    m = re.search(r'All:([\d.]+)', e)
    return round(float(m.group(1)), 5) if m else None


def main():
    os.makedirs('enc', exist_ok=True)
    if not os.path.isdir('seq_detail'):
        build(0.485, 'detail')
        build(0.215, 'soft')

    px = {'4k': 2160 * 3240, '2k': 1440 * 2160}
    rows = []
    for c in ['detail', 'soft']:
        for br in BR:
            p4, p2 = f'enc/{c}_4k_{br}.mp4', f'enc/{c}_2k_{br}.mp4'
            if not os.path.exists(p4):
                enc(f'seq_{c}/f%03d.png', p4, br)
            if not os.path.exists(p2):
                enc(f'ref2k_{c}/f%03d.png', p2, br)
            s4, s2 = os.path.getsize(p4), os.path.getsize(p2)
            f4, f2, f1 = (f'{d}/f%03d.png' for d in (f'seq_{c}', f'ref2k_{c}', f'ref1080_{c}'))
            r = dict(content=c, br=br, s4=s4, s2=s2, ratio=round(s4 / s2, 3),
                     bpp4=round(s4 * 8 / NFRAME * FPS / px['4k'], 2),
                     bpp2=round(s2 * 8 / NFRAME * FPS / px['2k'], 2),
                     # scored against the reference for each display size
                     a4_4k=ssim(p4, f4),
                     a4_2k=ssim(p4, f2, 'scale=1440:2160:flags=lanczos'),
                     a4_1k=ssim(p4, f1, 'scale=1080:1620:flags=lanczos'),
                     a2_4k=ssim(p2, f4, 'scale=2160:3240:flags=lanczos'),
                     a2_2k=ssim(p2, f2),
                     a2_1k=ssim(p2, f1, 'scale=1080:1620:flags=lanczos'))
            rows.append(r)
            print(f"{c:7}{br:5} 4K {s4/1e6:6.2f}MB  2K {s2/1e6:6.2f}MB r={r['ratio']:.3f} "
                  f"@4K {r['a4_4k']:.4f}/{r['a2_4k']:.4f}  @2K {r['a4_2k']:.4f}/{r['a2_2k']:.4f}  "
                  f"@1K {r['a4_1k']:.4f}/{r['a2_1k']:.4f}", flush=True)
    json.dump(rows, open('nhip90_rows.json', 'w'), indent=1)

    # equal-file-size gaps, interpolated in bytes
    print('\n=== delta SSIM (4K arm - 2K arm) at equal bytes; + means 4K file wins ===')
    for c in ['detail', 'soft']:
        rs = [r for r in rows if r['content'] == c]
        x4 = np.array([r['s4'] for r in rs]); x2 = np.array([r['s2'] for r in rs])
        order4 = np.argsort(x4); order2 = np.argsort(x2)
        for disp, k4, k2 in [('4K screen', 'a4_4k', 'a2_4k'),
                             ('2K screen', 'a4_2k', 'a2_2k'),
                             ('1080 screen', 'a4_1k', 'a2_1k')]:
            y4 = np.array([r[k4] for r in rs])[order4]; y2 = np.array([r[k2] for r in rs])[order2]
            y4c, y2c = np.array([r[k4] for r in rs]), np.array([r[k2] for r in rs])
            for mb in [2.0, 3.0, 4.0, 5.0, 5.5]:
                b = mb * 1e6
                if b < max(x4.min(), x2.min()) or b > min(x4.max(), x2.max()):
                    continue
                g = np.interp(b, x4[order4], y4c[order4]) - np.interp(b, x2[order2], y2c[order2])
                print(f'{c:7}{disp:11}{mb:4.1f}MB  {g:+.4f}')


if __name__ == '__main__':
    sys.exit(main())
