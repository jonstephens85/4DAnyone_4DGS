"""Score a rendered held-out camera against the exported ground truth.

Full-image PSNR is dominated by the black background (the subject covers ~14%
of the frame), so it flatters every model and hides real differences. This
reports foreground PSNR, a detail ratio, and a colour-outlier check that
catches the failure mode where a Gaussian's fixed colour diverges.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from PIL import Image
from scipy.ndimage import binary_erosion


def load_gt(export, camera, frame):
    rgba = np.asarray(Image.open(export/f'cam{camera:02d}'/f'{frame:04d}.png'))
    alpha = rgba[:, :, 3:4].astype(np.float32)/255
    # The adapter premultiplies at load time; match it so renders compare fairly.
    return (rgba[:, :, :3]*alpha).round().astype(np.uint8), rgba[:, :, 3] > 0


def psnr(a, b, mask):
    d = (a.astype(np.float64)/255 - b.astype(np.float64)/255)**2
    return 10*np.log10(1/max(d[mask].mean(), 1e-12))


def detail(img, mask):
    g = np.asarray(Image.fromarray(img).convert('L'), np.float64)/255
    gy, gx = np.gradient(g)
    return np.hypot(gx, gy)[mask].mean()*1000


def outliers(img, mask):
    # Blue exceeding green inside the silhouette: the magenta divergence.
    px = img[mask].astype(np.float64)/255
    return float(((px[:, 2]-px[:, 1]) > 0.15).mean())*100


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('--export', type=Path, required=True)
    p.add_argument('--renders', type=Path, required=True,
                   help='a model test/ours_<iter>/renders directory')
    p.add_argument('--camera', type=int, required=True)
    p.add_argument('--frame', type=int, default=42, help='frame for the detail report')
    args = p.parse_args()
    m = json.loads((args.export/'fdanyone.json').read_text())
    frames = int(m['output']['frames_per_video'])
    full, fg = [], []
    for i in range(frames):
        gt, a = load_gt(args.export, args.camera, i)
        r = np.asarray(Image.open(args.renders/f'{i:05d}.png').convert('RGB'))
        m3 = a[..., None].repeat(3, 2)
        full.append(psnr(r, gt, np.ones_like(m3, bool)))
        fg.append(psnr(r, gt, m3))
    gt, a = load_gt(args.export, args.camera, args.frame)
    inner = binary_erosion(a, np.ones((9, 9)))   # ignore the silhouette edge
    r = np.asarray(Image.open(args.renders/f'{args.frame:05d}.png').convert('RGB'))
    dg, dr = detail(gt, inner), detail(r, inner)
    print(f'camera {args.camera}, {frames} frames')
    print(f'  full-image PSNR  {np.mean(full):6.2f} dB')
    print(f'  foreground PSNR  {np.mean(fg):6.2f} dB   (min {np.min(fg):.2f} @ frame {int(np.argmin(fg))})')
    print(f'  detail @{args.frame}       {dr:6.2f} vs GT {dg:.2f}  ->  {dr/dg*100:.1f}% of GT')
    print(f'  colour outliers  {outliers(r, inner):5.2f}%   (GT {outliers(gt, inner):.2f}%)')


if __name__ == '__main__':
    main()
