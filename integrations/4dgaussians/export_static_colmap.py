"""Export one timestamp of a 4DAnyone dataset as a COLMAP model for static 3DGS.

This is the control experiment for the dynamic reconstruction: identical images,
masks and calibration, but a single instant and no deformation network. Poses
are known exactly, so no structure-from-motion is run.

Image order matches camera id, so vanilla 3DGS's `--eval` holdout (every 8th
index) leaves cameras 00, 08, 16, 24, 32 and 40 unseen -- camera 40 being the
same held-out view used to score the dynamic model.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import numpy as np
from PIL import Image


def rotmat2qvec(r):
    # COLMAP's rotation-matrix to quaternion conversion (w, x, y, z).
    xx, yx, zx, xy, yy, zy, xz, yz, zz = r.flat
    k = np.array([
        [xx-yy-zz, 0, 0, 0],
        [yx+xy, yy-xx-zz, 0, 0],
        [zx+xz, zy+yz, zz-xx-yy, 0],
        [yz-zy, zx-xz, xy-yx, xx+yy+zz]]) / 3.0
    vals, vecs = np.linalg.eigh(k)
    qvec = vecs[[3, 0, 1, 2], np.argmax(vals)]
    return -qvec if qvec[0] < 0 else qvec


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('export_dir', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--frame', type=int, default=42)
    args = p.parse_args()
    m = json.loads((args.export_dir/'fdanyone.json').read_text())
    frames = int(m['output']['frames_per_video'])
    if not 0 <= args.frame < frames:
        p.error(f'Frame must be in [0,{frames-1}]')
    sparse = args.output/'sparse/0'
    sparse.mkdir(parents=True, exist_ok=False)
    (args.output/'images').mkdir()
    cams, imgs = [], []
    for c in sorted(m['cameras'], key=lambda c: c['camera_id']):
        cid = c['camera_id']
        k = np.asarray(c['K'], dtype=float)
        w, h = c['image_width'], c['image_height']
        cams.append(f'{cid+1} PINHOLE {w} {h} {k[0,0]} {k[1,1]} {k[0,2]} {k[1,2]}')
        w2c = np.linalg.inv(np.asarray(c['camera_to_world'], dtype=float))
        q = rotmat2qvec(w2c[:3, :3])
        t = w2c[:3, 3]
        name = f'cam{cid:02d}.png'
        imgs.append(f'{cid+1} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {cid+1} {name}\n')
        # Vanilla 3DGS keeps the raw RGB as ground truth, so the background is
        # zeroed here (the 4DGaussians adapter premultiplies at load time).
        rgba = np.asarray(Image.open(args.export_dir/f'cam{cid:02d}'/f'{args.frame:04d}.png'))
        alpha = rgba[:, :, 3:4].astype(np.float32)/255
        rgb = (rgba[:, :, :3]*alpha).round().astype(np.uint8)
        # Written without an alpha channel on purpose: with alpha present 3DGS
        # masks the render before the loss, leaving the background unsupervised
        # and free to fill with floaters. The 4DGaussians adapter supervises the
        # full black-background image, so this matches it.
        Image.fromarray(rgb).save(args.output/'images'/name)
    (sparse/'cameras.txt').write_text('\n'.join(cams)+'\n')
    # COLMAP images.txt alternates a pose line with a (here empty) 2D-point line.
    (sparse/'images.txt').write_text(''.join(line+'\n' for line in imgs))
    seed = np.load(args.export_dir/'seed.npz', allow_pickle=False)
    pts = seed['points']
    lines = [f'{i+1} {x} {y} {z} 128 128 128 0' for i, (x, y, z) in enumerate(pts)]
    (sparse/'points3D.txt').write_text('\n'.join(lines)+'\n')
    print(f'Wrote {len(cams)} cameras and {len(pts)} seed points for frame '
          f'{args.frame} to {args.output}')


if __name__ == '__main__':
    main()
