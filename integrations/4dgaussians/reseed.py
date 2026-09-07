"""Regenerate seed.npz for an existing export with a denser point cloud.

Densification grows from the seed, so a 40k motion-envelope cloud is a weak
start for a detailed reconstruction. This rebuilds only the initialization,
leaving the already-segmented images and the manifest untouched.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
import numpy as np


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path, help='the 4DAnyone clip directory')
    p.add_argument('export_dir', type=Path, help='the exported dataset to reseed')
    p.add_argument('--points', type=int, default=200_000)
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry
    manifest = json.loads((args.export_dir/'fdanyone.json').read_text())
    if manifest['version'] != 1:
        raise ValueError('Unsupported export format')
    root = args.result_dir.resolve()
    motion_dir = root.parent.parent/'gvhmr/results'/root.name
    geometry = _body_geometry(MotionResult.load(motion_dir),
                              Path('models/4danyone/smplx_to_goliath70.pt').resolve(),
                              Path('third_party/GVHMR').resolve(), 'cpu')
    vertices = geometry.vertices_world.reshape(-1, 3)
    rng = np.random.default_rng(42)
    count = min(args.points, len(vertices))
    points = vertices[rng.choice(len(vertices), count, replace=False)]
    points = points + rng.normal(0, .002, points.shape)
    np.savez_compressed(args.export_dir/'seed.npz', points=points.astype(np.float32),
                        colors=np.full(points.shape, .5, dtype=np.float32))
    # points3D.ply is rebuilt from seed.npz by the loader on the next run.
    stale = args.export_dir/'points3D.ply'
    if stale.exists():
        stale.unlink()
    print(f'Seeded {count} points from {len(vertices)} body vertices into {args.export_dir}')


if __name__ == '__main__':
    main()
