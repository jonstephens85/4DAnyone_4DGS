"""Write the 3DGS-style cameras.json SIBR needs to recognize a model directory.

4DGaussians does not emit cameras.json, so SIBR_remoteGaussian_app cannot
determine the dataset type. The export manifest already holds every value.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import shutil
import numpy as np


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('source', type=Path, help='exported dataset holding fdanyone.json')
    p.add_argument('model', type=Path, help='trained model directory to annotate')
    args = p.parse_args()
    m = json.loads((args.source/'fdanyone.json').read_text())
    entries = []
    for c in sorted(m['cameras'], key=lambda c: c['camera_id']):
        c2w = np.asarray(c['camera_to_world'], dtype=float)
        k = np.asarray(c['K'], dtype=float)
        # SIBR's loadJSON reads the 3DGS convention: camera centre plus the
        # camera-to-world rotation, which is exactly what the manifest stores.
        entries.append(dict(id=c['camera_id'], img_name=f"cam{c['camera_id']:02d}",
                            width=c['image_width'], height=c['image_height'],
                            position=c2w[:3, 3].tolist(),
                            rotation=[r.tolist() for r in c2w[:3, :3]],
                            fx=k[0, 0], fy=k[1, 1]))
    (args.model/'cameras.json').write_text(json.dumps(entries, indent=2)+'\n')
    # getParsedGaussianData also resolves <path>/input.ply for the proxy mesh.
    ply = args.source/'points3D.ply'
    if ply.exists() and not (args.model/'input.ply').exists():
        shutil.copyfile(ply, args.model/'input.ply')
    print(f'Wrote {len(entries)} cameras to {args.model/"cameras.json"}')


if __name__ == '__main__':
    main()
