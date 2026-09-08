"""Export a masked dynamic dataset with known cameras for the 4DGaussians adapter."""
from pathlib import Path
import argparse
import copy
import json
import sys
import numpy as np
from PIL import Image


def select_views(camera_ids, train_views=None, test_views=(), video_camera=None):
    """Resolve disjoint training/evaluation sets and an independent playback view."""
    available = set(camera_ids)
    test = sorted(set(test_views))
    train = sorted(set(train_views) if train_views is not None else available - set(test))
    if not train or set(train) & set(test):
        raise ValueError('Training cameras must be nonempty and disjoint from test cameras')
    selected = sorted(set(train + test))
    if set(selected) - available:
        raise ValueError('Unknown camera IDs')
    video = train[0] if video_camera is None else video_camera
    if video not in selected:
        raise ValueError('Playback camera must be included in the export')
    return train, test, selected, video


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('result_dir', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--train-views', type=int, nargs='+', help='Default: every camera except test views')
    p.add_argument('--test-views', type=int, nargs='+', default=[], help='Optional held-out cameras')
    p.add_argument('--video-camera', type=int, help='Playback camera; default: first training camera')
    p.add_argument('--seed-points', type=int, default=200000)
    p.add_argument('--scale', type=float, default=1.0)
    args = p.parse_args()
    if not 0 < args.scale <= 1 or args.seed_points < 1:
        p.error('Scale must be in (0,1] and seed-points must be positive.')
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from fdanyone.video import iter_rgb_video
    from fdanyone.foreground import predict_foreground_masks
    from fdanyone.assets import resolve_foreground_model
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry
    root = args.result_dir.resolve()
    metadata = json.loads((root/'metadata.json').read_text())
    cameras = json.loads((root/'cameras.json').read_text())['cameras']
    by_id = {c['camera_id']: c for c in cameras}
    try:
        args.train_views, args.test_views, selected, video_camera = select_views(
            by_id, args.train_views, args.test_views, args.video_camera)
    except ValueError as exc:
        p.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = dict(version=1, source_clip=root.name, output=metadata['output'], cameras=[],
                    train_views=args.train_views, test_views=args.test_views, video_camera=video_camera,
                    world_frame='canonical_human_world', background='black')
    for cid in selected:
        c = copy.deepcopy(by_id[cid])
        print(f'Segmenting camera {cid:02d}', flush=True)
        frames = tuple(iter_rgb_video(root/c['video']))
        if len(frames) != metadata['output']['frames_per_video']:
            raise ValueError(f'Frame count mismatch for camera {cid}')
        masks = predict_foreground_masks(frames, resolve_foreground_model('models'), 'cuda:0')
        size = (round(c['image_width']*args.scale), round(c['image_height']*args.scale))
        k = np.asarray(c['K'], dtype=float)
        k[0] *= size[0]/c['image_width']; k[1] *= size[1]/c['image_height']
        c.update(K=k.tolist(), image_width=size[0], image_height=size[1])
        folder = args.output/f'cam{cid:02d}'; folder.mkdir()
        for i, (frame, mask) in enumerate(zip(frames, masks)):
            rgba = np.concatenate([frame, ((mask>=128)*255).astype(np.uint8)[...,None]], axis=-1)
            Image.fromarray(rgba).resize(size, Image.Resampling.LANCZOS).save(folder/f'{i:04d}.png')
        manifest['cameras'].append(c)
    print('Building motion-based initialization', flush=True)
    motion_dir = root.parent.parent/'gvhmr/results'/root.name
    geometry = _body_geometry(MotionResult.load(motion_dir), Path('models/4danyone/smplx_to_goliath70.pt').resolve(),
                              Path('third_party/GVHMR').resolve(), 'cpu')
    vertices = geometry.vertices_world.reshape(-1,3)
    rng = np.random.default_rng(42)
    points = vertices[rng.choice(len(vertices), min(args.seed_points,len(vertices)), replace=False)]
    points = points + rng.normal(0,.002,points.shape)
    np.savez_compressed(args.output/'seed.npz', points=points.astype(np.float32),
                        colors=np.full(points.shape,.5,dtype=np.float32))
    # Write the manifest last so incomplete exports cannot be used for training.
    (args.output/'fdanyone.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print(f'Completed {len(selected)} cameras at {size}: {args.output}', flush=True)


if __name__ == '__main__':
    main()
