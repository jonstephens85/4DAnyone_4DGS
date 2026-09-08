"""Lazy, calibrated multi-view sequence loader for 4DAnyone exports."""
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from PIL import Image
from scene.dataset_readers import CameraInfo, SceneInfo, getNerfppNorm, storePly
from utils.graphics_utils import BasicPointCloud, focal2fov


class Frames:
    def __init__(self, root, manifest, ids):
        self.root = Path(root)
        self.cameras = [c for c in manifest['cameras'] if c['camera_id'] in ids]
        self.frames = int(manifest['output']['frames_per_video'])
        self.samples = [(c, i) for c in self.cameras for i in range(self.frames)]
        for c in self.cameras:
            k = np.asarray(c['K'])
            w, h = c['image_width'], c['image_height']
            if not np.allclose([k[0,2],k[1,2]], [w/2,h/2]) or not np.isclose(k[0,1],0):
                raise ValueError('This rasterizer requires centered, zero-skew intrinsics')

    def __len__(self):
        return len(self.samples)

    @lru_cache(maxsize=32)
    def __getitem__(self, index):
        c, frame = self.samples[index]
        path = self.root/f"cam{c['camera_id']:02d}"/f'{frame:04d}.png'
        with Image.open(path) as image:
            rgba = np.asarray(image.convert('RGBA'), dtype=np.float32)/255
        rgb = rgba[:,:,:3] * rgba[:,:,3:4]
        w2c = np.linalg.inv(np.asarray(c['camera_to_world']))
        k = c['K']; w,h = c['image_width'],c['image_height']
        return CameraInfo(uid=index, R=w2c[:3,:3].T, T=w2c[:3,3],
            FovX=focal2fov(k[0][0],w), FovY=focal2fov(k[1][1],h),
            image=torch.from_numpy(rgb.copy()).permute(2,0,1), image_path=str(path),
            image_name=f"cam{c['camera_id']:02d}_{frame:04d}", width=w,height=h,
            time=frame/max(1,self.frames-1),mask=None)


def read_fdanyone(root):
    root = Path(root)
    m = json.loads((root/'fdanyone.json').read_text())
    if m['version'] != 1 or m['world_frame'] != 'canonical_human_world':
        raise ValueError('Unsupported export format')
    if set(m['train_views']) & set(m['test_views']):
        raise ValueError('Train/test cameras overlap')
    if not m['train_views']:
        raise ValueError('Dataset has no training cameras')
    train = Frames(root,m,m['train_views']); test = Frames(root,m,m['test_views'])
    seed = np.load(root/'seed.npz',allow_pickle=False)
    xyz, rgb = seed['points'],seed['colors']
    ply = root/'points3D.ply'
    storePly(str(ply),xyz,rgb*255)
    norm = getNerfppNorm([train[i*train.frames] for i in range(len(train.cameras))])
    # Playback is independent of evaluation; preserve old exports' default.
    fallback = (m['test_views'] or m['train_views'])[0]
    video_id = m.get('video_camera', fallback)
    if video_id not in {c['camera_id'] for c in m['cameras']}:
        raise ValueError('Unknown playback camera')
    video = Frames(root,m,[video_id])
    return SceneInfo(point_cloud=BasicPointCloud(points=xyz,colors=rgb,normals=np.zeros_like(xyz)),
                     train_cameras=train,test_cameras=test,video_cameras=video,
                     nerf_normalization=norm,ply_path=str(ply),maxtime=1)
