"""Validate all frames and camera/seed geometry before CUDA training."""
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

root=Path(sys.argv[1])
m=json.loads((root/'fdanyone.json').read_text())
assert not set(m['train_views']) & set(m['test_views'])
assert set(m['train_views']+m['test_views']) == {c['camera_id'] for c in m['cameras']}
seed=np.load(root/'seed.npz',allow_pickle=False)
assert seed['points'].shape == seed['colors'].shape
assert np.isfinite(seed['points']).all()
for c in m['cameras']:
    pose=np.array(c['camera_to_world']); k=np.array(c['K'])
    assert np.allclose(pose[3],[0,0,0,1])
    assert np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3))
    assert np.isclose(np.linalg.det(pose[:3,:3]),1)
    local=np.array([.1,.2,3.])
    world=pose[:3,:3]@local+pose[:3,3]
    w2c=np.linalg.inv(pose)
    assert np.allclose(w2c[:3,:3]@world+w2c[:3,3],local)
    assert np.allclose([k[0,2],k[1,2]],[c['image_width']/2,c['image_height']/2])
    files=sorted((root/f"cam{c['camera_id']:02d}").glob('*.png'))
    assert len(files)==m['output']['frames_per_video']
    for i,path in enumerate(files):
        assert path.name==f'{i:04d}.png'
        with Image.open(path) as image:
            assert image.mode=='RGBA' and image.size==(c['image_width'],c['image_height'])
            alpha=np.asarray(image)[:,:,3]
            assert (alpha>0).any() and (alpha==0).any()
print(f"Validated {len(m['cameras'])} cameras, {len(files)} frames each, disjoint split, masks and geometry.")
