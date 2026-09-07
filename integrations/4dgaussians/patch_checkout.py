"""Apply the small adapter/modern-environment changes to the pinned checkout."""
from pathlib import Path
import shutil
import sys

root = Path(sys.argv[1])
shutil.copyfile(Path(__file__).with_name('fdanyone_loader.py'), root/'scene/fdanyone_loader.py')
p = root/'scene/__init__.py'
s = p.read_text()
if 'read_fdanyone' not in s:
    old = '        if os.path.exists(os.path.join(args.source_path, "sparse")):'
    if old not in s:
        raise RuntimeError('Unrecognized upstream Scene loader')
    s = s.replace(old, '''        if os.path.exists(os.path.join(args.source_path, "fdanyone.json")):
            from scene.fdanyone_loader import read_fdanyone
            scene_info = read_fdanyone(args.source_path)
            dataset_type = "fdanyone"
        elif os.path.exists(os.path.join(args.source_path, "sparse")):''')
    p.write_text(s)
for name in ['train.py', 'render.py']:
    p = root/name
    s = p.read_text().replace('import mmcv', 'import mmengine as mmcv')
    if name == 'train.py':
        # Full-resolution PNG decoding starves the GPU on a single process.
        for workers in ('num_workers=16', 'num_workers=32', 'num_workers=0'):
            s = s.replace(workers, 'num_workers=8')
    p.write_text(s)
p = root/'submodules/simple-knn/simple_knn.cu'
s = p.read_text()
if '#include <cfloat>' not in s:
    p.write_text('#include <cfloat>\n' + s)
p = root/'train.py'
s = p.read_text()
s = s.replace('                loader = iter(viewpoint_stack_loader)\n\n        else:',
              '                loader = iter(viewpoint_stack_loader)\n                viewpoint_cams = next(loader)\n\n        else:')
p.write_text(s)
# Checkpoints are locally generated full optimizer/model snapshots, not weights-only.
p = root/'train.py'
s = p.read_text().replace('torch.load(checkpoint)', 'torch.load(checkpoint, weights_only=False)')
p.write_text(s)
p = root/'train.py'
s = p.read_text()
# The hard densification cap stops well short of a detailed human reconstruction.
s = s.replace('shape[0]<360000', 'shape[0]<800000')
p.write_text(s)
