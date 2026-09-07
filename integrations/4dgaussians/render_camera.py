"""Render one exported camera at one timestamp from a trained dynamic model."""
from argparse import ArgumentParser
from pathlib import Path
import json
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/4DGaussians'))
import numpy as np
import torch
from PIL import Image
from mmengine import Config
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.params_utils import merge_hparams
from scene import Scene, GaussianModel
from gaussian_renderer import render


def main():
    parser = ArgumentParser(description=__doc__)
    lp = ModelParams(parser); hp = ModelHiddenParams(parser); pp = PipelineParams(parser)
    parser.add_argument('--configs', required=True)
    parser.add_argument('--iteration', type=int, required=True)
    parser.add_argument('--camera', type=int, required=True)
    parser.add_argument('--frame', type=int, default=42)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args = merge_hparams(args, Config.fromfile(args.configs))
    data = lp.extract(args); pipe = pp.extract(args)
    with torch.no_grad():
        gaussians = GaussianModel(data.sh_degree, hp.extract(args))
        scene = Scene(data, gaussians, load_iteration=args.iteration, shuffle=False)
        manifest = json.loads((Path(data.source_path)/'fdanyone.json').read_text())
        frames = int(manifest['output']['frames_per_video'])
        train_ids = sorted(manifest['train_views'])
        if args.camera not in train_ids:
            raise SystemExit(f'Camera {args.camera} is not a training view of this export')
        # Frames enumerates cameras in ascending id, each with every timestamp.
        index = train_ids.index(args.camera)*frames + args.frame
        cam = scene.getTrainCameras()[index]
        bg = torch.tensor([0, 0, 0], dtype=torch.float32, device='cuda')
        px = render(cam, gaussians, pipe, bg, stage='fine', cam_type=scene.dataset_type)['render']
        img = (px.clamp(0, 1)*255).byte().permute(1, 2, 0).cpu().numpy()
        Image.fromarray(img).save(args.out)
        print(f'camera {args.camera} frame {args.frame} -> {args.out}')


if __name__ == '__main__':
    main()
