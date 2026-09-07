"""Serve a trained dynamic model to SIBR_remoteGaussian_app without training."""
from argparse import ArgumentParser
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/4DGaussians'))
import torch
from mmengine import Config
from arguments import ModelParams, PipelineParams, ModelHiddenParams
from utils.params_utils import merge_hparams
from scene import Scene, GaussianModel
from gaussian_renderer import render, network_gui


def recv_exact(sock, length):
    data = bytearray()
    while len(data) < length:
        part = sock.recv(length-len(data))
        if not part:
            raise ConnectionError('Viewer disconnected')
        data.extend(part)
    return data


def read_message():
    import json
    length = int.from_bytes(recv_exact(network_gui.conn,4), 'little')
    if not 0 < length <= 1024*1024:
        raise ValueError('Invalid viewer message length')
    return json.loads(recv_exact(network_gui.conn,length).decode('utf-8'))


def main():
    parser=ArgumentParser(description=__doc__)
    lp=ModelParams(parser); hp=ModelHiddenParams(parser); pp=PipelineParams(parser)
    parser.add_argument('--configs',required=True)
    parser.add_argument('--iteration',type=int,default=14000)
    parser.add_argument('--port',type=int,default=6017)
    parser.add_argument('--time',type=float,help='Hold a normalized time in [0,1]; default: loop animation')
    args=parser.parse_args()
    if args.time is not None and not 0 <= args.time <= 1:
        parser.error('--time must be in [0,1]')
    args=merge_hparams(args,Config.fromfile(args.configs))
    data=lp.extract(args); pipe=pp.extract(args)
    with torch.no_grad():
        gaussians=GaussianModel(data.sh_degree,hp.extract(args))
        scene=Scene(data,gaussians,load_iteration=args.iteration,shuffle=False)
        background=torch.tensor([0,0,0],dtype=torch.float32,device='cuda')
        manifest=__import__('json').loads((Path(data.source_path)/'fdanyone.json').read_text())
        from fractions import Fraction
        duration=(manifest['output']['frames_per_video']-1)/float(Fraction(manifest['output']['fps']))
        network_gui.read=read_message
        network_gui.init('127.0.0.1',args.port)
        print(f'SIBR server ready on 127.0.0.1:{args.port}; Ctrl+C to stop.',flush=True)
        started=time.monotonic()
        try:
            while True:
                if network_gui.conn is None:
                    network_gui.try_connect()
                    if network_gui.conn is None:
                        time.sleep(.05)
                        continue
                    network_gui.conn.settimeout(30)
                try:
                    camera, _, pipe.convert_SHs_python, pipe.compute_cov3D_python, _, scale = network_gui.receive()
                    payload=None
                    if camera is not None:
                        if camera.image_width*camera.image_height > 4096*4096:
                            raise ValueError('Viewer resolution exceeds 4096² pixels')
                        camera.time=args.time if args.time is not None else ((time.monotonic()-started)%duration)/duration
                        pixels=render(camera,gaussians,pipe,background,scale,stage='fine',cam_type=scene.dataset_type)['render']
                        payload=memoryview((pixels.clamp(0,1)*255).byte().permute(1,2,0).contiguous().cpu().numpy())
                    network_gui.send(payload,data.source_path)
                except Exception as exc:
                    print(f'Viewer connection closed: {exc}',flush=True)
                    network_gui.conn.close()
                    network_gui.conn=None
        finally:
            if network_gui.conn is not None:
                network_gui.conn.close()
            network_gui.listener.close()


if __name__=='__main__':
    main()
