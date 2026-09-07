"""Request one calibrated frame from serve_viewer.py using the SIBR wire protocol.

Verifies the render server end to end without needing the SIBR client build.
"""
from argparse import ArgumentParser
from pathlib import Path
import json
import socket
import sys
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'work/4DGaussians'))
from utils.graphics_utils import getProjectionMatrix, focal2fov


def receive(sock, count):
    buf = bytearray()
    while len(buf) < count:
        chunk = sock.recv(count-len(buf))
        if not chunk:
            raise ConnectionError('Server disconnected')
        buf.extend(chunk)
    return bytes(buf)


def main():
    p = ArgumentParser(description=__doc__)
    p.add_argument('export', type=Path, help='exported dataset holding fdanyone.json')
    p.add_argument('--camera', type=int, required=True)
    p.add_argument('--port', type=int, default=6019)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    m = json.loads((args.export/'fdanyone.json').read_text())
    c = next(c for c in m['cameras'] if c['camera_id'] == args.camera)
    w, h = c['image_width'], c['image_height']
    k = np.array(c['K'])
    view = np.linalg.inv(c['camera_to_world']).T.copy()
    projection = getProjectionMatrix(.01, 100, focal2fov(k[0, 0], w), focal2fov(k[1, 1], h)).numpy().T
    full = view@projection
    view[:, 1:3] *= -1
    full[:, 1] *= -1
    message = dict(resolution_x=w, resolution_y=h, train=False,
                   fov_x=focal2fov(k[0, 0], w), fov_y=focal2fov(k[1, 1], h),
                   z_near=.01, z_far=100, shs_python=False, rot_scale_python=False,
                   keep_alive=True, scaling_modifier=1,
                   view_matrix=view.reshape(-1).tolist(),
                   view_projection_matrix=full.reshape(-1).tolist())
    with socket.create_connection(('127.0.0.1', args.port), timeout=30) as s:
        data = json.dumps(message).encode()
        s.sendall(len(data).to_bytes(4, 'little')+data)
        pixels = receive(s, w*h*3)
        length = int.from_bytes(receive(s, 4), 'little')
        source = receive(s, length).decode()
        Image.frombytes('RGB', (w, h), pixels).save(args.out)
        assert max(pixels) > 0, 'Server returned an empty frame'
        print(f'Received {w}x{h} RGB frame for camera {args.camera}; server dataset: {source}')


if __name__ == '__main__':
    main()
