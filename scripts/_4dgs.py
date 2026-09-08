"""Shared launcher helpers; all reconstruction dependencies stay in work/4dgs-env."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHECKOUT = ROOT / 'work/4DGaussians'
PYTHON = ROOT / 'work/4dgs-env/bin/python'


def run(arguments, cwd=CHECKOUT):
    if not PYTHON.is_file() or not (CHECKOUT/'train.py').is_file():
        raise ValueError('Install the separate 4DGS environment first; see docs/4dgs.md.')
    subprocess.run([str(PYTHON), *map(str, arguments)], cwd=cwd, check=True)


def model_options(args):
    model = args.model.expanduser().resolve()
    saved = json.loads((model/'run.json').read_text()) if (model/'run.json').exists() else {}
    data = args.data or saved.get('data')
    config = args.config or (model/'config.py' if (model/'config.py').exists() else None)
    iteration = args.iteration or saved.get('iterations')
    if data is None or config is None or iteration is None:
        raise ValueError('For an older model, supply --data, --config, and --iteration.')
    return model, Path(data).expanduser().resolve(), Path(config).expanduser().resolve(), int(iteration)


def add_model_options(parser):
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--data', type=Path, help='Default: saved training dataset')
    parser.add_argument('--config', type=Path, help='Default: saved model configuration')
    parser.add_argument('--iteration', type=int, help='Default: saved final iteration')
