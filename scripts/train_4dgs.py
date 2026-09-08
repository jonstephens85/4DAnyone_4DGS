"""Train a dynamic Gaussian model using the separate 4DGS environment."""
import argparse
import json
from pathlib import Path
import pprint
import runpy
import subprocess
from _4dgs import ROOT, CHECKOUT, PYTHON, run


def settings(iterations, coarse_iterations, batch_size):
    groups = runpy.run_path(str(ROOT/'integrations/4dgaussians/default.py'))
    config = {k: v for k, v in groups.items() if k.endswith('Params')}
    config['OptimizationParams'].update(iterations=iterations, coarse_iterations=coarse_iterations,
        batch_size=batch_size, position_lr_max_steps=iterations,
        densify_until_iter=min(20000, max(1, iterations-1)))
    return config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--iterations',type=int,default=30000,help='Fine-stage iterations (default: 30000)')
    p.add_argument('--coarse-iterations',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=2)
    p.add_argument('--evaluate',action='store_true',help='Evaluate held-out cameras at the final iteration')
    p.add_argument('--port',type=int,default=6019)
    p.add_argument('--dry-run',action='store_true',help='Print resolved settings without writing or training')
    args=p.parse_args()
    if min(args.iterations,args.coarse_iterations,args.batch_size)<1:
        p.error('Iteration counts and batch size must be positive')
    try:
        data=args.data.expanduser().resolve(); output=args.output.expanduser().resolve()
        manifest=json.loads((data/'fdanyone.json').read_text())
        if not manifest['train_views']:
            raise ValueError('Dataset has no training cameras')
        if args.evaluate and not manifest['test_views']:
            raise ValueError('--evaluate requires an export with --test-views')
        config=settings(args.iterations,args.coarse_iterations,args.batch_size)
        if args.dry_run:
            print(json.dumps(dict(data=str(data),output=str(output),train_views=manifest['train_views'],
                                 evaluate=args.evaluate,config=config),indent=2)); return
        if not PYTHON.is_file():
            raise ValueError('Set up the 4DGS environment first: docs/4dgs.md')
        # Patch before launch so a fresh installation uses the current adapter.
        run([ROOT/'integrations/4dgaussians/patch_checkout.py',CHECKOUT],cwd=ROOT)
        output.mkdir(parents=True,exist_ok=False)
        config_path=output/'config.py'
        config_path.write_text('\n\n'.join(f'{key} = {pprint.pformat(value)}' for key,value in config.items())+'\n')
        command=[CHECKOUT/'train.py','-s',data,'--model_path',output,'--configs',config_path,
                 '--port',str(args.port),'--save_iterations',str(args.iterations),
                 '--checkpoint_iterations',str(args.iterations),
                 '--test_iterations',str(args.iterations if args.evaluate else -1)]
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=CHECKOUT,text=True).strip()
        (output/'run.json').write_text(json.dumps(dict(data=str(data),iterations=args.iterations,
            coarse_iterations=args.coarse_iterations,evaluate=args.evaluate,
            upstream_revision=revision,command=list(map(str,command))),indent=2)+'\n')
        run(command)
    except (OSError,ValueError,KeyError,subprocess.CalledProcessError) as exc:
        p.exit(1,f'error: {exc}\n')


if __name__=='__main__': main()
