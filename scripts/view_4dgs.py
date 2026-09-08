"""Prepare SIBR camera files and serve an animated model to its remote client."""
import argparse
import subprocess
from _4dgs import ROOT, add_model_options, model_options, run


def main():
    p=argparse.ArgumentParser(description=__doc__); add_model_options(p)
    p.add_argument('--port',type=int,default=6019)
    p.add_argument('--time',type=float,help='Freeze normalized time in [0,1]')
    args=p.parse_args()
    if args.time is not None and not 0<=args.time<=1: p.error('--time must be in [0,1]')
    try:
        model,data,config,iteration=model_options(args)
        run([ROOT/'integrations/4dgaussians/write_sibr_cameras.py',data,model],cwd=ROOT)
        command=[ROOT/'integrations/4dgaussians/serve_viewer.py','-s',data,'--model_path',model,
                 '--configs',config,'--iteration',str(iteration),'--port',str(args.port)]
        if args.time is not None: command += ['--time',str(args.time)]
        print(f'Connect SIBR_remoteGaussian_app --port {args.port} --path {model}',flush=True)
        run(command,cwd=ROOT)
    except (OSError,ValueError,subprocess.CalledProcessError) as exc: p.exit(1,f'error: {exc}\n')


if __name__=='__main__': main()
