"""Render the saved playback camera through the full action."""
import argparse
import subprocess
from _4dgs import CHECKOUT, add_model_options, model_options, run


def main():
    p=argparse.ArgumentParser(description=__doc__); add_model_options(p)
    args=p.parse_args()
    try:
        model,data,config,iteration=model_options(args)
        run([CHECKOUT/'render.py','-s',data,'--model_path',model,'--configs',config,
             '--iteration',str(iteration),'--skip_train','--skip_test'])
        print(f'Video: {model}/video/ours_{iteration}/video_rgb.mp4')
    except (OSError,ValueError,subprocess.CalledProcessError) as exc: p.exit(1,f'error: {exc}\n')


if __name__=='__main__': main()
