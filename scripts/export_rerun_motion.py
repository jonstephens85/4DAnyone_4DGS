"""Export cached conditioning keypoints for the lightweight Rerun environment."""
from pathlib import Path
import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("motion_dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    parser.add_argument("--gvhmr-root", type=Path, default=Path("third_party/GVHMR"))
    args = parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import numpy as np
    from fdanyone.motion.result import MotionResult
    from fdanyone.skeleton.pipeline import _body_geometry
    from fdanyone.skeleton.keypoints import LINKS, VISIBLE_KEYPOINT_IDS, KEYPOINT_NAMES, keypoint_color

    motion = MotionResult.load(args.motion_dir)
    geometry = _body_geometry(motion, (args.model_dir / "4danyone/smplx_to_goliath70.pt").resolve(),
                              args.gvhmr_root.resolve(), "cpu")
    ids = sorted(VISIBLE_KEYPOINT_IDS)
    mapping = {original: index for index, original in enumerate(ids)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("xb") as output:
        np.savez_compressed(output, positions=geometry.keypoints_world[:, ids],
            bones=np.array([[mapping[a], mapping[b]] for _, a, b, _, _ in LINKS]),
            colors=np.array([keypoint_color(i) for i in ids], dtype=np.uint8),
            bone_colors=np.array([color for _, _, _, color, _ in LINKS], dtype=np.uint8),
            names=np.array([KEYPOINT_NAMES[i] for i in ids]),
            fps_num=motion.fps.numerator, fps_den=motion.fps.denominator,
            clip=args.motion_dir.name, world_frame="canonical_human_world")
    print(f"Saved {motion.num_frames} frames of 3D motion to {args.out}")


if __name__ == "__main__":
    main()
