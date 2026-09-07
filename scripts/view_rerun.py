"""Inspect existing 4DAnyone results without importing the inference pipeline."""
from __future__ import annotations

import argparse
import colorsys
import json
from fractions import Fraction
from pathlib import Path
import sys
import tempfile


def load_run(directory: Path) -> tuple[list[dict], dict]:
    cameras = json.loads((directory / "cameras.json").read_text())["cameras"]
    metadata = json.loads((directory / "metadata.json").read_text())
    if not cameras:
        raise ValueError("The result contains no cameras.")
    ids = [int(c["camera_id"]) for c in cameras]
    if len(ids) != len(set(ids)):
        raise ValueError("Camera IDs must be unique.")
    if Fraction(str(metadata["output"]["fps"])) <= 0:
        raise ValueError("Output FPS must be positive.")
    return sorted(cameras, key=lambda c: int(c["camera_id"])), metadata


def select_cameras(cameras: list[dict], views: list[int] | None, layer: int | None,
                   max_views: int) -> list[dict]:
    candidates = [c for c in cameras if layer is None or c.get("layer_index") == layer]
    if views is not None:
        by_id = {int(c["camera_id"]): c for c in candidates}
        missing = set(views) - by_id.keys()
        if missing:
            raise ValueError(f"Unknown camera IDs for this selection: {sorted(missing)}")
        return [by_id[i] for i in dict.fromkeys(views)]
    if not candidates:
        raise ValueError("No cameras match the selected layer.")
    # Interleave rings by yaw so a large multi-ring rig covers more than just
    # the front/back directions when only a few videos are selected.
    candidates = sorted(candidates, key=lambda c: (float(c["yaw"]), int(c["camera_id"])))
    count = min(max_views, len(candidates))
    return [candidates[i * len(candidates) // count] for i in range(count)]


def validate_geometry(cameras: list[dict]) -> None:
    import numpy as np
    for c in cameras:
        k, pose = np.asarray(c["K"]), np.asarray(c["camera_to_world"])
        if k.shape != (3, 3) or not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
            raise ValueError(f"Invalid intrinsics for camera {c['camera_id']}")
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-5)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1)):
            raise ValueError(f"Invalid pose for camera {c['camera_id']}")
        if int(c["image_width"]) <= 0 or int(c["image_height"]) <= 0:
            raise ValueError("Camera image dimensions must be positive.")


def log_video(recording, entity: str, path: Path, fps: Fraction, frames: int) -> None:
    import numpy as np
    import rerun as rr
    asset = rr.AssetVideo(path=path)
    timestamps = asset.read_frame_timestamps_nanos()
    if len(timestamps) != frames:
        raise ValueError(f"{path}: expected {frames} frames, found {len(timestamps)}")
    # Shared output clock selects corresponding frame indices, even if the MP4
    # time base rounds individual presentation timestamps differently.
    times = np.array([round(Fraction(i, 1) / fps * 1_000_000_000) for i in range(frames)], dtype=np.int64)
    recording.log(entity, asset, static=True)
    recording.send_columns(entity, indexes=[rr.TimeColumn("time", duration=times.astype("timedelta64[ns]")),
                                           rr.TimeColumn("frame", sequence=np.arange(frames))],
                           columns=rr.VideoFrameReference.columns(timestamp=timestamps))


def camera_pages(cameras: list[dict]) -> list[tuple[int, list[list[dict]]]]:
    """Group neighboring yaw directions into pages of four within each ring."""
    layers = sorted({int(c.get("layer_index", 0)) for c in cameras})
    result = []
    for layer in layers:
        ring = sorted((c for c in cameras if int(c.get("layer_index", 0)) == layer),
                      key=lambda c: float(c["yaw"]))
        result.append((layer, [ring[i:i + 4] for i in range(0, len(ring), 4)]))
    return result


def video_panes(cameras: list[dict], skeletons: bool):
    import rerun.blueprint as rrb
    panes = []
    for c in cameras:
        camera_id = int(c["camera_id"])
        panes.append(rrb.Spatial2DView(origin=f"world/cameras/{camera_id:02d}/image",
            name=f"View {camera_id:02d} · pitch {c.get('pitch', 0):g}° · yaw {c['yaw']:g}°"))
        if skeletons:
            panes.append(rrb.Spatial2DView(origin=f"skeletons/{camera_id:02d}", name=f"Pose {camera_id:02d}"))
    return panes


def playback_layout(selected: list[dict], embedded: list[dict], browse_all: bool, skeletons: bool):
    import rerun.blueprint as rrb
    overview = rrb.Grid(*video_panes(selected, skeletons), grid_columns=3, name="Overview")
    if not browse_all:
        return overview
    rings = []
    for layer, pages in camera_pages(embedded):
        tabs = []
        for page in pages:
            label = f"Views {page[0]['camera_id']:02d}–{page[-1]['camera_id']:02d}"
            tabs.append(rrb.Grid(*video_panes(page, skeletons), grid_columns=2, name=label))
        pitch = pages[0][0].get("pitch", 0)
        rings.append(rrb.Tabs(*tabs, active_tab=0, name=f"Layer {layer} · {pitch:g}°"))
    return rrb.Tabs(overview, *rings, active_tab=0)


def load_motion(path: Path, clip: str, frames: int, fps: Fraction) -> dict:
    import numpy as np
    with np.load(path, allow_pickle=False) as archive:
        motion = {key: archive[key] for key in archive.files}
    if str(motion["clip"]) != clip or str(motion["world_frame"]) != "canonical_human_world":
        raise ValueError("Motion clip or coordinate frame does not match this result.")
    if Fraction(int(motion["fps_num"]), int(motion["fps_den"])) != fps:
        raise ValueError("Motion FPS does not match the videos.")
    points, bones = motion["positions"], motion["bones"]
    if points.ndim != 3 or points.shape[0] != frames or points.shape[2] != 3 or not np.isfinite(points).all():
        raise ValueError("Motion must contain finite 3D points for every video frame.")
    if bones.ndim != 2 or bones.shape[1] != 2 or not np.issubdtype(bones.dtype, np.integer) or np.any(bones < 0) or np.any(bones >= points.shape[1]):
        raise ValueError("Motion contains invalid bone endpoints.")
    return motion


def log_motion(recording, motion: dict, fps: Fraction) -> None:
    import rerun as rr
    for index, points in enumerate(motion["positions"]):
        recording.set_time("frame", sequence=index)
        recording.set_time("time", duration=float(Fraction(index, 1) / fps))
        recording.log("world/body/joints", rr.Points3D(points, radii=.012,
            colors=motion["colors"], labels=motion["names"].tolist(), show_labels=False))
        recording.log("world/body/bones", rr.LineStrips3D(points[motion["bones"]],
            radii=.006, colors=motion["bone_colors"]))
    recording.reset_time()


def build_recording(args) -> None:
    import numpy as np
    import rerun as rr
    import rerun.blueprint as rrb
    cameras, metadata = load_run(args.result_dir)
    selected = select_cameras(cameras, args.views, args.layer, args.max_views)
    embedded = cameras if args.browse_all else selected
    validate_geometry(cameras)
    fps = Fraction(str(metadata["output"]["fps"]))
    frames = int(metadata["output"]["frames_per_video"])
    if frames <= 0:
        raise ValueError("Frame count must be positive.")
    motion = load_motion(args.motion, args.result_dir.name, frames, fps) if args.motion else None
    for c in embedded:
        for key in (["video", "skeleton_video"] if args.skeletons else ["video"]):
            path = args.result_dir / c[key]
            if not path.is_file():
                raise ValueError(f"Missing {key}: {path}")
    if args.save.exists():
        raise ValueError(f"Recording already exists: {args.save}. Choose a new --save path.")
    args.save.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=args.save.parent, suffix=".rrd", delete=False) as handle:
        temporary = Path(handle.name)
    recording = rr.RecordingStream("4danyone-results")
    try:
        recording.save(str(temporary))
        recording.log("world", rr.ViewCoordinates.RIGHT_HAND_Y_UP, static=True)
        for c in cameras:
            entity = f"world/cameras/{int(c['camera_id']):02d}"
            pose = np.asarray(c["camera_to_world"])
            rgb = [int(v * 255) for v in colorsys.hsv_to_rgb(float(c["yaw"]) % 360 / 360, .65, 1)]
            recording.log(entity, rr.Transform3D(mat3x3=pose[:3, :3], translation=pose[:3, 3]), static=True)
            recording.log(entity, rr.Pinhole(image_from_camera=c["K"],
                resolution=[c["image_width"], c["image_height"]],
                camera_xyz=rr.ViewCoordinates.RDF, image_plane_distance=.25, color=rgb), static=True)
            if c not in embedded:
                continue
            print(f"Loading view {c['camera_id']:02d} (layer {c.get('layer_index', 0)}, yaw {c['yaw']:g}°)", flush=True)
            log_video(recording, f"{entity}/image", args.result_dir / c["video"], fps, frames)
            if args.skeletons:
                skeleton = f"skeletons/{int(c['camera_id']):02d}"
                log_video(recording, skeleton, args.result_dir / c["skeleton_video"], fps, frames)
        if motion is not None:
            log_motion(recording, motion, fps)
        recording.send_blueprint(rrb.Blueprint(
            rrb.Vertical(rrb.Spatial3DView(origin="world", name="Camera rig",
                             contents=[f"world/cameras/{int(c['camera_id']):02d}" for c in cameras] + ["world/body/**"]),
                         playback_layout(selected, embedded, args.browse_all, args.skeletons), row_shares=[1, 2]),
            rrb.TimePanel(timeline="time", play_state="playing", loop_mode="all")))
        recording.flush()
        recording.disconnect()
        # Publish only a complete recording; never replace an existing file.
        import os
        os.link(temporary, args.save)
    finally:
        recording.disconnect()
        temporary.unlink(missing_ok=True)
    print(f"Saved {args.save} ({args.save.stat().st_size / 1e6:.1f} MB)")
    if not args.no_open:
        from open_rerun import open_recording
        open_recording(args.save, args.ffmpeg)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--views", type=int, nargs="+", help="Camera IDs to play (overrides --max-views)")
    parser.add_argument("--layer", type=int, help="Restrict playback to this pitch layer")
    parser.add_argument("--max-views", type=int, default=6, help="Evenly sample this many videos (default: 6)")
    parser.add_argument("--motion", type=Path, help="3D motion NPZ from scripts/export_rerun_motion.py")
    parser.add_argument("--browse-all", action="store_true", help="Embed every video with an overview and per-layer tabs of four")
    parser.add_argument("--skeletons", action="store_true", help="Add synchronized pose-conditioning videos")
    parser.add_argument("--save", type=Path, help="Recording path (default: outputs/rerun/<clip>.rrd)")
    parser.add_argument("--no-open", action="store_true", help="Save without launching the native viewer")
    parser.add_argument("--ffmpeg", type=Path, help="FFmpeg >= 5.1 executable for native playback")
    args = parser.parse_args()
    if args.max_views < 1:
        parser.error("--max-views must be positive")
    args.save = args.save or Path("outputs/rerun") / f"{args.result_dir.name}.rrd"
    try:
        build_recording(args)
    except ImportError as exc:
        parser.exit(1, f"Missing viewer dependency: {exc}. Install requirements-rerun.txt in .venv.\n")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
