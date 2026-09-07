"""Open an RRD with a validated FFmpeg >= 5.1 on the viewer's PATH."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys


def open_recording(recording: Path, ffmpeg: Path | None = None) -> None:
    local = Path(sys.executable).parent / "ffmpeg"
    executable = str(ffmpeg) if ffmpeg else str(local) if local.is_file() else shutil.which("ffmpeg")
    if not executable:
        raise ValueError("FFmpeg >= 5.1 is required. Pass --ffmpeg /path/to/ffmpeg.")
    executable = str(Path(executable).absolute())
    result = subprocess.run([executable, "-version"], capture_output=True, text=True, check=True)
    match = re.search(r"ffmpeg version (\d+)\.(\d+)", result.stdout)
    if not match or tuple(map(int, match.groups())) < (5, 1):
        raise ValueError(f"Unsupported FFmpeg: {result.stdout.splitlines()[0]}. Pass --ffmpeg /path/to/ffmpeg (>= 5.1).")
    if Path(executable).name != "ffmpeg":
        raise ValueError("The FFmpeg executable must be named ffmpeg so Rerun can find it on PATH.")
    environment = os.environ.copy()
    environment["PATH"] = str(Path(executable).parent) + os.pathsep + environment.get("PATH", "")
    print(f"Opening {recording} with {result.stdout.splitlines()[0]}", flush=True)
    subprocess.run([str(Path(sys.executable).with_name("rerun")), str(recording)], env=environment, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("recording", type=Path)
    parser.add_argument("--ffmpeg", type=Path)
    args = parser.parse_args()
    try:
        open_recording(args.recording, args.ffmpeg)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"error: {exc}\n")


if __name__ == "__main__":
    main()
