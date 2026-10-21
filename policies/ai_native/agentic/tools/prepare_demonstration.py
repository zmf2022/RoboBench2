# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract chronological three-camera video montages at 1 Hz, capped at 120."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path

from PIL import Image


def sample_indices(frame_count: int, control_hz: float = 15.0, cap: int = 120) -> list[int]:
    if frame_count < 1 or not math.isfinite(control_hz) or control_hz <= 0 or cap < 2:
        raise ValueError("Need nonempty video, positive control rate and cap >= 2")
    # Include both endpoints; round each timestamp to the nearest recorded frame.
    indices = sorted(
        {round(t * control_hz) for t in range(math.floor((frame_count - 1) / control_hz) + 1)} | {frame_count - 1}
    )
    if len(indices) > cap:
        indices = [round(i * (frame_count - 1) / (cap - 1)) for i in range(cap)]
    return indices


def prepare(video: Path, output: Path, *, control_hz=15.0):
    if output.exists():
        raise FileExistsError(f"Demonstration already frozen at {output}")
    ffmpeg, ffprobe = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise RuntimeError("Install system FFmpeg (ffmpeg and ffprobe) and add both executables to PATH")
    video = video.resolve()
    probe = subprocess.run(
        [ffprobe, "-v", "error", "-select_streams", "v:0", "-count_frames",
         "-show_entries", "stream=width,height,avg_frame_rate,nb_read_frames", "-of", "json", str(video)],
        check=True, capture_output=True, text=True,
    )
    streams = json.loads(probe.stdout)["streams"]
    if not streams:
        raise ValueError("Video contains no video stream")
    metadata = streams[0]
    frame_count = int(metadata["nb_read_frames"])
    fps = float(Fraction(metadata["avg_frame_rate"]))
    if not math.isclose(fps, control_hz, abs_tol=0.01):
        raise ValueError("Video FPS does not match control cadence; explicit timestamp mapping required")
    width, height = metadata["width"], metadata["height"]
    if width % 3 or not 1 <= (width / 3) / height <= 2.5:
        raise ValueError("Expected three horizontal camera views")
    selected = sample_indices(frame_count, control_hz)
    output.mkdir(parents=True)
    paths = []
    selection = "+".join(f"eq(n\\,{i})" for i in selected)
    command = [ffmpeg, "-v", "error", "-nostdin", "-noautorotate", "-i", str(video),
               "-map", "0:v:0", "-vf", f"select={selection}", "-vsync", "0",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "pipe:1"]
    # File-backed stderr avoids blocking decoding on a full error pipe.
    with tempfile.TemporaryFile() as errors, subprocess.Popen(command, stdout=subprocess.PIPE, stderr=errors) as decoder:
        try:
            for i in selected:
                data = decoder.stdout.read(width * height * 3)
                if len(data) != width * height * 3:
                    raise ValueError("Decoder did not produce every selected frame")
                frame = Image.frombytes("RGB", (width, height), data)
                frame.thumbnail((1920, 640), Image.Resampling.BILINEAR)
                name = f"frame-{i:06d}.jpg"
                frame.save(output / name, quality=85)
                paths.append(name)
            decoder.stdout.close()
            if decoder.wait():
                errors.seek(0)
                raise RuntimeError(f"FFmpeg decoding failed: {errors.read().decode(errors='replace')}")
        finally:
            if decoder.poll() is None:
                decoder.kill()
            decoder.stdout.close()
    timestamps = [i / control_hz for i in selected]
    manifest = dict(frames=paths, timestamps_s=timestamps, layout="three-horizontal")
    if len(paths) != len(selected):
        raise ValueError("Decoder did not produce every selected frame")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    with video.open("rb") as source:
        video_sha256 = hashlib.file_digest(source, "sha256").hexdigest()
    provenance = dict(
        source_video=str(video.resolve()),
        video_sha256=video_sha256,
        control_hz=control_hz,
        frame_count=frame_count,
        indices=selected,
        note="Times are relative to first recorded frame; endpoint offsets may differ from reset/termination.",
    )
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-hz", type=float, default=15.0)
    args = parser.parse_args()
    prepare(args.video, args.output, control_hz=args.control_hz)


if __name__ == "__main__":
    main()
