# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load image-only demonstrations; source metadata stays in host-side provenance."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import math
from pathlib import Path

from PIL import Image


def load_demonstration(path: Path, *, image_size: int, jpeg_quality: int) -> tuple[list[dict], dict]:
    raw = path.read_bytes()
    manifest = json.loads(raw)
    frames = manifest.get("frames")
    if not isinstance(frames, list) or not 1 <= len(frames) <= 120:
        raise ValueError("frames must contain 1–120 chronological images")
    if set(manifest) == {"sample_hz", "frames"}:
        hz = manifest["sample_hz"]
        if isinstance(hz, bool) or not isinstance(hz, (float, int)) or not math.isfinite(hz) or hz <= 0:
            raise ValueError("sample_hz must be finite and positive")
        timestamps = [i / hz for i in range(len(frames))]
        timing = f"They were sampled chronologically at {hz:g} Hz. "
    elif set(manifest) == {"frames", "timestamps_s", "layout"}:
        timestamps = manifest["timestamps_s"]
        if manifest["layout"] != "three-horizontal" or len(timestamps) != len(frames):
            raise ValueError("Expected three-horizontal layout with one timestamp per frame")
        if any(type(t) not in (float, int) or not math.isfinite(t) or t < 0 for t in timestamps):
            raise ValueError("Demonstration timestamps must be finite nonnegative seconds")
        if any(b <= a for a, b in zip(timestamps, timestamps[1:])):
            raise ValueError("Demonstration timestamps must increase")
        timing = "Images are chronological; labels give seconds of simulated time from the first recorded frame. "
    else:
        raise ValueError("Demonstration manifest must contain images and timing only")
    blocks = [
        {
            "type": "text",
            "text": (
                "Perform the task shown in these images. "
                + timing
                + "This is a previous episode; object positions may differ. "
                "Subsequent images and feedback describe your live episode."
            ),
        }
    ]
    hashes = []
    for timestamp, name in zip(timestamps, frames):
        if not isinstance(name, str):
            raise ValueError("Each frame must be an image path")
        data = (path.parent / name).read_bytes()
        hashes.append(hashlib.sha256(data).hexdigest())
        with Image.open(io.BytesIO(data)) as image:
            image = image.convert("RGB")
            # Demonstrations use three equal camera views arranged horizontally.
            # Match the live per-camera size, rather than shrinking all three
            # cameras into the width allocated to one live image.
            image.thumbnail((3 * image_size, image_size), Image.Resampling.BILINEAR)
            output = io.BytesIO()
            image.save(output, format="JPEG", quality=jpeg_quality)
        blocks.append({"type": "text", "text": f"Demonstration time: {timestamp:.3f} s"})
        blocks.append(
            {"type": "image", "mimeType": "image/jpeg", "data": base64.b64encode(output.getvalue()).decode("ascii")}
        )
    return blocks, {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "timestamps_s": timestamps,
        "frame_sha256": hashes,
    }
