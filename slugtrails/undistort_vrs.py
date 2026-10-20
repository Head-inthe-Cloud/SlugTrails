#!/usr/bin/env python3
"""Undistort blurred Aria VRS cameras to upright pinhole MP4s.

The SlugTrails package ships ``vrs/recording.vrs`` with three blurred fisheye
cameras (color ``camera-rgb``, grayscale ``camera-slam-left`` and
``camera-slam-right``) plus an already-undistorted RGB MP4. This script
rebuilds that RGB video and writes the same kind of video for both SLAM
cameras.

The mapping matches the shipped RGB extract: a linear pinhole whose focal
length is the camera's focal length, bilinear resampling, then a 90° rotation
on Aria Gen1 so the saved frame is upright.

Usage::

  python -m slugtrails.undistort_vrs --vrs recording.vrs --out ./undistorted
  python -m slugtrails.undistort_vrs --vrs recording.vrs --streams camera-rgb
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

RGB_STREAM = "camera-rgb"
SLAM_LEFT_STREAM = "camera-slam-left"
SLAM_RIGHT_STREAM = "camera-slam-right"
DEFAULT_STREAMS = (RGB_STREAM, SLAM_LEFT_STREAM, SLAM_RIGHT_STREAM)


def _require_aria():
    try:
        from projectaria_tools.core import calibration, data_provider
        from projectaria_tools.core.calibration import DeviceVersion
        from projectaria_tools.core.image import InterpolationMethod
    except ImportError as exc:
        raise SystemExit(
            "projectaria-tools is required. Install with: pip install -e '.[vrs]'"
        ) from exc
    return calibration, data_provider, DeviceVersion, InterpolationMethod


def _to_bgr(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def _capture_timestamp_ns(image_data) -> Optional[int]:
    if len(image_data) > 1 and hasattr(image_data[1], "capture_timestamp_ns"):
        return int(image_data[1].capture_timestamp_ns)
    return None


def _open_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    for fourcc_name in ("avc1", "mp4v"):
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*fourcc_name), fps, size
        )
        if writer.isOpened():
            return writer
        writer.release()
    raise RuntimeError(f"could not open a video writer for {path}")


def _intrinsics_json(stream_label: str, linear_calib) -> dict:
    width, height = (int(v) for v in linear_calib.get_image_size())
    fx, fy = (float(v) for v in linear_calib.get_focal_lengths())
    cx, cy = (float(v) for v in linear_calib.get_principal_point())
    zeros = {"k0": 0.0, "k1": 0.0, "k2": 0.0, "k3": 0.0}
    return {
        "stream_label": stream_label,
        "resolution": {"width": width, "height": height},
        "intrinsics": {
            "focal_lengths": {"fx": fx, "fy": fy},
            "principal_point": {"cx": cx, "cy": cy},
            "model_name": "LINEAR",
            "distortion": {
                "model": "Linear",
                "note": "No distortion parameters (pinhole/linear model)",
                "radial": zeros,
                "tangential": {"p0": 0.0, "p1": 0.0},
                "thin_prism": {"s0": 0.0, "s1": 0.0, "s2": 0.0, "s3": 0.0},
            },
        },
    }


def undistort_stream(
    vrs_path: Path,
    output_dir: Path,
    stream_label: str,
    *,
    max_frames: Optional[int] = None,
) -> Path:
    """Write ``{stream}_undistorted.mp4``, timestamps, and linear intrinsics."""
    calibration, data_provider, DeviceVersion, InterpolationMethod = _require_aria()

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    if not provider:
        raise RuntimeError(f"could not open VRS: {vrs_path}")

    device_calib = provider.get_device_calibration()
    rotate = device_calib.get_device_version() == DeviceVersion.Gen1
    stream_id = provider.get_stream_id_from_label(stream_label)
    if stream_id is None:
        known = []
        for label in DEFAULT_STREAMS:
            if provider.get_stream_id_from_label(label) is not None:
                known.append(label)
        raise RuntimeError(
            f"{stream_label!r} is not in {vrs_path.name}. Present: {known or 'none'}"
        )

    src_calib = device_calib.get_camera_calib(stream_label)
    if src_calib is None:
        raise RuntimeError(f"no calibration for {stream_label}")

    image_config = provider.get_image_configuration(stream_id)
    width = int(image_config.image_width)
    height = int(image_config.image_height)
    fps = float(image_config.nominal_rate_hz)
    if fps <= 0:
        fps = 30.0

    linear_calib = calibration.get_linear_camera_calibration(
        int(src_calib.get_image_size()[0]),
        int(src_calib.get_image_size()[1]),
        src_calib.get_focal_lengths()[0],
        "pinhole",
        src_calib.get_transform_device_camera(),
    )
    out_w, out_h = (height, width) if rotate else (width, height)
    saved_calib = (
        calibration.rotate_camera_calib_cw90deg(linear_calib) if rotate else linear_calib
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / f"{stream_label}_undistorted.mp4"
    timestamps_path = output_dir / f"{stream_label}_frame_timestamps.npy"
    intrinsics_path = output_dir / f"{stream_label}_intrinsics_undistorted.json"

    num_frames = int(provider.get_num_data(stream_id))
    if max_frames is not None:
        num_frames = min(num_frames, max_frames)
    writer = _open_writer(video_path, fps, (out_w, out_h))
    timestamps: list[int] = []
    try:
        for frame_idx in range(num_frames):
            image_data = provider.get_image_data_by_index(stream_id, frame_idx)
            if image_data is None or len(image_data) == 0:
                continue
            image = image_data[0].to_numpy_array()
            undistorted = calibration.distort_by_calibration(
                image, linear_calib, src_calib, InterpolationMethod.BILINEAR
            )
            if rotate:
                undistorted = np.rot90(undistorted, k=3)
            writer.write(_to_bgr(undistorted))
            ts = _capture_timestamp_ns(image_data)
            timestamps.append(0 if ts is None else ts)
            if frame_idx == 0 or (frame_idx + 1) % 200 == 0 or frame_idx + 1 == num_frames:
                print(
                    f"  {stream_label}: {frame_idx + 1}/{num_frames}",
                    flush=True,
                )
    finally:
        writer.release()

    np.save(timestamps_path, np.asarray(timestamps, dtype=np.int64))
    intrinsics_path.write_text(json.dumps(_intrinsics_json(stream_label, saved_calib), indent=2))
    print(f"wrote {video_path} ({len(timestamps)} frames, {out_w}x{out_h})")
    return video_path


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vrs", type=Path, required=True, help="Blurred Aria .vrs")
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Directory for *_undistorted.mp4, timestamps, and intrinsics JSON",
    )
    parser.add_argument(
        "--streams",
        nargs="+",
        default=list(DEFAULT_STREAMS),
        help="Stream labels (default: camera-rgb camera-slam-left camera-slam-right)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Stop each stream after N frames (for a short check)",
    )
    args = parser.parse_args(argv)
    vrs_path = args.vrs.expanduser().resolve()
    if not vrs_path.is_file():
        print(f"ERROR: VRS not found: {vrs_path}", file=sys.stderr)
        return 1
    for label in args.streams:
        undistort_stream(
            vrs_path,
            args.out.expanduser().resolve(),
            label,
            max_frames=args.max_frames,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
