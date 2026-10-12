#!/usr/bin/env python3
"""Rerun viewer for SlugTrails 1.0 — blurred RGB + MPS point cloud (+ closed traj).

Shortest path toward Aria Studio / ``viewer_mps``:

1. Call Aria's ``log_mps_to_rerun`` with **no VRS** so it logs closed-loop
   trajectory + semidense points (same helpers Aria Studio uses).
2. Stream **anonymized** ``camera-rgb_undistorted.mp4`` from the 1.0 package
   (VRS is not public; stock Aria would show raw faces). Per-frame **frustum**
   uses the same Aria helpers as Studio: static ``Pinhole`` + ``log_camera_pose``,
   with intrinsics/extrinsics read from a **local** ``.vrs`` (images still blurred MP4).

Semidense points load from the package path
``recordings/<id>/mps/slam/semidense_points.csv.gz``. A local Aria ``.vrs``
(``--vrs`` or ``SLUGTRAILS_ARIA_ROOT``) is optional and used only for the
3D camera frustum; RGB stays the anonymized MP4.

Triple-view blueprint: **MPS** ``/world`` | **floor plan** ``/map`` | **video**
``/video``. Map CSVs skip ``half_walls`` / ``flor_other`` /
``wall_column_elevator`` / ``outline``; doors+windows colored; green thin
``gt_aligned`` traj; orange camera triangle follows video time.

Usage::

  python -m slugtrails.view_rerun --recording BE_0_0
  python -m slugtrails.view_rerun --recording BE_0_0 --max-frames 300 --stride 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from slugtrails.paths import (
    list_recording_ids,
    mps_closed_csv_path,
    resolve_package_root,
    resolve_pointcloud_npy,
    resolve_semidense_points,
    resolve_vrs_path,
    rgb_timestamps_path,
    rgb_video_path,
)
from slugtrails.rerun_map import (
    DEFAULT_HFOV_RAD,
    hfov_from_pinhole,
    load_gt_transforms,
    log_map_camera_at_time,
    log_static_map,
    nearest_gt_index,
    send_map_blueprint,
)


def _log_npy_pointcloud(path: Path, *, max_points: int = 500_000) -> None:
    import rerun as rr

    pts = np.load(path)
    if pts.ndim != 2 or pts.shape[1] < 3:
        raise ValueError(f"expected (N,3+) point cloud, got {pts.shape} in {path}")
    xyz = pts[:, :3]
    if len(xyz) > max_points:
        idx = np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64)
        xyz = xyz[idx]
    rr.log("world/points", rr.Points3D(xyz, radii=0.01), static=True)
    print(f"logged pointcloud.npy → world/points ({len(xyz)} pts) from {path}")


def _load_rgb_camera_calibration(vrs_path: Path):
    """Pinhole calib matching SlugTrails upright undistorted MP4.

    Extract / Aria Gen1 path: linear (undistorted) calib, then
    ``rotate_camera_calib_cw90deg`` so frustum roll matches the MP4 (already
    ``np.rot90(..., k=3)`` in ``aria_processor/video_converter.py``).
    """
    from projectaria_tools.core import calibration, data_provider
    from projectaria_tools.core.calibration import DeviceVersion
    from projectaria_tools.core.stream_id import StreamId

    provider = data_provider.create_vrs_data_provider(str(vrs_path))
    device_calibration = provider.get_device_calibration()
    rgb_stream_id = StreamId("214-1")
    rgb_stream_label = provider.get_label_from_stream_id(rgb_stream_id)
    rgb_camera_calibration = device_calibration.get_camera_calib(rgb_stream_label)
    rgb_linear = calibration.get_linear_camera_calibration(
        int(rgb_camera_calibration.get_image_size()[0]),
        int(rgb_camera_calibration.get_image_size()[1]),
        rgb_camera_calibration.get_focal_lengths()[0],
        "pinhole",
        rgb_camera_calibration.get_transform_device_camera(),
    )
    # HF / extract video is display-upright; Gen1 raw sensor is sideways.
    if device_calibration.get_device_version() == DeviceVersion.Gen1:
        rgb_linear = calibration.rotate_camera_calib_cw90deg(rgb_linear)
    return device_calibration, rgb_stream_label, rgb_linear


def _hfov_from_calib(camera_calibration) -> float:
    w = float(camera_calibration.get_image_size()[0])
    fx = float(camera_calibration.get_focal_lengths()[0])
    return hfov_from_pinhole(w, fx)


def _log_blurred_rgb(
    video: Path,
    timestamps: Path,
    traj_csv: Path,
    vrs_path: Optional[Path],
    *,
    gt_transforms: Optional[np.ndarray] = None,
    gt_timestamps: Optional[np.ndarray] = None,
    log_map_camera: bool = False,
    stride: int,
    max_frames: Optional[int],
    jpeg_quality: int,
    image_stride: int,
) -> int:
    """Log blurred RGB + per-frame device/camera pose (Aria ``viewer_mps`` parity)."""
    import rerun as rr
    from projectaria_tools.core import mps
    from projectaria_tools.tools.viewer_mps.rerun_viewer_mps import (
        log_Aria_glasses_outline,
        log_RGB_camera_calibration,
        log_camera_pose,
    )

    trajectory_data = mps.read_closed_loop_trajectory(str(traj_csv))
    if trajectory_data is not None and len(trajectory_data) == 0:
        trajectory_data = mps.read_open_loop_trajectory(str(traj_csv))

    rgb_stream_label = "camera-rgb"
    device_calibration = None
    camera_calibration = None
    map_hfov = DEFAULT_HFOV_RAD
    if vrs_path is not None and vrs_path.is_file():
        device_calibration, rgb_stream_label, camera_calibration = (
            _load_rgb_camera_calibration(vrs_path)
        )
        log_RGB_camera_calibration(
            camera_calibration, rgb_stream_label, image_stride
        )
        log_Aria_glasses_outline(device_calibration)
        map_hfov = _hfov_from_calib(camera_calibration)
        print(
            f"  camera calib + glasses outline from {vrs_path.name} "
            f"(map HFOV={np.degrees(map_hfov):.1f}°)"
        )
    else:
        print(
            "WARNING: no local .vrs — logging RGB only (no 3D frustum); "
            f"map wedge uses default HFOV={np.degrees(map_hfov):.1f}°. "
            "Set SLUGTRAILS_ARIA_ROOT or pass --vrs.",
            flush=True,
        )

    ts = np.load(timestamps)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {video}")
    n_vid = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    n = min(len(ts), n_vid) if n_vid > 0 else len(ts)
    logged = 0
    try:
        for i in range(0, n, max(1, stride)):
            if max_frames is not None and logged >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, bgr = cap.read()
            if not ok:
                break
            if image_stride > 1:
                bgr = bgr[::image_stride, ::image_stride]
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            t_ns = int(ts[i])
            rr.set_time_nanos("device_time", t_ns)
            rr.set_time_sequence("timestamp", t_ns)
            if camera_calibration is not None and trajectory_data:
                log_camera_pose(
                    trajectory_data,
                    t_ns,
                    camera_calibration,
                    rgb_stream_label,
                )
            compressed = rr.Image(rgb).compress(jpeg_quality=jpeg_quality)
            # Frustum-attached image (3D) + dedicated video pane entity.
            rr.log(f"world/device/{rgb_stream_label}", compressed)
            rr.log("/video/rgb", compressed)
            if (
                log_map_camera
                and gt_transforms is not None
                and gt_timestamps is not None
            ):
                gi = nearest_gt_index(gt_timestamps, t_ns)
                log_map_camera_at_time(gt_transforms[gi], hfov_rad=map_hfov)
            logged += 1
            if logged % 50 == 0:
                print(f"  rgb frames logged: {logged}", flush=True)
    finally:
        cap.release()
    print(f"logged blurred RGB + pose: {logged} frames from {video.name}")
    return logged


def run(
    root: Path,
    recording_id: str,
    *,
    points: Optional[Path] = None,
    pointcloud_npy: Optional[Path] = None,
    aria_root: Optional[Path] = None,
    vrs: Optional[Path] = None,
    stride: int = 10,
    max_frames: Optional[int] = None,
    jpeg_quality: int = 75,
    image_stride: int = 2,
    web: bool = False,
    rrd_output: Optional[Path] = None,
    skip_rgb: bool = False,
    skip_points: bool = False,
    skip_map: bool = False,
) -> int:
    import rerun as rr

    try:
        from projectaria_tools.tools.viewer_mps.rerun_viewer_mps import (
            log_mps_to_rerun,
        )
    except ImportError as e:
        print(
            "ERROR: need projectaria-tools with viewer_mps "
            f"(import failed: {e})",
            file=sys.stderr,
        )
        return 1

    video = rgb_video_path(root, recording_id)
    ts_path = rgb_timestamps_path(root, recording_id)
    traj_csv = mps_closed_csv_path(root, recording_id)
    if not video.is_file():
        raise FileNotFoundError(video)
    if not ts_path.is_file():
        raise FileNotFoundError(ts_path)
    if not traj_csv.is_file():
        raise FileNotFoundError(traj_csv)

    semidense = None if skip_points else resolve_semidense_points(
        root, recording_id, explicit=points, aria_root=aria_root
    )
    npy_pcd = None
    if not skip_points and semidense is None:
        npy_pcd = resolve_pointcloud_npy(
            root, recording_id, explicit=pointcloud_npy
        )

    print("SlugTrails Rerun viewer (Aria-style MPS + blurred 1.0 RGB)")
    print(f"  root:       {root}")
    print(f"  recording:  {recording_id}")
    print(f"  video:      {video}")
    print(f"  trajectory: {traj_csv}")
    vrs_path = resolve_vrs_path(recording_id, aria_root=aria_root, explicit=vrs)
    print(f"  semidense:  {semidense or '(none)'}")
    print(f"  pointcloud: {npy_pcd or '(none)'}")
    print(f"  vrs (calib): {vrs_path or '(none)'}")
    if not skip_points and semidense is None and npy_pcd is None:
        print(
            "WARNING: no point cloud found. Expected "
            "recordings/<id>/mps/slam/semidense_points.csv.gz in the package, "
            "or pass --points. Continuing with traj+RGB.",
            flush=True,
        )

    # Mirror Aria ``viewer_mps.py``: init → serve/spawn → log_mps_to_rerun.
    # We pass ``rrd_output_path`` into log_mps so Aria's save path stays intact;
    # for desktop we spawn here (Aria does ``rr.spawn()`` after init).
    rr.init("SlugTrails 1.0", spawn=(rrd_output is None and not web))
    if web:
        rr.serve_web()
    elif rrd_output is None:
        rr.spawn()

    gt_T: Optional[np.ndarray] = None
    gt_ts: Optional[np.ndarray] = None
    send_map_blueprint(show_map=not skip_map, show_video=not skip_rgb)
    if not skip_map:
        log_static_map(root, recording_id)
        gt_T, gt_ts = load_gt_transforms(root, recording_id)

    # Same entry as Aria Studio / viewer_mps — no VRS (we log blurred RGB next).
    log_mps_to_rerun(
        vrs_path=None,
        trajectory_files=[str(traj_csv)],
        points_files=[str(semidense)] if semidense is not None else [],
        eye_gaze_file=None,
        wrist_and_palm_poses_file=None,
        hand_tracking_results_file=None,
        should_rectify_image=False,
        down_sampling_factor=image_stride,
        jpeg_quality=jpeg_quality,
        rrd_output_path=str(rrd_output) if rrd_output is not None else None,
    )

    if npy_pcd is not None and semidense is None:
        _log_npy_pointcloud(npy_pcd)

    if not skip_rgb:
        _log_blurred_rgb(
            video,
            ts_path,
            traj_csv,
            vrs_path,
            gt_transforms=gt_T,
            gt_timestamps=gt_ts,
            log_map_camera=not skip_map,
            stride=stride,
            max_frames=max_frames,
            jpeg_quality=jpeg_quality,
            image_stride=image_stride,
        )

    if web:
        import time

        print("Web viewer running — Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down.")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument(
        "--recording",
        type=str,
        default="BE_0_0",
        help="Recording id under recordings/",
    )
    ap.add_argument(
        "--points",
        type=Path,
        default=None,
        help="MPS semidense_points.csv.gz (default: recordings/<id>/mps/slam/ in the package)",
    )
    ap.add_argument(
        "--pointcloud-npy",
        type=Path,
        default=None,
        help="Fallback pointcloud.npy if semidense CSV unavailable",
    )
    ap.add_argument(
        "--aria-root",
        type=Path,
        default=None,
        help="Root containing BE|E2|PS/mps_*/slam/ (or set SLUGTRAILS_ARIA_ROOT)",
    )
    ap.add_argument(
        "--vrs",
        type=Path,
        default=None,
        help="Local Aria .vrs for camera frustum (calib only; RGB stays blurred MP4)",
    )
    ap.add_argument("--stride", type=int, default=10, help="Keep every Nth RGB frame")
    ap.add_argument("--max-frames", type=int, default=None, help="Cap logged RGB frames")
    ap.add_argument("--jpeg-quality", type=int, default=75)
    ap.add_argument(
        "--image-stride",
        type=int,
        default=2,
        help="Spatial downsample (same idea as Aria down_sampling_factor)",
    )
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--rrd", type=Path, default=None, help="Save .rrd instead of spawn")
    ap.add_argument("--skip-rgb", action="store_true")
    ap.add_argument("--skip-points", action="store_true")
    ap.add_argument(
        "--skip-map",
        action="store_true",
        help="Disable 2D floor-plan view and map-frame camera triangle",
    )
    ap.add_argument(
        "--list",
        action="store_true",
        help="List recording ids and exit",
    )
    args = ap.parse_args(argv)
    root = resolve_package_root(args.root)
    if args.list:
        for rid in list_recording_ids(root):
            print(rid)
        return 0
    recs = list_recording_ids(root)
    if args.recording not in recs:
        print(f"ERROR: {args.recording!r} not in {recs}", file=sys.stderr)
        return 1
    return run(
        root,
        args.recording,
        points=args.points,
        pointcloud_npy=args.pointcloud_npy,
        aria_root=args.aria_root,
        vrs=args.vrs,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        jpeg_quality=args.jpeg_quality,
        image_stride=max(1, args.image_stride),
        web=args.web,
        rrd_output=args.rrd,
        skip_rgb=args.skip_rgb,
        skip_points=args.skip_points,
        skip_map=args.skip_map,
    )


if __name__ == "__main__":
    raise SystemExit(main())
