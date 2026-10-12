"""2D floor-plan layer for ``view_rerun`` (map frame, separate from MPS ``world/``)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from slugtrails.paths import (
    MAP_LAYER_FILES,
    gt_positions_path,
    load_recording_json,
    recording_dir,
    scene_dir,
)
from slugtrails.view_map import load_segments_csv

# Layers omitted from 2D map view (per public viewer toggles).
MAP_EXCLUDED_LAYERS = frozenset(
    {"half_walls", "flor_other", "wall_column_elevator", "outline"}
)

# Doors/windows highlighted; everything else one neutral CAD color.
MAP_DEFAULT_COLOR = (60, 60, 60, 255)
MAP_LAYER_COLORS: dict[str, tuple[int, int, int, int]] = {
    "doors": (220, 50, 50, 255),
    "windows": (40, 140, 220, 255),
}

# Match Aria MPS point-cloud green in the 3D view.
GT_TRAJ_COLOR = (50, 200, 80, 220)
GT_TRAJ_RADIUS = 0.04
CAMERA_COLOR = (255, 140, 0, 255)
CAMERA_RADIUS = 0.08
# Aria RGB linear (after Gen1 CW90) ≈ 98.22°; override from VRS when available.
DEFAULT_HFOV_RAD = float(np.deg2rad(98.22))
FRUSTUM_RANGE_M = 1.2


def floor_stem_for_recording(root: Path, recording_id: str) -> str:
    card = load_recording_json(root, recording_id)
    stem = card.get("floor_stem")
    if stem:
        return str(stem)
    parts = recording_id.rsplit("_", 1)
    return parts[0] if len(parts) == 2 else recording_id


def to_rerun_xy(xy: np.ndarray) -> np.ndarray:
    """Map meters (+Y up, origin bottom-left) → Rerun Spatial2D (+Y down).

    Negate Y so the floor plan is not vertically flipped in the viewer.
    """
    out = np.asarray(xy, dtype=np.float64).copy()
    out[..., 1] *= -1.0
    return out


def _segments_to_strips(segments: np.ndarray) -> list[np.ndarray]:
    if segments is None or len(segments) == 0:
        return []
    return [to_rerun_xy(seg) for seg in segments]


def load_gt_transforms(
    root: Path, recording_id: str
) -> tuple[np.ndarray, np.ndarray]:
    base = recording_dir(root, recording_id) / "gt_aligned"
    T = np.load(base / "trajectory_transforms.npy")
    ts = np.load(base / "trajectory_timestamps.npy")
    if len(T) != len(ts):
        raise ValueError(f"gt length mismatch {len(T)} vs {len(ts)}")
    return T, ts


def nearest_gt_index(timestamps_ns: np.ndarray, query_ts_ns: int) -> int:
    idx = int(np.searchsorted(timestamps_ns, query_ts_ns))
    idx = int(np.clip(idx, 0, len(timestamps_ns) - 1))
    if idx > 0 and (
        idx >= len(timestamps_ns)
        or abs(timestamps_ns[idx - 1] - query_ts_ns)
        < abs(timestamps_ns[idx] - query_ts_ns)
    ):
        return idx - 1
    return idx


def look_yaw_xy(T_world: np.ndarray) -> float:
    """Planar look heading (rad) from camera optical axis (+Z) projected to XY.

    Matches ``pose_to_ypr`` / paper floor-plan frustum: ``gt_aligned`` is Aria
    ``T_world_camera``, +Z forward. (Old arrow put the fat base *behind* the
    pose, which looked 180° wrong; FOV wedge opens *along* this yaw.)
    """
    fwd = np.asarray(T_world[:3, 2], dtype=np.float64)
    return float(np.arctan2(fwd[1], fwd[0]))


def hfov_from_pinhole(image_width: float, fx: float) -> float:
    """Horizontal FOV (rad): ``2 * atan(W / (2 fx))``."""
    return float(2.0 * np.arctan(float(image_width) / (2.0 * float(fx))))


def camera_triangle_xy(
    T_world: np.ndarray,
    *,
    hfov_rad: float = DEFAULT_HFOV_RAD,
    range_m: float = FRUSTUM_RANGE_M,
) -> np.ndarray:
    """Closed FOV wedge in map XY: apex at pose, opening = ``hfov_rad``."""
    x, y = float(T_world[0, 3]), float(T_world[1, 3])
    yaw = look_yaw_xy(T_world)
    half = 0.5 * float(hfov_rad)
    a0, a1 = yaw - half, yaw + half
    apex = np.array([x, y], dtype=np.float64)
    left = apex + range_m * np.array([np.cos(a0), np.sin(a0)])
    right = apex + range_m * np.array([np.cos(a1), np.sin(a1)])
    return np.stack([apex, left, right, apex], axis=0)


def send_map_blueprint(*, show_map: bool = True, show_video: bool = True) -> None:
    """Default: MPS left (wide); Floor plan + Video stacked on the right."""
    import rerun as rr
    import rerun.blueprint as rrb

    right: list = []
    if show_map:
        right.append(rrb.Spatial2DView(origin="/map", name="Floor plan"))
    if show_video:
        right.append(rrb.Spatial2DView(origin="/video", name="Video"))

    mps = rrb.Spatial3DView(origin="/world", name="MPS")
    if not right:
        root = mps
    elif len(right) == 1:
        root = rrb.Horizontal(mps, right[0], column_shares=[3, 1])
    else:
        root = rrb.Horizontal(
            mps,
            rrb.Vertical(*right, row_shares=[1] * len(right), name="Vertical"),
            column_shares=[3, 1],
        )

    rr.send_blueprint(
        rrb.Blueprint(
            root,
            rrb.TimePanel(state="expanded"),
            rrb.SelectionPanel(state="collapsed"),
        ),
        make_active=True,
        make_default=True,
    )


def log_static_map(root: Path, recording_id: str) -> str:
    """Log CAD layers + gt path under ``/map`` (static). Returns floor stem."""
    import rerun as rr

    stem = floor_stem_for_recording(root, recording_id)
    sdir = scene_dir(root, stem)

    for layer, fname in MAP_LAYER_FILES.items():
        if layer in MAP_EXCLUDED_LAYERS:
            continue
        strips = _segments_to_strips(load_segments_csv(sdir / fname))
        if not strips:
            continue
        color = MAP_LAYER_COLORS.get(layer, MAP_DEFAULT_COLOR)
        rr.log(
            f"/map/layers/{layer}",
            rr.LineStrips2D(strips, colors=color, radii=0.06),
            static=True,
        )

    gt_xy = np.load(gt_positions_path(root, recording_id))
    if gt_xy.ndim == 2 and gt_xy.shape[0] >= 2 and gt_xy.shape[1] >= 2:
        # Downsample a bit for thin polyline performance; keep shape.
        xy = to_rerun_xy(gt_xy[:, :2].astype(np.float64))
        if len(xy) > 4000:
            xy = xy[:: max(1, len(xy) // 4000)]
        rr.log(
            "/map/gt_trajectory",
            rr.LineStrips2D(
                [xy],
                colors=GT_TRAJ_COLOR,
                radii=GT_TRAJ_RADIUS,
            ),
            static=True,
        )

    print(
        f"  map: stem={stem}, "
        f"layers={len(MAP_LAYER_FILES) - len(MAP_EXCLUDED_LAYERS)} "
        f"(doors/windows colored; traj green thin)"
    )
    return stem


def log_map_camera_at_time(
    T_world: np.ndarray,
    *,
    hfov_rad: float = DEFAULT_HFOV_RAD,
) -> None:
    import rerun as rr

    tri = to_rerun_xy(camera_triangle_xy(T_world, hfov_rad=hfov_rad))
    apex = tri[0]
    rr.log(
        "/map/camera",
        rr.LineStrips2D([tri], colors=CAMERA_COLOR, radii=CAMERA_RADIUS),
    )
    rr.log(
        "/map/camera_apex",
        rr.Points2D([apex], colors=CAMERA_COLOR, radii=0.12),
    )
