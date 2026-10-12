"""SlugTrails 1.0 package layout helpers (HF tree or local staging copy)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

# Class CSV basenames under maps/<stem>/
MAP_LAYER_FILES: dict[str, str] = {
    "walls": "walls.csv",
    "doors": "doors.csv",
    "windows": "windows.csv",
    "columns": "columns.csv",
    "stairs": "stairs.csv",
    "elevators": "elevators.csv",
    "handrails": "handrails.csv",
    "half_walls": "half_walls.csv",
    "flor_other": "flor_other.csv",
    "wall_column_elevator": "wall_column_elevator.csv",
}

DEFAULT_ROOT_CANDIDATES = (
    Path("data/hf_package/slugtrails-1.0"),
    Path("slugtrails-1.0"),
)


def resolve_package_root(root: Optional[Path] = None) -> Path:
    """Return an existing SlugTrails 1.0 package root."""
    if root is not None:
        root = Path(root).expanduser().resolve()
        if not (root / "maps").is_dir():
            raise FileNotFoundError(f"no maps/ under package root: {root}")
        return root
    for cand in DEFAULT_ROOT_CANDIDATES:
        p = cand.expanduser().resolve()
        if (p / "maps").is_dir():
            return p
    raise FileNotFoundError(
        "Could not find SlugTrails 1.0 package. Pass --root "
        "(e.g. data/hf_package/slugtrails-1.0 or an HF download dir)."
    )


def list_scene_stems(root: Path) -> list[str]:
    maps = root / "maps"
    stems = sorted(
        p.name
        for p in maps.iterdir()
        if p.is_dir() and not p.name.startswith(".") and (p / "SCENE.json").is_file()
    )
    if stems:
        return stems
    # fallback: dirs with walls.csv
    return sorted(
        p.name
        for p in maps.iterdir()
        if p.is_dir() and (p / "walls.csv").is_file()
    )


def scene_dir(root: Path, stem: str) -> Path:
    return root / "maps" / stem


def load_scene_json(root: Path, stem: str) -> dict[str, Any]:
    path = scene_dir(root, stem) / "SCENE.json"
    if not path.is_file():
        return {"stem": stem, "recordings": [], "files": []}
    return json.loads(path.read_text())


def recording_dir(root: Path, recording_id: str) -> Path:
    return root / "recordings" / recording_id


def gt_positions_path(root: Path, recording_id: str) -> Path:
    return recording_dir(root, recording_id) / "gt_aligned" / "trajectory_positions.npy"


def recordings_for_stem(root: Path, stem: str) -> list[str]:
    scene = load_scene_json(root, stem)
    recs = list(scene.get("recordings") or [])
    if recs:
        return [r for r in recs if recording_dir(root, r).is_dir()]
    # fallback: recording ids that start with stem + "_"
    rec_root = root / "recordings"
    if not rec_root.is_dir():
        return []
    return sorted(
        p.name
        for p in rec_root.iterdir()
        if p.is_dir() and (p.name == stem or p.name.startswith(stem + "_"))
    )


def list_recording_ids(root: Path) -> list[str]:
    rec_root = root / "recordings"
    if not rec_root.is_dir():
        return []
    return sorted(
        p.name
        for p in rec_root.iterdir()
        if p.is_dir() and (p / "RECORDING.json").is_file()
    )


def load_recording_json(root: Path, recording_id: str) -> dict[str, Any]:
    path = recording_dir(root, recording_id) / "RECORDING.json"
    return json.loads(path.read_text())


def rgb_video_path(root: Path, recording_id: str) -> Path:
    return recording_dir(root, recording_id) / "rgb" / "camera-rgb_undistorted.mp4"


def rgb_timestamps_path(root: Path, recording_id: str) -> Path:
    return (
        recording_dir(root, recording_id) / "rgb" / "camera-rgb_frame_timestamps.npy"
    )


def mps_closed_csv_path(root: Path, recording_id: str) -> Path:
    return (
        recording_dir(root, recording_id)
        / "mps"
        / "closed"
        / "closed_loop_trajectory.csv"
    )


# Prefer package ``mps/slam/semidense_points.csv.gz``; else local Aria MPS tree.
_DEFAULT_ARIA_MPS_REL: dict[str, str] = {
    "BE_0_0": "BE/mps_BE_0_0_bri_vrs",
    "BE_1_0": "BE/mps_BE_1_0_vrs",
    "BE_2_0": "BE/mps_BE_2_0_mat_vrs",
    "E2_2_0": "E2/mps_E2_2_0_Bri_vrs",
    "E2_3_0": "E2/mps_E2-3-0_vrs",
    "PS_2_0": "PS/mps_PS_2_0_mat_vrs",
}


def resolve_vrs_path(
    recording_id: str,
    *,
    aria_root: Optional[Path] = None,
    explicit: Optional[Path] = None,
) -> Optional[Path]:
    """Local Aria ``.vrs`` for device/camera calibration (not shipped on HF 1.0)."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None

    import os

    candidates: list[Path] = []
    if aria_root is not None:
        candidates.append(Path(aria_root))
    env = os.environ.get("SLUGTRAILS_ARIA_ROOT") or os.environ.get(
        "SLUGTRAILS_RAW_DATA_ROOT"
    )
    if env:
        candidates.append(Path(env).expanduser())
    # Optional local Aria tree via SLUGTRAILS_ARIA_ROOT / SLUGTRAILS_RAW_DATA_ROOT

    prefix = recording_id.split("_")[0]  # BE, E2, PS
    for base in candidates:
        if not base.is_dir():
            continue
        scene = base / prefix
        if scene.is_dir():
            matches = sorted(scene.glob(f"{recording_id}*.vrs"))
            if matches:
                return matches[0]
        for p in base.rglob(f"{recording_id}*.vrs"):
            if p.is_file():
                return p
    return None


def resolve_semidense_points(
    root: Path,
    recording_id: str,
    *,
    explicit: Optional[Path] = None,
    aria_root: Optional[Path] = None,
) -> Optional[Path]:
    """Find MPS ``semidense_points.csv.gz`` (Aria Studio / viewer_mps format).

    Order: ``--points``, package-local path, ``SLUGTRAILS_ARIA_ROOT`` / default
    Aria tree, ``source_dir_name`` under common roots.
    """
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None

    local = (
        recording_dir(root, recording_id)
        / "mps"
        / "slam"
        / "semidense_points.csv.gz"
    )
    if local.is_file():
        return local

    import os

    candidates: list[Path] = []
    if aria_root is not None:
        candidates.append(Path(aria_root))
    env = os.environ.get("SLUGTRAILS_ARIA_ROOT") or os.environ.get(
        "SLUGTRAILS_RAW_DATA_ROOT"
    )
    if env:
        candidates.append(Path(env).expanduser())
    # Optional local Aria tree via SLUGTRAILS_ARIA_ROOT / SLUGTRAILS_RAW_DATA_ROOT

    rel = _DEFAULT_ARIA_MPS_REL.get(recording_id)
    card = load_recording_json(root, recording_id)
    mps_name = (card.get("mps") or {}).get("source_dir_name")

    for base in candidates:
        if not base.is_dir():
            continue
        if rel:
            p = base / rel / "slam" / "semidense_points.csv.gz"
            if p.is_file():
                return p
        if mps_name:
            for p in base.rglob(f"{mps_name}/slam/semidense_points.csv.gz"):
                return p
    return None


def resolve_pointcloud_npy(
    root: Path,
    recording_id: str,
    *,
    explicit: Optional[Path] = None,
    processed_root: Optional[Path] = None,
) -> Optional[Path]:
    """Optional ``pointcloud.npy`` (SlugTrails extract; not in HF 1.0)."""
    if explicit is not None:
        p = Path(explicit).expanduser().resolve()
        return p if p.is_file() else None
    packaged = recording_dir(root, recording_id) / "pointcloud.npy"
    if packaged.is_file():
        return packaged
    import os

    proc = processed_root
    if proc is None:
        env = os.environ.get("SLUGTRAILS_PROCESSED_ROOT")
        proc = Path(env) if env else root.parent.parent / "processed"
        # root = .../hf_package/slugtrails-1.0 → parent.parent = data/
        if not (Path(proc) / recording_id).is_dir():
            proc = Path("data/processed")
    p = Path(proc) / recording_id / "extracted" / "pointcloud.npy"
    return p if p.is_file() else None
