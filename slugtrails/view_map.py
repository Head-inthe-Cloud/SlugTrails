#!/usr/bin/env python3
"""Official 2D floor-plan viewer for SlugTrails 1.0 (HF package layout).

View-only. Scene radio, per-layer CSV toggles, optional ``gt_aligned`` trajectory.

Usage (any cwd; pass --root unless the package sits at a default path)::

  python -m slugtrails.view_map
  python -m slugtrails.view_map --root data/hf_package/slugtrails-1.0 --stem BE_2
  python -m slugtrails.view_map --smoke   # headless load check
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np

from slugtrails.paths import (
    MAP_LAYER_FILES,
    gt_positions_path,
    list_scene_stems,
    load_scene_json,
    recordings_for_stem,
    resolve_package_root,
    scene_dir,
)

# Draw order / default visibility (label → on)
LAYER_ORDER = (
    "walls",
    "doors",
    "windows",
    "columns",
    "stairs",
    "elevators",
    "handrails",
    "half_walls",
    "flor_other",
    "wall_column_elevator",
    "outline",
    "gt_trajectory",
)

LAYER_COLORS: dict[str, str] = {
    "walls": "#222222",
    "doors": "#d62728",
    "windows": "#2ca02c",
    "columns": "#7f7f7f",
    "stairs": "#9467bd",
    "elevators": "#8c564b",
    "handrails": "#e377c2",
    "half_walls": "#ff7f0e",
    "flor_other": "#bcbd22",
    "wall_column_elevator": "#1f77b4",
    "outline": "#17becf",
    "gt_trajectory": "#e41a1c",
}

DEFAULT_ON = {
    "walls": True,
    "doors": True,
    "windows": True,
    "columns": False,
    "stairs": False,
    "elevators": False,
    "handrails": False,
    "half_walls": False,
    "flor_other": False,
    "wall_column_elevator": False,
    "outline": True,
    "gt_trajectory": True,
}


def load_segments_csv(path: Path) -> np.ndarray:
    """Headerless x1,y1,x2,y2 → (N, 2, 2)."""
    if not path.is_file():
        return np.zeros((0, 2, 2), dtype=np.float64)
    rows: list[np.ndarray] = []
    with path.open() as f:
        for row in csv.reader(f):
            if len(row) < 4:
                continue
            try:
                x1, y1, x2, y2 = map(float, row[:4])
            except ValueError:
                continue
            rows.append(np.array([[x1, y1], [x2, y2]], dtype=np.float64))
    if not rows:
        return np.zeros((0, 2, 2), dtype=np.float64)
    return np.stack(rows, axis=0)


def load_outline_rings(path: Path) -> list[np.ndarray]:
    """Return list of (M, 2) rings (exterior + holes)."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    rings: list[np.ndarray] = []
    ext = data.get("exterior") or data.get("outline")
    if ext:
        rings.append(np.asarray(ext, dtype=np.float64))
    for hole in data.get("holes") or []:
        rings.append(np.asarray(hole, dtype=np.float64))
    return [r for r in rings if r.ndim == 2 and r.shape[0] >= 2]


def load_gt_xy(
    root: Path,
    recording_id: str,
    *,
    downsample: int = 30,
) -> np.ndarray:
    path = gt_positions_path(root, recording_id)
    if not path.is_file():
        return np.zeros((0, 2), dtype=np.float64)
    xyz = np.load(path)
    if xyz.ndim != 2 or xyz.shape[1] < 2:
        raise ValueError(f"bad gt positions shape {xyz.shape} in {path}")
    xy = xyz[:, :2]
    if downsample > 1:
        xy = xy[::downsample]
    return np.asarray(xy, dtype=np.float64)


def load_scene_layers(root: Path, stem: str) -> dict[str, Any]:
    sdir = scene_dir(root, stem)
    layers: dict[str, Any] = {}
    for name, fname in MAP_LAYER_FILES.items():
        layers[name] = load_segments_csv(sdir / fname)
    layers["outline"] = load_outline_rings(sdir / "outline.json")
    recs = recordings_for_stem(root, stem)
    layers["_recordings"] = recs
    layers["_recording"] = recs[0] if recs else None
    if layers["_recording"]:
        layers["gt_trajectory"] = load_gt_xy(root, layers["_recording"])
    else:
        layers["gt_trajectory"] = np.zeros((0, 2), dtype=np.float64)
    layers["_scene"] = load_scene_json(root, stem)
    return layers


def _seg_lines(arr: np.ndarray) -> list:
    if arr is None or len(arr) == 0:
        return []
    return [((float(s[0, 0]), float(s[0, 1])), (float(s[1, 0]), float(s[1, 1]))) for s in arr]


def smoke(root: Path, stem: Optional[str] = None) -> int:
    stems = list_scene_stems(root)
    if not stems:
        print(f"ERROR: no scenes under {root / 'maps'}", file=sys.stderr)
        return 1
    stem = stem or stems[0]
    if stem not in stems:
        print(f"ERROR: stem {stem!r} not in {stems}", file=sys.stderr)
        return 1
    layers = load_scene_layers(root, stem)
    print(f"root={root}")
    print(f"scenes={stems}")
    print(f"stem={stem} recordings={layers['_recordings']}")
    for name in LAYER_ORDER:
        if name == "outline":
            print(f"  {name}: {len(layers['outline'])} rings")
        elif name == "gt_trajectory":
            print(f"  {name}: {len(layers['gt_trajectory'])} pts "
                  f"(rec={layers['_recording']})")
        else:
            print(f"  {name}: {len(layers[name])} segs")
    return 0


def run_viewer(
    root: Path,
    *,
    stem: Optional[str] = None,
    traj_downsample: int = 30,
) -> int:
    import matplotlib

    if matplotlib.get_backend().lower() == "agg":
        try:
            matplotlib.use("TkAgg")
        except Exception:
            try:
                matplotlib.use("MacOSX")
            except Exception:
                pass

    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.widgets import CheckButtons, RadioButtons

    stems = list_scene_stems(root)
    if not stems:
        raise SystemExit(f"no scenes under {root / 'maps'}")
    if stem is None or stem not in stems:
        stem = stems[0]

    st: dict[str, Any] = {
        "stem": stem,
        "layers": load_scene_layers(root, stem),
        "visible": {k: DEFAULT_ON.get(k, False) for k in LAYER_ORDER},
        "traj_downsample": traj_downsample,
    }
    # Reload traj with requested downsample
    rec = st["layers"]["_recording"]
    if rec:
        st["layers"]["gt_trajectory"] = load_gt_xy(
            root, rec, downsample=traj_downsample
        )

    fig, ax = plt.subplots(figsize=(12, 9))
    plt.subplots_adjust(left=0.05, right=0.72, bottom=0.06, top=0.94)

    collections: dict[str, LineCollection] = {}
    for name in LAYER_ORDER:
        if name in ("outline", "gt_trajectory"):
            continue
        lw = 1.4 if name == "walls" else 0.9
        alpha = 0.35 if name == "wall_column_elevator" else 0.9
        coll = LineCollection(
            [],
            colors=LAYER_COLORS[name],
            linewidths=lw,
            alpha=alpha,
            zorder=2,
        )
        ax.add_collection(coll)
        collections[name] = coll

    outline_lines: list = []
    traj_line = ax.plot([], [], "-", color=LAYER_COLORS["gt_trajectory"], lw=1.2, zorder=4)[0]
    traj_start = ax.plot([], [], "o", color="#4daf4a", ms=7, zorder=5, label="start")[0]
    traj_end = ax.plot([], [], "s", color="#984ea3", ms=7, zorder=5, label="end")[0]
    title = ax.set_title("")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")

    # Scene radio
    n_stem = max(len(stems), 1)
    h_radio = min(0.45, 0.04 * n_stem + 0.08)
    rax = fig.add_axes([0.74, 0.92 - h_radio, 0.24, h_radio])
    rax.set_title("Scene", fontsize=9)
    map_radio = RadioButtons(rax, stems, active=stems.index(st["stem"]))

    # Layer checks (two columns of labels for height)
    labels = list(LAYER_ORDER)
    actives = [st["visible"][k] for k in labels]
    h_check = min(0.55, 0.028 * len(labels) + 0.06)
    cax = fig.add_axes([0.74, 0.92 - h_radio - h_check - 0.02, 0.24, h_check])
    cax.set_title("Layers", fontsize=9)
    checks = CheckButtons(cax, labels, actives)

    help_ax = fig.add_axes([0.74, 0.02, 0.24, 0.12])
    help_ax.axis("off")
    help_ax.text(
        0,
        0.5,
        "View-only · map meters\n"
        "gt_trajectory = gt_aligned XY\n"
        "green○ start · purple□ end",
        fontsize=8,
        va="center",
        family="monospace",
    )

    def _clear_outline() -> None:
        for ln in outline_lines:
            ln.remove()
        outline_lines.clear()

    def _fit() -> None:
        pts: list[list[float]] = []
        for name in LAYER_ORDER:
            if not st["visible"].get(name, False):
                continue
            if name == "outline":
                for ring in st["layers"]["outline"]:
                    pts.extend(ring.tolist())
            elif name == "gt_trajectory":
                xy = st["layers"]["gt_trajectory"]
                if len(xy):
                    pts.extend(xy.tolist())
            else:
                arr = st["layers"][name]
                for seg in arr:
                    pts.append(seg[0].tolist())
                    pts.append(seg[1].tolist())
        if not pts:
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
            return
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        pad = 2.0
        ax.set_xlim(min(xs) - pad, max(xs) + pad)
        ax.set_ylim(min(ys) - pad, max(ys) + pad)

    def redraw(*, refit: bool = False) -> None:
        for name, coll in collections.items():
            if st["visible"].get(name, False):
                coll.set_segments(_seg_lines(st["layers"][name]))
                coll.set_visible(True)
            else:
                coll.set_segments([])
                coll.set_visible(False)

        _clear_outline()
        if st["visible"].get("outline", False):
            for ring in st["layers"]["outline"]:
                if len(ring) < 2:
                    continue
                # close ring
                xs = list(ring[:, 0]) + [ring[0, 0]]
                ys = list(ring[:, 1]) + [ring[0, 1]]
                (ln,) = ax.plot(
                    xs,
                    ys,
                    "-",
                    color=LAYER_COLORS["outline"],
                    lw=1.5,
                    alpha=0.85,
                    zorder=1,
                )
                outline_lines.append(ln)

        xy = st["layers"]["gt_trajectory"]
        if st["visible"].get("gt_trajectory", False) and len(xy) > 0:
            traj_line.set_data(xy[:, 0], xy[:, 1])
            traj_start.set_data([xy[0, 0]], [xy[0, 1]])
            traj_end.set_data([xy[-1, 0]], [xy[-1, 1]])
            traj_line.set_visible(True)
            traj_start.set_visible(True)
            traj_end.set_visible(True)
        else:
            traj_line.set_data([], [])
            traj_start.set_data([], [])
            traj_end.set_data([], [])

        rec = st["layers"]["_recording"]
        n_traj = len(xy) if xy is not None else 0
        title.set_text(
            f"{st['stem']}  ·  map meters  ·  "
            f"GT rec={rec or '—'} ({n_traj} pts @ 1/{st['traj_downsample']})"
        )
        if refit:
            _fit()
        fig.canvas.draw_idle()

    def on_scene(label: str) -> None:
        st["stem"] = label
        st["layers"] = load_scene_layers(root, label)
        rec = st["layers"]["_recording"]
        if rec:
            st["layers"]["gt_trajectory"] = load_gt_xy(
                root, rec, downsample=st["traj_downsample"]
            )
        print(
            f"scene {label}: rec={rec} "
            f"walls={len(st['layers']['walls'])} "
            f"traj={len(st['layers']['gt_trajectory'])}",
            flush=True,
        )
        redraw(refit=True)

    def on_layer(_label: str) -> None:
        status = checks.get_status()
        for name, on in zip(labels, status):
            st["visible"][name] = bool(on)
        redraw(refit=False)

    map_radio.on_clicked(on_scene)
    checks.on_clicked(on_layer)
    redraw(refit=True)
    plt.show()
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root",
        type=Path,
        default=None,
        help="SlugTrails 1.0 package root (default: data/hf_package/slugtrails-1.0)",
    )
    ap.add_argument("--stem", type=str, default=None, help="Initial scene stem")
    ap.add_argument(
        "--traj-downsample",
        type=int,
        default=30,
        help="Keep every Nth gt_aligned pose (default 30 ≈ 1 Hz from 30 Hz)",
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Headless load check (no GUI)",
    )
    args = ap.parse_args(argv)
    root = resolve_package_root(args.root)
    if args.smoke:
        return smoke(root, args.stem)
    return run_viewer(
        root, stem=args.stem, traj_downsample=max(1, args.traj_downsample)
    )


if __name__ == "__main__":
    raise SystemExit(main())
