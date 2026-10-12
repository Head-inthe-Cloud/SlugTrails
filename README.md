# <img src="./slugicon_nobg.png" alt="SlugTrails icon" width="80" />    SlugTrails

## SlugTrails: An Egocentric Benchmark for Floor Plan Localization in Large Buildings

### [ArXiv preprint](https://arxiv.org/abs/2609.19876) · [Hugging Face dataset](https://huggingface.co/datasets/Head-inthe-Cloud/SlugTrails)

SlugTrails is a benchmark and dataset for **floor-plan-based indoor visual localization** in large-scale public buildings. It provides floor-plan-aligned egocentric recordings, standardized evaluation protocols, and tools for exploring the public **SlugTrails 1.0** package.

Existing floor-plan localization benchmarks predominantly focus on residential-scale environments. SlugTrails targets larger indoor spaces—especially **circulation** (corridors, lobbies, connectors)—where repetition and geometric ambiguity are hard.

---

## Dataset (SlugTrails 1.0)

**https://huggingface.co/datasets/Head-inthe-Cloud/SlugTrails**

About 10 GB. Anonymized RGB, Project Aria MPS trajectories, semidense point clouds, floor-plan maps, factor-graph ground truth in the map frame, and official evaluation / finetune manifests. File layout and checksums are on the dataset card (`README.md` on the Hub).

Install the Hub CLI (`pip install -U huggingface_hub`), then download the whole package:

```bash
hf download Head-inthe-Cloud/SlugTrails \
  --repo-type dataset \
  --local-dir ./slugtrails-1.0
```

Older installs still accept `huggingface-cli download` with the same flags.

### Where to put it

`--root` is the directory that **directly** contains `maps/` and `recordings/`. The command above creates that directory as `./slugtrails-1.0` in the current working directory.

```
slugtrails-1.0/
  CHECKSUMS.sha256
  RELEASE.json
  README.md
  manifests/
  maps/<scene>/          # BE_0, BE_1, BE_2, E2_2, E2_3, PS_2
  recordings/<id>/       # BE_0_0, BE_1_0, BE_2_0, E2_2_0, E2_3_0, PS_2_0
```

If you omit `--root`, both viewers search the current working directory for:

1. `./slugtrails-1.0`
2. `./data/hf_package/slugtrails-1.0`

Any other location is fine when you pass `--root /path/to/slugtrails-1.0`. `slugtrails-1.0/` is listed in `.gitignore`, so the download can sit beside this clone without being committed.

Check the download from inside the package root:

```bash
cd slugtrails-1.0
sha256sum -c CHECKSUMS.sha256
# macOS, if sha256sum is not installed:
shasum -a 256 -c CHECKSUMS.sha256
```

---

## Viewers (this repo)

Python ≥ 3.10. From a clone of this repository:

```bash
pip install -e .
# Rerun viewer (point cloud + video). The map viewer uses only the base install.
pip install -e ".[rerun]"
```

Console scripts after install: `slugtrails-view-map`, `slugtrails-view-rerun`. The `python -m` forms below work the same.

### 2D floor plan

```bash
python -m slugtrails.view_map --root ./slugtrails-1.0
```

Opens a matplotlib window: scene radio (`BE_0`, `BE_1`, `BE_2`, `E2_2`, `E2_3`, `PS_2`), per-layer CAD toggles (walls, doors, windows, and the rest of the map CSVs), and the `gt_aligned` trajectory for that scene’s recording. Coordinates are meters in the floor-plan frame.

```bash
python -m slugtrails.view_map --root ./slugtrails-1.0 --stem BE_2
python -m slugtrails.view_map --root ./slugtrails-1.0 --smoke   # load check, no window
```

### Rerun (point cloud + blurred RGB + map)

Needs the `[rerun]` extra (`rerun-sdk`, `projectaria-tools`).

```bash
python -m slugtrails.view_rerun --root ./slugtrails-1.0 --recording BE_0_0
python -m slugtrails.view_rerun --root ./slugtrails-1.0 --list
```

Recording ids: `BE_0_0`, `BE_1_0`, `BE_2_0`, `E2_2_0`, `E2_3_0`, `PS_2_0`.

The window is three panes: MPS 3D (trajectory + semidense cloud), floor plan, and the anonymized RGB video. Semidense points come from `recordings/<id>/mps/slam/semidense_points.csv.gz` inside the package. RGB is `recordings/<id>/rgb/camera-rgb_undistorted.mp4`.

Useful flags:

```bash
# shorter first look
python -m slugtrails.view_rerun --root ./slugtrails-1.0 --recording BE_0_0 \
  --max-frames 300 --stride 5

# browser instead of the desktop viewer
python -m slugtrails.view_rerun --root ./slugtrails-1.0 --recording BE_0_0 --web

# write a .rrd instead of opening a window
python -m slugtrails.view_rerun --root ./slugtrails-1.0 --recording BE_0_0 \
  --rrd BE_0_0.rrd --max-frames 300
```

A local Aria `.vrs` is optional. Pass `--vrs /path/to/BE_0_0.vrs` or set `SLUGTRAILS_ARIA_ROOT` to a directory that contains `BE/`, `E2/`, or `PS/` with the matching `.vrs`. That file supplies camera calibration for the 3D frustum. SlugTrails 1.0 already includes the anonymized MP4; the video pane uses that file.

---

## Benchmark

SlugTrails evaluates floor-plan localization methods across single-frame, multi-view, and sequential settings, including transfer from residential-scale data and fine-tuning on large-building sessions.

Full evaluation harness and baseline adapters will continue to land in this repository; protocol details and frozen splits ship with the HF package (`manifests/`).

---

## Paper

Cheng & Manduchi — *SlugTrails: An Egocentric Benchmark for Floor Plan Localization in Large Buildings* (2026). [arXiv:2609.19876](https://arxiv.org/abs/2609.19876)

---

## License

SlugTrails dataset and the code in this repository are released under
**[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/)** (see `LICENSE`).

---

## Citation

```bibtex
@inproceedings{Cheng2026SlugTrailsAE,
  title={SlugTrails: An Egocentric Benchmark for Floor Plan Localization in Large Buildings},
  author={Yunqian Cheng and Roberto Manduchi},
  year={2026}
}
```

Update the entry upon publication if the venue bibtex changes.

---

## Contact

Questions about the dataset or viewers: open a GitHub issue in this repository, or email ychen827@ucsc.edu.
