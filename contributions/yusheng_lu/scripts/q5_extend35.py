#!/usr/bin/env python3
"""Extend an EXISTING six-dataset Q5 Fresh IMP experiment to all 35 TabArena datasets.

Reuses completed IMP and timing pairs, never copies old IMP masks/rewind checkpoints.
Downloads ONLY missing tuned hyperparameters from the paper's Hugging Face archive.
Uses original repository's IMP implementation with a separate, 35-dataset launch config.

Run from /work/Q5_Timing_Experiment, in the CUDA-ready .venv:
 python -u q5_extend35.py --stage all 2>&1 | tee -a q5_extend35.log
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

HF_REPO = "samcaimingzhe/UCPH_ATDL_2026_Group16_Assignment3_LTH"
SEEDS = (42, 43, 44, 45, 46)


def run_command(argv, *, cwd, env):
    print("\n$ " + " ".join(str(a) for a in argv), flush=True)
    subprocess.run([str(a) for a in argv], cwd=str(cwd), env=env, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def get_datasets(project):
    import yaml
    f = project / "configs" / "tabarena35.yaml"
    names = list(yaml.safe_load(f.read_text(encoding="utf-8"))["datasets"])
    if len(names) != 35 or len(set(names)) != 35:
        raise RuntimeError(f"Expected exactly 35 unique datasets in {f}, found {len(names)}")
    return names


def create_safe_config(project, datasets):
    """Never modify the six-dataset config or its results; subset only controls launch enumeration."""
    import yaml
    origin = project / "configs" / "q5_fresh_imp.yaml"
    if not origin.exists():
        raise FileNotFoundError(
            f"Missing {origin}; first verify the original six-dataset Fresh IMP experiment."
        )
    old = yaml.safe_load(origin.read_text(encoding="utf-8"))
    mandatory = {
        "base": "tabarena35.yaml", "exp_name": "imp_global", "results_dir": "q5_fresh_archive",
        "controls": [], "save_full_rewind": True,
    }
    for k, expected in mandatory.items():
        if old.get(k) != expected:
            raise RuntimeError(f"Existing fresh IMP protocol differs: {k}: {old.get(k)!r} != {expected!r}")
    enlarged = copy.deepcopy(old)
    enlarged["dataset_subset"] = list(datasets)
    new_path = project / "configs" / "q5_fresh_imp_35.yaml"
    if new_path.exists():
        actual = yaml.safe_load(new_path.read_text(encoding="utf-8"))
        if actual != enlarged:
            raise RuntimeError(f"Existing 35-dataset config differs: {new_path}. Do not mix protocols.")
    else:
        new_path.write_text(yaml.safe_dump(enlarged, sort_keys=False), encoding="utf-8")
    return new_path


def fill_tuned_hps(source_tuned, project, datasets):
    """Download only absent HP JSONs; do not download unrelated old masks or checkpoints."""
    source_tuned.mkdir(parents=True, exist_ok=True)
    fresh_tuned = project / "q5_fresh_archive" / "tuned"
    fresh_tuned.mkdir(parents=True, exist_ok=True)
    missing = [ds for ds in datasets if not (source_tuned / f"{ds}.json").exists()]
    if missing:
        print(f"Fetching {len(missing)} missing tuned HP files (not masks) from Hugging Face", flush=True)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as e:
            raise RuntimeError("Install huggingface_hub: python -m pip install huggingface_hub") from e
        for i, ds in enumerate(missing, 1):
            try:
                hf_hub_download(
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    filename=f"tuned/{ds}.json",
                    local_dir=str(source_tuned.parent),
                )
            except Exception as e:
                raise RuntimeError(f"Cannot retrieve tuned HP for {ds}: {e}. Stop; do not retune silently.") from e
            print(f"  [{i}/{len(missing)}] downloaded tuned/{ds}.json", flush=True)
    checksums = {}
    for ds in datasets:
        src, dst = source_tuned / f"{ds}.json", fresh_tuned / f"{ds}.json"
        if not src.exists():
            raise FileNotFoundError(f"Missing tuned HP after download: {src}")
        doc = json.loads(src.read_text(encoding="utf-8"))
        hp = doc.get("hp", {})
        required = ("depth", "width", "lr", "weight_decay", "dropout", "batch_size", "epochs")
        if not all(k in hp for k in required):
            raise RuntimeError(f"Tuned HP is incomplete for {ds}: {src}")
        if dst.exists():
            if sha256(dst) != sha256(src):
                raise RuntimeError(f"Tuned HP conflict for {ds}: {src} vs {dst}. Investigate before continuing.")
        else:
            shutil.copy2(src, dst)
        checksums[ds] = sha256(dst)
    print(f"TUNED HP READY: {len(datasets)}/35", flush=True)
    return checksums


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("imp", "benchmark", "plot", "all"), default="all")
    parser.add_argument("--project", type=Path, default=Path("upstream_lth"))
    parser.add_argument("--tuned-source", type=Path, default=Path("archive_results/tuned"))
    parser.add_argument("--out", type=Path, default=Path("q5_fresh_results"))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--warmup-steps", type=int, default=40)
    parser.add_argument("--accuracy-margin-pp", type=float, default=1.0)
    args = parser.parse_args()

    top = Path(__file__).resolve().parent
    project = args.project.resolve()
    src = args.tuned_source.resolve()
    out = args.out.resolve()
    for required in (project / "lth/run_imp.py", top / "timing_fresh.py", top / "plot_efficiency_optimized.py", top / "q5_fresh_imp.py"):
        if not required.exists():
            raise FileNotFoundError(f"Missing required existing experiment file: {required}")

    from q5_fresh_imp import validate_run
    datasets = get_datasets(project)
    archive = project / "q5_fresh_archive"
    config35 = create_safe_config(project, datasets)
    hp_sha = fill_tuned_hps(src, project, datasets)
    out.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "description": "Incremental 35-dataset extension of original Q5 Fresh IMP",
        "datasets": datasets, "seeds": args.seeds, "stage": args.stage,
        "imp_config": str(config35), "archive": str(archive),
        "tuned_hp_sha256": hp_sha,
        "reuse_existing_results": True,
        "training_time_excludes_imp_discovery": True,
    }
    manifest_path = out / "fresh_imp_extension35_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    env.update({"CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "PYTHONUNBUFFERED": "1"})
    if args.stage != "plot":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; start the UCloud GPU job.")
        print(f"GPU={torch.cuda.get_device_name(0)} PyTorch={torch.__version__}", flush=True)

    if args.stage in ("imp", "all"):
        done_count = 0
        total_count = len(datasets) * len(args.seeds)
        for i, ds in enumerate(datasets, 1):
            for seed in args.seeds:
                if validate_run(archive, ds, seed):
                    print(f"SKIP completed IMP: {ds} seed{seed}", flush=True)
                else:
                    print(f"FRESH IMP START: {ds} seed{seed} [{i}/35]", flush=True)
                    run_command(
                        [sys.executable, "-u", "-m", "lth.run_imp", "--config", "configs/q5_fresh_imp_35.yaml", "--dataset", ds, "--seed", str(seed)],
                        cwd=project, env=env,
                    )
                    if not validate_run(archive, ds, seed):
                        raise RuntimeError(f"IMP finished but archive failed validation: {ds} seed{seed}")
                done_count += 1
                print(f"IMP progress: {done_count}/{total_count}", flush=True)

    if args.stage in ("benchmark", "all"):
        missing = [(ds, seed) for ds in datasets for seed in args.seeds if not validate_run(archive, ds, seed)]
        if missing:
            raise RuntimeError(f"Cannot benchmark yet; {len(missing)} IMP pairs missing, first: {missing[:5]}")
        run_command(
            [sys.executable, "-u", str(top / "timing_fresh.py"), "--project", str(project),
             "--archive", str(archive), "--out", str(out), "--datasets", *datasets,
             "--seeds", *map(str, args.seeds), "--accuracy-margin-pp", str(args.accuracy_margin_pp),
             "--warmup-steps", str(args.warmup_steps)],
            cwd=top, env=env,
        )

    if args.stage in ("plot", "all", "benchmark"):
        # Plotting 35 two-column panels creates tall images; PDFs are included for inspection.
        run_command(
            [sys.executable, "-u", str(top / "plot_efficiency_optimized.py"),
             "--out", str(out), "--datasets", *datasets], cwd=top, env=env
        )
    print(f"FINISHED. Results: {out}  Archive: {archive}", flush=True)


if __name__ == "__main__":
    main()
