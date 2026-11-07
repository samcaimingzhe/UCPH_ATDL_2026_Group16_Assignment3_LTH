#!/usr/bin/env python3
"""Q5: fresh, trace-consistent IMP discovery -> replay benchmark -> figures.

Uses the original repository's lth.run_imp rather than reimplementing IMP.
Original lth.run_imp performs the dense training, captures the full step-k
checkpoint, then runs 15 nested pruning rounds from that SAME checkpoint.

Output is isolated under <project>/q5_fresh_archive/imp_global/... .
Original/archive results are never used for masks or rewind checkpoints.
Only existing validation-selected tuned HYPERPARAMETERS are reused.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone

DEFAULT_DATASETS = ["splice", "SDSS17", "in_vehicle_coupon", "heloc", "churn", "MIC"]
DEFAULT_SEEDS = [42, 43, 44, 45, 46]


def run(cmd, *, cwd=None, env=None):
    print("\n$ " + " ".join(map(str, cmd)), flush=True)
    subprocess.run([str(s) for s in cmd], cwd=cwd, env=env, check=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def create_config(project, datasets, seeds):
    # Keep source exp_name as 'imp_global' so the existing timing benchmark
    # finds the expected folder. Isolate it using results_dir instead.
    import yaml
    rel_result = "q5_fresh_archive"
    config = {
        "base": "tabarena35.yaml",
        "exp_name": "imp_global",
        "results_dir": rel_result,
        "dataset_subset": datasets,
        "model_seeds": seeds,
        "controls": [],      # the Q5 question does not need reinit/random-mask controls
        "save_full_rewind": True,
    }
    path = project / "configs" / "q5_fresh_imp.yaml"
    if path.exists():
        existing = yaml.safe_load(path.read_text(encoding="utf-8"))
        if existing != config:
            # Guard against accidentally mixing settings in one results archive.
            archive = project / rel_result
            existing_runs = list(archive.glob("imp_global/*/seed*/result.json"))
            if existing_runs:
                raise RuntimeError(
                    f"Config {path} differs from the existing experiment, with "
                    f"{len(existing_runs)} completed runs. Keep identical config "
                    "or create a fresh project directory."
                )
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path, project / rel_result


def ensure_hp(source, result_root, datasets):
    # Tuned HPs are the only reused object: never import the old mask or checkpoint.
    out = result_root / "tuned"
    out.mkdir(parents=True, exist_ok=True)
    for ds in datasets:
        src = source / f"{ds}.json"
        dst = out / src.name
        if not src.exists():
            raise FileNotFoundError(f"Missing tuned hyperparameters: {src}")
        hp = json.loads(src.read_text())["hp"]
        for key in ("depth", "width", "lr", "weight_decay", "dropout", "batch_size", "epochs"):
            if key not in hp:
                raise ValueError(f"Missing hp '{key}' in {src}")
        if dst.exists() and sha256(dst) != sha256(src):
            raise RuntimeError(f"Tuned HPs already differ for {ds}: {dst}. Do not mix experiments.")
        if not dst.exists():
            shutil.copy2(src, dst)
        print(f"HP READY {ds}: {hp['depth']}x{hp['width']} epochs={hp['epochs']}", flush=True)


def validate_run(result_root, dataset, seed):
    path = result_root / "imp_global" / dataset / f"seed{seed}"
    resfile, masksfile, rewindfile = (path / n for n in ("result.json", "masks.npz", "rewind.pt"))
    if not resfile.exists():
        return False
    if not (masksfile.exists() and rewindfile.exists()):
        raise RuntimeError(f"A result.json exists but mask/rewind missing: {path}; investigate before rerunning")
    result = json.loads(resfile.read_text())
    if int(result["seeds"]["model_seed"]) != seed or result["dataset"] != dataset:
        raise RuntimeError(f"Unexpected dataset/seed in {resfile}")
    if result.get("exp_name") != "imp_global":
        raise RuntimeError(f"Unexpected exp_name in {resfile}")
    if result.get("config", {}).get("controls") != []:
        raise RuntimeError(f"IMP result has different controls than the fresh Q5 protocol: {resfile}")
    rounds = {int(x["round"]) for x in result["records"] if x["method"] == "ticket"}
    if rounds != set(range(16)):
        raise RuntimeError(f"Incomplete IMP rounds in {resfile}: {sorted(rounds)}")
    # The full rewind checkpoint must be present, not just weights.
    import torch
    ckpt = torch.load(rewindfile, map_location="cpu", weights_only=False)
    if set(("step", "model", "opt", "sched", "rng")) - set(ckpt):
        raise RuntimeError(f"Rewind checkpoint is incomplete: {rewindfile}")
    if int(ckpt["step"]) != int(result["rewind_step"]):
        raise RuntimeError(f"Rewind step mismatch: {rewindfile}")
    import numpy as np
    with np.load(masksfile) as masks:
        for r in range(1, 16):
            if not any(k.startswith(f"ticket/r{r}/") for k in masks.files):
                raise RuntimeError(f"Missing mask at round {r}: {masksfile}")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--project", type=Path, default=Path("upstream_lth"))
    ap.add_argument("--tuned-source", type=Path, default=Path("archive_results/tuned"))
    ap.add_argument("--out", type=Path, default=Path("q5_fresh_results"))
    ap.add_argument("--stage", choices=("imp", "benchmark", "all"), default="all")
    ap.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS)
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--gpu", default="0", help="CUDA-visible GPU index; default 0")
    ap.add_argument("--accuracy-margin-pp", type=float, default=1.0)
    ap.add_argument("--warmup-steps", type=int, default=40)
    args = ap.parse_args()
    top = Path(__file__).resolve().parent
    project = args.project.resolve()
    source = args.tuned_source.resolve()
    out = args.out.resolve()
    if not (project / "lth" / "run_imp.py").exists():
        raise FileNotFoundError(f"Original repo not found: {project}")
    if not (project / "configs" / "tabarena35.yaml").exists():
        raise FileNotFoundError("Missing base tabarena35.yaml in original repo")
    if not (top / "timing_fresh.py").exists():
        raise FileNotFoundError("Upload timing_fresh.py alongside q5_fresh_imp.py")
    if not (top / "plot_efficiency_optimized.py").exists():
        raise FileNotFoundError("Upload plot_efficiency_optimized.py alongside q5_fresh_imp.py")
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("Duplicate datasets/seeds")
    import yaml
    base = yaml.safe_load((project / "configs" / "tabarena35.yaml").read_text())
    invalid = [ds for ds in args.datasets if ds not in base["datasets"]]
    if invalid:
        raise ValueError(f"Unknown datasets: {invalid}")
    config, archive = create_config(project, args.datasets, args.seeds)
    ensure_hp(source, archive, args.datasets)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
                "CUDA_VISIBLE_DEVICES": args.gpu})
    # CUDA is required for this experiment's declared GPU timing results.
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA GPU found; choose UCloud GPU PyTorch application.")
    print(f"GPU: {torch.cuda.get_device_name(0)}  PyTorch: {torch.__version__}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "study": "Fresh IMP mask and rewind state from SAME dense trajectory",
        "datasets": args.datasets, "seeds": args.seeds,
        "stage": args.stage,
        "source": str(project), "config": str(config), "archive": str(archive),
        "out": str(out), "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__, "hyperparameters_sha256":
            {ds: sha256(archive / "tuned" / f"{ds}.json") for ds in args.datasets},
        "protocol": {"pruning_rounds": 15, "rewind_frac": 0.02,
                     "controls": [], "accuracy_margin_pp": args.accuracy_margin_pp,
                     "training_time_excludes_imp_discovery": True},
    }
    if not (out / "fresh_imp_manifest.json").exists():
        (out / "fresh_imp_manifest.json").write_text(json.dumps(manifest, indent=2))

    if args.stage in ("imp", "all"):
        for ds in args.datasets:
            for seed in args.seeds:
                if validate_run(archive, ds, seed):
                    print(f"SKIP completed IMP: {ds} seed{seed}", flush=True)
                    continue
                print(f"FRESH IMP START {ds} seed{seed} (1 dense + 15 ticket rounds)", flush=True)
                run([sys.executable, "-u", "-m", "lth.run_imp", "--config", "configs/q5_fresh_imp.yaml",
                     "--dataset", ds, "--seed", str(seed)], cwd=project, env=env)
                if not validate_run(archive, ds, seed):
                    raise RuntimeError(f"IMP completed without a valid result: {ds} seed{seed}")
                print(f"FRESH IMP COMPLETE {ds} seed{seed}", flush=True)

    if args.stage in ("benchmark", "all"):
        missing = [f"{ds}/seed{seed}" for ds in args.datasets for seed in args.seeds
                   if not validate_run(archive, ds, seed)]
        if missing:
            raise RuntimeError(f"IMP unfinished for {len(missing)} combinations: {missing[:10]}")
        run([sys.executable, "-u", str(top / "timing_fresh.py"),
             "--project", project, "--archive", archive, "--out", out,
             "--datasets", *args.datasets, "--seeds", *[str(s) for s in args.seeds],
             "--accuracy-margin-pp", str(args.accuracy_margin_pp),
             "--warmup-steps", str(args.warmup_steps)], cwd=top, env=env)
        run([sys.executable, "-u", str(top / "plot_efficiency_optimized.py"),
             "--out", out, "--datasets", *args.datasets], cwd=top, env=env)
    print("Q5 complete. Fresh IMP files:", archive, "\nFigures/CSV:", out, flush=True)


if __name__ == "__main__":
    main()
