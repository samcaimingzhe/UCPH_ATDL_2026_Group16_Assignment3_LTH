"""Enumerate jobs for a config and run them in parallel worker processes, one GPU per
worker (round-robin). Finished runs are skipped.

stages:
  tune       dense Optuna search per dataset
  imp        IMP on clean data, every (dataset, seed)
  noise      IMP on every non-zero noise level, every (dataset, seed); needs `imp` done first
             (clean_ticket control). label_flip only for classification datasets.
  baselines  dense_l1 / dense_strong_reg on clean data and every noise level
"""
import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from lth.common import ROOT, load_config
from lth.run_imp import noise_tag, run_dir
from lth.tune import tuned_path


def noise_settings(cfg):
    for kind, levels in cfg["noise"]["levels"].items():
        for i, lv in enumerate(levels):
            if lv != 0:
                yield kind, i, lv


def enumerate_jobs(cfg, stage):
    jobs = []
    for ds in cfg.get("dataset_subset") or cfg["datasets"]:  # subset: run only these, keep the full table
        if stage == "tune":
            if not tuned_path(cfg, ds).exists():
                jobs.append(["lth.tune", "--dataset", ds])
        elif stage == "imp":
            for s in cfg["model_seeds"]:
                if not (run_dir(cfg, ds, s) / "result.json").exists():
                    jobs.append(["lth.run_imp", "--dataset", ds, "--seed", str(s)])
        elif stage == "noise":
            for kind, i, lv in noise_settings(cfg):
                for s in cfg["model_seeds"]:
                    if not (run_dir(cfg, ds, s, noise_tag(kind, lv)) / "result.json").exists():
                        jobs.append(["lth.run_imp", "--dataset", ds, "--seed", str(s), "--noise", kind, "--level-idx", str(i)])
        elif stage == "baselines":
            exp = cfg["baselines"]["exp_name"]
            for kind, i, lv in [(None, 0, 0), *noise_settings(cfg)]:
                tag = noise_tag(kind, lv)
                if not all((run_dir(cfg, ds, s, tag, exp_name=exp) / "result.json").exists() for s in cfg["model_seeds"]):
                    jobs.append(["lth.run_baselines", "--dataset", ds]
                                + (["--noise", kind, "--level-idx", str(i)] if kind else []))
    return jobs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/default.yaml"))
    ap.add_argument("--stage", choices=["tune", "imp", "noise", "baselines"], required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--gpus", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = ap.parse_args()
    cfg = load_config(args.config)
    gpus = args.gpus.split(",")
    jobs = enumerate_jobs(cfg, args.stage)
    print(f"{len(jobs)} jobs pending", flush=True)
    log_dir = ROOT / cfg["results_dir"] / "logs" / cfg["exp_name"]
    log_dir.mkdir(parents=True, exist_ok=True)

    def run(i_job):
        i, job = i_job
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpus[i % len(gpus)], OMP_NUM_THREADS="4")
        name = "_".join(job[1:][1::2]) + f"_{args.stage}"
        with open(log_dir / f"{name}.log", "w") as log:
            rc = subprocess.call([sys.executable, "-m", *job, "--config", args.config], cwd=ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT)
        print(f"{'OK ' if rc == 0 else 'FAIL'} {name}", flush=True)
        return rc

    with ThreadPoolExecutor(args.workers) as ex:
        rcs = list(ex.map(run, enumerate(jobs)))
    sys.exit(int(any(rcs)))


if __name__ == "__main__":
    main()
