"""Exploratory validation-only stopping policies replayed from archived IMP rounds.

This is a separate, post-hoc analysis. It is not a measured elapsed-time result,
and it does not compare masks at a common density.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from analysis.compute_tradeoff import inventory
from lth.budget_protocol import imp_cost, load_yaml, select_prefix


def stop_round(tickets, patience, min_round=3):
    """Stop after `patience` consecutive non-improvements in validation accuracy.

    Ties do not improve the best. Test values are never accessed by this rule.
    """
    if patience < 1 or min_round < 0:
        raise ValueError("Invalid stopping parameters")
    best = float(tickets[0]["metrics"]["val"]["accuracy"])
    stale = 0
    for record in tickets[1:]:
        value = float(record["metrics"]["val"]["accuracy"])
        if value > best:
            best, stale = value, 0
        else:
            stale += 1
        if record["round"] >= min_round and stale >= patience:
            return record["round"]
    return tickets[-1]["round"]


def percentile_interval(values, rng, draws):
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.asarray(values)[samples].mean(axis=1)
    return [float(x) for x in np.percentile(means, (2.5, 97.5))]


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", required=True, type=Path)
    parser.add_argument("--configs", nargs="+", type=Path,
                        default=[Path("configs/tabarena35.yaml"),
                                 Path("configs/tabarena35_fixed512.yaml")])
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    if args.out.exists() or args.bootstrap < 100:
        raise SystemExit("Output exists or bootstrap draw count is too small")
    cfgs = [load_yaml(path) for path in args.configs]
    inventory_rows, runs, hashes = inventory(args.results_root, cfgs)
    if len(runs) != 350 or len(inventory_rows) != len(runs):
        raise SystemExit("Expected 350 complete and valid clean IMP runs")
    args.out.mkdir(parents=True, exist_ok=False)
    rows = []
    for run, tickets, source in runs:
        full = select_prefix(tickets, 15)
        full_cost = imp_cost(run["total_steps"], run["rewind_step"], 15)["end_to_end_steps"] / run["total_steps"]
        for name, budget in [("fixed_5", 5)] + [(f"patience_{p}", stop_round(tickets, p)) for p in (2, 3, 4)]:
            selected = select_prefix(tickets, budget)
            cost = imp_cost(run["total_steps"], run["rewind_step"], budget)["end_to_end_steps"] / run["total_steps"]
            rows.append(dict(experiment=run["exp_name"], dataset=run["dataset"],
                             seed=run["seeds"]["model_seed"], policy=name,
                             stop_round=budget, selected_round=selected["round"],
                             selected_density=selected["remaining_frac"],
                             normalized_steps=cost, full_normalized_steps=full_cost,
                             val_accuracy=selected["metrics"]["val"]["accuracy"],
                             test_accuracy=selected["metrics"]["test"]["accuracy"],
                             delta_test_pp=100 * (selected["metrics"]["test"]["accuracy"]
                                                  - full["metrics"]["test"]["accuracy"]),
                             source=source))
    write_csv(args.out / "adaptive_seed.csv", rows)
    rng = np.random.default_rng(20261105)
    summaries, datasets = [], []
    for experiment in ("imp_global", "imp_fixed_w512"):
        for policy in ("fixed_5", "patience_2", "patience_3", "patience_4"):
            subset = [r for r in rows if r["experiment"] == experiment and r["policy"] == policy]
            names = sorted({r["dataset"] for r in subset})
            assert len(names) == 35 and all(sum(r["dataset"] == d for r in subset) == 5 for d in names)
            grouped = []
            for dataset in names:
                five = [r for r in subset if r["dataset"] == dataset]
                row = dict(experiment=experiment, policy=policy, dataset=dataset,
                           mean_delta_test_pp=float(np.mean([r["delta_test_pp"] for r in five])),
                           mean_cost=float(np.mean([r["normalized_steps"] for r in five])),
                           mean_full_cost=float(np.mean([r["full_normalized_steps"] for r in five])),
                           mean_stop_round=float(np.mean([r["stop_round"] for r in five])),
                           mean_density=float(np.mean([r["selected_density"] for r in five])))
                datasets.append(row)
                grouped.append(row)
            differences = [d["mean_delta_test_pp"] for d in grouped]
            low, high = percentile_interval(differences, rng, args.bootstrap)
            summaries.append(dict(experiment=experiment, policy=policy, n_datasets=35,
                                  mean_cost=float(np.mean([r["normalized_steps"] for r in subset])),
                                  mean_full_cost=float(np.mean([r["full_normalized_steps"] for r in subset])),
                                  mean_cost_reduction_pct=float(100 * (1 - np.mean([r["normalized_steps"] for r in subset]) /
                                                                   np.mean([r["full_normalized_steps"] for r in subset]))),
                                  mean_delta_test_pp=float(np.mean(differences)),
                                  ci_low_pp=low, ci_high_pp=high,
                                  mean_density=float(np.mean([r["selected_density"] for r in subset])),
                                  fraction_dense=float(np.mean([r["selected_round"] == 0 for r in subset])),
                                  fraction_stop_before_full=float(np.mean([r["stop_round"] < 15 for r in subset])),
                                  mean_stop_round=float(np.mean([r["stop_round"] for r in subset]))))
    write_csv(args.out / "adaptive_dataset.csv", datasets)
    write_csv(args.out / "adaptive_summary.csv", summaries)
    comparisons = []
    for experiment in ("imp_global", "imp_fixed_w512"):
        base = {r["dataset"]: r for r in datasets
                if r["experiment"] == experiment and r["policy"] == "fixed_5"}
        for policy in ("patience_2", "patience_3", "patience_4"):
            other = {r["dataset"]: r for r in datasets
                     if r["experiment"] == experiment and r["policy"] == policy}
            assert base.keys() == other.keys()
            for measure in ("mean_delta_test_pp", "mean_cost"):
                paired = [other[d][measure] - base[d][measure] for d in sorted(base)]
                low, high = percentile_interval(paired, rng, args.bootstrap)
                comparisons.append(dict(experiment=experiment, policy=policy,
                                        comparator="fixed_5", measure=measure,
                                        mean_difference=float(np.mean(paired)),
                                        ci_low=low, ci_high=high))
    write_csv(args.out / "paired_vs_fixed5.csv", comparisons)
    (args.out / "manifest.json").write_text(json.dumps(dict(input_sha256=hashes,
          policies={"primary_exploratory": "patience_3", "sensitivity": ["patience_2", "patience_4"],
                    "minimum_stop_round": 3, "comparison": "full prefix 15; fixed_5 comparator"},
          selection="highest validation accuracy up to executed stop; earlier round wins ties",
          uncertainty="pointwise dataset-cluster percentile bootstrap, 10000 resamples, seed 20261105",
          scope="retrospective, not pre-registered; optimizer updates exclude controls/HPO/I-O/evaluation"),
          indent=2, sort_keys=True) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
