"""Audit original result.json files and replay fixed IMP budget prefixes.

No training, no torch/OpenML imports. A missing run is never filled with a
published mean. Analysis is strict: all configured datasets/seeds must exist.
"""
import argparse
import csv
import json
import math
from pathlib import Path

from lth.budget_protocol import (BASE_COMMIT, DEFAULT_PREFIXES, claim_output,
                                 imp_cost, load_yaml, select_prefix, sha256, write_json)


def csv_write(path, rows):
    if not rows:
        Path(path).write_text('')
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def validate_run(run, dataset, seed, cfg):
    """Reject inconsistent budgets/metrics/counts before reading test outcomes."""
    if run.get('dataset') != dataset or run.get('seeds', {}).get('model_seed') != seed:
        raise ValueError('Dataset/seed differs from requested run')
    if run.get('task_id') != cfg['datasets'][dataset]:
        raise ValueError('Task id differs from configured task')
    if run.get('exp_name') != cfg['exp_name']:
        raise ValueError('Experiment differs from config')
    if run.get('noise', {}).get('kind') != 'none':
        raise ValueError('Expected a clean-data run')
    if run.get('seeds', {}).get('split_seed') != cfg['split_seed']:
        raise ValueError('Data split mismatch')
    pruning = run.get('config', {}).get('pruning', {})
    for key in ('mode', 'rate', 'out_factor', 'rewind_frac', 'rounds'):
        if pruning.get(key) != cfg['pruning'][key]:
            raise ValueError('Pruning protocol mismatch: ' + key)
    if pruning['mode'] != 'global':
        raise ValueError('This analysis is predeclared for global IMP')
    hp = run['hp']
    total = run['total_steps']
    rewind = run['rewind_step']
    imp_cost(total, rewind, 0)
    if total != hp['epochs'] * math.ceil(run['n_train'] / hp['batch_size']):
        raise ValueError('T differs from epochs x batches')
    if rewind != round(pruning['rewind_frac'] * total):
        raise ValueError('Rewind step mismatch')
    records = run['records']
    keys = [(r['round'], r['method']) for r in records]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate round/method records')
    tickets = sorted([r for r in records if r['method'] == 'ticket'], key=lambda x: x['round'])
    if [r['round'] for r in tickets] != list(range(pruning['rounds'] + 1)):
        raise ValueError('Missing/extra ticket rounds')
    dense = [r for r in records if r['method'] == 'dense' and r['round'] == 0]
    if len(dense) != 1 or dense[0]['metrics'] != tickets[0]['metrics']:
        raise ValueError('Round-zero dense/ticket alias mismatch')
    sizes = tickets[0]['kept']
    if not sizes or any(type(v) is not int or v < 1 for v in sizes.values()):
        raise ValueError('Invalid dense layer sizes')
    previous = sizes
    for rec in tickets:
        for split in ('val', 'test'):
            for metric in ('accuracy', 'log_loss'):
                value = rec['metrics'][split][metric]
                if not math.isfinite(value) or (metric == 'accuracy' and not 0 <= value <= 1):
                    raise ValueError('Invalid ' + split + '/' + metric)
        kept = rec['kept']
        if set(kept) != set(sizes):
            raise ValueError('Layer names differ across rounds')
        if any(type(kept[n]) is not int or not 0 <= kept[n] <= previous[n] for n in sizes):
            raise ValueError('Invalid/nonmonotone kept counts')
        if not math.isclose(sum(kept.values()) / sum(sizes.values()), rec['remaining_frac'], abs_tol=1e-8):
            raise ValueError('Density and kept counts disagree')
        for n in sizes:
            if not math.isclose(1 - kept[n] / sizes[n], rec['layer_sparsity'][n], abs_tol=1e-8):
                raise ValueError('Layer sparsity and kept counts disagree')
        hidden = sum(sizes[n] for n in sizes if n != 'out')
        expected_hidden = round(hidden * (1 - pruning['rate']) ** rec['round'])
        expected_output = round(sizes['out'] * (1 - pruning['rate'] * pruning['out_factor']) ** rec['round'])
        if sum(kept[n] for n in sizes if n != 'out') != expected_hidden or kept['out'] != expected_output:
            raise ValueError('Global pruning target does not match recorded counts')
        previous = kept
    return tickets


def inventory(root, configs):
    rows, runs, hashes = [], [], {}
    seen = set()
    for cfg in configs:
        datasets = cfg.get('dataset_subset') or list(cfg['datasets'])
        for ds in datasets:
            for seed in cfg['model_seeds']:
                key = cfg['exp_name'], ds, seed
                if key in seen:
                    raise ValueError('Duplicate expected run across configs')
                seen.add(key)
                path = Path(root) / cfg['exp_name'] / ds / f'seed{seed}' / 'result.json'
                row = dict(experiment=cfg['exp_name'], dataset=ds, seed=seed,
                           relative_path=str(path.relative_to(root)))
                if not path.is_file():
                    row.update(status='missing', detail='Original result.json is unavailable')
                else:
                    try:
                        run = json.loads(path.read_text())
                        tickets = validate_run(run, ds, seed, cfg)
                        row.update(status='valid', detail='', total_steps=run['total_steps'],
                                   rewind_step=run['rewind_step'], ticket_rounds=len(tickets),
                                   run_runtime_s=run.get('runtime_s'), runtime_scope='IMP+controls; not ticket search')
                        hashes[str(path.relative_to(root))] = sha256(path)
                        runs.append((run, tickets, str(path.relative_to(root))))
                    except (KeyError, TypeError, ValueError) as exc:
                        row.update(status='invalid', detail=str(exc))
                rows.append(row)
    return rows, runs, hashes


def prefix_rows(runs, budgets):
    rows = []
    for run, tickets, source in runs:
        if max(budgets) > max(r['round'] for r in tickets):
            raise ValueError('Requested prefix exceeds available ticket rounds')
        for budget in budgets:
            selected = select_prefix(tickets, budget)
            cost = imp_cost(run['total_steps'], run['rewind_step'], budget)
            rows.append(dict(experiment=run['exp_name'], dataset=run['dataset'],
                             seed=run['seeds']['model_seed'], prefix_round=budget,
                             selected_round=selected['round'], selected_sparse=selected['round'] > 0,
                             remaining_frac=selected['remaining_frac'],
                             val_accuracy=selected['metrics']['val']['accuracy'],
                             test_accuracy=selected['metrics']['test']['accuracy'],
                             test_log_loss=selected['metrics']['test']['log_loss'],
                             **cost, normalized_steps=cost['end_to_end_steps'] / run['total_steps'],
                             source=source, hpo_cost='not_recorded', wallclock_search_s=None))
    return rows


def summarize(rows, output, bootstrap=10000):
    import numpy as np
    import pandas as pd
    frame = pd.DataFrame(rows)
    means = frame.groupby(['experiment', 'dataset', 'prefix_round'], as_index=False).agg(
        accuracy=('test_accuracy', 'mean'), log_loss=('test_log_loss', 'mean'),
        density=('remaining_frac', 'mean'), cost=('normalized_steps', 'mean'),
        fraction_sparse=('selected_sparse', 'mean'))
    csv_write(output / 'per_dataset_prefix.csv', means.to_dict('records'))
    rng = np.random.default_rng(20261102)
    summaries = []
    for exp, group in means.groupby('experiment'):
        full_budget = group.prefix_round.max()
        ref = group[group.prefix_round == full_budget].set_index('dataset')
        draws = rng.integers(0, len(ref), size=(bootstrap, len(ref)))
        for budget, g in group.groupby('prefix_round'):
            g = g.set_index('dataset').loc[ref.index]
            delta = 100 * (g.accuracy - ref.accuracy).to_numpy()
            # Cluster resampling: dataset means, all seeds stay together.
            ci = np.quantile(delta[draws].mean(axis=1), [.025, .975])
            summaries.append(dict(experiment=exp, prefix_round=int(budget), n_datasets=len(delta),
                                  mean_accuracy=float(g.accuracy.mean()), mean_delta_vs_full_pp=float(delta.mean()),
                                  median_delta_vs_full_pp=float(np.median(delta)), ci_low_pp=float(ci[0]), ci_high_pp=float(ci[1]),
                                  mean_normalized_steps=float(g.cost.mean()), mean_density=float(g.density.mean()),
                                  fraction_selected_sparse=float(g.fraction_sparse.mean()),
                                  bootstrap_replicates=bootstrap, uncertainty='dataset-cluster percentile; fixed split'))
    csv_write(output / 'summary_prefix.csv', summaries)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for exp in means.experiment.unique():
        g = [r for r in summaries if r['experiment'] == exp]
        x = [r['mean_normalized_steps'] for r in g]
        y = [r['mean_delta_vs_full_pp'] for r in g]
        axes[0].plot(x, y, 'o-', label=exp)
        axes[0].fill_between(x, [r['ci_low_pp'] for r in g], [r['ci_high_pp'] for r in g], alpha=.15)
        axes[1].plot(x, [r['mean_density'] for r in g], 'o-', label=exp)
    axes[0].set(xlabel='Cumulative optimizer steps / dense steps', ylabel='Accuracy vs full prefix (pp)')
    axes[1].set(xlabel='Cumulative optimizer steps / dense steps', ylabel='Selected weight density')
    for ax in axes:
        ax.legend(fontsize=7)
        ax.grid(alpha=.2)
    fig.tight_layout()
    for ext in ('png', 'pdf'):
        fig.savefig(output / ('cost_prefix.' + ext), dpi=200)
    plt.close(fig)


def published_check(assets, output):
    """Check published summaries, not raw seeds. Does not produce prefix curves."""
    import numpy as np
    import pandas as pd
    rows = []
    sources = {}
    for setting in ('tuned', 'untuned'):
        path = assets / f'per_dataset_{setting}.csv'
        per = pd.read_csv(path)
        sources[str(path.name)] = sha256(path)
        for table in ('table1_ticket_vs_controls.csv', 'table2_ticket_vs_dense.csv'):
            sources[table] = sha256(assets / table)
            for rec in pd.read_csv(assets / table).to_dict('records'):
                if rec['setting'] != setting:
                    continue
                a, b = rec['method'], rec['vs']
                diff = per['acc_' + a] - per['acc_' + b]
                p = per[a + '_vs_' + b + '_p_bh']
                estimate = float(100 * diff.mean())
                better = int(((p < .05) & (diff > 0)).sum())
                worse = int(((p < .05) & (diff < 0)).sum())
                rows.append(dict(setting=setting, method=a, reference=b, n_datasets=len(per),
                                 recomputed_mean_difference_pp=estimate, published_mean_difference_pp=rec['mean_diff_pp'],
                                 delta_from_rounding_pp=estimate-rec['mean_diff_pp'],
                                 mean_matches_within_0_002pp=abs(estimate-rec['mean_diff_pp']) <= .002,
                                 better_bh=better, worse_bh=worse,
                                 counts_match=better == rec['sig_better_bh'] and worse == rec['sig_worse_bh'],
                                 median_difference_pp=float(100 * diff.median()),
                                 min_difference_pp=float(100 * diff.min()), max_difference_pp=float(100 * diff.max()),
                                 evidence='rounded published dataset means; not raw seed verification'))
    csv_write(output / 'published_summary_checks.csv', rows)
    write_json(output / 'published_summary_manifest.json', dict(base_commit=BASE_COMMIT, source_hashes=sources,
               scope='Consistency check only; original seeds and rounds unavailable',
               all_checks_pass=all(r['mean_matches_within_0_002pp'] and r['counts_match'] for r in rows)))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command in ('inventory', 'analyze'):
        p = sub.add_parser(command)
        p.add_argument('--results-root', type=Path, required=True)
        p.add_argument('--configs', nargs='+', type=Path, default=[Path('configs/tabarena35.yaml'), Path('configs/tabarena35_fixed512.yaml')])
        p.add_argument('--out', type=Path, required=True)
        if command == 'analyze':
            p.add_argument('--prefix-rounds', type=int, nargs='+', default=list(DEFAULT_PREFIXES))
            p.add_argument('--bootstrap', type=int, default=10000)
    p = sub.add_parser('published-check')
    p.add_argument('--assets', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'published-check':
        out = claim_output(args.out)
        rows = published_check(args.assets, out)
        print(json.dumps({'comparisons': len(rows), 'output': str(out)}))
        return 0 if all(r['counts_match'] and r['mean_matches_within_0_002pp'] for r in rows) else 2
    configs = [load_yaml(p) for p in args.configs]
    inv, runs, hashes = inventory(args.results_root, configs)
    out = claim_output(args.out)
    csv_write(out / 'inventory.csv', inv)
    counts = {status: sum(r['status'] == status for r in inv) for status in ('valid', 'missing', 'invalid')}
    write_json(out / 'manifest.json', dict(base_commit=BASE_COMMIT, input_hashes=hashes, counts=counts,
               config_hashes={str(p): sha256(p) for p in args.configs},
               hpo='not available from recorded trial best_epoch', cost_unit='optimizer updates, not FLOPs/seconds'))
    print(json.dumps(counts))
    if args.command == 'inventory':
        return 0 if counts['valid'] == len(inv) else 2
    if counts['missing'] or counts['invalid']:
        raise SystemExit('Analysis blocked: missing/invalid original runs. See inventory.csv; no performance curves written.')
    budgets = sorted(set(args.prefix_rounds))
    if not budgets or budgets[0] != 0 or args.bootstrap < 100:
        raise SystemExit('Need nonnegative prefixes including 0 and at least 100 bootstrap draws')
    rows = prefix_rows(runs, budgets)
    csv_write(out / 'cost_prefix.csv', rows)
    summarize(rows, out, args.bootstrap)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
