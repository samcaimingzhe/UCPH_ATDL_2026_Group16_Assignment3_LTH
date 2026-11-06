"""Hash-verified retrospective sparse-only IMP budget sensitivity analysis."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def select(records, budget, ceiling):
    feasible = [r for r in records if 0 < r['round'] <= budget
                and r['remaining_frac'] <= ceiling]
    return min(feasible, key=lambda r: (-r['metrics']['val']['accuracy'], r['round'])) if feasible else None


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, default=Path(__file__).with_name('source_manifest.json'))
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    if args.out.exists():
        raise SystemExit('Output exists; choose a fresh directory')
    source = json.loads(args.manifest.read_text())
    hashes = source['input_hashes']
    if len(hashes) != 350:
        raise ValueError('Expected exactly 350 archived runs')
    rows = []
    feasible_counts = {5: 0, 8: 0, 10: 0, 15: 0}
    for relative, expected in sorted(hashes.items()):
        path = args.root / relative
        payload = path.read_bytes()
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f'Hash mismatch: {relative}')
        run = json.loads(payload)
        exp, dataset, seed, _ = Path(relative).parts
        assert run['exp_name'] == exp and run['dataset'] == dataset
        assert run['seeds']['model_seed'] == int(seed.removeprefix('seed'))
        tickets = sorted([r for r in run['records'] if r['method'] == 'ticket'], key=lambda r: r['round'])
        assert [r['round'] for r in tickets] == list(range(16))
        T, k = run['total_steps'], run['rewind_step']
        assert 0 <= k < T
        for budget in feasible_counts:
            feasible_counts[budget] += select(tickets, budget, .2) is not None
        for budget in (10, 15):
            chosen = select(tickets, budget, .2)
            if chosen is None:
                raise ValueError('No sparse candidate; do not silently discard this run')
            assert 0 < chosen['round'] <= budget and chosen['remaining_frac'] <= .2
            rows.append(dict(experiment=exp,dataset=dataset,seed=run['seeds']['model_seed'],
                             budget=budget,selected_round=chosen['round'],density=chosen['remaining_frac'],
                             accuracy=chosen['metrics']['test']['accuracy'],log_loss=chosen['metrics']['test']['log_loss'],
                             dense_accuracy=tickets[0]['metrics']['test']['accuracy'],
                             cost=(T+budget*(T-k))/T))
    args.out.mkdir(parents=True)
    write_csv(args.out/'seed_results.csv', rows)
    grouped=[]
    for exp in ('imp_global','imp_fixed_w512'):
        datasets=sorted({r['dataset'] for r in rows if r['experiment']==exp})
        assert len(datasets)==35
        for ds in datasets:
            both=[]
            for b in (10,15):
                five=[r for r in rows if (r['experiment'],r['dataset'],r['budget'])==(exp,ds,b)]
                assert len(five)==5 and len({r['seed'] for r in five})==5
                both.append({key:float(np.mean([r[key] for r in five]))
                             for key in ('accuracy','log_loss','cost','density','dense_accuracy')})
            a,b=both
            grouped.append(dict(experiment=exp,dataset=ds,delta_accuracy_pp=100*(a['accuracy']-b['accuracy']),
                                delta_log_loss=a['log_loss']-b['log_loss'],cost10=a['cost'],cost15=b['cost'],
                                density10=a['density'],density15=b['density'],
                                sparse10_vs_dense_pp=100*(a['accuracy']-a['dense_accuracy']),
                                sparse15_vs_dense_pp=100*(b['accuracy']-b['dense_accuracy'])))
    write_csv(args.out/'dataset_results.csv',grouped)
    summary=[]
    for exp in ('imp_global','imp_fixed_w512'):
        group=[r for r in grouped if r['experiment']==exp]
        rng=np.random.default_rng(20261105)
        idx=rng.integers(0,35,(10000,35))
        d=np.array([r['delta_accuracy_pp'] for r in group])
        low,high=np.percentile(d[idx].mean(axis=1),[2.5,97.5])
        result=dict(experiment=exp,n_datasets=35,mean_delta_accuracy_pp=float(d.mean()),
                    median_delta_accuracy_pp=float(np.median(d)),ci_low_pp=float(low),ci_high_pp=float(high),
                    datasets_positive=int(sum(d>1e-10)),datasets_tied=int(sum(abs(d)<=1e-10)),datasets_negative=int(sum(d< -1e-10)))
        for key in ('cost10','cost15','density10','density15','sparse10_vs_dense_pp','sparse15_vs_dense_pp','delta_log_loss'):
            result['mean_'+key]=float(np.mean([r[key] for r in group]))
        result['accounted_update_reduction_pct']=100*(1-result['mean_cost10']/result['mean_cost15'])
        summary.append(result)
    write_csv(args.out/'summary.csv',summary)
    (args.out/'checks.json').write_text(json.dumps(dict(verified_hashes=350,
        density_ceiling=.2,feasible_run_counts=feasible_counts,bootstrap=10000,seed=20261105,
        uncertainty='pointwise dataset-cluster percentile; fixed split',
        scope='post-hoc sensitivity; ceiling not identical densities; no wall-clock measurement'),indent=2))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    main()
