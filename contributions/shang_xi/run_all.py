"""Reproduce the fixed-prefix and sparse-candidate analyses without training."""
import argparse
import csv
import datetime
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess
import sys
import uuid


def difference(actual, expected, fields):
    with actual.open(newline='') as f:
        a=list(csv.DictReader(f))
    with expected.open(newline='') as f:
        b=list(csv.DictReader(f))
    if len(a)!=len(b): raise ValueError('Reference row count mismatch')
    keys=['experiment']+(['prefix_round'] if 'prefix_round' in a[0] else [])
    if [[r[k] for k in keys] for r in a] != [[r[k] for k in keys] for r in b]:
        raise ValueError('Reference row identities differ')
    return max(abs(float(x[k])-float(y[k])) for x,y in zip(a,b) for k in fields)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inputs-root',type=Path,required=True)
    args=ap.parse_args()
    root=Path(__file__).resolve().parent
    inputs=args.inputs_root.resolve()
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    out=root/'runs'/('review_'+stamp+'_'+uuid.uuid4().hex[:6])
    out.mkdir(parents=True,exist_ok=False)
    metadata={'created_utc':stamp,'python':sys.version,'platform':platform.platform(),
              'dependencies':{n:importlib.metadata.version(n) for n in ('numpy','pandas','matplotlib','PyYAML')},
              'scope':'archived-result reanalysis, no model training'}
    with (out/'execution.log').open('w') as log:
        for cmd in [
            [sys.executable,'-m','analysis.compute_tradeoff','analyze','--results-root',str(inputs),
             '--bootstrap','10000','--out',str(out/'fixed')],
            [sys.executable,'sparse_budget_audit.py','--root',str(inputs),'--out',str(out/'sparse')]]:
            log.write(json.dumps(cmd)+'\n');log.flush()
            subprocess.run(cmd,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    metadata['fixed_max_difference']=difference(out/'fixed/summary_prefix.csv',root/'reference_outputs/fixed/summary_prefix.csv',
        ['mean_accuracy','mean_delta_vs_full_pp','ci_low_pp','ci_high_pp','mean_normalized_steps','mean_density'])
    metadata['sparse_max_difference']=difference(out/'sparse/summary.csv',root/'reference_outputs/sparse/summary.csv',
        ['mean_delta_accuracy_pp','ci_low_pp','ci_high_pp','mean_cost10','mean_cost15','mean_density10','mean_density15'])
    metadata['within_1e_minus_9']=all(metadata[k]<=1e-9 for k in ('fixed_max_difference','sparse_max_difference'))
    (out/'verification.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps({'output':str(out),**metadata},indent=2))
    if not metadata['within_1e_minus_9']:
        raise SystemExit('Differences found. Preserve outputs and inspect; do not alter results to match.')


if __name__=='__main__':main()
