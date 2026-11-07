#!/usr/bin/env python3
"""Create cleaned two Q5 figures and auditable dataset summaries.

Run with the existing q5_results (no GPU required). Curves are displayed only where
>=80% of available paired seeds contribute, so a single long timing outlier cannot
create a misleading mean tail. Raw seed curves remain visible. No extrapolation.
"""
import argparse, csv, json, math
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DS_DEFAULT=['splice','SDSS17','in_vehicle_coupon','heloc','churn','MIC']

def load(path):
    pairs=defaultdict(dict)
    for sf in sorted(path.glob('*/seed*/summary.json')):
        hf=sf.parent/'history.csv'
        if not hf.exists(): continue
        summary=json.loads(sf.read_text())
        with hf.open(newline='') as f:
            h=list(csv.DictReader(f))
        if not h: continue
        pairs[summary['dataset']][int(summary['seed'])]={'s':summary,'h':h}
    return pairs

def get_curve(history,model,xfield):
    sub=[r for r in history if r['model']==model]
    x=np.array([float(r[xfield]) for r in sub],float)
    y=np.array([float(r['val_accuracy'])*100 for r in sub],float)
    idx=np.isfinite(x)&np.isfinite(y)
    x,y=x[idx],y[idx]
    if len(x)<2: return None
    inds=np.argsort(x,kind='stable')
    x,y=x[inds],y[inds]
    x,unique_idx=np.unique(x,return_index=True)
    return x,y[unique_idx]

def aggregate(curves,coverage=0.8):
    """Mean of aligned validation curves; never extend a shorter seed's curve."""
    n=len(curves); required=max(1,math.ceil(coverage*n))
    xx=np.concatenate([x for x,_ in curves]); xx=np.unique(xx)
    # Discrete observed epoch positions make the graph faithful to measurements.
    mat=np.full((n,len(xx)),np.nan)
    for j,(x,y) in enumerate(curves):
        within=(xx>=x[0])&(xx<=x[-1])
        mat[j,within]=np.interp(xx[within],x,y)
    ct=np.sum(np.isfinite(mat),axis=0)
    valid=ct>=required
    mean=np.full(len(xx),np.nan)
    sd=np.full(len(xx),np.nan)
    if np.any(valid):
        mean[valid]=np.nanmean(mat[:,valid],axis=0)
        sd[valid]=np.nanstd(mat[:,valid],axis=0)
    return xx,mean,sd,valid,required

def plot_two(out, pairs, datasets, coverage):
    for field,filename,label in [
        ('iterations_including_prefix','figure1_accuracy_vs_iterations_optimized','Cumulative training iterations'),
        ('time_including_prefix_s','figure2_accuracy_vs_wallclock_optimized','Cumulative GPU training time (s)')]:
        ncols=2 if len(datasets)>1 else 1; nrows=math.ceil(len(datasets)/ncols)
        fig,axs=plt.subplots(nrows,ncols,figsize=(13 if ncols==2 else 7,4.2*nrows),squeeze=False)
        for i,ds in enumerate(datasets):
            ax=axs[i//ncols][i%ncols]
            group=pairs[ds]
            for model,name,color in [('dense','Dense MLP','tab:blue'),('ticket','Fixed-mask sparse ticket','tab:orange')]:
                curves=[]
                for sid,run in sorted(group.items()):
                    v=get_curve(run['h'],model,field)
                    if v is None:continue
                    curves.append(v)
                    ax.plot(*v,alpha=0.12,lw=.7,color=color)
                if curves:
                    x,mean,sd,valid,req=aggregate(curves,coverage)
                    ax.plot(x,mean,color=color,lw=2,label=f'{name} (n={len(curves)})')
                    ax.fill_between(x,mean-sd,mean+sd,where=valid,color=color,alpha=.12)
            targets=[run['s']['target_val_accuracy']*100 for run in group.values()]
            if targets:
                ax.axhline(np.mean(targets),color='grey',linestyle=':',lw=1,label='Mean target (dense best −1 pp)')
            ax.set_title(f'{ds}  (n={len(group)})')
            ax.set_xlabel(label)
            ax.set_ylabel('Validation accuracy (%)')
            ax.grid(alpha=.2);ax.legend(loc='best',fontsize=7)
        for i in range(len(datasets),ncols*nrows): axs[i//ncols][i%ncols].axis('off')
        fig.suptitle('Dense vs validation-selected fixed-mask sparse subnetworks\n'
                     f'Mean/SD shown only with ≥{coverage:.0%} of seeds; rewind prefix included, IMP search excluded',fontsize=11)
        fig.tight_layout(rect=[0,0,1,.955]);fig.savefig(out/(filename+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(out/(filename+'.pdf'),bbox_inches='tight');plt.close(fig)
        print('Saved',filename+'.png / .pdf')

def write_pair_csv(out, pairs, datasets):
    """Auditable one-row-per-(dataset, seed) summary, including failures to reach target."""
    rows=[]
    for ds in datasets:
        for seed, run in sorted(pairs[ds].items()):
            s=run['s']; d=s.get('dense_time_to_target'); t=s.get('ticket_time_to_target')
            rows.append({
                'dataset':ds, 'seed':seed, 'ticket_round':s['ticket_round'],
                'rewind_step':s['rewind_step'],
                'checkpoint_source':s.get('checkpoint_source',''),
                'dense_final_test_accuracy':s['dense_final_test_accuracy'],
                'ticket_final_test_accuracy':s['ticket_final_test_accuracy'],
                'test_accuracy_difference_pp':s['test_accuracy_difference_pp'],
                'comparable_final_test_accuracy':s['comparable_final_test_accuracy'],
                'dense_total_train_time_s':s['dense_total_train_time_s'],
                'ticket_plus_prefix_total_train_time_s':s['ticket_plus_prefix_total_train_time_s'],
                'fixed_budget_time_ratio':s['training_time_ratio_with_prefix'],
                'target_val_accuracy':s['target_val_accuracy'],
                'dense_target_reached':d is not None, 'ticket_target_reached':t is not None,
                'dense_target_steps':d['iterations'] if d else '',
                'ticket_target_steps':t['iterations'] if t else '',
                'dense_target_seconds':d['time_seconds'] if d else '',
                'ticket_target_seconds':t['time_seconds'] if t else '',
                'time_to_target_steps_ratio':(t['iterations']/d['iterations']) if d and t and d['iterations'] else '',
                'time_to_target_seconds_ratio':(t['time_seconds']/d['time_seconds']) if d and t and d['time_seconds'] else '',
            })
    if rows:
        dest=out/'summary_all_pairs.csv'
        with dest.open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print('Saved',dest)


def summarize(out,pairs,datasets):
    rows=[]
    for ds in datasets:
        group=pairs[ds]
        if not group:continue
        acc=[];full=[];target=[];steps=[];passed_acc=0;passed_120=0;both=0
        for _,run in sorted(group.items()):
            s=run['s'];da=s['dense_time_to_target'];ta=s['ticket_time_to_target']
            acc.append(float(s['test_accuracy_difference_pp']))
            passed_acc+=int(bool(s['comparable_final_test_accuracy']))
            full.append(float(s['training_time_ratio_with_prefix']))
            if da and ta and da['time_seconds']>0:
                rr=ta['time_seconds']/da['time_seconds']
                target.append(rr)
                steps.append(ta['iterations']/da['iterations'])
                passed_120+=int(rr<=1.2)
                both+=int(bool(s['comparable_final_test_accuracy']) and rr<=1.2)
        rows.append(dict(dataset=ds,n_seeds=len(group),n_acc_within_minus1pp=passed_acc,
                         mean_test_accuracy_delta_pp=np.mean(acc),median_fixed_budget_time_ratio=np.median(full),
                         median_time_to_target_ratio=np.median(target) if target else np.nan,
                         median_steps_to_target_ratio=np.median(steps) if steps else np.nan,
                         n_target_reached=len(target),n_target_time_ratio_le_1_2=passed_120,
                         n_joint_accuracy_and_time_le_1_2=both))
    sf=out/'dataset_summary_optimized.csv'
    with sf.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    for row in rows:
        print(f"{row['dataset']:20s} n={row['n_seeds']}  accuracy Δ={row['mean_test_accuracy_delta_pp']:+.3f}pp  "
              f"full-time ratio={row['median_fixed_budget_time_ratio']:.3f}  "
              f"target-time ratio={row['median_time_to_target_ratio']:.3f}")
    print('Saved',sf)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=Path('q5_results'))
    ap.add_argument('--datasets',nargs='*',default=DS_DEFAULT)
    ap.add_argument('--coverage',type=float,default=0.8)
    args=ap.parse_args()
    if not 0<args.coverage<=1:raise ValueError('coverage must be in (0,1]')
    out=args.out.resolve();pairs=load(out)
    dss=[d for d in args.datasets if d in pairs]
    if not dss:raise SystemExit('No completed summary/history pairs under '+str(out))
    plot_two(out,pairs,dss,args.coverage)
    write_pair_csv(out,pairs,dss)
    summarize(out,pairs,dss)

if __name__=='__main__':main()
