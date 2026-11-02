import sys; sys.path.insert(0, "/home/vpm582/lth_tabular")
import numpy as np, pandas as pd
from analysis.load import load_results
from common import load_config
from data.preprocess import load_dataset
cfg = load_config("/home/vpm582/lth_tabular/configs/tabarena35.yaml")
imp = load_results("/home/vpm582/lth_tabular/results/imp_global"); imp = imp[(imp.noise_kind == "none") & imp.dataset.isin(cfg["datasets"])]
base = load_results("/home/vpm582/lth_tabular/results/baselines"); base = base[(base.noise_kind == "none") & base.dataset.isin(cfg["datasets"])]
rows = []
for ds, g in imp.groupby("dataset"):
    d = load_dataset(cfg["datasets"][ds], 0, 10000)
    maj = np.bincount(d.y_test).max() / len(d.y_test)
    tk = g[g.method == "ticket"]
    v = tk[tk.split == "val"].set_index(["model_seed", "round"]).accuracy
    best_r = v.groupby("model_seed").idxmax()
    te = tk[tk.split == "test"].set_index(["model_seed", "round"])
    T = te.loc[list(best_r)].reset_index().set_index("model_seed")[["accuracy", "log_loss"]]
    dense = {"dense": g[(g.method == "dense")]}
    b = base[base.dataset == ds]
    for m in ("dense_l1", "dense_strong_reg"): dense[m] = b[b.method == m]
    val_acc = {m: x[x.split == "val"].accuracy.mean() for m, x in dense.items()}
    bm = max(val_acc, key=val_acc.get)
    D = dense[bm][dense[bm].split == "test"].set_index("model_seed")[["accuracy", "log_loss"]]
    da = T.accuracy - D.accuracy; dl = D.log_loss - T.log_loss  # positive = ticket better
    rows.append(dict(dataset=ds, n_train=len(d.X_train), n_in=d.n_inputs, n_cls=d.n_classes, maj=maj,
                     best_dense=bm, dense_acc=D.accuracy.mean(), acc_gap=da.mean(), acc_z=da.mean() / (da.std() + 1e-9),
                     acc_wins=int((da > 0).sum()), acc_losses=int((da < 0).sum()), ll_gap=dl.mean(), ll_wins=int((dl > 0).sum()),
                     degenerate=D.accuracy.mean() - maj < 0.005))
t = pd.DataFrame(rows).set_index("dataset").sort_values("acc_z", ascending=False)
t.to_csv("/home/vpm582/lth_tabular/results/analysis/tables/tabarena35_selection_v2.csv")
pd.set_option("display.width", 250)
print(t.assign(acc_gap=100 * t.acc_gap, dense_acc=100 * t.dense_acc, maj=100 * t.maj).round(2).to_string())
