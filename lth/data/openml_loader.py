"""Download Grinsztajn benchmark tasks from OpenML with a local parquet cache."""
import json
from pathlib import Path

import pandas as pd

from lth.common import ROOT

TARGET_COL = "__target__"


def fetch_task(task_id: int, cache_dir=ROOT / "cache"):
    """Return (X dataframe, y series, meta dict). meta has name/task_type/categorical."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pq = cache_dir / f"task_{task_id}.parquet"
    meta_path = cache_dir / f"task_{task_id}.json"
    if pq.exists() and meta_path.exists():
        df = pd.read_parquet(pq)
        meta = json.loads(meta_path.read_text())
    else:
        import openml

        task = openml.tasks.get_task(task_id, download_data=True, download_qualities=False)
        ds = task.get_dataset()
        X, y, cat_ind, names = ds.get_data(target=task.target_name, dataset_format="dataframe")
        task_type = (
            "classification"
            if task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION
            else "regression"
        )
        meta = {
            "task_id": task_id,
            "dataset_id": ds.dataset_id,
            "name": ds.name,
            "task_type": task_type,
            "categorical": [n for n, c in zip(names, cat_ind) if c],
        }
        df = X.copy()
        df[TARGET_COL] = y.astype(str) if task_type == "classification" else y.astype(float)
        df.to_parquet(pq)
        meta_path.write_text(json.dumps(meta, indent=1))
    y = df.pop(TARGET_COL)
    return df, y, meta
