"""Dependency-light protocol, provenance, and cost accounting (no torch import)."""
import hashlib
import json
import math
import os
from pathlib import Path

BASE_COMMIT = '452ecb414b2e82967d642e84d3b317f8ebd50ba4'
DEFAULT_PREFIXES = (0, 1, 3, 5, 10, 15)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def object_hash(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, allow_nan=False).encode()).hexdigest()


def load_yaml(path, _seen=None):
    import yaml
    path = Path(path).resolve()
    seen = set() if _seen is None else set(_seen)
    if path in seen:
        raise ValueError('Cyclic YAML inheritance')
    seen.add(path)
    cfg = yaml.safe_load(path.read_text())
    base = cfg.pop('base', None)
    if not base:
        return cfg
    parent = load_yaml(path.parent / base, seen)
    def merge(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and isinstance(d.get(k), dict):
                merge(d[k], v)
            else:
                d[k] = v
    merge(parent, cfg)
    return parent


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + '\n')
    os.replace(temporary, path)


def imp_cost(total_steps, rewind_step, round_idx):
    """Optimizer updates only. Excludes controls, evaluation, I/O, and HPO."""
    if (type(total_steps) is not int or total_steps < 1 or
            type(rewind_step) is not int or not 0 <= rewind_step < total_steps or
            type(round_idx) is not int or round_idx < 0):
        raise ValueError('Invalid T, k or round')
    if round_idx == 0:
        return {'search_steps': 0, 'final_train_steps': total_steps,
                'end_to_end_steps': total_steps}
    final = total_steps - rewind_step
    search = total_steps + (round_idx - 1) * final
    return {'search_steps': search, 'final_train_steps': final,
            'end_to_end_steps': search + final}


def one_shot_cost(total_steps, rewind_step, fraction):
    imp_cost(total_steps, rewind_step, 0)
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Search fraction must be in (0, 1]')
    search = math.floor(fraction * total_steps)
    if search < rewind_step:
        raise ValueError('Search endpoint precedes rewind checkpoint')
    return {'search_steps': search, 'final_train_steps': total_steps - rewind_step,
            'end_to_end_steps': search + total_steps - rewind_step}


def select_prefix(ticket_records, budget):
    """Selection uses validation accuracy only; exact ties choose earliest round."""
    candidates = [r for r in ticket_records if r['round'] <= budget]
    if not candidates:
        raise ValueError('No ticket candidates within prefix')
    def key(r):
        value = float(r['metrics']['val']['accuracy'])
        if not math.isfinite(value):
            raise ValueError('Non-finite validation accuracy')
        return -value, r['round']
    return min(candidates, key=key)


def claim_output(path):
    """Create a new output directory, refuse existing/symlink destinations."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir(exist_ok=False)
    return path
