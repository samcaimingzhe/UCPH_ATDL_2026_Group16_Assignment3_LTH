"""Mask bookkeeping, magnitude pruning and random-mask controls.

Masks are dicts {layer_name: bool tensor shaped like that layer's weight}. Only Linear
weights are masked; biases and norm parameters are never pruned.
"""
import numpy as np
import torch


class Masker:
    def __init__(self, model, masks=None):
        self.layers = model.named_linears()
        self.masks = masks or {n: torch.ones_like(l.weight, dtype=torch.bool) for n, l in self.layers.items()}
        self.masks = {n: m.to(self.layers[n].weight.device) for n, m in self.masks.items()}

    @torch.no_grad()
    def apply(self):
        for n, l in self.layers.items():
            l.weight.mul_(self.masks[n])

    def sizes(self):
        return {n: m.numel() for n, m in self.masks.items()}

    def kept(self):
        return {n: int(m.sum()) for n, m in self.masks.items()}

    def layer_sparsity(self):
        return {n: 1 - int(m.sum()) / m.numel() for n, m in self.masks.items()}

    def remaining_frac(self):
        return sum(self.kept().values()) / sum(self.sizes().values())

    def to_numpy(self):
        return {n: m.cpu().numpy() for n, m in self.masks.items()}


def target_kept(sizes, round_idx, rate, out_factor, out_name="out"):
    """Target number of kept weights after `round_idx` rounds. Computed from the round
    index directly (not compounded), so rounding error never exceeds one weight.
    Returns (kept for hidden layers in total, kept for the output layer)."""
    hidden = sum(v for k, v in sizes.items() if k != out_name)
    k_hidden = int(round(hidden * (1 - rate) ** round_idx))
    k_out = int(round(sizes[out_name] * (1 - rate * out_factor) ** round_idx))
    return k_hidden, k_out


def _topk_mask(scores, k):
    flat = scores.reshape(-1)
    m = torch.zeros_like(flat, dtype=torch.bool)
    if k > 0:
        m[torch.topk(flat, k, sorted=False).indices] = True
    return m.view_as(scores)


@torch.no_grad()
def magnitude_prune(model, masks, round_idx, rate=0.2, out_factor=0.5, mode="global", min_keep_frac=0.01):
    """Return new masks for round `round_idx` (>=1) from the trained weights of `model`.
    Already-pruned weights get score -inf so masks are nested across rounds."""
    layers = model.named_linears()
    sizes = {n: m.numel() for n, m in masks.items()}
    scores = {
        n: torch.where(masks[n], l.weight.abs(), torch.tensor(-float("inf"), device=l.weight.device))
        for n, l in layers.items()
    }
    k_hidden, k_out = target_kept(sizes, round_idx, rate, out_factor)
    hidden = [n for n in layers if n != "out"]
    if mode == "global":
        new = {"out": _topk_mask(scores["out"], k_out)}
        flat = torch.cat([scores[n].reshape(-1) for n in hidden])
        keep = _topk_mask(flat, k_hidden)
        offset = 0
        for n in hidden:
            new[n] = keep[offset : offset + sizes[n]].view_as(scores[n])
            offset += sizes[n]
    elif mode == "uniform":
        new = {"out": _topk_mask(scores["out"], k_out)}
        for n in hidden:
            new[n] = _topk_mask(scores[n], int(round(sizes[n] * (1 - rate) ** round_idx)))
    elif mode.startswith("single_"):
        # prune one layer only; its kept count is chosen so the network total matches what
        # `global` keeps at this round. If the layer is too small for that, it is clipped at
        # min_keep_frac of the layer and the run records the (higher) actual total.
        target = single_layer_name(hidden, mode)
        total_target = k_hidden + k_out
        others = sum(v for n, v in sizes.items() if n != target)
        floor = max(1, int(round(min_keep_frac * sizes[target])))
        k = min(int(masks[target].sum()), max(total_target - others, floor))
        new = {n: masks[n].clone() for n in layers}
        new[target] = _topk_mask(scores[target], k)
    else:
        raise ValueError(mode)
    return new


def single_layer_name(hidden, mode):
    """hidden = ["fc0", ..., "fc{d-1}"] (fc0 is the input layer, fc{d-1} produces the last
    hidden representation). single_middle picks the middle hidden-to-hidden layer."""
    if mode == "single_input":
        return hidden[0]
    if mode == "single_last":
        return hidden[-1]
    if mode == "single_middle":
        if len(hidden) < 3:
            raise ValueError("single_middle needs depth >= 3")
        return hidden[len(hidden) // 2]
    raise ValueError(mode)


def random_mask_layerwise(ticket_masks, seed):
    """Same kept count per layer as the ticket, positions uniformly random within each layer."""
    rng = np.random.RandomState(seed)
    out = {}
    for n, m in ticket_masks.items():
        flat = np.zeros(m.numel(), dtype=bool)
        flat[rng.choice(m.numel(), int(m.sum()), replace=False)] = True
        out[n] = torch.from_numpy(flat.reshape(tuple(m.shape)))
    return out


def random_mask_global(ticket_masks, seed):
    """Same total kept count as the ticket, positions uniformly random across all layers."""
    rng = np.random.RandomState(seed)
    sizes = [m.numel() for m in ticket_masks.values()]
    total_kept = int(sum(int(m.sum()) for m in ticket_masks.values()))
    flat = np.zeros(sum(sizes), dtype=bool)
    flat[rng.choice(sum(sizes), total_kept, replace=False)] = True
    out, off = {}, 0
    for (n, m), s in zip(ticket_masks.items(), sizes):
        out[n] = torch.from_numpy(flat[off : off + s].reshape(tuple(m.shape)))
        off += s
    return out
