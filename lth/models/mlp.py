import torch
from torch import nn


class MLP(nn.Module):
    """Stack of Linear -> [LayerNorm] -> ReLU -> Dropout, then a Linear output head.

    `self.linears` holds every Linear in forward order; the last one is the output layer.
    Layer names used by the pruning code are "fc0".."fc{depth-1}" and "out".
    """

    def __init__(self, d_in, d_out, depth=3, width=256, dropout=0.0, layernorm=False):
        super().__init__()
        dims = [d_in] + [width] * depth
        self.linears = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.linears.append(nn.Linear(dims[-1], d_out))
        self.norms = nn.ModuleList(nn.LayerNorm(width) if layernorm else nn.Identity() for _ in range(depth))
        self.drop = nn.Dropout(dropout)

    def layer_names(self):
        return [f"fc{i}" for i in range(len(self.linears) - 1)] + ["out"]

    def named_linears(self):
        return dict(zip(self.layer_names(), self.linears))

    def forward(self, x):
        for lin, norm in zip(self.linears[:-1], self.norms):
            x = self.drop(torch.relu(norm(lin(x))))
        return self.linears[-1](x)


def build_mlp(data, hp) -> MLP:
    return MLP(
        data.n_inputs,
        data.n_outputs,
        depth=hp["depth"],
        width=hp["width"],
        dropout=hp.get("dropout", 0.0),
        layernorm=hp.get("layernorm", False),
    )
