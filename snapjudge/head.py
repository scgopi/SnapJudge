import mlx.core as mx
import mlx.nn as nn

KIND = {"noul": 0, "choice": 1, "score": 2}


class OptionHead(nn.Module):
    """Scores each option from the prompt's final hidden state and the hidden state at the end of that option's line."""

    def __init__(self, d=2560, h=512):
        super().__init__()
        self.pl, self.po = nn.Linear(d, h), nn.Linear(d, h)
        self.kind = nn.Embedding(3, h)
        self.drop = nn.Dropout(0.2)
        self.mlp = nn.Sequential(nn.Linear(3 * h, h), nn.GELU(), nn.Linear(h, 1))

    def __call__(self, last, options, kind):
        a = self.pl(self.drop(last)) + self.kind(kind)
        b = self.po(self.drop(options))
        a = mx.broadcast_to(a[:, None, :], b.shape)
        return self.mlp(self.drop(mx.concatenate([a, b, a * b], -1)))[..., 0]
