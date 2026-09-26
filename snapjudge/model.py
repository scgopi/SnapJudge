import json
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache

from .head import KIND, OptionHead

LETTERS = "ABCDEFGHIJKL"
PACKAGE_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = PACKAGE_ROOT / "weights"
MODELS_DIR = PACKAGE_ROOT / "models"


def default_backbone(config):
    names = config["backbone"].get("folders", ["snapjudge-text", "snapjudge-vision"])
    for name in names:
        if (MODELS_DIR / name / "config.json").exists():
            return MODELS_DIR / name
    return MODELS_DIR / names[0]


SYSTEM_PROMPT = "You are a classifier. You output one option letter."
NOUL_QUESTION = "Which statements are true of `state`?"


class SnapJudge:
    """Answers typed questions about a text `state` with calibrated probabilities, fully on-device."""

    def __init__(self, backbone=None, weights=DEFAULT_WEIGHTS, cache_limit_bytes=1 << 30):
        weights = Path(weights)
        self.config = json.loads((weights / "config.json").read_text())
        self.depth = self.config["backbone"]["layers"]
        self.max_options = self.config["max_options"]
        backbone = Path(backbone) if backbone else default_backbone(self.config)
        if not (backbone / "config.json").exists():
            raise FileNotFoundError(f"backbone not found at {backbone}; run scripts/download_model.py first (--variant 2b for the 2B model)")
        # MLX keeps freed buffers per input shape; without a cap the cache grows across varied prompt lengths.
        mx.set_cache_limit(cache_limit_bytes)
        self.model, self.tok = load(str(backbone))
        self.inner = getattr(self.model, "language_model", self.model).model
        self.hf = self.tok._tokenizer
        dim = self.config["backbone"]["hidden_size"]
        self.heads = []
        for path in sorted(weights.glob("head-*.safetensors")):
            head = OptionHead(d=dim)
            head.load_weights(str(path))
            head.eval()
            self.heads.append(head)
        n = np.load(weights / "norm.npz")
        self.norm = {k: mx.array(n[k]) for k in n.files}
        t = self.config["temperature"]
        self.temperature = [t["noul"], t["choice"], t["score"]]

    def _prompt(self, state, question, options):
        lines = [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
        user = f"state: {state}\n\nQuestion: {question}\n" + "\n".join(lines) + \
            "\nAnswer with the letter of the correct option."
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}]
        s = self.tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)
        enc = self.hf(s, return_offsets_mapping=True, add_special_tokens=False)
        starts = [a for a, _ in enc["offset_mapping"]]
        pos, cursor = [], s.index("Question: ")
        for line in lines:
            end = s.index(line, cursor) + len(line)
            cursor = end
            pos.append(max(i for i, a in enumerate(starts) if a < end))
        return enc["input_ids"], pos

    def _new_caches(self):
        return [ArraysCache(2) if getattr(l, "is_linear", False) else KVCache() for l in self.inner.layers[:self.depth]]

    def _layers(self, h, caches):
        # Linear-attention (recurrent) layers take no mask; right padding after real tokens cannot leak backwards.
        kv = next(c for c in caches if isinstance(c, KVCache))
        mask = create_attention_mask(h, kv)
        for layer, c in zip(self.inner.layers[:self.depth], caches):
            h = layer(h, None if getattr(layer, "is_linear", False) else mask, c)
        return h

    def _features(self, state, items):
        """Encode the shared `state` prefix once, then every question's own tokens as one padded batch."""
        prompts = [self._prompt(state, q, o) for q, o in items]
        P = min(len(ids) for ids, _ in prompts) - 1
        for ids, _ in prompts[1:]:
            P = next((i for i, (a, b) in enumerate(zip(prompts[0][0][:P], ids)) if a != b), P)
        P = min([P] + [min(pos) for _, pos in prompts])
        base = self._new_caches()
        if P > 0:
            self._layers(self.inner.embed_tokens(mx.array([prompts[0][0][:P]])), base)
            mx.eval([c.state for c in base])
        B, L = len(prompts), max(len(ids) - P for ids, _ in prompts)
        pad = self.tok.pad_token_id or 0
        batch = mx.array([ids[P:] + [pad] * (L - len(ids) + P) for ids, _ in prompts])
        caches = self._new_caches()
        if P > 0:
            for c, n in zip(base, caches):
                if isinstance(c, KVCache):
                    n.state = (mx.repeat(c.keys[..., :c.offset, :], B, axis=0), mx.repeat(c.values[..., :c.offset, :], B, axis=0))
                else:
                    n[0], n[1] = mx.repeat(c[0], B, axis=0), mx.repeat(c[1], B, axis=0)
        h = self._layers(self.inner.embed_tokens(batch), caches).astype(mx.float32)
        out = []
        for b, (ids, pos) in enumerate(prompts):
            last = (h[b, len(ids) - P - 1] - self.norm["mu_h"]) / self.norm["sd_h"]
            opts = (h[b, mx.array([p - P for p in pos])] - self.norm["mu_o"]) / self.norm["sd_o"]
            out.append((last, opts, len(ids) - P))
        return out, P

    def _probs(self, feats, kind):
        last, opts, _ = feats
        k = mx.array([KIND[kind]])
        z = mx.mean(mx.stack([h(last[None], opts[None], k)[0] for h in self.heads]), 0) / self.temperature[KIND[kind]]
        p = mx.sigmoid(z) if kind == "noul" else mx.softmax(z)
        return np.array(p).tolist()

    def classify(self, request):
        """Answer every question in `request`; see README.md for the request and response format."""
        state = request["state"] if isinstance(request["state"], str) else json.dumps(request["state"])
        plan = []
        for name, q in request["questions"].items():
            t = q["type"]
            if t == "noul":
                plan.append((name, t, NOUL_QUESTION, [q["instructions"]], None))
            elif t == "choice":
                keys = list(q["criteria"])[:self.max_options]
                plan.append((name, t, q["instructions"], [q["criteria"][k] for k in keys], keys))
            elif t == "score":
                plan.append((name, t, q["instructions"], list(q["criteria"])[:self.max_options], None))
            else:
                raise ValueError(f"unknown question type {t!r} for {name!r}")
        feats, shared = self._features(state, [(question, opts) for _, _, question, opts, _ in plan])
        answers = {}
        for (name, t, _, opts, keys), f in zip(plan, feats):
            p = self._probs(f, t)
            if t == "noul":
                answers[name] = {"type": "noul", "noul": round(p[0], 4)}
            elif t == "choice":
                probs = {k: round(v, 4) for k, v in zip(keys, p)}
                best = max(probs, key=probs.get)
                answers[name] = {"type": "choice", "choice": best, "confidence": probs[best], "probabilities": probs}
            else:
                answers[name] = {"type": "score", "score": round(float(np.dot(p, range(len(p)))), 2),
                                 "confidence": round(max(p), 4),
                                 "legend": {str(i): l for i, l in enumerate(opts)},
                                 "probabilities": {str(i): round(v, 4) for i, v in enumerate(p)}}
        tokens = shared + sum(f[2] for f in feats)
        return {"model": f"snapjudge-{self.config['version']}", "answers": answers,
                "usage": {"input_tokens": tokens, "output_tokens": 0}}
