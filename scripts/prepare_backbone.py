"""Build a SnapJudge backbone from the original Qwen3.5-4B (MLX, 4-bit) weights instead of downloading the release.

    python scripts/prepare_backbone.py                          # text-only -> models/snapjudge-text
    python scripts/prepare_backbone.py --variant vision         # keeps the vision tower -> models/snapjudge-vision
    python scripts/prepare_backbone.py --source DIR             # trim an existing local copy
    python scripts/prepare_backbone.py --size 2b                # 2B model -> models/snapjudge-2b
"""
import argparse, glob, json, re, shutil
from pathlib import Path

import mlx.core as mx
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parent.parent


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--size", choices=["4b", "2b"], default="4b")
    size = pre.parse_known_args()[0].size
    cfg = json.loads((ROOT / ("weights" if size == "4b" else "weights-2b") / "config.json").read_text())["backbone"]
    ap = argparse.ArgumentParser(parents=[pre])
    ap.add_argument("--repo", default=cfg["repo"])
    ap.add_argument("--source", help="local directory with the full backbone (skips the download)")
    ap.add_argument("--variant", choices=["text", "vision"], default="text")
    ap.add_argument("--out", help="output directory (default: models/snapjudge-<variant>)")
    ap.add_argument("--layers", type=int, default=cfg["layers"])
    args = ap.parse_args()

    src = Path(args.source) if args.source else Path(snapshot_download(args.repo))
    default_name = f"snapjudge-{args.variant}" if size == "4b" else "snapjudge-2b"
    out = Path(args.out or ROOT / "models" / default_name)
    out.mkdir(parents=True, exist_ok=True)

    weights = {}
    for f in sorted(glob.glob(str(src / "model*.safetensors"))):
        weights.update(mx.load(f))

    def keep(name):
        m = re.search(r"(?:language_model\.)?model\.layers\.(\d+)\.", name)
        if "vision" in name:
            return args.variant == "vision"
        return not m or int(m.group(1)) < args.layers

    kept = {k: v for k, v in weights.items() if keep(k)}
    mx.save_safetensors(str(out / "model.safetensors"), kept, metadata={"format": "mlx"})

    for f in src.iterdir():
        if f.is_file() and not f.name.startswith("model") and (args.variant == "vision" or "processor" not in f.name):
            shutil.copy(f, out / f.name)
    # Written after the copy above so the original config.json cannot overwrite the trimmed layer count.
    config = json.loads((src / "config.json").read_text())
    text = config.get("text_config", config)
    text["num_hidden_layers"] = args.layers
    if text.get("layer_types"):
        text["layer_types"] = text["layer_types"][:args.layers]
    (out / "config.json").write_text(json.dumps(config, indent=2))

    vision = sum(1 for k in kept if "vision" in k)
    size = (out / "model.safetensors").stat().st_size / 1e9
    print(f"wrote {out}: {len(kept)} tensors ({vision} vision), {size:.2f} GB, {args.layers} language layers")


if __name__ == "__main__":
    main()
