"""Download a SnapJudge backbone from the GitHub release and assemble it under models/.

    python scripts/download_model.py                   # text-only model  -> models/snapjudge-text
    python scripts/download_model.py --variant vision  # with vision tower -> models/snapjudge-vision
    python scripts/download_model.py --variant both
    python scripts/download_model.py --variant 2b      # 2B model -> models/snapjudge-2b (use with --size 2b)

Uses the GitHub CLI (`gh`) when available, which also works while the repository is private;
otherwise falls back to plain HTTPS downloads, which need the repository to be public.
"""
import argparse, hashlib, os, shutil, subprocess, tarfile, urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASSETS = {
    "text": ["snapjudge-language-{tag}.safetensors", "snapjudge-text-{tag}-files.tar.gz"],
    "vision": ["snapjudge-language-{tag}.safetensors", "snapjudge-vision-tower-{tag}.safetensors",
               "snapjudge-vision-{tag}-files.tar.gz"],
    "2b": ["snapjudge-2b-language-{tag}.safetensors", "snapjudge-2b-{tag}-files.tar.gz"],
}


def fetch(repo, tag, name, cache):
    dest = cache / name
    if dest.exists():
        return dest
    if shutil.which("gh"):
        subprocess.run(["gh", "release", "download", tag, "-R", repo, "-p", name, "-D", str(cache), "--clobber"], check=True)
    else:
        url = f"https://github.com/{repo}/releases/download/{tag}/{name}"
        print(f"downloading {url}")
        urllib.request.urlretrieve(url, dest)
    return dest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def link(src, dst):
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy(src, dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["text", "vision", "both", "2b"], default="text")
    ap.add_argument("--tag", default="v0.2.0")
    ap.add_argument("--repo", default="scgopi/SnapJudge")
    ap.add_argument("--dest", default=str(ROOT / "models"))
    args = ap.parse_args()

    dest = Path(args.dest)
    cache = dest / ".download"
    cache.mkdir(parents=True, exist_ok=True)
    sums = fetch(args.repo, args.tag, "SHA256SUMS", cache)
    expected = dict(reversed(line.split()) for line in sums.read_text().splitlines() if line.strip())

    for variant in (["text", "vision"] if args.variant == "both" else [args.variant]):
        files = {}
        for pattern in ASSETS[variant]:
            name = pattern.format(tag=args.tag)
            path = fetch(args.repo, args.tag, name, cache)
            if sha256(path) != expected.get(name):
                raise SystemExit(f"checksum mismatch for {name}; delete {path} and retry")
            files[name] = path
        out = dest / f"snapjudge-{variant}"
        out.mkdir(parents=True, exist_ok=True)
        with tarfile.open(files[f"snapjudge-{variant}-{args.tag}-files.tar.gz"]) as t:
            t.extractall(out, filter="data")
        if variant == "2b":
            link(files[f"snapjudge-2b-language-{args.tag}.safetensors"], out / "model.safetensors")
            print(f"ready: {out}")
            continue
        language = files[f"snapjudge-language-{args.tag}.safetensors"]
        if variant == "text":
            link(language, out / "model.safetensors")
        else:
            link(language, out / "model-00001-of-00002.safetensors")
            link(files[f"snapjudge-vision-tower-{args.tag}.safetensors"], out / "model-00002-of-00002.safetensors")
        print(f"ready: {out}")


if __name__ == "__main__":
    main()
