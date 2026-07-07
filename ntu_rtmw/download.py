import argparse
import sys
import urllib.request
from pathlib import Path

from .constants import (
    MODELS_DIR,
    RTMDET_WEIGHTS_PATH,
    RTMDET_WEIGHTS_URL,
    RTMW_WEIGHTS_PATH,
    RTMW_WEIGHTS_URL,
)


def download_file(url, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".part")
    print("download {} -> {}".format(url, path), flush=True)
    with urllib.request.urlopen(url) as response, temp.open("wb") as f:
        total = int(response.headers.get("Content-Length", "0") or "0")
        done = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if total:
                print("\r  {:.1f}%".format(done * 100 / total), end="", flush=True)
    if total:
        print(flush=True)
    temp.replace(path)
    return path


def ensure_weights(url, path, label, force=False):
    path = Path(path)
    if path.exists() and path.stat().st_size > 1024 * 1024 and not force:
        print("{} weights ready: {}".format(label, path), flush=True)
        return path
    try:
        return download_file(url, path)
    except Exception as exc:
        raise SystemExit(
            "Failed to download {} weights. You can download manually:\n"
            "{}\n"
            "and save it as:\n"
            "{}".format(label, url, path)
        ) from exc


def ensure_rtmw_weights(force=False):
    return ensure_weights(RTMW_WEIGHTS_URL, RTMW_WEIGHTS_PATH, "RTMW", force=force)


def ensure_rtmdet_weights(force=False):
    return ensure_weights(RTMDET_WEIGHTS_URL, RTMDET_WEIGHTS_PATH, "RTMDet", force=force)


def parser():
    p = argparse.ArgumentParser(description="Download RTMW and RTMDet weights into the local models directory.")
    p.add_argument("--force", action="store_true")
    p.add_argument("--models-dir", default=str(MODELS_DIR), help="Shown for convenience; weights path is fixed.")
    p.add_argument("--rtmw-only", action="store_true")
    return p


def main():
    args = parser().parse_args()
    if Path(args.models_dir).resolve() != MODELS_DIR.resolve():
        print("Using fixed project models dir: {}".format(MODELS_DIR), flush=True)
    ensure_rtmw_weights(force=args.force)
    if not args.rtmw_only:
        ensure_rtmdet_weights(force=args.force)


if __name__ == "__main__":
    sys.exit(main())
