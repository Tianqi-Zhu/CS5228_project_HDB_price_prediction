"""Download a licensed TabPFN-3 or TabPFN-3.5-Fast checkpoint.

Fetches bounded ranges in parallel, verifies the model hash supplied by the
Hub, and only moves a complete checkpoint into the SDK cache.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
from pathlib import Path
import re

import requests


MODELS = {
    "v3": ("tabpfn_3", "tabpfn-v3-regressor-v3_default.ckpt"),
    "v3.5-fast": ("tabpfn_3_5", "tabpfn-v3.5-fast-20260909.safetensors"),
}


def fetch_part(index: int, start: int, end: int, total: int, url: str, parts: Path) -> Path:
    path = parts / f"{index:04d}.part"
    expected = end - start + 1
    if path.exists() and path.stat().st_size == expected:
        return path
    last_error = None
    for _ in range(3):
        try:
            with requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}"},
                stream=True,
                timeout=(15, 60),
            ) as response:
                response.raise_for_status()
                if response.status_code != 206 or response.headers.get("Content-Range") != f"bytes {start}-{end}/{total}":
                    raise RuntimeError(f"Unexpected range response for part {index}")
                with path.open("wb") as output:
                    for chunk in response.iter_content(1024 * 1024):
                        output.write(chunk)
            if path.stat().st_size != expected:
                raise RuntimeError(f"Short response for part {index}")
            return path
        except (OSError, requests.RequestException, RuntimeError) as exc:
            last_error = exc
    raise RuntimeError(f"Part {index} failed") from last_error


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-version", choices=MODELS, default="v3")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=4)
    args = parser.parse_args()
    repo, filename = MODELS[args.model_version]
    url = f"https://huggingface.co/Prior-Labs/{repo}/resolve/main/{filename}"
    target = Path(".cache/tabpfn") / filename
    part_bytes = (1 if args.model_version == "v3.5-fast" else 8) * 1024 * 1024
    parts = target.with_suffix(".parts")
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        print(f"Checkpoint already present: {target}", flush=True)
        return
    response = requests.head(url, allow_redirects=False, timeout=(15, 30))
    response.raise_for_status()
    total = int(response.headers["X-Linked-Size"])
    etag = response.headers["X-Linked-ETag"].strip('"')
    parts.mkdir(exist_ok=True)
    bounds = [(i, start, min(start + part_bytes, total) - 1, total)
              for i, start in enumerate(range(0, total, part_bytes))]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = [pool.submit(fetch_part, *bound, url, parts) for bound in bounds]
        for number, job in enumerate(as_completed(jobs), 1):
            job.result()
            if number % 8 == 0 or number == len(bounds):
                print(f"Downloaded {number}/{len(bounds)} parts", flush=True)

    temporary = target.with_suffix(".assembling")
    digest = hashlib.sha256()
    with temporary.open("wb") as output:
        for index in range(len(bounds)):
            with (parts / f"{index:04d}.part").open("rb") as part:
                for chunk in iter(lambda: part.read(1024 * 1024), b""):
                    digest.update(chunk)
                    output.write(chunk)
    if temporary.stat().st_size != total:
        raise RuntimeError("Assembled checkpoint has incorrect size")
    if not re.fullmatch(r"[0-9a-fA-F]{64}", etag) or digest.hexdigest() != etag.lower():
        raise RuntimeError("Checkpoint SHA-256 does not match model host")
    temporary.replace(target)
    print(f"Checkpoint ready: {target} ({total:,} bytes)", flush=True)


if __name__ == "__main__":
    main()
