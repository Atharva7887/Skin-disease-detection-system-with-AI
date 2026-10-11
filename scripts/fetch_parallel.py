"""Resumable, multi-connection downloader (HTTP Range) for large dataset archives.

    python scripts/fetch_parallel.py URL OUT_FILE [--parts 8]

Each part is saved as OUT_FILE.part<i> and resumes where it stopped; the parts are joined
into OUT_FILE once all are complete and the final size matches the server's.
"""
import argparse
import os
import shutil
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 1 << 20


def total_size(url):
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as r:
        return int(r.headers["Content-Length"])


def fetch_part(url, path, start, end):
    done = os.path.getsize(path) if os.path.exists(path) else 0
    if start + done > end:
        return
    req = urllib.request.Request(url, headers={"Range": f"bytes={start + done}-{end}"})
    with urllib.request.urlopen(req, timeout=60) as r, open(path, "ab") as f:
        while block := r.read(CHUNK):
            f.write(block)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    ap.add_argument("out")
    ap.add_argument("--parts", type=int, default=8)
    ap.add_argument("--retries", type=int, default=20)
    a = ap.parse_args()

    n = total_size(a.url)
    step = -(-n // a.parts)
    ranges = [(i, i * step, min((i + 1) * step, n) - 1) for i in range(a.parts)]
    print(f"{n / 1e6:.0f} MB in {a.parts} parts", flush=True)

    def work(r):
        i, s, e = r
        path = f"{a.out}.part{i}"
        for attempt in range(a.retries):
            try:
                fetch_part(a.url, path, s, e)
                if os.path.getsize(path) == e - s + 1:
                    return
            except Exception as exc:  # network hiccup: retry and resume
                print(f"part {i} retry {attempt + 1}: {exc}", flush=True)
        raise RuntimeError(f"part {i} failed")

    with ThreadPoolExecutor(a.parts) as ex:
        list(ex.map(work, ranges))

    with open(a.out, "wb") as out:
        for i, _, _ in ranges:
            with open(f"{a.out}.part{i}", "rb") as p:
                shutil.copyfileobj(p, out, CHUNK)
    if os.path.getsize(a.out) != n:
        sys.exit("size mismatch after join")
    for i, _, _ in ranges:
        os.remove(f"{a.out}.part{i}")
    print("done", flush=True)


if __name__ == "__main__":
    main()
