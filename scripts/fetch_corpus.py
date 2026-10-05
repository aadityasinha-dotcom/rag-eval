"""Fetch the raw corpus into corpus/ from the manifest in scripts/sources.txt.

Usage:
    python scripts/fetch_corpus.py [--manifest PATH] [--dest DIR] [--force]

Manifest line:  <url> [<target>] [subdir=<path/inside/archive>] [sha256=<hex>]

Stdlib only, so it runs before the project venv exists. Files that already
exist are skipped unless --force is given, so re-running is cheap.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MANIFEST = ROOT / "scripts" / "sources.txt"
DEFAULT_DEST = ROOT / "corpus"
ARCHIVE_SUFFIXES = (".zip", ".tar.gz", ".tgz", ".tar")


@dataclass(frozen=True)
class Source:
    url: str
    target: str | None = None  # path relative to corpus/
    subdir: str | None = None  # archives only: extract just this directory, prefix stripped
    sha256: str | None = None


def read_manifest(path: Path) -> list[Source]:
    entries: list[Source] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        url, rest = parts[0], parts[1:]
        opts = {k: v for k, _, v in (p.partition("=") for p in rest if "=" in p)}
        positional = [p for p in rest if "=" not in p]
        if len(positional) > 1:
            raise ValueError(f"bad manifest line: {raw!r}")
        unknown = set(opts) - {"subdir", "sha256"}
        if unknown:
            raise ValueError(f"unknown option(s) {sorted(unknown)} in: {raw!r}")
        entries.append(
            Source(
                url=url,
                target=positional[0] if positional else None,
                subdir=opts.get("subdir", "").strip("/") or None,
                sha256=opts.get("sha256"),
            )
        )
    return entries


def is_archive(url: str) -> bool:
    return url.lower().endswith(ARCHIVE_SUFFIXES)


def download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "rag-eval-fetch/0.1"})
    with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as out:
        shutil.copyfileobj(resp, out)
    tmp.replace(dest)


def verify_sha256(path: Path, expected: str) -> None:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    if h.hexdigest() != expected.lower():
        raise RuntimeError(f"sha256 mismatch for {path.name}: got {h.hexdigest()}, want {expected}")


def _strip_subdir(name: str, subdir: str | None) -> str | None:
    """Path of an archive member relative to `subdir`, or None if it lies outside it."""
    if subdir is None:
        return name
    prefix = subdir + "/"
    if name == subdir or name.startswith(prefix):
        return name[len(prefix) :]
    return None


def safe_extract_dir(archive: Path, into: Path, subdir: str | None = None) -> int:
    """Extract (regular files only) into `into`, refusing paths that escape it.

    With `subdir`, only members under that directory are extracted and the
    prefix is removed, so archive `root/doc/html/x.html` with subdir
    `root/doc/html` lands at `into/x.html`. Returns the number of files written.
    """
    into.mkdir(parents=True, exist_ok=True)
    resolved = into.resolve()
    written = 0

    def dest_for(member_name: str) -> Path | None:
        rel = _strip_subdir(member_name, subdir)
        if not rel:
            return None
        dest = (resolved / rel).resolve()
        if not dest.is_relative_to(resolved):
            raise RuntimeError(f"refusing to extract {member_name!r} outside {into}")
        return dest

    if archive.name.lower().endswith(".zip"):
        with zipfile.ZipFile(archive) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                dest = dest_for(info.filename)
                if dest is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
    else:
        with tarfile.open(archive) as tf:
            for member in tf:
                if not member.isfile():
                    continue
                dest = dest_for(member.name)
                if dest is None:
                    continue
                src = tf.extractfile(member)
                if src is None:
                    continue
                dest.parent.mkdir(parents=True, exist_ok=True)
                with src, dest.open("wb") as out:
                    shutil.copyfileobj(src, out)
                written += 1
    if written == 0:
        raise RuntimeError(f"nothing extracted from {archive.name} (subdir={subdir!r})")
    return written


def fetch_entry(src: Source, dest_root: Path, force: bool) -> None:
    url = src.url
    if is_archive(url):
        out_dir = dest_root / src.target if src.target else dest_root
        marker = out_dir / ".fetched"
        if marker.exists() and marker.read_text(encoding="utf-8").strip() == url and not force:
            print(f"skip   {url} (already extracted to {out_dir})")
            return
        print(f"fetch  {url}")
        with tempfile.TemporaryDirectory() as tmp:
            archive = Path(tmp) / Path(url).name
            download(url, archive)
            if src.sha256:
                verify_sha256(archive, src.sha256)
                print("sha256 ok")
            where = f" [{src.subdir}]" if src.subdir else ""
            print(f"unpack {archive.name}{where} -> {out_dir}")
            n = safe_extract_dir(archive, out_dir, src.subdir)
            print(f"wrote  {n} files")
        marker.write_text(url + "\n", encoding="utf-8")
        return

    out_file = dest_root / (src.target or Path(url).name)
    if out_file.exists() and not force:
        print(f"skip   {url} (exists: {out_file})")
        return
    print(f"fetch  {url} -> {out_file}")
    download(url, out_file)
    if src.sha256:
        verify_sha256(out_file, src.sha256)
        print("sha256 ok")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    entries = read_manifest(args.manifest)
    if not entries:
        print(f"No sources listed in {args.manifest}. Add URLs there first.", file=sys.stderr)
        return 1
    for src in entries:
        fetch_entry(src, args.dest, args.force)
    print(f"corpus ready in {args.dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
