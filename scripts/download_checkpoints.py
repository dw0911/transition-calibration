# -*- coding: utf-8 -*-
"""Download the companion checkpoint data package.

Usage:
    # 1. Set the download URL (a .zip or .tar.gz archive containing the checkpoints/ layout)
    #    -- either via environment variable ...
    export CHECKPOINT_URL=https://github.com/dw0911/transition-calibration/releases/download/v1.0/checkpoints_pems.zip
    python scripts/download_checkpoints.py

    #    -- or as a command-line argument ...
    python scripts/download_checkpoints.py --url https://example.com/checkpoints_pems.zip

    # 2. Optionally verify the archive integrity against an expected SHA-256:
    python scripts/download_checkpoints.py --sha256 <expected-hex>

The archive must unpack to the repository's `checkpoints/` directory with the layout documented
in `checkpoints/README.md` (e.g. `PEMS04_r3_4split_seed42/best.pt`).
"""
import argparse
import hashlib
import io
import os
import sys
import tarfile
import urllib.request
import zipfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_DEST = os.path.join(REPO_ROOT, 'checkpoints')


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_archive(data: bytes, dest: str):
    """Extract a .zip or .tar.gz archive in memory into `dest` (paths sanitized)."""
    if data[:2] == b'PK':                       # zip magic
        zf = zipfile.ZipFile(io.BytesIO(data))
        for info in zf.infolist():
            # sanitize: skip absolute / parent-traversing paths
            name = info.filename.replace('\\', '/')
            if name.startswith('/') or '..' in name.split('/'):
                continue
            target = os.path.normpath(os.path.join(dest, name))
            if not target.startswith(os.path.normpath(dest)):
                continue
            if info.is_dir():
                os.makedirs(target, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb') as f:
                    f.write(zf.read(info))
    else:                                       # assume tar.gz
        tf = tarfile.open(fileobj=io.BytesIO(data), mode='r:gz')
        for member in tf.getmembers():
            name = member.name.replace('\\', '/')
            if name.startswith('/') or '..' in name.split('/'):
                continue
            target = os.path.normpath(os.path.join(dest, name))
            if not target.startswith(os.path.normpath(dest)):
                continue
            if member.isdir():
                os.makedirs(target, exist_ok=True)
            elif member.isfile():
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with open(target, 'wb') as f:
                    f.write(tf.extractfile(member).read())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default=os.environ.get('CHECKPOINT_URL', ''),
                    help='URL of the checkpoint archive (.zip or .tar.gz)')
    ap.add_argument('--sha256', default='', help='expected SHA-256 of the archive')
    ap.add_argument('--dest', default=DEFAULT_DEST, help='destination directory (default: <repo>/checkpoints)')
    args = ap.parse_args()

    if not args.url:
        print('ERROR: no --url provided (or CHECKPOINT_URL env var is empty).', file=sys.stderr)
        sys.exit(2)

    print(f'Downloading {args.url} ...', flush=True)
    with urllib.request.urlopen(args.url, timeout=120) as resp:
        data = resp.read()

    if args.sha256:
        got = sha256_of(data)
        if got != args.sha256.lower():
            print(f'SHA-256 mismatch: expected {args.sha256}, got {got}', file=sys.stderr)
            sys.exit(1)
        print(f'SHA-256 OK ({got})')

    os.makedirs(args.dest, exist_ok=True)
    extract_archive(data, args.dest)
    print(f'Extracted into {args.dest}')
    print('Verify the layout with: ls ' + os.path.join(args.dest, 'PEMS04_r3_4split_seed42'))


if __name__ == '__main__':
    main()
