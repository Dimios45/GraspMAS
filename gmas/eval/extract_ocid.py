"""Selectively extract OCID-VLG: annotations + only the RGB images referenced
by the test split (full extraction won't fit in the disk quota).

Usage: python extract_ocid.py <zip> <dest_root>
"""
import json
import sys
import zipfile
from pathlib import Path

zip_path, dest = sys.argv[1], Path(sys.argv[2])
zf = zipfile.ZipFile(zip_path)
names = zf.namelist()

# detect an optional single top-level folder inside the zip
tops = {n.split("/", 1)[0] for n in names if "/" in n}
prefix = (tops.pop() + "/") if len(tops) == 1 else ""
print(f"zip members: {len(names)}, prefix: {prefix!r}")

# 1) all annotation JSONs (refer/) — small
ann = [n for n in names if "/refer/" in n or n.startswith(prefix + "refer/")]
print(f"extracting {len(ann)} annotation files ...")
zf.extractall(dest, members=ann)

# 2) collect image paths referenced by every *_expressions.json we extracted
refer_root = dest / prefix / "refer"
wanted = set()
for j in refer_root.rglob("*_expressions.json"):
    data = json.load(open(j))["data"]
    for s in data:
        rel = s["image_filename"].replace(",", "/rgb/")
        wanted.add(rel)
        wanted.add(rel.replace("/rgb/", "/depth/"))  # depth used by some tools
print(f"referenced images: {len(wanted)}")

members = [n for n in names if n.removeprefix(prefix) in wanted]
print(f"matching zip members: {len(members)}")
missing = wanted - {n.removeprefix(prefix) for n in members}
if missing:
    print(f"WARNING: {len(missing)} referenced files not in zip, e.g. {sorted(missing)[:3]}")

done = 0
for m in members:
    zf.extract(m, dest)
    done += 1
    if done % 500 == 0:
        print(f"  {done}/{len(members)}")
print(f"extracted {done} images into {dest}")
