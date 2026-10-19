"""Zip the CONTENTS of a submission folder (run.py + metadata.yaml at the zip root, assets/ inside).

Usage: python tools/package.py --src submission --out dist/<name>.zip
"""
import argparse
import filecmp
import os
import zipfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
META = os.path.join(ROOT, "starter_kit", "examples", "task1", "sample_code_submission", "metadata.yaml")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=os.path.join(ROOT, "submission"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = os.path.abspath(a.src)
    assert os.path.isfile(os.path.join(src, "run.py")), "run.py missing"
    assert filecmp.cmp(os.path.join(src, "metadata.yaml"), META, shallow=False), "metadata.yaml must be unchanged"
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with zipfile.ZipFile(a.out, "w", zipfile.ZIP_DEFLATED) as zf:
        for dp, dn, fn in os.walk(src):
            dn[:] = [d for d in dn if d != "__pycache__" and not d.startswith(".")]
            for f in sorted(fn):
                if f.endswith(".pyc"):
                    continue
                full = os.path.join(dp, f)
                arc = os.path.relpath(full, src).replace(os.sep, "/")
                # model weights are already dense; store them to save time
                zf.write(full, arc, compress_type=zipfile.ZIP_STORED if f.endswith(".pt") else zipfile.ZIP_DEFLATED)
    with zipfile.ZipFile(a.out) as zf:
        print(f"{'size':>12}  name")
        for i in zf.infolist():
            print(f"{i.file_size:>12}  {i.filename}")
        names = zf.namelist()
    assert "run.py" in names and "metadata.yaml" in names, "run.py/metadata.yaml must be at the zip root"
    print(f"zip: {a.out}  {os.path.getsize(a.out) / 2**20:.1f} MB")


if __name__ == "__main__":
    main()
