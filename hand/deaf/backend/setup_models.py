"""
Run this once before starting the server:

  python setup_models.py

Both PyTorch model checkpoints live inside assets/best_gloss_model.pt/ as
extracted zip directories.  This script re-zips each one and places it in
backend/model_weights/ where main.py expects them.
"""

import os
import zipfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_ASSETS = os.path.join(_HERE, "..", "assets", "best_gloss_model.pt")
_OUT = os.path.join(_HERE, "model_weights")

# (inner_dir_name_inside_assets,  target_zip_filename)
_MODELS = [
    ("best_gloss_model",   "best_gloss_model.pt"),
    ("best_model (1).pth", "best_model.pth"),
]

os.makedirs(_OUT, exist_ok=True)


def _zip_model_dir(src_dir: str, dst_zip: str):
    """
    Re-create the PyTorch zip archive from an extracted model directory.
    torch.load expects the archive root to contain the model sub-folder.
    """
    parent = os.path.dirname(src_dir)
    with zipfile.ZipFile(dst_zip, "w", compression=zipfile.ZIP_STORED) as zf:
        for root, _, files in os.walk(src_dir):
            for fname in files:
                abs_path = os.path.join(root, fname)
                # arc name is relative to the parent so the sub-folder name
                # appears at the top level inside the zip
                arc_name = os.path.relpath(abs_path, parent)
                zf.write(abs_path, arc_name)
    size_kb = os.path.getsize(dst_zip) // 1024
    print(f"  ✓  {os.path.basename(src_dir)}  →  {dst_zip}  ({size_kb} KB)")


def main():
    print("Setting up model weights …\n")
    for inner_dir, target_name in _MODELS:
        src = os.path.join(_ASSETS, inner_dir)
        dst = os.path.join(_OUT, target_name)

        if not os.path.isdir(src):
            print(f"  ✗  Source not found: {src}")
            continue

        _zip_model_dir(src, dst)

    print("\nDone.  Start the server with:")
    print("  cd backend && uvicorn main:app --reload --host 0.0.0.0 --port 8000\n")


if __name__ == "__main__":
    main()
