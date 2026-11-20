"""Command line interface.

    python -m mr_sleuthing scan.nii.gz                    # axial, central 9 slices
    python -m mr_sleuthing scan.nii.gz --plane sagittal   # through-plane view (best for slice-axis models)
    python -m mr_sleuthing scan.nii.gz --plane both --out results/
    python -m mr_sleuthing slice.png                       # a single 2-D image also works
"""
import argparse
from pathlib import Path

from .detector import DEFAULT_WEIGHTS, HallucinationDetector, save_overlay


def main(argv=None):
    ap = argparse.ArgumentParser(prog="mr_sleuthing",
                                 description="Detect deep-learning MRI reconstruction and localize the evidence.")
    ap.add_argument("inputs", nargs="+", help="NIfTI volume(s) or 2-D image(s) (.nii/.nii.gz/.npy/.png/.jpg)")
    ap.add_argument("--plane", choices=["axial", "sagittal", "both"], default="axial")
    ap.add_argument("--slices", type=int, default=9, help="central slices analyzed per volume (default 9)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--out", default=None, help="folder for Grad-CAM figures (omit to skip figures)")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    ap.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    a = ap.parse_args(argv)

    det = HallucinationDetector(a.weights, a.device, a.threshold)
    planes = ["axial", "sagittal"] if a.plane == "both" else [a.plane]
    out = Path(a.out) if a.out else None
    if out:
        out.mkdir(parents=True, exist_ok=True)
    for path in a.inputs:
        for plane in planes:
            r = det.detect(path, plane=plane, n_slices=a.slices, localize=out is not None)
            print(r.summary())
            if out:
                stem = Path(path).name.replace(".nii.gz", "").replace(".nii", "")
                print("  Grad-CAM ->", save_overlay(r, out / f"{stem}_{r.plane}_gradcam.png"))
            if r.plane == "2-D":
                break


if __name__ == "__main__":
    main()
