"""Run the detector on the bundled example: one held-out subject (BraTS 001, T1) as the original acquisition
and as reconstructed by each of the three architectures at R = 4.

    python run_demo.py              # prints a table and writes Grad-CAM figures to demo_output/
"""
from pathlib import Path

from mr_sleuthing import HallucinationDetector, save_overlay

EX = Path(__file__).resolve().parent / "examples"
CASES = [  # (file, ground truth, plane where that architecture's artifacts are most visible)
    ("BRATS_001_T1_original.nii.gz", "original", "axial"),
    ("BRATS_001_T1_original.nii.gz", "original", "sagittal"),
    ("BRATS_001_T1_modelA_dcnn_R4.nii.gz", "Model A (unrolled CNN)", "axial"),
    ("BRATS_001_T1_modelB_zssr_R4.nii.gz", "Model B (zero-shot SR)", "sagittal"),
    ("BRATS_001_T1_modelC_ppn_R4_slice77.nii.gz", "Model C (diffusion)", "axial"),
]

if __name__ == "__main__":
    out = Path("demo_output"); out.mkdir(exist_ok=True)
    det = HallucinationDetector()
    print(f"device: {det.device}\n")
    print(f"{'input':44s} {'truth':24s} {'plane':9s} {'P(recon)':>8s}  verdict")
    for fname, truth, plane in CASES:
        r = det.detect(EX / fname, plane=plane)
        print(f"{fname:44s} {truth:24s} {r.plane:9s} {r.probability:8.3f}  {r.verdict}")
        save_overlay(r, out / f"{fname.replace('.nii.gz', '')}_{r.plane}.png")
    print(f"\nGrad-CAM figures written to {out}/")
