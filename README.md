# MR-Sleuthing

**Reference-free, cross-architecture detection and localization of deep-learning MRI reconstruction.**

Deep-learning (DL) reconstruction lets MRI scans be acquired faster, but the networks can *hallucinate*:
they add, remove or reshape plausible anatomy in a way a reader cannot see and that global metrics such as
PSNR and SSIM miss. MR-Sleuthing gives you a detector that looks at a single brain MR image, with **no fully
sampled reference**, and returns

* `P(recon)`, the probability that the image was produced by a DL reconstruction, and
* a **Grad-CAM map** showing where in the image the evidence lies.

The detector is a single meta-classifier trained on reconstructions from three architecturally distinct
reconstructors, so it learns what DL reconstruction leaves behind in general rather than the quirks of one
network:

| | Reconstructor | Family | Characteristic failure mode |
|---|---|---|---|
| Model A | Unrolled CNN with conjugate-gradient data consistency (MoDL family) | supervised, physics-guided | smoothing of fine detail |
| Model B | Zero-shot super-resolution (ZSSR) | self-supervised, test-time | through-plane interpolation |
| Model C | Predictor-projector-noisor diffusion (PPN) | generative prior | synthesis of fine texture |

## Quick start

```bash
git clone https://github.com/imr-framework/mr_sleuthing.git
cd mr_sleuthing
pip install -r requirements.txt        # or: pip install -e .
python run_demo.py
```

The demo runs the detector on one **held-out** subject (Medical Segmentation Decathlon BraTS 001, T1), first
as the original acquisition and then as reconstructed by each of the three models at acceleration R = 4:

```
input                                        truth                    plane     P(recon)  verdict
BRATS_001_T1_original.nii.gz                 original                 axial        0.034  likely an original acquisition
BRATS_001_T1_original.nii.gz                 original                 sagittal     0.001  likely an original acquisition
BRATS_001_T1_modelA_dcnn_R4.nii.gz           Model A (unrolled CNN)   axial        0.962  likely DL-reconstructed
BRATS_001_T1_modelB_zssr_R4.nii.gz           Model B (zero-shot SR)   sagittal     1.000  likely DL-reconstructed
BRATS_001_T1_modelC_ppn_R4_slice77.nii.gz    Model C (diffusion)      2-D          0.999  likely DL-reconstructed
```

Grad-CAM figures for each case are written to `demo_output/`.

## Test your own images

```bash
python -m mr_sleuthing scan.nii.gz                               # axial view, central 9 slices
python -m mr_sleuthing scan.nii.gz --plane both --out results/   # axial + sagittal, save Grad-CAM figures
python -m mr_sleuthing a.nii.gz b.nii.gz c.png                   # several files; 2-D images also work
```

Or from Python:

```python
from mr_sleuthing import HallucinationDetector, save_overlay

det = HallucinationDetector()                        # uses CUDA, Apple MPS or CPU automatically
result = det.detect("scan.nii.gz", plane="sagittal")
print(result.summary())                              # P(recon), verdict, per-slice range
print(result.slice_probabilities)                    # one probability per analyzed slice
save_overlay(result, "scan_gradcam.png")             # most suspicious slice + Grad-CAM evidence
```

Inputs can be NIfTI volumes (`.nii`, `.nii.gz`, reoriented automatically) or single 2-D slices
(`.nii`, `.npy`, `.png`, `.jpg`). For a volume, the detector scores the central slices in the chosen plane and
reports their mean. Inputs are expected to be skull-stripped, like the training data. Preprocessing follows
training, except that the brain mask is computed from the input itself (in training it came from the paired
original, which a real test case does not have); on the bundled example this changes `P(recon)` by at most 0.002. **Use `--plane sagittal` (or `both`) for reconstructors that work along the slice axis**,
such as super-resolution networks: their interpolation is often invisible in-plane and obvious through-plane.

## Performance

Held-out subjects (no subject appears in both training and test sets), from the SPIE Medical Imaging 2027
paper below:

| Setting | AUC |
|---|---|
| Joint meta-classifier (this release), all conditions | **0.968** |
| Per condition (3 models x 2 planes x R = 4, 8; Model C axial only) | 0.91 to 1.00 |
| Model B, axial vs sagittal, R = 4 | 0.91 vs 0.99 |
| Leave-one-model-out (detector never saw that architecture): A / B / C | 0.86 / 0.83 / 0.97 |

The leave-one-model-out row is the closest estimate of how the detector behaves on a reconstructor it was
not trained on. Continual learning with replay and elastic weight consolidation, which adds new architectures
without forgetting old ones, reached 0.977 / 0.972 / 0.999 on Models A / B / C (see `training/`).

## Scope and limitations

* Trained on brain MRI (T1-weighted, T2-weighted and FLAIR) at accelerations R = 4 and 8. Behavior on other
  anatomy, contrasts or accelerations is untested.
* The detector answers "was this image DL-reconstructed, and where is the evidence?". A high `P(recon)` is a
  flag for closer inspection, not a diagnosis of a specific hallucination. Grad-CAM localization has been
  assessed qualitatively and by agreement between two backbones (ResNet-18 and DenseNet-121); quantitative
  localization against synthetic ground truth is in progress.
* Research software. Not for clinical use.

## Repository layout

```
mr_sleuthing/        detector package (preprocessing, model, Grad-CAM, CLI)
weights/             meta_joint_deploy.pt, the trained ResNet-18 meta-classifier (45 MB)
examples/            held-out BraTS 001 T1: original + Model A, B and C reconstructions (see ATTRIBUTION.md)
run_demo.py          end-to-end demo
tests/               pytest smoke tests on the examples
training/            the scripts that built the training slices and trained the meta-classifier
```

Run the tests with `pip install pytest && pytest -q`.

## Citation

If you use MR-Sleuthing, please cite:

> A. Shah, A. Sharma, and S. Geethanath, "MR-Sleuthing: cross-architecture detection and localization of MRI
> reconstruction hallucinations via explainable continual learning," *Proc. SPIE Medical Imaging 2027*,
> paper 14522-15 (to appear).

## Acknowledgments

Research reported here was supported by the National Institute of Mental Health of the National Institutes of
Health under Award Number R21MH136520 and by the Johns Hopkins Provost's DELTA award. Developed at the
Accessible Magnetic Resonance Laboratory, Johns Hopkins University School of Medicine.

## License

Code and weights: MIT (see `LICENSE`). The example images are derived from the Medical Segmentation Decathlon
and are distributed under CC BY-SA 4.0 (see `examples/ATTRIBUTION.md`).
