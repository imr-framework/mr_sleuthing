# Example data attribution

All example images come from subject **BRATS_001** (T1-weighted) of the **Medical Segmentation Decathlon**,
Task 01 (brain tumour), which redistributes the BraTS challenge data under the
[Creative Commons Attribution-ShareAlike 4.0 International license (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

> M. Antonelli, A. Reinke, S. Bakas, et al., "The Medical Segmentation Decathlon," *Nat. Commun.* 13, 4128 (2022).
> http://medicaldecathlon.com/

Modifications: the original volume was skull-stripped (HD-BET) and intensity-normalized. The three reconstructions
were produced from it by retrospective 4-fold undersampling followed by reconstruction with Model A (unrolled CNN),
Model B (ZSSR) and Model C (PPN diffusion; one axial slice, z = 77). Model A and B volumes are stored as scaled
16-bit integers (maximum relative error below 1e-5). These derived files are shared under the same CC BY-SA 4.0
license. This subject is in the detector's held-out test split.

| File | Content |
|---|---|
| `BRATS_001_T1_original.nii.gz` | original acquisition (240 x 240 x 155) |
| `BRATS_001_T1_modelA_dcnn_R4.nii.gz` | Model A reconstruction, R = 4 |
| `BRATS_001_T1_modelB_zssr_R4.nii.gz` | Model B reconstruction, R = 4 |
| `BRATS_001_T1_modelC_ppn_R4_slice77.nii.gz` | Model C reconstruction of axial slice 77, R = 4 (320 x 320) |
