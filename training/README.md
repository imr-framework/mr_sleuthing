# Training the meta-classifier

These are the scripts that produced `weights/meta_joint_deploy.pt`. They are included for transparency and
reproducibility; the reconstructions themselves are not distributed here.

1. **`meta_build_cache_v2.py`** builds the training cache. It expects a data folder (set `MRS_DATAPOOL`) with
   * `pool/<subject>.nii.gz`: skull-stripped original volumes, and
   * `recons/dcnn/`, `recons/zssr/`, `recons/ppn/`: Model A, B and C reconstructions of those subjects at R = 4 and 8.

   It extracts central axial and sagittal slices (axial only for Model C), applies the detector's preprocessing
   (with each reconstructed slice masked by the brain mask of its paired original), and assigns each subject to the train or test split deterministically, so no subject is in both.
2. **`meta_train_v2.py`** trains and evaluates the ResNet-18 detector: the joint deployment model (saved as
   `meta_joint_deploy.pt`), continual learning with experience replay and EWC, a naive fine-tuning baseline, and
   leave-one-model-out generalization.

```bash
export MRS_DATAPOOL=/path/to/datapool MRS_OUT=outputs/meta_classifier
python training/meta_build_cache_v2.py
python training/meta_train_v2.py
```

Pool sources: Medical Segmentation Decathlon (BraTS), NFBS, and OpenNeuro releases.
