"""Smoke tests on the bundled held-out example (BraTS 001, T1). Run with:  pytest -q"""
from pathlib import Path

import numpy as np
import pytest

from mr_sleuthing import HallucinationDetector

EX = Path(__file__).resolve().parent.parent / "examples"


@pytest.fixture(scope="module")
def det():
    return HallucinationDetector(device="cpu")


@pytest.mark.parametrize("plane", ["axial", "sagittal"])
def test_original_is_not_flagged(det, plane):
    r = det.detect(EX / "BRATS_001_T1_original.nii.gz", plane=plane, localize=False)
    assert not r.is_reconstruction and r.probability < 0.2


@pytest.mark.parametrize("fname,plane", [
    ("BRATS_001_T1_modelA_dcnn_R4.nii.gz", "axial"),
    ("BRATS_001_T1_modelB_zssr_R4.nii.gz", "sagittal"),
    ("BRATS_001_T1_modelC_ppn_R4_slice77.nii.gz", "axial"),
])
def test_each_architecture_is_flagged(det, fname, plane):
    r = det.detect(EX / fname, plane=plane, localize=False)
    assert r.is_reconstruction and r.probability > 0.8


def test_gradcam_map(det):
    r = det.detect(EX / "BRATS_001_T1_modelA_dcnn_R4.nii.gz", localize=True)
    assert r.cam.shape == (320, 320) and np.isfinite(r.cam).all() and 0 <= r.cam.min() <= r.cam.max() <= 1


def test_cli_runs(tmp_path, capsys):
    from mr_sleuthing.cli import main
    main([str(EX / "BRATS_001_T1_original.nii.gz"), "--plane", "both", "--out", str(tmp_path), "--device", "cpu"])
    assert "P(recon)" in capsys.readouterr().out and len(list(tmp_path.glob("*.png"))) == 2
