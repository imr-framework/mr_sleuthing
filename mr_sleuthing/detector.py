"""Cross-architecture detector of deep-learning MRI reconstruction (the MR-Sleuthing meta-classifier).

The detector is a ResNet-18 trained jointly on originals and on reconstructions from three architecturally
distinct reconstructors (an unrolled data-consistency CNN, a zero-shot super-resolution network, and a diffusion
model) at accelerations R = 4 and 8, in axial and sagittal slices. Given a brain MR image, it returns the
probability that the image was produced by a DL reconstruction, P(recon), and a Grad-CAM map of where the
evidence lies. No reference (fully sampled) image is needed.

Preprocessing follows training: reorientation to RAS, radiological display orientation, resize to 320 x 320,
1st to 99th percentile normalization, a brain mask at 0.03 of the normalized intensity, and ImageNet normalization
of a 3-channel copy. One difference is unavoidable: in training, each reconstructed slice was masked with the brain
mask of its paired original, which is not available at test time, so here the mask is computed from the input image
itself. On the bundled held-out example the two choices change P(recon) by at most 0.002.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence, Union

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import zoom
from torchvision import models, transforms

SIZE = 320
MASK_THRESHOLD = 0.03
DEFAULT_WEIGHTS = Path(__file__).resolve().parent.parent / "weights" / "meta_joint_deploy.pt"
_NORM = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ------------------------------------------------------------------------------------------- preprocessing
def normalize01(s: np.ndarray) -> np.ndarray:
    """Scale to [0, 1] between the 1st and 99th percentiles (as in training)."""
    s = np.abs(s).astype(np.float32)
    lo, hi = np.percentile(s, 1), np.percentile(s, 99)
    return np.clip((s - lo) / max(hi - lo, 1e-6), 0, 1)


def to_ras(volume: np.ndarray, affine: np.ndarray) -> np.ndarray:
    ornt = nib.orientations.io_orientation(affine)
    return nib.orientations.apply_orientation(
        volume, nib.orientations.ornt_transform(ornt, nib.orientations.axcodes2ornt(("R", "A", "S"))))


def axial_slice(ras: np.ndarray, z: int) -> np.ndarray:
    return np.fliplr(np.flipud(ras[:, :, z].T))


def sagittal_slice(ras: np.ndarray, x: int) -> np.ndarray:
    return np.flipud(ras[x, :, :].T)


def prepare_slice(sl: np.ndarray) -> np.ndarray:
    """Any 2-D slice -> the 320 x 320, masked, [0, 1] image the network was trained on."""
    o = normalize01(sl)
    if o.shape != (SIZE, SIZE):
        o = zoom(o, (SIZE / o.shape[0], SIZE / o.shape[1]), order=1)
    mask = (o > MASK_THRESHOLD).astype(np.float32)
    return normalize01(o * mask)


def to_tensor(img: np.ndarray) -> torch.Tensor:
    return _NORM(torch.from_numpy(np.repeat(img.astype(np.float32)[None], 3, 0)))


def central_indices(n: int, k: int) -> List[int]:
    start = max(0, n // 2 - k // 2)
    return list(range(start, min(n, start + k)))


def load_image(path: Union[str, Path]):
    """Return (array, affine) for NIfTI, or (array, None) for .npy / .png / .jpg 2-D images."""
    path = str(path)
    if path.endswith((".nii", ".nii.gz")):
        v = nib.load(path)
        a = np.asarray(v.dataobj, dtype=np.float32)
        if a.ndim == 4:
            a = a[..., 0]
        return np.squeeze(a) if a.ndim == 3 and 1 in a.shape else a, v.affine
    if path.endswith(".npy"):
        return np.load(path).astype(np.float32), None
    from PIL import Image
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32), None


# ------------------------------------------------------------------------------------------- model
def load_model(weights: Union[str, Path] = DEFAULT_WEIGHTS, device: Optional[torch.device] = None) -> nn.Module:
    device = device or default_device()
    m = models.resnet18()
    m.fc = nn.Linear(m.fc.in_features, 2)
    m.load_state_dict(torch.load(str(weights), map_location="cpu", weights_only=True))
    return m.to(device).eval()


def grad_cam(model: nn.Module, img: np.ndarray, device: torch.device) -> np.ndarray:
    """Grad-CAM on layer4 for the reconstruction class, upsampled to 320 x 320 and scaled to [0, 1]."""
    acts, grads = {}, {}
    h1 = model.layer4.register_forward_hook(lambda m, i, o: acts.__setitem__("a", o.detach()))
    h2 = model.layer4.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("g", go[0].detach()))
    try:
        x = to_tensor(img).unsqueeze(0).to(device).requires_grad_(True)
        model.zero_grad()
        out = model(x)
        out[0, 1].backward()
    finally:
        h1.remove(); h2.remove()
    w = grads["g"].mean((2, 3), keepdim=True)
    cam = torch.relu((w * acts["a"]).sum(1))[0].cpu().numpy()
    cam = zoom(cam, (SIZE / cam.shape[0], SIZE / cam.shape[1]), order=1)
    return (cam - cam.min()) / (np.ptp(cam) + 1e-8)


# ------------------------------------------------------------------------------------------- detection
@dataclass
class Detection:
    path: str
    plane: str
    probability: float                       # mean P(recon) over the analyzed slices
    slice_probabilities: List[float]
    slice_indices: List[int]
    threshold: float = 0.5
    image: Optional[np.ndarray] = field(default=None, repr=False)    # most suspicious slice (320 x 320)
    cam: Optional[np.ndarray] = field(default=None, repr=False)      # its Grad-CAM map

    @property
    def is_reconstruction(self) -> bool:
        return self.probability >= self.threshold

    @property
    def verdict(self) -> str:
        return "likely DL-reconstructed" if self.is_reconstruction else "likely an original acquisition"

    def summary(self) -> str:
        p = np.asarray(self.slice_probabilities)
        return (f"{Path(self.path).name}: P(recon) = {self.probability:.3f} -> {self.verdict} "
                f"({len(p)} {self.plane} slice{'s' if len(p) != 1 else ''}, range {p.min():.2f} to {p.max():.2f})")


class HallucinationDetector:
    """Reference-free detector of DL-reconstructed MR images with Grad-CAM localization.

    >>> det = HallucinationDetector()
    >>> result = det.detect("scan.nii.gz", plane="axial")
    >>> print(result.summary())
    """

    def __init__(self, weights: Union[str, Path] = DEFAULT_WEIGHTS, device: Optional[str] = None,
                 threshold: float = 0.5):
        self.device = torch.device(device) if device else default_device()
        self.model = load_model(weights, self.device)
        self.threshold = threshold

    @torch.no_grad()
    def probabilities(self, images: Sequence[np.ndarray]) -> List[float]:
        batch = torch.stack([to_tensor(im) for im in images]).to(self.device)
        return torch.softmax(self.model(batch), 1)[:, 1].float().cpu().tolist()

    def slices(self, array: np.ndarray, affine, plane: str, n_slices: int):
        """Prepared slices and their indices; a 2-D input is returned as a single slice."""
        if array.ndim == 2:
            return [prepare_slice(array)], [0]
        ras = to_ras(array, affine) if affine is not None else array
        if plane == "axial":
            idx = central_indices(ras.shape[2], n_slices)
            return [prepare_slice(axial_slice(ras, i)) for i in idx], idx
        if plane == "sagittal":
            idx = central_indices(ras.shape[0], n_slices)
            return [prepare_slice(sagittal_slice(ras, i)) for i in idx], idx
        raise ValueError("plane must be 'axial' or 'sagittal'")

    def detect(self, path: Union[str, Path], plane: str = "axial", n_slices: int = 9,
               localize: bool = True) -> Detection:
        array, affine = load_image(path)
        imgs, idx = self.slices(array, affine, plane, n_slices)
        probs = self.probabilities(imgs)
        res = Detection(str(path), plane if array.ndim == 3 else "2-D", float(np.mean(probs)), probs, idx,
                        self.threshold)
        if localize:
            k = int(np.argmax(probs))
            res.image, res.cam = imgs[k], grad_cam(self.model, imgs[k], self.device)
        return res


def save_overlay(result: Detection, out_png: Union[str, Path]) -> Path:
    """Write a two-panel figure: the most suspicious slice and its Grad-CAM overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    k = int(np.argmax(result.slice_probabilities))
    fig, ax = plt.subplots(1, 2, figsize=(8, 4.3))
    ax[0].imshow(result.image, cmap="gray")
    ax[0].set_title(f"Input ({result.plane}, slice {result.slice_indices[k]})")
    ax[1].imshow(result.image, cmap="gray")
    ax[1].imshow(result.cam, cmap="jet", alpha=0.45)
    ax[1].set_title(f"Grad-CAM evidence, P(recon) = {result.probability:.2f}")
    for a in ax:
        a.axis("off")
    fig.suptitle(f"{Path(result.path).name}: {result.verdict}", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out_png = Path(out_png)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return out_png
