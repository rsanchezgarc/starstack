import functools
from os import PathLike
from typing import Union, Literal

import torch
from scipy.spatial.transform import Rotation as R
from torch.utils.data import Dataset

from supervisedPoseBenchmark.ctf import apply_ctf
from supervisedPoseBenchmark.particlesStar import ParticlesSet


class ParticlesDataset(Dataset):
    def __init__(self, targetName: Union[PathLike, str],
                 partition: Literal[1, 2],
                 downloadPath: str,
                 apply_perImg_normalization: bool = True,
                 ctf_correction: Literal["none", "phase_flip", "wiener"] = "phase_flip"
                 ):

        super().__init__()
        self.targetName = targetName
        self.partition = partition
        self.downloadPath = downloadPath
        self.apply_perImg_normalization = apply_perImg_normalization
        self.ctf_correction = ctf_correction

        self._particles = None

    @property
    def starFname(self):
        return f"{self.targetName}_{self.partition}.star"

    @property
    def particles(self):
        if self._particles is None:
            self._particles = ParticlesSet(starFname=self.starFname, particlesDir=None)
        return self._particles

    @functools.lru_cache(1)
    def getParticleNormalizationMask(self, particleNumPixels,
                                     normalizationRadiusPixels=None,
                                     device=None):
        """
        Get the mask with 1s in the corners and 0 for the center
        :param particleNumPixels:
        :param normalizationRadiusPixels:
        :param device:
        :return:
        """
        radius = particleNumPixels // 2
        if normalizationRadiusPixels is None:
            normalizationRadiusPixels = radius
        ies, jes = torch.meshgrid(
            torch.linspace(-1 * radius, 1 * radius, particleNumPixels, dtype=torch.float32),
            torch.linspace(-1 * radius, 1 * radius, particleNumPixels, dtype=torch.float32),
            indexing="ij"
        )
        r = (ies ** 2 + jes ** 2) ** 0.5
        _normalizationMask = (r > normalizationRadiusPixels)
        _normalizationMask = _normalizationMask.to(device)
        return _normalizationMask

    def _normalize(self, img):
        backgroundMask = self.getParticleNormalizationMask(img.shape[-1])
        noiseRegion = img[:, backgroundMask]
        meanImg = noiseRegion.mean()
        stdImg = noiseRegion.std()
        return (img - meanImg) / stdImg

    def __getitem__(self, item):
        img, metadata = self.particles[item]

        if self.apply_perImg_normalization:
            img = self._normalize(img)

        rotMat = R.from_euler("ZYZ", [metadata[name] for name in ['rlnAngleRot', 'rlnAngleTilt', 'rlnAnglePsi']],
                              degrees=True).as_matrix()

        if self.ctf_correction != "none":
            ctf, wimg = apply_ctf(img, self.particles.sampling_rate, dfu=metadata["rlnDefocusU"],
                                  dfv=metadata["rlnDefocusV"],
                                  dfang=metadata["rlnDefocusAngle"], volt=float(self.particles.optics["rlnVoltage"][0]),
                                  cs=float(self.particles.optics["rlnSphericalAberration"][0]),
                                  w=float(self.particles.optics["rlnAmplitudeContrast"][0]),
                                  mode=self.ctf_correction)
            wimg = torch.clamp(wimg, img.min(), img.max())
            wimg = torch.nan_to_num(wimg, nan=img.mean())
            img = wimg

        return img, rotMat

    def __len__(self):
        return len(self.particles)
