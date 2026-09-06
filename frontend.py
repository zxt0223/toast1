"""Shared Mel-spectrogram frontend for training and evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

import torch
import torch.nn as nn
from torchaudio import transforms as T


@dataclass(frozen=True)
class FrontendConfig:
    sample_rate: int = 16000
    n_fft: int = 1024
    win_length: int = 1024
    hop_length: int = 512
    n_mels: int = 128
    f_min: float = 50.0
    f_max: float = 2500.0
    top_db: float = 80.0
    freq_mask_param: int = 15
    time_mask_param: int = 35
    normalize: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "FrontendConfig":
        if not values:
            return cls()
        valid_names = {field.name for field in fields(cls)}
        filtered = {key: value for key, value in values.items() if key in valid_names}
        return cls(**filtered)

    def to_dict(self) -> dict:
        return asdict(self)


class MelFrontend(nn.Module):
    def __init__(self, config: FrontendConfig | None = None) -> None:
        super().__init__()
        self.config = config or FrontendConfig()
        cfg = self.config
        self.mel = T.MelSpectrogram(
            sample_rate=cfg.sample_rate,
            n_fft=cfg.n_fft,
            win_length=cfg.win_length,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
            f_min=cfg.f_min,
            f_max=cfg.f_max,
            power=2.0,
            center=True,
            pad_mode="reflect",
        )
        self.to_db = T.AmplitudeToDB(stype="power", top_db=cfg.top_db)
        self.frequency_mask = T.FrequencyMasking(cfg.freq_mask_param)
        self.time_mask = T.TimeMasking(cfg.time_mask_param)

    @staticmethod
    def _standardize(spectrograms: torch.Tensor) -> torch.Tensor:
        mean = spectrograms.mean(dim=(-2, -1), keepdim=True)
        std = spectrograms.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        return (spectrograms - mean) / std

    def linear_mel(self, waveforms: torch.Tensor) -> torch.Tensor:
        return self.mel(waveforms.float())

    def finish(
        self,
        linear_mel: torch.Tensor,
        augment: bool = False,
    ) -> torch.Tensor:
        spectrograms = self.to_db(linear_mel.float().clamp_min(1e-10))
        if self.config.normalize:
            spectrograms = self._standardize(spectrograms)
        if augment:
            spectrograms = self.frequency_mask(spectrograms)
            spectrograms = self.time_mask(spectrograms)
        return spectrograms

    def forward(
        self,
        waveforms: torch.Tensor,
        augment: bool = False,
    ) -> torch.Tensor:
        return self.finish(self.linear_mel(waveforms), augment=augment)