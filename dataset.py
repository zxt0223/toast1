"""Dataset utilities for the ICBHI respiratory-sound benchmark."""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path
from typing import Optional, Sequence

import pandas as pd
import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio import transforms as T


CLASS_NAMES = ("Normal", "Crackle", "Wheeze", "Both")

DEVICE_MAP = {
    "AKGC417L": 0,
    "LittC2SE": 1,
    "Litt3200": 1,
    "Meditron": 2,
    "WelchAllyn": 3,
}


def safe_torch_load(path: os.PathLike | str):
    """Load a trusted local cache on CPU across PyTorch versions."""

    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _binary_flag(value) -> bool:
    if pd.isna(value):
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(int(value))


def label_from_flags(crackles, wheezes) -> int:
    """Map crackle/wheeze flags to Normal, Crackle, Wheeze, or Both."""

    has_crackle = _binary_flag(crackles)
    has_wheeze = _binary_flag(wheezes)
    if has_crackle and has_wheeze:
        return 3
    if has_crackle:
        return 1
    if has_wheeze:
        return 2
    return 0


class ICBHIDataset(Dataset):
    """
    Load respiratory cycles described by ``metadata.csv``.

    Each item is ``(waveform, class_label, device_label, patient_id)``.
    Waveforms are mono, resampled, and deterministically fixed to the target
    duration. Unknown recording devices are labelled ``-1``.
    """

    CACHE_VERSION = "fixed_cycle_v2"
    REQUIRED_COLUMNS = {
        "filepath",
        "onset",
        "offset",
        "crackles",
        "wheezes",
        "split",
    }

    def __init__(
        self,
        data_path: os.PathLike | str,
        split: Optional[str],
        metadatafile: os.PathLike | str = "metadata.csv",
        duration: float = 8.0,
        samplerate: int = 16000,
        fade_samples_ratio: int = 16,
        pad_type: str = "circular",
        cache: bool = True,
        cache_dir: Optional[os.PathLike | str] = None,
    ) -> None:
        super().__init__()

        self.data_path = Path(data_path).expanduser().resolve()
        metadata_path = Path(metadatafile).expanduser()
        if not metadata_path.is_absolute():
            metadata_path = self.data_path / metadata_path
        self.csv_path = metadata_path.resolve()

        if not self.csv_path.is_file():
            raise FileNotFoundError(f"找不到 metadata.csv: {self.csv_path}")

        self.split = None if split is None else str(split)
        self.duration = float(duration)
        self.samplerate = int(samplerate)
        self.target_samples = int(round(self.duration * self.samplerate))
        self.pad_type = str(pad_type)

        if self.duration <= 0 or self.target_samples <= 0:
            raise ValueError("duration 和 samplerate 必须为正数。")
        if fade_samples_ratio <= 0:
            raise ValueError("fade_samples_ratio 必须为正整数。")
        if self.pad_type not in {"circular", "zero"}:
            raise ValueError("pad_type 只能是 'circular' 或 'zero'。")

        self.fade_samples = max(1, self.samplerate // int(fade_samples_ratio))
        self.fade = T.Fade(
            fade_in_len=self.fade_samples,
            fade_out_len=self.fade_samples,
            fade_shape="linear",
        )
        self.fade_out = T.Fade(
            fade_in_len=0,
            fade_out_len=self.fade_samples,
            fade_shape="linear",
        )

        frame = pd.read_csv(self.csv_path)
        frame = frame.loc[
            :, ~frame.columns.astype(str).str.startswith("Unnamed")
        ].copy()
        missing = self.REQUIRED_COLUMNS.difference(frame.columns)
        if missing:
            raise ValueError(f"metadata.csv 缺少字段: {sorted(missing)}")

        if self.split is not None:
            frame = frame[frame["split"].astype(str) == self.split].copy()
        frame.reset_index(drop=True, inplace=True)
        if frame.empty:
            raise ValueError(f"metadata.csv 中没有 split={self.split!r} 的样本。")
        self.df = frame

        csv_digest = hashlib.sha256(self.csv_path.read_bytes()).hexdigest()[:12]
        split_tag = "all" if self.split is None else self.split
        duration_tag = f"{self.duration:g}".replace(".", "p")
        cache_root = (
            Path(cache_dir).expanduser().resolve()
            if cache_dir is not None
            else self.data_path
        )
        self.cache_path = cache_root / (
            f"icbhi_{split_tag}_sr{self.samplerate}_dur{duration_tag}_"
            f"{self.pad_type}_{self.CACHE_VERSION}_{csv_digest}.pth"
        )

        loaded = False
        if cache and self.cache_path.is_file():
            try:
                payload = safe_torch_load(self.cache_path)
                self._load_cache(payload)
                loaded = True
                print(f"📦 已加载缓存: {self.cache_path}")
            except (KeyError, RuntimeError, TypeError, ValueError, EOFError) as error:
                print(f"⚠️ 缓存无效，将重新生成: {error}")

        if not loaded:
            (
                self.data,
                self.labels,
                self.device_labels,
                self.patient_ids,
                self.sample_ids,
            ) = self._build_dataset()
            if cache:
                cache_root.mkdir(parents=True, exist_ok=True)
                torch.save(self._cache_payload(), self.cache_path)
                print(f"💾 已生成缓存: {self.cache_path}")

        print(
            f"📦 ICBHI [{split_tag}] | samples={len(self)} | "
            f"patients={len(set(self.patient_ids))}"
        )

    def _cache_payload(self) -> dict:
        return {
            "cache_version": self.CACHE_VERSION,
            "samplerate": self.samplerate,
            "target_samples": self.target_samples,
            "pad_type": self.pad_type,
            "data": self.data,
            "labels": self.labels,
            "device_labels": self.device_labels,
            "patient_ids": self.patient_ids,
            "sample_ids": self.sample_ids,
        }

    def _load_cache(self, payload: dict) -> None:
        if payload.get("cache_version") != self.CACHE_VERSION:
            raise ValueError("缓存版本不匹配。")
        if int(payload.get("samplerate", -1)) != self.samplerate:
            raise ValueError("缓存采样率不匹配。")
        if int(payload.get("target_samples", -1)) != self.target_samples:
            raise ValueError("缓存长度不匹配。")
        if payload.get("pad_type") != self.pad_type:
            raise ValueError("缓存补齐方式不匹配。")

        self.data = payload["data"].float().contiguous()
        self.labels = payload["labels"].long()
        self.device_labels = payload["device_labels"].long()
        self.patient_ids = [str(value) for value in payload["patient_ids"]]
        self.sample_ids = [str(value) for value in payload["sample_ids"]]

        sample_count = int(self.labels.numel())
        if self.data.shape != (sample_count, 1, self.target_samples):
            raise ValueError(f"缓存波形形状异常: {tuple(self.data.shape)}")
        if len(self.patient_ids) != sample_count or len(self.sample_ids) != sample_count:
            raise ValueError("缓存元数据长度不一致。")

    def _resolve_audio_path(self, filepath: str) -> Path:
        path = Path(filepath).expanduser()
        if not path.is_absolute():
            path = self.data_path / path
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(f"找不到音频文件: {path}")
        return path

    @staticmethod
    def _patient_id(row: pd.Series) -> str:
        if "patient" in row.index and not pd.isna(row["patient"]):
            return str(row["patient"])
        stem = Path(str(row["filepath"])).stem
        return stem.split("_", maxsplit=1)[0]

    @staticmethod
    def _device_label(row: pd.Series) -> int:
        candidates = []
        if "device" in row.index and not pd.isna(row["device"]):
            candidates.append(str(row["device"]))
        candidates.append(str(row["filepath"]))
        for text in candidates:
            for device_name, label in DEVICE_MAP.items():
                if device_name in text:
                    return label
        return -1

    def _load_cycle(self, row: pd.Series) -> torch.Tensor:
        filepath = self._resolve_audio_path(str(row["filepath"]))
        onset = float(row["onset"])
        offset = float(row["offset"])
        if not math.isfinite(onset) or not math.isfinite(offset) or offset <= onset:
            raise ValueError(
                f"无效呼吸周期: {filepath}, onset={onset}, offset={offset}"
            )

        sample_rate = int(torchaudio.info(str(filepath)).sample_rate)
        frame_offset = max(0, int(onset * sample_rate))
        num_frames = max(1, int((offset - onset) * sample_rate))
        waveform, loaded_rate = torchaudio.load(
            str(filepath),
            frame_offset=frame_offset,
            num_frames=num_frames,
        )
        if waveform.ndim != 2 or waveform.size(-1) == 0:
            raise RuntimeError(f"读取到空音频: {filepath}")
        if int(loaded_rate) != sample_rate:
            sample_rate = int(loaded_rate)
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != self.samplerate:
            waveform = torchaudio.functional.resample(
                waveform,
                orig_freq=sample_rate,
                new_freq=self.samplerate,
            )
        return self.fade(waveform.float())

    def _fix_length(self, waveform: torch.Tensor) -> torch.Tensor:
        length = int(waveform.size(-1))
        if length <= 0:
            raise RuntimeError("不能补齐空波形。")
        if length >= self.target_samples:
            return waveform[..., : self.target_samples].contiguous()

        if self.pad_type == "circular":
            repeat_count = math.ceil(self.target_samples / length)
            fixed = waveform.repeat(1, repeat_count)[..., : self.target_samples]
            return self.fade_out(fixed).contiguous()

        fixed = waveform.new_zeros((1, self.target_samples))
        fixed[..., :length] = waveform
        return fixed.contiguous()

    def _build_dataset(self):
        waveforms = []
        labels = []
        devices = []
        patients = []
        sample_ids = []

        for _, row in self.df.iterrows():
            waveform = self._fix_length(self._load_cycle(row))
            label = label_from_flags(row["crackles"], row["wheezes"])
            patient = self._patient_id(row)
            sample_id = (
                f"{row['filepath']}|{float(row['onset']):.3f}-"
                f"{float(row['offset']):.3f}"
            )

            waveforms.append(waveform)
            labels.append(label)
            devices.append(self._device_label(row))
            patients.append(patient)
            sample_ids.append(sample_id)

        if not waveforms:
            raise RuntimeError("数据集中没有可用音频。")

        return (
            torch.stack(waveforms, dim=0),
            torch.tensor(labels, dtype=torch.long),
            torch.tensor(devices, dtype=torch.long),
            patients,
            sample_ids,
        )

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, index: int):
        return (
            self.data[index],
            self.labels[index],
            self.device_labels[index],
            self.patient_ids[index],
        )


class ICBHIView(Dataset):
    """A zero-copy indexed view used for patient-disjoint train/validation splits."""

    def __init__(self, base: ICBHIDataset, indices: Sequence[int]) -> None:
        self.base = base
        self.indices = [int(index) for index in indices]
        self.labels = base.labels[self.indices]
        self.device_labels = base.device_labels[self.indices]
        self.patient_ids = [base.patient_ids[index] for index in self.indices]
        self.sample_ids = [base.sample_ids[index] for index in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        return self.base[self.indices[index]]