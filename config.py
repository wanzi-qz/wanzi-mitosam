"""Configuration objects.

Every entry point of MitoSAM is driven by the same dataclasses, so a YAML file
or CLI flags produce exactly the same run as the Python API.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = ROOT / "configs" / "default.yaml"


@dataclass
class DataConfig:
    """Where images and masks live and how they are fed to the network."""

    image_dir: str = "data/sample/images"
    mask_dir: str = "data/sample/masks"
    patch_size: int = 512
    patches_per_image: int = 16
    val_split: float = 0.2
    seed: int = 42
    augment: bool = True


@dataclass
class ModelConfig:
    """MitoSAM = frozen SAM ViT backbone + ResNet-CBAM adapter + mask decoder."""

    sam_checkpoint: str | None = "weights/sam_vit_b_01ec64.pth"
    sam_type: str = "vit_b"          # vit_b (fast) | vit_l | vit_h (thesis setting)
    adapter_channels: int = 256      # SAM ViT-B/H image embedding width
    use_cbam: bool = True
    use_adapter: bool = True
    train_decoder: bool = True
    train_prompt_encoder: bool = False
    input_size: int = 1024


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 1
    lr: float = 1e-4
    weight_decay: float = 1e-4
    prompt_points: int = 10
    amp: bool = True
    grad_clip: float = 1.0
    num_workers: int = 0
    checkpoint_dir: str = "weights"
    device: str = "auto"


@dataclass
class FeatureConfig:
    """Feature families extracted per cell / field of view."""

    min_area: int = 50
    max_area: int = 12_000
    pixel_size_um: float = 0.1
    texture_bins: int = 64
    lbp_points: int = 24
    lbp_radius: int = 3
    ripley_max_radius_px: float = 80.0
    ripley_steps: int = 16
    families: list[str] = field(default_factory=lambda: ["morphology", "texture", "spatial"])


@dataclass
class EvalConfig:
    n_features: int = 30
    n_splits: int = 5
    n_repeats: int = 10
    random_state: int = 42
    models: list[str] = field(
        default_factory=lambda: ["logreg", "random_forest", "xgboost", "lightgbm"]
    )
    feature_subsets: list[str] = field(
        default_factory=lambda: [
            "morphology_only", "texture_only", "spatial_only",
            "morphology_texture", "texture_spatial", "morphology_spatial",
            "full_features",
        ]
    )
    shap_samples: int = 200


@dataclass
class MitoSAMConfig:
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "MitoSAMConfig":
        """Load config from YAML, falling back to defaults for missing keys."""
        cfg = cls()
        if path is None or not Path(path).exists():
            return cfg
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for section, values in raw.items():
            target = getattr(cfg, section, None)
            if target is None or not isinstance(values, dict):
                continue
            for key, value in values.items():
                if hasattr(target, key):
                    setattr(target, key, value)
        return cfg

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(self.to_dict(), sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path
