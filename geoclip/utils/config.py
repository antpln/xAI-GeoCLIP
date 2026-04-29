import yaml
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ModelConfig:
    clip_backbone: str = "ViT-B/16"
    freeze_layers: int = 9
    rff_num_scales: int = 10
    rff_dim: int = 256
    mlp_hidden: int = 1024
    embedding_dim: int = 512


@dataclass
class DataConfig:
    mode: str = "hf"              # hf | subset | streaming | sharded | local
    subset_size: Optional[int] = 50000
    num_shards: Optional[int] = None
    streaming: bool = False
    num_workers: int = 4
    pin_memory: bool = True
    local_files_only: bool = False
    # local mode
    zip_dir: Optional[str] = None
    train_csv: Optional[str] = None
    val_csv: Optional[str] = None
    val_zip_dir: Optional[str] = None


@dataclass
class GalleryConfig:
    size: int = 10000
    strategy: str = "train_sample"
    cache_path: str = "gallery.pt"


@dataclass
class TrainingConfig:
    pretrained_weights_dir: Optional[str] = None  # path to pre_trained_weights folder; None = skip
    batch_size: int = 128
    epochs: int = 30
    warmup_epochs: int = 2
    lr_clip: float = 1e-5   # low LR for the pretrained ViT backbone
    lr_gps: float = 1e-4    # higher LR for the GPS encoder (trained from scratch)
    lr_temp: float = 1e-3   # learnable temperature converges quickly
    weight_decay: float = 0.1
    amp: bool = True
    log_every: int = 50
    eval_every: int = 1
    checkpoint_dir: str = "checkpoints/"
    # Hard geographic negative mining
    hard_neg_swap_prob: float = 0.5
    hard_neg_min_dist_km: float = 500.0
    # Attention entropy regularization (run every N steps to limit memory overhead)
    lambda_attn: float = 0.01
    attn_reg_every: int = 4


@dataclass
class EvalConfig:
    thresholds_km: List[int] = field(default_factory=lambda: [1, 25, 200, 750, 2500])


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    gallery: GalleryConfig = field(default_factory=GalleryConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvalConfig = field(default_factory=EvalConfig)


def load_config(path: str) -> Config:
    """Load a YAML config file. Missing sections fall back to dataclass defaults."""
    with open(path) as f:
        raw = yaml.safe_load(f)

    cfg = Config()
    if "model" in raw:
        cfg.model = ModelConfig(**raw["model"])
    if "data" in raw:
        cfg.data = DataConfig(**raw["data"])
    if "gallery" in raw:
        cfg.gallery = GalleryConfig(**raw["gallery"])
    if "training" in raw:
        cfg.training = TrainingConfig(**raw["training"])
    if "evaluation" in raw:
        cfg.evaluation = EvalConfig(**raw["evaluation"])
    return cfg
