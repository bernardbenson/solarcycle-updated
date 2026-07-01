"""
Configuration management for solar cycle prediction models.
Uses Pydantic for validation and YAML for human-readable configs.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from pydantic import BaseModel, Field, validator
from enum import Enum


class AttentionType(str, Enum):
    SCALED_DOT = "scaled_dot"
    BAHDANAU = "bahdanau"


class HeadType(str, Enum):
    MSE = "mse"
    QUANTILE = "quantile"
    COMBINED = "combined"


class NormalizationMethod(str, Enum):
    STANDARD = "standard"
    ROBUST = "robust"
    NONE = "none"


class TransformMethod(str, Enum):
    IDENTITY = "identity"
    LOG1P = "log1p"
    SQRT = "sqrt"


class OptimizerType(str, Enum):
    ADAM = "adam"
    ADAMW = "adamw"
    SGD = "sgd"


class SchedulerType(str, Enum):
    COSINE_WITH_WARMUP = "cosine_with_warmup"
    WARMUP_COSINE = "warmup_cosine"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"
    STEP = "step"
    NONE = "none"


class WaveNetConfig(BaseModel):
    """WaveNet encoder configuration."""
    stacks: int = Field(default=3, ge=1, le=10, description="Number of WaveNet stacks")
    layers_per_stack: int = Field(default=4, ge=1, le=10, description="Layers per stack")
    kernel_size: int = Field(default=3, ge=2, le=7, description="Convolution kernel size")
    base_dilation: int = Field(default=1, ge=1, description="Base dilation rate")
    channels: int = Field(default=128, ge=32, le=512, description="Number of channels")
    dropout: float = Field(default=0.2, ge=0.0, le=0.8, description="Dropout rate")


class NormalizationConfig(BaseModel):
    """Data normalization configuration."""
    method: NormalizationMethod = Field(default=NormalizationMethod.ROBUST)
    transform: TransformMethod = Field(default=TransformMethod.SQRT)
    quantile_range: tuple = Field(default=(25.0, 75.0), description="IQR range for robust scaling")
    
    @validator('quantile_range')
    def validate_quantile_range(cls, v):
        if len(v) != 2 or v[0] >= v[1] or v[0] < 0 or v[1] > 100:
            raise ValueError("quantile_range must be (lower, upper) with 0 <= lower < upper <= 100")
        return v


class ModelConfig(BaseModel):
    """Model architecture configuration."""
    name: str = Field(default="WaveNetAttnSeq2Seq", description="Model name")
    input_dim: int = Field(default=1, ge=1, description="Input feature dimension")
    d_model: int = Field(default=128, ge=32, le=512, description="Model hidden dimension")
    output_size: int = Field(default=132, ge=1, description="Output sequence length")
    
    # WaveNet encoder
    wavenet: WaveNetConfig = Field(default_factory=WaveNetConfig)
    
    # Encoder LSTM
    encoder_bilstm_hidden: int = Field(default=128, ge=32, le=512, description="Encoder BiLSTM hidden size")
    
    # Decoder (non-autoregressive attention decoder)
    decoder_lstm_hidden: int = Field(default=128, ge=32, le=512, description="Decoder hidden size (legacy)")
    decoder_layers: int = Field(default=2, ge=1, le=8, description="Transformer decoder layers")
    decoder_heads: int = Field(default=4, ge=1, le=16, description="Decoder attention heads")
    attention: AttentionType = Field(default=AttentionType.SCALED_DOT)
    
    # Precursor conditioning dimension (0 = no conditioning). Derived from
    # data.use_terminator by resolve_derived_dims(); do not set by hand.
    cond_dim: int = Field(default=0, ge=0, description="Precursor conditioning vector size")

    # Prediction head
    head: HeadType = Field(default=HeadType.QUANTILE)
    quantiles: List[float] = Field(default=[0.1, 0.5, 0.9], description="Quantile levels")
    
    # Regularization
    dropout: float = Field(default=0.2, ge=0.0, le=0.8, description="Global dropout rate")
    
    # MC-Dropout
    mc_dropout_samples: int = Field(default=30, ge=1, le=100, description="MC-Dropout samples")
    
    @validator('quantiles')
    def validate_quantiles(cls, v):
        if not all(0 < q < 1 for q in v):
            raise ValueError("All quantiles must be between 0 and 1")
        if len(set(v)) != len(v):
            raise ValueError("Quantiles must be unique")
        return sorted(v)


class TrainingConfig(BaseModel):
    """Training configuration."""
    epochs: int = Field(default=200, ge=1, le=1000, description="Maximum training epochs")
    batch_size: int = Field(default=32, ge=1, le=256, description="Batch size")
    
    # Teacher forcing
    teacher_forcing: float = Field(default=0.5, ge=0.0, le=1.0, description="Initial teacher forcing ratio")
    teacher_forcing_decay: float = Field(default=0.95, ge=0.0, le=1.0, description="Teacher forcing decay")
    teacher_forcing_min: float = Field(default=0.1, ge=0.0, le=1.0, description="Minimum teacher forcing ratio")
    
    # Optimization
    optimizer: OptimizerType = Field(default=OptimizerType.ADAMW)
    lr: float = Field(default=1e-3, gt=0, le=1.0, description="Learning rate")
    weight_decay: float = Field(default=1e-4, ge=0.0, le=1.0, description="Weight decay")
    
    # Scheduling
    scheduler: SchedulerType = Field(default=SchedulerType.COSINE_WITH_WARMUP)
    warmup_epochs: int = Field(default=5, ge=0, description="Warmup epochs for cosine scheduler")
    scheduler_patience: int = Field(default=5, ge=1, description="Patience for ReduceLROnPlateau")
    scheduler_factor: float = Field(default=0.5, gt=0.0, le=1.0, description="LR reduction factor")
    
    # Early stopping
    early_stop_patience: int = Field(default=20, ge=1, description="Early stopping patience")
    
    # Regularization
    grad_clip_norm: float = Field(default=1.0, gt=0.0, description="Gradient clipping norm")
    
    # Performance
    amp: bool = Field(default=False, description="Use automatic mixed precision (CUDA only)")

    # Weight EMA (exponential moving average) for smoother validation curves
    use_ema: bool = Field(default=False, description="Track an EMA of weights for eval/checkpoint")
    ema_decay: float = Field(default=0.999, ge=0.5, le=0.9999, description="EMA decay rate")

    # Validation
    val_ratio: float = Field(default=0.2, gt=0.0, lt=1.0, description="Validation set ratio")


class CVConfig(BaseModel):
    """Cross-validation configuration."""
    enabled: bool = Field(default=True, description="Enable cross-validation")
    n_folds: int = Field(default=5, ge=2, le=10, description="Number of CV folds")
    test_size: int = Field(default=132, ge=1, description="Test set size per fold")
    min_train_size: int = Field(default=1000, ge=100, description="Minimum training size")
    gap_size: int = Field(default=0, ge=0, description="Gap between train and test")
    method: str = Field(default="rolling", description="CV method: 'rolling' or 'blocked'")


class DataConfig(BaseModel):
    """Data configuration."""
    target_col: str = Field(default="sunspot_number", description="Target column name")
    start_year: int = Field(default=1749, ge=1700, le=2100, description="Starting year for data")
    input_window: int = Field(default=528, ge=100, description="Input window size (months)")
    prediction_horizon: int = Field(default=132, ge=1, description="Prediction horizon (months)")
    
    # Features
    add_features: bool = Field(default=True, description="Add engineered features")
    feature_window: int = Field(default=13, ge=1, description="Window for feature engineering")

    # Precursors (exogenous input channels + terminator conditioning).
    # Empty precursor_cols => univariate (backward compatible).
    precursor_cols: List[str] = Field(default=[], description="Exogenous columns fed as extra input channels")
    geomag_mask: bool = Field(default=True, description="Add a binary availability-mask channel for precursors")
    use_terminator: bool = Field(default=False, description="Condition on the terminator/cycle-length precursor")

    # Normalization
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class ExperimentConfig(BaseModel):
    """Complete experiment configuration."""
    experiment_name: str = Field(default="wavenet_seq2seq", description="Experiment name")
    seed: int = Field(default=42, ge=0, description="Random seed")
    device: str = Field(default="auto", description="Device: 'auto', 'cpu', 'cuda'")
    
    # Components
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    cv: CVConfig = Field(default_factory=CVConfig)
    
    # Output
    output_dir: str = Field(default="data/experiments", description="Output directory")
    save_predictions: bool = Field(default=True, description="Save prediction arrays")
    save_model: bool = Field(default=True, description="Save trained model")
    
    # Logging
    log_interval: int = Field(default=10, ge=1, description="Logging interval (epochs)")
    plot_training: bool = Field(default=True, description="Generate training plots")
    plot_predictions: bool = Field(default=True, description="Generate prediction plots")


def resolve_derived_dims(config: ExperimentConfig) -> ExperimentConfig:
    """Derive model input_dim / cond_dim from the data config.

    Keeps train() and load_trained() in agreement on tensor shapes:
    - input_dim = sunspot (1) + one channel per precursor column + an optional
      availability-mask channel.
    - cond_dim  = 1 when the terminator/cycle-length precursor is enabled, else 0.
    """
    n_precursors = len(config.data.precursor_cols)
    mask_channel = 1 if (config.data.geomag_mask and n_precursors > 0) else 0
    config.model.input_dim = 1 + n_precursors + mask_channel
    config.model.cond_dim = 1 if config.data.use_terminator else 0
    return config


def load_config(config_path: Union[str, Path]) -> ExperimentConfig:
    """Load configuration from YAML file."""
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)

    return resolve_derived_dims(ExperimentConfig(**config_dict))


def save_config(config: ExperimentConfig, config_path: Union[str, Path]) -> None:
    """Save configuration to YAML file."""
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_path, 'w') as f:
        yaml.dump(config.dict(), f, default_flow_style=False, indent=2)


def create_default_configs():
    """Create default configuration files."""
    configs_dir = Path("solar/configs")
    configs_dir.mkdir(parents=True, exist_ok=True)
    
    # Base configuration
    base_config = ExperimentConfig()
    save_config(base_config, configs_dir / "base.yaml")
    
    # Seq2Seq quantile configuration
    seq2seq_config = ExperimentConfig(
        experiment_name="seq2seq_quantile",
        model=ModelConfig(
            name="WaveNetAttnSeq2Seq",
            head=HeadType.QUANTILE,
            quantiles=[0.1, 0.5, 0.9],
            wavenet=WaveNetConfig(stacks=3, layers_per_stack=4, channels=128, dropout=0.2),
            d_model=128,
            encoder_bilstm_hidden=128,
            decoder_lstm_hidden=128,
            attention=AttentionType.SCALED_DOT
        ),
        training=TrainingConfig(
            epochs=200,
            batch_size=32,
            teacher_forcing=0.6,
            optimizer=OptimizerType.ADAMW,
            lr=1e-3,
            scheduler=SchedulerType.COSINE_WITH_WARMUP,
            warmup_epochs=5,
            early_stop_patience=20
        )
    )
    save_config(seq2seq_config, configs_dir / "seq2seq_quantile.yaml")
    
    # TCN ablation configuration
    tcn_config = ExperimentConfig(
        experiment_name="ablation_tcn",
        model=ModelConfig(
            name="TCNOnly",
            head=HeadType.MSE,
            d_model=128
        ),
        training=TrainingConfig(
            epochs=150,
            batch_size=32,
            teacher_forcing=0.0,  # Not applicable for TCN
            optimizer=OptimizerType.ADAMW,
            lr=1e-3,
            scheduler=SchedulerType.REDUCE_ON_PLATEAU
        )
    )
    save_config(tcn_config, configs_dir / "ablation_tcn.yaml")
    
    print(f"Created default configs in {configs_dir}")
    return configs_dir


def validate_config(config: ExperimentConfig) -> List[str]:
    """Validate configuration and return list of warnings/errors."""
    warnings = []
    
    # Check model-training compatibility
    if config.model.head == HeadType.MSE and config.training.teacher_forcing > 0:
        warnings.append("Teacher forcing is typically used with sequence generation, not MSE head")
    
    # Check data consistency
    if config.data.input_window < config.data.prediction_horizon:
        warnings.append("Input window smaller than prediction horizon may limit performance")
    
    # Check CV settings
    if config.cv.enabled and config.cv.test_size > config.data.input_window // 2:
        warnings.append("CV test size is large relative to input window")
    
    # Check quantiles for non-quantile heads
    if config.model.head == HeadType.MSE and len(config.model.quantiles) > 1:
        warnings.append("Multiple quantiles specified for MSE head")
    
    return warnings


if __name__ == "__main__":
    # Test configuration system
    print("Testing configuration system...")
    
    # Create default configurations
    configs_dir = create_default_configs()
    
    # Test loading
    base_config = load_config(configs_dir / "base.yaml")
    print(f"Loaded base config: {base_config.experiment_name}")
    
    seq2seq_config = load_config(configs_dir / "seq2seq_quantile.yaml")
    print(f"Loaded seq2seq config: {seq2seq_config.experiment_name}")
    
    # Test validation
    warnings = validate_config(seq2seq_config)
    if warnings:
        print("Config warnings:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("No config warnings found")
    
    # Test creating custom config
    custom_config = ExperimentConfig(
        experiment_name="custom_test",
        model=ModelConfig(
            head=HeadType.COMBINED,
            quantiles=[0.05, 0.5, 0.95],
            wavenet=WaveNetConfig(stacks=2, channels=64)
        ),
        training=TrainingConfig(
            epochs=100,
            lr=2e-3,
            batch_size=16
        )
    )
    
    save_config(custom_config, configs_dir / "custom_test.yaml")
    print("Created custom config")
    
    # Validate custom config
    custom_warnings = validate_config(custom_config)
    print(f"Custom config validation: {len(custom_warnings)} warnings")
    
    print("\n✅ Configuration system implemented and tested!")