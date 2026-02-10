"""
Configuration Loader - Load and provide access to config.yaml

Usage:
    from config_loader import config

    # Access configuration
    url = config.api.casca_url
    fps_min = config.slo_targets.fps.min
    learning_rate = config.ppo.learning_rate
"""

import os
import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional


# Configuration file path (relative to this file's location)
CONFIG_DIR = Path(__file__).parent
DEFAULT_CONFIG_FILE = CONFIG_DIR / "config-redacted.yaml"


@dataclass
class APIConfig:
    casca_url: str = "http://localhost:8080/v0"
    emma_url: str = "http://localhost:8081"
    timeout: int = 10


@dataclass
class EnvironmentConfig:
    wait_seconds: int = 25
    max_episode_steps: int = 144
    slos: List[str] = field(default_factory=lambda: ["meanFPS", "meanPower"])
    config_params: List[str] = field(default_factory=lambda: ["EncodingThreadCount"])


@dataclass
class FPSTargets:
    min: int = 20
    max: int = 31


@dataclass
class SLOTargets:
    fps: FPSTargets = field(default_factory=FPSTargets)


@dataclass
class EncodingThreadCountRange:
    min: int = 0
    max: int = 16


@dataclass
class ActionSpace:
    encoding_thread_count: EncodingThreadCountRange = field(default_factory=EncodingThreadCountRange)


@dataclass
class PPOConfig:
    learning_rate: float = 0.0003
    n_steps: int = 144
    batch_size: int = 36
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5


@dataclass
class TrainingConfig:
    total_timesteps: int = 2880
    checkpoint_freq: int = 60
    save_dir: str = "./models"
    log_dir: str = "./logs"
    tensorboard_dir: str = "./tensorboard_logs"


@dataclass
class CarbonConfig:
    csv_file: str = "carbon_intensity_austria_2024_hourly.csv"
    default_value: float = 300.0


@dataclass
class RobustnessConfig:
    max_retries: int = 3
    retry_delay: float = 10.0


@dataclass
class CausalityTestConfig:
    sample_interval: int = 25
    phase_duration: int = 600
    action_low: int = 2
    action_high: int = 16


@dataclass
class Config:
    """Main configuration class - contains all config items"""
    api: APIConfig = field(default_factory=APIConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    slo_targets: SLOTargets = field(default_factory=SLOTargets)
    action_space: ActionSpace = field(default_factory=ActionSpace)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    carbon: CarbonConfig = field(default_factory=CarbonConfig)
    robustness: RobustnessConfig = field(default_factory=RobustnessConfig)
    causality_test: CausalityTestConfig = field(default_factory=CausalityTestConfig)


def _dict_to_dataclass(data: dict, cls):
    """Recursively convert dictionary to dataclass"""
    if not isinstance(data, dict):
        return data

    # Get dataclass fields
    field_types = {f.name: f.type for f in cls.__dataclass_fields__.values()}

    kwargs = {}
    for key, value in data.items():
        if key in field_types:
            field_type = field_types[key]
            # Check if it's a nested dataclass
            if hasattr(field_type, '__dataclass_fields__'):
                kwargs[key] = _dict_to_dataclass(value, field_type)
            else:
                kwargs[key] = value

    return cls(**kwargs)


def load_config(config_file: Optional[str] = None) -> Config:
    """
    Load configuration file

    Args:
        config_file: Configuration file path, defaults to config.yaml

    Returns:
        Config object
    """
    if config_file is None:
        config_file = DEFAULT_CONFIG_FILE

    config_path = Path(config_file)

    if not config_path.exists():
        print(f"Warning: Config file not found: {config_path}")
        print("Using default configuration")
        return Config()

    with open(config_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    if data is None:
        return Config()

    # Build configuration object
    config = Config()

    if 'api' in data:
        config.api = _dict_to_dataclass(data['api'], APIConfig)

    if 'environment' in data:
        config.environment = _dict_to_dataclass(data['environment'], EnvironmentConfig)

    if 'slo_targets' in data:
        slo_data = data['slo_targets']
        if 'fps' in slo_data:
            fps_targets = FPSTargets(**slo_data['fps'])
        else:
            fps_targets = FPSTargets()
        config.slo_targets = SLOTargets(fps=fps_targets)

    if 'action_space' in data:
        action_data = data['action_space']
        if 'encoding_thread_count' in action_data:
            etc_range = EncodingThreadCountRange(**action_data['encoding_thread_count'])
        else:
            etc_range = EncodingThreadCountRange()
        config.action_space = ActionSpace(encoding_thread_count=etc_range)

    if 'ppo' in data:
        config.ppo = _dict_to_dataclass(data['ppo'], PPOConfig)

    if 'training' in data:
        config.training = _dict_to_dataclass(data['training'], TrainingConfig)

    if 'carbon' in data:
        config.carbon = _dict_to_dataclass(data['carbon'], CarbonConfig)

    if 'robustness' in data:
        config.robustness = _dict_to_dataclass(data['robustness'], RobustnessConfig)

    if 'causality_test' in data:
        config.causality_test = _dict_to_dataclass(data['causality_test'], CausalityTestConfig)

    return config


# Global configuration instance (lazy loading)
_config: Optional[Config] = None


def get_config() -> Config:
    """Get global configuration instance"""
    global _config
    if _config is None:
        _config = load_config()
    return _config


def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload configuration"""
    global _config
    _config = load_config(config_file)
    return _config


# Convenient access
config = get_config()


if __name__ == "__main__":
    # Test configuration loading
    print("=" * 60)
    print("Configuration Loading Test")
    print("=" * 60)

    cfg = load_config()

    print(f"\nAPI Configuration:")
    print(f"  CASCA URL: {cfg.api.casca_url}")
    print(f"  Timeout: {cfg.api.timeout}s")

    print(f"\nSLO Targets:")
    print(f"  FPS range: {cfg.slo_targets.fps.min} - {cfg.slo_targets.fps.max}")

    print(f"\nAction Space:")
    print(f"  EncodingThreadCount: {cfg.action_space.encoding_thread_count.min} - {cfg.action_space.encoding_thread_count.max}")

    print(f"\nPPO Hyperparameters:")
    print(f"  Learning rate: {cfg.ppo.learning_rate}")
    print(f"  n_steps: {cfg.ppo.n_steps}")
    print(f"  batch_size: {cfg.ppo.batch_size}")
    print(f"  gamma: {cfg.ppo.gamma}")

    print(f"\nTraining Configuration:")
    print(f"  Total timesteps: {cfg.training.total_timesteps}")
    print(f"  Checkpoint frequency: {cfg.training.checkpoint_freq}")

    print(f"\nCausality Test Configuration:")
    print(f"  Action low: {cfg.causality_test.action_low}")
    print(f"  Action high: {cfg.causality_test.action_high}")

    print("\n" + "=" * 60)
    print("Configuration loaded successfully!")
    print("=" * 60)
