"""
PPO Training Script - Based on ServiceGym with Discrete Action Space

All configuration parameters are centrally managed in config.yaml
"""
import os
import sys
import time
import numpy as np
import torch
import requests
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add ServiceGym to path (relative to script directory)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(SCRIPT_DIR, 'ServiceGym'))
from gym_service import ServiceEnv, ServiceReward, FinishingCondition

# Import robust wrapper
from robust_service_env import RobustServiceEnvWrapper

# Import discrete action wrapper
from discrete_action_wrapper import DiscreteActionWrapper

# Import carbon cache
from carbon_cache_hourly import CarbonIntensityCacheHourly

# Import config loader
from config_loader import get_config

# Load configuration
cfg = get_config()


class ServiceAPIConfig:
    """Service API Configuration - loaded from config.yaml"""
    CASCA_URL = cfg.api.casca_url
    EMMA_URL = cfg.api.emma_url
    WAIT_SECONDS = cfg.environment.wait_seconds

    # SLOs configuration
    SLOS = cfg.environment.slos

    # Configurable parameters
    CONFIG_PARAMS = cfg.environment.config_params

class CustomReward(ServiceReward):
    """
    Custom Reward Function - Using Historical Carbon Data

    Changes:
    - No longer fetches carbon intensity from EMMA API in real-time
    - Uses pre-loaded 2024 Austria historical data (hourly granularity)
    - Automatically maps current time to same period last year
    - No interpolation, uses actual hourly values
    """
    def __init__(self, carbon_cache, target_fps_min=None, target_fps_max=None):
        """
        Initialize reward function

        Args:
            carbon_cache: CarbonIntensityCacheHourly instance
            target_fps_min: FPS lower bound (from config.yaml, default 20)
            target_fps_max: FPS upper bound (from config.yaml, default 31)
        """
        self.carbon_cache = carbon_cache
        self.target_fps_min = target_fps_min if target_fps_min is not None else cfg.slo_targets.fps.min
        self.target_fps_max = target_fps_max if target_fps_max is not None else cfg.slo_targets.fps.max

    def get_carbon_intensity(self):
        """
        Get carbon intensity from cache

        Returns:
            float: carbon intensity (gCO2eq/kWh)
        """
        current_time = datetime.now()
        return self.carbon_cache.get_carbon_intensity(current_time)

    def calculate_carbon_footprint(self, mean_power):
        """
        Calculate carbon footprint (unit: grams gCO2eq)

        Formula: carbon_footprint_g = (meanPower / 1000) * (wait_seconds/3600) * carbon_intensity

        Note: Although meanPower is the average over the past 5 minutes,
              we calculate the carbon footprint for the current step (wait_seconds)

        Args:
            mean_power: Average power (Watts, from past 5 minutes average)

        Returns:
            float: Carbon footprint (gCO2eq, estimated for current step)
        """
        carbon_intensity = self.get_carbon_intensity()
        time_period_hours = cfg.environment.wait_seconds / 3600.0
        energy_kwh = (mean_power / 1000.0) * time_period_hours  # kWh
        carbon_footprint_g = energy_kwh * carbon_intensity  # gCO2eq

        return carbon_footprint_g

    def calculate_reward(self, last_obs, current_obs):
        """
        Calculate reward

        Reward function:
        - Within SLO (fps_min <= fps <= fps_max): reward = 1 / carbon
        - SLO violation: reward = -carbon

        Args:
            last_obs: Previous observation (unused)
            current_obs: Current observation [meanFPS, meanPower]

        Returns:
            float: Reward value
        """
        fps = current_obs[0]  # meanFPS
        mean_power = current_obs[1]  # meanPower

        # Calculate carbon footprint (in grams)
        carbon = self.calculate_carbon_footprint(mean_power)

        # Prevent division by zero
        carbon = max(carbon, 0.001)

        # Apply reward function
        if self.target_fps_min <= fps <= self.target_fps_max:
            reward = 1.0 / carbon  # Within SLO: lower carbon = higher reward
        else:
            reward = -carbon  # SLO violation: higher carbon = larger penalty

        return reward

class TrainingCallback(BaseCallback):
    """Training Monitoring Callback - with detailed logging"""
    def __init__(self, save_path, carbon_cache=None, verbose=1):
        super().__init__(verbose)
        self.save_path = save_path
        self.carbon_cache = carbon_cache  # Carbon cache for real carbon intensity
        self.episode_rewards = []
        self.episode_lengths = []

        # Detailed log files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.step_log_file = f"{save_path}/step_log_{timestamp}.csv"
        self.episode_log_file = f"{save_path}/episode_log_{timestamp}.csv"

        # Create CSV files with headers
        import csv
        with open(self.step_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "episode", "step", "global_step",
                "meanFPS", "meanPower", "carbon_footprint_g", "carbon_intensity",
                "action_EncodingThreadCount", "reward",
                "in_slo_range", "terminated", "truncated"
            ])

        with open(self.episode_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp", "episode", "total_steps", "episode_reward",
                "mean_fps", "mean_power", "mean_carbon", "mean_reward",
                "slo_satisfaction_rate", "duration_minutes"
            ])

        # Episode statistics
        self.current_episode = 0
        self.episode_start_time = time.time()
        self.episode_fps_list = []
        self.episode_power_list = []
        self.episode_carbon_list = []
        self.episode_reward_list = []
        self.episode_in_slo_list = []

    def _on_step(self) -> bool:
        # Get current information
        obs = self.locals.get("obs_tensor", self.locals.get("new_obs"))
        actions = self.locals.get("actions")
        rewards = self.locals.get("rewards")
        dones = self.locals.get("dones")
        infos = self.locals.get("infos", [{}])

        # Log detailed information for each step
        if obs is not None and actions is not None:
            import csv

            # Extract observation values
            if hasattr(obs, 'cpu'):
                obs_np = obs.cpu().numpy()[0]
            else:
                obs_np = np.array(obs[0]) if isinstance(obs, list) else obs

            mean_fps = float(obs_np[0])
            mean_power = float(obs_np[1])

            # Calculate carbon footprint
            reward_value = float(rewards[0]) if rewards is not None else 0.0

            # Estimate carbon from reward
            in_slo = cfg.slo_targets.fps.min <= mean_fps <= cfg.slo_targets.fps.max
            if in_slo and reward_value > 0:
                carbon_footprint_g = 1.0 / reward_value
            elif not in_slo and reward_value < 0:
                carbon_footprint_g = -reward_value
            else:
                # Direct calculation using wait_seconds
                carbon_intensity = self.carbon_cache.get_carbon_intensity() if self.carbon_cache else cfg.carbon.default_value
                carbon_footprint_g = (mean_power / 1000.0) * (cfg.environment.wait_seconds/3600.0) * carbon_intensity

            # Extract action
            action_value = int(actions[0]) if actions is not None else 0

            # Get termination info
            done = bool(dones[0]) if dones is not None else False
            info = infos[0] if infos else {}
            terminated = info.get("TimeLimit.truncated", False) == False and done
            truncated = info.get("TimeLimit.truncated", False)

            # Get real carbon intensity
            current_carbon_intensity = self.carbon_cache.get_carbon_intensity() if self.carbon_cache else cfg.carbon.default_value

            # Write to step log
            with open(self.step_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.current_episode,
                    self.num_timesteps % cfg.environment.max_episode_steps,
                    self.num_timesteps,
                    f"{mean_fps:.2f}",
                    f"{mean_power:.2f}",
                    f"{carbon_footprint_g:.4f}",
                    f"{current_carbon_intensity:.2f}",
                    action_value,
                    f"{reward_value:.6f}",
                    "Yes" if in_slo else "No",
                    "Yes" if terminated else "No",
                    "Yes" if truncated else "No"
                ])

            # Accumulate episode statistics
            self.episode_fps_list.append(mean_fps)
            self.episode_power_list.append(mean_power)
            self.episode_carbon_list.append(carbon_footprint_g)
            self.episode_reward_list.append(reward_value)
            self.episode_in_slo_list.append(1 if in_slo else 0)

        # Episode end statistics
        if dones is not None and dones[0]:
            episode_duration = (time.time() - self.episode_start_time) / 60.0  # minutes

            # Calculate statistics
            episode_reward = sum(self.episode_reward_list)
            total_steps = len(self.episode_reward_list)
            mean_fps = np.mean(self.episode_fps_list) if self.episode_fps_list else 0
            mean_power = np.mean(self.episode_power_list) if self.episode_power_list else 0
            mean_carbon = np.mean(self.episode_carbon_list) if self.episode_carbon_list else 0
            mean_reward = np.mean(self.episode_reward_list) if self.episode_reward_list else 0
            slo_rate = np.mean(self.episode_in_slo_list) * 100 if self.episode_in_slo_list else 0

            # Write to episode log
            import csv
            with open(self.episode_log_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    datetime.now().isoformat(),
                    self.current_episode,
                    total_steps,
                    f"{episode_reward:.4f}",
                    f"{mean_fps:.2f}",
                    f"{mean_power:.2f}",
                    f"{mean_carbon:.4f}",
                    f"{mean_reward:.6f}",
                    f"{slo_rate:.1f}%",
                    f"{episode_duration:.1f}"
                ])

            # Console output
            print(f"\n{'='*80}")
            print(f"Episode {self.current_episode} completed:")
            print(f"  Total steps: {total_steps}")
            print(f"  Episode reward: {episode_reward:.4f}")
            print(f"  Mean FPS: {mean_fps:.2f} (target: {cfg.slo_targets.fps.min}-{cfg.slo_targets.fps.max})")
            print(f"  Mean power: {mean_power:.2f} W")
            print(f"  Mean carbon: {mean_carbon:.4f} gCO2eq")
            print(f"  Mean reward: {mean_reward:.6f}")
            print(f"  SLO satisfaction rate: {slo_rate:.1f}%")
            print(f"  Duration: {episode_duration:.1f} minutes")
            print(f"{'='*80}\n")

            # Reset episode statistics
            self.current_episode += 1
            self.episode_start_time = time.time()
            self.episode_fps_list = []
            self.episode_power_list = []
            self.episode_carbon_list = []
            self.episode_reward_list = []
            self.episode_in_slo_list = []

            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(total_steps)

        return True

def test_service_api_connection(url):
    """Pre-training smoke test - verify Service API connection"""
    print("\n[Smoke Test] Running smoke test...")
    print(f"Checking Service API: {url}")

    try:
        # Test SLOs endpoint
        print("  Testing /slos endpoint...", end=" ")
        resp = requests.get(f"{url}/slos", timeout=5)
        resp.raise_for_status()
        slos = resp.json()
        print(f"OK (found {len(slos)} SLOs: {slos})")

        # Test config endpoint
        print("  Testing /config endpoint...", end=" ")
        resp = requests.get(f"{url}/config", timeout=5)
        resp.raise_for_status()
        config = resp.json()
        print(f"OK (found {len(config)} config params: {config})")

        # Verify required SLOs exist
        print("  Verifying required SLOs...", end=" ")
        slo_ids = [s["id"] for s in slos] if isinstance(slos, list) else list(slos.keys())
        for slo in ServiceAPIConfig.SLOS:
            if slo not in slo_ids:
                print(f"FAILED - missing SLO: {slo}")
                return False
        print("OK")

        # Verify required config params exist
        print("  Verifying required config params...", end=" ")
        param_ids = [p["id"] for p in config] if isinstance(config, list) else list(config.keys())
        for param in ServiceAPIConfig.CONFIG_PARAMS:
            if param not in param_ids:
                print(f"FAILED - missing config param: {param}")
                return False
        print("OK")

        print("\n[Smoke Test] PASSED! Service API is ready.\n")
        return True

    except requests.Timeout:
        print(f"\n[ERROR] Connection timeout: {url}")
        print("Please check if Service API is running")
        return False
    except requests.ConnectionError:
        print(f"\n[ERROR] Cannot connect: {url}")
        print("Please check:")
        print("  1. Is Service API running?")
        print("  2. Is the URL correct?")
        print("  3. Is network connection working?")
        return False
    except requests.HTTPError as e:
        print(f"\n[ERROR] HTTP error: {e}")
        return False
    except Exception as e:
        print(f"\n[ERROR] Unknown error: {e}")
        return False

def make_env(casca_url, reward_fn, max_episode_steps=None, rank=0, save_dir=None):
    """Environment factory function (with robust wrapper)"""
    # Use config defaults
    if max_episode_steps is None:
        max_episode_steps = cfg.environment.max_episode_steps
    if save_dir is None:
        save_dir = cfg.training.save_dir

    def _init():
        import requests
        from gymnasium.wrappers import TimeLimit
        session = requests.Session()

        # Create finishing condition - using step count
        class StepCountFinisher(FinishingCondition):
            def __init__(self, max_steps):
                self.max_steps = max_steps
                self.current_step = 0

            def terminate_or_truncate(self, obs, reward):
                self.current_step += 1
                # Check if max steps reached
                truncate = self.current_step >= self.max_steps

                # Check for severe SLO violation (early termination)
                fps = obs[0] if len(obs) > 0 else 0
                terminate = fps < 10  # Early termination if FPS is extremely low

                if truncate or terminate:
                    self.current_step = 0  # Reset counter

                return terminate, truncate

        env = ServiceEnv(
            casca_url=casca_url,
            session=session,
            reward=reward_fn,
            finisher=StepCountFinisher(max_episode_steps),
            wait_seconds=ServiceAPIConfig.WAIT_SECONDS,
            slos=ServiceAPIConfig.SLOS,
            config_params=ServiceAPIConfig.CONFIG_PARAMS
        )

        # Add discrete action wrapper (to improve exploration)
        env = DiscreteActionWrapper(env)

        # Add robust wrapper (error handling and retry)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        error_log_file = f"{save_dir}/api_error_log_{timestamp}.csv"
        env = RobustServiceEnvWrapper(
            env,
            max_retries=cfg.robustness.max_retries,
            retry_delay=cfg.robustness.retry_delay,
            error_log_file=error_log_file
        )

        # Wrap with Monitor to record episode statistics
        env = Monitor(env, filename=f"./logs/monitor_{rank}")
        return env

    return _init

def train_ppo():
    """Main training function - discrete action space version"""
    print("[PPO Training] Starting PPO training - Discrete Action Space Version")
    print(f"Configuration (from config.yaml):")
    print(f"  - Episode length: {cfg.environment.max_episode_steps} steps ({cfg.environment.wait_seconds}s each = {cfg.environment.max_episode_steps * cfg.environment.wait_seconds / 60:.0f} min)")
    print(f"  - Total timesteps: {cfg.training.total_timesteps}")
    print(f"  - Expected episodes: {cfg.training.total_timesteps // cfg.environment.max_episode_steps}")
    print(f"  - Expected training time: {cfg.training.total_timesteps * cfg.environment.wait_seconds / 3600:.1f} hours")
    print(f"  - SLO target: FPS {cfg.slo_targets.fps.min}-{cfg.slo_targets.fps.max}")
    print(f"  - Action range: {cfg.action_space.encoding_thread_count.min}-{cfg.action_space.encoding_thread_count.max}")
    print(f"  - Using discrete action space for better exploration")


    # Smoke test: verify Service API connection
    if not test_service_api_connection(ServiceAPIConfig.CASCA_URL):
        print("\n[ABORT] Smoke test failed, terminating training")
        sys.exit(1)

    # Check CUDA availability
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./models/ppo_test_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs("./logs", exist_ok=True)

    print(f"\nLogs and models save directory: {save_dir}")

    # Initialize Carbon Intensity cache
    print("\n[Loading] Loading Carbon Intensity historical data...")
    carbon_cache = CarbonIntensityCacheHourly(
        csv_file=cfg.carbon.csv_file,
        default_value=cfg.carbon.default_value
    )

    # Create reward function (using carbon cache, SLO targets from config)
    reward_fn = CustomReward(
        carbon_cache=carbon_cache,
        target_fps_min=cfg.slo_targets.fps.min,
        target_fps_max=cfg.slo_targets.fps.max
    )

    # Create single environment (for detailed logging)
    env_fn = make_env(
        casca_url=ServiceAPIConfig.CASCA_URL,
        reward_fn=reward_fn,
        max_episode_steps=cfg.environment.max_episode_steps,
        rank=0,
        save_dir=save_dir
    )
    env = DummyVecEnv([env_fn])

    # PPO hyperparameters (from config.yaml)
    ppo_config = {
        "policy": "MlpPolicy",
        "learning_rate": cfg.ppo.learning_rate,
        "n_steps": cfg.ppo.n_steps,
        "batch_size": cfg.ppo.batch_size,
        "n_epochs": cfg.ppo.n_epochs,
        "gamma": cfg.ppo.gamma,
        "gae_lambda": cfg.ppo.gae_lambda,
        "clip_range": cfg.ppo.clip_range,
        "ent_coef": cfg.ppo.ent_coef,
        "vf_coef": cfg.ppo.vf_coef,
        "max_grad_norm": cfg.ppo.max_grad_norm,
        "device": device,
        "tensorboard_log": cfg.training.tensorboard_dir,
        "verbose": 1
    }

    # Create PPO model
    model = PPO(env=env, **ppo_config)

    print(f"\nModel architecture:")
    print(f"- Policy network: {model.policy}")
    print(f"- Observation space: {env.observation_space}")
    print(f"- Action space: {env.action_space}")

    # Setup callbacks
    callbacks = [
        TrainingCallback(save_dir, carbon_cache=carbon_cache),
        CheckpointCallback(
            save_freq=cfg.training.checkpoint_freq,
            save_path=save_dir,
            name_prefix="ppo_checkpoint"
        )
    ]

    # Start training (total timesteps from config)
    total_timesteps = cfg.training.total_timesteps
    print(f"\n{'='*80}")
    print(f"Starting training for {total_timesteps} steps...")
    print(f"{'='*80}\n")

    start_time = time.time()

    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        progress_bar=True
    )

    training_duration = (time.time() - start_time) / 3600.0  # hours

    # Save final model
    final_model_path = f"{save_dir}/ppo_final.zip"
    model.save(final_model_path)

    # Get API error statistics
    try:
        # Extract actual environment from DummyVecEnv
        actual_env = env.envs[0]
        # Find RobustServiceEnvWrapper (may be wrapped by Monitor)
        while hasattr(actual_env, 'env'):
            if isinstance(actual_env, RobustServiceEnvWrapper):
                error_summary = actual_env.get_error_summary()
                break
            actual_env = actual_env.env
        else:
            error_summary = None
    except Exception as e:
        print(f"[WARNING] Cannot get error statistics: {e}")
        error_summary = None

    print(f"\n{'='*80}")
    print(f"[DONE] Training completed!")
    print(f"{'='*80}")
    print(f"Training duration: {training_duration:.2f} hours")
    print(f"Model saved: {final_model_path}")
    print(f"\nLog files:")
    print(f"  - Step log: {save_dir}/step_log_*.csv")
    print(f"  - Episode log: {save_dir}/episode_log_*.csv")
    print(f"  - TensorBoard: ./tensorboard_logs/")

    if error_summary:
        print(f"\n[Statistics] Service API error statistics:")
        print(f"  - Get observation failures: {error_summary['get_observation_errors']}")
        print(f"  - Perform action failures: {error_summary['perform_action_errors']}")
        print(f"  - Total Service API errors: {error_summary['total_errors']}")

    # Carbon data statistics
    carbon_stats = carbon_cache.get_statistics()
    print(f"\n[Statistics] Carbon Intensity data statistics:")
    print(f"  - Total queries: {carbon_stats['total_queries']:,}")
    print(f"  - Cache hit rate: {carbon_stats['hit_rate']:.1f}%")
    print(f"  - Data source: Austria 2024 historical data (hourly)")

    # Overall evaluation
    total_errors = error_summary['total_errors'] if error_summary else 0
    if total_errors == 0:
        print(f"\n[SUCCESS] Perfect run! No Service API errors")
    elif total_errors < 10:
        print(f"\n[OK] Training mostly stable, few Service API errors ({total_errors})")
    else:
        print(f"\n[WARNING] Many Service API errors occurred ({total_errors})")
        print(f"  Consider checking network connection or API service status")

    print(f"{'='*80}\n")

    return model, save_dir

def export_to_onnx(model, save_dir):
    """Export model to ONNX format for Jetson deployment"""
    print("\n[Export] Exporting model to ONNX format...")

    # Get model's observation space
    obs_shape = model.observation_space.shape

    # Create dummy input
    dummy_input = torch.randn(1, *obs_shape).to(model.device)

    # Export policy network
    onnx_path = f"{save_dir}/ppo_policy.onnx"

    # Extract actor network (for inference)
    actor = model.policy.actor
    actor.eval()

    # Export to ONNX
    torch.onnx.export(
        actor,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=11,
        input_names=['observation'],
        output_names=['action'],
        dynamic_axes={
            'observation': {0: 'batch_size'},
            'action': {0: 'batch_size'}
        }
    )

    print(f"[DONE] ONNX model exported: {onnx_path}")

    # Verify ONNX model
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("[DONE] ONNX model verification passed")

    return onnx_path

if __name__ == "__main__":
    try:
        # Train model
        model, save_dir = train_ppo()

        print(f"\n[COMPLETE] Training finished! All logs saved to {save_dir}")
        print(f"\nNext steps:")
        print(f"  1. Analyze log files and plot training curves")
        print(f"  2. Decide if parameter tuning is needed based on results")
        print(f"  3. If results are good, consider longer training runs")

    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Training interrupted by user")
        print("Model and logs saved up to current progress")
    except Exception as e:
        print(f"\n\n[ERROR] Training error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
