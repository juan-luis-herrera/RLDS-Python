# PPO Training Implementation for ServiceGym

## Overview

This repository contains a **Proximal Policy Optimization (PPO)** implementation designed to interact with the ServiceGym platform. The agent learns to optimize `EncodingThreadCount` to maintain FPS within SLO bounds while minimizing carbon footprint.

**Key Achievement**: The PPO agent successfully learns the optimal policy, achieving 95-100% SLO satisfaction rate after training.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        PPO Agent                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────┐ │
│  │ MlpPolicy   │───▶│ PPO (SB3)   │───▶│ TrainingCallback    │ │
│  │ (Actor-     │    │             │    │ (Logging/Stats)     │ │
│  │  Critic)    │    └─────────────┘    └─────────────────────┘ │
│  └─────────────┘            │                                   │
└─────────────────────────────┼───────────────────────────────────┘
                              │ action (EncodingThreadCount)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Environment Wrappers                        │
│  ┌───────────────────┐    ┌────────────────────────────────┐   │
│  │ DiscreteAction    │───▶│ RobustServiceEnvWrapper        │   │
│  │ Wrapper           │    │ (Error handling, Retry logic)  │   │
│  │ (Box→Discrete)    │    └────────────────────────────────┘   │
│  └───────────────────┘                    │                     │
└───────────────────────────────────────────┼─────────────────────┘
                                            │
                                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                     ServiceGym (Your Platform)                  │
│  ServiceEnv ◄──── REST API ────► CASCA Service                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## File Structure

```
PPO_Training_Release/
│
├── train_ppo_service_discrete.py   # Main entry point
├── config.yaml                      # Centralized configuration
├── config_loader.py                 # Configuration management
│
├── discrete_action_wrapper.py       # Discrete action space wrapper
├── robust_service_env.py            # Error handling wrapper
├── carbon_cache_hourly.py           # Carbon intensity data cache
│
├── carbon_intensity_austria_2024_hourly.csv  # Historical carbon data
├── requirements_training.txt        # Python dependencies
└── ServiceGym/                      # Your platform (dependency)
```

---

## Core Components

### 1. Reward Function (`CustomReward`)

```python
# Location: train_ppo_service_discrete.py, line 52-137

class CustomReward(ServiceReward):
    def calculate_reward(self, last_obs, current_obs):
        fps = current_obs[0]      # meanFPS from ServiceGym
        power = current_obs[1]    # meanPower from ServiceGym

        carbon = self.calculate_carbon_footprint(power)

        if self.target_fps_min <= fps <= self.target_fps_max:
            reward = 1.0 / carbon    # Within SLO: minimize carbon
        else:
            reward = -carbon         # SLO violation: penalize

        return reward
```

**Design Rationale**:
- Inverse relationship with carbon when SLO satisfied → agent learns to minimize carbon
- Direct penalty when SLO violated → agent learns to maintain FPS
- Carbon footprint calculated using historical Austria 2024 hourly data

### 2. Discrete Action Wrapper (`DiscreteActionWrapper`)

```python
# Location: discrete_action_wrapper.py

# Problem: PPO with Box action space outputs continuous values clustered around 0
# Solution: Convert to Discrete space for uniform exploration

Original: Box(low=0, high=16) → PPO outputs ~0.01, -0.02, 0.15...
Wrapped:  Discrete(17)        → PPO outputs 0, 1, 2, ..., 16
```

**Why needed**: ServiceGym's `EncodingThreadCount` is integer [0-16], but PPO's Gaussian policy for Box spaces doesn't explore well in small integer ranges.

### 3. Robust Environment Wrapper (`RobustServiceEnvWrapper`)

```python
# Location: robust_service_env.py

Features:
├── Automatic retry (3 attempts, 10s interval)
├── Fallback to cached observation on API failure
├── Detailed error logging to CSV
└── Action preprocessing for DummyVecEnv compatibility
```

**Why needed**: Network instability during long training runs (24+ hours) would otherwise crash training.

### 4. Carbon Intensity Cache (`CarbonIntensityCacheHourly`)

```python
# Location: carbon_cache_hourly.py

# Maps current time to same period last year
# Uses hourly granularity, no interpolation
current_time (2025-12-17 14:30) → lookup (2024-12-17 14:00) → 285.5 gCO2eq/kWh
```

**Why needed**: Provides realistic carbon intensity variation without real-time EMMA API dependency.

---

## Configuration Management

All parameters are centralized in `config.yaml` and loaded via `config_loader.py`.

### config.yaml Structure

```yaml
# API endpoints
api:
  casca_url: "http://localhost:8080/v0"

# Environment timing
environment:
  wait_seconds: 180        # Action interval (critical parameter)
  max_episode_steps: 48    # Steps per episode

# SLO targets
slo_targets:
  fps:
    min: 20
    max: 31

# Action space bounds
action_space:
  encoding_thread_count:
    min: 0
    max: 16

# PPO hyperparameters
ppo:
  learning_rate: 0.0003
  n_steps: 48              # = max_episode_steps (collect full episode)
  batch_size: 12           # n_steps / 4
  ent_coef: 0.02           # Increased for exploration
  ...

# Training settings
training:
  total_timesteps: 480     # 10 episodes
  checkpoint_freq: 48      # Save every episode
```

### Accessing Configuration

```python
from config_loader import get_config

cfg = get_config()

# Access any parameter
url = cfg.api.casca_url
fps_min = cfg.slo_targets.fps.min
lr = cfg.ppo.learning_rate
```

### Modifying Parameters

Simply edit `config.yaml` - no code changes needed. All files read from this single source.

---

## Key Design Decisions

### 1. Action Interval: 180 seconds

**Experiment**: Action sweep test showed FPS response time of 75-402 seconds (mean: 182s).

**Decision**: `wait_seconds = 180` ensures the agent observes the effect of its action before taking the next one.

### 2. Episode Length: 48 steps

**Reasoning**:
- 48 steps × 180s = 144 minutes = 2.4 hours per episode
- Long enough to observe carbon intensity variation
- Short enough for reasonable training iteration

### 3. Entropy Coefficient: 0.02

**Issue**: Non-monotonic relationship between EncodingThreadCount and FPS (optimal at ~10, not 16).

**Solution**: Higher entropy coefficient encourages exploration of the non-convex action space.

### 4. Discrete Action Space

**Issue**: PPO's continuous policy outputs clustered around 0 for small integer ranges.

**Solution**: `DiscreteActionWrapper` converts to categorical distribution for uniform exploration.

---

## Running the Training

```bash
# 1. Install dependencies
pip install -r requirements_training.txt

# 2. Ensure ServiceGym API is accessible
curl http://localhost:8080/v0/slos

# 3. Run training
python train_ppo_service_discrete.py
```

### Output Structure

```
./models/ppo_test_YYYYMMDD_HHMMSS/
├── step_log_*.csv          # Per-step metrics
├── episode_log_*.csv       # Per-episode summary
├── api_error_log_*.csv     # API errors (if any)
├── ppo_checkpoint_*.zip    # Periodic checkpoints
└── ppo_final.zip           # Final trained model
```

---

## Training Flow

```
1. Initialize
   ├── Load config.yaml
   ├── Load carbon intensity data
   ├── Create ServiceEnv + Wrappers
   └── Initialize PPO model

2. Training Loop (for each timestep)
   ├── Agent selects action (discrete 0-16)
   ├── DiscreteActionWrapper converts to continuous
   ├── RobustWrapper sends to ServiceGym API
   ├── Wait 180 seconds
   ├── Get observation [meanFPS, meanPower]
   ├── Calculate reward (carbon-based)
   └── PPO updates policy

3. Logging
   ├── TrainingCallback logs each step
   ├── Episode summary on completion
   └── Checkpoints saved periodically

4. Completion
   ├── Save final model
   └── Print statistics
```

---

## Extending the Implementation

### Adding New SLO Metrics

1. Add to `config.yaml`:
```yaml
slo_targets:
  fps:
    min: 20
    max: 31
  latency:           # New metric
    max: 100
```

2. Modify `CustomReward.calculate_reward()` to incorporate new metric.

### Adding New Config Parameters

1. Add to `config.yaml`
2. Add dataclass in `config_loader.py`
3. Access via `cfg.new_section.new_param`

### Adjusting Training Duration

Edit `config.yaml`:
```yaml
training:
  total_timesteps: 960    # Double the training (20 episodes)
```

---

## Dependencies

- **stable-baselines3**: PPO implementation
- **gymnasium**: Environment interface
- **torch**: Neural network backend
- **pyyaml**: Configuration loading
- **ServiceGym**: Your platform (provided separately)

---

