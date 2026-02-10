import gymnasium as gym
import numpy as np
from requests import Session
from abc import ABC, abstractmethod
import time

class ServiceReward(ABC):

    @abstractmethod
    def calculate_reward(self, prev_obs_space: gym.spaces.Space, curr_obs_space: gym.spaces.Space) -> float:
        pass

class FinishingCondition(ABC):

    @abstractmethod
    def terminate_or_truncate(self, observation: gym.spaces.Space, reward: float) -> tuple[float, float]:
        pass

class ServiceEnv(gym.Env):
    SLOS_ENDPOINT = "/slos"
    CONFIG_PARAMS_ENDPOINT = "/config"

    def __init__(self, casca_url: str, session: Session, reward: ServiceReward, finisher: FinishingCondition, wait_seconds: float, slos: list[str] = [], config_params: list[str] = []):
        self._session = session
        self._url = casca_url
        self.action_space, self._conf_data = self._create_action_space(config_params)
        self.observation_space, self._slo_space = self._create_observation_space(slos, self.action_space)
        self._reward = reward
        self._finisher = finisher
        self._t = wait_seconds
        self._last_observation = self._get_observation()

    def _create_observation_space(self, slos: list[str], action_space: gym.Space) -> tuple[gym.spaces.Space, list[str]]:
        slo_discovery = self._session.get(f"{self._url}{self.SLOS_ENDPOINT}")
        discovered_slos = slo_discovery.json()
        spaces = []
        slo_list = []
        if slos is None or len(slos) == 0:
            slos = list(slos.keys())
        for slo in discovered_slos:
            if slo["id"] in slos:
                if slo["type"] in ["range", "target_value"]:
                    low = slo.get("description", {}).get("minValue")
                    high = slo.get("description", {}).get("maxValue")
                    if low is not None and high is not None:
                        space = gym.spaces.Box(low=low, high=high, dtype=float)
                    elif low is not None:
                        space = gym.spaces.Box(low=low, dtype=float)
                    elif high is not None:
                        space = gym.spaces.Box(high=high, dtype=float)
                    else:
                        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
                elif slo["type"] in ["integer_range", "integer_target_value"]:
                    low = slo.get("description", {}).get("minValue")
                    high = slo.get("description", {}).get("maxValue")
                    if low is not None and high is not None:
                        space = gym.spaces.Box(low=low, high=high, dtype=int)
                    elif low is not None:
                        space = gym.spaces.Box(low=low, dtype=int)
                    elif high is not None:
                        space = gym.spaces.Box(high=high, dtype=int)
                    else:
                        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=int)
                elif slo["type"] == "boolean":
                    space = gym.spaces.MultiBinary(1)
                else:
                    raise RuntimeError(f"SLO {slo['id']} of type {slo['type']} unsupported")
                spaces.append(space)
                slo_list.append(slo['id'])
        spaces.append(action_space)
        return gym.spaces.flatten_space(gym.spaces.Tuple(spaces)), slo_list
    
    def _create_action_space(self, config_params: list[str]) -> tuple[gym.spaces.Space, dict[str, list|None]]:
        config_discovery = self._session.get(f"{self._url}{self.CONFIG_PARAMS_ENDPOINT}")
        discovered_params = config_discovery.json()
        spaces = []
        param_data = {}
        if config_params is None or len(config_params) == 0:
            config_params = [c["id"] for c in discovered_params]
        for param in discovered_params:
            if param["id"] in config_params:
                if param["type"] == "range":
                    low = param.get("description", {}).get("minValue")
                    high = param.get("description", {}).get("maxValue")
                    if low is not None and high is not None:
                        space = gym.spaces.Box(low=low, high=high, dtype=float)
                    elif low is not None:
                        space = gym.spaces.Box(low=low, dtype=float)
                    elif high is not None:
                        space = gym.spaces.Box(high=high, dtype=float)
                    else:
                        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=float)
                    param_list = None
                elif param["type"] == "integer_range":
                    low = param.get("description", {}).get("minValue")
                    high = param.get("description", {}).get("maxValue")
                    if low is not None and high is not None:
                        space = gym.spaces.Box(low=low, high=high, dtype=int)
                    elif low is not None:
                        space = gym.spaces.Box(low=low, dtype=int)
                    elif high is not None:
                        space = gym.spaces.Box(high=high, dtype=int)
                    else:
                        space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=int)
                    param_list = None
                elif "value_list" in param["type"]:
                    value_list = param.get("description", {}).get("valueList", [])
                    if len(value_list) <= 1:
                        continue
                    space = gym.spaces.Discrete(len(value_list))
                    param_list = value_list
                elif param["type"] == "boolean":
                    space = gym.spaces.MultiBinary(1)
                    param_list = None
                else:
                    raise RuntimeError(f"Parameter {param['id']} of type {param['type']} unsupported")
                spaces.append(space)
                param_data[param["id"]] = param_list
        return gym.spaces.flatten_space(gym.spaces.Tuple(spaces)), param_data
    
    def _get_observation(self) -> np.ndarray:
        slo_discovery = self._session.get(f"{self._url}{self.SLOS_ENDPOINT}")
        discovered_slos = slo_discovery.json()
        config_discovery = self._session.get(f"{self._url}{self.CONFIG_PARAMS_ENDPOINT}")
        discovered_params = config_discovery.json()
        observations = list(map(lambda s: s["value"], filter(lambda x: x["id"] in self._slo_space, discovered_slos)))
        for param in discovered_params:
            if param["id"] in self._conf_data:
                lst = self._conf_data[param["id"]]
                if lst is None:
                    obs_val = param.get("value", np.nan)
                else:
                    try:
                        obs_val = lst.index(param.get("value", np.nan))
                    except ValueError:
                        obs_val = np.nan
                observations.append(obs_val)
        return np.array(observations)
    
    def _perform_action(self, action):
        for param, act in zip(list(self._conf_data.keys()), action):
            transformer = self._conf_data[param]
            if transformer is None:
                value = act
            else:
                value = transformer[act]
            if isinstance(value, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
                value = int(value)
            elif isinstance(value, (np.float_, np.float16, np.float32, np.float64)):
                value = float(value)
            elif isinstance(value, (np.bool_)):
                value = bool(value)
            self._session.put(f"{self._url}{self.CONFIG_PARAMS_ENDPOINT}/{param}", json={"value": value})
        
    def reset(self, *, seed=None, options=None):
        self._last_observation = self._get_observation()
        return self._last_observation, {}
    
    def step(self, action):
        self._perform_action(action)
        time.sleep(self._t)
        next_obs = self._get_observation()
        reward = self._reward.calculate_reward(self._last_observation, next_obs)
        self._last_observation = next_obs
        finished, truncated = self._finisher.terminate_or_truncate(next_obs, reward)
        return next_obs, reward, finished, truncated, {}