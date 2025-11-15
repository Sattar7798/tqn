# tcddpg/env_rc_model.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces

class RCMultiZoneEnv(gym.Env):
    """
    Minimal reduced-order RC multi-zone building model.
    5 zones: N, S, E, W, Core.
    State: zone temps + outdoor temp + setpoint.
    Action: [supply_air_temperature, vav_flow_factor].
    """
    metadata = {"render_modes": []}

    def __init__(self, cfg):
        super().__init__()

        self.dt = cfg.get("timestep_min", 5) * 60  # 5 min timestep
        self.n_zones = cfg.get("n_zones", 5)
        self.episode_len = cfg.get("episode_len_steps", 288)

        # simple RC parameters (placeholders but stable)
        self.C = np.ones(self.n_zones) * 2e5  # thermal capacitance
        self.Ro = np.ones(self.n_zones) * 2.0 # resistances to outside

        # action space: sat + flow
        self.action_space = spaces.Box(
            low=np.array([15.0, 0.1]),
            high=np.array([30.0, 1.0]),
            dtype=np.float32
        )

        # observation: zone temps + outdoor + setpoint
        low = np.array([-10]*self.n_zones + [-20, 10], dtype=np.float32)
        high = np.array([50]*self.n_zones + [50, 35], dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high)

        self.reset_state()

    def reset_state(self):
        self.t = 0
        self.zone_temps = 22.0 + 0.3*np.random.randn(self.n_zones)
        return self._get_obs()

    def _get_obs(self):
        outdoor = 10 + 10*np.sin(self.t / 24 * 2*np.pi)
        setp = 23 if 8 <= (self.t % 24) < 18 else 26
        return np.concatenate([self.zone_temps, [outdoor, setp]]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = self.reset_state()
        return obs, {}

    def step(self, action):
        sat, flow = action

        # simple RC dynamics
        outdoor = 10 + 10*np.sin(self.t / 24 * 2*np.pi)
        gains = (sat - self.zone_temps) * flow * 180  # simplistic HVAC influence

        # update temps
        self.zone_temps += self.dt / self.C * (
            (outdoor - self.zone_temps) / self.Ro + gains
        )

        # 🔹 NEW: clamp zone temperatures to a realistic safe range
        self.zone_temps = np.clip(self.zone_temps, 10.0, 35.0)

        # reward (very simplified minimal version)
        comfort_drift = np.maximum(0, np.abs(self.zone_temps - 23) - 2).sum()
        energy = abs(sat - 12) * flow * 0.7
        reward = -(energy + comfort_drift)

        self.t += 1
        done = self.t >= self.episode_len
        obs = self._get_obs()

        return obs, reward, done, False, {
            "energy": energy,
            "comfort_drift": comfort_drift
        }



def make_rc_env(cfg):
    return RCMultiZoneEnv(cfg)
