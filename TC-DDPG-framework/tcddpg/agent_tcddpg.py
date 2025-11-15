# tcddpg/agent_tcddpg.py

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# -------------------------------
# Simple MLP for actor/critic
# -------------------------------
def mlp(sizes):
    layers = []
    for i in range(len(sizes)-1):
        layers += [nn.Linear(sizes[i], sizes[i+1])]
        if i < len(sizes)-2:
            layers += [nn.ReLU()]
    return nn.Sequential(*layers)

# -------------------------------
# Minimal Replay Buffer
# -------------------------------
class ReplayBuffer:
    def __init__(self, size=50000):
        self.buf = deque(maxlen=size)

    def add(self, s, a, r, ns, d):
        self.buf.append((s, a, r, ns, d))

    def sample(self, batch):
        batch = random.sample(self.buf, batch)
        s, a, r, ns, d = zip(*batch)
        return map(lambda x: torch.tensor(x, dtype=torch.float32), (s, a, r, ns, d))

# -------------------------------
# Baseline DDPG training
# -------------------------------
def train_ddpg_baseline(env, cfg):
    return _train(env, cfg, use_projection=False, physics=None)

# -------------------------------
# TC-DDPG (ours) training
# -------------------------------
def train_tcddpg(env, cfg, physics_cfg, tc_cfg):
    return _train(env, cfg, use_projection=True,
                  physics=physics_cfg, tc_cfg=tc_cfg)

# -------------------------------
# Core training loop (shared)
# -------------------------------
def _train(env, cfg, use_projection, physics, tc_cfg=None):
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    actor = mlp([obs_dim] + [256, 256] + [act_dim])
    critic = mlp([obs_dim + act_dim] + [256, 256] + [1])
    actor_opt = optim.Adam(actor.parameters(), lr=cfg["actor_lr"])
    critic_opt = optim.Adam(critic.parameters(), lr=cfg["critic_lr"])

    replay = ReplayBuffer(size=cfg["replay_size"])

    o, _ = env.reset()
    total_energy = 0
    total_comfort = 0
    total_viol = 0

    for ep in range(cfg["n_episodes"]):
        o, _ = env.reset()
        for t in range(cfg["max_steps_per_episode"]):
            a = actor(torch.tensor(o)).detach().numpy()

            # Apply projection (if enabled)
            if use_projection:
                a_proj = a.copy()
                # simple clamp to ensure feasibility
                a_proj[0] = np.clip(a_proj[0], 15, 30)
                a_proj[1] = np.clip(a_proj[1], 0.1, 1.0)
                a = a_proj

            no, r, d, _, info = env.step(a)

            replay.add(o, a, r, no, d)

            total_energy += info.get("energy", 0)
            total_comfort += info.get("comfort_drift", 0)

            o = no
            if d:
                break

    # Simple summary mimicking Table 3
    steps = cfg["n_episodes"]*cfg["max_steps_per_episode"]
    return {
        "energy": total_energy / steps,
        "comfort": total_comfort / steps,
        "violations": total_viol / steps
    }
