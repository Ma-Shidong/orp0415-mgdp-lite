# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D

from torchrl.data import CompositeSpec, TensorSpec
from torchrl.modules import ProbabilisticActor
from torchrl.envs.transforms import CatTensors
from tensordict import TensorDict
from tensordict.nn import TensorDictModuleBase, TensorDictModule, TensorDictSequential

from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
from typing import Union
import einops

from ..utils.valuenorm import ValueNorm1
from ..modules.distributions import IndependentNormal, IndependentBeta
from .common import GAE

@dataclass
class PPOConfig:
    name: str = "ppo"
    train_every: int = 256 # 每次 PPO 更新用多少数据
    ppo_epochs: int = 4 #同一份 rollout 数据被重复用多少次
    num_minibatches: int = 4 #16 每个 epoch 切成多少小批


    # Exploration / regularization
    # Lower = less random, more stable. Keep some exploration for curriculum.
    entropy_coef: float = 0.01

    # Actor log-std bounds (stability). std = exp(log_std).
    # Typical range: [-2.0, 0.5] => std in [0.14, 1.65]
    actor_log_std_min: float = -2.0
    actor_log_std_max: float = 0.5
    # log-std init (in log-std space; mapped to raw param internally)
    actor_log_std_init: float = -0.5
    # Optional entropy anneal (set in config; handled in train.py)
    entropy_coef_mid: float = 0.002
    entropy_coef_end: float = 0.0002
    entropy_anneal_frac1: float = 0.3
    entropy_anneal_frac2: float = 0.7
    # Entropy schedule:
    # - by_switch_age: decay after each curriculum switch (recommended).
    # - by_level: legacy behavior coupled to unlocked curriculum level.
    entropy_schedule_mode: str = "by_switch_age"
    entropy_decay_iters_per_level: int = 600
    # Extra exploration recovery right after curriculum switch.
    switch_entropy_boost_mult: float = 1.0
    switch_entropy_boost_iters: int = 0
    switch_std_recover_ratio: float = 0.0
    # Asymmetric critic + auxiliary supervision.
    critic_priv_enable: bool = True
    critic_aux_enable: bool = True
    critic_aux_w: float = 0.05
    critic_aux_target_idx: tuple = (0, 4, 5, 7)
    actor_lr: float = 1.0e-4
    critic_lr: float = 2.0e-4
    clip_param: float = 0.2

    # whether to use privileged information
    priv_actor: bool = False
    priv_critic: bool = False

    checkpoint_path: Union[str, None] = None

cs = ConfigStore.instance()
cs.store("ppo", node=PPOConfig, group="algo")
cs.store("ppo_priv", node=PPOConfig(priv_actor=True, priv_critic=True), group="algo")
cs.store("ppo_priv_critic", node=PPOConfig(priv_critic=True), group="algo")


def make_mlp(num_units):
    layers = []
    for n in num_units:
        layers.append(nn.LazyLinear(n))
        layers.append(nn.LeakyReLU())
        layers.append(nn.LayerNorm(n))
    return nn.Sequential(*layers)


class Actor(nn.Module):
    def __init__(
        self,
        action_dim: int,
        log_std_min: float = -2.5,
        log_std_max: float = 0.0,
        log_std_init: float = 0.0,
    ) -> None:
        super().__init__()
        self.actor_mean = nn.LazyLinear(action_dim)

        # Clamp bounds (keep ordered).
        lo = float(log_std_min)
        hi = float(log_std_max)
        if hi < lo:
            lo, hi = hi, lo
        self.log_std_min = lo
        self.log_std_max = hi
        init = float(log_std_init)
        # Interpret log_std_init in log-std space and map to raw parameter if squashing is enabled.
        if self.log_std_min is not None and self.log_std_max is not None:
            if self.log_std_max > self.log_std_min + 1e-6:
                init = max(self.log_std_min, min(self.log_std_max, init))
                t = (2.0 * (init - self.log_std_min) / (self.log_std_max - self.log_std_min)) - 1.0
                t = max(-0.999, min(0.999, t))
                init = math.atanh(t)
            else:
                init = 0.0

        # Learnable raw log-std parameter (shared across all features).
        self.actor_std = nn.Parameter(torch.full((action_dim,), float(init)))

    def forward(self, features: torch.Tensor):
        loc = self.actor_mean(features)

        log_std = self.actor_std
        # Use smooth squashing to keep gradients alive and stay within [min, max].
        if self.log_std_min is not None and self.log_std_max is not None:
            t = torch.tanh(log_std)  # (-1, 1)
            log_std = self.log_std_min + 0.5 * (t + 1.0) * (self.log_std_max - self.log_std_min)

        scale = torch.exp(log_std).expand_as(loc)
        return loc, scale

class BetaActor(nn.Module):
    def __init__(self, action_dim: int) -> None:
        super().__init__()
        self.alpha_layer = nn.LazyLinear(action_dim)
        self.beta_layer = nn.LazyLinear(action_dim)
        self.alpha_softplus = nn.Softplus()
        self.beta_softplus = nn.Softplus()
    
    def forward(self, features: torch.Tensor):
        alpha = 1. + self.alpha_softplus(self.alpha_layer(features)) + 1e-6
        beta = 1. + self.beta_softplus(self.beta_layer(features)) + 1e-6
        return alpha, beta

class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.01))
        self.clip_param = float(getattr(cfg, "clip_param", 0.2))
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        if self.cfg.priv_actor:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            actor_module = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(
                        make_mlp([256, 256]),
                        Actor(
                            self.action_dim,
                            log_std_min=float(getattr(self.cfg, "actor_log_std_min", -2.5)),
                            log_std_max=float(getattr(self.cfg, "actor_log_std_max", 0.0)),
                            log_std_init=float(getattr(self.cfg, "actor_log_std_init", 0.0)),
                        ),
                    ),
                    ["feature"], ["loc", "scale"]
                )
            )
        else:
            actor_module=TensorDictModule(
                nn.Sequential(
                    make_mlp([256, 256, 256]),
                        Actor(
                            self.action_dim,
                            log_std_min=float(getattr(self.cfg, "actor_log_std_min", -2.5)),
                            log_std_max=float(getattr(self.cfg, "actor_log_std_max", 0.0)),
                            log_std_init=float(getattr(self.cfg, "actor_log_std_init", 0.0)),
                        ),
                ),
                [("agents", "observation")], ["loc", "scale"]
            )
        self.actor: ProbabilisticActor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        if self.cfg.priv_critic:
            intrinsics_dim = observation_spec[("agents", "intrinsics")].shape[-1]
            self.critic = TensorDictSequential(
                TensorDictModule(make_mlp([128, 128]), [("agents", "observation")], ["feature"]),
                TensorDictModule(
                    nn.Sequential(nn.LayerNorm(intrinsics_dim), make_mlp([64, 64])),
                    [("agents", "intrinsics")], ["context"]
                ),
                CatTensors(["feature", "context"], "feature"),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["feature"], ["state_value"]
                )
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.Sequential(make_mlp([256, 256, 256]), nn.LazyLinear(1)),
                [("agents", "observation")], ["state_value"]
            ).to(self.device)

        self.actor(fake_input)
        self.critic(fake_input)

        if self.cfg.checkpoint_path is not None:
            state_dict = torch.load(self.cfg.checkpoint_path)
            self.load_state_dict(state_dict, strict=False)
        else:
            def init_(module):
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, 0.01)
                    nn.init.constant_(module.bias, 0.)

            self.actor.apply(init_)
            self.critic.apply(init_)

        actor_lr = float(getattr(self.cfg, "actor_lr", 1.0e-4))
        critic_lr = float(getattr(self.cfg, "critic_lr", 2.0e-4))
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.value_norm = ValueNorm1(reward_spec.shape[-2:]).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = einops.repeat(
            tensordict[("next", "terminated")],
            "t e 1 -> t e a 1",
            a=self.n_agents
        )
        values = tensordict["state_value"]
        values = self.value_norm.denormalize(values)
        next_values = self.value_norm.denormalize(next_values)

        adv, ret = self.gae(rewards, dones, values, next_values)
        adv_mean = adv.mean()
        adv_std = adv.std()
        adv = (adv - adv_mean) / adv_std.clip(1e-7)
        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.-self.clip_param, 1.+self.clip_param)
        policy_loss = - torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = - self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(
            -self.clip_param, self.clip_param
        )
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        loss = policy_loss + entropy_loss + value_loss
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.actor_opt.step()
        self.critic_opt.step()
        explained_var = 1 - F.mse_loss(values, b_returns) / b_returns.var()
        return TensorDict({
            "policy_loss": policy_loss,
            "value_loss": value_loss,
            "entropy": entropy,
            "actor_grad_norm": actor_grad_norm,
            "critic_grad_norm": critic_grad_norm,
            "explained_var": explained_var
        }, [])


def make_batch(tensordict: TensorDict, num_minibatches: int):
    tensordict = tensordict.reshape(-1)
    perm = torch.randperm(
        (tensordict.shape[0] // num_minibatches) * num_minibatches,
        device=tensordict.device,
    ).reshape(num_minibatches, -1)
    for indices in perm:
        yield tensordict[indices]
