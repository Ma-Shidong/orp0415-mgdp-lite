import logging
import os
import itertools
import random
from contextlib import nullcontext

import hydra
import torch
import wandb
from hydra.utils import to_absolute_path

from tqdm import tqdm
from omegaconf import OmegaConf

from resources import init_simulation_app
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.utils import set_exploration_type, ExplorationType
from resources.utils.torchrl import SyncDataCollector
from resources.utils.torchrl.transforms import (
    FromMultiDiscreteAction,
    FromDiscreteAction,
    ravel_composite
)

from resources.utils.wandb import init_wandb
from resources.utils.torchrl import RenderCallback, EpisodeStats

from setproctitle import setproctitle
from torchrl.envs.transforms import TransformedEnv, InitTracker, Compose

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from resources.learning.ppo.ppo import PPOConfig, make_mlp, make_batch, Actor, IndependentNormal, GAE, ValueNorm1
from resources.learning.modules.rnn import GRU
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Use TF32 on Ampere+ for faster matmul with minimal accuracy impact.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

def make_sequence_batch(tensordict: TensorDict, num_minibatches: int):
    if len(tensordict.batch_size) < 2:
        yield tensordict
        return
    n_envs = int(tensordict.batch_size[1])
    if n_envs <= 0:
        return
    num_minibatches = max(1, min(int(num_minibatches), n_envs))
    td_device = tensordict.device
    if td_device is None:
        td_device = tensordict[("agents", "action")].device
    env_perm = torch.randperm(n_envs, device=td_device)
    for env_idx in torch.tensor_split(env_perm, num_minibatches):
        if env_idx.numel() == 0:
            continue
        yield tensordict[:, env_idx]


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        model_cfg: Optional[Any],
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.model_cfg = model_cfg
        self.device = device

        self.entropy_coef = float(getattr(cfg, "entropy_coef", 0.01))
        self.clip_param = float(getattr(cfg, "clip_param", 0.2))
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.995, 0.95)
        self._critic_priv_key = ("agents", "observation_central")
        obs_keys = set(observation_spec.keys(True, True))
        self.use_critic_priv = bool(getattr(cfg, "critic_priv_enable", True)) and (self._critic_priv_key in obs_keys)
        self.use_critic_aux = bool(getattr(cfg, "critic_aux_enable", False))
        self.critic_aux_w = float(getattr(cfg, "critic_aux_w", 0.0))
        aux_idx_cfg = getattr(cfg, "critic_aux_target_idx", (0, 4, 5, 7))
        self.critic_aux_target_idx = tuple(int(i) for i in aux_idx_cfg)
        self.bptt_len = max(1, int(getattr(cfg, "bptt_len", 64)))
        self.sequence_num_minibatches = max(1, int(getattr(cfg, "sequence_num_minibatches", getattr(cfg, "num_minibatches", 4))))

        temporal_cfg = getattr(model_cfg, "temporal", None) if model_cfg is not None else None
        self.temporal_enable = bool(getattr(temporal_cfg, "enable", False))
        self.temporal_type = str(getattr(temporal_cfg, "type", "gru")).lower()
        self.temporal_hidden_size = int(getattr(temporal_cfg, "hidden_size", 128))
        if self.temporal_enable and self.temporal_type != "gru":
            raise NotImplementedError(f"Unsupported temporal core: {self.temporal_type}")
        if self.temporal_enable and self.temporal_hidden_size != 128:
            raise ValueError("Current GRU temporal core requires hidden_size=128 to preserve residual dimensions.")

        teacher_cfg = getattr(model_cfg, "teacher_student", None) if model_cfg is not None else None
        self.teacher_enable = bool(getattr(teacher_cfg, "enable", False)) and self.use_critic_priv
        self.distill_feature_w_start = float(getattr(teacher_cfg, "distill_feature_w", 0.05)) if teacher_cfg is not None else 0.05
        self.distill_feature_w_end = float(getattr(teacher_cfg, "distill_feature_w_end", 0.15)) if teacher_cfg is not None else 0.15
        self.distill_priv_w_start = float(getattr(teacher_cfg, "distill_priv_w", 0.05)) if teacher_cfg is not None else 0.05
        self.distill_priv_w_end = float(getattr(teacher_cfg, "distill_priv_w_end", 0.10)) if teacher_cfg is not None else 0.10
        self.distill_ramp_iters = max(1, int(getattr(teacher_cfg, "ramp_iters", 1000))) if teacher_cfg is not None else 1000
        self.curriculum_level = 0
        self._l4_distill_iters = 0
        self._current_distill_feature_w = 0.0
        self._current_distill_priv_w = 0.0
        self._rollout_hidden = None

        fake_input = observation_spec.zero()

        self.cnn = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]),
            nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]),
            nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128),
            nn.LayerNorm(128),
        )
        self.feature_mlp = make_mlp([256, 256])
        self.encoder = TensorDictSequential(
            TensorDictModule(self.cnn, [("agents", "observation", "lidar")], ["_cnn_feature"]),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(self.feature_mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.temporal_core = GRU(128, self.temporal_hidden_size).to(self.device) if self.temporal_enable else None

        self.actor = ProbabilisticActor(
            TensorDictModule(
                Actor(
                    self.action_dim,
                    log_std_min=float(getattr(cfg, "actor_log_std_min", -2.5)),
                    log_std_max=float(getattr(cfg, "actor_log_std_max", 0.0)),
                    log_std_init=float(getattr(cfg, "actor_log_std_init", 0.0)),
                ),
                ["_feature"],
                ["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        if self.use_critic_priv:
            self.critic = TensorDictSequential(
                CatTensors(["_feature", self._critic_priv_key], "_critic_feature", del_keys=False),
                TensorDictModule(
                    nn.Sequential(make_mlp([256, 256]), nn.LazyLinear(1)),
                    ["_critic_feature"],
                    ["state_value"],
                ),
            ).to(self.device)
        else:
            self.critic = TensorDictModule(
                nn.LazyLinear(1), ["_feature"], ["state_value"]
            ).to(self.device)

        if self.use_critic_aux and self.critic_aux_w > 0.0 and self.use_critic_priv:
            aux_dim = max(1, len(self.critic_aux_target_idx))
            self.critic_aux_head = TensorDictModule(
                nn.Sequential(
                    nn.LazyLinear(128),
                    nn.ELU(),
                    nn.LazyLinear(aux_dim),
                ),
                ["_feature"],
                ["critic_aux_pred"],
            ).to(self.device)
        else:
            self.critic_aux_head = None

        if self.teacher_enable:
            priv_dim = int(observation_spec[self._critic_priv_key].shape[-1])
            self.teacher_feature_head = nn.Sequential(
                nn.LazyLinear(128),
                nn.ELU(),
                nn.LayerNorm(128),
                nn.LazyLinear(self.temporal_hidden_size),
            ).to(self.device)
            self.student_priv_head = nn.Sequential(
                nn.LazyLinear(128),
                nn.ELU(),
                nn.LazyLinear(priv_dim),
            ).to(self.device)
            self.teacher_priv_head = nn.Sequential(
                nn.LazyLinear(128),
                nn.ELU(),
                nn.LazyLinear(priv_dim),
            ).to(self.device)
        else:
            self.teacher_feature_head = None
            self.student_priv_head = None
            self.teacher_priv_head = None

        fake_td = observation_spec.zero()
        self._encode_features(fake_td, use_rollout_state=False)
        self.actor(fake_td)
        self.critic(fake_td)
        if self.critic_aux_head is not None:
            self.critic_aux_head(fake_td)
        if self.teacher_enable:
            self._compute_teacher_losses(fake_td)
        self.reset_rollout_state()

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)
        if self.teacher_feature_head is not None:
            self.teacher_feature_head.apply(init_)
        if self.student_priv_head is not None:
            self.student_priv_head.apply(init_)
        if self.teacher_priv_head is not None:
            self.teacher_priv_head.apply(init_)

        actor_lr = float(getattr(cfg, "actor_lr", 1.5e-4))
        critic_lr = float(getattr(cfg, "critic_lr", 3.0e-4))
        encoder_lr = actor_lr
        encoder_params = list(self.encoder.parameters())
        if self.temporal_core is not None:
            encoder_params.extend(list(self.temporal_core.parameters()))
        if self.teacher_feature_head is not None:
            encoder_params.extend(list(self.teacher_feature_head.parameters()))
        if self.student_priv_head is not None:
            encoder_params.extend(list(self.student_priv_head.parameters()))
        if self.teacher_priv_head is not None:
            encoder_params.extend(list(self.teacher_priv_head.parameters()))
        self._encoder_grad_params = [p for p in encoder_params if p.requires_grad]
        self.encoder_opt = torch.optim.Adam(self._encoder_grad_params, lr=encoder_lr) if self._encoder_grad_params else None
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.value_norm = ValueNorm1(1).to(self.device)

    def set_curriculum_level(self, level: int) -> None:
        self.curriculum_level = int(level)

    def get_rollout_state(self):
        if self._rollout_hidden is None:
            return None
        return self._rollout_hidden.detach().clone()

    def set_rollout_state(self, state) -> None:
        if state is None:
            self._rollout_hidden = None
        else:
            self._rollout_hidden = state.detach().clone().to(self.device)

    def reset_rollout_state(self, batch_size: Optional[int] = None) -> None:
        if not self.temporal_enable:
            self._rollout_hidden = None
            return
        if batch_size is None:
            self._rollout_hidden = None
            return
        self._rollout_hidden = torch.zeros(batch_size, self.temporal_hidden_size, device=self.device)

    def _get_is_initial(self, tensordict: TensorDict, feature: torch.Tensor) -> Optional[torch.Tensor]:
        is_initial = None
        try:
            is_initial = tensordict.get("is_init", None)
        except Exception:
            is_initial = None
        if is_initial is None:
            return None
        is_initial = torch.as_tensor(is_initial, device=feature.device)
        if feature.ndim == 2:
            if is_initial.ndim == 1:
                return is_initial.unsqueeze(-1).to(feature.dtype)
            return is_initial.reshape(feature.shape[0], -1)[..., :1].to(feature.dtype)
        if feature.ndim == 3:
            if is_initial.ndim == 2:
                return is_initial.unsqueeze(-1).to(feature.dtype)
            return is_initial.reshape(feature.shape[0], feature.shape[1], -1)[..., :1].to(feature.dtype)
        return None

    def _run_cnn(self, lidar: torch.Tensor) -> torch.Tensor:
        leading = lidar.shape[:-3]
        x = lidar.reshape(-1, *lidar.shape[-3:])
        y = self.cnn(x)
        return y.reshape(*leading, -1)

    def _apply_last_dim(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor:
        leading = tensor.shape[:-1]
        y = module(tensor.reshape(-1, tensor.shape[-1]))
        return y.reshape(*leading, -1)

    def _run_temporal(
        self,
        cnn_feature: torch.Tensor,
        is_initial: Optional[torch.Tensor],
        hidden_state: Optional[torch.Tensor],
        use_rollout_state: bool,
    ):
        if not self.temporal_enable or self.temporal_core is None:
            return cnn_feature, hidden_state

        if cnn_feature.ndim == 2:
            if is_initial is None:
                is_initial = torch.zeros(cnn_feature.shape[0], 1, device=cnn_feature.device, dtype=cnn_feature.dtype)
            h_in = self._rollout_hidden if use_rollout_state else hidden_state
            if h_in is None or h_in.shape[0] != cnn_feature.shape[0]:
                h_in = torch.zeros(cnn_feature.shape[0], self.temporal_hidden_size, device=cnn_feature.device, dtype=cnn_feature.dtype)
            out, h_out = self.temporal_core(cnn_feature, h=h_in, is_initial=is_initial)
            if use_rollout_state:
                self._rollout_hidden = h_out.detach()
            return out, h_out

        if cnn_feature.ndim != 3:
            raise RuntimeError(f"Unexpected temporal feature shape: {tuple(cnn_feature.shape)}")

        if is_initial is None:
            is_initial = torch.zeros(*cnn_feature.shape[:2], 1, device=cnn_feature.device, dtype=cnn_feature.dtype)
        x = cnn_feature.permute(1, 0, 2)
        seq_is_initial = is_initial.permute(1, 0, 2)
        h_in = hidden_state
        if h_in is not None and h_in.ndim > 2:
            h_in = h_in.reshape(h_in.shape[0], -1)[..., : self.temporal_hidden_size]
        out, h_out = self.temporal_core(x, h=h_in, is_initial=seq_is_initial)
        out = out.permute(1, 0, 2)
        if h_out.ndim == 3:
            h_last = h_out[:, -1]
        else:
            h_last = h_out
        return out, h_last

    def _encode_features(
        self,
        tensordict: TensorDict,
        use_rollout_state: bool,
        hidden_state: Optional[torch.Tensor] = None,
    ):
        lidar = tensordict[("agents", "observation", "lidar")]
        state = tensordict[("agents", "observation", "state")]
        cnn_feature = self._run_cnn(lidar)
        is_initial = self._get_is_initial(tensordict, cnn_feature)
        cnn_feature_mem, next_hidden = self._run_temporal(
            cnn_feature,
            is_initial=is_initial,
            hidden_state=hidden_state,
            use_rollout_state=use_rollout_state,
        )
        feature_input = torch.cat([cnn_feature_mem, state], dim=-1)
        feature = self._apply_last_dim(self.feature_mlp, feature_input)
        tensordict.set("_cnn_feature", cnn_feature)
        tensordict.set("_cnn_feature_mem", cnn_feature_mem)
        tensordict.set("_feature", feature)
        return next_hidden

    def _compute_aux_loss(self, tensordict: TensorDict) -> torch.Tensor:
        if self.critic_aux_head is None or self.critic_aux_w <= 0.0:
            return torch.zeros((), device=self.device)
        if self._critic_priv_key not in tensordict.keys(True, True):
            return torch.zeros((), device=self.device)
        aux_pred = self.critic_aux_head(tensordict)["critic_aux_pred"]
        aux_target = tensordict[self._critic_priv_key]
        while aux_target.ndim > aux_pred.ndim and aux_target.shape[-2] == 1:
            aux_target = aux_target.squeeze(-2)
        try:
            aux_target = aux_target[..., list(self.critic_aux_target_idx)]
        except Exception:
            aux_target = torch.zeros_like(aux_pred)
        if aux_target.shape != aux_pred.shape:
            if aux_target.numel() == aux_pred.numel():
                aux_target = aux_target.reshape_as(aux_pred)
            else:
                aux_target = torch.zeros_like(aux_pred)
        return F.smooth_l1_loss(aux_pred, aux_target.detach())

    def _compute_teacher_losses(self, tensordict: TensorDict):
        zero = torch.zeros((), device=self.device)
        if not self.teacher_enable or self.teacher_feature_head is None or self.student_priv_head is None or self.teacher_priv_head is None:
            return zero, zero, zero
        if self._critic_priv_key not in tensordict.keys(True, True):
            return zero, zero, zero

        student_feature = tensordict["_cnn_feature_mem"]
        priv_target = tensordict[self._critic_priv_key]
        teacher_input = torch.cat([student_feature, priv_target], dim=-1)
        teacher_feature = self._apply_last_dim(self.teacher_feature_head, teacher_input)
        student_priv_pred = self._apply_last_dim(self.student_priv_head, student_feature)
        teacher_priv_pred = self._apply_last_dim(self.teacher_priv_head, teacher_feature)
        tensordict.set("teacher_feature", teacher_feature)
        tensordict.set("student_priv_pred", student_priv_pred)
        tensordict.set("teacher_priv_pred", teacher_priv_pred)
        feature_loss = F.mse_loss(student_feature, teacher_feature.detach())
        priv_recon_loss = F.smooth_l1_loss(student_priv_pred, priv_target.detach())
        teacher_recon_loss = F.smooth_l1_loss(teacher_priv_pred, priv_target.detach())
        return feature_loss, priv_recon_loss, teacher_recon_loss

    def _current_distill_weights(self):
        if not self.teacher_enable or self.curriculum_level < 4:
            return 0.0, 0.0
        progress = min(1.0, float(self._l4_distill_iters) / float(self.distill_ramp_iters))
        feature_w = self.distill_feature_w_start + (self.distill_feature_w_end - self.distill_feature_w_start) * progress
        priv_w = self.distill_priv_w_start + (self.distill_priv_w_end - self.distill_priv_w_start) * progress
        return float(feature_w), float(priv_w)

    def forward(self, tensordict: TensorDict):
        self._encode_features(tensordict, use_rollout_state=True)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude(
            "loc",
            "scale",
            "_feature",
            "_cnn_feature",
            "_cnn_feature_mem",
            "_critic_feature",
            "critic_aux_pred",
            "teacher_feature",
            "student_priv_pred",
            "teacher_priv_pred",
            inplace=True,
        )
        return tensordict

    def _compute_losses(self, tensordict: TensorDict):
        dist = self.actor.get_dist(tensordict)
        log_probs = dist.log_prob(tensordict[("agents", "action")])
        entropy = dist.entropy()

        adv = tensordict["adv"]
        ratio = torch.exp(log_probs - tensordict["sample_log_prob"]).unsqueeze(-1)
        surr1 = adv * ratio
        surr2 = adv * ratio.clamp(1.0 - self.clip_param, 1.0 + self.clip_param)
        policy_loss = -torch.mean(torch.min(surr1, surr2)) * self.action_dim
        entropy_loss = -self.entropy_coef * torch.mean(entropy)

        b_values = tensordict["state_value"]
        b_returns = tensordict["ret"]
        values = self.critic(tensordict)["state_value"]
        values_clipped = b_values + (values - b_values).clamp(-self.clip_param, self.clip_param)
        value_loss_clipped = self.critic_loss_fn(b_returns, values_clipped)
        value_loss_original = self.critic_loss_fn(b_returns, values)
        value_loss = torch.max(value_loss_original, value_loss_clipped)

        aux_loss = self._compute_aux_loss(tensordict)
        feature_distill_loss, priv_recon_loss, teacher_recon_loss = self._compute_teacher_losses(tensordict)

        distill_feature_w = self._current_distill_feature_w
        distill_priv_w = self._current_distill_priv_w
        loss = (
            policy_loss
            + entropy_loss
            + value_loss
            + (self.critic_aux_w * aux_loss)
            + (distill_feature_w * feature_distill_loss)
            + (distill_priv_w * (priv_recon_loss + teacher_recon_loss))
        )
        explained_var = 1.0 - F.mse_loss(values, b_returns) / b_returns.var().clamp_min(1.0e-6)
        metrics = TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "critic_aux_loss": aux_loss,
                "feature_distill_loss": feature_distill_loss,
                "priv_recon_loss": priv_recon_loss,
                "teacher_priv_recon_loss": teacher_recon_loss,
                "entropy": torch.mean(entropy),
                "explained_var": explained_var,
            },
            [],
        )
        return loss, metrics

    def _step_minibatch(self, tensordict: TensorDict, hidden_state: Optional[torch.Tensor] = None):
        next_hidden = self._encode_features(tensordict, use_rollout_state=False, hidden_state=hidden_state)
        loss, metrics = self._compute_losses(tensordict)

        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()

        if self._encoder_grad_params:
            nn.utils.clip_grad.clip_grad_norm_(self._encoder_grad_params, 5)
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)

        if self.encoder_opt is not None:
            self.encoder_opt.step()
        self.actor_opt.step()
        self.critic_opt.step()

        metrics.set("actor_grad_norm", torch.as_tensor(actor_grad_norm, device=self.device))
        metrics.set("critic_grad_norm", torch.as_tensor(critic_grad_norm, device=self.device))
        metrics.set("distill_feature_w", torch.as_tensor(self._current_distill_feature_w, device=self.device))
        metrics.set("distill_priv_w", torch.as_tensor(self._current_distill_priv_w, device=self.device))
        return metrics, next_hidden

    def _update_sequence(self, tensordict: TensorDict):
        infos = []
        hidden_state = None
        time_steps = int(tensordict.batch_size[0])
        for start in range(0, time_steps, self.bptt_len):
            chunk = tensordict[start : start + self.bptt_len]
            info, hidden_state = self._step_minibatch(chunk, hidden_state=hidden_state)
            infos.append(info)
            if hidden_state is not None:
                hidden_state = hidden_state.detach()
        return infos

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            self._encode_features(next_tensordict, use_rollout_state=False)
            next_values = self.critic(next_tensordict)["state_value"]
        rewards = tensordict[("next", "agents", "reward")]
        dones = tensordict[("next", "terminated")]

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

        self._current_distill_feature_w, self._current_distill_priv_w = self._current_distill_weights()

        infos = []
        for _ in range(self.cfg.ppo_epochs):
            if self.temporal_enable:
                batch_iter = make_sequence_batch(tensordict, self.sequence_num_minibatches)
                for minibatch in batch_iter:
                    infos.extend(self._update_sequence(minibatch))
            else:
                batch_iter = make_batch(tensordict, self.cfg.num_minibatches)
                for minibatch in batch_iter:
                    info, _ = self._step_minibatch(minibatch)
                    infos.append(info)

        if self.teacher_enable and self.curriculum_level >= 4:
            self._l4_distill_iters += 1

        infos = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        out = {k: v.item() for k, v in infos.items()}
        out["distill_l4_iters"] = float(self._l4_distill_iters)
        out["temporal_enable"] = 1.0 if self.temporal_enable else 0.0
        return out


@dataclass
class CurriculumLevel:
    name: str
    success_ge: float
    static_obs_num_total: int = 0
    static_obs_platform_width: float = 0.0
    dynamic_obs_active: int = 0


class SuccessCurriculum:
    def __init__(self, cfg: Dict[str, Any]):
        self.enable = bool(cfg.get("enable", False))
        self.metric = cfg.get("metric", "done_success")  # ✅ done_success
        self.ema_alpha = float(cfg.get("ema_alpha", 0.05))
        self.check_every_iters = int(cfg.get("check_every_iters", 100))  # ✅ 100 iter
        self.patience = int(cfg.get("patience", 3))
        self.cooldown = int(cfg.get("cooldown", 0))
        self.monotonic = bool(cfg.get("monotonic", True))
        self.reset_ema_on_switch = bool(cfg.get("reset_ema_on_switch", True))

        self.levels: List[CurriculumLevel] = []
        for lv in cfg.get("levels", []):
            self.levels.append(
                CurriculumLevel(
                    name=str(lv.get("name", f"L{len(self.levels)}")),
                    success_ge=float(lv.get("success_ge", 0.0)),
                    static_obs_num_total=int(
                        lv.get(
                            "static_obs_num_total",
                            lv.get("static_obs_num_per_grid", lv.get("static_obs_num_per_gird", 0)),
                        )
                    ),
                    static_obs_platform_width=float(lv.get("static_obs_platform_width", 0.0)),
                    dynamic_obs_active=int(lv.get("dynamic_obs_active", 0)),
                )
            )

        self.level = int(cfg.get("start_level", 0))
        self.ema_success = 0.0
        self._pat = 0
        self._cool = 0
        self._last_switch_iter = -1
        self._last_check_idx = -1

    def maybe_update(self, it: int, success_raw: float) -> Dict[str, float]:
        enabled = bool(self.enable and len(self.levels) > 0)
        next_level = self.level + 1
        has_next = next_level < len(self.levels)
        threshold = self.levels[next_level].success_ge if has_next else -1.0
        check_idx = -1
        if enabled and self.check_every_iters > 0:
            check_idx = int(it) // int(self.check_every_iters)
        checks_passed = max(0, check_idx - self._last_check_idx)
        check_hit = checks_passed > 0
        out = {
            "curriculum/ema_success": self.ema_success,
            "curriculum/level": float(self.level),
            "curriculum/switch": 0.0,
            "curriculum/raw_success": float(success_raw),
            "curriculum/enabled": 1.0 if enabled else 0.0,
            "curriculum/check": 1.0 if check_hit else 0.0,
            "curriculum/check_idx": float(check_idx),
            "curriculum/checks_passed": float(checks_passed),
            "curriculum/check_every_iters": float(self.check_every_iters),
            "curriculum/patience": float(self.patience),
            "curriculum/cooldown_left": float(self._cool),
            "curriculum/threshold": float(threshold),
            "curriculum/patience_count": float(self._pat),
            "curriculum/last_switch_iter": float(self._last_switch_iter),
            "curriculum/iters_since_switch": float(0 if self._last_switch_iter < 0 else max(0, int(it) - int(self._last_switch_iter))),
        }
        if not enabled:
            return out

        # EMA update
        if self.ema_success == 0.0:
            self.ema_success = float(success_raw)
        else:
            self.ema_success = (1.0 - self.ema_alpha) * self.ema_success + self.ema_alpha * float(success_raw)
        out["curriculum/ema_success"] = self.ema_success

        if not check_hit:
            return out
        self._last_check_idx = check_idx

        # cooldown
        if self._cool > 0:
            self._cool = max(0, self._cool - checks_passed)
            out["curriculum/cooldown_left"] = float(self._cool)
            if self._cool > 0:
                return out

        # decide next level threshold (use NEXT level's success_ge)
        if not has_next:
            out["curriculum/ema_success"] = self.ema_success
            return out

        threshold = self.levels[next_level].success_ge
        if self.ema_success >= threshold:
            self._pat += checks_passed
        else:
            self._pat = 0
        out["curriculum/patience_count"] = float(self._pat)

        if self._pat >= self.patience:
            # switch!
            self.level = next_level
            self._pat = 0
            self._cool = self.cooldown
            self._last_switch_iter = int(it)
            out["curriculum/switch"] = 1.0
            out["curriculum/level"] = float(self.level)
            out["curriculum/patience_count"] = float(self._pat)
            out["curriculum/cooldown_left"] = float(self._cool)
            out["curriculum/last_switch_iter"] = float(self._last_switch_iter)
            out["curriculum/iters_since_switch"] = 0.0

            if self.reset_ema_on_switch:
                self.ema_success = 0.0
            out["curriculum/ema_success"] = self.ema_success

        return out


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)
    simulation_app = init_simulation_app(cfg)
    run = init_wandb(cfg)
    setproctitle(run.name)
    print(OmegaConf.to_yaml(cfg))

    from resources.envs.isaac_env import IsaacEnv
    from resources.envs.single import Env

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    # curriculum manager
    curr_cfg = {}
    if hasattr(cfg.task, "success_curriculum"):
        curr_cfg = OmegaConf.to_container(cfg.task.success_curriculum, resolve=True)
    curr_mgr = SuccessCurriculum(curr_cfg)

    transforms = [InitTracker()]

    if cfg.task.get("ravel_obs", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation"))
        transforms.append(transform)
    if cfg.task.get("ravel_obs_central", False):
        transform = ravel_composite(base_env.observation_spec, ("agents", "observation_central"))
        transforms.append(transform)
    if (
        cfg.task.get("flatten_intrinsics", True)
        and ("agents", "intrinsics") in base_env.observation_spec.keys(True)
        and isinstance(base_env.observation_spec[("agents", "intrinsics")], CompositeSpec)
    ):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "intrinsics"), start_dim=-1))

    action_transform: str = cfg.task.get("action_transform", None)
    if action_transform is not None:
        if action_transform.startswith("multidiscrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromMultiDiscreteAction(nbins=nbins)
            transforms.append(transform)
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transform = FromDiscreteAction(nbins=nbins)
            transforms.append(transform)
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = PPOPolicy(
        cfg.algo,
        getattr(cfg, "model", None),
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

    resume_checkpoint = cfg.get("resume_checkpoint", None)
    if resume_checkpoint:
        ckpt_path = str(resume_checkpoint)
        if not os.path.isabs(ckpt_path):
            try:
                ckpt_path = to_absolute_path(ckpt_path)
            except Exception:
                ckpt_path = os.path.abspath(ckpt_path)
        ckpt_path = os.path.abspath(ckpt_path)
        checkpoint = torch.load(ckpt_path, map_location=base_env.device)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        incompatible = policy.load_state_dict(checkpoint, strict=False)
        missing = list(getattr(incompatible, "missing_keys", []))
        unexpected = list(getattr(incompatible, "unexpected_keys", []))
        logging.info(
            "Loaded resume checkpoint %s (missing=%d, unexpected=%d)",
            ckpt_path,
            len(missing),
            len(unexpected),
        )
        if missing:
            logging.info("Missing keys (first 16): %s", missing[:16])
        if unexpected:
            logging.info("Unexpected keys (first 16): %s", unexpected[:16])

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    print(f"frames_per_batch {frames_per_batch}")
    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = cfg.get("max_iters", -1)
    # Guardrails: avoid silent no-op training.
    if max_iters == 0:
        logging.warning(
            "[fix] max_iters=0 would make training do zero iterations. "
            "Treating it as unlimited (<=0) and relying on total_frames."
        )
        max_iters = -1
    if total_frames <= 0:
        raise ValueError(
            f"total_frames must be > 0 for training (got {total_frames}). "
            f"Check your config and frames_per_batch={frames_per_batch}."
        )
    eval_interval = cfg.get("eval_interval", -1)
    eval_num_episodes = int(cfg.get("eval_num_episodes", 64))
    eval_max_steps = int(cfg.get("eval_max_steps", base_env.max_episode_length))
    eval_video_max_steps = int(cfg.get("eval_video_max_steps", eval_max_steps))
    save_interval = cfg.get("save_interval", -1)
    record_video = bool(cfg.get("record_video", True))
    eval_seed_mode = str(cfg.get("eval_seed_mode", "random")).lower()
    eval_seed = cfg.get("eval_seed", None)

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0]=="stats"
    ]
    episode_stats = EpisodeStats(stats_keys)
    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    eval_counter = 0
    last_switch_iter_for_entropy_boost = -10**9
    switch_entropy_boost_iters = max(0, int(getattr(cfg.algo, "switch_entropy_boost_iters", 0)))
    switch_entropy_boost_mult = max(1.0, float(getattr(cfg.algo, "switch_entropy_boost_mult", 1.0)))

    def _next_eval_seed() -> Optional[int]:
        nonlocal eval_counter
        if eval_seed_mode == "fixed":
            seed = int(eval_seed) if eval_seed is not None else int(cfg.seed)
        elif eval_seed_mode == "sequence":
            base_seed = int(eval_seed) if eval_seed is not None else int(cfg.seed)
            seed = base_seed + eval_counter
        elif eval_seed_mode == "none":
            seed = None
        else:
            seed = random.randint(0, 2**31 - 1)
        eval_counter += 1
        return seed

    def _find_actor_core_module(module: Optional[nn.Module]) -> Optional[nn.Module]:
        """Recursively find the module that owns actor_std/log_std bounds."""
        if module is None:
            return None
        if hasattr(module, "actor_std"):
            return module
        child = getattr(module, "module", None)
        if isinstance(child, nn.Module):
            found = _find_actor_core_module(child)
            if found is not None:
                return found
        for sub in module.children():
            found = _find_actor_core_module(sub)
            if found is not None:
                return found
        return None

    def _update_done_gap_metrics(info: Dict[str, Any]) -> None:
        train_success = info.get("train/stats.done_success")
        eval_success = info.get("eval/stats.done_success")
        if isinstance(train_success, (int, float)) and isinstance(eval_success, (int, float)):
            info["eval_gap_success"] = float(train_success - eval_success)

        train_height_low = info.get("train/stats.done_height_low")
        eval_height_low = info.get("eval/stats.done_height_low")
        if isinstance(train_height_low, (int, float)) and isinstance(eval_height_low, (int, float)):
            info["eval_gap_done_height_low"] = float(eval_height_low - train_height_low)

    def _reset_actor_std_on_switch() -> Dict[str, float]:
        """On curriculum switch, partially restore exploration by lifting actor log-std."""
        out: Dict[str, float] = {}
        try:
            actor_core = _find_actor_core_module(getattr(policy, "actor", None))
            if actor_core is None or not hasattr(actor_core, "actor_std"):
                out["train/switch_std_reset_applied"] = 0.0
                return out

            raw = actor_core.actor_std.data
            lo = float(getattr(actor_core, "log_std_min", -2.5))
            hi = float(getattr(actor_core, "log_std_max", 0.5))
            if hi < lo:
                lo, hi = hi, lo

            if hi > lo + 1.0e-6:
                cur_log_std = lo + 0.5 * (torch.tanh(raw) + 1.0) * (hi - lo)
            else:
                cur_log_std = raw

            target = float(getattr(cfg.algo, "actor_log_std_init", -0.25))
            target = max(lo, min(hi, target))
            tgt = torch.full_like(cur_log_std, target)
            recover_ratio = float(getattr(cfg.algo, "switch_std_recover_ratio", 0.35))
            recover_ratio = max(0.0, min(1.0, recover_ratio))
            lifted_log_std = cur_log_std + recover_ratio * (tgt - cur_log_std)
            new_log_std = torch.maximum(cur_log_std, lifted_log_std)

            if hi > lo + 1.0e-6:
                t = (2.0 * (new_log_std - lo) / (hi - lo)) - 1.0
                t = t.clamp(-0.999, 0.999)
                new_raw = torch.atanh(t)
            else:
                new_raw = new_log_std

            actor_core.actor_std.data.copy_(new_raw)
            out["train/switch_std_reset_applied"] = 1.0
            out["train/switch_std_recover_ratio"] = float(recover_ratio)
            out["train/switch_log_std_before"] = float(cur_log_std.mean().item())
            out["train/switch_log_std_after"] = float(new_log_std.mean().item())
            out["train/switch_log_std_delta"] = float((new_log_std - cur_log_std).mean().item())
        except Exception:
            out["train/switch_std_reset_applied"] = 0.0
            out["train/switch_std_reset_error"] = 1.0
        return out

    @torch.no_grad()
    def evaluate(
        seed: Optional[int] = None,
        render_video: bool = False,
        exploration_type: Optional[ExplorationType] = ExplorationType.MODE,
        prefix: str = "eval",
    ):

        # Stats-only eval does not need viewport rendering. For headless runs,
        # forcing render here can wake up the Vulkan viewport path and crash the
        # process with device-lost/pagefault errors.
        base_env.enable_render(bool(render_video))
        base_env.eval()
        env.eval()
        if seed is not None:
            env.set_seed(int(seed))
        policy.reset_rollout_state()

        render_callback = None

        if render_video:
            render_callback = RenderCallback(interval=2)

        # Streamed eval to avoid storing full rollouts on GPU.
        stats_keys = [
            k for k in base_env.observation_spec.keys(True, True)
            if isinstance(k, tuple) and k[0] == "stats"
        ]
        episode_stats = EpisodeStats(stats_keys)
        td = env.reset()

        ctx = set_exploration_type(exploration_type) if exploration_type is not None else nullcontext()
        with ctx:
            max_steps = eval_video_max_steps if render_video else eval_max_steps
            max_steps = max(1, int(max_steps))
            target_episodes = max(1, int(eval_num_episodes))
            for _ in range(max_steps):
                if render_video and render_callback:
                    render_callback(env)
                td = policy(td)
                td = env.step(td)
                episode_stats.add(td)
                # advance to next step
                td = td["next"]
                if len(episode_stats) >= target_episodes:
                    break

        base_env.enable_render(not cfg.headless)
        env.reset()
        policy.reset_rollout_state()

        if len(episode_stats) == 0:
            return {f"{prefix}/episode_samples": 0.0}

        traj_stats = episode_stats.pop()
        if len(traj_stats) > target_episodes:
            traj_stats = traj_stats[:target_episodes]
        # NOTE: EpisodeStats.pop() returns a TensorDict that may contain nested
        # TensorDict values (e.g., {"stats": TensorDict(...)}). Using
        # `.items(True, True)` flattens it into leaf tensors, consistent with
        # the training logger below.
        traj_stats = traj_stats.cpu()
        stats_mean = {k: v.float().mean().item() for k, v in traj_stats.items(True, True)}

        info = {f"{prefix}/episode_samples": float(len(traj_stats))}
        if seed is not None:
            info[f"{prefix}/seed"] = int(seed)
        skip_substrings = ("curriculum",)
        skip_keys = {
            "stats.truncated",
            "stats.terminated",
            "stats.temporal_reset_rate",
            "stats.done_any",
        }
        avg_speed_val = stats_mean.get(("stats", "avg_speed"))
        if avg_speed_val is not None:
            info[f"{prefix}/avg_speed"] = float(avg_speed_val)

        for k, v in stats_mean.items():
            # k is usually a tuple like ("stats", "done_success")
            if isinstance(k, tuple):
                key_str = ".".join(map(str, k))
            else:
                key_str = str(k)
            if key_str in skip_keys or any(s in key_str for s in skip_substrings):
                continue
            info[f"{prefix}/{key_str}"] = float(v)

        if render_video and render_callback:
            # log video
            info["recording"] = wandb.Video(
                render_callback.get_video_array(axes="t c h w"),
                fps=0.5 / (cfg.sim.dt * cfg.sim.substeps),
                format="mp4"
            )

        return info
    
    # NOTE: In some forks, training used `for i in range(max_iters)` which
    # makes the loop empty when max_iters == -1. We treat max_iters <= 0 as
    # "no iteration limit" and iterate the collector until total_frames.
    iterator = itertools.islice(collector, max_iters) if max_iters > 0 else iter(collector)

    def _log_scalar(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    pbar = tqdm(iterator)
    env.train()
    for i, data in enumerate(pbar):
        info = {"iter": i, "env_frames": collector._frames, "rollout_fps": collector._fps}
        log_ready = False
        td = data.to_tensordict()
        episode_stats.add(data.to_tensordict())

        # Optional: log batch-level stats more frequently to keep W&B curves fresh
        log_interval_iters = int(cfg.task.get("log_interval_iters", 0))
        if log_interval_iters > 0 and (i % log_interval_iters == 0):
            try:
                batch_stats = td["next"]["stats"]
                batch_mean = {k: v.mean().item() for k, v in batch_stats.items(True, True)}
                for k, v in batch_mean.items():
                    key_str = ".".join(k)
                    # keep batch logs lightweight and avoid curriculum/meta noise
                    if key_str.startswith("stats.done_ratio_"):
                        continue
                    if key_str.startswith("stats.curriculum_"):
                        continue
                    info[f"train_batch/{key_str}"] = v
            except Exception:
                # do not fail training if batch stats are missing
                pass

        # Entropy schedule (decoupled from level by default):
        # - by_switch_age: decay within each level using iterations since last switch.
        # - by_level: legacy behavior coupled to unlocked curriculum level.
        try:
            ent_start = float(getattr(cfg.algo, "entropy_coef", 0.01))
            ent_mid = float(getattr(cfg.algo, "entropy_coef_mid"))
            ent_end = float(getattr(cfg.algo, "entropy_coef_end"))
            f1 = float(getattr(cfg.algo, "entropy_anneal_frac1", 0.3))
            f2 = float(getattr(cfg.algo, "entropy_anneal_frac2", 0.7))
            mode = str(getattr(cfg.algo, "entropy_schedule_mode", "by_switch_age")).lower()
            if mode in ("by_level", "level"):
                max_level = 0
                if hasattr(cfg.task, "success_curriculum") and hasattr(cfg.task.success_curriculum, "levels"):
                    try:
                        max_level = max(0, len(cfg.task.success_curriculum.levels) - 1)
                    except Exception:
                        max_level = 0
                unlocked_level = 0
                try:
                    unlocked_level = int(getattr(curr_mgr, "level", 0))
                except Exception:
                    unlocked_level = 0
                if max_level <= 0:
                    progress = 0.0
                else:
                    progress = float(max(0, min(unlocked_level, max_level))) / float(max_level)
            else:
                decay_iters = int(getattr(cfg.algo, "entropy_decay_iters_per_level", 600))
                decay_iters = max(1, decay_iters)
                last_switch_iter = int(getattr(curr_mgr, "_last_switch_iter", -1))
                if last_switch_iter < 0:
                    age = int(i)
                else:
                    age = max(0, int(i) - last_switch_iter)
                progress = min(1.0, float(age) / float(decay_iters))
                info["train/entropy_age_iters"] = float(age)
                info["train/entropy_schedule_progress"] = float(progress)
            if progress <= f1:
                ent_coef = ent_start
            elif progress <= f2:
                ent_coef = ent_start + (ent_mid - ent_start) * ((progress - f1) / max(1e-6, (f2 - f1)))
            else:
                ent_coef = ent_mid + (ent_end - ent_mid) * ((progress - f2) / max(1e-6, (1.0 - f2)))
            switch_age = int(i) - int(last_switch_iter_for_entropy_boost)
            if switch_entropy_boost_iters > 0 and switch_age >= 0 and switch_age < switch_entropy_boost_iters:
                fade = 1.0 - (float(switch_age) / float(max(1, switch_entropy_boost_iters)))
                boost = 1.0 + (switch_entropy_boost_mult - 1.0) * fade
                ent_coef *= boost
                info["train/entropy_switch_boost"] = float(boost)
                info["train/entropy_switch_age_iters"] = float(switch_age)
            else:
                info["train/entropy_switch_boost"] = 1.0
            policy.entropy_coef = float(ent_coef)
            info["train/entropy_coef"] = float(ent_coef)
        except Exception:
            pass

        if len(episode_stats) >= cfg.task.env.num_envs:
            log_ready = True
            info.update(
                {
                    "frames_per_batch": frames_per_batch,
                    "train/episode_samples": len(episode_stats),
                }
            )

            stats_td = episode_stats.pop()
            stats_mean = {k: v.mean().item() for k, v in stats_td.items(True, True)}

            # ✅ success_raw from done_success
            metric = str(getattr(curr_mgr, "metric", "done_success") or "done_success")
            if metric.startswith("stats."):
                metric = metric.split("stats.", 1)[1]
            success_raw = float(
                stats_mean.get(("stats", metric), stats_mean.get(("stats", "done_success"), 0.0))
            )

            # curriculum update + log curves
            cinfo = curr_mgr.maybe_update(i, success_raw)
            info.update(cinfo)

            # ✅ if switched, request level + reset to apply immediately
            if cinfo.get("curriculum/switch", 0.0) > 0.5:
                try:
                    info.update(_reset_actor_std_on_switch())
                    last_switch_iter_for_entropy_boost = int(i)
                    info["train/switch_entropy_boost_applied"] = 1.0 if switch_entropy_boost_iters > 0 else 0.0
                    base_env.request_curriculum_level(curr_mgr.level)
                    env.reset()  # apply now
                    policy.reset_rollout_state()
                except Exception as e:
                    print(f"[Curriculum] switch failed: {e}")

            # log stats (filter duplicates)
            avg_speed_val = stats_mean.get(("stats", "avg_speed"))
            if avg_speed_val is not None:
                info["train/avg_speed"] = float(avg_speed_val)

            for k, v in stats_mean.items():
                key_str = ".".join(k)
                # 过滤 done_ratio_*（和 done_*重复）
                if key_str.startswith("stats.done_ratio_"):
                    continue
                if key_str == "stats.curriculum_static_obs_num_per_grid":
                    info["curriculum/static_obs_num_total"] = v
                    continue
                if key_str == "stats.curriculum_dobs_active":
                    info["curriculum/dobs_active"] = v
                    continue
                if key_str == "stats.curriculum_level":
                    info["curriculum/level_env"] = v
                    continue
                if key_str == "stats.curriculum_reset":
                    info["curriculum/reset_env"] = v
                    continue
                info[f"train/{key_str}"] = v

        policy.set_curriculum_level(curr_mgr.level)
        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0 and i != 0:
            logging.info(f"Eval at {collector._frames} steps.") 
            try:
                info.update(
                    evaluate(
                        seed=_next_eval_seed(),
                        render_video=False,
                        exploration_type=ExplorationType.MODE,
                        prefix="eval",
                    )
                )
                _update_done_gap_metrics(info)
            except Exception as e:
                logging.exception(f"Eval failed (skipping this eval): {e}")
            env.train()
            base_env.train()

        if save_interval > 0 and i % save_interval == 0 and i != 0:
            try:
                ckpt_path = os.path.join(run.dir, f"checkpoint_{collector._frames}.pt")
                torch.save(policy.state_dict(), ckpt_path)
                logging.info(f"Saved checkpoint to {str(ckpt_path)}")
            except AttributeError:
                logging.warning(f"Policy {policy} does not implement `.state_dict()`")

        run.log(info)
        if log_ready:
            _update_done_gap_metrics(info)
            print(OmegaConf.to_yaml({k: v for k, v in info.items() if _log_scalar(v)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})


    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    try:
        info.update(
            evaluate(
                seed=_next_eval_seed(),
                exploration_type=ExplorationType.MODE,
                prefix="eval",
            )
        )
        _update_done_gap_metrics(info)
    except Exception as e:
        logging.exception(f"Final eval failed: {e}")
    run.log(info)

    try:
        ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
        torch.save(policy.state_dict(), ckpt_path)

        model_artifact = wandb.Artifact(
            f"{cfg.task.name}-{cfg.algo.name.lower()}",
            type="model",
            description=f"{cfg.task.name}-{cfg.algo.name.lower()}",
            metadata=dict(cfg))

        model_artifact.add_file(ckpt_path)
        wandb.save(ckpt_path)
        run.log_artifact(model_artifact)

        logging.info(f"Saved checkpoint to {str(ckpt_path)}")
    except AttributeError:
        logging.warning(f"Policy {policy} does not implement `.state_dict()`")

    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()
