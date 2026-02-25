import logging
import os
import itertools

import hydra
import torch
import wandb

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
from torch.func import vmap
from einops.layers.torch import Rearrange
from resources.learning.ppo.ppo import PPOConfig, make_mlp, make_batch, Actor, IndependentNormal, GAE, ValueNorm1
from tensordict import TensorDict
from tensordict.nn import TensorDictSequential, TensorDictModule, TensorDictModuleBase
from torchrl.envs.transforms import CatTensors
from torchrl.modules import ProbabilisticActor

from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# Use TF32 on Ampere+ for faster matmul with minimal accuracy impact.
torch.set_float32_matmul_precision("high")
torch.backends.cuda.matmul.allow_tf32 = True

class PPOPolicy(TensorDictModuleBase):

    def __init__(self, cfg: PPOConfig, observation_spec: CompositeSpec, action_spec: CompositeSpec, reward_spec: TensorSpec, device):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.03 # 熵正则系数，鼓励策略保持随机性（探索），避免过早“变得很确定”
        self.clip_param = 0.25 # 限制策略更新幅度，越大越敢于更新
        self.critic_loss_fn = nn.HuberLoss(delta=10)
        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.995, 0.95)

        fake_input = observation_spec.zero()

        cnn = nn.Sequential(
            nn.LazyConv2d(out_channels=4, kernel_size=[5, 3], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 1], padding=[2, 1]), nn.ELU(),
            nn.LazyConv2d(out_channels=16, kernel_size=[5, 3], stride=[2, 2], padding=[2, 1]), nn.ELU(),
            Rearrange("n c w h -> n (c w h)"),
            nn.LazyLinear(128), nn.LayerNorm(128)
        )
        mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(cnn, [("agents", "observation", "lidar")], ["_cnn_feature"]), 
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True
        ).to(self.device)

        self.critic = TensorDictModule(
            nn.LazyLinear(1), ["_feature"], ["state_value"]
        ).to(self.device)

        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.)

        self.actor.apply(init_)
        self.critic.apply(init_)
        # 学习率
        # self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=3e-4)
        # self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        # self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=1e-4)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=2e-4)
        
        self.value_norm = ValueNorm1(1).to(self.device)

    def __call__(self, tensordict: TensorDict):
        self.encoder(tensordict)
        self.actor(tensordict)
        self.critic(tensordict)
        tensordict.exclude("loc", "scale", "_feature", inplace=True)
        return tensordict

    def train_op(self, tensordict: TensorDict):
        next_tensordict = tensordict["next"]
        with torch.no_grad():
            next_tensordict = vmap(self.encoder)(next_tensordict)
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

        infos = []
        for epoch in range(self.cfg.ppo_epochs):
            batch = make_batch(tensordict, self.cfg.num_minibatches)
            for minibatch in batch:
                infos.append(self._update(minibatch))

        infos: TensorDict = torch.stack(infos).to_tensordict()
        infos = infos.apply(torch.mean, batch_size=[])
        return {k: v.item() for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
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
        self.encoder_opt.zero_grad()
        self.actor_opt.zero_grad()
        self.critic_opt.zero_grad()
        loss.backward()
        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5)
        self.encoder_opt.step()
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
        self._last_switch_iter = -10**9
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
            out["curriculum/switch"] = 1.0
            out["curriculum/level"] = float(self.level)
            out["curriculum/patience_count"] = float(self._pat)
            out["curriculum/cooldown_left"] = float(self._cool)

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
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device
    )

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
    save_interval = cfg.get("save_interval", -1)
    record_video = bool(cfg.get("record_video", True))

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

    @torch.no_grad()
    def evaluate(
        seed: int = 0,
        render_video: bool = False,
        exploration_type: ExplorationType = ExplorationType.MODE,
        prefix: str = "eval",
    ):

        base_env.enable_render(True)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

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

        with set_exploration_type(exploration_type):
            for _ in range(base_env.max_episode_length):
                if render_video and render_callback:
                    render_callback(env)
                td = policy(td)
                td = env.step(td)
                episode_stats.add(td)
                # advance to next step
                td = td["next"]
                if len(episode_stats) >= env.num_envs:
                    break

        base_env.enable_render(not cfg.headless)
        env.reset()

        traj_stats = episode_stats.pop()
        # NOTE: EpisodeStats.pop() returns a TensorDict that may contain nested
        # TensorDict values (e.g., {"stats": TensorDict(...)}). Using
        # `.items(True, True)` flattens it into leaf tensors, consistent with
        # the training logger below.
        traj_stats = traj_stats.cpu()
        stats_mean = {k: v.float().mean().item() for k, v in traj_stats.items(True, True)}

        info = {}
        for k, v in stats_mean.items():
            # k is usually a tuple like ("stats", "done_success")
            if isinstance(k, tuple):
                key_str = ".".join(map(str, k))
            else:
                key_str = str(k)
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

    pbar = tqdm(iterator)
    env.train()
    for i, data in enumerate(pbar):
        info = {"env_frames": collector._frames, "rollout_fps": collector._fps}
        episode_stats.add(data.to_tensordict())

        if len(episode_stats) >= cfg.task.env.num_envs:
            info = {
                "iter": i,
                "env_frames": collector._frames,
                "frames_per_batch": frames_per_batch,
                "rollout_fps": collector._fps,
                "train/episode_samples": len(episode_stats),
            }

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
                    base_env.request_curriculum_level(curr_mgr.level)
                    env.reset()  # apply now
                except Exception as e:
                    print(f"[Curriculum] switch failed: {e}")

            # log stats (filter duplicates)
            for k, v in stats_mean.items():
                key_str = ".".join(k)
                # 过滤 done_ratio_*（和 done_*重复）
                if key_str.startswith("stats.done_ratio_"):
                    continue
                info[f"train/{key_str}"] = v

        info.update(policy.train_op(data.to_tensordict()))

        if eval_interval > 0 and i % eval_interval == 0 and i != 0:
            logging.info(f"Eval at {collector._frames} steps.") 
            try:
                # Eval with deterministic action selection (distribution mode / mean)
                info.update(evaluate(render_video=False, exploration_type=ExplorationType.MODE, prefix="eval_mode"))
                # Eval with stochastic action sampling (helps detect multi-modal policies)
                _sample_expl = getattr(ExplorationType, "RANDOM", None) or getattr(ExplorationType, "SAMPLE", None)
                if _sample_expl is not None:
                    info.update(evaluate(render_video=False, exploration_type=_sample_expl, prefix="eval_sample"))
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
            
            if record_video:
                logging.info(f"Eval and render video at {collector._frames} steps.")
                try:
                    info.update(evaluate(render_video=True, exploration_type=ExplorationType.MODE, prefix="eval_mode"))
                    _sample_expl = getattr(ExplorationType, "RANDOM", None) or getattr(ExplorationType, "SAMPLE", None)
                    if _sample_expl is not None:
                        info.update(evaluate(render_video=False, exploration_type=_sample_expl, prefix="eval_sample"))
                except Exception as e:
                    logging.exception(f"Video eval failed (skipping video): {e}")
            else:
                logging.info(f"Eval at {collector._frames} steps (video disabled).")
                try:
                    info.update(evaluate(render_video=False, exploration_type=ExplorationType.MODE, prefix="eval_mode"))
                    _sample_expl = getattr(ExplorationType, "RANDOM", None) or getattr(ExplorationType, "SAMPLE", None)
                    if _sample_expl is not None:
                        info.update(evaluate(render_video=False, exploration_type=_sample_expl, prefix="eval_sample"))
                except Exception as e:
                    logging.exception(f"Eval failed (skipping this eval): {e}")

        run.log(info)
        print(OmegaConf.to_yaml({k: v for k, v in info.items() if isinstance(v, float)}))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})


    logging.info(f"Final Eval at {collector._frames} steps.")
    info = {"env_frames": collector._frames}
    try:
        info.update(evaluate(exploration_type=ExplorationType.MODE, prefix="eval_mode"))
        _sample_expl = getattr(ExplorationType, "RANDOM", None) or getattr(ExplorationType, "SAMPLE", None)
        if _sample_expl is not None:
            info.update(evaluate(exploration_type=_sample_expl, prefix="eval_sample"))
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
