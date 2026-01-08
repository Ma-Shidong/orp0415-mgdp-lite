import gc
import logging
import os

import hydra
import torch
import wandb

# import resources.envs.single  # noqa: F401  # 触发 Env 注册

from hydra.utils import to_absolute_path
from omegaconf import OmegaConf
from setproctitle import setproctitle
from tqdm import tqdm

import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential
from torchrl.data import CompositeSpec, TensorSpec
from torchrl.envs.transforms import CatTensors, Compose, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import ProbabilisticActor

from resources import init_simulation_app
from resources.learning.ppo.ppo import (
    Actor,
    GAE,
    IndependentNormal,
    PPOConfig,
    ValueNorm1,
    make_batch,
    make_mlp,
)
from resources.utils.torchrl import EpisodeStats, RenderCallback, SyncDataCollector
from resources.utils.torchrl.transforms import FromDiscreteAction, FromMultiDiscreteAction, ravel_composite
from resources.utils.wandb import init_wandb


class BatchConv2dWrapper(nn.Module):
    """Make a Conv2d-based CNN accept leading batch dims like [E,T,C,H,W].

    It flattens leading dims -> [N,C,H,W], runs cnn, then unflattens back.
    """

    def __init__(self, cnn: nn.Module):
        super().__init__()
        self.cnn = cnn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim < 4:
            raise RuntimeError(f"[BatchConv2dWrapper] Expected >=4D, got {tuple(x.shape)}")
        leading = x.shape[:-3]
        x = x.reshape(-1, *x.shape[-3:])  # [N, C, H, W]
        y = self.cnn(x)                   # [N, F]
        return y.reshape(*leading, -1)    # [..., F]


class PPOPolicy(TensorDictModuleBase):

    def __init__(
        self,
        cfg: PPOConfig,
        observation_spec: CompositeSpec,
        action_spec: CompositeSpec,
        reward_spec: TensorSpec,
        device,
    ):
        super().__init__()
        self.cfg = cfg
        self.device = device

        self.entropy_coef = 0.001
        self.clip_param = 0.1
        self.critic_loss_fn = nn.HuberLoss(delta=10)

        self.n_agents, self.action_dim = action_spec.shape[-2:]
        self.gae = GAE(0.99, 0.95)

        fake_input = observation_spec.zero()

        cnn = nn.Sequential(
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
        mlp = make_mlp([256, 256])

        self.encoder = TensorDictSequential(
            TensorDictModule(
                BatchConv2dWrapper(cnn),
                [("agents", "observation", "lidar")],
                ["_cnn_feature"],
            ),
            CatTensors(["_cnn_feature", ("agents", "observation", "state")], "_feature", del_keys=False),
            TensorDictModule(mlp, ["_feature"], ["_feature"]),
        ).to(self.device)

        self.actor = ProbabilisticActor(
            TensorDictModule(Actor(self.action_dim), ["_feature"], ["loc", "scale"]),
            in_keys=["loc", "scale"],
            out_keys=[("agents", "action")],
            distribution_class=IndependentNormal,
            return_log_prob=True,
        ).to(self.device)

        self.critic = TensorDictModule(nn.LazyLinear(1), ["_feature"], ["state_value"]).to(self.device)

        # materialize lazy modules
        self.encoder(fake_input)
        self.actor(fake_input)
        self.critic(fake_input)

        def init_(module):
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, 0.01)
                nn.init.constant_(module.bias, 0.0)

        self.actor.apply(init_)
        self.critic.apply(init_)

        lr = float(getattr(cfg, "lr", 5e-4))
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
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
            self.encoder(next_tensordict)
            next_values = self.critic(next_tensordict)["state_value"]

        rewards = tensordict[("next", "agents", "reward")]     # [..., 1]
        terminated = tensordict[("next", "terminated")]        # [..., 1]
        truncated = tensordict[("next", "truncated")]          # [..., 1]
        dones = (terminated | truncated)

        values = self.value_norm.denormalize(tensordict["state_value"])
        next_values = self.value_norm.denormalize(next_values)

        # Ensure GAE sees [E,T,...] (your GAE uses shape[1] as T)
        if rewards.ndim >= 2 and rewards.shape[0] < rewards.shape[1]:
            rewards = rewards.transpose(0, 1)
            dones = dones.transpose(0, 1)
            values = values.transpose(0, 1)
            next_values = next_values.transpose(0, 1)

            adv, ret = self.gae(rewards, dones, values, next_values)

            adv = adv.transpose(0, 1)
            ret = ret.transpose(0, 1)
        else:
            adv, ret = self.gae(rewards, dones, values, next_values)

        adv = (adv - adv.mean()) / adv.std().clip(1e-7)

        self.value_norm.update(ret)
        ret = self.value_norm.normalize(ret)

        tensordict.set("adv", adv)
        tensordict.set("ret", ret)

        infos = []
        for _ in range(self.cfg.ppo_epochs):
            for minibatch in make_batch(tensordict, self.cfg.num_minibatches):
                infos.append(self._update(minibatch))

        if not infos:
            return {}

        infos: TensorDict = torch.stack(infos).to_tensordict().apply(torch.mean, batch_size=[])
        return {k: float(v.detach().cpu()) for k, v in infos.items()}

    def _update(self, tensordict: TensorDict):
        self.encoder(tensordict)
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
        value_loss = torch.max(self.critic_loss_fn(b_returns, values), self.critic_loss_fn(b_returns, values_clipped))

        loss = policy_loss + entropy_loss + value_loss
        self.encoder_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        loss.backward()

        actor_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 5.0)
        critic_grad_norm = nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 5.0)

        self.encoder_opt.step()
        self.actor_opt.step()
        self.critic_opt.step()

        explained_var = 1.0 - F.mse_loss(values, b_returns) / b_returns.var().clamp_min(1e-8)

        return TensorDict(
            {
                "policy_loss": policy_loss,
                "value_loss": value_loss,
                "entropy": entropy,
                "actor_grad_norm": actor_grad_norm,
                "critic_grad_norm": critic_grad_norm,
                "explained_var": explained_var,
            },
            [],
        )


def _infer_time_dim(x: torch.Tensor, num_envs: int) -> int:
    """Return time dimension index for tensors shaped [T,E,...] or [E,T,...]."""
    if x.ndim < 2:
        return 0
    if x.shape[0] == num_envs:
        return 1  # [E,T,...]
    return 0      # [T,E,...] (common)


def _iter_success_from_stats(td: TensorDict, num_envs: int, key=("next", "stats", "flight_success")):
    x = td.get(key, None)
    if x is None:
        return None
    x = x.squeeze(-1)
    tdim = _infer_time_dim(x, num_envs)
    # success per env if any timestep succeeded in this rollout chunk
    success_env = x.max(dim=tdim).values
    return float(success_env.float().mean().item())


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(cfg):
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.resolve(cfg)
    OmegaConf.set_struct(cfg, False)

    simulation_app = init_simulation_app(cfg)

    # keep torch device consistent with cfg.sim.device
    if torch.cuda.is_available() and str(cfg.sim.device).startswith("cuda:"):
        try:
            torch.cuda.set_device(int(str(cfg.sim.device).split(":")[1]))
        except Exception as e:
            print("[WARN] torch.cuda.set_device failed:", e)

    run = init_wandb(cfg)
    setproctitle(run.name)

    print(OmegaConf.to_yaml(cfg))

    from resources.envs.isaac_env import IsaacEnv  # noqa: WPS433
    from resources.envs.single import Env  # noqa: F401  # 触发 Env 子类注册进 REGISTRY

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=cfg.headless)

    transforms = [InitTracker()]

    if cfg.task.get("ravel_obs", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation")))
    if cfg.task.get("ravel_obs_central", False):
        transforms.append(ravel_composite(base_env.observation_spec, ("agents", "observation_central")))
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
            transforms.append(FromMultiDiscreteAction(nbins=nbins))
        elif action_transform.startswith("discrete"):
            nbins = int(action_transform.split(":")[1])
            transforms.append(FromDiscreteAction(nbins=nbins))
        else:
            raise NotImplementedError(f"Unknown action transform: {action_transform}")

    env = TransformedEnv(base_env, Compose(*transforms)).train()
    env.set_seed(cfg.seed)

    policy = PPOPolicy(
        cfg.algo,
        env.observation_spec,
        env.action_spec,
        env.reward_spec,
        device=base_env.device,
    )

    resume_ckpt = cfg.get("resume_checkpoint")
    if resume_ckpt:
        ckpt_path = to_absolute_path(resume_ckpt)
        state_dict = torch.load(ckpt_path, map_location=base_env.device)
        missing, unexpected = policy.load_state_dict(state_dict, strict=False)
        logging.info(f"Resumed policy from {ckpt_path} (missing: {missing}, unexpected: {unexpected})")

    frames_per_batch = env.num_envs * int(cfg.algo.train_every)
    print(f"frames_per_batch {frames_per_batch}")

    total_frames = cfg.get("total_frames", -1) // frames_per_batch * frames_per_batch
    max_iters = int(cfg.get("max_iters", -1))
    eval_interval = int(cfg.get("eval_interval", -1))
    save_interval = int(cfg.get("save_interval", -1))
    log_interval = int(cfg.get("log_interval", 100))  # default: log every 100 iters to reduce overhead
    print_interval = int(cfg.get("print_interval", 50))

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    stats_keys = [
        k for k in base_env.observation_spec.keys(True, True)
        if isinstance(k, tuple) and k[0] == "stats"
    ]
    episode_stats = EpisodeStats(stats_keys)

    def _alias_reward_metrics(info: dict, prefix: str) -> None:
        """Add a small set of reward-related keys for W&B dashboards.

        We keep this intentionally minimal to avoid cluttering the run with too many metrics.
        Expected source keys format:
          - {prefix}/stats.<stat_key>    (e.g., train/stats.return, eval/stats.reward_goal)
        """
        def _copy(stat_key: str, name: str) -> None:
            src = f"{prefix}/stats.{stat_key}"
            if src in info and isinstance(info[src], (float, int)):
                info[f"{prefix}/reward/{name}"] = info[src]

        # Total episode reward
        _copy("return", "return")

        # Key reward components (most informative)
        _copy("reward_goal", "goal")
        _copy("reward_velocity", "velocity")
        _copy("reward_safety", "safety")

        # Task progress
        _copy("goal_gate", "goalgate")

        # Motion state summary
        _copy("avg_speed", "avg_speed")
        _copy("avg_acc", "avg_acc")

    collector = SyncDataCollector(
        env,
        policy=policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=cfg.sim.device,
        return_same_td=True,
    )

    @torch.no_grad()
    def evaluate(iter_idx: int, seed: int = 0, render_video: bool = False,
                 exploration_type: ExplorationType = ExplorationType.MODE):
        base_env.enable_render(render_video)
        base_env.eval()
        env.eval()
        env.set_seed(seed)

        render_callback = RenderCallback(interval=2) if render_video else None

        eval_max_steps = int(cfg.get("eval_max_steps", 512))
        if render_video:
            eval_max_steps = int(cfg.get("eval_video_max_steps", 128))

        with set_exploration_type(exploration_type):
            rollout_td = env.rollout(
                max_steps=eval_max_steps,
                policy=policy,
                callback=render_callback,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=True,
            )

        done = rollout_td.get(("next", "done"))
        if done is not None and done.shape[-1] == 1:
            done = done.squeeze(-1)
        done = done.cpu()

        stats_td = rollout_td.get(("next", "stats")).cpu()
        del rollout_td

        if render_video:
            gc.collect()

        done_long = done.long()
        has_done = done_long.any(dim=0)
        first_done = torch.argmax(done_long, dim=0)
        first_done = torch.where(has_done, first_done, torch.full_like(first_done, done.shape[0] - 1)).long()

        def take_first_episode(tensor: torch.Tensor):
            idx = first_done.view(1, -1, *([1] * (tensor.ndim - 2)))
            out = torch.take_along_dim(tensor, idx, dim=0)
            return out.squeeze(0)

        traj_stats = {k: take_first_episode(v) for k, v in stats_td.items()}

        info = {"iter": iter_idx}
        info.update({f"eval/stats.{k}": torch.mean(v.float()).item() for k, v in traj_stats.items()})
        _alias_reward_metrics(info, "eval")

        if render_video and render_callback:
            vid = render_callback.get_video_array(axes="t c h w")
            info["recording"] = wandb.Video(vid, fps=0.5 / (cfg.sim.dt * cfg.sim.substeps), format="mp4")
            del vid
            if hasattr(render_callback, "clear"):
                render_callback.clear()
            elif hasattr(render_callback, "reset"):
                render_callback.reset()

        del stats_td, traj_stats, done
        if render_video:
            gc.collect()

        base_env.enable_render(not cfg.headless)
        env.reset()
        env.train()
        base_env.train()
        return info

    pbar = tqdm(collector)
    env.train()

    for i, data in enumerate(pbar):
        td = data if isinstance(data, TensorDict) else data.to_tensordict()

        info = {
            "iter": i,
            "env_frames": int(collector._frames),
            "frames_per_batch": int(frames_per_batch),
            "rollout_fps": float(collector._fps),
        }

        # ---- per-iter success (avoid episode accumulator ambiguities) ----
        iter_succ = _iter_success_from_stats(td, num_envs=base_env.num_envs)
        if iter_succ is not None:
            info["train/iter_flight_success"] = iter_succ

        # ---- episode-level stats (first-episode-per-env in this buffer) ----
        episode_stats.add(td)
        if len(episode_stats) >= base_env.num_envs:
            stats = {
                "train/" + (".".join(k) if isinstance(k, tuple) else str(k)): torch.mean(v.float()).item()
                for k, v in episode_stats.pop().items(True, True)
            }
            info.update(stats)
            _alias_reward_metrics(info, "train")

        info.update(policy.train_op(td))

        if eval_interval > 0 and i % eval_interval == 0 and i != 0:
            logging.info(f"Eval at iter={i}, env_frames={collector._frames}.")
            info.update(evaluate(iter_idx=i, render_video=False))

        if save_interval > 0 and i % save_interval == 0 and i != 0:
            ckpt_path = os.path.join(run.dir, f"checkpoint_iter_{i}_frames_{collector._frames}.pt")
            torch.save(policy.state_dict(), ckpt_path)
            logging.info(f"Saved checkpoint to {ckpt_path}")
            info.update(evaluate(iter_idx=i, render_video=True))

        if log_interval > 0 and (i % log_interval == 0):
            # reduce W&B payload: drop redundant / noisy keys
            drop_exact = {"iter", "env_frames", "frames_per_batch"}
            drop_substr = ("truncated", "timeout", "nan")
            log_info = {}
            for k, v in info.items():
                if k in drop_exact:
                    continue
                if any(s in str(k) for s in drop_substr):
                    continue
                log_info[k] = v
            run.log(log_info, step=i)

        if print_interval > 0 and (i % print_interval == 0):
            printable = {k: v for k, v in info.items() if isinstance(v, (float, int))}
            print(OmegaConf.to_yaml(printable))

        pbar.set_postfix({"rollout_fps": collector._fps, "frames": collector._frames})

        if max_iters > 0 and i >= max_iters - 1:
            break

    logging.info(f"Final Eval at iter={i}, env_frames={collector._frames}.")
    final_info = {"iter": i, "env_frames": int(collector._frames)}
    final_info.update(evaluate(iter_idx=i))
    run.log(final_info, step=i)

    ckpt_path = os.path.join(run.dir, "checkpoint_final.pt")
    torch.save(policy.state_dict(), ckpt_path)
    wandb.save(ckpt_path)
    wandb.finish()

    simulation_app.close()


if __name__ == "__main__":
    main()