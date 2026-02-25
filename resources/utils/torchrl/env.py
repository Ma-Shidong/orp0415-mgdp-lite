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
import numpy as np
import einops
from tqdm import tqdm
from typing import Optional, Sequence, Any, Dict

from dataclasses import dataclass
from torchrl.envs import EnvBase
from torchrl.data import TensorSpec, CompositeSpec
from tensordict import TensorDictBase



@dataclass
class AgentSpec:
    name: str
    n: int
    observation_key: Optional[str] = "observation"
    action_key: Optional[str] = None
    state_key: Optional[str] = None
    reward_key: Optional[str] = None
    done_key: Optional[str] = None

    _env: Optional[EnvBase] = None

    @property
    def observation_spec(self) -> TensorSpec:
        return self._env.observation_spec[self.observation_key]

    @property
    def action_spec(self) -> TensorSpec:
        if self.action_key is None:
            return self._env.action_spec
        try:
            return self._env.input_spec["full_action_spec"][self.action_key]
        except:
            return self._env.action_spec[self.action_key]

    @property
    def state_spec(self) -> TensorSpec:
        if self.state_key is None:
            raise ValueError()
        return self._env.observation_spec[self.state_key]

    @property
    def reward_spec(self) -> TensorSpec:
        if self.reward_key is None:
            return self._env.reward_spec
        try:
            return self._env.output_spec["full_reward_spec"][self.reward_key]
        except:
            return self._env.reward_spec[self.reward_key]

    @property
    def done_spec(self) -> TensorSpec:
        if self.done_key is None:
            return self._env.done_spec
        try:
            return self._env.output_spec["full_done_spec"][self.done_key]
        except:
            return self._env.done_spec[self.done_key]


class RenderCallback:

    def __init__(self, interval: int=2):
        self.interval = interval
        self.frames = []
        self.i = 0
        self.t = tqdm(desc="Rendering")

    def __call__(self, env, *args):
        if self.i % self.interval == 0:
            frame = env.render(mode="rgb_array")
            self.frames.append(frame)
            self.t.update(self.interval)
        self.i += 1
        return self.i

    def get_video_array(self, axes: str = "t c h w"):
        return einops.rearrange(np.stack(self.frames), "t h w c -> " + axes)


class EpisodeStats:
    def __init__(self, in_keys: Sequence[str] = None):
        self.in_keys = in_keys
        self._stats = []
        self._episodes = 0

    def add(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict = tensordict["next"]
        done = next_tensordict.get("done", None)
        if done is None:
            return len(self)

        # ---- normalize done shape to match next_tensordict.batch_size ----
        batch_size = next_tensordict.batch_size  # e.g. torch.Size([E]) or torch.Size([T, E])
        if done.shape == batch_size + torch.Size([1]):
            # common case: [E,1] or [T,E,1]
            done = done.squeeze(-1)
        elif done.shape == batch_size:
            # already ok: [E] or [T,E]
            pass
        else:
            # tolerate some reshapeable cases, otherwise fail loudly
            if done.shape[-1] == 1 and done.shape[:-1] == batch_size:
                done = done.squeeze(-1)
            elif done.numel() == int(torch.tensor(batch_size).prod().item()):
                done = done.reshape(*batch_size)
            else:
                raise RuntimeError(
                    f"[EpisodeStats.add] Unexpected done shape: {tuple(done.shape)}, "
                    f"expected {tuple(batch_size)} or {tuple(batch_size + torch.Size([1]))}."
                )

        # ---- collect stats on finished envs ----
        if done.any():
            self._episodes += done.sum().item()
            next_selected = next_tensordict.select(*self.in_keys)
            # boolean indexing is now safe because done matches batch_size exactly
            self._stats.extend(next_selected[done].cpu().unbind(0))

        return len(self)

    def pop(self):
        stats: TensorDictBase = torch.stack(self._stats).to_tensordict()
        self._stats.clear()
        return stats

    def __len__(self):
        return len(self._stats)

def create_env(cfg, headless: bool = True):
    """
    Create TorchRL env with ORP env registry.
    This is used by some framework wrappers.
    """
    from resources.envs.isaac_env import IsaacEnv
    from resources.utils.torchrl.transforms import wrap_env

    if cfg.task.name not in IsaacEnv.REGISTRY:
        raise KeyError(
            f"[create_env] Unknown task name: {cfg.task.name}. "
            f"Available: {list(IsaacEnv.REGISTRY.keys())}"
        )

    env_class = IsaacEnv.REGISTRY[cfg.task.name]
    base_env = env_class(cfg, headless=headless)
    env = wrap_env(base_env)
    return env


def check_env(env) -> None:
    """
    Basic sanity checks for TorchRL env specs.
    """
    # TorchRL env has input_spec / output_spec; not all wrappers expose both.
    if hasattr(env, "observation_spec"):
        _ = env.observation_spec
    if hasattr(env, "action_spec"):
        _ = env.action_spec
    if hasattr(env, "reward_spec"):
        _ = env.reward_spec

    # One step dry-run (safe guard)
    try:
        td = env.reset()
        a = env.action_spec.zero()
        td["agents", "action"] = a if ("agents", "action") in td.keys(True, True) else a
        env.step(td)
    except Exception as e:
        raise RuntimeError(f"[check_env] env dry-run failed: {e}") from e


def update_policy_weights_(policy: torch.nn.Module, state_dict: Dict[str, Any], strict: bool = False) -> None:
    """
    Load weights into policy/actor/critic.
    Framework may call this hook.
    """
    missing, unexpected = policy.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        print(f"[update_policy_weights_] missing={missing}, unexpected={unexpected}")