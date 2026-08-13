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


import datetime
import logging
import os
from typing import Any

import wandb
from omegaconf import OmegaConf

def dict_flatten(x, parent_key="", sep="."):
    items = {}
    if isinstance(x, dict):
        for k, v in x.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(dict_flatten(v, new_key, sep=sep))
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            new_key = f"{parent_key}{sep}{i}" if parent_key else str(i)
            items.update(dict_flatten(v, new_key, sep=sep))
    else:
        items[parent_key] = x
    return items


def _cfg_get(node: Any, key: str, default=None):
    try:
        return node.get(key, default)
    except Exception:
        return getattr(node, key, default)


def _build_wandb_settings(disable_sys: bool, init_timeout: float):
    settings_kwargs = {}
    if init_timeout is not None:
        settings_kwargs["init_timeout"] = float(init_timeout)
    if disable_sys:
        settings_kwargs["x_disable_stats"] = True
        settings_kwargs["x_disable_meta"] = True
    try:
        return wandb.Settings(**settings_kwargs)
    except Exception:
        settings_kwargs.pop("x_disable_meta", None)
        try:
            return wandb.Settings(**settings_kwargs)
        except Exception:
            settings_kwargs.pop("x_disable_stats", None)
            if disable_sys:
                settings_kwargs["_disable_stats"] = True
            return wandb.Settings(**settings_kwargs)


def init_wandb(cfg):
    """Initialize WandB.

    If only `run_id` is given, resume from the run specified by `run_id`.
    If only `run_path` is given, start a new run from that specified by `run_path`,
        possibly restoring trained models.

    Otherwise, start a fresh new run.

    """
    wandb_cfg = cfg.wandb
    time_str = datetime.datetime.now().strftime("%m-%d_%H-%M")
    run_name = f"{wandb_cfg.run_name}/{time_str}"
    kwargs = dict(
        project=wandb_cfg.project,
        group=wandb_cfg.group,
        entity=wandb_cfg.entity,
        name=run_name,
        mode=wandb_cfg.mode,
        tags=wandb_cfg.tags,
    )
    if wandb_cfg.run_id is not None:
        kwargs["allow_val_change"] = True
        kwargs["id"] = wandb_cfg.run_id
        kwargs["resume"] = "must"
        print(f"Resume training from run {wandb_cfg.run_id}!")
    else:
        kwargs["allow_val_change"] = False
        kwargs["id"] = wandb.util.generate_id()
        new_id = kwargs["id"]
        print(f"starting a new run: {new_id}")
    # Disable W&B system metrics / metadata collection by default to reduce overhead.
    # Can be overridden via cfg.wandb.disable_system_metrics=false
    disable_sys = bool(_cfg_get(wandb_cfg, "disable_system_metrics", True))
    init_timeout = float(_cfg_get(wandb_cfg, "init_timeout", 300.0))
    fallback_mode = str(_cfg_get(wandb_cfg, "fallback_mode", "offline") or "").lower()
    kwargs["settings"] = _build_wandb_settings(disable_sys=disable_sys, init_timeout=init_timeout)

    try:
        run = wandb.init(**kwargs)
    except Exception as exc:
        logging.exception("wandb.init failed in mode=%s", kwargs.get("mode"))
        mode = str(kwargs.get("mode", "") or "").lower()
        if mode == "online" and fallback_mode in {"offline", "disabled"}:
            retry_kwargs = dict(kwargs)
            retry_kwargs["mode"] = fallback_mode
            logging.warning(
                "Retrying wandb.init in fallback mode=%s after online init failure: %s",
                fallback_mode,
                exc,
            )
            run = wandb.init(**retry_kwargs)
        else:
            raise
    cfg_dict = dict_flatten(OmegaConf.to_container(cfg))
    run.config.update(cfg_dict, allow_val_change = kwargs["allow_val_change"])
    return run
