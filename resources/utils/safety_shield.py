from typing import Dict, List

import torch


def apply_target_acc_safety_shield(
    target_acc: torch.Tensor,
    min_depth,
    z,
    virtual_ground: float,
    safety_dis: float,
    acc_ref: float,
    soft_margin: float = 0.8,
    floor_margin: float = 0.35,
    floor_gain: float = 4.0,
    floor_bias_max: float = 2.0,
) -> Dict[str, object]:
    """Inference-only safety filter for target acceleration commands."""

    target_acc_before = target_acc.clone()
    device = target_acc.device
    dtype = target_acc.dtype
    batch = int(target_acc.shape[0])

    min_depth_t = torch.as_tensor(min_depth, device=device, dtype=dtype).reshape(-1)
    if min_depth_t.numel() == 1:
        min_depth_t = min_depth_t.expand(batch)
    else:
        min_depth_t = min_depth_t[:batch]

    z_t = torch.as_tensor(z, device=device, dtype=dtype).reshape(-1)
    if z_t.numel() == 1:
        z_t = z_t.expand(batch)
    else:
        z_t = z_t[:batch]

    out = target_acc.clone()
    scale_xy = torch.ones(batch, device=device, dtype=dtype)
    floor_bias = torch.zeros(batch, device=device, dtype=dtype)
    active = torch.zeros(batch, device=device, dtype=torch.bool)

    hard_mask = min_depth_t < float(safety_dis)
    soft_mask = (~hard_mask) & (min_depth_t < float(safety_dis + soft_margin))
    floor_limit = float(virtual_ground + floor_margin)
    floor_mask = z_t < floor_limit

    if soft_mask.any():
        scale_xy_soft = ((min_depth_t[soft_mask] - float(safety_dis)) / max(float(soft_margin), 1.0e-6)).clamp(0.0, 1.0)
        scale_xy[soft_mask] = scale_xy_soft
        out[soft_mask, :2] = out[soft_mask, :2] * scale_xy_soft.unsqueeze(-1)
        active[soft_mask] = True

    if hard_mask.any():
        out[hard_mask, :2] = 0.0
        out[hard_mask, 2] = torch.clamp(out[hard_mask, 2], min=0.0)
        scale_xy[hard_mask] = 0.0
        active[hard_mask] = True

    if floor_mask.any():
        floor_bias_val = ((floor_limit - z_t[floor_mask]).clamp(min=0.0) * float(floor_gain)).clamp(max=float(floor_bias_max))
        floor_bias[floor_mask] = floor_bias_val
        out[floor_mask, 2] = torch.clamp(out[floor_mask, 2], min=0.0) + floor_bias_val
        active[floor_mask] = True

    out = torch.clamp(out, -float(acc_ref), float(acc_ref))

    reasons: List[str] = []
    for idx in range(batch):
        tags = []
        if bool(hard_mask[idx]):
            tags.append("hard_collision")
        elif bool(soft_mask[idx]):
            tags.append("soft_collision")
        if bool(floor_mask[idx]):
            tags.append("floor_guard")
        reasons.append(",".join(tags) if tags else "none")

    return {
        "target_acc": out,
        "shield_active": active,
        "shield_scale_xy": scale_xy,
        "shield_floor_bias": floor_bias,
        "shield_reason": reasons,
        "target_acc_before": target_acc_before,
        "target_acc_after": out.clone(),
    }
