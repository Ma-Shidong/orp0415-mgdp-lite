#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/home/csj/msd/orp0415/orp"
CONDA_SH="/home/csj/anaconda3/etc/profile.d/conda.sh"
CONDA_ENV="orp"

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="/media/share/csj/msd/orp_runs/four_exp_${STAMP}"
TMP_ROOT="/media/share/csj/msd/orp_tmp/four_exp_${STAMP}"

mkdir -p "$RUN_ROOT" "$TMP_ROOT"

# Isaac Sim's conda activation script reads shell-specific variables such as
# ZSH_VERSION. Keep strict mode for the launcher, but relax nounset while
# activation hooks run.
set +u
source "$CONDA_SH"
conda activate "$CONDA_ENV"
set -u

cd "$REPO_ROOT/scripts"

BASE_OVERRIDES=(
  headless=true
  wandb.mode=offline
  resume_checkpoint=null
  task.env.num_envs=1024
  task.env.max_episode_length=1200
  algo.train_every=64
  total_frames=655360000
  max_iters=10000
  eval_interval=500
  save_interval=500
  record_video=false
  task.success_curriculum.enable=true
  seed=0
)

launch_exp() {
  local gpu="$1"
  local name="$2"
  shift 2

  local run_dir="${RUN_ROOT}/${name}"
  local tmp_dir="${TMP_ROOT}/${name}"
  mkdir -p "$run_dir" "$tmp_dir"/{xdg_cache,torch_extensions,omni_cache}

  {
    printf 'cd %s/scripts\n' "$REPO_ROOT"
    printf 'CUDA_VISIBLE_DEVICES=%s SIM_DEVICE=cuda:0 python train.py ...\n\n' "$gpu"
    printf 'Overrides:\n'
    printf '  %s\n' "${BASE_OVERRIDES[@]}" "$@"
  } > "${run_dir}/command.txt"

  (
    export CUDA_VISIBLE_DEVICES="$gpu"
    export SIM_DEVICE="cuda:0"
    export TMPDIR="$tmp_dir"
    export XDG_CACHE_HOME="${tmp_dir}/xdg_cache"
    export TORCH_EXTENSIONS_DIR="${tmp_dir}/torch_extensions"
    export OMNI_USER_CACHE_DIR="${tmp_dir}/omni_cache"
    export WANDB_DIR="$run_dir"
    export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

    exec > >(tee "${run_dir}/train.log") 2>&1
    exec python train.py "${BASE_OVERRIDES[@]}" "$@"
  ) &

  local pid=$!
  echo "$pid" > "${run_dir}/pid.txt"
  echo "Started ${name} on GPU ${gpu}, pid=${pid}"
}

launch_exp 0 "exp1_mgdpv2_orp_current_reward" \
  task.input_mode=mgdp_lite_v2 \
  task.reward_mode=p2m \
  model.policy_variant=orp \
  task.flow_update_period=2

launch_exp 1 "exp2_original_p2m_input_net_reward" \
  task.input_mode=p2m_original \
  task.reward_mode=p2m_original \
  model.policy_variant=p2m_original \
  task.flow_update_period=1 \
  algo.entropy_coef=0.001 \
  algo.entropy_coef_mid=0.001 \
  algo.entropy_coef_end=0.001 \
  algo.clip_param=0.1 \
  algo.actor_lr=0.0005 \
  algo.critic_lr=0.0005 \
  algo.critic_priv_enable=false \
  algo.critic_aux_enable=false \
  model.temporal.enable=false \
  model.teacher_student.enable=false

launch_exp 2 "exp3_mgdpv2_orp_original_reward" \
  task.input_mode=mgdp_lite_v2 \
  task.reward_mode=p2m_original \
  model.policy_variant=orp \
  task.flow_update_period=2

launch_exp 3 "exp4_mgdpv2_no_ch3_orp_current_reward" \
  task.input_mode=mgdp_lite_v2_no_ch3 \
  task.reward_mode=p2m \
  model.policy_variant=orp \
  task.flow_update_period=2

cat > "${RUN_ROOT}/README.txt" <<EOF
Four experiments started at ${STAMP}

Run root:
  ${RUN_ROOT}

Temp/cache root:
  ${TMP_ROOT}

Monitor:
  tail -f ${RUN_ROOT}/exp1_mgdpv2_orp_current_reward/train.log
  tail -f ${RUN_ROOT}/exp2_original_p2m_input_net_reward/train.log
  tail -f ${RUN_ROOT}/exp3_mgdpv2_orp_original_reward/train.log
  tail -f ${RUN_ROOT}/exp4_mgdpv2_no_ch3_orp_current_reward/train.log

Stop all:
  kill \$(cat ${RUN_ROOT}/exp*/pid.txt)
EOF

echo
echo "All four jobs launched."
echo "RUN_ROOT=${RUN_ROOT}"
echo "TMP_ROOT=${TMP_ROOT}"
echo "Waiting for jobs. Press Ctrl+C only if you want to stop this launcher; jobs may keep running."

wait
