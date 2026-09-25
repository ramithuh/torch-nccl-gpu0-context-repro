#!/bin/bash
#SBATCH --job-name=gpu0ctx
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:15:00
#SBATCH --output=results/%x-%j.out
#
# Runs every case on one 2-GPU allocation. Submit from the repo root:
#   NIGHTLY=/path/to/env_torch_nightly STABLE=/path/to/env_torch_stable sbatch run.sbatch
#   ... sbatch run.sbatch nightly    # only cases whose name contains "nightly"
#
# Each case prints rank 1's CUDA contexts per step; a "gpu0:" entry on rank 1
# is the bug.
set -u
FILTER=${1:-}
NIGHTLY=${NIGHTLY:?set NIGHTLY to a python env with the newer torch}
STABLE=${STABLE:?set STABLE to a python env with the older torch}
NEW_NCCL=$NIGHTLY/lib/python3.11/site-packages/nvidia/nccl/lib/libnccl.so.2
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 NCCL_DEBUG=VERSION

run() {  # run <name> <env> [VAR=value ...] -- <repro args>
  local name=$1 env=$2; shift 2
  local vars=(); while [ "$1" != "--" ]; do vars+=("$1"); shift; done; shift
  [[ -n "$FILTER" && "$name" != *"$FILTER"* ]] && return
  echo "===== $name $(date +%T)"
  env PATH="$env/bin:$PATH" "${vars[@]}" \
    timeout -k 10 180 torchrun --standalone --nproc_per_node=2 repro.py "$@" 2>&1 |
    stdbuf -oL grep -vE "^\s*$"
  echo "===== $name exit=${PIPESTATUS[0]} $(date +%T)"   # 124 = timed out
}

hostname
nvidia-smi --query-gpu=index,name,driver_version --format=csv,noheader
run nightly_nccl_device_id      "$NIGHTLY"                        -- --variant device_id
run nightly_nccl_no_device_id   "$NIGHTLY"                        -- --variant no_device_id
run nightly_nccl_legacy         "$NIGHTLY" TORCH_DIST_USE_NCCL2=0 -- --variant device_id
run nightly_gloo                "$NIGHTLY"                        -- --variant no_device_id --backend gloo
run stable_nccl_device_id       "$STABLE"                         -- --variant device_id
run stable_torch_new_nccl       "$STABLE" LD_PRELOAD="$NEW_NCCL"  -- --variant device_id
echo "===== done"
