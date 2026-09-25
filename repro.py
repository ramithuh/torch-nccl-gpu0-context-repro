"""Minimal repro: does rank 1 get a CUDA context on GPU 0 during NCCL setup?

No Lightning, no model, no data. Launch on one node with 2 GPUs:

    torchrun --nproc_per_node=2 repro.py --variant device_id
    torchrun --nproc_per_node=2 repro.py --variant no_device_id

Each rank prints which GPUs its own process holds a context on (nvidia-smi,
by PID) after every step. Related upstream: pytorch#149119, #126381, #163741.
"""

import argparse
import os
import subprocess
import time

import torch
import torch.distributed as dist


def contexts() -> str:
    """GPUs on which this PID holds a context, e.g. 'gpu0:426MiB gpu1:500MiB'."""
    try:
        idx = dict(
            line.split(", ")
            for line in subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,index", "--format=csv,noheader"], text=True
            ).splitlines()
        )
        apps = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,gpu_uuid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).splitlines()
        mine = [a.split(", ") for a in apps if a.split(", ")[0] == str(os.getpid())]
        return " ".join(f"gpu{idx[u]}:{m}MiB" for _, u, m in mine) or "none"
    except Exception as exc:  # never fail the repro on reporting
        return f"error({exc})"


def report(rank: int, stage: str) -> None:
    time.sleep(2)  # let context creation settle before nvidia-smi samples
    print(f"REPRO rank={rank} stage={stage} contexts=[{contexts()}]", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["device_id", "no_device_id"], required=True)
    parser.add_argument("--backend", default="nccl", choices=["nccl", "gloo"])
    args = parser.parse_args()

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)

    report(rank, "start")
    torch.cuda.set_device(device)
    torch.zeros(1, device=device)  # force our own context first
    report(rank, "after_set_device")

    kwargs = {"device_id": device} if args.variant == "device_id" else {}
    dist.init_process_group(args.backend, **kwargs)
    report(rank, "after_init_process_group")

    dist.barrier(device_ids=[local_rank] if args.backend == "nccl" else None)
    report(rank, "after_barrier")

    x = torch.ones(1024, device=device if args.backend == "nccl" else "cpu")
    dist.all_reduce(x)
    torch.cuda.synchronize()
    report(rank, "after_all_reduce")

    obj = [{"rank": rank}]
    dist.broadcast_object_list(obj, src=0)  # Lightning uses this during setup
    report(rank, "after_broadcast_object_list")

    dist.destroy_process_group()
    report(rank, "after_destroy")
    print(
        f"REPRO rank={rank} variant={args.variant} backend={args.backend} torch={torch.__version__} "
        f"nccl={'.'.join(map(str, torch.cuda.nccl.version()))} "
        f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}",
        flush=True,
    )


if __name__ == "__main__":
    main()
