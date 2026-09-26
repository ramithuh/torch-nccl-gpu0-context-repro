#!/bin/bash
# Per-node health report from Slurm accounting (all users), split by GPU count.
#
#   ./slurm_node_report.sh [-s YYYY-MM-DD] node1 [node2 ...]
#
# For each node: current state + drain reason, then job outcomes since the start
# date grouped by GPUs per job (1, 2-4, 5-8). Multi-GPU failure rate matters most
# for DDP/NCCL runs: a node can look fine for 1-2 GPU jobs yet fail 8-GPU ones.
# Failure counts include ordinary user errors, so compare nodes against each
# other rather than reading a rate in isolation.
set -u
SINCE=$(date -d '7 days ago' +%F)
if [ "${1:-}" = "-s" ]; then SINCE=$2; shift 2; fi
[ $# -gt 0 ] || { echo "usage: $0 [-s YYYY-MM-DD] node [node ...]" >&2; exit 1; }

printf "since %s (all users, one row per job allocation)\n\n" "$SINCE"
for node in "$@"; do
  info=$(scontrol show node "$node" 2>/dev/null)
  state=$(grep -oE 'State=\S+' <<<"$info")
  reason=$(grep -oE 'Reason=.*' <<<"$info" | head -1)
  echo "===== $node  ${state:-State=?}  ${reason}"
  sacct -a -X -n -P -N "$node" -S "$SINCE" -o State,AllocTRES 2>/dev/null |
    awk -F'|' '
      {
        n = 0
        if (match($2, /gres\/gpu[^=]*=[0-9]+/)) { s = substr($2, RSTART, RLENGTH); sub(/.*=/, "", s); n = s + 0 }
        if (n == 0) next
        b = (n == 1) ? "1 GPU" : (n <= 4) ? "2-4 GPU" : "5-8 GPU"
        st = $1; sub(/ .*/, "", st)
        total[b]++
        if (st == "COMPLETED") ok[b]++
        else if (st == "FAILED" || st == "NODE_FAIL" || st ~ /OUT_OF_ME/) bad[b]++
        else if (st == "TIMEOUT") to[b]++
        else if (st == "RUNNING") run[b]++
      }
      END {
        printf "  %-8s %6s %9s %7s %8s %8s %9s\n", "bucket", "jobs", "completed", "failed", "timeout", "running", "fail rate"
        split("1 GPU,2-4 GPU,5-8 GPU", order, ",")
        for (i = 1; i <= 3; i++) {
          b = order[i]; if (!total[b]) continue
          fin = ok[b] + bad[b]
          printf "  %-8s %6d %9d %7d %8d %8d %8s\n", b, total[b], ok[b], bad[b], to[b], run[b], fin ? sprintf("%.0f%%", 100 * bad[b] / fin) : "-"
        }
      }'
  echo
done
