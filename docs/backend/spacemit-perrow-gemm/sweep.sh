#!/usr/bin/env bash
# Layer sweep for a per-layer slope: pp156, -fa 1, r=10, 2 interleaved rounds.
# usage: OURS=<bin dir> BASE=<bin dir> MODELS=<dir with pr_L{2,4,8,12,16}.gguf> [FUSED=<fused_bench> WZ=<layer dir>] sweep.sh
set -uo pipefail
bench() {  # tag bin model
  echo "=== $1 model=$3"
  env GGML_SPACEMIT_BACKEND=1 LD_LIBRARY_PATH=$2 $2/llama-bench -m $MODELS/$3.gguf --device SPACEMIT0 \
    -p 156 -n 0 -fa 1 -r ${REPS:-10} -o md 2>&1 | grep -E "^\|.*pp156|error|abort|failed"
}
for round in 1 2; do
  echo "##### round $round $(date +%T)"
  [ -n "${FUSED:-}" ] && LD_LIBRARY_PATH=$OURS $FUSED $WZ 156 8 10 2>&1 | grep 'fused 16L'
  for L in 2 4 8 12 16; do
    bench ours $OURS pr_L$L
    bench base $BASE pr_L$L
  done
done
