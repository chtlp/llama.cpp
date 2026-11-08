#!/usr/bin/env bash
# KL divergence against F16 reference logits, Qwen3-1.7B 28 layers, wikitext-2 -c 512 --chunks 32.
# usage: BIN=<board build/bin> REF=<f16ref.kld> TXT=<wiki.test.raw> Q8=<Q8_0.gguf> PR=<pr_L28.gguf> F16=<F16.gguf> kld.sh
# The reference comes from an x86 CPU build on the F16 model:
#   llama-perplexity -m F16.gguf -f wiki.test.raw -c 512 --chunks 32 --kl-divergence-base f16ref.kld
run() {  # tag model env...
  local tag=$1 m=$2; shift 2
  echo "=== $tag $(date +%T)"
  env GGML_SPACEMIT_BACKEND=1 LD_LIBRARY_PATH=$BIN "$@" $BIN/llama-perplexity -m $m --device SPACEMIT0 -f $TXT -c 512 \
      --chunks 32 -fa on --kl-divergence-base $REF --kl-divergence 2>&1 \
    | grep -E 'Mean PPL\(Q\)|Mean    KLD|99.0%   KLD|99.9%   KLD|Median  KLD|Same top p|RMS Δp|error|fault|abort'
}
run q8-32x32         $Q8  GGML_SPACEMIT_PERROW=0
run pr-rot0          $PR  GGML_SPACEMIT_PERROW_ROT=0
run pr-rot1          $PR  GGML_SPACEMIT_PERROW_ROT=1
run pr-rot2          $PR  GGML_SPACEMIT_PERROW_ROT=2
run q8-perrow-rot2   $Q8  GGML_SPACEMIT_PERROW=2 GGML_SPACEMIT_PERROW_ROT=2
run f16-perrow-rot2  $F16 GGML_SPACEMIT_PERROW=2 GGML_SPACEMIT_PERROW_ROT=2
run f16-perrow-rot1  $F16 GGML_SPACEMIT_PERROW=2 GGML_SPACEMIT_PERROW_ROT=1
run f16-perrow-rot0  $F16 GGML_SPACEMIT_PERROW=2 GGML_SPACEMIT_PERROW_ROT=0
