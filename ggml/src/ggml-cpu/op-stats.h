#pragma once

// Lightweight aggregated per-op timing for the CPU graph loop.
// Enabled at runtime with GGML_OP_STATS=1; the summary is printed to stderr at
// exit and, if GGML_OP_STATS_FILE is set, also written there as TSV.

#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

bool    ggml_op_stats_enabled(void);
int64_t ggml_op_stats_now_ns(void);

// called by every thread for every executed node (t_end is before the barrier)
void ggml_op_stats_record(int ith, int node_n, int64_t t_begin, int64_t t_end);

// called by thread 0 after all threads have finished the graph
void ggml_op_stats_graph_done(const struct ggml_cgraph * cgraph, int nth);

// called by the launching thread with the wall time of the whole graph launch
void ggml_op_stats_add_launch(const struct ggml_cgraph * cgraph, int64_t ns);

// n_tokens of the graph being computed (from the embedding GET_ROWS node), -1 if unknown
int ggml_op_stats_graph_ntok(const struct ggml_cgraph * cgraph);

#ifdef __cplusplus
}
#endif
