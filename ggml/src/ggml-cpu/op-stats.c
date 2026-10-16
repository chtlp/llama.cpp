#include "op-stats.h"

#include "ggml-impl.h"

#include <ctype.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define OP_STATS_MAX_THREADS 64
#define OP_STATS_TABLE_SIZE  16384
#define OP_STATS_KEY_LEN     96

typedef struct {
    int64_t * t_begin;
    int64_t * t_end;
    int       cap;
} op_stats_thread;

typedef struct {
    int     used;
    int     ntok;
    char    key[OP_STATS_KEY_LEN];
    int64_t calls;
    int64_t wall_ns;
    int64_t busy_ns;
    int64_t nthr;
    double  flops;
    double  bytes;  // dst + all srcs, nbytes (MUL_MAT_ID counts every expert of src0)
    char    shape[160];
} op_stats_entry;

typedef struct {
    int     used;
    int     ntok;
    int64_t graphs;
    int64_t graph_ns;   // first node begin -> last node end
    int64_t launch_ns;  // whole launch incl. tile start/stop
} op_stats_graph_entry;

static int             g_enabled = -1;
static pthread_once_t  g_once    = PTHREAD_ONCE_INIT;
static op_stats_thread g_threads[OP_STATS_MAX_THREADS];
static op_stats_entry  g_table[OP_STATS_TABLE_SIZE];
static op_stats_graph_entry g_graphs[64];
static pthread_mutex_t g_mutex = PTHREAD_MUTEX_INITIALIZER;

static void op_stats_dump(void);

static void op_stats_init(void) {
    const char * e = getenv("GGML_OP_STATS");
    g_enabled      = e && e[0] == '1';
    if (g_enabled) {
        atexit(op_stats_dump);
    }
}

bool ggml_op_stats_enabled(void) {
    pthread_once(&g_once, op_stats_init);
    return g_enabled;
}

int64_t ggml_op_stats_now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t) ts.tv_sec * 1000000000LL + ts.tv_nsec;
}

void ggml_op_stats_record(int ith, int node_n, int64_t t_begin, int64_t t_end) {
    if (ith < 0 || ith >= OP_STATS_MAX_THREADS) {
        return;
    }
    op_stats_thread * th = &g_threads[ith];
    if (node_n >= th->cap) {
        int cap = th->cap ? th->cap : 4096;
        while (cap <= node_n) {
            cap *= 2;
        }
        th->t_begin = (int64_t *) realloc(th->t_begin, cap * sizeof(int64_t));
        th->t_end   = (int64_t *) realloc(th->t_end, cap * sizeof(int64_t));
        memset(th->t_begin + th->cap, 0, (cap - th->cap) * sizeof(int64_t));
        memset(th->t_end + th->cap, 0, (cap - th->cap) * sizeof(int64_t));
        th->cap = cap;
    }
    th->t_begin[node_n] = t_begin;
    th->t_end[node_n]   = t_end;
}

int ggml_op_stats_graph_ntok(const struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes && i < 64; i++) {
        const struct ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_GET_ROWS && (strcmp(t->name, "inp_embd") == 0 || strcmp(t->name, "embd") == 0 ||
                                          (t->src[0] && strcmp(t->src[0]->name, "token_embd.weight") == 0))) {
            return (int) t->ne[1];
        }
    }
    // graphs without a token embedding lookup (e.g. an MTP draft head fed with hidden states):
    // the activation rows of the first matmul
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const struct ggml_tensor * t = cgraph->nodes[i];
        if (t->op == GGML_OP_MUL_MAT && t->src[1]) {
            return (int) t->src[1]->ne[1];
        }
    }
    return -1;
}

static uint64_t fnv1a(const char * s, int ntok) {
    uint64_t h = 1469598103934665603ULL ^ (uint64_t) ntok;
    for (; *s; s++) {
        h ^= (unsigned char) *s;
        h *= 1099511628211ULL;
    }
    return h;
}

static op_stats_entry * table_get(int ntok, const char * key) {
    uint64_t h = fnv1a(key, ntok);
    for (int probe = 0; probe < OP_STATS_TABLE_SIZE; probe++) {
        op_stats_entry * e = &g_table[(h + probe) % OP_STATS_TABLE_SIZE];
        if (!e->used) {
            e->used = 1;
            e->ntok = ntok;
            snprintf(e->key, sizeof(e->key), "%s", key);
            return e;
        }
        if (e->ntok == ntok && strcmp(e->key, key) == 0) {
            return e;
        }
    }
    return NULL;
}

static op_stats_graph_entry * graph_get(int ntok) {
    for (size_t i = 0; i < sizeof(g_graphs) / sizeof(g_graphs[0]); i++) {
        op_stats_graph_entry * g = &g_graphs[i];
        if (!g->used) {
            g->used = 1;
            g->ntok = ntok;
            return g;
        }
        if (g->ntok == ntok) {
            return g;
        }
    }
    return NULL;
}

// op name + tensor name with the trailing "-<layer>" removed
static void node_key(const struct ggml_tensor * t, char * buf, size_t n) {
    char name[GGML_MAX_NAME];
    snprintf(name, sizeof(name), "%s", t->name);
    char * dash = strrchr(name, '-');
    if (dash && dash[1] != '\0') {
        char * p = dash + 1;
        while (*p && isdigit((unsigned char) *p)) {
            p++;
        }
        if (*p == '\0' || *p == ' ') {
            memmove(dash, p, strlen(p) + 1);
        }
    }
    const char * op = t->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(t)) : ggml_op_name(t->op);
    if ((t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->src[0]) {
        // split matmuls by weight type, e.g. MUL_MAT:ffn_up[q8_0]
        snprintf(buf, n, "%s:%s[%s]", op, name[0] ? name : "?", ggml_type_name(t->src[0]->type));
        return;
    }
    if (getenv("GGML_OP_STATS_SHAPES")) {
        // split by output shape, e.g. RMS_NORM:norm{128x32x512}
        snprintf(buf, n, "%s:%s{%lldx%lldx%lld}", op, name[0] ? name : "?", (long long) t->ne[0], (long long) t->ne[1],
                 (long long) t->ne[2]);
        return;
    }
    snprintf(buf, n, "%s:%s", op, name[0] ? name : "?");
}

static double node_bytes(const struct ggml_tensor * t) {
    double b = (double) ggml_nbytes(t);
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (t->src[i]) {
            b += (double) ggml_nbytes(t->src[i]);
        }
    }
    return b;
}

static void node_shape(const struct ggml_tensor * t, char * buf, size_t n) {
    int off = snprintf(buf, n, "dst=%lldx%lldx%lld", (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2]);
    for (int i = 0; i < 3 && off < (int) n; i++) {
        const struct ggml_tensor * s = t->src[i];
        if (s) {
            off += snprintf(buf + off, n - off, " s%d=%lldx%lldx%lld%s", i, (long long) s->ne[0], (long long) s->ne[1],
                            (long long) s->ne[2], ggml_is_contiguous(s) ? "" : "(nc)");
        }
    }
}

static double node_flops(const struct ggml_tensor * t) {
    switch (t->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return 2.0 * (double) t->src[0]->ne[0] * (double) ggml_nelements(t);
        case GGML_OP_FLASH_ATTN_EXT:
            {
                const struct ggml_tensor * q = t->src[0];
                const struct ggml_tensor * k = t->src[1];
                const struct ggml_tensor * v = t->src[2];
                return 2.0 * (double) q->ne[1] * (double) q->ne[2] * (double) q->ne[3] * (double) k->ne[1] *
                       (double) (k->ne[0] + v->ne[0]);
            }
        default:
            return 0.0;
    }
}

void ggml_op_stats_graph_done(const struct ggml_cgraph * cgraph, int nth) {
    if (nth > OP_STATS_MAX_THREADS) {
        nth = OP_STATS_MAX_THREADS;
    }
    const int ntok = ggml_op_stats_graph_ntok(cgraph);

    pthread_mutex_lock(&g_mutex);

    int64_t first_begin = 0;
    int64_t prev_end    = 0;
    int64_t sync_ns     = 0;
    char    key[OP_STATS_KEY_LEN];

    for (int i = 0; i < cgraph->n_nodes; i++) {
        int64_t b_min = 0, e_max = 0, busy = 0, cnt = 0;
        for (int ith = 0; ith < nth; ith++) {
            op_stats_thread * th = &g_threads[ith];
            if (i >= th->cap || th->t_begin[i] == 0) {
                continue;
            }
            const int64_t b = th->t_begin[i];
            const int64_t e = th->t_end[i];
            b_min           = cnt == 0 || b < b_min ? b : b_min;
            e_max           = e > e_max ? e : e_max;
            busy += e - b;
            cnt++;
            th->t_begin[i] = 0;
            th->t_end[i]   = 0;
        }
        if (cnt == 0) {
            continue;
        }
        if (first_begin == 0) {
            first_begin = b_min;
        } else if (b_min > prev_end) {
            sync_ns += b_min - prev_end;
        }
        prev_end = e_max;

        node_key(cgraph->nodes[i], key, sizeof(key));
        op_stats_entry * e = table_get(ntok, key);
        if (e) {
            e->calls++;
            e->wall_ns += e_max - b_min;
            e->busy_ns += busy;
            e->nthr += cnt;
            e->flops += node_flops(cgraph->nodes[i]);
            e->bytes += node_bytes(cgraph->nodes[i]);
            if (!e->shape[0]) {
                node_shape(cgraph->nodes[i], e->shape, sizeof(e->shape));
            }
        }
    }

    op_stats_entry * es = table_get(ntok, "(barrier gaps)");
    if (es) {
        es->calls++;
        es->wall_ns += sync_ns;
    }
    op_stats_graph_entry * g = graph_get(ntok);
    if (g && first_begin) {
        g->graphs++;
        g->graph_ns += prev_end - first_begin;
    }

    pthread_mutex_unlock(&g_mutex);
}

void ggml_op_stats_add_launch(const struct ggml_cgraph * cgraph, int64_t ns) {
    const int ntok = ggml_op_stats_graph_ntok(cgraph);
    pthread_mutex_lock(&g_mutex);
    op_stats_graph_entry * g = graph_get(ntok);
    if (g) {
        g->launch_ns += ns;
    }
    pthread_mutex_unlock(&g_mutex);
}

static int cmp_wall_desc(const void * a, const void * b) {
    const op_stats_entry * x = *(const op_stats_entry * const *) a;
    const op_stats_entry * y = *(const op_stats_entry * const *) b;
    if (x->ntok != y->ntok) {
        return x->ntok < y->ntok ? -1 : 1;
    }
    return x->wall_ns < y->wall_ns ? 1 : (x->wall_ns > y->wall_ns ? -1 : 0);
}

static void op_stats_dump(void) {
    pthread_mutex_lock(&g_mutex);

    op_stats_entry * rows[OP_STATS_TABLE_SIZE];
    int              n_rows = 0;
    for (int i = 0; i < OP_STATS_TABLE_SIZE; i++) {
        if (g_table[i].used) {
            rows[n_rows++] = &g_table[i];
        }
    }
    qsort(rows, n_rows, sizeof(rows[0]), cmp_wall_desc);

    FILE *       tsv  = NULL;
    const char * path = getenv("GGML_OP_STATS_FILE");
    if (path && path[0]) {
        tsv = fopen(path, "w");
        if (tsv) {
            fprintf(tsv, "kind\tntok\tkey\tcalls\twall_ms\tbusy_ms\tnthr\tgflop\tgbyte\tshape\n");
        }
    }

    for (size_t gi = 0; gi < sizeof(g_graphs) / sizeof(g_graphs[0]); gi++) {
        const op_stats_graph_entry * g = &g_graphs[gi];
        if (!g->used || g->graphs == 0) {
            continue;
        }
        const double graph_ms = g->graph_ns / 1e6;
        fprintf(stderr, "\n== ggml op stats: ntok=%d graphs=%lld graph_wall=%.1f ms launch_wall=%.1f ms ==\n", g->ntok,
                (long long) g->graphs, graph_ms, g->launch_ns / 1e6);
        fprintf(stderr, "%-44s %8s %11s %6s %7s %9s\n", "op:tensor", "calls", "wall_ms", "%", "util%", "GFLOP/s");
        if (tsv) {
            fprintf(tsv, "graph\t%d\t-\t%lld\t%.3f\t%.3f\t0\t0\t0\t-\n", g->ntok, (long long) g->graphs, graph_ms,
                    g->launch_ns / 1e6);
        }
        for (int i = 0; i < n_rows; i++) {
            const op_stats_entry * e = rows[i];
            if (e->ntok != g->ntok) {
                continue;
            }
            const double wall_ms = e->wall_ns / 1e6;
            // util: busy thread-time / (avg threads per call * wall)
            const double nthr_avg = e->calls ? (double) e->nthr / e->calls : 0.0;
            const double util     = e->wall_ns && nthr_avg > 0 ? 100.0 * e->busy_ns / (nthr_avg * e->wall_ns) : 0.0;
            const double gflops   = e->wall_ns ? e->flops / e->wall_ns : 0.0;
            if (100.0 * wall_ms / graph_ms >= 0.1) {
                fprintf(stderr, "%-44s %8lld %11.1f %6.2f %7.1f %9.1f\n", e->key, (long long) e->calls, wall_ms,
                        100.0 * wall_ms / graph_ms, util, gflops);
            }
            if (tsv) {
                fprintf(tsv, "op\t%d\t%s\t%lld\t%.3f\t%.3f\t%.2f\t%.3f\t%.4f\t%s\n", e->ntok, e->key, (long long) e->calls,
                        wall_ms, e->busy_ns / 1e6, nthr_avg, e->flops / 1e9, e->bytes / 1e9, e->shape[0] ? e->shape : "-");
            }
        }
    }

    if (tsv) {
        fclose(tsv);
    }
    pthread_mutex_unlock(&g_mutex);
}
