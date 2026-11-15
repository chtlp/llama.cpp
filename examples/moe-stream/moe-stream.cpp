// Linux-only experiment: bounded direct-I/O buffers for real single-token MoE decode.
#include "llama.h"
#include "ggml-cpu.h"
#include "ggml-backend.h"
#include "gguf.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fcntl.h>
#include <fstream>
#include <map>
#include <mutex>
#include <numeric>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <thread>
#include <unistd.h>
#include <vector>

using clock_type = std::chrono::steady_clock;
static double now_us() {
    return std::chrono::duration<double, std::micro>(clock_type::now().time_since_epoch()).count();
}
static size_t align_up(size_t x) { return (x + 4095) & ~size_t(4095); }
static void require(bool ok, const char * message) {
    if (!ok) { std::fprintf(stderr, "fatal: %s (%s)\n", message, std::strerror(errno)); std::abort(); }
}
static void read_full(int fd, void * dst, size_t bytes, size_t offset) {
    size_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd, (char *) dst + done, bytes - done, offset + done);
        if (n < 0 && errno == EINTR) { continue; }
        require(n > 0, "pread");
        done += n;
    }
}

struct options {
    std::string model, pack, trace, modes = "native,memory,bulk,sync,stream";
    std::string prompt = "Explain why a small buffer can overlap storage reads and computation.";
    int layer = 10, threads = 8, tokens = 12, misses = 2, slots = 4, io_threads = 2;
    int io_cpu_base = -1;
    size_t chunk = 128 * 1024;
    bool prepare = false;
};
struct tensor_spec {
    size_t original, packed, bytes;
};
static std::map<std::string, tensor_spec> get_specs(const options & opt) {
    gguf_init_params p = {true, nullptr};
    auto * g = gguf_init_from_file(opt.model.c_str(), p);
    require(g != nullptr, "open GGUF metadata");
    std::map<std::string, tensor_spec> result;
    size_t offset = 0;
    for (const char * part : {"gate", "up", "down"}) {
        std::string name = "blk." + std::to_string(opt.layer) + ".ffn_" + part + "_exps.weight";
        int64_t id = gguf_find_tensor(g, name.c_str());
        require(id >= 0, "requested expert tensor missing");
        auto * ne = gguf_get_tensor_ne(g, id);
        require(ne[2] > 1 && ne[3] == 1, "expected 3D expert tensor");
        offset = align_up(offset);
        result[name] = {gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id), offset,
                        gguf_get_tensor_size(g, id)};
        offset += gguf_get_tensor_size(g, id);
    }
    gguf_free(g);
    return result;
}
static void prepare_pack(const options & opt, const std::map<std::string, tensor_spec> & specs) {
    int src = open(opt.model.c_str(), O_RDONLY);
    int dst = open(opt.pack.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0644);
    require(src >= 0 && dst >= 0, "open pack (will not overwrite)");
    std::vector<char> buffer(4 * 1024 * 1024);
    size_t end = 0;
    for (const auto & entry : specs) {
        const auto & s = entry.second;
        for (size_t n = 0; n < s.bytes;) {
            size_t bytes = std::min(buffer.size(), s.bytes - n);
            read_full(src, buffer.data(), bytes, s.original + n);
            size_t written = 0;
            while (written < bytes) {
                ssize_t rc = pwrite(dst, buffer.data() + written, bytes - written, s.packed + n + written);
                if (rc < 0 && errno == EINTR) { continue; }
                require(rc > 0, "write pack");
                written += rc;
            }
            n += bytes;
        }
        end = std::max(end, s.packed + s.bytes);
        std::printf("PACK %s source=%zu packed=%zu bytes=%zu\n", entry.first.c_str(), s.original, s.packed, s.bytes);
    }
    require(ftruncate(dst, align_up(end)) == 0 && fsync(dst) == 0, "flush pack");
    close(src);
    close(dst);
}

// All GGML workers participate. Release/acquire also publishes the current job.
struct barrier {
    std::atomic<unsigned> count{0}, phase{0};
    void wait(unsigned n) {
        unsigned old = phase.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) == n - 1) {
            count.store(0, std::memory_order_relaxed);
            phase.fetch_add(1, std::memory_order_release);
        } else {
            unsigned spins = 0;
            while (phase.load(std::memory_order_acquire) == old) {
                if (++spins % 1024 == 0) { std::this_thread::yield(); }
            }
        }
    }
};
struct job {
    int rank, expert;
    int64_t row, rows;
    bool missing;
    size_t offset, bytes;
};
struct slot {
    void * data = nullptr;
    int job_id = -1;
    size_t delta = 0;
};
struct event {
    std::string mode, kind, tensor;
    int step, rank, expert, slot_id;
    int64_t row;
    bool missing;
    double start, end;
    size_t useful, physical;
};

struct stream_context {
    const options & opt;
    std::map<std::string, tensor_spec> specs;
    int fd;
    std::string mode;
    bool active = false;
    int step = 0;
    barrier sync;
    std::vector<slot> slots;
    std::vector<std::thread> workers;
    std::mutex mutex, events_mutex;
    std::condition_variable cv;
    bool stop = false, running = false;
    std::vector<job> jobs;
    std::vector<int> read_jobs, resident_jobs;
    std::deque<int> free_slots, ready_slots;
    size_t next_read = 0, resident_next = 0, processed = 0;
    int current_job = -1, current_slot = -1;
    const char * current_data = nullptr;
    std::vector<float> quant;
    size_t quant_row = 0, weight_row = 0;
    const ggml_type_traits_cpu * traits = nullptr;
    std::string tensor_name;
    std::vector<event> events;
    std::vector<void *> bulk_buffers;
    double compute_start = 0, op_start = 0, target_us = 0;
    size_t payload_bytes = 0, read_bytes = 0, io_calls = 0;

    stream_context(const options & o, const std::map<std::string, tensor_spec> & s) : opt(o), specs(s) {
        fd = open(opt.pack.c_str(), O_RDONLY | O_DIRECT);
        require(fd >= 0, "open O_DIRECT pack (no buffered fallback)");
        struct stat st{};
        require(fstat(fd, &st) == 0, "stat pack");
        for (const auto & e : specs) {
            require(size_t(st.st_size) >= align_up(e.second.packed + e.second.bytes), "pack too small");
        }
        slots.resize(opt.slots);
        for (auto & s0 : slots) {
            require(posix_memalign(&s0.data, 4096, align_up(opt.chunk + 8192)) == 0, "allocate slot");
        }
        for (int i = 0; i < opt.io_threads; ++i) {
            workers.emplace_back([this, i] {
                if (opt.io_cpu_base >= 0) {
                    cpu_set_t cpus;
                    CPU_ZERO(&cpus);
                    CPU_SET(opt.io_cpu_base + i, &cpus);
                    require(pthread_setaffinity_np(pthread_self(), sizeof(cpus), &cpus) == 0, "pin I/O worker");
                }
                io_loop();
            });
        }
    }
    ~stream_context() {
        { std::lock_guard<std::mutex> lock(mutex); stop = true; }
        cv.notify_all();
        for (auto & t : workers) { t.join(); }
        for (auto & s : slots) { free(s.data); }
        for (void * p : bulk_buffers) { free(p); }
        close(fd);
    }
    void add_event(const event & e) {
        std::lock_guard<std::mutex> lock(events_mutex);
        if (e.kind == "io") {
            payload_bytes += e.useful;
            read_bytes += e.physical;
            ++io_calls;
        }
        if (!opt.trace.empty()) { events.push_back(e); }
    }
    size_t load_job(int jid, void * ptr, int sid) {
        const job & j = jobs[jid];
        size_t base = j.offset & ~size_t(4095);
        size_t delta = j.offset - base;
        size_t bytes = align_up(delta + j.bytes);
        double start = now_us();
        read_full(fd, ptr, bytes, base);
        double end = now_us();
        add_event({mode, "io", tensor_name, step, j.rank, j.expert, sid, j.row, true, start, end, j.bytes, bytes});
        return delta;
    }
    void io_loop() {
        for (;;) {
            int sid, jid;
            {
                std::unique_lock<std::mutex> lock(mutex);
                cv.wait(lock, [&] { return stop || (running && next_read < read_jobs.size() && !free_slots.empty()); });
                if (stop) { return; }
                sid = free_slots.front(); free_slots.pop_front();
                jid = read_jobs[next_read++];
            }
            size_t delta = load_job(jid, slots[sid].data, sid);
            {
                std::lock_guard<std::mutex> lock(mutex);
                slots[sid].delta = delta;
                slots[sid].job_id = jid;
                ready_slots.push_back(sid);
            }
            cv.notify_all();
        }
    }
    void begin(ggml_tensor * dst) {
        op_start = now_us();
        auto * w = dst->src[0];
        auto * x = dst->src[1];
        auto * ids = dst->src[2];
        tensor_name = w->name;
        require(w->nb[1] == ggml_row_size(w->type, w->ne[0]) &&
                w->nb[2] == w->nb[1] * w->ne[1], "expert layout must be contiguous");
        require(x->type == GGML_TYPE_F32 && ids->type == GGML_TYPE_I32 &&
                dst->type == GGML_TYPE_F32 && dst->nb[0] == sizeof(float), "unexpected activation layout");
        traits = ggml_get_type_traits_cpu(w->type);
        auto * qtraits = ggml_get_type_traits_cpu(traits->vec_dot_type);
        require(traits->vec_dot && qtraits->from_float, "quantized dot unavailable");
        quant_row = ggml_row_size(traits->vec_dot_type, x->ne[0]);
        quant.resize((quant_row * x->ne[1] + sizeof(float) - 1) / sizeof(float));
        for (int64_t i = 0; i < x->ne[1]; ++i) {
            qtraits->from_float((const float *) ((const char *) x->data + i * x->nb[1]),
                               (char *) quant.data() + i * quant_row, x->ne[0]);
        }
        weight_row = w->nb[1];
        int64_t rows_per_job = std::max<int64_t>(1, opt.chunk / weight_row);
        require(weight_row <= opt.chunk, "chunk must hold at least one full quantized row");
        jobs.clear(); read_jobs.clear(); resident_jobs.clear();
        resident_next = next_read = processed = 0;
        current_job = current_slot = -1;
        free_slots.clear(); ready_slots.clear();
        for (int i = 0; i < opt.slots; ++i) { free_slots.push_back(i); }
        for (int rank = 0; rank < ids->ne[0]; ++rank) {
            int expert = *(const int32_t *) ((const char *) ids->data + rank * ids->nb[0]);
            require(expert >= 0 && expert < w->ne[2], "expert index out of range");
            bool missing = rank >= ids->ne[0] - opt.misses && mode != "memory";
            for (int64_t row = 0; row < w->ne[1]; row += rows_per_job) {
                int64_t rows = std::min(rows_per_job, w->ne[1] - row);
                int jid = jobs.size();
                size_t off = specs.at(tensor_name).packed + expert * w->nb[2] + row * weight_row;
                jobs.push_back({rank, expert, row, rows, missing, off, size_t(rows) * weight_row});
                (missing ? read_jobs : resident_jobs).push_back(jid);
            }
        }
        if (mode == "bulk") {
            // Whole missing matrices are loaded before compute; use the same compute row chunks.
            for (void * p : bulk_buffers) { free(p); }
            bulk_buffers.assign(ids->ne[0], nullptr);
            for (int rank = 0; rank < ids->ne[0]; ++rank) {
                if (rank < ids->ne[0] - opt.misses) { continue; }
                int expert = *(const int32_t *) ((const char *) ids->data + rank * ids->nb[0]);
                size_t off = specs.at(tensor_name).packed + expert * w->nb[2];
                size_t base = off & ~size_t(4095), delta = off - base;
                size_t bytes = align_up(delta + w->nb[2]);
                require(posix_memalign(&bulk_buffers[rank], 4096, bytes) == 0, "allocate bulk buffer");
                double start = now_us();
                read_full(fd, bulk_buffers[rank], bytes, base);
                add_event({mode, "io", tensor_name, step, rank, expert, -1, 0, true,
                           start, now_us(), w->nb[2], bytes});
            }
        }
        if (mode == "stream") {
            { std::lock_guard<std::mutex> lock(mutex); running = true; }
            cv.notify_all();
        }
    }
    void select(ggml_tensor * dst) {
        auto * w = dst->src[0];
        current_slot = -1;
        if (processed == jobs.size()) { current_job = -1; return; }
        if (mode == "stream") {
            std::unique_lock<std::mutex> lock(mutex);
            if (ready_slots.empty() && resident_next == resident_jobs.size()) {
                cv.wait(lock, [&] { return !ready_slots.empty(); });
            }
            if (!ready_slots.empty()) {
                current_slot = ready_slots.front(); ready_slots.pop_front();
                current_job = slots[current_slot].job_id;
                current_data = (const char *) slots[current_slot].data + slots[current_slot].delta;
            } else {
                current_job = resident_jobs[resident_next++];
                const auto & j = jobs[current_job];
                current_data = (const char *) w->data + j.expert * w->nb[2] + j.row * weight_row;
            }
        } else {
            current_job = processed;
            const auto & j = jobs[current_job];
            if (j.missing && mode == "sync") {
                size_t delta = load_job(current_job, slots[0].data, 0);
                current_data = (const char *) slots[0].data + delta;
            } else if (j.missing && mode == "bulk") {
                size_t off = specs.at(tensor_name).packed + j.expert * w->nb[2];
                current_data = (const char *) bulk_buffers[j.rank] + (off & 4095) + j.row * weight_row;
            } else {
                current_data = (const char *) w->data + j.expert * w->nb[2] + j.row * weight_row;
            }
        }
        compute_start = now_us();
    }
    bool compute(ggml_tensor * dst, int ith, int nth) {
        if (!active || mode == "native" || dst->src[2]->ne[1] != 1 ||
            dst->src[1]->ne[2] != 1 || specs.count(dst->src[0]->name) == 0) { return false; }
        if (ith == 0) { begin(dst); }
        sync.wait(nth);
        for (;;) {
            if (ith == 0) { select(dst); }
            sync.wait(nth);
            if (current_job < 0) { break; }
            const auto & j = jobs[current_job];
            const char * q = (const char *) quant.data() + (j.rank % dst->src[1]->ne[1]) * quant_row;
            float * out = (float *) ((char *) dst->data + j.rank * dst->nb[1]);
            for (int64_t r = ith; r < j.rows; r += nth) {
                traits->vec_dot(dst->src[0]->ne[0], out + j.row + r, 0,
                                current_data + r * weight_row, 0, q, 0, 1);
            }
            sync.wait(nth);
            if (ith == 0) {
                add_event({mode, "compute", tensor_name, step, j.rank, j.expert, current_slot,
                           j.row, j.missing, compute_start, now_us(), j.bytes, 0});
                if (current_slot >= 0) {
                    { std::lock_guard<std::mutex> lock(mutex); free_slots.push_back(current_slot); }
                    cv.notify_all();
                }
                ++processed;
            }
        }
        if (ith == 0) {
            { std::lock_guard<std::mutex> lock(mutex); running = false; }
            double end = now_us();
            target_us += end - op_start;
            add_event({mode, "op", tensor_name, step, -1, -1, -1, 0, false, op_start, end, 0, 0});
        }
        sync.wait(nth);
        return true;
    }
};
static bool hook(ggml_tensor * dst, int ith, int nth, void * user) {
    return ((stream_context *) user)->compute(dst, ith, nth);
}
static uint64_t proc_read_bytes() {
    std::ifstream f("/proc/self/io");
    std::string key; uint64_t value;
    while (f >> key >> value) { if (key == "read_bytes:") { return value; } }
    return 0;
}
static std::vector<std::string> split(const std::string & s) {
    std::vector<std::string> out;
    size_t start = 0;
    do {
        size_t end = s.find(',', start);
        out.push_back(s.substr(start, end - start));
        if (end == std::string::npos) { break; }
        start = end + 1;
    } while (true);
    return out;
}
int main(int argc, char ** argv) {
    options opt;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--prepare") { opt.prepare = true; continue; }
        if (k == "--help") {
            std::puts("llama-moe-stream --model GGUF --pack FILE [--prepare] [--modes native,memory,bulk,sync,stream]\n"
                      "  --layer 10 --chunk-kib 128 --slots 4 --io-threads 2 --io-cpu-base 8 --misses 2 --threads 8 --tokens 12 --trace FILE");
            return 0;
        }
        require(i + 1 < argc, "missing option value");
        std::string v = argv[++i];
        if (k == "--model") { opt.model = v; }
        else if (k == "--pack") { opt.pack = v; }
        else if (k == "--modes") { opt.modes = v; }
        else if (k == "--trace") { opt.trace = v; }
        else if (k == "--prompt") { opt.prompt = v; }
        else if (k == "--layer") { opt.layer = std::stoi(v); }
        else if (k == "--chunk-kib") { opt.chunk = std::stoul(v) * 1024; }
        else if (k == "--slots") { opt.slots = std::stoi(v); }
        else if (k == "--io-threads") { opt.io_threads = std::stoi(v); }
        else if (k == "--io-cpu-base") { opt.io_cpu_base = std::stoi(v); }
        else if (k == "--misses") { opt.misses = std::stoi(v); }
        else if (k == "--threads") { opt.threads = std::stoi(v); }
        else if (k == "--tokens") { opt.tokens = std::stoi(v); }
        else { require(false, "unknown option"); }
    }
    require(!opt.model.empty() && !opt.pack.empty(), "model and pack required");
    require(opt.slots > 0 && opt.io_threads > 0 && opt.threads > 0 && opt.tokens > 2 &&
            opt.misses >= 0 && opt.misses <= 8 && opt.io_cpu_base >= -1 &&
            opt.io_cpu_base + opt.io_threads < CPU_SETSIZE, "invalid numeric option");
    auto specs = get_specs(opt);
    if (opt.prepare) { prepare_pack(opt, specs); return 0; }
    auto modes = split(opt.modes);
    require(modes[0] == "native", "first mode must be native for reference logits/tokens");
    for (auto & m : modes) {
        require(m == "native" || m == "memory" || m == "bulk" || m == "sync" || m == "stream", "unknown mode");
    }
    ggml_backend_load_all();
    llama_backend_init();
    auto mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    mp.load_mode = LLAMA_LOAD_MODE_MMAP;
    mp.use_extra_bufts = false;
    auto * model = llama_model_load_from_file(opt.model.c_str(), mp);
    require(model != nullptr, "load model");
    const auto * vocab = llama_model_get_vocab(model);
    int n_vocab = llama_vocab_n_tokens(vocab);
    int n_prompt = -llama_tokenize(vocab, opt.prompt.data(), opt.prompt.size(), nullptr, 0, true, true);
    require(n_prompt > 0, "tokenize prompt size");
    std::vector<llama_token> prompt(n_prompt);
    require(llama_tokenize(vocab, opt.prompt.data(), opt.prompt.size(), prompt.data(), n_prompt, true, true) == n_prompt,
            "tokenize prompt");
    auto cp = llama_context_default_params();
    cp.n_ctx = std::max(256, n_prompt + opt.tokens + 32);
    cp.n_batch = cp.n_ubatch = std::max(64, n_prompt);
    cp.n_threads = cp.n_threads_batch = opt.threads;
    auto * ctx = llama_init_from_model(model, cp);
    require(ctx != nullptr, "create context");
    std::vector<llama_token> teacher;
    std::vector<std::vector<float>> reference;
    bool passed = true;
    {
        stream_context stream(opt, specs);
        ggml_cpu_set_moe_stream_callback(hook, &stream);
        for (const auto & mode : modes) {
            stream.active = false;
            stream.mode = mode;
            stream.target_us = 0;
            stream.payload_bytes = stream.read_bytes = stream.io_calls = 0;
            for (void * p : stream.bulk_buffers) { free(p); }
            stream.bulk_buffers.clear();
            llama_memory_clear(llama_get_memory(ctx), true);
            require(llama_decode(ctx, llama_batch_get_one(prompt.data(), prompt.size())) == 0, "prefill");
            auto * logits = llama_get_logits_ith(ctx, -1);
            llama_token token = std::max_element(logits, logits + n_vocab) - logits;
            std::vector<double> times;
            double max_abs = 0, squared = 0;
            size_t compared = 0;
            int top1_mismatch = 0;
            uint64_t io_before = proc_read_bytes();
            for (int step = 0; step < opt.tokens; ++step) {
                if (mode == "native") { teacher.push_back(token); }
                else { token = teacher[step]; }
                stream.step = step;
                stream.active = true;
                double start = now_us();
                require(llama_decode(ctx, llama_batch_get_one(&token, 1)) == 0, "decode");
                times.push_back(now_us() - start);
                stream.active = false;
                logits = llama_get_logits_ith(ctx, -1);
                if (mode == "native") { reference.emplace_back(logits, logits + n_vocab); }
                else {
                    for (int j = 0; j < n_vocab; ++j) {
                        double d = double(logits[j]) - reference[step][j];
                        require(std::isfinite(d), "non-finite logits");
                        max_abs = std::max(max_abs, std::abs(d)); squared += d * d; ++compared;
                    }
                    auto & r = reference[step];
                    top1_mismatch += (std::max_element(logits, logits + n_vocab) - logits !=
                                      std::max_element(r.begin(), r.end()) - r.begin());
                }
                token = std::max_element(logits, logits + n_vocab) - logits;
            }
            uint64_t io_after = proc_read_bytes();
            size_t useful = stream.payload_bytes, physical = stream.read_bytes, calls = stream.io_calls;
            std::vector<double> steady(times.begin() + 2, times.end());
            std::sort(steady.begin(), steady.end());
            double mean = std::accumulate(steady.begin(), steady.end(), 0.0) / steady.size();
            bool ok = max_abs <= 0.001 && top1_mismatch == 0;
            passed = passed && ok;
            std::printf("RESULT {\"mode\":\"%s\",\"chunk\":%zu,\"slots\":%d,\"misses\":%d,\"tokens\":%d,"
                        "\"mean_decode_us\":%.3f,\"median_decode_us\":%.3f,\"p95_decode_us\":%.3f,\"tps\":%.5f,"
                        "\"target_total_us\":%.3f,\"io_payload_bytes\":%zu,\"io_padded_bytes\":%zu,\"io_calls\":%zu,"
                        "\"proc_read_bytes\":%llu,\"max_abs_logits\":%.9g,\"rmse_logits\":%.9g,\"top1_mismatch\":%d,\"pass\":%s}\n",
                        mode.c_str(), opt.chunk, opt.slots, opt.misses, opt.tokens, mean, steady[steady.size()/2],
                        steady[std::min(steady.size()-1, size_t(std::ceil(steady.size()*0.95)-1))], 1e6 / mean,
                        stream.target_us, useful, physical, calls, (unsigned long long) (io_after - io_before),
                        max_abs, compared ? std::sqrt(squared / compared) : 0, top1_mismatch, ok ? "true" : "false");
            std::fflush(stdout);
        }
        ggml_cpu_set_moe_stream_callback(nullptr, nullptr);
        if (!opt.trace.empty()) {
            std::ofstream out(opt.trace);
            require(out.good(), "open trace");
            out.precision(17);
            for (const auto & e : stream.events) {
                out << "{\"mode\":\"" << e.mode << "\",\"kind\":\"" << e.kind << "\",\"tensor\":\"" << e.tensor
                    << "\",\"step\":" << e.step << ",\"rank\":" << e.rank << ",\"expert\":" << e.expert
                    << ",\"slot\":" << e.slot_id << ",\"row\":" << e.row << ",\"missing\":" << (e.missing ? "true" : "false")
                    << ",\"start_us\":" << e.start << ",\"end_us\":" << e.end << ",\"useful\":" << e.useful
                    << ",\"physical\":" << e.physical << "}\n";
            }
        }
    }
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return passed ? 0 : 2;
}
