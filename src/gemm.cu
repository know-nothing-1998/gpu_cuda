#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "build_info.h"
#include "kernel_registry.cuh"
#include "tuned_config.h"
#include "warptiled.cuh"

namespace {
constexpr double kAtol = 1e-3, kRtol = 1e-4;
bool cleanup_failed = false;

void cuda_check(cudaError_t status, const char* expr, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(expr) + " at line " + std::to_string(line) +
                                 ": " + cudaGetErrorString(status));
}
void blas_check(cublasStatus_t status, const char* expr, int line) {
    if (status != CUBLAS_STATUS_SUCCESS)
        throw std::runtime_error(std::string(expr) + " at line " + std::to_string(line) +
                                 ": cuBLAS status " + std::to_string(static_cast<int>(status)));
}
#define CUDA_CHECK(expr) cuda_check((expr), #expr, __LINE__)
#define BLAS_CHECK(expr) blas_check((expr), #expr, __LINE__)
void cleanup(cudaError_t status, const char* what) noexcept {
    if (status != cudaSuccess) {
        std::cerr << "cleanup error: " << what << ": " << cudaGetErrorString(status) << '\n';
        cleanup_failed = true;
    }
}

struct Stream {
    cudaStream_t value = nullptr;
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&value, cudaStreamNonBlocking)); }
    ~Stream() { if (value) cleanup(cudaStreamDestroy(value), "stream"); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;
};
struct Handle {
    cublasHandle_t value = nullptr;
    Handle() { BLAS_CHECK(cublasCreate(&value)); }
    ~Handle() {
        if (value && cublasDestroy(value) != CUBLAS_STATUS_SUCCESS) {
            std::cerr << "cleanup error: cuBLAS handle\n";
            cleanup_failed = true;
        }
    }
    Handle(const Handle&) = delete;
    Handle& operator=(const Handle&) = delete;
};
struct Event {
    cudaEvent_t value = nullptr;
    Event() { CUDA_CHECK(cudaEventCreate(&value)); }
    ~Event() { if (value) cleanup(cudaEventDestroy(value), "event"); }
    Event(const Event&) = delete;
    Event& operator=(const Event&) = delete;
};
struct DeviceBuffer {
    float* value = nullptr;
    explicit DeviceBuffer(size_t bytes) { CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&value), bytes)); }
    ~DeviceBuffer() { if (value) cleanup(cudaFree(value), "device buffer"); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};
struct Context {
    const WarptiledVariant* warp_variant = nullptr;
    Stream stream;
    Handle blas;
    cudaDeviceProp prop{};
    int runtime = 0, driver = 0, blas_version = 0;
    cublasMath_t math_mode{};
    explicit Context(const WarptiledVariant* variant = nullptr) : warp_variant(variant) {
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
        CUDA_CHECK(cudaDriverGetVersion(&driver));
        BLAS_CHECK(cublasSetStream(blas.value, stream.value));
        BLAS_CHECK(cublasSetPointerMode(blas.value, CUBLAS_POINTER_MODE_HOST));
        BLAS_CHECK(cublasSetMathMode(blas.value, CUBLAS_PEDANTIC_MATH));
        BLAS_CHECK(cublasGetMathMode(blas.value, &math_mode));
        BLAS_CHECK(cublasGetVersion(blas.value, &blas_version));
    }
};

struct Shape { int m, n, k; };
struct Options {
    std::string mode = "info", kernel = "cublas", dtype = "fp32", csv;
    Shape shape{0, 0, 0};
    uint32_t seed = 42;
    int warmup = 20, iters = 100, rounds = 5;
    bool has_shape = false, help = false;
    WarptiledConfig warp_config = default_warptiled_config();
    const WarptiledVariant* warp_variant = nullptr;
};
uint64_t unsigned_value(const std::string& s, uint64_t limit, const std::string& key) {
    if (s.empty() || s.find_first_not_of("0123456789") != std::string::npos)
        throw std::runtime_error(key + " requires an unsigned integer");
    uint64_t value = 0;
    for (char ch : s) {
        const auto digit = static_cast<uint64_t>(ch - '0');
        if (value > (limit - digit) / 10) throw std::runtime_error(key + " is out of range");
        value = value * 10 + digit;
    }
    return value;
}
Options parse(int argc, char** argv) {
    Options o;
    std::set<std::string> seen;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        if (key == "--help" || key == "-h") { o.help = true; continue; }
        if (!seen.insert(key).second) throw std::runtime_error("duplicate option: " + key);
        if (i + 1 == argc) throw std::runtime_error("missing value for " + key);
        const std::string value = argv[++i];
        if (key == "--mode") o.mode = value;
        else if (key == "--kernel") o.kernel = value;
        else if (key == "--dtype") o.dtype = value;
        else if (key == "--csv") o.csv = value;
        else if (key == "--seed") o.seed = static_cast<uint32_t>(unsigned_value(value, UINT32_MAX, key));
        else {
            if (key != "--m" && key != "--n" && key != "--k" && key != "--warmup" &&
                key != "--iters" && key != "--rounds" && key != "--bm" && key != "--bn" &&
                key != "--bk" && key != "--wm" && key != "--wn" && key != "--tm" &&
                key != "--tn" && key != "--wniter")
                throw std::runtime_error("unknown option: " + key);
            const int v = static_cast<int>(unsigned_value(value, INT32_MAX, key));
            if (v == 0 && key != "--warmup") throw std::runtime_error(key + " must be positive");
            if (key == "--m") o.shape.m = v;
            else if (key == "--n") o.shape.n = v;
            else if (key == "--k") o.shape.k = v;
            else if (key == "--warmup") o.warmup = v;
            else if (key == "--iters") o.iters = v;
            else if (key == "--rounds") o.rounds = v;
            else if (key == "--bm") o.warp_config.bm = v;
            else if (key == "--bn") o.warp_config.bn = v;
            else if (key == "--bk") o.warp_config.bk = v;
            else if (key == "--wm") o.warp_config.wm = v;
            else if (key == "--wn") o.warp_config.wn = v;
            else if (key == "--tm") o.warp_config.tm = v;
            else if (key == "--tn") o.warp_config.tn = v;
            else o.warp_config.wniter = v;
        }
    }
    if (o.help) return o;
    if (o.mode != "info" && o.mode != "list" && o.mode != "check" && o.mode != "bench" && o.mode != "profile")
        throw std::runtime_error("unknown mode: " + o.mode);
    if (o.dtype != "fp32") throw std::runtime_error("unsupported dtype: " + o.dtype + "; M0 supports fp32 only");
    int warp_parameters = 0;
    for (const char* key : {"--bm", "--bn", "--bk", "--wm", "--wn", "--tm", "--tn", "--wniter"})
        warp_parameters += static_cast<int>(seen.count(key));
    if (warp_parameters && o.kernel != "warptiled")
        throw std::runtime_error("tile parameter selection requires --kernel warptiled");
    if (warp_parameters != 0 && warp_parameters != 8)
        throw std::runtime_error("specify all of --bm --bn --bk --wm --wn --tm --tn --wniter or none");
    o.warp_variant = find_warptiled_variant(o.warp_config);
    if (!o.warp_variant)
        throw std::runtime_error("warptiled configuration is not compiled; use --mode list --kernel warptiled "
                                 "to see compiled configurations. Add the tuple to benchmarks/warptiling.json "
                                 "or set GEMM_WARP_* in CMake, then rebuild.");
    const auto* custom = find_custom_kernel(o.kernel);
    if (o.kernel != "cublas" && !custom)
        throw std::runtime_error("unknown/unimplemented kernel: " + o.kernel);
    if (custom && !custom->available() && o.mode != "info" && o.mode != "list")
        throw std::runtime_error(o.kernel + " is not implemented; complete " + custom->source +
                                 " and enable " + o.kernel + "_available()");
    const int dims = (o.shape.m > 0) + (o.shape.n > 0) + (o.shape.k > 0);
    if (dims != 0 && dims != 3) throw std::runtime_error("specify all of --m --n --k or none");
    o.has_shape = dims == 3;
    if ((o.mode == "bench" || o.mode == "profile") && !o.has_shape)
        throw std::runtime_error(o.mode + " requires --m --n --k");
    if (o.shape.k > 4096)
        throw std::runtime_error("K > 4096 is outside the current numerical validation contract");
    if ((o.mode == "info" || o.mode == "list") && o.has_shape)
        throw std::runtime_error(o.mode + " does not accept a shape");
    if (!o.csv.empty() && o.mode != "bench") throw std::runtime_error("--csv is supported only in bench mode");
    if (!o.csv.empty() && std::filesystem::exists(o.csv))
        throw std::runtime_error("CSV already exists; choose a new path: " + o.csv);
    if (o.mode == "profile") {
        if (!seen.count("--iters")) o.iters = 1;
        if (!seen.count("--rounds")) o.rounds = 1;
        if (o.rounds != 1) throw std::runtime_error("profile requires --rounds 1");
    }
    return o;
}

size_t checked_mul(size_t a, size_t b) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a)
        throw std::runtime_error("matrix size/index overflow");
    return a * b;
}
size_t checked_add(size_t a, size_t b) {
    if (b > std::numeric_limits<size_t>::max() - a) throw std::runtime_error("total allocation overflow");
    return a + b;
}
struct Layout {
    size_t a, b, c, a_bytes, b_bytes, c_bytes;
    explicit Layout(Shape s)
        : a(checked_mul(s.m, s.k)), b(checked_mul(s.k, s.n)), c(checked_mul(s.m, s.n)),
          a_bytes(checked_mul(a, sizeof(float))), b_bytes(checked_mul(b, sizeof(float))),
          c_bytes(checked_mul(c, sizeof(float))) {}
    size_t bytes(int outputs) const {
        return checked_add(checked_add(a_bytes, b_bytes), checked_mul(c_bytes, outputs));
    }
};
void check_memory(const Layout& layout, int outputs) {
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    if (layout.bytes(outputs) > free_bytes)
        throw std::runtime_error("insufficient GPU memory: requested " + std::to_string(layout.bytes(outputs)) +
                                 " bytes, free " + std::to_string(free_bytes));
}

// Fixed mapping avoids implementation-dependent uniform_real_distribution.
// mt19937 upper 24 bits -> an exactly representable FP32 value in [-1, 1).
struct Inputs { std::vector<float> a, b; };
Inputs generate(Shape s, const Layout& l, uint32_t seed, const std::string& pattern) {
    Inputs in{std::vector<float>(l.a), std::vector<float>(l.b)};
    std::mt19937 rng(seed);
    auto fill = [&](std::vector<float>& x) {
        for (float& v : x) {
            if (pattern == "integer") v = static_cast<float>(static_cast<int>(rng() % 5) - 2);
            else v = static_cast<float>(rng() >> 8) * 0x1p-23f - 1.0f;
        }
    };
    fill(in.a); fill(in.b); // One RNG stream, A first then B.
    if (pattern == "zero") std::fill(in.a.begin(), in.a.end(), 0.0f);
    else if (pattern == "identity") {
        std::fill(in.a.begin(), in.a.end(), 0.0f);
        for (int i = 0; i < std::min(s.m, s.k); ++i) in.a[static_cast<size_t>(i) * s.k + i] = 1.0f;
    }
    return in;
}

void launch(Context& ctx, const std::string& kernel, Shape s, const float* a, const float* b, float* c) {
    if (kernel == "cublas") {
        const float alpha = 1.0f, beta = 0.0f;
        // Row-major C=A*B is column-major C^T=B^T*A^T, without a copy.
        BLAS_CHECK(cublasGemmEx(ctx.blas.value, CUBLAS_OP_N, CUBLAS_OP_N,
            s.n, s.m, s.k, &alpha, b, CUDA_R_32F, s.n, a, CUDA_R_32F, s.k,
            &beta, c, CUDA_R_32F, s.n, CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT));
    } else if (const auto* custom = find_custom_kernel(kernel)) {
        if (!custom->available()) throw std::runtime_error(kernel + " is not implemented");
        const auto selected_launch = kernel == "warptiled" && ctx.warp_variant
            ? ctx.warp_variant->launch : custom->launch;
        CUDA_CHECK(selected_launch(a, b, c, s.m, s.n, s.k, ctx.stream.value));
    } else throw std::runtime_error("unregistered kernel: " + kernel);
    CUDA_CHECK(cudaGetLastError());
}

struct Reference {
    std::vector<size_t> positions;
    std::vector<double> values;
    std::string scope;
};
Reference cpu_reference(Shape s, const Layout& l, const Inputs& in) {
    Reference ref;
    // Bound CPU work; large cases always have an independent fixed sample.
    const bool full = l.c <= 65536 && static_cast<double>(l.c) * s.k <= 8000000;
    ref.scope = full ? "cpu_double_full" : "cpu_double_sample";
    if (full) {
        ref.positions.resize(l.c);
        std::iota(ref.positions.begin(), ref.positions.end(), size_t{0});
    } else {
        std::set<size_t> positions;
        for (int r : {0, std::min(1, s.m - 1), s.m / 3, s.m / 2, std::max(0, s.m - 2), s.m - 1})
            for (int c : {0, std::min(1, s.n - 1), s.n / 3, s.n / 2, std::max(0, s.n - 2), s.n - 1})
                positions.insert(static_cast<size_t>(r) * s.n + c);
        for (size_t i = 0; i < 17; ++i) {
            // Multiplication is safe because c_bytes was checked above; divide first anyway.
            positions.insert((l.c - 1) / 16 * i + (l.c - 1) % 16 * i / 16);
        }
        ref.positions.assign(positions.begin(), positions.end());
    }
    ref.values.reserve(ref.positions.size());
    for (size_t pos : ref.positions) {
        const size_t row = pos / s.n, col = pos % s.n;
        double sum = 0;
        for (int p = 0; p < s.k; ++p)
            sum += static_cast<double>(in.a[row * s.k + p]) * in.b[static_cast<size_t>(p) * s.n + col];
        ref.values.push_back(sum);
    }
    return ref;
}
struct Metrics {
    std::string scope;
    size_t compared = 0, failed = 0, nonfinite = 0;
    double max_abs = 0, relative_frob = 0;
    bool valid() const { return failed == 0 && nonfinite == 0; }
};
template <typename Index, typename Value>
Metrics compare(const std::vector<float>& actual, size_t count, const std::string& scope,
                Index index, Value value) {
    Metrics result;
    result.scope = scope; result.compared = count;
    for (float v : actual) if (!std::isfinite(v)) ++result.nonfinite;
    result.failed = result.nonfinite;
    double err2 = 0, ref2 = 0;
    for (size_t i = 0; i < count; ++i) {
        const double a = actual[index(i)], r = value(i);
        if (!std::isfinite(a)) continue; // Already counted by the full finite scan.
        if (!std::isfinite(r)) { ++result.failed; continue; }
        const double error = std::abs(a - r);
        result.max_abs = std::max(result.max_abs, error);
        err2 += error * error; ref2 += r * r;
        if (error > kAtol + kRtol * std::abs(r)) ++result.failed;
    }
    result.relative_frob = std::sqrt(err2) / std::max(std::sqrt(ref2), 1e-30);
    if (result.nonfinite) result.max_abs = result.relative_frob = std::numeric_limits<double>::infinity();
    return result;
}
Metrics compare_cpu(const std::vector<float>& output, const Reference& ref) {
    return compare(output, ref.positions.size(), ref.scope,
        [&](size_t i) { return ref.positions[i]; }, [&](size_t i) { return ref.values[i]; });
}
struct Validation {
    Metrics primary, cpu;
    bool valid() const { return primary.valid() && cpu.valid(); }
};
void show_validation(const std::string& kernel, Shape s, const std::string& pattern, const Validation& v) {
    std::cout << "check kernel=" << kernel << " shape=" << s.m << 'x' << s.n << 'x' << s.k
        << " pattern=" << pattern << " status=" << (v.valid() ? "valid" : "invalid")
        << " reference=" << v.primary.scope << " compared=" << v.primary.compared
        << " max_abs=" << v.primary.max_abs << " relative_frob=" << v.primary.relative_frob
        << " failed=" << v.primary.failed << " nonfinite=" << v.primary.nonfinite
        << " cpu_scope=" << v.cpu.scope << " cpu_compared=" << v.cpu.compared
        << " cpu_max_abs=" << v.cpu.max_abs << " cpu_relative_frob=" << v.cpu.relative_frob
        << " cpu_failed=" << v.cpu.failed << '\n';
}
struct Workload {
    Shape shape;
    Layout layout;
    Inputs inputs;
    DeviceBuffer a, b, target;
    Workload(Context& ctx, Shape s, uint32_t seed, const std::string& pattern)
        : shape(s), layout(s), inputs(generate(s, layout, seed, pattern)),
          a(layout.a_bytes), b(layout.b_bytes), target(layout.c_bytes) {
        CUDA_CHECK(cudaMemcpyAsync(a.value, inputs.a.data(), layout.a_bytes, cudaMemcpyHostToDevice, ctx.stream.value));
        CUDA_CHECK(cudaMemcpyAsync(b.value, inputs.b.data(), layout.b_bytes, cudaMemcpyHostToDevice, ctx.stream.value));
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream.value));
    }
};
std::vector<float> checked_output(Context& ctx, Workload& w, const std::string& kernel, float* output) {
    // All-one FP32 bit pattern is NaN; catches beta=0 violations and unwritten C.
    CUDA_CHECK(cudaMemsetAsync(output, 0xff, w.layout.c_bytes, ctx.stream.value));
    launch(ctx, kernel, w.shape, w.a.value, w.b.value, output);
    std::vector<float> host(w.layout.c);
    CUDA_CHECK(cudaMemcpyAsync(host.data(), output, w.layout.c_bytes, cudaMemcpyDeviceToHost, ctx.stream.value));
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream.value));
    return host;
}
std::vector<Validation> validate(Context& ctx, Workload& w, const std::string& kernel,
                                 float* baseline, const std::string& pattern) {
    const auto ref = cpu_reference(w.shape, w.layout, w.inputs);
    const auto target = checked_output(ctx, w, kernel, w.target.value);
    Validation self;
    self.cpu = compare_cpu(target, ref);
    self.primary = self.cpu;
    std::vector<Validation> values;
    if (kernel != "cublas") {
        const auto lib = checked_output(ctx, w, "cublas", baseline);
        Validation bv;
        bv.cpu = compare_cpu(lib, ref); bv.primary = bv.cpu;
        // CPU is full on small cases; also require an all-element library comparison.
        self.primary = compare(target, lib.size(), "cublas_full",
            [](size_t i) { return i; }, [&](size_t i) { return static_cast<double>(lib[i]); });
        values = {self, bv};
        show_validation(kernel, w.shape, pattern, self);
        show_validation("cublas", w.shape, pattern, bv);
    } else {
        values = {self};
        show_validation(kernel, w.shape, pattern, self);
    }
    return values;
}

double event_batch(Context& ctx, Workload& w, const std::string& kernel, float* c, int iters,
                   Event& start, Event& stop) {
    CUDA_CHECK(cudaEventRecord(start.value, ctx.stream.value));
    for (int i = 0; i < iters; ++i) launch(ctx, kernel, w.shape, w.a.value, w.b.value, c);
    CUDA_CHECK(cudaEventRecord(stop.value, ctx.stream.value));
    CUDA_CHECK(cudaEventSynchronize(stop.value));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, start.value, stop.value));
    if (!std::isfinite(elapsed) || elapsed <= 0) throw std::runtime_error("non-positive/invalid event duration; increase --iters");
    return static_cast<double>(elapsed) / iters;
}
struct Timing {
    std::vector<double> samples;
    double median = 0, minimum = 0, maximum = 0;
    void summarize() {
        auto sorted = samples;
        std::sort(sorted.begin(), sorted.end());
        minimum = sorted.front(); maximum = sorted.back();
        median = sorted[sorted.size() / 2];
        if (sorted.size() % 2 == 0) median = (median + sorted[sorted.size() / 2 - 1]) / 2;
    }
};
std::string csv_quote(const std::string& s) {
    std::string out = "\"";
    for (char c : s) { if (c == '"') out += '"'; out += c; }
    return out + '"';
}
std::string number(double v) {
    std::ostringstream out; out << std::setprecision(12) << v; return out.str();
}
std::string warp_configuration(const WarptiledConfig& config) {
    std::ostringstream out;
    out << "BM=" << config.bm << ";BN=" << config.bn
        << ";BK=" << config.bk << ";WM=" << config.wm
        << ";WN=" << config.wn << ";TM=" << config.tm
        << ";TN=" << config.tn << ";WNITER=" << config.wniter
        << ";THREADS=" << 32 * (config.bm / config.wm) * (config.bn / config.wn);
    return out.str();
}
void write_csv(const Options& o, const Context& ctx, const std::vector<Validation>& v,
               const std::vector<Timing>& times, bool valid) {
    if (o.csv.empty()) return;
    std::ofstream out(o.csv);
    if (!out) throw std::runtime_error("cannot open CSV: " + o.csv);
    out << "requested_kernel,actual_kernel,m,n,k,input_dtype,accum_dtype,output_dtype,seed,generator,status,"
           "reference_scope,compared,max_abs,relative_frob,failed,nonfinite,cpu_scope,cpu_compared,cpu_max_abs,cpu_relative_frob,cpu_failed,"
           "warmup,iters,rounds,median_ms,min_ms,max_ms,gflops,cublas_over_self,round_mean_ms,source_sha256,build_type,"
           "cuda_architectures,gpu,cuda_runtime,cuda_driver,cublas_version,math_mode,compute_type,"
           "tuned_bm,tuned_bn,tuned_bk,tuned_tm,tuned_tn,warptiled_config\n";
    for (size_t i = 0; i < v.size(); ++i) {
        const auto& a = v[i];
        const std::string kernel = i == 0 ? o.kernel : "cublas";
        std::string samples;
        if (valid) for (double t : times[i].samples) { if (!samples.empty()) samples += ';'; samples += number(t); }
        const double flop = 2.0 * o.shape.m * o.shape.n * o.shape.k;
        out << o.kernel << ',' << kernel << ',' << o.shape.m << ',' << o.shape.n << ',' << o.shape.k
            << ",fp32,fp32,fp32," << o.seed << ",mt19937_u24_v1," << (valid ? "valid" : "invalid")
            << ',' << a.primary.scope << ',' << a.primary.compared << ',' << number(a.primary.max_abs)
            << ',' << number(a.primary.relative_frob) << ',' << a.primary.failed << ',' << a.primary.nonfinite
            << ',' << a.cpu.scope << ',' << a.cpu.compared << ',' << number(a.cpu.max_abs)
            << ',' << number(a.cpu.relative_frob) << ',' << a.cpu.failed
            << ',' << o.warmup << ',' << o.iters << ',' << o.rounds
            << ',' << (valid ? number(times[i].median) : "NA")
            << ',' << (valid ? number(times[i].minimum) : "NA")
            << ',' << (valid ? number(times[i].maximum) : "NA")
            << ',' << (valid ? number(flop / (1e6 * times[i].median)) : "NA")
            << ',' << (valid ? number(times.back().median / times[i].median) : "NA")
            << ',' << csv_quote(valid ? samples : "NA") << ',' << GEMM_SOURCE_SHA256 << ',' << GEMM_BUILD_CONFIG
            << ',' << csv_quote(GEMM_CUDA_ARCHITECTURES) << ',' << csv_quote(ctx.prop.name)
            << ',' << ctx.runtime << ',' << ctx.driver << ',' << ctx.blas_version << ',' << static_cast<int>(ctx.math_mode)
            << ",CUBLAS_COMPUTE_32F_PEDANTIC";
        if (kernel == "tuned")
            out << ',' << gemm_tuned_config::BM << ',' << gemm_tuned_config::BN
                << ',' << gemm_tuned_config::BK << ',' << gemm_tuned_config::TM
                << ',' << gemm_tuned_config::TN;
        else
            out << ",,,,,";
        out << ',' << csv_quote(kernel == "warptiled" ? warp_configuration(o.warp_variant->config) : "") << '\n';
    }
    out.flush();
    if (!out) throw std::runtime_error("failed to write CSV: " + o.csv);
}

bool run_case(Context& ctx, const Options& o, Shape s, const std::string& pattern) {
    const bool with_baseline = o.kernel != "cublas" && o.mode != "profile";
    check_memory(Layout(s), with_baseline ? 2 : 1);
    Workload w(ctx, s, o.seed, pattern);
    if (o.mode == "profile") {
        for (int i = 0; i < o.warmup; ++i) launch(ctx, o.kernel, s, w.a.value, w.b.value, w.target.value);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream.value));
        for (int i = 0; i < o.iters; ++i) launch(ctx, o.kernel, s, w.a.value, w.b.value, w.target.value);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream.value));
        std::cout << "profile requested=" << o.kernel << " actual=" << o.kernel
                  << " warmup=" << o.warmup << " target_operations=" << o.iters
                  << " rounds=1 validation=not_run (run check separately)\n";
        return true;
    }
    // The library and self outputs are independent buffers, with identical A/B.
    // Allocate the baseline only when there is an actual second implementation.
    std::vector<Validation> validation;
    std::vector<Timing> times(with_baseline ? 2 : 1);
    auto work = [&](float* baseline) {
        validation = validate(ctx, w, o.kernel, baseline, pattern);
        const bool valid = std::all_of(validation.begin(), validation.end(), [](const Validation& v) { return v.valid(); });
        if (o.mode != "bench") return valid;
        if (!valid) { write_csv(o, ctx, validation, times, false); return false; }
        for (int i = 0; i < o.warmup; ++i) {
            launch(ctx, o.kernel, s, w.a.value, w.b.value, w.target.value);
            if (with_baseline) launch(ctx, "cublas", s, w.a.value, w.b.value, baseline);
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream.value));
        Event start, stop;
        for (int r = 0; r < o.rounds; ++r) {
            // A/B, B/A, A/B... to reduce a systematic first/second bias.
            for (size_t j = 0; j < times.size(); ++j) {
                const size_t i = r % 2 ? times.size() - 1 - j : j;
                const auto& kernel = i == 0 ? o.kernel : std::string("cublas");
                const double ms = event_batch(ctx, w, kernel, i == 0 ? w.target.value : baseline, o.iters, start, stop);
                times[i].samples.push_back(ms);
                std::cout << "round=" << r + 1 << " kernel=" << kernel << " mean_ms=" << ms
                          << " batch_ms=" << ms * o.iters << '\n';
            }
        }
        for (auto& timing : times) timing.summarize();
        for (size_t i = 0; i < times.size(); ++i) {
            const auto& t = times[i];
            std::cout << "bench kernel=" << (i == 0 ? o.kernel : "cublas") << " status=valid"
                << " median_ms=" << t.median << " min_ms=" << t.minimum << " max_ms=" << t.maximum
                << " gflops=" << (2.0 * s.m * s.n * s.k / (1e6 * t.median))
                << " cublas_over_self=" << times.back().median / t.median << '\n';
            if (t.minimum * o.iters < 50 || t.maximum * o.iters > 200)
                std::cout << "timing_note: batch outside 50-200 ms; adjust --iters equally for both kernels and record it\n";
        }
        write_csv(o, ctx, validation, times, true);
        return true;
    };
    if (with_baseline) { DeviceBuffer baseline(w.layout.c_bytes); return work(baseline.value); }
    return work(nullptr);
}

void info(const Context& ctx) {
    const auto& p = ctx.prop;
    const auto warp = ctx.warp_variant ? ctx.warp_variant->config : default_warptiled_config();
    int device = 0, device_count = 0, sm_clock_khz = 0, memory_clock_khz = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    // CUDA 13 no longer exposes clocks in cudaDeviceProp. These attributes
    // report device clock properties, not live clocks during a workload.
    CUDA_CHECK(cudaDeviceGetAttribute(&sm_clock_khz, cudaDevAttrClockRate, device));
    CUDA_CHECK(cudaDeviceGetAttribute(&memory_clock_khz, cudaDevAttrMemoryClockRate, device));
    size_t free_bytes = 0, total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    std::cout << "device=" << device << "\ncuda_device_count=" << device_count
        << "\ngpu=" << p.name << "\ncompute_capability=" << p.major << '.' << p.minor
        << "\nsm_count=" << p.multiProcessorCount << "\nglobal_memory_bytes=" << p.totalGlobalMem
        << "\nfree_memory_bytes=" << free_bytes << "\nruntime_total_memory_bytes=" << total_bytes
        << "\nmemory_bus_width_bits=" << p.memoryBusWidth
        << "\nreported_sm_clock_khz=" << sm_clock_khz
        << "\nreported_memory_clock_khz=" << memory_clock_khz
        << "\nl2_bytes=" << p.l2CacheSize
        << "\npersisting_l2_max_bytes=" << p.persistingL2CacheMaxSize
        << "\nconstant_memory_bytes=" << p.totalConstMem
        << "\nglobal_l1_cache_supported=" << p.globalL1CacheSupported
        << "\nlocal_l1_cache_supported=" << p.localL1CacheSupported
        << "\nwarp_size=" << p.warpSize << "\nmax_threads_per_block=" << p.maxThreadsPerBlock
        << "\nmax_threads_per_sm=" << p.maxThreadsPerMultiProcessor
        << "\nmax_warps_per_sm=" << p.maxThreadsPerMultiProcessor / p.warpSize
        << "\nmax_blocks_per_sm=" << p.maxBlocksPerMultiProcessor
        << "\nmax_block_dims=" << p.maxThreadsDim[0] << 'x' << p.maxThreadsDim[1] << 'x' << p.maxThreadsDim[2]
        << "\nregisters_per_block=" << p.regsPerBlock << "\nregisters_per_sm=" << p.regsPerMultiprocessor
        << "\nregister_bytes_per_sm=" << static_cast<size_t>(p.regsPerMultiprocessor) * sizeof(uint32_t)
        << "\nshared_bytes_per_block=" << p.sharedMemPerBlock << "\nshared_bytes_per_sm=" << p.sharedMemPerMultiprocessor
        << "\nshared_bytes_per_block_optin=" << p.sharedMemPerBlockOptin
        << "\nshared_reserved_bytes_per_block=" << p.reservedSharedMemPerBlock
        << "\nmax_grid=" << p.maxGridSize[0] << 'x' << p.maxGridSize[1] << 'x' << p.maxGridSize[2]
        << "\nasync_engine_count=" << p.asyncEngineCount
        << "\nconcurrent_kernels=" << p.concurrentKernels
        << "\nstream_priorities_supported=" << p.streamPrioritiesSupported
        << "\nunified_addressing=" << p.unifiedAddressing
        << "\nmanaged_memory=" << p.managedMemory
        << "\nconcurrent_managed_access=" << p.concurrentManagedAccess
        << "\ncan_map_host_memory=" << p.canMapHostMemory
        << "\nintegrated_gpu=" << p.integrated << "\necc_enabled=" << p.ECCEnabled
        << "\ntcc_driver=" << p.tccDriver
        << "\ncuda_runtime=" << ctx.runtime << "\ncuda_driver=" << ctx.driver
        << "\ncublas_version=" << ctx.blas_version << "\nmath_mode=" << static_cast<int>(ctx.math_mode)
        << " (CUBLAS_PEDANTIC_MATH)\ncompute_type=CUBLAS_COMPUTE_32F_PEDANTIC"
        << "\ninput_dtype=fp32\naccum_dtype=fp32\noutput_dtype=fp32\nalpha=1\nbeta=0"
        << "\ngenerator=mt19937_u24_v1\nsource_sha256=" << GEMM_SOURCE_SHA256
        << "\nbuild_type=" << GEMM_BUILD_CONFIG << "\ncuda_architectures=" << GEMM_CUDA_ARCHITECTURES
        << "\ncuda_compiler=" << GEMM_CUDA_COMPILER << "\nhost_compiler=" << GEMM_HOST_COMPILER
        << "\ntuned_bm=" << gemm_tuned_config::BM << "\ntuned_bn=" << gemm_tuned_config::BN
        << "\ntuned_bk=" << gemm_tuned_config::BK << "\ntuned_tm=" << gemm_tuned_config::TM
        << "\ntuned_tn=" << gemm_tuned_config::TN
        << "\nwarptiled_bm=" << warp.bm << "\nwarptiled_bn=" << warp.bn
        << "\nwarptiled_bk=" << warp.bk << "\nwarptiled_wm=" << warp.wm
        << "\nwarptiled_wn=" << warp.wn << "\nwarptiled_tm=" << warp.tm
        << "\nwarptiled_tn=" << warp.tn << "\nwarptiled_wniter=" << warp.wniter
        << "\nwarptiled_threads=" << 32 * (warp.bm / warp.wm) * (warp.bn / warp.wn)
        << "\nwarptiled_config=" << warp_configuration(warp)
        << "\nwarptiled_compiled_configs=" << warptiled_variants().size()
        << "\ncublas_available=true\n";
    for (const auto& kernel : custom_kernels())
        std::cout << kernel.name << "_available=" << (kernel.available() ? "true" : "false") << '\n';
}
void list_kernels() {
    std::cout << "kernel=cublas available=true source=src/gemm.cu description=\"strict FP32 cuBLAS baseline\"\n";
    for (const auto& kernel : custom_kernels())
        std::cout << "kernel=" << kernel.name << " available=" << (kernel.available() ? "true" : "false")
                  << " source=" << kernel.source << " description=\"" << kernel.description << "\"\n";
}
int run(int argc, char** argv) {
    const Options o = parse(argc, argv);
    if (o.help) {
        std::cout << "gemm --mode list|info|check|bench|profile [--kernel NAME] [--dtype fp32]\n"
                     "     [--m M --n N --k K] [--seed 42] [--warmup 20] [--iters 100] [--rounds 5]\n"
                     "     [--bm BM --bn BN --bk BK --wm WM --wn WN --tm TM --tn TN --wniter WNITER]\n"
                     "     [--csv NEW_FILE] (bench only; does not overwrite)\n"
                     "Tile options select a compiled warptiled configuration; supply all eight or none.\n"
                     "Use --mode list --kernel warptiled to see compiled configurations.\n"
                     "check: omit all dimensions for the fixed suite. bench/profile: all dimensions required.\n"
                     "profile defaults: warmup=20, iters=1, rounds=1; no correctness/reference operations.\n"
                     "list: show registered kernels without initializing CUDA. K <= 4096.\n"
                     "Unavailable implementations fail explicitly; no fallback.\n";
        list_kernels();
        return 0;
    }
    if (o.mode == "list") {
        list_kernels();
        std::cout << "selected_warptiled_config " << warp_configuration(o.warp_variant->config) << '\n';
        for (const auto& variant : warptiled_variants())
            std::cout << "warptiled_config " << warp_configuration(variant.config) << '\n';
        return 0;
    }
    CUDA_CHECK(cudaSetDevice(0));
    Context ctx(o.warp_variant);
    std::cout << std::setprecision(10);
    if (o.mode == "info") { info(ctx); return 0; }
    std::cout << "source_sha256=" << GEMM_SOURCE_SHA256 << " build_type=" << GEMM_BUILD_CONFIG
              << " seed=" << o.seed << " requested=" << o.kernel << " actual=" << o.kernel << '\n';
    if (o.kernel == "tuned")
        std::cout << "tuned_config BM=" << gemm_tuned_config::BM << " BN=" << gemm_tuned_config::BN
                  << " BK=" << gemm_tuned_config::BK << " TM=" << gemm_tuned_config::TM
                  << " TN=" << gemm_tuned_config::TN << '\n';
    if (o.kernel == "warptiled")
        std::cout << "warptiled_config " << warp_configuration(o.warp_variant->config) << '\n';
    if (o.mode == "bench" && std::string(GEMM_BUILD_CONFIG) != "Release")
        throw std::runtime_error("bench requires a Release build");
    if (o.mode == "check" && !o.has_shape) {
        const std::vector<Shape> shapes = {{1,1,1}, {2,2,3}, {17,19,23}, {65,97,33},
                                         {15,16,17}, {16,17,16}, {33,31,32}, {17,17,17}};
        bool valid = true;
        for (Shape s : shapes) for (const std::string pattern : {"random", "zero", "identity", "integer"})
            valid = run_case(ctx, o, s, pattern) && valid;
        std::cout << "suite_cases=" << shapes.size() * 4 << " status=" << (valid ? "valid" : "invalid") << '\n';
        return valid ? 0 : 2;
    }
    return run_case(ctx, o, o.shape, "random") ? 0 : 2;
}
} // namespace

int main(int argc, char** argv) {
    try {
        const int result = run(argc, argv); // Resources are destroyed before checking cleanup status.
        return cleanup_failed ? 1 : result;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
