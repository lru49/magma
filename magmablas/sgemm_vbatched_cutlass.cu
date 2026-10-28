/*
    -- MAGMA (version 1.1) -- F-SGEMM-CUTLASS (magma-hpc F-track)

    Float vbatched NN GEMM through CUTLASS's grouped SIMT FP32 GEMM (cutlass::gemm::device::GemmGrouped,
    OpClassSimt, Sm80 arch tag, column-major A, B, C; BSD-3 headers from ext/cutlass, included at build time only).
    Selected by MAGMA_SGEMM_VBATCHED_NN_VERSION=CUTLASS | CUTLASS128 | CUTLASS64 in sgemm_vbatched_core.cu.

    Semantics are those of gemm_template_vbatched_nn_kernel (gemm_template_kernel_vbatched.cuh): per problem i
      M = m[i] - max(Ai,Ci), N = n[i] - max(Bj,Cj), K = k[i] - max(Aj,Bi), each clipped to max_m/max_n/max_k,
      A_i = dA[i] + Aj*lda[i] + Ai (etc.), problems with out-of-range offsets, M<=0, N<=0, K<0, (K==0 && beta==1)
      or a NULL pointer are skipped (problem size 0 -> no tiles).
    A tiny device kernel builds CUTLASS's per-problem GemmCoord / int64 leading-dimension / pointer arrays from
    MAGMA's device int arrays in a per-stream scratch buffer (no host sync per call; the buffer is reused call after
    call because kernels on one stream are ordered). C = alpha*A*B + beta*C with the same per-element FMA order
    as MAGMA's kernel, so results are bit-identical to the default dispatch (verified in bench/sgemm-cutlass).

    MAGMA_SGEMM_CUTLASS_PLANT (compile-time, verification only): the LAST problem of every call gets no update
    (its tiles are skipped) -> the solve compare must FAIL.
*/
// std and CUTLASS headers first: magma_internal.h defines min/max function-like macros that break <chrono>/<limits>
#include <cstdint>
#include <mutex>
#include <vector>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/gemm/kernel/gemm_grouped.h"
#include "cutlass/gemm/kernel/default_gemm_grouped.h"
#include "cutlass/gemm/device/gemm_grouped.h"

#include "magma_internal.h"

namespace {

template <int TBM, int TBN, int TBK, int WM, int WN, int STAGES>
struct SgemmGroupedCfg {
    using GemmKernel = typename cutlass::gemm::kernel::DefaultGemmGrouped<
        float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
        float, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 1,
        float, cutlass::layout::ColumnMajor, float,
        cutlass::arch::OpClassSimt, cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<TBM, TBN, TBK>,
        cutlass::gemm::GemmShape<WM, WN, TBK>,
        cutlass::gemm::GemmShape<1, 1, 1>,
        cutlass::epilogue::thread::LinearCombination<float, 1, float, float>,
        cutlass::gemm::threadblock::GemmBatchedIdentityThreadblockSwizzle,
        STAGES,
        cutlass::gemm::kernel::GroupScheduleMode::kDeviceOnly>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmGrouped<GemmKernel>;
};

using Cfg128x64  = SgemmGroupedCfg<128,  64, 8, 32, 64, 2>;   // tag CUTLASS    (default: best in the microbenchmark)
using Cfg128x128 = SgemmGroupedCfg<128, 128, 8, 32, 64, 2>;   // tag CUTLASS128
using Cfg64x64   = SgemmGroupedCfg< 64,  64, 8, 32, 32, 2>;   // tag CUTLASS64

// per-stream scratch: GemmCoord[cap] | int64 lda,ldb,ldc [3*cap] | float* A,B,C [3*cap]
struct Scratch {
    cudaStream_t stream = nullptr;
    int cap = 0;
    cutlass::gemm::GemmCoord* sizes = nullptr;
    int64_t* ld = nullptr;
    float** ptr = nullptr;
};

Scratch* get_scratch(cudaStream_t stream, int batchCount)
{
    static std::mutex mtx;
    static std::vector<Scratch*> table;
    std::lock_guard<std::mutex> lock(mtx);
    Scratch* s = nullptr;
    for (Scratch* t : table) if (t->stream == stream) { s = t; break; }
    if (s == nullptr) { s = new Scratch(); s->stream = stream; table.push_back(s); }
    if (s->cap < batchCount) {
        int cap = 1024; while (cap < batchCount) cap *= 2;
        size_t bytes = (size_t)cap * (sizeof(cutlass::gemm::GemmCoord) + 3 * sizeof(int64_t) + 3 * sizeof(float*));
        bytes = (bytes + 255) & ~(size_t)255;
        void* buf = nullptr;
        if (cudaMalloc(&buf, bytes) != cudaSuccess) { (void)cudaGetLastError(); return nullptr; }
        if (s->sizes != nullptr) { cudaStreamSynchronize(stream); cudaFree(s->sizes); }   // growth only: rare
        s->sizes = (cutlass::gemm::GemmCoord*)buf;
        s->ld    = (int64_t*)(s->sizes + cap);
        s->ptr   = (float**)(s->ld + 3 * cap);
        s->cap   = cap;
    }
    return s;
}

__global__ void sgemm_vbatched_cutlass_setup_kernel(
    int batchCount, int max_m, int max_n, int max_k, float beta,
    const magma_int_t* __restrict__ M, const magma_int_t* __restrict__ N, const magma_int_t* __restrict__ K,
    float const * const * __restrict__ Aarray, int Ai, int Aj, const magma_int_t* __restrict__ LDA,
    float const * const * __restrict__ Barray, int Bi, int Bj, const magma_int_t* __restrict__ LDB,
    float              ** __restrict__ Carray, int Ci, int Cj, const magma_int_t* __restrict__ LDC,
    cutlass::gemm::GemmCoord* __restrict__ sizes, int64_t* __restrict__ ld, float** __restrict__ ptr)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= batchCount) return;
    int my_M = (int)M[b], my_N = (int)N[b], my_K = (int)K[b];
    bool skip = (my_M < Ai || my_K < Aj || my_K < Bi || my_N < Bj || my_M < Ci || my_N < Cj);
    my_M -= max(Ai, Ci); my_N -= max(Bj, Cj); my_K -= max(Aj, Bi);
    my_M = min(my_M, max_m); my_N = min(my_N, max_n); my_K = min(my_K, max_k);
    if (my_M <= 0 || my_N <= 0 || my_K < 0) skip = true;
    if (my_K == 0 && beta == 1.0f) skip = true;
    const float* A = Aarray[b]; const float* B = Barray[b]; float* C = Carray[b];
    if (A == NULL || B == NULL || C == NULL) skip = true;
#ifdef MAGMA_SGEMM_CUTLASS_PLANT
    if (b == batchCount - 1) skip = true;   // PLANT: the last problem of every call gets no update at all (alpha = 0)
#endif
    const int64_t lda = (int64_t)LDA[b], ldb = (int64_t)LDB[b], ldc = (int64_t)LDC[b];
    if (skip) { my_M = 0; my_N = 0; my_K = 0; }
    sizes[b] = cutlass::gemm::GemmCoord(my_M, my_N, my_K);
    ld[b] = lda; ld[batchCount + b] = ldb; ld[2 * batchCount + b] = ldc;
    ptr[b]                  = skip ? NULL : (float*)(A + lda * Aj + Ai);
    ptr[batchCount + b]     = skip ? NULL : (float*)(B + ldb * Bj + Bi);
    ptr[2 * batchCount + b] = skip ? NULL : (C + ldc * Cj + Ci);
}

template <typename Cfg>
int run_grouped(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float               **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    using Gemm = typename Cfg::Gemm;
    static const int threadblock_count = Gemm::sufficient();   // SM count x occupancy, once per config
    if (threadblock_count <= 0) return -1;
    cudaStream_t stream = queue->cuda_stream();
    Scratch* s = get_scratch(stream, (int)batchCount);
    if (s == nullptr) return -2;

    const int threads = 128;
    sgemm_vbatched_cutlass_setup_kernel<<<((int)batchCount + threads - 1) / threads, threads, 0, stream>>>(
        (int)batchCount, (int)max_m, (int)max_n, (int)max_k, beta,
        m, n, k, dA_array, (int)Ai, (int)Aj, ldda, dB_array, (int)Bi, (int)Bj, lddb, dC_array, (int)Ci, (int)Cj, lddc,
        s->sizes, s->ld, s->ptr);

    typename Gemm::EpilogueOutputOp::Params epilogue(alpha, beta);
    typename Gemm::Arguments args(
        s->sizes, (int)batchCount, threadblock_count, epilogue,
        s->ptr, s->ptr + batchCount, s->ptr + 2 * batchCount, s->ptr + 2 * batchCount,
        s->ld, s->ld + batchCount, s->ld + 2 * batchCount, s->ld + 2 * batchCount,
        nullptr);
    Gemm gemm;
    if (gemm.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) return -3;
    if (gemm.run(stream) != cutlass::Status::kSuccess) return -4;
    return 0;
}

} // namespace

// returns 0 on success (GEMM launched on the queue's stream), nonzero if the caller must fall back to the MAGMA kernel
__attribute__((visibility("hidden")))
int magmablas_sgemm_vbatched_nn_cutlass(
    int config,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float               **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    if (batchCount <= 0) return 0;
    switch (config) {
        case 0:  return run_grouped<Cfg128x64 >(max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
        case 1:  return run_grouped<Cfg128x128>(max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
        case 2:  return run_grouped<Cfg64x64  >(max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
        default: return -5;
    }
}
