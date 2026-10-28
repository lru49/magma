/*
    -- MAGMA (version 2.10.0) -- magma-hpc S track addition (BSD-3 like the rest of MAGMA)
       S-TOPCHAIN: single-right-hand-side triangular solve of ONE tall matrix (256 < n <= 4096) per batch entry with
       ONE COOPERATIVE kernel instead of the recursion (256-row pipelined leaves joined by batched-gemv launches:
       27..87 launches per level for the 800..3200-row top-of-tree fronts of the JOREK-proxy solve, 40 % of the solve).
       Grid = (ceil(n/256), batchCount), all CTAs co-resident (cooperative launch). CTA c (solve order) owns rows
       [256 bc, 256 bc + 256) with bc = c for lower, nbig-1-c for upper; it applies the updates from the earlier
       CTAs' x (read from B after an acquire on that CTA's flag), then runs the G-GEMV3 pipelined 32-row scheme on
       its own block (one warp per 32-row sub-block, shared-memory counter), writes x to B, and publishes its flag
       (release, gpu scope). Flags live in a small device buffer allocated once; an epoch per launch avoids resets.
       Arithmetic per row: alpha*b, then the off-diagonal terms in solve order (earlier CTAs' 256-column blocks
       chunked by 32, then the own CTA's earlier 32-row blocks), then the diagonal chain: the same blocking order as
       trsm_nrhs1_pipe_kernel; compared at 1e-12. Deadlock-free: a CTA only waits on lower solve-order CTAs, which
       never wait on it, and the cooperative launch guarantees co-residency (else the caller falls back).
       MAGMA_TRSM_NRHS1_COOP=0 disables (A/B fallback inside one library).

       @generated from magmablas/ztrsm_nrhs1_coop.cu, normal z -> c, Tue Oct 27 20:46:18 2026
*/
#include "magma_internal.h"
#define PRECISION_c
#include "magma_templates.h"
#include "trsm_template_kernel_vbatched.cuh"   // make_FloatingPoint, div, shfl helpers
#include "trsm_nrhs1_warp.cuh"
#include "trsm_nrhs1_blk.cuh"
#include <cstdlib>

#define TRSM_NRHS1_COOP_ROWS   256
#define TRSM_NRHS1_COOP_MAX_N  4096
#define TRSM_NRHS1_COOP_MAXB   16
#define TRSM_NRHS1_COOP_FSTRIDE 32

static __device__ __forceinline__ int trsm_nrhs1_ld_acquire_gpu(const int* p)
{
    int v;
    asm volatile("ld.acquire.gpu.global.b32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
    return v;
}
static __device__ __forceinline__ void trsm_nrhs1_st_release_gpu(int* p, int v)
{
    asm volatile("st.release.gpu.global.b32 [%0], %1;" :: "l"(p), "r"(v) : "memory");
}

template<typename T, bool LOWER>
static __global__
__launch_bounds__(TRSM_NRHS1_COOP_ROWS)
void trsm_nrhs1_coop_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n, int batchCount, int* flags, int epoch )
{
    constexpr bool EARLY = (sizeof(T) <= 8);
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int w       = tid >> 5;                // sub-block (solve order) inside this CTA
    const int batchid = blockIdx.y;
    const int c       = blockIdx.x;              // CTA position in the solve order
    if (batchid >= batchCount) return;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );
    if(my_m <= 0 || my_n <= 0) return;
    const int s    = my_m;
    const int nbig = (s + TRSM_NRHS1_COOP_ROWS - 1) / TRSM_NRHS1_COOP_ROWS;
    if (c >= nbig) return;                       // CTA-uniform; never a producer anyone waits for
    const int bc     = LOWER ? c : (nbig - 1 - c);
    const int r0c    = TRSM_NRHS1_COOP_ROWS * bc;
    const int rows_c = min(TRSM_NRHS1_COOP_ROWS, s - r0c);
    const int nblk   = (rows_c + 31) / 32;

    __shared__ T   sx[TRSM_NRHS1_COOP_ROWS];
    __shared__ int s_done;
    if (tid == 0) s_done = 0;
    __syncthreads();

    const int la = (int)ldda[batchid], lb = (int)lddb[batchid];
    const T* A = Aarray[batchid] + Aj * la + Ai;
    T*       B = Barray[batchid] + Bj * lb + Bi;
    __builtin_assume(__isGlobal(A));
    __builtin_assume(__isGlobal(B));
    int* myflags = flags + batchid * TRSM_NRHS1_COOP_FSTRIDE;

    const T zero = make_FloatingPoint(0.0, 0.0);
    const T one  = make_FloatingPoint(1.0, 0.0);
    const bool unit = (diag != MagmaNonUnit);
    const bool warp_on = (w < nblk);             // warp-uniform
    const int b     = LOWER ? w : (nblk - 1 - w);
    const int r0l   = 32 * b;                    // local row offset of this warp's block
    const int r0    = r0c + r0l;                 // absolute
    const int rb    = warp_on ? min(32, rows_c - r0l) : 0;
    const bool valid = warp_on && (lane < rb);
    const int i     = r0 + lane;

    T acc = zero;
    if (valid) acc = alpha * B[i];

    T a[32];
    T dinv = one;
    #define TRSM_NRHS1_COOP_LOAD_DIAG()                                                             \
        {                                                                                           \
            const T* pc = A + (size_t)r0 * la + i;                                                   \
            _Pragma("unroll")                                                                       \
            for (int j = 0; j < 32; j++) {                                                          \
                const bool need = valid && (j < rb) && (LOWER ? (j < lane) : (j > lane));           \
                a[j] = need ? *pc : zero;                                                           \
                pc += la;                                                                           \
            }                                                                                       \
            if (valid && !unit) dinv = div(one, A[(size_t)i * la + i]);                             \
        }
    if (EARLY) TRSM_NRHS1_COOP_LOAD_DIAG();

    // 1. updates from the earlier CTAs (solve order), 256 columns each in 32-column chunks; x read from B after the acquire
    for (int q = 0; q < c; q++) {
        const int bq   = LOWER ? q : (nbig - 1 - q);
        const int c0q  = TRSM_NRHS1_COOP_ROWS * bq;
        const int cols = min(TRSM_NRHS1_COOP_ROWS, s - c0q);
        if (lane == 0) { while (trsm_nrhs1_ld_acquire_gpu(myflags + q) < epoch) { __nanosleep(100); } }
        __syncwarp();
        for (int k = 0; k < TRSM_NRHS1_COOP_ROWS / 32; k++) {
            const int c0 = c0q + 32 * k;
            const int cb = min(32, cols - 32 * k);
            if (cb <= 0) break;                    // warp-uniform
            T p[32];
            const T* pc = A + (size_t)c0 * la + i;
            #pragma unroll
            for (int j = 0; j < 32; j++) { p[j] = (valid && j < cb) ? *pc : zero; pc += la; }
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                if (j < cb) { const T xv = B[c0 + j]; acc -= p[j] * xv; }
            }
        }
    }
    // 2. updates from this CTA's earlier 32-row blocks (pipelined through shared memory)
    if (warp_on) {
        for (int q = 0; q < w; q++) {
            const int c0l = 32 * (LOWER ? q : (nblk - 1 - q));
            const int cb  = min(32, rows_c - c0l);
            T p[32];
            const T* pc = A + (size_t)(r0c + c0l) * la + i;
            #pragma unroll
            for (int j = 0; j < 32; j++) { p[j] = (valid && j < cb) ? *pc : zero; pc += la; }
            while (trsm_nrhs1_ld_acquire(&s_done) <= q) { }
            #pragma unroll
            for (int j = 0; j < 32; j++) { const T xv = sx[c0l + j]; if (j < cb) acc -= p[j] * xv; }
        }
    }
    if (!EARLY) TRSM_NRHS1_COOP_LOAD_DIAG();
    #undef TRSM_NRHS1_COOP_LOAD_DIAG

    // 3. diagonal block (G-TRSV2 scheme, 32 unrolled steps)
    if (!valid) acc = zero;
    #define TRSM_NRHS1_COOP_STEP(XJ)                                                                \
        _Pragma("unroll")                                                                           \
        for (int jj = 0; jj < 32; jj++) {                                                           \
            const int j  = LOWER ? jj : (31 - jj);                                                  \
            const T   xj = trsm_nrhs1_shfl(XJ, j);                                                  \
            const T   up = acc - xj * a[j];                                                         \
            const bool dep = (j < rb) && (LOWER ? (lane > j) : (lane < j));                         \
            acc = (j < rb && lane == j) ? xj : (dep ? up : acc);                                    \
        }
    if (warp_on) {
        if (unit) { TRSM_NRHS1_COOP_STEP(acc) }
        else      { TRSM_NRHS1_COOP_STEP(acc * dinv) }
    }
    #undef TRSM_NRHS1_COOP_STEP

    // 4. publish inside the CTA, write back, then publish the CTA's flag for the later CTAs
    if (valid) sx[r0l + lane] = acc;
    __threadfence_block();
    __syncwarp();
    if (warp_on && lane == 0) trsm_nrhs1_st_release(&s_done, w + 1);
    if (valid) B[i] = acc;
    __threadfence();
    __syncthreads();
    if (tid == 0) trsm_nrhs1_st_release_gpu(myflags + c, epoch);
}

static inline bool trsm_nrhs1_coop_enabled()
{
    static const bool on = []() { const char *e = getenv("MAGMA_TRSM_NRHS1_COOP"); return (e && atoi(e) == 1); }();   // default OFF (window 109 NO-GO)
    return on;
}

// flags: one small device buffer per process (zeroed once); epoch increases per launch so no reset is needed
static int* trsm_nrhs1_coop_flags(int& epoch)
{
    static int* d_flags = nullptr;
    static int  ep = 0;
    if (!d_flags) {
        if (cudaMalloc((void**)&d_flags, sizeof(int) * TRSM_NRHS1_COOP_MAXB * TRSM_NRHS1_COOP_FSTRIDE) != cudaSuccess) return nullptr;
        cudaMemset(d_flags, 0, sizeof(int) * TRSM_NRHS1_COOP_MAXB * TRSM_NRHS1_COOP_FSTRIDE);
    }
    epoch = ++ep;
    return d_flags;
}

/// returns 0 when the solve was issued, -1 when the caller must use the recursion
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_ctrsm_nrhs1_coop(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaFloatComplex alpha,
        magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaFloatComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    if (!trsm_nrhs1_coop_enabled()) return -1;
    if (max_m <= TRSM_NRHS1_COOP_ROWS || max_m > TRSM_NRHS1_COOP_MAX_N || batchCount < 1 || batchCount > TRSM_NRHS1_COOP_MAXB) return -1;
    const int nbig = (int)magma_ceildiv(max_m, (magma_int_t)TRSM_NRHS1_COOP_ROWS);
    if (nbig > TRSM_NRHS1_COOP_FSTRIDE) return -1;
    int epoch = 0;
    int* flags = trsm_nrhs1_coop_flags(epoch);
    if (!flags) return -1;
    // co-residency check (cooperative launch would fail otherwise)
    int nsm = 0, per_sm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, queue->device());
    void* kern = (uplo == MagmaLower) ? (void*)trsm_nrhs1_coop_kernel<magmaFloatComplex, true>
                                      : (void*)trsm_nrhs1_coop_kernel<magmaFloatComplex, false>;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kern, TRSM_NRHS1_COOP_ROWS, 0) != cudaSuccess) return -1;
    if ((long)nbig * batchCount > (long)per_sm * nsm) return -1;
    dim3 grid(nbig, (unsigned int)batchCount, 1), threads(TRSM_NRHS1_COOP_ROWS, 1, 1);
    int bc = (int)batchCount, mm = (int)max_m, nn = (int)max_n, ai = (int)Ai, aj = (int)Aj, bi = (int)Bi, bj = (int)Bj;
    void* args[] = { &diag, &m, &n, &alpha, &dA_array, &ai, &aj, &ldda, &dB_array, &bi, &bj, &lddb, &mm, &nn, &bc, &flags, &epoch };
    cudaError_t err = cudaLaunchCooperativeKernel(kern, grid, threads, args, 0, queue->cuda_stream());
    if (err != cudaSuccess) { cudaGetLastError(); return -1; }
    return 0;
}
