/*
    G-GEMV2: single-right-hand-side (nrhs = 1) blocked triangular solve for up to TRSM_NRHS1_BLK_MAX (256) rows,
    ONE CTA per matrix, ONE kernel per recursion leaf.
    Replaces, on the nrhs1 path of trsm_vbatched_core (side=Left, transA=NoTrans, max_n == 1), the recursion levels between
    32 and 256 rows: the old path solved 32-row leaves with trsm_nrhs1_warp_kernel and joined them with one PREP + one tiny
    batched-gemv launch per recursion node (264 nodes per side per solve in the JOREK-proxy STRUMPACK solve, 5.3 ms of mostly
    2-CTA launches, window 26). Here the block is substituted in 32-row steps INSIDE the kernel:
      step I (ascending for lower, descending for upper):
        1. warp 0 solves the 32x32 diagonal block with the G-TRSV2 scheme (lane i owns row i, x(j) broadcast by __shfl_sync);
        2. all threads update the rows still to be solved: y(i) -= A(i, 32I+j) * x(32I+j), j = 0..31 ascending
           (rows are the thread index, so the global reads of a column of the panel are coalesced);
      b / x live in shared memory (<= 256 elements).
    Arithmetic is deterministic: fixed block order, fixed j order per row, no atomics, no split-K. It is NOT bit-identical
    to the recursive path (different summation order of the off-diagonal terms); it is the same blocking algorithm with a
    different split, checked against the old path at tol 1e-12 in the window kit (bench/gemv2).
    Per-matrix m, n, ldda, lddb, Ai/Aj/Bi/Bj are interpreted exactly as trsm_nrhs1_warp_kernel does (same clamps).
    Not codegen'd (generic in T); included by ztrsm_small_vbatched.cu.

    G-GEMV3 (trsm_nrhs1_pipe_kernel below, used when the grid fits one wave: batchCount <= number of SMs):
    ncu on the kernel above (window 28) showed issue-active < 3.4 % and ~65 % barrier stalls: warp 0 alone runs the 32-step diagonal
    chain while 3 warps wait, the panel loads are issued only after that barrier, and every 32-row step costs two CTA barriers.
    The pipelined kernel gives every 32-row block its OWN warp (block b of the solve order -> warp w, up to 8 warps):
      - warp w keeps its 32 rows' right-hand side in registers (lane = row);
      - for each earlier block q of the solve order it issues the 32 panel loads A(rows, block q) BEFORE waiting, waits for
        block q's x (a monotonic per-CTA counter in shared memory, release/acquire, no CTA barrier), then applies the 32 terms;
      - then it solves its own diagonal block with the G-TRSV2 shuffle scheme (fully unrolled 32 steps, compile-time trip
        count, the partial-block guard only predicates the update, so the shuffles are straight-line) and publishes x.
      The diagonal-block loads and the NonUnit reciprocal are issued at kernel start (off the chain) for 4/8-byte T; for
      double-complex they are issued after the panel loop (register budget).
      So the panel memory latency overlaps the previous block's diagonal chain, warps for later blocks absorb earlier panels
      while they wait, and the serial path per block is: counter hand-off + one 32-term DFMA chain + the 32-step diagonal chain.
      Unit diagonal: x = acc (the old code multiplied by dinv = 1, an exact no-op), removing one dependent DMUL per step.
    Arithmetic per row is the SAME sequence as trsm_nrhs1_blk_kernel (alpha*b, then for each block in solve order the 32 panel
    terms j ascending, then the diagonal terms in solve order, then x = acc * dinv), so both kernels give the same values
    (bit-identical intent; the window kit still compares at tol 1e-12). Upper and lower share one code path (block index
    reversed for upper). The old kernel stays for grids larger than one wave (1 CTA of 256 threads per SM after this change)
    and as an A/B fallback: MAGMA_TRSM_NRHS1_PIPE=0 forces it.
*/
#ifndef TRSM_NRHS1_BLK_CUH
#define TRSM_NRHS1_BLK_CUH

#include <stdlib.h>
#include "trsm_nrhs1_warp.cuh"

#define TRSM_NRHS1_BLK_MAX     256   // rows per leaf block (shared-memory vector length)
#define TRSM_NRHS1_BLK_THREADS 128   // threads per CTA (4 warps); the panel update loops over rows with this stride

template<typename T, bool LOWER>
static __global__
__launch_bounds__(TRSM_NRHS1_BLK_THREADS)
void trsm_nrhs1_blk_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n, int batchCount )
{
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int batchid = blockIdx.x;              // one CTA per matrix
    if (batchid >= batchCount) return;           // CTA-uniform
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    // same offset checks / clamps as trsm_nrhs1_warp_kernel
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );
    if(my_m <= 0 || my_n <= 0) return;           // CTA-uniform
    const int s = min( my_m, (int)TRSM_NRHS1_BLK_MAX );

    const int la = (int)ldda[batchid], lb = (int)lddb[batchid];
    const T* A = Aarray[batchid] + Aj * la + Ai;
    T*       B = Barray[batchid] + Bj * lb + Bi;

    const T zero = make_FloatingPoint(0.0, 0.0);
    const T one  = make_FloatingPoint(1.0, 0.0);

    __shared__ T sx[TRSM_NRHS1_BLK_MAX];         // right-hand side, overwritten by the solution
    for (int i = tid; i < s; i += TRSM_NRHS1_BLK_THREADS) sx[i] = alpha * B[i];
    __syncthreads();

    const int nblk = (s + 31) / 32;
    for (int step = 0; step < nblk; step++) {    // CTA-uniform loop
        const int I  = LOWER ? step : (nblk - 1 - step);
        const int r0 = 32 * I;
        const int rb = min(32, s - r0);          // rows in this diagonal block

        // 1. diagonal block, warp 0 (G-TRSV2 scheme: x(j) broadcast, dependents do acc -= x(j) * A(i,j), j in solve order)
        if (tid < 32) {
            T a[32];
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                const bool need = (lane < rb) && (j < rb) && (LOWER ? (j < lane) : (j > lane));
                a[j] = need ? A[(size_t)(r0 + j) * la + (r0 + lane)] : zero;
            }
            T dinv = one, acc = zero;
            if (lane < rb) {
                if (diag == MagmaNonUnit) dinv = div(one, A[(size_t)(r0 + lane) * la + (r0 + lane)]);
                acc = sx[r0 + lane];
            }
            if (LOWER) {
                #pragma unroll
                for (int j = 0; j < 32; j++) {
                    if (j >= rb) break;          // warp-uniform
                    T xj = acc * dinv;
                    xj = trsm_nrhs1_shfl(xj, j);
                    if (lane == j)       acc = xj;
                    else if (lane > j)   acc -= xj * a[j];
                }
            } else {
                #pragma unroll
                for (int j = 31; j >= 0; j--) {
                    if (j >= rb) continue;       // warp-uniform
                    T xj = acc * dinv;
                    xj = trsm_nrhs1_shfl(xj, j);
                    if (lane == j)       acc = xj;
                    else if (lane < j)   acc -= xj * a[j];
                }
            }
            if (lane < rb) sx[r0 + lane] = acc;
        }
        __syncthreads();

        // 2. off-diagonal panel update of the rows still to be solved with the 32 new unknowns x(r0..r0+rb-1)
        const int u0 = LOWER ? (r0 + 32) : 0;    // rows below (lower) / above (upper) the diagonal block
        const int u1 = LOWER ? s         : r0;
        for (int i = u0 + tid; i < u1; i += TRSM_NRHS1_BLK_THREADS) {
            T av[32];
            #pragma unroll
            for (int j = 0; j < 32; j++)         // independent loads first (coalesced over i), then the fixed-order FMA chain
                av[j] = (j < rb) ? A[(size_t)(r0 + j) * la + i] : zero;
            T t = sx[i];
            #pragma unroll
            for (int j = 0; j < 32; j++)
                if (j < rb) t -= av[j] * sx[r0 + j];
            sx[i] = t;
        }
        __syncthreads();
    }
    for (int i = tid; i < s; i += TRSM_NRHS1_BLK_THREADS) B[i] = sx[i];
}

// host wrapper: Left / NoTrans / max_n == 1 / 32 < max_m <= TRSM_NRHS1_BLK_MAX (the caller checks this)
static inline bool trsm_nrhs1_blk_applies(magma_side_t side, magma_trans_t transA, magma_int_t max_m, magma_int_t max_n)
{
    return side == MagmaLeft && transA == MagmaNoTrans && max_n == 1 && max_m >= 1 && max_m <= TRSM_NRHS1_BLK_MAX;
}

// ---------------------------------------------------------------------------------------------------------------------------
// G-GEMV3 pipelined leaf
#define TRSM_NRHS1_PIPE_THREADS TRSM_NRHS1_BLK_MAX   // one warp per 32-row block, 8 warps for 256 rows

static __device__ __forceinline__ int trsm_nrhs1_ld_acquire(const int* p)
{
    int v;
    asm volatile("ld.acquire.cta.shared.b32 %0, [%1];" : "=r"(v) : "r"((unsigned int)__cvta_generic_to_shared(p)) : "memory");
    return v;
}
static __device__ __forceinline__ void trsm_nrhs1_st_release(int* p, int v)
{
    asm volatile("st.release.cta.shared.b32 [%0], %1;" :: "r"((unsigned int)__cvta_generic_to_shared(p)), "r"(v) : "memory");
}

template<typename T, bool LOWER>
static __global__
__launch_bounds__(TRSM_NRHS1_PIPE_THREADS)
void trsm_nrhs1_pipe_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n, int batchCount )
{
    // diagonal-block row and reciprocal loaded at kernel start (off the chain) when the register budget allows
    constexpr bool EARLY = (sizeof(T) <= 8);
    const int tid     = threadIdx.x;
    const int lane    = tid & 31;
    const int w       = tid >> 5;                // position of this warp's block in the solve order
    const int batchid = blockIdx.x;              // one CTA per matrix
    if (batchid >= batchCount) return;           // CTA-uniform
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    // same offset checks / clamps as trsm_nrhs1_warp_kernel / trsm_nrhs1_blk_kernel
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );
    if(my_m <= 0 || my_n <= 0) return;           // CTA-uniform
    const int s    = min( my_m, (int)TRSM_NRHS1_BLK_MAX );
    const int nblk = (s + 31) / 32;              // <= blockDim.x / 32 (host launches 32 * ceil(max_m / 32) threads)

    __shared__ T   sx[TRSM_NRHS1_BLK_MAX];       // solved x, published block by block
    __shared__ int s_done;                       // number of blocks (in solve order) whose x is published
    if (tid == 0) s_done = 0;
    __syncthreads();                             // the only CTA barrier
    if (w >= nblk) return;                       // warp-uniform; no later CTA barrier

    const int la = (int)ldda[batchid], lb = (int)lddb[batchid];
    const T* A = Aarray[batchid] + Aj * la + Ai;
    T*       B = Barray[batchid] + Bj * lb + Bi;
    __builtin_assume(__isGlobal(A));             // global loads (LDG), not generic
    __builtin_assume(__isGlobal(B));

    const T zero = make_FloatingPoint(0.0, 0.0);
    const T one  = make_FloatingPoint(1.0, 0.0);
    const bool unit = (diag != MagmaNonUnit);

    const int b     = LOWER ? w : (nblk - 1 - w);   // block owned by this warp
    const int r0    = 32 * b;
    const int rb    = min(32, s - r0);              // rows in this block
    const bool valid = (lane < rb);
    const int i     = r0 + lane;                    // this lane's row

    T acc = zero;
    if (valid) acc = alpha * B[i];

    T a[32];
    T dinv = one;
    #define TRSM_NRHS1_PIPE_LOAD_DIAG()                                                             \
        {                                                                                           \
            const T* pc = A + (size_t)r0 * la + i;      /* column r0 + j, row i */                  \
            _Pragma("unroll")                                                                       \
            for (int j = 0; j < 32; j++) {                                                          \
                const bool need = valid && (j < rb) && (LOWER ? (j < lane) : (j > lane));           \
                a[j] = need ? *pc : zero;                                                           \
                pc += la;                                                                           \
            }                                                                                       \
            if (valid && !unit) dinv = div(one, A[(size_t)i * la + i]);                             \
        }
    if (EARLY) TRSM_NRHS1_PIPE_LOAD_DIAG();

    // panel updates from the blocks earlier in the solve order (same per-row term order as trsm_nrhs1_blk_kernel)
    for (int q = 0; q < w; q++) {                  // warp-uniform trip count
        const int c0 = 32 * (LOWER ? q : (nblk - 1 - q));
        const int cb = min(32, s - c0);
        T p[32];
        const T* pc = A + (size_t)c0 * la + i;     // column c0 + j, row i
        #pragma unroll
        for (int j = 0; j < 32; j++) {             // independent loads, issued before the wait
            p[j] = (valid && j < cb) ? *pc : zero;
            pc += la;
        }
        while (trsm_nrhs1_ld_acquire(&s_done) <= q) { }
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            const T xv = sx[c0 + j];               // in bounds (c0 + 31 < TRSM_NRHS1_BLK_MAX); used only for j < cb
            if (j < cb) acc -= p[j] * xv;
        }
    }
    if (!EARLY) TRSM_NRHS1_PIPE_LOAD_DIAG();
    #undef TRSM_NRHS1_PIPE_LOAD_DIAG

    // diagonal block: G-TRSV2 scheme, 32 unrolled steps (compile-time trip count), j in solve order
    if (!valid) acc = zero;
    // branch-free step: the update is computed on every lane and selected (same value as the old guarded update)
    #define TRSM_NRHS1_PIPE_STEP(XJ)                                                                \
        _Pragma("unroll")                                                                           \
        for (int jj = 0; jj < 32; jj++) {                                                           \
            const int j  = LOWER ? jj : (31 - jj);                                                  \
            const T   xj = trsm_nrhs1_shfl(XJ, j);                                                  \
            const T   up = acc - xj * a[j];                                                         \
            const bool dep = (j < rb) && (LOWER ? (lane > j) : (lane < j));                         \
            acc = (j < rb && lane == j) ? xj : (dep ? up : acc);                                    \
        }
    if (unit) { TRSM_NRHS1_PIPE_STEP(acc) }
    else      { TRSM_NRHS1_PIPE_STEP(acc * dinv) }
    #undef TRSM_NRHS1_PIPE_STEP

    // publish x of this block, then write it back
    if (valid) sx[i] = acc;
    __threadfence_block();
    __syncwarp();
    if (lane == 0) trsm_nrhs1_st_release(&s_done, w + 1);
    if (valid) B[i] = acc;
}

// MAGMA_TRSM_NRHS1_PIPE=0 forces the G-GEMV2 kernel for every grid (A/B fallback inside one library)
static inline bool trsm_nrhs1_pipe_enabled()
{
    static const bool on = []() {
        const char *e = getenv("MAGMA_TRSM_NRHS1_PIPE");
        return !(e && atoi(e) == 0);
    }();
    return on;
}

template<typename T>
static void trsm_nrhs1_blk_vbatched(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    int nsm = 0;
    cudaDeviceGetAttribute(&nsm, cudaDevAttrMultiProcessorCount, queue->device());
    const int pipe_threads = 32 * (int)magma_ceildiv(min(max_m, (magma_int_t)TRSM_NRHS1_BLK_MAX), (magma_int_t)32);
    dim3 threads(TRSM_NRHS1_BLK_THREADS, 1, 1);
    for (magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount - i);
        dim3 grid( (unsigned int)ibatch, 1, 1 );
        // G-GEMV3: pipelined kernel while the grid fits one wave (its 256-thread CTA allows ~1 CTA per SM)
        if (trsm_nrhs1_pipe_enabled() && ibatch <= (magma_int_t)nsm) {
            dim3 pthreads(pipe_threads, 1, 1);
            if (uplo == MagmaLower)
                trsm_nrhs1_pipe_kernel<T, true><<< grid, pthreads, 0, queue->cuda_stream() >>>
                (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
            else
                trsm_nrhs1_pipe_kernel<T, false><<< grid, pthreads, 0, queue->cuda_stream() >>>
                (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
            continue;
        }
        if (uplo == MagmaLower)
            trsm_nrhs1_blk_kernel<T, true><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
        else
            trsm_nrhs1_blk_kernel<T, false><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
    }
}
#endif
