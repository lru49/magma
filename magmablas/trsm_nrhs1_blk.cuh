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
*/
#ifndef TRSM_NRHS1_BLK_CUH
#define TRSM_NRHS1_BLK_CUH

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
    dim3 threads(TRSM_NRHS1_BLK_THREADS, 1, 1);
    for (magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount - i);
        dim3 grid( (unsigned int)ibatch, 1, 1 );
        if (uplo == MagmaLower)
            trsm_nrhs1_blk_kernel<T, true><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
        else
            trsm_nrhs1_blk_kernel<T, false><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
    }
}
#endif
