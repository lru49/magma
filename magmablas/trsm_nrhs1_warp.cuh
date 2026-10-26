/*
    G-TRSV2: single-right-hand-side (nrhs = 1) small triangular solve, one warp per matrix.
    Replaces trsm_template_vbatched_lNL/lNU_kernel<T,NB,NRHS=32> for side=Left, transA=NoTrans, max_n == 1, max_m <= 32.
    Lane i owns row i of A and x(i). Row i of A (strictly lower / upper triangle) is read once into registers, coalesced
    per column; then x is produced by forward (lower) / back (upper) substitution: the owner lane of x(j) broadcasts it with
    __shfl_sync and every dependent lane does acc -= x(j) * A(i,j) in the SAME order as the old kernel (j ascending for
    lower, descending for upper), so results are bit-identical to the template kernel. A non-unit diagonal is applied as
    multiplication by the reciprocal (as the old kernel does). Per-matrix m, n, ldda, lddb, Ai/Aj/Bi/Bj are interpreted
    exactly as trsm_template_vbatched_lNL_kernel does.
    Not codegen'd (generic in T); included by ztrsm_small_vbatched.cu.
*/
#ifndef TRSM_NRHS1_WARP_CUH
#define TRSM_NRHS1_WARP_CUH

#define TRSM_NRHS1_WARPS 4   // matrices (warps) per block

template<typename T>
static __device__ __inline__ T trsm_nrhs1_shfl(T v, int src)
{
    T r;
    const unsigned int* pv = (const unsigned int*)&v;
    unsigned int* pr = (unsigned int*)&r;
    #pragma unroll
    for (int w = 0; w < (int)(sizeof(T) / 4); w++)
        pr[w] = __shfl_sync(0xffffffffu, pv[w], src);
    return r;
}

template<typename T, bool LOWER>
static __global__
__launch_bounds__(32 * TRSM_NRHS1_WARPS)
void trsm_nrhs1_warp_kernel(
        magma_diag_t diag, magma_int_t* m, magma_int_t* n,
        T alpha, T** Aarray, int Ai, int Aj, magma_int_t* ldda,
                 T** Barray, int Bi, int Bj, magma_int_t* lddb,
        int max_m, int max_n, int batchCount )
{
    const int lane    = threadIdx.x & 31;
    const int batchid = blockIdx.x * TRSM_NRHS1_WARPS + (threadIdx.x >> 5);
    if (batchid >= batchCount) return;           // warp-uniform
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    // same offset checks / clamps as trsm_template_vbatched_lNL_kernel
    if( my_m < Ai || my_m < Aj ) return;
    if( my_m < Bi || my_n < Bj ) return;
    my_m -= max( Ai, max( Aj, Bi ) );
    my_n -= Bj;
    my_m = min( my_m, max_m );
    my_n = min( my_n, max_n );
    if(my_m <= 0 || my_n <= 0) return;           // warp-uniform
    my_m = min( my_m, 32 );

    const int la = (int)ldda[batchid], lb = (int)lddb[batchid];
    const T* A = Aarray[batchid] + Aj * la + Ai;
    T*       B = Barray[batchid] + Bj * lb + Bi;

    const T zero = make_FloatingPoint(0.0, 0.0);
    const T one  = make_FloatingPoint(1.0, 0.0);

    T a[32];
    #pragma unroll
    for (int j = 0; j < 32; j++) {
        const bool need = (lane < my_m) && (j < my_m) && (LOWER ? (j < lane) : (j > lane));
        a[j] = need ? A[j * la + lane] : zero;
    }
    T dinv = one, acc = zero;
    if (lane < my_m) {
        if (diag == MagmaNonUnit) dinv = div(one, A[lane * la + lane]);
        acc = alpha * B[lane];
    }

    if (LOWER) {
        #pragma unroll
        for (int j = 0; j < 32; j++) {
            if (j >= my_m) break;                // warp-uniform
            T xj = acc * dinv;
            xj = trsm_nrhs1_shfl(xj, j);
            if (lane == j)       acc = xj;
            else if (lane > j)   acc -= xj * a[j];
        }
    } else {
        #pragma unroll
        for (int j = 31; j >= 0; j--) {
            if (j >= my_m) continue;             // warp-uniform
            T xj = acc * dinv;
            xj = trsm_nrhs1_shfl(xj, j);
            if (lane == j)       acc = xj;
            else if (lane < j)   acc -= xj * a[j];
        }
    }
    if (lane < my_m) B[lane] = acc;
}

// host wrapper: Left / NoTrans / max_n == 1 / 1 <= max_m <= 32 (the caller checks this, see trsm_nrhs1_warp_applies)
static inline bool trsm_nrhs1_warp_applies(magma_side_t side, magma_trans_t transA, magma_int_t max_m, magma_int_t max_n)
{
    return side == MagmaLeft && transA == MagmaNoTrans && max_n == 1 && max_m >= 1 && max_m <= 32;
}

template<typename T>
static void trsm_nrhs1_warp_vbatched(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t* m, magma_int_t* n,
    T alpha, T** dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
             T** dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(32 * TRSM_NRHS1_WARPS, 1, 1);
    for (magma_int_t i = 0; i < batchCount; i += max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount - i);
        dim3 grid( magma_ceildiv(ibatch, (magma_int_t)TRSM_NRHS1_WARPS), 1, 1 );
        if (uplo == MagmaLower)
            trsm_nrhs1_warp_kernel<T, true><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
        else
            trsm_nrhs1_warp_kernel<T, false><<< grid, threads, 0, queue->cuda_stream() >>>
            (diag, m+i, n+i, alpha, dA_array+i, Ai, Aj, ldda+i, dB_array+i, Bi, Bj, lddb+i, max_m, max_n, (int)ibatch);
    }
}
#endif
