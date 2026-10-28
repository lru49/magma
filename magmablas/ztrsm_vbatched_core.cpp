/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Ahmad Abdelfattah

*/
#include <mutex>
#include "magma_internal.h"
#include "trsm_nrhs1_stop.h"

#define PRECISION_z
//-------------------------

magma_int_t magma_get_ztrsm_vbatched_nb(magma_int_t max_n)
{
    if      ( max_n > 2048 ) return 2048;
    else if ( max_n > 1024 ) return 1024;
    else if ( max_n >  512 ) return 512;
    else if ( max_n >  256 ) return 256;
    else if ( max_n >  128 ) return 128;
    else if ( max_n >   64 ) return  64;
    else if ( max_n >   32 ) return  32;
    else if ( max_n >   16 ) return  16;
    else if ( max_n >    8 ) return   8;
    else if ( max_n >    4 ) return   4;
    else if ( max_n >    2 ) return   2;
    else return 1;
}
///////////////////////////////////////////////////////////////////////////////////////////////////
// G-TRSV1 helpers (ztrsm_vbatched_nrhs1.cu, hidden symbols)
extern "C" __attribute__((visibility("hidden"))) size_t magma_ztrsm_nrhs1_ws_bytes(magma_int_t batchCount);
extern "C" __attribute__((visibility("hidden"))) void
magma_ztrsm_nrhs1_update(
    void *ws, magma_int_t M, magma_int_t K, magmaDoubleComplex alpha, magmaDoubleComplex beta,
    magma_int_t Ai, magma_int_t Aj, magma_int_t Bi, magma_int_t Bj, magma_int_t Ci, magma_int_t Cj,
    magma_int_t *m, magma_int_t *n,
    magmaDoubleComplex **dA, magma_int_t *ldda, magmaDoubleComplex **dB, magma_int_t *lddb,
    magma_int_t batchCount, magma_queue_t queue);

// G-GEMV2: blocked nrhs=1 leaf, 32 < max_m <= 256 rows, one CTA per matrix (ztrsm_small_vbatched.cu, hidden symbol)
extern "C" __attribute__((visibility("hidden"))) void
magma_ztrsm_nrhs1_blk(
    magma_uplo_t uplo, magma_diag_t diag,
    magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
    magmaDoubleComplex alpha,
    magmaDoubleComplex **dA, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    magmaDoubleComplex **dB, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    magma_int_t batchCount, magma_queue_t queue );

extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_ztrsm_nrhs1_coop(
        magma_uplo_t uplo, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue );
// recursion; nrhs1_ws != NULL (only for Left/NoTrans with max_n == 1) routes the off-diagonal updates to batched gemv
static void
ztrsm_vbatched_core_rec(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue, void *nrhs1_ws )
{
#define dA_array(i,j) dA_array, i, j
#define dB_array(i,j) dB_array, i, j

    const magmaDoubleComplex c_one        = MAGMA_Z_ONE;
    const magmaDoubleComplex c_negone     = MAGMA_Z_NEG_ONE;
    const magmaDoubleComplex c_neg_ialpha = MAGMA_Z_DIV(c_negone, alpha);

    if(max_m == 0 || max_n == 0) return;

    magma_int_t max_nrowA = (side == MagmaLeft ? max_m : max_n);
    magma_int_t shape = 0;
    if      (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 0; } // lNL
    else if (side == MagmaLeft   && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 1; } // lNU
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 2; } // lTL | lCL
    else if (side == MagmaLeft   && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 3; } // lTU | lCU
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaLower) { shape = 4; } // rNL
    else if (side == MagmaRight  && transA == MagmaNoTrans  && uplo == MagmaUpper) { shape = 5; } // rNU
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaLower) { shape = 6; } // rTL | rCL
    else if (side == MagmaRight  && transA != MagmaNoTrans  && uplo == MagmaUpper) { shape = 7; } // rTU | rCU

    // use the fixed-size batched function to get nb
    magma_int_t batrsm_stop_nb = magma_get_ztrsm_batched_stop_nb(side, max_m, max_n);
    // G-GEMV2: on the nrhs=1 path the recursion stops at trsm_nrhs1_stop() rows (256) instead of 32; leaves of 33..256 rows go to
    // the blocked one-CTA-per-matrix kernel, which does the sub-block off-diagonal updates inside one launch
    if (nrhs1_ws) batrsm_stop_nb = max(batrsm_stop_nb, (magma_int_t)trsm_nrhs1_stop());
    // S-TOPCHAIN: a tall single-RHS triangular solve (Left/NoTrans, n > the leaf size) goes to one cooperative
    // kernel per batch instead of the recursion (falls back when the launch cannot be made co-resident)
    if (nrhs1_ws && (shape == 0 || shape == 1) && max_nrowA > batrsm_stop_nb) {
        if (magma_ztrsm_nrhs1_coop(uplo, diag, max_m, max_n, m, n, alpha,
                                   dA_array(Ai, Aj), ldda, dB_array(Bi, Bj), lddb, batchCount, queue) == 0)
            return;
    }
    // stopping condition
    if(max_nrowA <= batrsm_stop_nb){
        if (nrhs1_ws && max_nrowA > 32) {
            magma_ztrsm_nrhs1_blk(uplo, diag, max_m, max_n, m, n, alpha,
                                  dA_array(Ai, Aj), ldda, dB_array(Bi, Bj), lddb, batchCount, queue );
            return;
        }
        magmablas_ztrsm_small_vbatched(
                side, uplo, transA, diag,
                max_m, max_n, m, n, alpha,
                dA_array(Ai, Aj), ldda,
                dB_array(Bi, Bj), lddb,
                batchCount, queue );
        return;
    }

    switch(shape)
    {
        case 0: // lNl
            {
                const int m2 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                //printf("trsm1: m1 = %d, m2 = %d\n", m1, m2);
                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

                if (nrhs1_ws) {
                    magma_ztrsm_nrhs1_update(
                        nrhs1_ws, m2, m1, c_negone, alpha,
                        Ai+m1, Aj, Bi, Bj, Bi+m1, Bj,
                        m, n, dA_array, ldda, dB_array, lddb, batchCount, queue );
                } else
                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        c_negone, dA_array, Ai+m1, Aj, ldda,
                                  dB_array, Bi,    Bj, lddb,
                        alpha   , dB_array, Bi+m1, Bj, lddb,
                        batchCount, queue );

                //printf("trsm2: m1 = %d, m2 = %d\n", m1, m2);
                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, c_one,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 1: // lNU
            {
                const int m1 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m2 = max_m - m1;

                //printf("trsm1: m1 = %d, m2 = %d\n", m1, m2);
                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue, nrhs1_ws );

                if (nrhs1_ws) {
                    magma_ztrsm_nrhs1_update(
                        nrhs1_ws, m1, m2, c_neg_ialpha, c_one,
                        Ai, Aj+m1, Bi+m1, Bj, Bi, Bj,
                        m, n, dA_array, ldda, dB_array, lddb, batchCount, queue );
                } else
                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        c_neg_ialpha, dA_array, Ai, Aj+m1, ldda,
                                      dB_array, Bi+m1, Bj, lddb,
                        c_one       , dB_array, Bi,    Bj, lddb,
                        batchCount, queue );

                //printf("trsm2: m1 = %d, m2 = %d\n", m1, m2);
                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

            }
            break;
        case 2: // lTL || lCL
            {
                const int m2 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m1 = max_m - m2;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, alpha,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m1, max_n, m2,
                        m, n, m,
                        c_neg_ialpha, dA_array, Ai+m1, Aj, ldda,
                                      dB_array, Bi+m1, Bj, lddb,
                        c_one,        dB_array, Bi,    Bj, lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 3: // lTU | lCU
            {
                const int m1 = magma_get_ztrsm_vbatched_nb(max_m);
                const int m2 = max_m - m1;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m1, max_n, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        transA, MagmaNoTrans,
                        m2, max_n, m1,
                        m, n, m,
                        c_negone, dA_array, Ai,    Aj+m1, ldda,
                                  dB_array, Bi,    Bj,    lddb,
                        alpha   , dB_array, Bi+m1, Bj,    lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        m2, max_n, m, n, c_one,
                        dA_array(Ai+m1, Aj+m1), ldda,
                        dB_array(Bi+m1, Bj),    lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 4: // rNL
             {
                const int n2 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        c_neg_ialpha, dB_array, Bi,    Bj+n1, lddb,
                                      dA_array, Ai+n1, Aj,    ldda,
                        c_one       , dB_array, Bi,    Bj,    lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 5: // rNU
            {
                const int n1 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        c_negone, dB_array, Bi, Bj,    lddb,
                                  dA_array, Ai, Aj+n1, ldda,
                        alpha   , dB_array, Bi, Bj+n1, lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, c_one,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 6: // rTL | rCL
            {
                const int n1 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n2 = max_n - n1;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n2, n1,
                        m, n, n,
                        c_negone, dB_array, Bi,    Bj,    lddb,
                                  dA_array, Ai+n1, Aj,    ldda,
                        alpha   , dB_array, Bi,    Bj+n1, lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, c_one,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        case 7: // rTU | rCU
            {
                const int n2 = magma_get_ztrsm_vbatched_nb(max_n);
                const int n1 = max_n - n2;

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n2, m, n, alpha,
                        dA_array(Ai+n1, Aj+n1), ldda,
                        dB_array(Bi, Bj+n1),    lddb,
                        batchCount, queue, nrhs1_ws );

                magmablas_zgemm_vbatched_core(
                        MagmaNoTrans, transA,
                        max_m, n1, n2,
                        m, n, n,
                        c_neg_ialpha, dB_array, Bi, Bj+n1, lddb,
                                      dA_array, Ai, Aj+n1, ldda,
                        c_one       , dB_array, Bi, Bj,    lddb,
                        batchCount, queue );

                ztrsm_vbatched_core_rec(
                        side, uplo, transA, diag,
                        max_m, n1, m, n, alpha,
                        dA_array(Ai, Aj), ldda,
                        dB_array(Bi, Bj), lddb,
                        batchCount, queue, nrhs1_ws );
            }
            break;
        default:; // propose something
    }
#undef dA_array
#undef dB_array
}
///////////////////////////////////////////////////////////////////////////////////////////////////

// G-TRSV1: private stream-ordered pool that keeps its memory (release threshold = max). The default pool
// (threshold 0) hands memory back at every stream sync, so each cudaMallocAsync cost ~0.4 ms in the
// JOREK-proxy solve (window 16: 600 calls = 245.7 ms). One pool per device, created on first use; NULL on failure.
static cudaMemPool_t z_trsm_nrhs1_pool()
{
    static std::mutex mtx;
    static cudaMemPool_t pools[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return NULL; }
    std::lock_guard<std::mutex> lock(mtx);
    if (pools[dev] == NULL) {
        cudaMemPoolProps props = {};
        props.allocType     = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id   = dev;
        cudaMemPool_t p = NULL;
        if (cudaMemPoolCreate(&p, &props) != cudaSuccess) { cudaGetLastError(); return NULL; }
        cuuint64_t keep = ~(cuuint64_t)0;
        cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &keep);
        pools[dev] = p;
    }
    return pools[dev];
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// G-TRSV1: Left/NoTrans with a single right-hand side (max_n == 1) does its off-diagonal updates as batched gemv.
// The pointer/extent workspace is stream-ordered (cudaMallocFromPoolAsync/cudaFreeAsync on the queue's stream, private
// pool above): no host sync,
// the call stays fully asynchronous. If the allocation fails the original gemm path is used.
extern "C" void
magmablas_ztrsm_vbatched_core(
        magma_side_t side, magma_uplo_t uplo, magma_trans_t transA, magma_diag_t diag,
        magma_int_t max_m, magma_int_t max_n, magma_int_t* m, magma_int_t* n,
        magmaDoubleComplex alpha,
        magmaDoubleComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
        magmaDoubleComplex **dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
        magma_int_t batchCount, magma_queue_t queue )
{
    // G-GEMV2: ws is the nrhs=1 path flag. A real workspace is allocated only if at least one update node exists
    // (max_m > trsm_nrhs1_stop() = 256 rows); a block that fits one leaf needs no workspace, so a non-NULL sentinel
    // (never dereferenced or freed) selects the blocked leaf kernel. If the allocation fails ws stays NULL = original path.
    void *ws = NULL;
    bool ws_real = false;
    if (max_m > 0 && max_n == 1 && batchCount > 0 && side == MagmaLeft && transA == MagmaNoTrans)
    {
        if (max_m > max(magma_get_ztrsm_batched_stop_nb(side, max_m, max_n), (magma_int_t)trsm_nrhs1_stop()))   // at least one update happens
        {
            cudaStream_t st = magma_queue_get_cuda_stream(queue);
            cudaMemPool_t pool = z_trsm_nrhs1_pool();
            if (pool == NULL || cudaMallocFromPoolAsync(&ws, magma_ztrsm_nrhs1_ws_bytes(batchCount), pool, st) != cudaSuccess) { ws = NULL; cudaGetLastError(); }
            else ws_real = true;
        }
        else ws = (void*)&ws;   // sentinel: no update node, only the leaf kernel
    }
    ztrsm_vbatched_core_rec(side, uplo, transA, diag, max_m, max_n, m, n, alpha,
                            dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, batchCount, queue, ws);
    if (ws_real) cudaFreeAsync(ws, magma_queue_get_cuda_stream(queue));
}
