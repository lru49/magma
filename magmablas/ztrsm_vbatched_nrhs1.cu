/*
    -- MAGMA (version 2.0) --
       @precisions normal z -> s d c

    G-TRSV1: N=1 (single right-hand side) off-diagonal update of ztrsm_vbatched_core as a batched GEMV.
    One tiny launch computes, per matrix, the displaced A/x/y pointers and the effective extent (the same clamp
    gemm_template_vbatched_nn_kernel applies), then zgemv_vbatched_max_nocheck does y = alpha*A*x + beta*y.
*/
#include "magma_internal.h"
#include "trsm_nrhs1_clamp.h"

#define PRECISION_z

#define HIDDEN __attribute__((visibility("hidden")))

__global__ void ztrsm_nrhs1_prep_kernel(
    const magma_int_t *m, const magma_int_t *n,
    magmaDoubleComplex **dA, const magma_int_t *ldda, magmaDoubleComplex **dB, const magma_int_t *lddb,
    magmaDoubleComplex **sA, magmaDoubleComplex **sx, magmaDoubleComplex **sy,
    magma_int_t *rows, magma_int_t *cols, magma_int_t *ones,
    int M, int K, int Ai, int Aj, int Bi, int Bj, int Ci, int Cj, int batch)
{
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= batch) return;
    int er, ec;
    trsm_nrhs1_clamp((int)m[k], (int)n[k], (int)m[k], M, K, Ai, Aj, Bi, Bj, Ci, Cj, &er, &ec);
    magmaDoubleComplex *a = dA[k], *b = dB[k];
    if (a == NULL || b == NULL) { er = 0; ec = 0; }
    ones[k] = 1; rows[k] = er; cols[k] = ec;
    if (er > 0) {
        sA[k] = a + Ai + (size_t)ldda[k] * Aj;
        sx[k] = b + Bi + (size_t)lddb[k] * Bj;
        sy[k] = b + Ci + (size_t)lddb[k] * Cj;
    } else { sA[k] = NULL; sx[k] = NULL; sy[k] = NULL; }
}

// workspace layout: sA[batch], sx[batch], sy[batch] (pointers) then rows[batch], cols[batch], ones[batch] (magma_int_t)
extern "C" HIDDEN size_t magma_ztrsm_nrhs1_ws_bytes(magma_int_t batchCount)
{
    return (size_t)batchCount * (3 * sizeof(magmaDoubleComplex*) + 3 * sizeof(magma_int_t));
}

extern "C" HIDDEN void
magma_ztrsm_nrhs1_update(
    void *ws, magma_int_t M, magma_int_t K,
    magmaDoubleComplex alpha, magmaDoubleComplex beta,
    magma_int_t Ai, magma_int_t Aj, magma_int_t Bi, magma_int_t Bj, magma_int_t Ci, magma_int_t Cj,
    magma_int_t *m, magma_int_t *n,
    magmaDoubleComplex **dA, magma_int_t *ldda, magmaDoubleComplex **dB, magma_int_t *lddb,
    magma_int_t batchCount, magma_queue_t queue)
{
    magmaDoubleComplex **sA = (magmaDoubleComplex**)ws, **sx = sA + batchCount, **sy = sx + batchCount;
    magma_int_t *rows = (magma_int_t*)(sy + batchCount), *cols = rows + batchCount, *ones = cols + batchCount;
    ztrsm_nrhs1_prep_kernel<<<(batchCount + 127) / 128, 128, 0, queue->cuda_stream()>>>(
        m, n, dA, ldda, dB, lddb, sA, sx, sy, rows, cols, ones, M, K, Ai, Aj, Bi, Bj, Ci, Cj, batchCount);
    magmablas_zgemv_vbatched_max_nocheck(MagmaNoTrans, rows, cols, alpha, sA, ldda /*unused below*/, sx, ones, beta, sy, ones,
                                         batchCount, M, K, queue);
}
