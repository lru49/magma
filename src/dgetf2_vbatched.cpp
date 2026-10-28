/*
   -- MAGMA (version 2.0) --
   Univ. of Tennessee, Knoxville
   Univ. of California, Berkeley
   Univ. of Colorado, Denver
   @date

   @author Azzam Haidar
   @author Tingxing Dong

   @generated from src/zgetf2_vbatched.cpp, normal z -> d, Tue Oct 27 22:04:46 2026
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include <cstdlib>

extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_dgetf2_tall_vbatched(
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t *m, magma_int_t *n,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue);

// F-ONLINEPIVOT (magmablas/dgetf2_onlinepiv_vbatched.cu): update kernel that also emits per-tile pivot partials of the
// next column, and the swap kernel that reduces them (replaces the idamax launch); MAGMA_GETF2_ONLINE_PIVOT=1 enables.
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_dscal_dger_onlinepiv_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        double *pval, int *pidx, magma_int_t npart,
        magma_int_t batchCount, magma_queue_t queue);
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_dswap_onlinepiv_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t *info_array, magma_int_t gbstep,
        const double *pval, const int *pidx, magma_int_t npart, magma_int_t ger_max_m,
        magma_int_t batchCount, magma_queue_t queue);

static int dgetf2_online_pivot_mode()
{
    static int mode = -1;
    if (mode < 0) {
        const char *e = getenv("MAGMA_GETF2_ONLINE_PIVOT");
        mode = (e && atoi(e) == 1) ? 1 : 0;                 // default OFF: the per-column path is untouched
    }
    return mode;
}

static magma_int_t
magma_dgetf2_vbatched_v1(
    magma_int_t *m, magma_int_t *n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
#define dA_array(i,j) dA_array, (i), (j)
#define ipiv_array(i) ipiv_array, (i)

    magma_int_t arginfo = 0;

    magma_int_t j;
    magma_int_t nb = 8;

    // F-ONLINEPIVOT scratch: the queue's idle dCarray (maxBatch pointers) holds the per-tile partials between the update
    // kernel of column j and the swap kernel of column j+1 (nothing else is enqueued on this queue in between).
    const int online = dgetf2_online_pivot_mode();
    double *pval = NULL; int *pidx = NULL; magma_int_t pcap = 0;
    if (online) {
        size_t bytes = (size_t)queue->get_maxBatch() * sizeof(void*);
        pcap = (magma_int_t)(bytes / (sizeof(double) + sizeof(int)));
        char *scratch = (char*)queue->get_dCarray();
        pval = (double*)scratch;
        pidx = (int*)(scratch + (size_t)pcap * sizeof(double));
    }
    magma_int_t have_partials = 0, npart_prev = 0, germ_prev = 0;

    for(j=0; j < max_minmn; j+=nb) {
        // panel (the swap is done on the entire width)
        magma_int_t ib = min(nb, max_minmn-j);
        for(magma_int_t jj = 0; jj < ib; jj++) {
            magma_int_t gbj = j+jj;
            // idamax (skipped when the previous update kernel emitted the pivot partials of this column)
            if (!have_partials)
                magma_idamax_vbatched(max_m-gbj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, ipiv_array(Ai+gbj), info_array, gbj, gbstep, batchCount, queue);

            // dswap (the online variant first reduces the partials to ipiv/info exactly as idamax would)
            if (have_partials)
                magma_dswap_onlinepiv_vbatched(max_n, m, n, dA_array(Ai+gbj, Aj), ldda, ipiv_array, gbj, info_array, gbstep,
                                               pval, pidx, npart_prev, germ_prev, batchCount, queue);
            else
                magma_dswap_vbatched(max_n, m, n, dA_array(Ai+gbj, Aj), ldda, ipiv_array, gbj, batchCount, queue);

            // scal+ger (+ pivot partials of column gbj+1 when it belongs to this block and the scratch holds them)
            magma_int_t germ = max_m-gbj, npart = magma_ceildiv(germ, 256);
            if (online && jj < ib-1 && germ >= 2 && batchCount*npart <= pcap) {
                magma_dscal_dger_onlinepiv_vbatched( germ, ib-jj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, info_array, gbj, gbstep,
                                                      pval, pidx, npart, batchCount, queue);
                have_partials = 1; npart_prev = npart; germ_prev = germ;
            }
            else {
                magma_dscal_dger_vbatched( germ, ib-jj, m, n, dA_array(Ai+gbj, Aj+gbj), ldda, info_array, gbj, gbstep, batchCount, queue);
                have_partials = 0;
            }
        }

        // trsm
        magmablas_dtrsm_vbatched_core(
            MagmaLeft, MagmaLower, MagmaNoTrans, MagmaUnit,
            ib, max_n-j-ib, m, n, MAGMA_D_ONE,
            dA_array(Ai+j, Aj+j   ), ldda,
            dA_array(Ai+j, Aj+j+ib), ldda, batchCount, queue );

        // gemm
        magmablas_dgemm_vbatched_core(
            MagmaNoTrans, MagmaNoTrans,
            max_m-j-ib, max_n-j-ib, ib,
            m, n, minmn,
            MAGMA_D_NEG_ONE, dA_array, Ai+j+ib, Aj+j,    ldda,
                             dA_array, Ai+j,    Aj+j+ib, ldda,
            MAGMA_D_ONE,     dA_array, Ai+j+ib, Aj+j+ib, ldda,
            batchCount, queue );
    }

    return arginfo;

#undef dA_array
#undef ipiv_array
}



/***************************************************************************//**
    Purpose
    -------
    DGETF2 computes an LU factorization of a general M-by-N matrix A
    using partial pivoting with row interchanges.

    The factorization has the form
        A = P * L * U
    where P is a permutation matrix, L is lower triangular with unit
    diagonal elements (lower trapezoidal if m > n), and U is upper
    triangular (upper trapezoidal if m < n).

    This is the right-looking Level 3 BLAS version of the algorithm.

    This is a batched version that factors batchCount M-by-N matrices in parallel.
    dA, ipiv, and info become arrays with one entry per matrix.

    Arguments
    ---------
    @param[in]
    m       INTEGER
            The number of rows of each matrix A.  M >= 0.

    @param[in]
    n       INTEGER
            The number of columns of each matrix A.  N >= 0.

    @param[in,out]
    dA_array    Array of pointers, dimension (batchCount).
            Each is a DOUBLE PRECISION array on the GPU, dimension (LDDA,N).
            On entry, each pointer is an M-by-N matrix to be factored.
            On exit, the factors L and U from the factorization
            A = P*L*U; the unit diagonal elements of L are not stored.

    @param[in]
    ai      INTEGER
            Row offset for A.

    @param[in]
    aj      INTEGER
            Column offset for A.

    @param[in]
    ldda    INTEGER
            The leading dimension of each array A.  LDDA >= max(1,M).

    @param[out]
    ipiv_array  Array of pointers, dimension (batchCount), for corresponding matrices.
            Each is an INTEGER array, dimension (min(M,N))
            The pivot indices; for 1 <= i <= min(M,N), row i of the
            matrix was interchanged with row IPIV(i).

    @param[out]
    info_array  Array of INTEGERs, dimension (batchCount), for corresponding matrices.
      -     = 0:  successful exit
      -     < 0:  if INFO = -i, the i-th argument had an illegal value
                  or another error occured, such as memory allocation failed.
      -     > 0:  if INFO = i, U(i,i) is exactly zero. The factorization
                  has been completed, but the factor U is exactly
                  singular, and division by zero will occur if it is used
                  to solve a system of equations.

    @param[in]
    gbstep  INTEGER
            internal use.

    @param[in]
    batchCount  INTEGER
                The number of matrices to operate on.

    @param[in]
    queue   magma_queue_t
            Queue to execute in.

    this is an internal routine that might have many assumption.


    @ingroup magma_getf2_batched
*******************************************************************************/
extern "C" magma_int_t
magma_dgetf2_vbatched(
    magma_int_t *m, magma_int_t *n, magma_int_t* minmn,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_minmn, magma_int_t max_mxn,
    double **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t *info_array,
    magma_int_t gbstep, magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t arginfo = -1;

    // first, test the fused panel
    arginfo = magma_dgetf2_fused_vbatched(
                max_m, max_n, max_minmn, max_mxn,
                m, n,
                dA_array, Ai, Aj, ldda,
                ipiv_array, Ai, info_array,
                batchCount, queue);
    if(arginfo == 0) return arginfo;

    // F-TALLPANEL: one fused CTA per matrix for panels too tall for the fused kernels (MAGMA_GETF2_TALL=0 disables)
    arginfo = magma_dgetf2_tall_vbatched(
                max_m, max_n, m, n,
                dA_array, Ai, Aj, ldda,
                ipiv_array, Ai, info_array, gbstep,
                batchCount, queue);
    if(arginfo == 0) return arginfo;

    magma_dgetf2_vbatched_v1(
        m, n,
        minmn, max_m, max_n, max_minmn, max_mxn,
        dA_array, Ai, Aj, ldda,
        ipiv_array, info_array, gbstep,
        batchCount, queue);

    return 0;
}

