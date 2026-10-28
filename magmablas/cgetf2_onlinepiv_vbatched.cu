/*
    -- MAGMA (version 2.10.0) -- magma-hpc F track addition (BSD-3 like the rest of MAGMA)
       F-ONLINEPIVOT: the pivot search of column j+1 fused into the rank-1 update of column j
       for the per-column path of the batched variable-size getf2 (cgetf2_vbatched_v1).
       Own translation unit (cgetf2_kernels_var.cu takes minutes to compile).

       @generated from magmablas/zgetf2_onlinepiv_vbatched.cu, normal z -> c, Tue Oct 27 23:30:03 2026
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "cgetf2_devicefunc.cuh"
#include <climits>

// F-ONLINEPIVOT (magma-hpc F track, 2026-10-27). In cgetf2_vbatched_v1 every column costs three dependent launches:
// icamax over the column, cswap of the pivot row, cscal_cgeru (scale + rank-1 update of the remaining block columns).
// The update kernel of column j already holds the updated values of column j+1 in registers, so it emits, per 256-row
// tile (CTA) and per matrix, the partial (max |a_{i,j+1}|, first row i of that max) into a small device scratch; the
// swap kernel of column j+1 reduces the <= ceil(m/256) partials of its matrix to the pivot and writes ipiv/info exactly
// as icamax_kernel_vbatched would, then swaps. The icamax launch and its pass over the column disappear for every
// column that is not the first of an 8-column block (the first one is updated by the blocked trsm/gemm, not by cgeru).
//
// Exactness: icamax_devfunc returns the first index of the maximum of (fabs(re)+fabs(im)) where NaN never wins (strict >
// from 0, then (value, index) max with the smaller index on ties). (max value, smallest index) is order independent once
// NaN is mapped to 0, so the tile reduce + partial reduce reproduce icamax bit for bit; the scale/update arithmetic is
// the verbatim expression of cscal_cgeru_generic_device.
//
// Partial format: two arrays indexed [batchid * npart + tile]: pval (float, |value| of the tile maximum, NaN -> 0) and
// pidx (int, row index relative to the first row of column j+1; INT_MAX when the tile holds no row of the column).
//
// Plant (-DMAGMA_PLANT_ONLINE_PIVOT): the consumer ignores the last partial of every matrix (the pivot can only be found
// in the first ceil(m/256)-1 tiles) -> occasionally wrong pivot -> must fail the 1e-12 solve compare.

#define ONLINEPIV_TBX 256

/******************************************************************************/
__global__ void
cscal_cgeru_1d_onlinepiv_kernel_vbatched(
        int max_m, int max_n,
        magma_int_t *M, magma_int_t *N,
        magmaFloatComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t *info_array, int step, int gbstep,
        float *pval, int *pidx, int npart)
{
    const int batchid = blockIdx.z;
    const int tx  = threadIdx.x;
    const int gtx = blockIdx.x * blockDim.x + tx;
    __shared__ float sx[ONLINEPIV_TBX / 32];
    __shared__ int    si[ONLINEPIV_TBX / 32];

    int my_M    = (int)M[batchid];
    int my_N    = (int)N[batchid];
    int my_ldda = (int)ldda[batchid];

    if( my_M <= Ai || my_N <= Aj ) return;      // uniform over the CTA: no partial needed (the consumer skips this matrix)
    my_M -= Ai;
    my_N -= Aj;
    my_M = min(my_M, max_m);
    my_N = min(my_N, max_n);

    magmaFloatComplex* dA = dA_array[batchid] + Aj * my_ldda + Ai;

    float v = MAGMA_D_ZERO;
    int    r = INT_MAX;
    if (gtx != 0 && gtx < my_M) {
        // === verbatim cscal_cgeru_generic_device ===
        magmaFloatComplex rA, reg;
        float rTmp;
        rA   = dA[0];
        rTmp = fabs(MAGMA_C_REAL( rA ) ) + fabs( MAGMA_C_IMAG( rA ) );

        reg = (rTmp == MAGMA_D_ZERO) ? MAGMA_C_ONE : MAGMA_C_DIV(MAGMA_C_ONE, rA);
        rA  = dA[ gtx ];
        rA *= reg;

        dA[ gtx ] = rA;
        #pragma unroll
        for(int i = 1; i < my_N; i++)
            dA[i * my_ldda + gtx] -= rA * dA[i * my_ldda + 0];
        // === end ===
        if (my_N >= 2) {
            magmaFloatComplex a1 = dA[ my_ldda + gtx ];               // updated value of column j+1 (just stored)
            v = fabs(MAGMA_C_REAL( a1 )) + fabs(MAGMA_C_IMAG( a1 ));   // same expression as icamax_devfunc
            if ( !(v > MAGMA_D_ZERO) ) v = MAGMA_D_ZERO;                // NaN never wins in icamax: treat as 0
            r = gtx - 1;                                                // row relative to the first row of column j+1
        }
    }
    // (max value, smallest index) reduce: warp shuffles, then the 8 warp results by thread 0 (order independent)
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, v, o);
        int    oi = __shfl_xor_sync(0xffffffff, r, o);
        if ( ov > v || (ov == v && oi < r) ) { v = ov; r = oi; }
    }
    if ((tx & 31) == 0) { sx[tx >> 5] = v; si[tx >> 5] = r; }
    __syncthreads();
    if (tx == 0) {
        #pragma unroll
        for (int w = 1; w < ONLINEPIV_TBX / 32; w++)
            if ( sx[w] > v || (sx[w] == v && si[w] < r) ) { v = sx[w]; r = si[w]; }
        pval[batchid * npart + blockIdx.x] = v;
        pidx[batchid * npart + blockIdx.x] = r;
    }
}

/******************************************************************************/
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_cscal_cgeru_onlinepiv_vbatched(
        magma_int_t max_M, magma_int_t max_N,
        magma_int_t *M, magma_int_t *N,
        magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t *info_array, magma_int_t step, magma_int_t gbstep,
        float *pval, int *pidx, magma_int_t npart,
        magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads(ONLINEPIV_TBX, 1, 1);
    if (npart != magma_ceildiv(max_M, ONLINEPIV_TBX)) return -1;

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid(npart, 1, ibatch);

        cscal_cgeru_1d_onlinepiv_kernel_vbatched<<<grid, threads, 0, queue->cuda_stream()>>>
        (max_M, max_N, M+i, N+i, dA_array+i, Ai, Aj, ldda+i, info_array+i, step, gbstep,
         pval + i*npart, pidx + i*npart, npart);
    }
    return 0;
}

/******************************************************************************/
// swap of column (Ai, Aj+piv_adjustment) whose pivot is first computed from the partials written by the update kernel
// of the previous column (which ran at rows Ai-1.., with ger_max_m = its max_m and npart tiles per matrix).
__global__
void cswap_onlinepiv_kernel_vbatched(
        int max_n, magma_int_t *M, magma_int_t *N,
        magmaFloatComplex **dA_array, int Ai, int Aj, magma_int_t* ldda,
        magma_int_t** ipiv_array, int piv_adjustment,
        magma_int_t *info_array, int gbstep,
        const float *pval, const int *pidx, int npart, int ger_max_m)
{
    const int batchid = blockIdx.x;
    const int tx      = threadIdx.x;
    const int my_ldda = (int)ldda[batchid];
    int my_M          = (int)M[batchid];
    int my_N          = (int)N[batchid];
    int my_minmn      = min(my_M, my_N);
    const int step    = piv_adjustment;
    const int Ajc     = Aj + piv_adjustment;          // the column's own column index (icamax was called at (Ai, Ajc))

    __shared__ int    jp;

    // ---- icamax_kernel_vbatched part: same guard, same outputs
    const bool do_piv = !( my_M <= Ai || my_N <= Ajc );
    if (do_piv) {
        magma_int_t *ipiv = ipiv_array[batchid] + Ai;
        magma_int_t *info = &info_array[batchid];
        int linfo = ( (gbstep+step) == 0) ? 0 : *info;
        // rows the producer handled for this matrix (relative to row Ai-1): min(M - (Ai-1), ger_max_m); tiles holding rows
        int rows_ger = min(my_M - (Ai-1), ger_max_m);
        int pm = magma_ceildiv(rows_ger, ONLINEPIV_TBX);
#ifdef MAGMA_PLANT_ONLINE_PIVOT
        if (pm > 1) pm -= 1;                            // PLANT: the last tile of every matrix is never searched
#endif
        if (tx < 32) {                                  // one warp: lanes strided over the tiles, then shuffle (max, min idx)
            const float *mv = pval + (size_t)batchid * npart;
            const int    *mi = pidx + (size_t)batchid * npart;
            float v = MAGMA_D_ZERO;
            int    r = INT_MAX;
            for (int p = tx; p < pm; p += 32) {
                float pv = mv[p];
                int    pi = mi[p];
                if ( pv > v || (pv == v && pi < r) ) { v = pv; r = pi; }
            }
            #pragma unroll
            for (int o = 16; o > 0; o >>= 1) {
                float ov = __shfl_xor_sync(0xffffffff, v, o);
                int    oi = __shfl_xor_sync(0xffffffff, r, o);
                if ( ov > v || (ov == v && oi < r) ) { v = ov; r = oi; }
            }
            if (tx == 0) {
                jp = r;
                *ipiv = r + step + 1;                       // Fortran indexing & adjust pivot
                linfo = ( v == MAGMA_D_ZERO && linfo == 0) ? (r+step+gbstep+1) : linfo;
                *info = (magma_int_t)linfo;
            }
        }
    }

    // ---- cswap_kernel_vbatched part (verbatim, the pivot read from shared when computed above)
    if( my_M <= Ai || my_N <= Aj || my_minmn <= Ai ) return;

    my_N -= Aj; // this is the maximum possible width
    my_N = min(my_N, max_n);

    if (threadIdx.x == 0 && !do_piv) {
        magma_int_t *ipiv = ipiv_array[batchid] + Ai;
        jp  = ipiv[0] - 1; // roll-back Fortran indexing
        jp -= piv_adjustment;
    }
    __syncthreads();

    if (jp == 0) return;

    magmaFloatComplex *dAorg = dA_array[batchid];
    magmaFloatComplex *dA    = dAorg + Aj * my_ldda + Ai;
    magmaFloatComplex *dA1   = dA;
    magmaFloatComplex *dA2   = dA + jp;

    magmaFloatComplex rA = dAorg[ (Aj+piv_adjustment) * my_ldda + Ai ];
    float rx_abs_max = fabs( MAGMA_C_REAL(rA) ) + fabs( MAGMA_C_IMAG(rA) );

    if(rx_abs_max == MAGMA_D_ZERO) return;

    cswap_device_v2(my_N, dA1, my_ldda, dA2, my_ldda );
}

/******************************************************************************/
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_cswap_onlinepiv_vbatched(
        magma_int_t max_n, magma_int_t *M, magma_int_t *N,
        magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
        magma_int_t** ipiv_array, magma_int_t piv_adjustment,
        magma_int_t *info_array, magma_int_t gbstep,
        const float *pval, const int *pidx, magma_int_t npart, magma_int_t ger_max_m,
        magma_int_t batchCount, magma_queue_t queue)
{
    dim3 grid(batchCount, 1, 1);
    dim3 threads(zamax, 1, 1);

    cswap_onlinepiv_kernel_vbatched<<< grid, threads, 0, queue->cuda_stream() >>>
    (max_n, M, N, dA_array, Ai, Aj, ldda, ipiv_array, piv_adjustment, info_array, gbstep,
     pval, pidx, npart, ger_max_m);

    return 0;
}
