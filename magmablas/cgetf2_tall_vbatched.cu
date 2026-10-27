/*
    -- MAGMA (version 2.10.0) -- magma-hpc F track addition (BSD-3 like the rest of MAGMA)
       F-TALLPANEL: fused LU of a tall narrow panel for the batched variable-size getf2.
       Kept in its own translation unit so an edit here recompiles in seconds instead of
       the five minutes cgetf2_kernels_var.cu takes (its 32 fused-kernel instantiations).

       @generated from magmablas/zgetf2_tall_vbatched.cu, normal z -> c, Tue Oct 27 19:04:58 2026
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "magma_templates.h"
#include "shuffle.cuh"
#include "cgetf2_devicefunc.cuh"
#include <cstdlib>

/******************************************************************************/
// F-TALLPANEL (magma-hpc F track, 2026-10-27): fused LU of a tall narrow panel (m up to CGETF2_TALL_MAX_M rows, n <= 32
// columns) with ONE thread block per matrix, for the panel sizes the register- and shared-memory-fused kernels cannot take
// (m above a few hundred rows). It replaces the per-column icamax / cswap / cscal_cgeru launch loop of cgetf2_vbatched_v1
// (3 launches per column: 8448 launches for the 3200-row root front of the JOREK proxy). Pivot choice = icamax_devfunc
// (first maximum of |re|+|im|), scaling = reciprocal multiply as in cscal_cgeru_generic_device, singular column => no swap and
// info set like the per-column path. The remaining panel columns are updated column by column (rank-1 from the staged pivot
// row), where cgetf2_vbatched_v1 applies a blocked trsm/gemm order every 8 columns: equal to rounding, not bitwise.
#define CGETF2_TALL_NT    1024
#define CGETF2_TALL_MAX_M 4096
#define CGETF2_TALL_MAX_N 32
__global__ void
cgetf2_tall_kernel_vbatched(
        magma_int_t *M, magma_int_t *N,
        magmaFloatComplex **dA_array, int Ai, int Aj, magma_int_t *ldda,
        magma_int_t **ipiv_array, int ipiv_i,
        magma_int_t *info_array, int gbstep )
{
    extern __shared__ float sdata[];
    const int batchid = blockIdx.x, tx = threadIdx.x;
    int my_M = (int)M[batchid], my_N = (int)N[batchid];
    const int my_ldda = (int)ldda[batchid];
    if (my_M <= Ai || my_N <= Aj) return;
    my_M -= Ai; my_N -= Aj;
    const int n     = min(my_N, CGETF2_TALL_MAX_N);
    const int minmn = min(my_M, n);
    magmaFloatComplex *dA   = dA_array[batchid] + (size_t)Aj * my_ldda + Ai;
    magma_int_t        *ipiv = ipiv_array[batchid] + ipiv_i;
    magma_int_t        *info = &info_array[batchid];
    float *shared_x   = sdata;
    int    *shared_idx = (int*)(shared_x + zamax);
    magmaFloatComplex *su = (magmaFloatComplex*)(shared_x + zamax + (zamax * sizeof(int)) / sizeof(float));
    __shared__ int sjp;
    __shared__ int ssing;
    int linfo = (gbstep == 0) ? 0 : (int)(*info);
    for (int j = 0; j < minmn; j++) {
        magmaFloatComplex *col = dA + (size_t)j * my_ldda;
        icamax_devfunc(my_M - j, col + j, 1, shared_x, shared_idx);   // all threads: it synchronizes inside
        if (tx == 0) {
            const int jp = shared_idx[0] + j;
            sjp   = jp;
            ssing = (shared_x[0] == MAGMA_D_ZERO);
            ipiv[j] = jp + 1;                       // Fortran indexing, relative to row Ai
            linfo = (shared_x[0] == MAGMA_D_ZERO && linfo == 0) ? (j + gbstep + 1) : linfo;
        }
        __syncthreads();
        const int jp = sjp;
        if (tx < n) {                               // swap rows j and jp across the panel, stage the pivot row
            magmaFloatComplex *c = dA + (size_t)tx * my_ldda;
            magmaFloatComplex piv = c[jp];
            if (jp != j && !ssing) { c[jp] = c[j]; c[j] = piv; }
            else piv = c[j];
            su[tx] = piv;
        }
        __syncthreads();
        const magmaFloatComplex rA0 = su[j];
        const float rTmp = fabs(MAGMA_C_REAL(rA0)) + fabs(MAGMA_C_IMAG(rA0));
        const magmaFloatComplex reg = (rTmp == MAGMA_D_ZERO) ? MAGMA_C_ONE : MAGMA_C_DIV(MAGMA_C_ONE, rA0);
        for (int i = j + 1 + tx; i < my_M; i += CGETF2_TALL_NT) {
            magmaFloatComplex rA = col[i];
            rA *= reg;
            col[i] = rA;
            for (int c = j + 1; c < n; c++)
                dA[(size_t)c * my_ldda + i] -= rA * su[c];
        }
        __syncthreads();
    }
    if (tx == 0) *info = (magma_int_t)linfo;
}

/******************************************************************************/
extern "C" __attribute__((visibility("hidden"))) magma_int_t
magma_cgetf2_tall_vbatched(
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t *m, magma_int_t *n,
    magmaFloatComplex **dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t *ldda,
    magma_int_t **ipiv_array, magma_int_t ipiv_i,
    magma_int_t *info_array, magma_int_t gbstep,
    magma_int_t batchCount, magma_queue_t queue)
{
    if (max_n > CGETF2_TALL_MAX_N || max_m > CGETF2_TALL_MAX_M || max_m < 1) return -1;
    static int enabled = -1;
    if (enabled < 0) { const char* e = getenv("MAGMA_GETF2_TALL"); enabled = (e && atoi(e) == 0) ? 0 : 1; }
    if (!enabled) return -1;
    const size_t shmem = zamax * sizeof(float) + zamax * sizeof(int) + CGETF2_TALL_MAX_N * sizeof(magmaFloatComplex);
    const magma_int_t max_batch = 65535;
    for (magma_int_t i = 0; i < batchCount; i += max_batch) {
        magma_int_t ib = min(max_batch, batchCount - i);
        dim3 grid(ib, 1, 1), threads(CGETF2_TALL_NT, 1, 1);
        cgetf2_tall_kernel_vbatched<<<grid, threads, shmem, queue->cuda_stream()>>>
            (m + i, n + i, dA_array + i, Ai, Aj, ldda + i, ipiv_array + i, ipiv_i, info_array + i, gbstep);
    }
    return 0;
}
