// G-TRSV1: per-matrix effective extent of one off-diagonal update of the N=1 (single right-hand side) trsm_vbatched
// recursion, computed exactly as gemm_template_vbatched_nn_kernel clamps (my_M, my_N, my_K) for the gemm call
// gemm(M, N=1, K, mM=m[k], mN=n[k], mK=m[k], A@(Ai,Aj), B@(Bi,Bj), C@(Ci,Cj)).
// Returns the gemv extent (rows = my_M, cols = my_K). rows = cols = 0 means "gemm would do nothing".
// Shared by the device kernel (ztrsm_vbatched_nrhs1.cu) and the host test (bench/trsv1).
#ifndef TRSM_NRHS1_CLAMP_H
#define TRSM_NRHS1_CLAMP_H
#ifdef __CUDACC__
#define TRSM_NRHS1_HD __host__ __device__
#else
#define TRSM_NRHS1_HD
#endif
TRSM_NRHS1_HD static inline void
trsm_nrhs1_clamp(int mM, int mN, int mK, int M, int K,
                 int Ai, int Aj, int Bi, int Bj, int Ci, int Cj, int *rows, int *cols)
{
    *rows = 0; *cols = 0;
    if (mM < Ai || mK < Aj) return;
    if (mK < Bi || mN < Bj) return;
    if (mM < Ci || mN < Cj) return;
    int eM = mM - (Ai > Ci ? Ai : Ci);
    int eN = mN - (Bj > Cj ? Bj : Cj);
    int eK = mK - (Aj > Bi ? Aj : Bi);
    if (eM > M) eM = M;
    if (eN > 1) eN = 1;
    if (eK > K) eK = K;
    if (eM <= 0 || eN <= 0 || eK <= 0) return;
    *rows = eM; *cols = eK;
}
#endif
