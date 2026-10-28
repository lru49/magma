/*
    -- MAGMA (version 1.1) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Jakub Kurzak
       @author Stan Tomov
       @author Mark Gates
       @author Azzam Haidar
       @author Ahmad Abdelfattah

*/

#include "magma_internal.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define PRECISION_s

#include "gemm_template_kernel_vbatched.cuh"

#include "gemm_config/sgemm_param_nn.h"
#include "gemm_config/sgemm_param_nt.h"
#include "gemm_config/sgemm_param_tn.h"
#include "gemm_config/sgemm_param_tt.h"

#define version(s,v) s ## _V_ ## v

/******************************************************************************/
// F-SGEMM-TILE: zero-risk settings screen for the float NN vbatched kernel.
//   MAGMA_SGEMM_VBATCHED_NN_VERSION=<tag>  forces ONE kernel version for every float vbatched NN call
//                                          (read once; unset / unknown tag => the default dispatch below is unchanged).
//   MAGMA_SGEMM_VBATCHED_NN_MIN_MN=<n>     apply the forced version only to calls with min(max_m, max_n) >= n
//                                          (default 0 = every call; use it to leave the tiny getrf panel calls alone).
//   MAGMA_SGEMM_VBATCHED_NN_LOG=1          prints one line per float vbatched NN call to stderr (census).
// <tag> is an index of magmablas/gemm_config/sgemm_param_nn.h (NN_V_<tag>) that is instantiated in the list below,
// or one of the off-table tiles X1..X3 defined here. The template parameters are
//   DIM_X, DIM_Y, BLK_M, BLK_N, BLK_K, dim_vec, DIM_XA, DIM_YA, DIM_XB, DIM_YB  (threads = DIM_X*DIM_Y,
//   per-thread register tile = (BLK_M/DIM_X) x (BLK_N/DIM_Y); loaders need BLK_M%DIM_XA == BLK_K%DIM_YA ==
//   BLK_K%DIM_XB == BLK_N%DIM_YB == 0).
#define NN_V_X1  8,  8,  64,  64, 16, 1,  8,  8,  8,  8    // 8x8 per thread,  64 threads
#define NN_V_X2 16,  8, 128,  64, 16, 1, 16,  8, 16,  8    // 8x8 per thread, 128 threads
#define NN_V_X3 16, 16, 128, 128, 16, 1, 16, 16, 16, 16    // 8x8 per thread, 256 threads

// (tag, numeric code): table indices keep their index as code; off-table tiles use 10001+
#define SGEMM_VB_NN_FORCED_VERSIONS(X) \
    X(346, 346) X(398, 398) X(417, 417) X(483, 483) X(504, 504) X(507, 507) X(510, 510) X(512, 512) \
    X(518, 518) X(519, 519) X(532, 532) X(550, 550) X(551, 551) X(557, 557) X(596, 596) X(598, 598) \
    X(X1, 10001) X(X2, 10002) X(X3, 10003)

static int sgemm_vbatched_nn_forced_version()
{
    // -1 = unset (default dispatch), otherwise the numeric code of the forced version
    static const int forced = []() -> int {
        const char* e = getenv("MAGMA_SGEMM_VBATCHED_NN_VERSION");
        if (e == NULL || *e == '\0') return -1;
        #define SGEMM_VB_NN_MATCH(tag, code) if (strcmp(e, #tag) == 0) return code;
        SGEMM_VB_NN_FORCED_VERSIONS(SGEMM_VB_NN_MATCH)
        #undef SGEMM_VB_NN_MATCH
        fprintf(stderr, "MAGMA_SGEMM_VBATCHED_NN_VERSION=%s unknown; using the default dispatch. Known tags:", e);
        #define SGEMM_VB_NN_LIST(tag, code) fprintf(stderr, " %s", #tag);
        SGEMM_VB_NN_FORCED_VERSIONS(SGEMM_VB_NN_LIST)
        #undef SGEMM_VB_NN_LIST
        fprintf(stderr, "\n");
        return -1;
    }();
    return forced;
}

static magma_int_t sgemm_vbatched_nn_forced_min_mn()
{
    static const magma_int_t min_mn = []() -> magma_int_t {
        const char* e = getenv("MAGMA_SGEMM_VBATCHED_NN_MIN_MN");
        return (e == NULL || *e == '\0') ? 0 : (magma_int_t) atoll(e);
    }();
    return min_mn;
}

static bool sgemm_vbatched_nn_log_enabled()
{
    static const bool on = []() -> bool {
        const char* e = getenv("MAGMA_SGEMM_VBATCHED_NN_LOG");
        return (e != NULL && *e != '\0' && *e != '0');
    }();
    return on;
}

static inline void sgemm_vbatched_nn_log(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, const char* tag)
{
    if (sgemm_vbatched_nn_log_enabled())
        fprintf(stderr, "MAGMA_SGEMM_VBATCHED_NN max_m=%lld max_n=%lld max_k=%lld batchCount=%lld version=%s\n",
                (long long)max_m, (long long)max_n, (long long)max_k, (long long)batchCount, tag);
}

// launch version <code> (default dispatch or forced); returns the tag used (NULL if the code is not instantiated)
static const char* sgemm_vbatched_nn_forced_launch(
    int code,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float               **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    switch (code) {
        #define SGEMM_VB_NN_CASE(tag, c) \
        case c: \
            gemm_template_vbatched_nn<float, version(NN,tag), 0, 0> \
            (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue); \
            return #tag;
        SGEMM_VB_NN_FORCED_VERSIONS(SGEMM_VB_NN_CASE)
        #undef SGEMM_VB_NN_CASE
        default: return NULL;
    }
}

/******************************************************************************/
extern "C" void
magmablas_sgemm_vbatched_core(
    magma_trans_t transA, magma_trans_t transB,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float               **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue )
{
    if(max_m <=0 || max_n <= 0 || max_k < 0) return;

    magma_int_t shape = 0;
    if      (transA == MagmaNoTrans   && transB == MagmaNoTrans)   { shape = 0; } // nn
    else if (transA == MagmaNoTrans   && transB == MagmaTrans)     { shape = 1; } // nt
    else if (transA == MagmaNoTrans   && transB == MagmaConjTrans) { shape = 2; } // nc
    else if (transA == MagmaTrans     && transB == MagmaNoTrans)   { shape = 3; } // tn
    else if (transA == MagmaTrans     && transB == MagmaTrans)     { shape = 4; } // tt
    else if (transA == MagmaTrans     && transB == MagmaConjTrans) { shape = 5; } // tc
    else if (transA == MagmaConjTrans && transB == MagmaNoTrans)   { shape = 6; } // cn
    else if (transA == MagmaConjTrans && transB == MagmaTrans)     { shape = 7; } // ct
    else if (transA == MagmaConjTrans && transB == MagmaConjTrans) { shape = 8; } // cc

    switch(shape)
    {
        case 0: // nn
            {
                // F-SGEMM-TILE: one launcher for the default dispatch and the env override, so every template
                // instance keeps a single call site (inlined, no new exported symbols; SYMCHECK stays identical)
                int code = sgemm_vbatched_nn_forced_version();
                if (code >= 0 && min(max_m, max_n) < sgemm_vbatched_nn_forced_min_mn()) code = -1;
                if (code < 0)
                {
                    if(max_k < 64)
                    {
                        if(max_k==8 && max_n==24) code = 512;
                        else if (max_n<32)        code = 510;
                        else                      code = 504;
                    }
                    else                          code = 518;
                }
                const char* tag = sgemm_vbatched_nn_forced_launch(code,
                    max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                sgemm_vbatched_nn_log(max_m, max_n, max_k, batchCount, tag ? tag : "?");
            }
            break;
        case 1: // nt
            {
                gemm_template_vbatched_nt<float, version(NT,734), 0, 0>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 2: // nc
            {
                gemm_template_vbatched_nt<float, version(NT,734), 0, 1>
                (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
            }
            break;
        case 3: // tn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<float, version(TN,654), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<float, version(TN,666), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 6: // cn
            {
                if(max_k < 64)
                {
                    gemm_template_vbatched_tn<float, version(TN,654), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
                else
                {
                    gemm_template_vbatched_tn<float, version(TN,666), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 4: // tt
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 0, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 0, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 5: // tc
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 0, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 0, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 7: // ct
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 1, 0>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 1, 0>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        case 8: // cc
            {
                if(max_k < 128)
                {
                    if(max_m < 128)
                    {
                        gemm_template_vbatched_tt<float, version(TT,275), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                    else
                    {
                        gemm_template_vbatched_tt<float, version(TT,312), 1, 1>
                        (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    }
                }
                else
                {
                    gemm_template_vbatched_tt<float, version(TT,312), 1, 1>
                    (max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                }
            }
            break;
        default:; // propose something
    }
}
