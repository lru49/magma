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
//   MAGMA_SGEMM_VBATCHED_NN_VERSION=<tag>  forces ONE kernel version for every float vbatched NN call above MIN_MN
//                                          (read once; unset => CUTLASS (adopted default); DEFAULT or 0 => stock dispatch;
//                                          unknown tag => stock dispatch with a message).
//   MAGMA_SGEMM_VBATCHED_NN_MIN_MN=<n>     apply the forced/default version only to calls with min(max_m, max_n) >= n
//                                          (default 64: the tiny getrf panel calls keep the stock dispatch).
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

// F-SGEMM-CUTLASS: grouped SIMT FP32 GEMM from CUTLASS (magmablas/sgemm_vbatched_cutlass.cu), codes 20000+ (config = code - 20000)
//   CUTLASS = 128x64x8 threadblock (default), CUTLASS128 = 128x128x8, CUTLASS64 = 64x64x8 (all 2-stage, warp 32x64 / 32x32).
// If the CUTLASS path refuses a call (returns nonzero) the default dispatch below runs instead.
#define SGEMM_VB_NN_CUTLASS_BASE 20000
#define SGEMM_VB_NN_CUTLASS_VERSIONS(X) \
    X(CUTLASS, 20000) X(CUTLASS128, 20001) X(CUTLASS64, 20002)

int magmablas_sgemm_vbatched_nn_cutlass(
    int config,
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k,
    magma_int_t* m, magma_int_t* n, magma_int_t* k,
    float alpha,
    float const * const * dA_array, magma_int_t Ai, magma_int_t Aj, magma_int_t* ldda,
    float const * const * dB_array, magma_int_t Bi, magma_int_t Bj, magma_int_t* lddb,
    float beta,
    float               **dC_array, magma_int_t Ci, magma_int_t Cj, magma_int_t* lddc,
    magma_int_t batchCount, magma_queue_t queue );

// F-SGEMM-CUTLASS adopted (fp32-7*, window 118): the DEFAULT for float vbatched NN calls with min(max_m, max_n) >= 64 is now
// the CUTLASS grouped GEMM (tag CUTLASS = 128x64x8). MAGMA_SGEMM_VBATCHED_NN_VERSION=DEFAULT (or 0) restores the stock
// MAGMA dispatch for every call; the other tags force one version as before. MAGMA_SGEMM_VBATCHED_NN_MIN_MN defaults to 64.
#define SGEMM_VB_NN_DEFAULT_CODE   20000   /* CUTLASS */
#define SGEMM_VB_NN_DEFAULT_MIN_MN 64

static int sgemm_vbatched_nn_forced_version()
{
    // -1 = stock MAGMA dispatch, otherwise the numeric code of the version used above MIN_MN
    static const int forced = []() -> int {
        const char* e = getenv("MAGMA_SGEMM_VBATCHED_NN_VERSION");
        if (e == NULL || *e == '\0') return SGEMM_VB_NN_DEFAULT_CODE;
        if (strcmp(e, "DEFAULT") == 0 || strcmp(e, "0") == 0) return -1;
        #define SGEMM_VB_NN_MATCH(tag, code) if (strcmp(e, #tag) == 0) return code;
        SGEMM_VB_NN_FORCED_VERSIONS(SGEMM_VB_NN_MATCH)
        SGEMM_VB_NN_CUTLASS_VERSIONS(SGEMM_VB_NN_MATCH)
        #undef SGEMM_VB_NN_MATCH
        fprintf(stderr, "MAGMA_SGEMM_VBATCHED_NN_VERSION=%s unknown; using the stock MAGMA dispatch. Known tags: DEFAULT", e);
        #define SGEMM_VB_NN_LIST(tag, code) fprintf(stderr, " %s", #tag);
        SGEMM_VB_NN_FORCED_VERSIONS(SGEMM_VB_NN_LIST)
        SGEMM_VB_NN_CUTLASS_VERSIONS(SGEMM_VB_NN_LIST)
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
        return (e == NULL || *e == '\0') ? SGEMM_VB_NN_DEFAULT_MIN_MN : (magma_int_t) atoll(e);
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

// F-SGEMM-CUTLASS census: MAGMA_SGEMM_VBATCHED_NN_DUMP=<file> (needs LOG=1) appends, for every float vbatched NN call with
// min(max_m, max_n) >= 64, one "#CALL max_m max_n max_k batchCount version Ai Aj Bi Bj Ci Cj" header and one
// "m n k lda ldb ldc" line per problem (the raw device arrays, synchronously copied back: census runs only, never timed).
static FILE* sgemm_vbatched_nn_dump_file()
{
    static FILE* f = []() -> FILE* {
        const char* e = getenv("MAGMA_SGEMM_VBATCHED_NN_DUMP");
        return (e == NULL || *e == '\0') ? NULL : fopen(e, "a");
    }();
    return f;
}

static void sgemm_vbatched_nn_dump(
    magma_int_t max_m, magma_int_t max_n, magma_int_t max_k, magma_int_t batchCount, const char* tag,
    magma_int_t* m, magma_int_t* n, magma_int_t* k, magma_int_t* ldda, magma_int_t* lddb, magma_int_t* lddc,
    magma_int_t Ai, magma_int_t Aj, magma_int_t Bi, magma_int_t Bj, magma_int_t Ci, magma_int_t Cj, magma_queue_t queue)
{
    FILE* f = sgemm_vbatched_nn_dump_file();
    if (f == NULL || !sgemm_vbatched_nn_log_enabled() || min(max_m, max_n) < 64 || batchCount <= 0) return;
    magma_int_t* h = (magma_int_t*) malloc(6 * batchCount * sizeof(magma_int_t));
    if (h == NULL) return;
    magma_int_t* src[6] = { m, n, k, ldda, lddb, lddc };
    magma_queue_sync(queue);
    for (int j = 0; j < 6; j++)
        magma_getvector(batchCount, sizeof(magma_int_t), src[j], 1, h + j * batchCount, 1, queue);
    fprintf(f, "#CALL %lld %lld %lld %lld %s %lld %lld %lld %lld %lld %lld\n", (long long)max_m, (long long)max_n, (long long)max_k,
            (long long)batchCount, tag, (long long)Ai, (long long)Aj, (long long)Bi, (long long)Bj, (long long)Ci, (long long)Cj);
    for (magma_int_t i = 0; i < batchCount; i++)
        fprintf(f, "%lld %lld %lld %lld %lld %lld\n", (long long)h[i], (long long)h[batchCount + i], (long long)h[2 * batchCount + i],
                (long long)h[3 * batchCount + i], (long long)h[4 * batchCount + i], (long long)h[5 * batchCount + i]);
    fflush(f);
    free(h);
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
                const char* cutlass_tag = NULL;
                if (code >= SGEMM_VB_NN_CUTLASS_BASE)
                {
                    // F-SGEMM-CUTLASS: grouped GEMM; a nonzero return means "not run" -> default dispatch below
                    int rc = magmablas_sgemm_vbatched_nn_cutlass(code - SGEMM_VB_NN_CUTLASS_BASE,
                        max_m, max_n, max_k, m, n, k, alpha, dA_array, Ai, Aj, ldda, dB_array, Bi, Bj, lddb, beta, dC_array, Ci, Cj, lddc, batchCount, queue);
                    if (rc == 0) {
                        #define SGEMM_VB_NN_CUTLASS_NAME(tag, c) if (code == c) cutlass_tag = #tag;
                        SGEMM_VB_NN_CUTLASS_VERSIONS(SGEMM_VB_NN_CUTLASS_NAME)
                        #undef SGEMM_VB_NN_CUTLASS_NAME
                    }
                    code = -1;
                }
                if (cutlass_tag != NULL)
                {
                    sgemm_vbatched_nn_log(max_m, max_n, max_k, batchCount, cutlass_tag);
                    sgemm_vbatched_nn_dump(max_m, max_n, max_k, batchCount, cutlass_tag, m, n, k, ldda, lddb, lddc, Ai, Aj, Bi, Bj, Ci, Cj, queue);
                    break;
                }
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
                sgemm_vbatched_nn_dump(max_m, max_n, max_k, batchCount, tag ? tag : "?", m, n, k, ldda, lddb, lddc, Ai, Aj, Bi, Bj, Ci, Cj, queue);
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
