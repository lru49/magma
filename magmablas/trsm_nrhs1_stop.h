// G-GEMV2: leaf size (rows) at which the nrhs = 1 trsm_vbatched recursion stops and hands the block to the blocked
// one-CTA-per-matrix kernel (trsm_nrhs1_blk.cuh). Default 256 (= TRSM_NRHS1_BLK_MAX). 32 restores the G-TRSV1/G-TRSV2 recursion
// exactly (32-row warp leaves + PREP/gemv updates), which gives an A/B fallback inside one library. Read once from the
// environment variable MAGMA_TRSM_NRHS1_STOP (integer 32..256); anything else = default.
#ifndef TRSM_NRHS1_STOP_H
#define TRSM_NRHS1_STOP_H
#include <stdlib.h>
static inline int trsm_nrhs1_stop()
{
    static const int stop = []() {
        const char *e = getenv("MAGMA_TRSM_NRHS1_STOP");
        int v = e ? atoi(e) : 256;
        return (v >= 32 && v <= 256) ? v : 256;
    }();
    return stop;
}
#endif
