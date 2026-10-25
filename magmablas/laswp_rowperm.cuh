/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       magma-hpc opt-g7: precision-independent building blocks of the batched row-interchange kernel
       zlaswp_rowperm_kernel_batched (magmablas/zlaswp_batched.cu), used by magma_zlaswp_rowserial_batched.

       The released rowserial kernel applies the swaps i1 = k1-1 .. k2-2 (0-based; i2 = ipiv[i1]-1) one after the
       other, one thread per column, so each swap touches two rows of a column-major matrix (two 8/16-byte elements
       ld apart per thread: uncoalesced) and the next swap waits for the previous one. Row swaps only move data, so
       the same result is obtained by
         1. composing the swap sequence into a permutation perm[] (B_new(r,:) = B_old(perm[r],:)) once per block,
            serially in shared memory (this is the only sequential part, ~n shared-memory steps), then
         2. gathering: columns are staged through shared memory with coalesced column reads and writes; only rows
            with perm[r] != r are read and written (they are exactly the rows whose content changes).
       The values written are copies of the values the released kernel would have moved there (no arithmetic), so the
       output is bitwise identical. Anything outside the fast path's preconditions (an ipiv entry outside 1..R) runs
       the released serial loop for that block's columns (serial_fallback, same statements as the released kernel).

       Everything here is __host__ __device__ so that anatomy/dlaswp_rowserial_batched/opt_g7/cpu_test.cpp can run the
       very same code on the CPU against a sequential-swap model. The kernel itself (barriers, launch) is in
       zlaswp_batched.cu.
*/
#ifndef MAGMA_LASWP_ROWPERM_CUH
#define MAGMA_LASWP_ROWPERM_CUH

#include <cstddef>

#ifdef __CUDACC__
#define LASWP_RP_HD __host__ __device__ __forceinline__
#else
#define LASWP_RP_HD inline
#endif

#define LASWP_RP_NTH          256          // max threads per block
#define LASWP_RP_SMEM_TARGET  (24 * 1024)  // bytes of tile + perm + ipiv per block: 4 blocks/SM on sm_86 (100 KB smem)
#define LASWP_RP_SMEM_MAX     (48 * 1024)  // never above the default dynamic smem limit (no opt-in attribute needed)
#define LASWP_RP_RMAX         2048         // rows of the permutation window (R = k2); larger -> released kernel
#define LASWP_RP_TARGET_BLOCKS 3936        // 82 SMs x 4 blocks x 12 waves: split columns over blocks below this
#define LASWP_RP_UNROLL       8            // loads in flight per thread in the staging loop

namespace magma_laswp_rp {

struct Config {
    int R;               // rows of the permutation window: rows 0..R-1 (R = k2)
    int C;               // columns per shared-memory tile
    int cols_per_block;  // columns handled by one block (multiple of C)
    int nblk_x;          // blocks per matrix (grid.x)
    int nthreads;        // threads per block
    size_t smem;         // dynamic shared memory bytes: R*C*elem (tile) + R*4 (perm) + R*4 (ipiv)
};

static inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Host-side launch configuration. Returns false when the fast path does not apply (the caller then launches the
// released kernel, or returns for the empty swap range). n = columns, k1/k2 = this routine's Fortran bounds.
static inline bool config(long long n, long long k1, long long k2, long long batch, int elem_bytes, Config& cf)
{
    if (n <= 0 || batch <= 0) return false;
    if (k1 < 1) return false;                       // released kernel would index ipiv[-1]: keep its behaviour
    if (k2 - 1 <= k1 - 1) return false;             // no swap in the released loop (caller: no-op)
    if (k2 > LASWP_RP_RMAX) return false;
    if (n > 0x7fffffffLL) return false;
    const int R = (int)k2;
    int C = (LASWP_RP_SMEM_TARGET - 8 * R) / (elem_bytes * R);
    if (C < 1) C = 1;
    if (C > n) C = (int)n;
    const size_t smem = (size_t)R * C * elem_bytes + (size_t)8 * R;
    if (smem > LASWP_RP_SMEM_MAX) return false;
    const int ntiles = ceil_div((int)n, C);
    long long q = (LASWP_RP_TARGET_BLOCKS + batch - 1) / batch;
    if (q < 1) q = 1;
    if (q > ntiles) q = ntiles;
    const int tiles_per_block = ceil_div(ntiles, (int)q);
    cf.R = R;
    cf.C = C;
    cf.cols_per_block = tiles_per_block * C;
    cf.nblk_x = ceil_div((int)n, cf.cols_per_block);
    int nt = ((R * C + 31) / 32) * 32;
    if (nt > LASWP_RP_NTH) nt = LASWP_RP_NTH;
    if (nt < 32) nt = 32;
    cf.nthreads = nt;
    cf.smem = smem;
    return true;
}

// Phase 1 (all threads): identity permutation, ipiv window staged 0-based into sipiv[0..nsw), validity flag.
// i_begin = k1-1, nsw = (k2-1) - (k1-1) swaps. Returns nonzero if this thread saw an entry outside 1..R.
template <class I>
LASWP_RP_HD int stage(int tid, int nth, const I* dipiv, int i_begin, int nsw, int R, int* sperm, int* sipiv)
{
    for (int r = tid; r < R; r += nth) sperm[r] = r;
    int bad = 0;
    for (int i = tid; i < nsw; i += nth) {
        const I p1 = dipiv[i_begin + i];
        const int ok = (p1 >= 1) && (p1 <= (I)R);
        bad |= !ok;
        sipiv[i] = ok ? (int)(p1 - 1) : 0;
    }
    return bad;
}

// Phase 2 (one thread): apply the swap sequence, in the released order, to the row-index vector.
// Swapping rows a and b of B (B(r,:) = B0(perm[r],:)) swaps perm[a] and perm[b].
LASWP_RP_HD void compose(int i_begin, int nsw, int* sperm, const int* sipiv)
{
    for (int i = 0; i < nsw; i++) {
        const int a = i_begin + i;
        const int b = sipiv[i];
        if (b != a) {
            const int t = sperm[a];
            sperm[a] = sperm[b];
            sperm[b] = t;
        }
    }
}

// Phase 3 (all threads): does any row move? (block-wide OR in the kernel)
LASWP_RP_HD int any_moved(int tid, int nth, int R, const int* sperm)
{
    int m = 0;
    for (int r = tid; r < R; r += nth) m |= (sperm[r] != r);
    return m;
}

// advance a flat tile index by nth: (r, c) with r in [0, R)
LASWP_RP_HD void step(int& r, int& c, int nth, int R)
{
    r += nth;
    while (r >= R) { r -= R; c++; }
}

// Phase 4 (all threads): stage columns c0 .. c0+cw-1, rows with perm[r] != r, into stile[c*R + r].
// Flat index e = c*R + r: consecutive threads read consecutive rows of one column (coalesced).
template <class T>
LASWP_RP_HD void load_tile(int tid, int nth, const T* dA, size_t lda, int c0, int cw, int R, const int* sperm, T* stile)
{
    const int cnt = R * cw;
    int r = tid, c = 0;
    while (r >= R) { r -= R; c++; }
    for (int e0 = tid; e0 < cnt; e0 += LASWP_RP_UNROLL * nth) {
        T   v[LASWP_RP_UNROLL];
        int rr[LASWP_RP_UNROLL], cc[LASWP_RP_UNROLL];
        bool ok[LASWP_RP_UNROLL];
        #pragma unroll
        for (int u = 0; u < LASWP_RP_UNROLL; u++) {
            ok[u] = (e0 + u * nth < cnt) && (sperm[r] != r);
            rr[u] = r;
            cc[u] = c;
            if (ok[u]) v[u] = dA[(size_t)r + (size_t)(c0 + c) * lda];
            step(r, c, nth, R);
        }
        #pragma unroll
        for (int u = 0; u < LASWP_RP_UNROLL; u++) {
            if (ok[u]) stile[cc[u] * R + rr[u]] = v[u];
        }
    }
}

// Phase 5 (all threads, after a barrier): B(r, c0+c) = staged B(perm[r], c0+c) for every moved row (coalesced writes).
template <class T>
LASWP_RP_HD void store_tile(int tid, int nth, T* dA, size_t lda, int c0, int cw, int R, const int* sperm, const T* stile)
{
    const int cnt = R * cw;
    int r = tid, c = 0;
    while (r >= R) { r -= R; c++; }
    #pragma unroll 4
    for (int e = tid; e < cnt; e += nth) {
        const int p = sperm[r];
        if (p != r) dA[(size_t)r + (size_t)(c0 + c) * lda] = stile[c * R + p];
        step(r, c, nth, R);
    }
}

// Fallback (all threads): the released rowserial loop, same statements and int index arithmetic, over this block's
// columns [c_begin, c_end) (columns are independent, so splitting them over blocks/threads does not change results).
template <class T, class I>
LASWP_RP_HD void serial_fallback(int tid, int nth, T* dA, int lda, int k1m, int k2m, const I* dipiv, int c_begin, int c_end)
{
    for (int col = c_begin + tid; col < c_end; col += nth) {
        T A1;
        for (int i1 = k1m; i1 < k2m; i1++) {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if (i2 != i1) {
                A1 = dA[i1 + col * lda];
                dA[i1 + col * lda] = dA[i2 + col * lda];
                dA[i2 + col * lda] = A1;
            }
        }
    }
}

}  // namespace magma_laswp_rp

#endif  // MAGMA_LASWP_ROWPERM_CUH
