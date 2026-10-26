/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @author Tingxing Dong
       @author Azzam Haidar

*/
#ifndef GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
#define GEMV_TEMPLATE_KERNEL_VBATCHED_CUH

#include "gemm_template_device_defs.cuh" // use make_FloatingPoint
#include "gemv_template_device.cuh"

/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
gemvn_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array, magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy)
{
    int batchid = blockIdx.z;

    int my_m = (int)m[batchid];
    if( blockIdx.x >= magma_ceildiv(my_m, TILE_SIZE) ) return;

    gemvn_template_device<T, DIM_X, DIM_Y, TILE_SIZE>
        ( my_m, (int)n[batchid], alpha, A_array[batchid], (int)lda[batchid], x_array[batchid], (int)incx[batchid], beta, y_array[batchid], (int)incy[batchid]);
}


/******************************************************************************/
// G-GEMV1: split-K for under-filled gemvn grids.
// The launch above has ceil(max_m/TILE_SIZE) x batchCount CTAs; on long, narrow shapes that is far below the 82 SMs
// (window 22 ncu: 36 launches with < 82 CTAs = 72.7 % of the long-gemv time, 8-16 CTAs typical, DRAM 36.7 % of peak).
// Here the n columns are cut into S chunks, CTA (x, s, z) writes the partial sum of chunk s to a workspace, and a second
// kernel adds the S partials in FIXED order s = 0..S-1 and applies alpha/beta. No atomics, so the result is deterministic.
// Per-chunk accumulation order inside a CTA is the same as in gemvn_template_device.
#define GEMVN_SPLIT_TARGET_CTAS 246   // 3 resident 512-thread CTAs x 82 SMs (about 3 waves of the old CTA count at 82 SMs)
#define GEMVN_SPLIT_MIN_COLS     32   // never cut chunks below this many columns
#define GEMVN_SPLIT_FILLED      164   // old grids with >= 2 x 82 CTAs keep the old path

// private stream-ordered pool, keeps its memory (see G-TRSV1 pool, release threshold max); NULL on failure.
// One pool per device, created on first use; lock-free (compare-and-swap) because <mutex> cannot be included after magma_internal.h
// (its min/max macros break libstdc++ headers). static: the library must export exactly the released symbol set (symcheck).
static inline cudaMemPool_t gemvn_split_pool()
{
    static cudaMemPool_t volatile pools[64] = {};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 64) { cudaGetLastError(); return NULL; }
    if (pools[dev] == NULL) {
        cudaMemPoolProps props = {};
        props.allocType     = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id   = dev;
        cudaMemPool_t p = NULL;
        if (cudaMemPoolCreate(&p, &props) != cudaSuccess) { cudaGetLastError(); return NULL; }
        cuuint64_t keep = ~(cuuint64_t)0;
        cudaMemPoolSetAttribute(p, cudaMemPoolAttrReleaseThreshold, &keep);
        if (!__sync_bool_compare_and_swap((void* volatile*)&pools[dev], (void*)NULL, (void*)p)) cudaMemPoolDestroy(p);   // lost the race
    }
    return pools[dev];
}

// number of column chunks for a launch (1 = keep the old path); also used by the host test
static inline int gemvn_split_count(int max_m, int max_n, int batch, int tile, int dimxy)
{
    if (dimxy != 512 || tile != 512) return 1;           // only the two long variants <512,1,512>, <128,4,512>
    long ctas = (long)((max_m + tile - 1) / tile) * batch;
    if (ctas <= 0 || ctas >= GEMVN_SPLIT_FILLED) return 1;
    int s = (int)((GEMVN_SPLIT_TARGET_CTAS + ctas - 1) / ctas);
    int smax = max_n / GEMVN_SPLIT_MIN_COLS;
    if (s > smax) s = smax;
    return s < 2 ? 1 : s;
}

template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
__global__ void
gemvn_split_kernel_vbatched(
    magma_int_t* m, magma_int_t* n,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array, magma_int_t* incx,
    T* ws, int ldw, int S, int chunk)
{
    int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    if( my_m <= 0 || my_n <= 0 ) return;
    if( blockIdx.x >= magma_ceildiv(my_m, TILE_SIZE) ) return;
    int c0 = blockIdx.y * chunk;
    if( c0 >= my_n ) return;
    int c1 = min(c0 + chunk, my_n);

    const T* A = A_array[batchid];
    const T* x = x_array[batchid];
    int LDA  = (int)lda[batchid];
    int INCX = (int)incx[batchid];
    T* part  = ws + ((size_t)batchid * S + blockIdx.y) * ldw;

    int tx = threadIdx.x % DIM_X;     // same thread mapping as gemvn_template_device
    int ty = threadIdx.x / DIM_X;
    __shared__ T sdata[DIM_X * DIM_Y];

    int st  = blockIdx.x * TILE_SIZE;
    int ed  = min(st + TILE_SIZE, magma_roundup(my_m, DIM_X));
    int ind = st + tx;
    for (int i = 0; i < (ed - st) / DIM_X; i++)
    {
        T res = make_FloatingPoint(0.0, 0.0);
        if (ind < my_m)
        {
            const T* Ai = A + ind;
            for (int col = c0 + ty; col < c1; col += DIM_Y)
                res += Ai[(size_t)col * LDA] * x[col * INCX];
        }
        if (DIM_Y == 1)
        {
            if (ind < my_m) part[ind] = res;
        }
        else
        {
            sdata[ty + tx * DIM_Y] = res;
            __syncthreads();
            if (DIM_Y > 16) {
                magma_sum_reduce< DIM_Y >( ty, sdata + tx * DIM_Y);
            }
            else if (ty == 0 && ind < my_m) {
                for (int j = 1; j < DIM_Y; j++)
                    sdata[tx * DIM_Y] += sdata[j + tx * DIM_Y];
            }
            if (ty == 0 && ind < my_m) part[ind] = sdata[tx * DIM_Y];
            __syncthreads();
        }
        ind += DIM_X;
    }
}

// y = alpha * (partial_0 + partial_1 + ... ) + beta * y, partials added in fixed order
template<typename T>
__global__ void
gemvn_split_reduce_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha, T beta,
    T** y_array, magma_int_t* incy,
    T const * ws, int ldw, int S, int chunk)
{
    int batchid = blockIdx.z;
    int my_m = (int)m[batchid];
    int my_n = (int)n[batchid];
    if( my_m <= 0 || my_n <= 0 ) return;
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if( row >= my_m ) return;
    int used = (my_n + chunk - 1) / chunk;   // chunks that were written for this matrix
    T res = make_FloatingPoint(0.0, 0.0);
    for (int s = 0; s < used; s++)
        res += ws[((size_t)batchid * S + s) * ldw + row];
    T* y = y_array[batchid];
    int INCY = (int)incy[batchid];
    y[row * INCY] = alpha*res + beta*y[row * INCY];
}

/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
static void gemvn_template_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dx_array, magma_int_t* incx,
    T beta, T** dy_array, magma_int_t* incy,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y);

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);

        // G-GEMV1: under-filled grid on a long variant -> split the columns over S chunks (fallback: old path)
        int S = gemvn_split_count((int)max_m, (int)max_n, (int)ibatch, TILE_SIZE, DIM_X*DIM_Y);
        if (S > 1) {
            int chunk = (int)((max_n + S - 1) / S);
            S = (int)((max_n + chunk - 1) / chunk);
            int ldw = (int)max_m;
            T* ws = NULL;
            cudaStream_t st = queue->cuda_stream();
            cudaMemPool_t pool = gemvn_split_pool();
            size_t bytes = (size_t)ibatch * S * ldw * sizeof(T);
            if (pool == NULL || cudaMallocFromPoolAsync((void**)&ws, bytes, pool, st) != cudaSuccess) { ws = NULL; cudaGetLastError(); }
            if (ws != NULL) {
                dim3 gridS( magma_ceildiv(max_m, TILE_SIZE), S, ibatch );
                gemvn_split_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE>
                <<< gridS, threads, 0, st >>>
                ( m+i, n+i, dA_array+i, ldda+i, dx_array+i, incx+i, ws, ldw, S, chunk );
                dim3 gridR( magma_ceildiv(max_m, 256), 1, ibatch );
                gemvn_split_reduce_kernel_vbatched<T>
                <<< gridR, 256, 0, st >>>
                ( m+i, n+i, alpha, beta, dy_array+i, incy+i, ws, ldw, S, chunk );
                cudaFreeAsync(ws, st);
                continue;
            }
        }

        dim3 grid( magma_ceildiv(max_m, TILE_SIZE), 1, ibatch );

        gemvn_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE>
        <<< grid, threads, 0, queue->cuda_stream() >>>
        ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
    }
}


/******************************************************************************/
template<typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE, magma_trans_t trans>
__global__ void
gemvc_kernel_vbatched(
    magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * A_array, magma_int_t* lda,
    T const * const * x_array,  magma_int_t* incx,
    T beta, T**  y_array, magma_int_t* incy)
{
    int batchid = blockIdx.z;

    int my_n = (int)n[batchid];
    if( blockIdx.x >= magma_ceildiv(my_n, TILE_SIZE) ) return;

    gemvc_template_device<T, DIM_X, DIM_Y, TILE_SIZE, trans>
        ( (int)m[batchid], (int)n[batchid], alpha, A_array[batchid], (int)lda[batchid], x_array[batchid], (int)incx[batchid], beta, y_array[batchid], (int)incy[batchid]);
}


/******************************************************************************/
template <typename T, const int DIM_X, const int DIM_Y, const int TILE_SIZE>
void gemvc_template_vbatched(
    magma_trans_t trans, magma_int_t* m, magma_int_t* n, T alpha,
    T const * const * dA_array, magma_int_t* ldda,
    T const * const * dx_array, magma_int_t* incx,
    T beta, T** dy_array, magma_int_t* incy,
    magma_int_t max_m, magma_int_t max_n,
    magma_int_t batchCount, magma_queue_t queue)
{
    magma_int_t max_batchCount = queue->get_maxBatch();
    dim3 threads ( DIM_X, DIM_Y );

    for(magma_int_t i=0; i<batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3 grid    ( magma_ceildiv(max_n, TILE_SIZE), 1, ibatch );

        if (trans == MagmaConjTrans) {
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaConjTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
        }
        else if (trans == MagmaTrans) {
            gemvc_kernel_vbatched<T, DIM_X, DIM_Y, TILE_SIZE, MagmaTrans>
            <<< grid, threads, 0, queue->cuda_stream() >>>
            ( m+i, n+i, alpha, dA_array+i, ldda+i, dx_array+i, incx+i, beta, dy_array+i, incy+i );
        }
    }
}

#endif  // GEMV_TEMPLATE_KERNEL_VBATCHED_CUH
