/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> s d c

       @author Azzam Haidar
       @author Tingxing Dong
*/
#include "magma_internal.h"
#include "batched_kernel_param.h"
#include "zlaswp_device.cuh"
#include "laswp_rowperm.cuh"   // magma-hpc opt-g7

/******************************************************************************/
// parallel swap the swaped dA(1:nb,i:n) is stored in dout
__global__
void zlaswp_rowparallel_kernel(
                                int n, int width, int height,
                                magmaDoubleComplex *dinput, int ldi,
                                magmaDoubleComplex *doutput, int ldo,
                                magma_int_t*  pivinfo)
{
    zlaswp_rowparallel_devfunc(n, width, height, dinput, ldi, doutput, ldo, pivinfo);
}


/******************************************************************************/
__global__
void zlaswp_rowparallel_kernel_batched(
                                int n, int width, int height,
                                magmaDoubleComplex **input_array, int input_i, int input_j, int ldi,
                                magmaDoubleComplex **output_array, int output_i, int output_j, int ldo,
                                magma_int_t** pivinfo_array)
{
    int batchid = blockIdx.z;
    zlaswp_rowparallel_devfunc( n, width, height,
                                input_array[batchid]  + input_j  * ldi +  input_i, ldi,
                                output_array[batchid] + output_j * ldo + output_i, ldo,
                                pivinfo_array[batchid]);
}


/******************************************************************************/
extern "C" void
magma_zlaswp_rowparallel_batched( magma_int_t n,
                       magmaDoubleComplex**  input_array, magma_int_t  input_i, magma_int_t  input_j, magma_int_t ldi,
                       magmaDoubleComplex** output_array, magma_int_t output_i, magma_int_t output_j, magma_int_t ldo,
                       magma_int_t k1, magma_int_t k2,
                       magma_int_t **pivinfo_array,
                       magma_int_t batchCount, magma_queue_t queue)
{
#define  input_array(i,j)  input_array, i, j
#define output_array(i,j) output_array, i, j

    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > 1024)
    {
        fprintf( stderr, "%s: n=%lld > 1024, not supported\n", __func__, (long long) n );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        if ( n < SWP_WIDTH) {
            size_t shmem = sizeof(magmaDoubleComplex) * height * n;
            zlaswp_rowparallel_kernel_batched
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, n, height, input_array+i, input_i, input_j, ldi, output_array+i, output_i, output_j, ldo, pivinfo_array+i );
        }
        else {
            size_t shmem = sizeof(magmaDoubleComplex) * height * SWP_WIDTH;
            zlaswp_rowparallel_kernel_batched
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, SWP_WIDTH, height, input_array+i, input_i, input_j, ldi, output_array+i, output_i, output_j, ldo, pivinfo_array+i );
        }
    }
#undef  input_array
#undef output_attay
}


/******************************************************************************/
extern "C" void
magma_zlaswp_rowparallel_native(
    magma_int_t n,
    magmaDoubleComplex* input, magma_int_t ldi,
    magmaDoubleComplex* output, magma_int_t ldo,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *pivinfo,
    magma_queue_t queue)
{
    if (n == 0 ) return;
    int height = k2-k1;
    if ( height  > MAX_NTHREADS)
    {
        fprintf( stderr, "%s: height=%lld > %lld, magma_zlaswp_rowparallel_q not supported\n",
                 __func__, (long long) n, (long long) MAX_NTHREADS );
    }

    int blocks = magma_ceildiv( n, SWP_WIDTH );
    dim3  grid(blocks, 1, 1);

    if ( n < SWP_WIDTH)
    {
        size_t shmem = sizeof(magmaDoubleComplex) * height * n;
        zlaswp_rowparallel_kernel
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, n, height, input, ldi, output, ldo, pivinfo );
    }
    else
    {
        size_t shmem = sizeof(magmaDoubleComplex) * height * SWP_WIDTH;
        zlaswp_rowparallel_kernel
            <<< grid, height, shmem, queue->cuda_stream() >>>
            ( n, SWP_WIDTH, height, input, ldi, output, ldo, pivinfo );
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
__global__ void zlaswp_rowserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaDoubleComplex* dA = dA_array[blockIdx.z];
    magma_int_t *dipiv = ipiv_array[blockIdx.z];

    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

    k1--;
    k2--;

    if (tid < n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}


/******************************************************************************/
// serial swap that does swapping one row by one row
__global__ void zlaswp_rowserial_kernel_native( int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;

    //k1--;
    //k2--;

    if (tid < n) {
        magmaDoubleComplex A1;

        for (int i1 = k1; i1 < k2; i1++)
        {
            int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
            if ( i2 != i1)
            {
                A1 = dA[i1 + tid * lda];
                dA[i1 + tid * lda] = dA[i2 + tid * lda];
                dA[i2 + tid * lda] = A1;
            }
        }
    }
}



/******************************************************************************/
// magma-hpc opt-g7: the rowserial swaps as one permutation gather (see laswp_rowperm.cuh).
// grid (nblk_x, 1, batch): block x handles columns [x*cols_per_block, min(n, (x+1)*cols_per_block)) of matrix z.
// Same result as zlaswp_rowserial_kernel_batched, bit for bit: values are only copied, and every row ends up holding
// the row the sequential swaps would have put there.
static __global__ void zlaswp_rowperm_kernel_batched(   // static: no new exported symbol (SYMCHECK)
        int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2,
        int R, int C, int cols_per_block, magma_int_t** ipiv_array )
{
    extern __shared__ __align__(16) unsigned char zlaswp_rp_smem[];
    magmaDoubleComplex* stile = (magmaDoubleComplex*)zlaswp_rp_smem;
    int* sperm = (int*)(stile + (size_t)R * C);
    int* sipiv = sperm + R;

    const int tid = threadIdx.x;
    const int nth = blockDim.x;
    magmaDoubleComplex* dA = dA_array[blockIdx.z];
    const magma_int_t* dipiv = ipiv_array[blockIdx.z];
    const int c_begin = blockIdx.x * cols_per_block;
    const int c_end   = min(n, c_begin + cols_per_block);
    if (c_begin >= c_end) return;                       // uniform over the block
    const int i_begin = k1 - 1;                         // released loop: i1 = k1-1 .. k2-2
    const int nsw     = (k2 - 1) - (k1 - 1);

    const int bad = magma_laswp_rp::stage(tid, nth, dipiv, i_begin, nsw, R, sperm, sipiv);
    if (__syncthreads_or(bad)) {
        magma_laswp_rp::serial_fallback(tid, nth, dA, lda, k1 - 1, k2 - 1, dipiv, c_begin, c_end);
        return;
    }
    if (tid == 0) magma_laswp_rp::compose(i_begin, nsw, sperm, sipiv);
    __syncthreads();
    if (! __syncthreads_or(magma_laswp_rp::any_moved(tid, nth, R, sperm))) return;

    for (int c0 = c_begin; c0 < c_end; c0 += C) {
        const int cw = min(C, c_end - c0);
        magma_laswp_rp::load_tile(tid, nth, (const magmaDoubleComplex*)dA, (size_t)lda, c0, cw, R, sperm, stile);
        __syncthreads();
        magma_laswp_rp::store_tile(tid, nth, dA, (size_t)lda, c0, cw, R, sperm, stile);
        __syncthreads();
    }
}

/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_rowserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0) return;

    // magma-hpc opt-g7: empty swap range (the released kernel's loop would not run) -> nothing to do
    if (n > 0 && k1 >= 1 && k2 - 1 <= k1 - 1) return;

    magma_int_t threads = min(n, MAX_NTHREADS);
    magma_int_t blocks  = magma_ceildiv( n, threads );
    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);

        // magma-hpc opt-g7: permutation gather when the window fits (always for getrs: k1 = 1, k2 = n <= 2048)
        magma_laswp_rp::Config cf;
        if (magma_laswp_rp::config( n, k1, k2, ibatch, (int)sizeof(magmaDoubleComplex), cf )) {
            dim3 pgrid(cf.nblk_x, 1, ibatch);
            zlaswp_rowperm_kernel_batched
            <<< pgrid, cf.nthreads, cf.smem, queue->cuda_stream() >>>
            (n, dA_array+i, lda, k1, k2, cf.R, cf.C, cf.cols_per_block, ipiv_array+i);
            continue;
        }

        dim3  grid(blocks, 1, ibatch);

        zlaswp_rowserial_kernel_batched
        <<< grid, threads, 0, queue->cuda_stream() >>>
        (n, dA_array+i, lda, k1, k2, ipiv_array+i);
    }
}



/******************************************************************************/
// serial swap that does swapping one row by one row, similar to LAPACK
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_rowserial_native(magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t* dipiv, magma_queue_t queue)
{
    if (n == 0) return;

    magma_int_t threads = min(n, MAX_NTHREADS);
    magma_int_t blocks  = magma_ceildiv( n, threads );
    dim3  grid(blocks, 1, 1);

    zlaswp_rowserial_kernel_native
    <<< grid, threads, 0, queue->cuda_stream() >>>
    (n, dA, lda, k1, k2, dipiv);
}



/******************************************************************************/
// serial swap that does swapping one column by one column
__device__ void zlaswp_columnserial_devfunc(int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    unsigned int tid = threadIdx.x + blockDim.x*blockIdx.x;
    k1--;
    k2--;
    if ( k1 < 0 || k2 < 0 ) return;


    if ( tid < n) {
        magmaDoubleComplex A1;
        if (k1 <= k2)
        {
            for (int i1 = k1; i1 <= k2; i1++)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        } else
        {

            for (int i1 = k1; i1 >= k2; i1--)
            {
                int i2 = dipiv[i1] - 1;  // Fortran index, switch i1 and i2
                if ( i2 != i1)
                {
                    A1 = dA[i1 * lda + tid];
                    dA[i1 * lda + tid] = dA[i2 * lda + tid];
                    dA[i2 * lda + tid] = A1;
                }
            }
        }
    }
}


__global__ void zlaswp_columnserial_kernel_batched( int n, magmaDoubleComplex **dA_array, int lda, int k1, int k2, magma_int_t** ipiv_array )
{
    magmaDoubleComplex* dA = dA_array[blockIdx.z];
    magma_int_t *dipiv = ipiv_array[blockIdx.z];

    zlaswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv);
}

__global__ void zlaswp_columnserial_kernel( int n, magmaDoubleComplex_ptr dA, int lda, int k1, int k2, magma_int_t* dipiv )
{
    zlaswp_columnserial_devfunc(n, dA, lda, k1, k2, dipiv);
}

/******************************************************************************/
// serial swap that does swapping one column by one column
// K1, K2 are in Fortran indexing
extern "C" void
magma_zlaswp_columnserial(
    magma_int_t n, magmaDoubleComplex_ptr dA, magma_int_t lda,
    magma_int_t k1, magma_int_t k2,
    magma_int_t *dipiv, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, ZLASWP_COL_NTH );
    dim3  grid(blocks, 1, 1);

    zlaswp_columnserial_kernel<<< grid, ZLASWP_COL_NTH, 0, queue->cuda_stream() >>>
    (n, dA, lda, k1, k2, dipiv);
}

extern "C" void
magma_zlaswp_columnserial_batched(magma_int_t n, magmaDoubleComplex** dA_array, magma_int_t lda,
                   magma_int_t k1, magma_int_t k2,
                   magma_int_t **ipiv_array,
                   magma_int_t batchCount, magma_queue_t queue)
{
    if (n == 0 ) return;

    int blocks = magma_ceildiv( n, ZLASWP_COL_NTH );

    magma_int_t max_batchCount = queue->get_maxBatch();

    for(magma_int_t i = 0; i < batchCount; i+=max_batchCount) {
        magma_int_t ibatch = min(max_batchCount, batchCount-i);
        dim3  grid(blocks, 1, ibatch);

        magma_int_t min_ZLASWP_COL_NTH__n = min(ZLASWP_COL_NTH, n);
        zlaswp_columnserial_kernel_batched
        <<< grid, min_ZLASWP_COL_NTH__n, 0, queue->cuda_stream() >>>
        (n, dA_array+i, lda, k1, k2, ipiv_array+i);
    }
}
