# Prompt: CSR SpMV numpy
### Begining of output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

# CSR SPMV numpy implementation

=======

# CSR SPMV numpy wrapper

### End of output
# Prompt: CSR SpMV numpy
def
### Begining of output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

def CSR_SPMV(A, x):
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i+1]):
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def csr_spmv_numpy(data, indices, indptr, x):
    y = np.zeros(indptr.shape[0] - 1)
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i+1]):
            y[i] += data[j] * x[indices[j]]
    return y

=======

def SPMV_numpy(A, x, y):
    for i in range(A.shape[0]):
        for j in A.indptr[i]: A.indices[j]:
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def spmv_csr_numpy(A, x):
    return A.dot(x)

=======

def spmv_csr(A, x):
    return A.dot(x)

=======

def spmv_csr_numpy(A, x):
    return A * x

=======

def csr_spmv_np(A, x):
    return A.dot(x)

=======

def csr_spmv_numpy(A, x):
    return A * x
### End of Output
# Prompt CSR SPMV CuPy
### Beginning of Output
Synthesizing 7/10 solutions (Duplicates hidden)

=======

# CSR SPMV CuPy Kernel

=======

# CSR SPMV CuPy kernel

=======

# CSR SPMV CuPy implementation

=======

# CSR SPMV CuPy
# Created: 2019-12-16 13:44:29

=======

# CSR SPMV CuPy
# Created: 2020-08-28 21:51:17.392311
### End of Output
# CSR SPMV CuPy
def
Synthesizing 10/10 solutions

=======

def spmv_csr_cupy(A, x):
    y = cupy.zeros(A.shape[0], dtype=x.dtype)
    cupyx.scipy.sparse.csr_matrix(A).dot(x, out=y)
    return y

=======

def spmv_cupy_csr(csr_data, csr_indices, csr_indptr, x, y):
    n = len(csr_indptr) - 1
    for i in range(n):
        for j in range(csr_indptr[i], csr_indptr[i + 1]):
            y[i] += csr_data[j] * x[csr_indices[j]]

=======

def spmv_csr_cupy(A, x):
    return cupy.sparse.csr_matrix.dot(A, x)

=======

def SPMV_cupy_csr(A, x):
    return cupy.sparse.csr_matrix.dot(A, x)

=======

def spmv_cupy_csr(x, row, col, data):
    """SPMV for CSR matrix"""
    n = row.shape[0] - 1
    y = cupy.zeros(n, dtype=cupy.float32)
    for i in range(n):
        for j in range(row[i], row[i + 1]):
            y[i] += data[j] * x[col[j]]
    return y

=======

def spmv_csr_cupy(alpha, csr, x, beta, y):
    return alpha * csr @ x + beta * y

=======

def spmv_cupy_csr(A, x):
    return (A * x).sum(axis=1)

=======

def spmv_csr_cupy():
    print("CSR SPMV CuPy")
    # Read matrix
    A = cupyx.scipy.sparse.load_npz("SPMV/CSR.npz")
    # Read vector
    x = cupy.load("SPMV/x.npy")
    # Perform SPMV
    y = A.dot(x)
    # Save vector
    cupy.save("SPMV/y.npy", y)
    print("CSR SPMV CuPy done")

=======

def SPMV_cupy(A, x):
    """SPMV for CuPy CSR matrix"""
    return cupy.sparse.csr_matrix.dot(A, x)

=======

def spmv_cupy_csr(a, x):
    return a.dot(x)
### End of Output
# Prompt: CSR SPMV pyCUDA
### Beginning of Output
Synthesizing 8/10 solutions (Duplicates hidden)

=======

# CSR SPMV pyCUDA implementation

import pycuda.driver as drv
import pycuda.autoinit
from pycuda.compiler import SourceModule
import numpy as np

mod = SourceModule("""
    __global__ void spmv(int n, float *val, int *row_ptr, int *col_ind, float *x, float *y)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n)
        {
            float tmp = 0;
            for (int j = row_ptr[i]; j < row_ptr[i+1]; j++)
            {
                tmp += val[j] * x[col_ind[j]];
            }
            y[i] = tmp;
        }
    }
""")

=======

# CSR SPMV pyCUDA kernel

=======

# CSR SPMV pyCUDA implementation

=======

# CSR SPMV pyCUDA code
### End of Output
# Prompt: CSR SPMV pyCUDA
def
### Beginning of Output
Synthesizing 9/10 solutions

=======

def spmv_csr(csrVal, csrRowPtr, csrColInd, x, y):
    # Copy data to GPU
    csrVal = csrVal.astype(numpy.float32)
    csrRowPtr = csrRowPtr.astype(numpy.int32)
    csrColInd = csrColInd.astype(numpy.int32)
    x = x.astype(numpy.float32)
    y = y.astype(numpy.float32)
    csrVal_gpu = gpuarray.to_gpu(csrVal)
    csrRowPtr_gpu = gpuarray.to_gpu(csrRowPtr)
    csrColInd_gpu = gpuarray.to_gpu(csrColInd)
    x_gpu = gpuarray.to_gpu(x)
    y_gpu = gpuarray.to_gpu(y)

    # Launch the kernel
    block = (256, 1, 1)
    grid = (int(math.ceil(float(csrVal.shape[0]) / block[0])), 1)
    spmv_csr_kernel(grid, block, (csrVal_gpu, csrRowPtr_gpu, csrColInd_gpu, x_gpu, y_gpu, numpy.int32(csrVal.shape[0])))

    # Copy result back to CPU
    y = y_gpu.get()

    return y

=======

def spmv_csr_gpu(csrVal, csrColInd, csrRowPtr, x):
    # Initialize variables
    nnz = csrVal.shape[0]
    n = csrRowPtr.shape[0] - 1
    y = np.zeros(n, dtype=np.float32)
    # CUDA kernel
    mod = SourceModule("""
    __global__ void spmv_csr_kernel(float *y, float *csrVal, int *csrColInd, int *csrRowPtr, float *x)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        for (int j = csrRowPtr[i]; j < csrRowPtr[i+1]; j++)
        {
            y[i] += csrVal[j] * x[csrColInd[j]];
        }
    }
    """)
    # Get kernel function
    spmv_csr_kernel = mod.get_function("spmv_csr_kernel")
    # Call kernel function
    spmv_csr_kernel(
        drv.Out(y), drv.In(csrVal), drv.In(csrColInd), drv.In(csrRowPtr), drv.In(x),
        block=(256, 1, 1), grid=(int(math.ceil(n/256.0)), 1))
    # Return
    return y

=======

def csr_spmv_cusparse(csr_val, csr_row_ptr, csr_col_ind, x):
    # Create a handle for CUSPARSE
    handle = cusparseCreate()
    # Create a dense vector for the result
    y = gpuarray.empty(x.shape, x.dtype)
    # Create a descriptor of the sparse matrix A
    descr = cusparseCreateMatDescr()
    # Configure the matrix descriptor
    cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL)
    cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO)
    # Carry out the sparse matrix-vector multiplication
    cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, x.shape[0], x.shape[0], csr_val.size, 
                   float(1.0), descr, csr_val.gpudata, csr_row_ptr.gpudata, csr_col_ind.gpudata, 
                   x.gpudata, float(0.0), y.gpudata)
    # Destroy the matrix descriptor
    cusparseDestroyMatDescr(descr)
    # Destroy the handle
    cusparseDestroy(handle)
    return y

=======

def CSR_SPMV(csrVal, csrRowPtr, csrColInd, vec):
    # CSR SPMV
    # csrVal - CSR value array
    # csrRowPtr - CSR row pointer array
    # csrColInd - CSR column index array
    # vec - input vector
    # return - output vector
    # initialize variables
    vecLen = vec.shape[0]
    outVec = np.zeros(vecLen, dtype=np.float32)
    # calculate
    for i in range(vecLen):
        for j in range(csrRowPtr[i], csrRowPtr[i+1]):
            outVec[i] += csrVal[j] * vec[csrColInd[j]]
    return outVec

=======

def csr_spmv(csr, x, y):
    # create the kernel
    kernel = """
    __global__ void csr_spmv(int *csrRowPtr, int *csrColInd, double *csrVal, double *x, double *y)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        double sum = 0;
        for(int i = csrRowPtr[row]; i < csrRowPtr[row + 1]; i++)
            sum += csrVal[i] * x[csrColInd[i]];
        y[row] = sum;
    }
    """
    # compile the kernel code
    mod = SourceModule(kernel)
    # get the kernel function from the compiled module
    csr_spmv = mod.get_function("csr_spmv")
    # call the kernel on the card
    csr_spmv(csr.rowPtr, csr.colInd, csr.val, x, y, block=(256, 1, 1), grid=(int(math.ceil(float(csr.shape[0]) / 256)), 1, 1))
    return y

=======

def SPMV(A, x, y):
    # A is a CSR matrix
    # x is a dense vector
    # y is a dense vector

    # Get the row and column indices
    row = A.row
    col = A.col
    data = A.data

    # Get the number of rows
    n = A.shape[0]

    # Get the number of nonzero elements
    nnz = A.nnz

    # Get the number of columns
    m = A.shape[1]

    # Get the number of nonzeros in each row
    nnz_row = A.indptr[1:] - A.indptr[:-1]

    # Get the start index of each row in the data, row, and col arrays
    start = A.indptr[:-1]

    # Create the kernel
    kernel = """
    __global__ void spmv(int n, int m, int nnz, int* row, int* col, double* data, double* x, double* y)
    {
        int tid = threadIdx.x + blockDim.x * blockIdx.x;
        if(tid < n)
        {
            // Get the number of nonzeros in this row
            int nnz_in_row = %(nnz_row)s[tid];

            // Get the start index of this row
            int start_index = %(start)s[tid];

            // Initialize the value of y to zero
            double temp = 0;

            // Loop over the nonzeros in this row
            for(int i = 0; i < nnz_in_row; i++)
            {
                // Get the column index of the current nonzero
                int index = col[start_index + i];

                // Get the value of the current nonzero
                double value = data[start_index + i];

                // Add the product of the value and the x vector to the temp variable
                temp += value * x[index];
            }

            // Set the value of y
            y[tid] = temp;
        }
    }
    """ % {"nnz_row": nnz_row, "start": start}

    # Compile the kernel
    mod = SourceModule(kernel)

    # Get the function from the compiled module
    spmv = mod.get_function("spmv")

    # Create the block and grid dimensions
    threads_per_block = 256
    blocks_per_grid = int(math.ceil(n / threads_per_block))

=======

def SPMV_csr_cuda(A, x, y, block_size):
    # Define the CUDA kernel
    kernel_code_template = """
    #define BLOCK_SIZE %(BLOCK_SIZE)d

    __global__ void spmv_csr_kernel(int num_rows, int num_cols, int *d_csrRowPtr, int *d_csrColInd, float *d_csrVal, float *d_x, float *d_y)
    {
        int row = blockIdx.x * blockDim.x + threadIdx.x;
        if(row < num_rows)
        {
            float dot = 0;
            for(int i = d_csrRowPtr[row]; i < d_csrRowPtr[row + 1]; i++)
            {
                dot += d_csrVal[i] * d_x[d_csrColInd[i]];
            }
            d_y[row] = dot;
        }
    }
    """
    # Get the kernel code from the template
    kernel_code = kernel_code_template % {
        'BLOCK_SIZE': block_size,
    }
    # Compile the CUDA kernel
    mod = SourceModule(kernel_code)
    spmv_csr_kernel = mod.get_function("spmv_csr_kernel")
    # Call the kernel on the card
    spmv_csr_kernel(A.shape[0], A.shape[1], A.indptr, A.indices, A.data, x, y, block=(block_size,1,1), grid=(int(math.ceil(A.shape[0]/block_size)),1))
    return y

=======

def CSR_SPMV(A, x):
    # A is a CSR matrix
    # x is a vector
    # y is the output vector
    y = pycuda.gpuarray.empty_like(x)
    # Get the kernel
    kernel = get_kernel("CSR_SPMV")
    # Run the kernel
    kernel(A.row, A.col, A.data, x, y, block=(256, 1, 1), grid=(1, 1))
    return y

=======

def csr_spmv_csr(A, x):
    # A is the sparse matrix in CSR format
    # x is the dense vector
    # y is the result of the multiplication
    # y = A*x

### End of Output
# Prompt: CSR SPMV numba
### Beginning of Output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

# CSR SPMV numba implementation

=======

# CSR SPMV numba
# Created on 2020/12/8 15:59
# Updated on 2020/12/8 15:59

=======

# CSR SPMV numba kernel
### End of Output
# Prompt: CSR SPMV numba
def
### egining of Output
Synthesizing 10/10 solutions (Duplicates hidden)

=======

def spmv_csr_numba(A, x):
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def spmv_csr_numba(A_data, A_indices, A_indptr, x):
    y = np.zeros(A_indptr.shape[0]-1)
    for i in range(A_indptr.shape[0]-1):
        for j in range(A_indptr[i], A_indptr[i+1]):
            y[i] += A_data[j] * x[A_indices[j]]
    return y

=======

def csr_spmv_numba(A, x):
    y = np.zeros(A.shape[0], dtype=np.float64)
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i+1]):
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def csr_spmv_numba(data, indices, indptr, x):
    y = np.zeros(indptr.shape[0] - 1, dtype=np.float64)
    for i in range(indptr.shape[0] - 1):
        for j in range(indptr[i], indptr[i + 1]):
            y[i] += data[j] * x[indices[j]]
    return y

=======

def spmv_csr_numba(A, x):
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        row_start = A.indptr[i]
        row_end = A.indptr[i+1]
        for k in range(row_start, row_end):
            y[i] += A.data[k] * x[A.indices[k]]
    return y

=======

def csr_spmv_numba(A, x):
    y = np.zeros(A.shape[0])
    for row in range(A.shape[0]):
        for col in range(A.indptr[row], A.indptr[row+1]):
            y[row] += A.data[col] * x[A.indices[col]]
    return y

=======

def csr_spmv_numba(A, x):
    y = np.zeros(A.shape[0], dtype=A.dtype)
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def SPMV_CSR_numba(A, x):
    y = np.zeros(A.shape[0])
    for i in range(A.shape[0]):
        for j in range(A.indptr[i], A.indptr[i + 1]):
            y[i] += A.data[j] * x[A.indices[j]]
    return y

=======

def csr_spmv_numba(data, indices, indptr, x):
    y = np.zeros(len(indptr) - 1, dtype=np.float64)
    for i in range(len(indptr) - 1):
        for k in range(indptr[i], indptr[i + 1]):
            y[i] += data[k] * x[indices[k]]
    return y
### End of Output
