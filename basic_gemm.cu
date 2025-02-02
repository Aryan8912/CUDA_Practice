#include <iostream>
#include <sstream>
#include <vector>

// Helper methods to check for erros
#include "helper.h"

// CUTLASS includes needed for single-precision GEMM kernel

// Defines cutlass::gemm::device::Gemm, the generic Gemm computation template class.
#include "cutlass/gemm/device/gemm.h"

/// Define a CUTLASS GEMM template and launch a GEMM kernel.
cudaError_t CutlassgemmNN(
    int M, 
    int N,
    int K,
    float alpha,
    float* A,
    int lda,
    float beta,
    float* C,
    int ldc) {
    // Difine type definition for single-precision CUTLASS GEMM with column-major
    // input matrices and 128x128x8 threadblock tile size (chosen by default).

    // To keep the interface manageable, several helpers are defined for plausible compositions
    // including the following example for single-precision GEMM. Typical values are used as

    using ColumnMajor = cutlass::layout::ColumnMajor;
    using CutlassGemm = cutlass::gemm::device::Gemm<float, // Data-type of A matrix
                                                    ColumnMajor, // Layout of a Matrix
                                                    float, // Data-type of B matrix
                                                    ColumnMajor, // Layout of B matrix
                                                    float, // Data-type of C matrix
                                                    ColumnMajor>; // Layout of C matrix
    // Define a CUTLASS GEMM type
    CutlassGemm gemm_operator;

    // Construct the CUTLASS GEMM arguments object.
    //
    // One of CUTLASS's design patterns is to define gemm argument objects that are constructible
    // in host code and passed to kernel by value. these may include pointers, strides, scalars,
    // and other arguments needed by GEMM and its components.
    //
    // The benefit of this pattern are (1.) a structured, composable strategy for passing host-constructible
    CutlassGemm::Argument args({M, N, K}, // Gemm Problem dimensions
                                {A, lda}, // Tensor-ref for source matrix A
                                {B, ldb}, // Tensor-ref for source matrix B
                                {C, ldc}, // Tensor-ref for source matrix C
                                {C, ldc}, // Tensor-ref for destination matrix D(may be different memory than source C matrix)
                                {alpha, beta}); // Scalars used in the Epilogue

    cutlass::Status status = gemm_operator(args);

    //
    // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
    //
    if (status != cutlass::Status::kSuccess){
        return cudaErrorUnknown;
    }

    // Return success, if no errors were encountered.
    return cudaSuccess;
    }

// The source code after this point in the file is generic CUDA using the CUDA Runtime API.
// and simple CUDA kernel to initiable matrices and compute the general matrix product.

/// Kernel to initialize a matrix with small integers.
__global__ void InitializeMatrix_kernel(
    float *matrix,
    int rows,
    int columns,
    int seed = 0) {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        int j = threadIdx.y + blockIdx.y * blockDim.y;
        
        if(i < rows && j < columns) {
            int offset = i + j * rows;

            // Generate arbitrary elements.
            int const k = 16807;
            int const m = 16;
            float value = float(((offset + seed) * k % m) - m / 2);
            matrix[offset] = value;
        }
    }

/// Simple function to initialize a matrix to arbitrary small intergers.
cudaError_t InitializeMatrix(float *matrix, int rows, int columns, int seed = 0) {
    dim3 block(16, 16);
    dim3 grid(
        (rows + block.x -1) / block.x,
        (columns + block.y - 1) / block.y
    );

    InitializeMatrix_kernel<<< grid, block>>>(matrix, rows, columns, seed);

    return cudaGetLastError();
}